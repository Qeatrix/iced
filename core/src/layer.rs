//! Compositor-layer primitive — a persistent registration unit for
//! cached, transform-animated content.
//!
//! A [`LayerSlot`] is a small, heap-allocated handle (held behind an
//! [`Arc`]) that a widget owns across frames. Layer-aware widgets (the
//! `Cached` widget and any custom animation widget) write their
//! per-frame parameters (transform, bounds, parent, clip) into the slot
//! during the event-propagation phase and call
//! [`Shell::register_layer`] to enroll the slot in this frame's
//! [`LayerRegistry`].
//!
//! After the widget-tree `draw` walk completes, the runtime calls
//! `Renderer::compose_layers`, which composites every registered slot
//! into the renderer's existing layer stack — parent transforms chained
//! by left-multiplication, clip bounds wrapped via `with_layer`, child
//! slots recursed into inside their parent's transform block. This
//! decouples a layer's transform-only animation from the surrounding
//! widget-tree walk, so a nested cached widget keeps animating even
//! when its parent's cache is fresh.
//!
//! No `unsafe` is used anywhere in this module — slot state is guarded
//! by a `std::sync::Mutex`, so a single lock acquisition covers all
//! field accesses for a slot per phase.
//!
//! [`Shell::register_layer`]: crate::Shell::register_layer
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

use crate::{Rectangle, Size, TextureCache, Transformation};

/// Enables debug outline drawing on layers if `ICED_DEBUG_LAYERS` is set.
pub static DEBUG_LAYERS: LazyLock<bool> =
    LazyLock::new(|| std::env::var_os("ICED_DEBUG_LAYERS").is_some());

/// A globally-unique identifier for a [`LayerSlot`]. Allocated
/// monotonically from a process-wide counter.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct LayerId(u64);

impl LayerId {
    /// Returns the raw integer value of the [`LayerId`].
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

/// State that mutates per frame inside a [`LayerSlot`]. Held behind a
/// single [`Mutex`] so each update or compose access requires just one
/// lock acquisition.
#[derive(Debug, Clone, Copy)]
pub struct LayerSlotData {
    /// Local transform applied during compositing. Composes with the
    /// parent layer's transform by left-multiplication.
    pub transform: Transformation,
    /// Bounds passed to `draw_cached_texture` during compositing. For
    /// a `Cached` widget this is the padded cache bounds; for a group
    /// layer with no recorded cache this can be the layer's outer
    /// rectangle (composite is a no-op when the cache is unrecorded).
    pub bounds: Rectangle,
    /// Identifier of the enclosing layer at registration time, or
    /// [`None`] for top-level layers.
    pub parent_id: Option<LayerId>,
    /// Optional clip rectangle. When set, [`Renderer::compose_layers`]
    /// wraps this slot's composite in a `with_layer` block using these
    /// bounds. Children inherit the clip naturally (they composite
    /// inside the parent's `with_layer`).
    pub clip_bounds: Option<Rectangle>,
}

impl Default for LayerSlotData {
    fn default() -> Self {
        Self {
            transform: Transformation::IDENTITY,
            bounds: Rectangle::with_size(Size::ZERO),
            parent_id: None,
            clip_bounds: None,
        }
    }
}

/// A persistent registration handle for a compositor layer.
///
/// Hold one of these (behind an [`Arc`]) on each frame the widget
/// exists; write into it during `update`; pass it to
/// [`Shell::register_layer`] so the runtime knows to composite it.
///
/// [`Shell::register_layer`]: crate::Shell::register_layer
#[derive(Debug)]
pub struct LayerSlot {
    id: LayerId,
    /// Texture-cache handle for the recorded contents of this layer.
    ///
    /// Group layers (e.g. content_stack) may never record into this
    /// cache — `draw_cached_texture` is a no-op on an unrecorded cache,
    /// so the compositor still walks the layer's children correctly
    /// inside its transform/clip block.
    pub cache: TextureCache,
    data: Mutex<LayerSlotData>,
    subtree_dirty: AtomicBool,
}

impl LayerSlot {
    /// Creates a new layer slot backed by the given texture cache,
    /// wrapped in an [`Arc`] so the slot can be cheaply cloned into
    /// the registry.
    pub fn new(cache: TextureCache) -> Arc<Self> {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Arc::new(Self {
            id: LayerId(NEXT.fetch_add(1, Ordering::Relaxed)),
            cache,
            data: Mutex::new(LayerSlotData::default()),
            subtree_dirty: AtomicBool::new(false),
        })
    }

    /// Returns the [`LayerId`] of this slot.
    pub fn id(&self) -> LayerId {
        self.id
    }

    /// Acquires the slot's mutex and runs `f` with mutable access to
    /// the per-frame data.
    pub fn write<R>(&self, f: impl FnOnce(&mut LayerSlotData) -> R) -> R {
        let mut guard = self.data.lock().expect("LayerSlot mutex poisoned");
        f(&mut *guard)
    }

    /// Acquires the slot's mutex and returns a copy of the per-frame
    /// data. [`LayerSlotData`] is `Copy`, so the lock is released
    /// before the caller observes the value.
    pub fn read(&self) -> LayerSlotData {
        *self.data.lock().expect("LayerSlot mutex poisoned")
    }

    /// Flags this slot as having a dirty cache somewhere in its subtree.
    ///
    /// Set during the `update` walk by a descendant `Cached` whose own
    /// cache is invalidated (it marks every ancestor slot via
    /// [`Shell::layer_stack_ancestors`]). It tells this slot's `draw`
    /// that it must *traverse* its content to reach that descendant —
    /// without re-rasterizing its own texture.
    ///
    /// [`Shell::layer_stack_ancestors`]: crate::Shell::layer_stack_ancestors
    pub fn mark_subtree_dirty(&self) {
        self.subtree_dirty.store(true, Ordering::Release);
    }

    /// Atomically reads the subtree-dirty flag and resets it to `false`.
    ///
    /// Consumed once per frame by this slot's `draw`. Returns `true` if a
    /// descendant marked this slot during the preceding `update` walk,
    /// then clears the flag so it does not persist into the next frame
    /// (frame-scoped, mirroring [`TextureCache::take_invalidated`]).
    ///
    /// [`TextureCache::take_invalidated`]: crate::TextureCache::take_invalidated
    pub fn take_subtree_dirty(&self) -> bool {
        self.subtree_dirty.swap(false, Ordering::AcqRel)
    }
}

/// Per-frame layer registry, owned by the runtime. Cleared each frame
/// before event propagation; populated by layer-aware widgets'
/// `update`; consumed by `Renderer::compose_layers`.
#[derive(Debug)]
pub struct LayerRegistry {
    /// Slots registered this frame, in first-registration order.
    pub(crate) registered: Vec<Arc<LayerSlot>>,
    /// O(1) dedup membership set so repeated `register_layer` calls on
    /// the same slot (a Cached widget that updates multiple times per
    /// frame, for example) only push once.
    pub(crate) registered_ids: HashSet<LayerId>,
    /// Stack pushed/popped during update propagation. A widget calling
    /// [`Shell::current_layer`] sees the top of this stack.
    ///
    /// [`Shell::current_layer`]: crate::Shell::current_layer
    pub(crate) stack: Vec<Arc<LayerSlot>>,
}

impl LayerRegistry {
    /// Creates a new, empty registry. Capacities are pre-reserved so
    /// the steady-state `register_layer` path does not allocate.
    pub fn new() -> Self {
        Self {
            registered: Vec::with_capacity(64),
            registered_ids: HashSet::with_capacity(64),
            stack: Vec::with_capacity(8),
        }
    }

    /// Clears all per-frame state. Called by the runtime at the start
    /// of every frame; capacity is retained.
    pub fn clear(&mut self) {
        self.registered.clear();
        self.registered_ids.clear();
        self.stack.clear();
    }

    /// Returns the slots registered this frame, in registration order.
    pub fn registered(&self) -> &[Arc<LayerSlot>] {
        &self.registered
    }

    /// Returns `true` if no slots have been registered this frame.
    pub fn is_empty(&self) -> bool {
        self.registered.is_empty()
    }
}

impl Default for LayerRegistry {
    fn default() -> Self {
        Self::new()
    }
}
