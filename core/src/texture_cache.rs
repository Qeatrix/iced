//! Cache widget contents into a persistent texture/pixmap that can be
//! cheaply re-rendered with [`Transformation`] applied on top of it.
//!
//! [`Transformation`]: crate::Transformation
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// A handle to a persistent backing-store (texture for GPU backends, pixmap
/// for CPU backends) that a widget can render its expensive contents into
/// once and then animate cheaply by replaying it under different
/// [`Transformation`]s.
///
/// The handle itself is lightweight (an id and a shared invalidation flag).
/// The actual backing store is owned by the renderer and keyed by [`id`].
///
/// Clones share the same backing store. Dropping every clone signals the
/// renderer that the backing store can be reclaimed on the next frame.
///
/// [`Transformation`]: crate::Transformation
/// [`id`]: Self::id
#[derive(Debug, Clone)]
pub struct TextureCache {
    id: Id,
    invalidated: Arc<AtomicBool>,
}

/// A stable identifier for a [`TextureCache`]. Used by renderers to key their
/// per-cache backing store storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Id(u64);

impl Id {
    /// Returns the raw integer value of the [`Id`].
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

impl TextureCache {
    /// Creates a new [`TextureCache`] handle with a freshly-allocated [`Id`].
    ///
    /// The cache starts in the invalidated state, so the first call to
    /// `draw_to_texture` will actually record drawing operations.
    pub fn new() -> Self {
        Self {
            id: Id(NEXT_ID.fetch_add(1, Ordering::Relaxed)),
            invalidated: Arc::new(AtomicBool::new(true)),
        }
    }

    /// Returns the stable [`Id`] of this cache.
    pub fn id(&self) -> Id {
        self.id
    }

    /// Marks the cache as invalidated. The next call to `draw_to_texture`
    /// will re-record drawing operations and re-render the backing store.
    pub fn invalidate(&self) {
        self.invalidated.store(true, Ordering::Release);
    }

    /// Atomically reads the invalidated flag and resets it to `false`.
    ///
    /// Renderers call this from `draw_to_texture` to decide whether the
    /// closure must be re-run.
    pub fn take_invalidated(&self) -> bool {
        self.invalidated.swap(false, Ordering::AcqRel)
    }

    /// Returns the current invalidation state without resetting it.
    pub fn is_invalidated(&self) -> bool {
        self.invalidated.load(Ordering::Acquire)
    }
}

impl Default for TextureCache {
    fn default() -> Self {
        Self::new()
    }
}
