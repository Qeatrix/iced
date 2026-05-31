//! A [`wgpu`] renderer for [Iced].
//!
//! ![The native path of the Iced ecosystem](https://github.com/iced-rs/iced/blob/0525d76ff94e828b7b21634fa94a747022001c83/docs/graphs/native.png?raw=true)
//!
//! [`wgpu`] supports most modern graphics backends: Vulkan, Metal, DX11, and
//! DX12 (OpenGL and WebGL are still WIP). Additionally, it will support the
//! incoming [WebGPU API].
//!
//! Currently, `iced_wgpu` supports the following primitives:
//! - Text, which is rendered using [`glyphon`].
//! - Quads or rectangles, with rounded borders and a solid background color.
//! - Clip areas, useful to implement scrollables or hide overflowing content.
//! - Images and SVG, loaded from memory or the file system.
//! - Meshes of triangles, useful to draw geometry freely.
//!
//! [Iced]: https://github.com/iced-rs/iced
//! [`wgpu`]: https://github.com/gfx-rs/wgpu-rs
//! [WebGPU API]: https://gpuweb.github.io/gpuweb/
//! [`glyphon`]: https://github.com/grovesNL/glyphon
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/iced-rs/iced/9ab6923e943f784985e9ef9ca28b10278297225d/docs/logo.svg"
)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![allow(missing_docs)]
pub mod layer;
pub mod primitive;
pub mod texture_cache;
pub mod window;

#[cfg(feature = "geometry")]
pub mod geometry;

mod buffer;
mod color;
mod engine;
mod quad;
mod text;
mod triangle;

#[cfg(any(feature = "image", feature = "svg"))]
#[path = "image/mod.rs"]
mod image;

#[cfg(not(any(feature = "image", feature = "svg")))]
#[path = "image/null.rs"]
mod image;

use buffer::Buffer;

use iced_debug as debug;
pub use iced_graphics as graphics;
pub use iced_graphics::core;

pub use wgpu;

pub use engine::Engine;
pub use layer::Layer;
pub use primitive::Primitive;

#[cfg(feature = "geometry")]
pub use geometry::Geometry;

use crate::core::layer::{LayerId, LayerRegistry, LayerSlot};
use crate::core::renderer;
use crate::core::{
    Background, Color, Font, Pixels, Point, Rectangle, Size, TextureCache, TextureRecordMode,
    Transformation,
};
use crate::graphics::mesh;
use crate::graphics::text::{Editor, Paragraph};
use crate::graphics::{Shell, Viewport};
use crate::layer::debug_layer_color;
use std::sync::Arc;

/// A [`wgpu`] graphics renderer for [`iced`].
///
/// [`wgpu`]: https://github.com/gfx-rs/wgpu-rs
/// [`iced`]: https://github.com/iced-rs/iced
pub struct Renderer {
    engine: Engine,
    settings: renderer::Settings,

    layers: layer::Stack,
    scale_factor: Option<f32>,

    quad: quad::State,
    triangle: triangle::State,
    text: text::State,
    text_viewport: text::Viewport,

    #[cfg(any(feature = "svg", feature = "image"))]
    image: image::State,

    // TODO: Centralize all the image feature handling
    #[cfg(any(feature = "svg", feature = "image"))]
    image_cache: std::cell::RefCell<image::Cache>,

    texture_cache: texture_cache::Storage,
    texture_cache_state: texture_cache::FrameState,

    /// Reused scratch index for `compose_layers`. Cleared at the
    /// start of every compose pass; capacity persists frame-to-frame
    /// so no allocation happens in steady state.
    compose_index: Vec<(LayerId, usize)>,

    staging_belt: wgpu::util::StagingBelt,
}

impl Renderer {
    pub fn new(engine: Engine, settings: renderer::Settings) -> Self {
        Self {
            settings,
            layers: layer::Stack::new(),
            scale_factor: None,

            quad: quad::State::new(),
            triangle: triangle::State::new(&engine.device, &engine.triangle_pipeline),
            text: text::State::new(),
            text_viewport: engine.text_pipeline.create_viewport(&engine.device),

            #[cfg(any(feature = "svg", feature = "image"))]
            image: image::State::new(),

            #[cfg(any(feature = "svg", feature = "image"))]
            image_cache: std::cell::RefCell::new(engine.create_image_cache()),

            texture_cache: texture_cache::Storage::new(),
            texture_cache_state: texture_cache::FrameState::new(),
            compose_index: Vec::with_capacity(64),

            // TODO: Resize belt smartly (?)
            // It would be great if the `StagingBelt` API exposed methods
            // for introspection to detect when a resize may be worth it.
            staging_belt: wgpu::util::StagingBelt::new(
                engine.device.clone(),
                buffer::MAX_WRITE_SIZE as u64,
            ),

            engine,
        }
    }

    /// Record commands that draw the current primitives to the target texture view.
    ///
    /// You must call [`finish`](Self::finish) and [`recall`](Self::recall) when submitting
    /// the resulting [`wgpu::CommandEncoder`].
    pub fn draw(
        &mut self,
        clear_color: Option<Color>,
        target: &wgpu::TextureView,
        viewport: &Viewport,
    ) -> wgpu::CommandEncoder {
        let mut encoder =
            self.engine
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("iced_wgpu encoder"),
                });

        // Flush pending texture caches into their backing textures using the
        // same encoder so the main pass can sample them.
        self.flush_pending_caches(&mut encoder);

        self.prepare(&mut encoder, viewport);
        self.render(&mut encoder, target, clear_color, viewport);

        self.quad.trim();
        self.triangle.trim();
        self.text.trim();
        self.texture_cache_state.trim();

        // TODO: Provide window id (?)
        self.engine.trim();

        #[cfg(any(feature = "svg", feature = "image"))]
        {
            self.image.trim();
            self.image_cache.borrow_mut().trim();
        }

        encoder
    }

    pub fn present(
        &mut self,
        clear_color: Option<Color>,
        _format: wgpu::TextureFormat,
        frame: &wgpu::TextureView,
        viewport: &Viewport,
    ) -> wgpu::SubmissionIndex {
        let encoder = self.draw(clear_color, frame, viewport);

        self.staging_belt.finish();
        let submission = self.engine.queue.submit([encoder.finish()]);
        self.staging_belt.recall();
        submission
    }

    /// Renders the current surface to an offscreen buffer.
    ///
    /// Returns RGBA bytes of the texture data.
    pub fn screenshot(&mut self, viewport: &Viewport, background_color: Color) -> Vec<u8> {
        #[derive(Clone, Copy, Debug)]
        struct BufferDimensions {
            width: u32,
            height: u32,
            unpadded_bytes_per_row: usize,
            padded_bytes_per_row: usize,
        }

        impl BufferDimensions {
            fn new(size: Size<u32>) -> Self {
                let unpadded_bytes_per_row = size.width as usize * 4; //slice of buffer per row; always RGBA
                let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize; //256
                let padded_bytes_per_row_padding =
                    (alignment - unpadded_bytes_per_row % alignment) % alignment;
                let padded_bytes_per_row = unpadded_bytes_per_row + padded_bytes_per_row_padding;

                Self {
                    width: size.width,
                    height: size.height,
                    unpadded_bytes_per_row,
                    padded_bytes_per_row,
                }
            }
        }

        let dimensions = BufferDimensions::new(viewport.physical_size());

        let texture_extent = wgpu::Extent3d {
            width: dimensions.width,
            height: dimensions.height,
            depth_or_array_layers: 1,
        };

        let texture = self.engine.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("iced_wgpu.offscreen.source_texture"),
            size: texture_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.engine.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.draw(Some(background_color), &view, viewport);

        let texture = crate::color::convert(
            &self.engine.device,
            &mut encoder,
            texture,
            if graphics::color::GAMMA_CORRECTION {
                wgpu::TextureFormat::Rgba8UnormSrgb
            } else {
                wgpu::TextureFormat::Rgba8Unorm
            },
        );

        let output_buffer = self.engine.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("iced_wgpu.offscreen.output_texture_buffer"),
            size: (dimensions.padded_bytes_per_row * dimensions.height as usize) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &output_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(dimensions.padded_bytes_per_row as u32),
                    rows_per_image: None,
                },
            },
            texture_extent,
        );

        self.staging_belt.finish();
        let index = self.engine.queue.submit([encoder.finish()]);
        self.staging_belt.recall();

        let slice = output_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});

        let _ = self.engine.device.poll(wgpu::PollType::Wait {
            submission_index: Some(index),
            timeout: None,
        });

        let mapped_buffer = slice.get_mapped_range();

        mapped_buffer
            .chunks(dimensions.padded_bytes_per_row)
            .fold(vec![], |mut acc, row| {
                acc.extend(&row[..dimensions.unpadded_bytes_per_row]);
                acc
            })
    }

    fn prepare(&mut self, encoder: &mut wgpu::CommandEncoder, viewport: &Viewport) {
        let scale_factor = viewport.scale_factor();

        self.text_viewport
            .update(&self.engine.queue, viewport.physical_size());

        let physical_bounds =
            Rectangle::<f32>::from(Rectangle::with_size(viewport.physical_size()));

        self.layers.merge();

        for layer in self.layers.iter() {
            let clip_bounds = layer.bounds * scale_factor;

            if physical_bounds
                .intersection(&clip_bounds)
                .and_then(Rectangle::snap)
                .is_none()
            {
                continue;
            }

            if !layer.quads.is_empty() {
                let prepare_span = debug::prepare(debug::Primitive::Quad);

                self.quad.prepare(
                    &self.engine.quad_pipeline,
                    &self.engine.device,
                    &mut self.staging_belt,
                    encoder,
                    &layer.quads,
                    viewport.projection(),
                    scale_factor,
                );

                prepare_span.finish();
            }

            if !layer.triangles.is_empty() {
                let prepare_span = debug::prepare(debug::Primitive::Triangle);

                self.triangle.prepare(
                    &self.engine.triangle_pipeline,
                    &self.engine.device,
                    &mut self.staging_belt,
                    encoder,
                    &layer.triangles,
                    Transformation::scale(scale_factor),
                    viewport.physical_size(),
                );

                prepare_span.finish();
            }

            if !layer.primitives.is_empty() {
                let prepare_span = debug::prepare(debug::Primitive::Shader);

                let mut primitive_storage = self
                    .engine
                    .primitive_storage
                    .write()
                    .expect("Write primitive storage");

                for instance in &layer.primitives {
                    instance.primitive.prepare(
                        &mut primitive_storage,
                        &self.engine.device,
                        &self.engine.queue,
                        self.engine.format,
                        &instance.bounds,
                        viewport,
                    );
                }

                prepare_span.finish();
            }

            #[cfg(any(feature = "svg", feature = "image"))]
            if !layer.images.is_empty() {
                let prepare_span = debug::prepare(debug::Primitive::Image);

                self.image.prepare(
                    &self.engine.image_pipeline,
                    &self.engine.device,
                    &mut self.staging_belt,
                    encoder,
                    &mut self.image_cache.borrow_mut(),
                    &layer.images,
                    viewport.projection(),
                    scale_factor,
                );

                prepare_span.finish();
            }

            if !layer.text.is_empty() {
                let prepare_span = debug::prepare(debug::Primitive::Text);

                self.text.prepare(
                    &self.engine.text_pipeline,
                    &self.engine.device,
                    &self.engine.queue,
                    &self.text_viewport,
                    encoder,
                    &layer.text,
                    layer.bounds,
                    Transformation::scale(scale_factor),
                );

                prepare_span.finish();
            }

            if !layer.cached_textures.is_empty() {
                // Allocate or reuse this layer's compositing state.
                if self.texture_cache_state.layers.len() <= self.texture_cache_state.prepare_layer {
                    self.texture_cache_state
                        .layers
                        .push(texture_cache::LayerState::new());
                }

                // The pipeline is guaranteed to exist if any cache is registered,
                // and `draw_cached_texture` would have rejected the call otherwise.
                let pipeline = self
                    .texture_cache
                    .pipeline
                    .as_ref()
                    .expect("texture_cache pipeline must exist when drawing cached textures");

                let layer_state =
                    &mut self.texture_cache_state.layers[self.texture_cache_state.prepare_layer];

                layer_state.ensure_capacity(
                    &self.engine.device,
                    pipeline,
                    layer.cached_textures.len() as u32,
                );

                for (index, instance) in layer.cached_textures.iter().enumerate() {
                    // `uv_max` tells the composite shader which sub-region of
                    // the (possibly over-allocated) cache texture actually
                    // holds content. Without it, an over-allocated texture
                    // would either show a transparent strip on the right /
                    // bottom or squish the content into the quad — see
                    // `texture_cache::Uniforms`.
                    let uv_max = self
                        .texture_cache
                        .entries
                        .get(&instance.cache_id)
                        .map(|entry| {
                            [
                                entry.physical_size.width as f32
                                    / entry.texture_capacity_size.width.max(1) as f32,
                                entry.physical_size.height as f32
                                    / entry.texture_capacity_size.height.max(1) as f32,
                            ]
                        })
                        .unwrap_or([1.0, 1.0]);

                    layer_state.write_instance(
                        &mut self.staging_belt,
                        encoder,
                        &self.engine.device,
                        pipeline,
                        index as u32,
                        instance,
                        viewport.projection(),
                        scale_factor,
                        uv_max,
                    );
                }

                self.texture_cache_state.prepare_layer += 1;
            }
        }
    }

    fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &wgpu::TextureView,
        clear_color: Option<Color>,
        viewport: &Viewport,
    ) {
        use std::mem::ManuallyDrop;

        let mut render_pass =
            ManuallyDrop::new(encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("iced_wgpu render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: frame,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: match clear_color {
                            Some(background_color) => wgpu::LoadOp::Clear({
                                let [r, g, b, a] =
                                    graphics::color::pack(background_color).components();

                                wgpu::Color {
                                    r: f64::from(r * a),
                                    g: f64::from(g * a),
                                    b: f64::from(b * a),
                                    a: f64::from(a),
                                }
                            }),
                            None => wgpu::LoadOp::Load,
                        },
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            }));

        let mut quad_layer = 0;
        let mut mesh_layer = 0;
        let mut text_layer = 0;
        // Seed from the frame-level running index: cached-texture layer states
        // are shared across the per-cache flush passes and the main pass (not
        // swapped like quad/text state), so this index must continue across
        // every `render` call in the frame to stay aligned with `prepare`.
        let mut cached_texture_layer = self.texture_cache_state.render_layer;

        #[cfg(any(feature = "svg", feature = "image"))]
        let mut image_layer = 0;

        let scale_factor = viewport.scale_factor();
        let physical_bounds =
            Rectangle::<f32>::from(Rectangle::with_size(viewport.physical_size()));

        let scale = Transformation::scale(scale_factor);

        for layer in self.layers.iter() {
            let Some(physical_bounds) =
                physical_bounds.intersection(&(layer.bounds * scale_factor))
            else {
                continue;
            };

            let Some(scissor_rect) = physical_bounds.snap() else {
                continue;
            };

            if !layer.quads.is_empty() {
                let render_span = debug::render(debug::Primitive::Quad);
                self.quad.render(
                    &self.engine.quad_pipeline,
                    quad_layer,
                    scissor_rect,
                    &layer.quads,
                    &mut render_pass,
                );
                render_span.finish();

                quad_layer += 1;
            }

            if !layer.triangles.is_empty() {
                let _ = ManuallyDrop::into_inner(render_pass);

                let render_span = debug::render(debug::Primitive::Triangle);
                mesh_layer += self.triangle.render(
                    &self.engine.triangle_pipeline,
                    encoder,
                    frame,
                    mesh_layer,
                    &layer.triangles,
                    physical_bounds,
                    scale,
                );
                render_span.finish();

                render_pass =
                    ManuallyDrop::new(encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("iced_wgpu render pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: frame,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    }));
            }

            if !layer.primitives.is_empty() {
                let render_span = debug::render(debug::Primitive::Shader);

                let primitive_storage = self
                    .engine
                    .primitive_storage
                    .read()
                    .expect("Read primitive storage");

                let mut need_render = Vec::new();

                for instance in &layer.primitives {
                    let bounds = instance.bounds * scale;

                    if let Some(clip_bounds) = (instance.bounds * scale)
                        .intersection(&physical_bounds)
                        .and_then(Rectangle::snap)
                    {
                        render_pass.set_viewport(
                            bounds.x,
                            bounds.y,
                            bounds.width,
                            bounds.height,
                            0.0,
                            1.0,
                        );

                        render_pass.set_scissor_rect(
                            clip_bounds.x,
                            clip_bounds.y,
                            clip_bounds.width,
                            clip_bounds.height,
                        );

                        let drawn = instance
                            .primitive
                            .draw(&primitive_storage, &mut render_pass);

                        if !drawn {
                            need_render.push((instance, clip_bounds));
                        }
                    }
                }

                render_pass.set_viewport(
                    0.0,
                    0.0,
                    viewport.physical_width() as f32,
                    viewport.physical_height() as f32,
                    0.0,
                    1.0,
                );

                render_pass.set_scissor_rect(
                    0,
                    0,
                    viewport.physical_width(),
                    viewport.physical_height(),
                );

                if !need_render.is_empty() {
                    let _ = ManuallyDrop::into_inner(render_pass);

                    for (instance, clip_bounds) in need_render {
                        instance
                            .primitive
                            .render(&primitive_storage, encoder, frame, &clip_bounds);
                    }

                    render_pass =
                        ManuallyDrop::new(encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("iced_wgpu render pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: frame,
                                depth_slice: None,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Load,
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                            multiview_mask: None,
                        }));
                }

                render_span.finish();
            }

            #[cfg(any(feature = "svg", feature = "image"))]
            if !layer.images.is_empty() {
                let render_span = debug::render(debug::Primitive::Image);
                self.image.render(
                    &self.engine.image_pipeline,
                    image_layer,
                    scissor_rect,
                    &mut render_pass,
                );
                render_span.finish();

                image_layer += 1;
            }

            if !layer.text.is_empty() {
                let render_span = debug::render(debug::Primitive::Text);
                text_layer += self.text.render(
                    &self.engine.text_pipeline,
                    &self.text_viewport,
                    text_layer,
                    &layer.text,
                    scissor_rect,
                    &mut render_pass,
                );
                render_span.finish();
            }

            if !layer.cached_textures.is_empty() {
                let pipeline = self
                    .texture_cache
                    .pipeline
                    .as_ref()
                    .expect("texture_cache pipeline must exist");

                let layer_state = &self.texture_cache_state.layers[cached_texture_layer];

                // Build (index, &texture_bind_group) pairs for instances whose
                // backing entry is still present.
                let bindings: Vec<(u32, &wgpu::BindGroup)> = layer
                    .cached_textures
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, instance)| {
                        self.texture_cache
                            .entries
                            .get(&instance.cache_id)
                            .map(|entry| (idx as u32, &entry.texture_bind_group))
                    })
                    .collect();

                if !bindings.is_empty() {
                    layer_state.render(pipeline, &bindings, scissor_rect, &mut render_pass);
                }

                cached_texture_layer += 1;
            }
        }

        let _ = ManuallyDrop::into_inner(render_pass);

        // Persist the running index so the next `render` call this frame (the
        // main pass after cache flushes, or the next cache) keeps consuming
        // layer states in lockstep with `prepare`.
        self.texture_cache_state.render_layer = cached_texture_layer;

        debug::layers_rendered(|| {
            self.layers
                .iter()
                .filter(|layer| {
                    !layer.is_empty()
                        && physical_bounds
                            .intersection(&(layer.bounds * scale_factor))
                            .is_some_and(|viewport| viewport.snap().is_some())
                })
                .count()
        });
    }

    /// Prepares currently mapped buffers for use in a submission.
    ///
    /// Usually, this method is only needed if you are calling [`Renderer::draw`] directly,
    /// instead of relying on [`Renderer::present`].
    ///
    /// You must call this method _before_ submitting the resulting [`wgpu::CommandEncoder`]
    /// of [`Renderer::draw`] to a [`wgpu::Queue`].
    pub fn finish(&mut self) {
        self.staging_belt.finish();
    }

    /// Recalls all of the closed buffers back to be reused.
    ///
    /// Usually, this method is only needed if you are calling [`Renderer::draw`] directly,
    /// instead of relying on [`Renderer::present`] to a [`wgpu::Queue`].
    ///
    /// You must call this method _after_ submitting the resulting [`wgpu::CommandEncoder`]
    /// of [`Renderer::draw`] to a [`wgpu::Queue`].
    pub fn recall(&mut self) {
        self.staging_belt.recall();
    }

    /// Allocates a fresh [`texture_cache::Entry`] (texture + bind group +
    /// per-cache `State`s) and inserts it into the storage. Recreates the
    /// texture if an entry with this id already exists but the physical
    /// size has changed.
    fn ensure_texture_cache_entry(
        &mut self,
        id: u64,
        size: Size<u32>,
        physical_size: Size<u32>,
        scale_factor: f32,
    ) {
        /// Geometric growth for one axis of the cache texture. Returns the
        /// smallest power-of-two ≥ `needed`, with a 128-px floor, but never
        /// shrinks below `current` so we don't oscillate when the user
        /// drag-resizes back and forth.
        fn grow_capacity(needed: u32, current: u32) -> u32 {
            const MIN_CAPACITY: u32 = 128;
            let target = needed.max(MIN_CAPACITY).next_power_of_two();
            target.max(current)
        }

        // Lazily initialize the sampling pipeline.
        if self.texture_cache.pipeline.is_none() {
            self.texture_cache.pipeline = Some(texture_cache::Pipeline::new(
                &self.engine.device,
                self.engine.format,
            ));
        }
        let pipeline = self
            .texture_cache
            .pipeline
            .as_ref()
            .expect("texture_cache pipeline initialized");

        // The existing texture's allocated dimensions. We can keep the texture
        // (skip the GPU realloc) whenever the new content fits inside it.
        let existing_capacity = self
            .texture_cache
            .entries
            .get(&id)
            .map(|e| e.texture_capacity_size);

        let fits = existing_capacity.map_or(false, |cap| {
            cap.width >= physical_size.width && cap.height >= physical_size.height
        });

        if fits {
            // Content shrank or grew within current capacity: no realloc,
            // just patch the metadata. The compose shader will read the new
            // `uv_max = physical_size / texture_capacity_size` and sample
            // only the content sub-region of the existing texture.
            if let Some(entry) = self.texture_cache.entries.get_mut(&id) {
                entry.size = size;
                entry.physical_size = physical_size;
                entry.scale_factor = scale_factor;
            }
            return;
        }

        // Grow the texture geometrically (round each dimension up to the next
        // power of two, with a 128-px floor). Rationale: during a drag-resize
        // the per-frame physical size walks up in pixels of 1; reallocating
        // the wgpu texture and its bind group every frame hammers the
        // allocator and dominates the stutter. A geometric grow amortizes
        // realloc to O(log max_dim) over the lifetime of the cache, at a
        // worst-case 2× memory overhead — the unused area is transparent and
        // explicitly excluded from the composite via `uv_max`.
        let target_capacity = Size::new(
            grow_capacity(
                physical_size.width,
                existing_capacity.map(|c| c.width).unwrap_or(0),
            ),
            grow_capacity(
                physical_size.height,
                existing_capacity.map(|c| c.height).unwrap_or(0),
            ),
        );

        let texture = self.engine.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("iced_wgpu.texture_cache.texture"),
            size: wgpu::Extent3d {
                width: target_capacity.width.max(1),
                height: target_capacity.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.engine.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let texture_bind_group = self
            .engine
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("iced_wgpu.texture_cache.texture_bind_group"),
                layout: &pipeline.texture_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                }],
            });

        // Update-in-place when an entry already exists: keep its per-cache
        // `quad`/`triangle`/`text`/`text_viewport`/`image` `State`s so the
        // already-warmed wgpu buffers, bind groups, and `TextRenderer`s are
        // reused across the realloc. The discarded `State`s on every resize
        // were the dominant source of first-resize stutter — none of them
        // hold anything tied to a specific texture-target size (`text_viewport`
        // is re-`update()`d with the new physical size in `prepare`).
        if let Some(entry) = self.texture_cache.entries.get_mut(&id) {
            entry.texture = texture;
            entry.view = view;
            entry.texture_bind_group = texture_bind_group;
            entry.size = size;
            entry.physical_size = physical_size;
            entry.texture_capacity_size = target_capacity;
            entry.scale_factor = scale_factor;
            return;
        }

        let entry = texture_cache::Entry {
            texture,
            view,
            texture_bind_group,
            size,
            physical_size,
            texture_capacity_size: target_capacity,
            scale_factor,
            quad: quad::State::new(),
            triangle: triangle::State::new(&self.engine.device, &self.engine.triangle_pipeline),
            text: text::State::new(),
            text_viewport: self
                .engine
                .text_pipeline
                .create_viewport(&self.engine.device),
            #[cfg(any(feature = "image", feature = "svg"))]
            image: image::State::new(),
        };

        let _ = self.texture_cache.entries.insert(id, entry);
    }

    /// Renders all pending texture-cache recordings into their respective
    /// backing textures. Must be called before the main `prepare`/`render`
    /// pass and use the same `encoder` so cache textures are populated
    /// before the main pass samples them.
    fn flush_pending_caches(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if self.texture_cache.pending.is_empty() {
            return;
        }

        let pending: Vec<_> = self.texture_cache.pending.drain(..).collect();

        for (id, layers) in pending {
            // Briefly take ownership of the entry to avoid a double mutable
            // borrow of self.texture_cache.entries while calling self.prepare.
            let Some(mut entry) = self.texture_cache.entries.remove(&id) else {
                continue;
            };

            // The viewport's `physical_size` must match the GPU texture's
            // allocated dimensions (not the content's `physical_size`), so the
            // orthographic projection lands content at its actual physical
            // pixels within the over-allocated texture. With the wrong size
            // here, NDC=±1 would map to the texture's edges, stretching
            // content across the entire texture.
            let cache_viewport =
                Viewport::with_physical_size(entry.texture_capacity_size, entry.scale_factor);

            // Swap dedicated state with main renderer's per-frame state.
            std::mem::swap(&mut self.quad, &mut entry.quad);
            std::mem::swap(&mut self.triangle, &mut entry.triangle);
            std::mem::swap(&mut self.text, &mut entry.text);
            std::mem::swap(&mut self.text_viewport, &mut entry.text_viewport);
            #[cfg(any(feature = "image", feature = "svg"))]
            std::mem::swap(&mut self.image, &mut entry.image);

            let saved_layers = std::mem::replace(&mut self.layers, layers);

            self.prepare(encoder, &cache_viewport);
            self.render(
                encoder,
                &entry.view,
                Some(Color::TRANSPARENT),
                &cache_viewport,
            );

            self.layers = saved_layers;

            // Trim the cache's State counters before swapping them back so
            // that subsequent re-records start from slot 0.
            self.quad.trim();
            self.triangle.trim();
            self.text.trim();
            #[cfg(any(feature = "image", feature = "svg"))]
            self.image.trim();

            // Swap back.
            std::mem::swap(&mut self.quad, &mut entry.quad);
            std::mem::swap(&mut self.triangle, &mut entry.triangle);
            std::mem::swap(&mut self.text, &mut entry.text);
            std::mem::swap(&mut self.text_viewport, &mut entry.text_viewport);
            #[cfg(any(feature = "image", feature = "svg"))]
            std::mem::swap(&mut self.image, &mut entry.image);

            let _ = self.texture_cache.entries.insert(id, entry);
        }
    }
}

/// Maximum compose-recursion depth. Guards against pathological
/// nesting or accidental cycles in layer `parent_id` graphs.
const MAX_LAYER_DEPTH: u32 = 64;

fn compose_one(renderer: &mut Renderer, slots: &[Arc<LayerSlot>], idx: usize, depth: u32) {
    use core::Renderer as _;

    if depth >= MAX_LAYER_DEPTH {
        debug_assert!(false, "layer depth exceeded {}", MAX_LAYER_DEPTH);
        return;
    }

    let slot = slots[idx].clone();
    let data = slot.read();
    let id = slot.id();

    // The compose body: paint this slot's cache (no-op when
    // unrecorded — that's the group-layer case, e.g. content_stack)
    // and recurse into children inside the parent's transform block.
    let paint = move |renderer: &mut Renderer| {
        renderer.draw_cached_texture(&slot.cache, data.bounds);

        // Recurse into children. `partition_point` finds the
        // contiguous run of `compose_index` whose parent_id matches.
        let start = renderer.compose_index.partition_point(|(p, _)| *p < id);
        let end = renderer.compose_index.partition_point(|(p, _)| *p <= id);

        // Snapshot the child indices into a local Vec so we don't
        // hold a borrow on `renderer.compose_index` across the
        // recursive call (which needs `&mut renderer`). In practice
        // the slice is small and short-lived.
        let children: Vec<usize> = renderer.compose_index[start..end]
            .iter()
            .map(|(_, i)| *i)
            .collect();

        for ci in children {
            compose_one(renderer, slots, ci, depth + 1);
        }
    };

    // Wrap in the slot's transform, then optionally in its clip.
    // `with_layer` must be outermost so the clip survives the
    // transform (clip bounds are absolute, not transform-relative).
    match data.clip_bounds {
        Some(clip) => renderer.with_layer(clip, move |r| {
            r.with_transformation(data.transform, paint);
        }),
        None => renderer.with_transformation(data.transform, paint),
    }
}

fn compose_outline_one(renderer: &mut Renderer, slots: &[Arc<LayerSlot>], idx: usize, depth: u32) {
    use core::Renderer as _;
    use core::text::Renderer as _;

    if depth >= MAX_LAYER_DEPTH {
        debug_assert!(false, "layer depth exceeded {}", MAX_LAYER_DEPTH);
        return;
    }

    let slot = slots[idx].clone();
    let data = slot.read();
    let id = slot.id();

    // let color = DEBUG_LAYER_COLORS[depth as usize % DEBUG_LAYER_COLORS.len()];
    let color = debug_layer_color(depth);
    let text_size = 14.0;
    let label_clip = data.bounds.expand(text_size * 2 as f32);

    let paint = move |renderer: &mut Renderer| {
        renderer.fill_quad(
            core::renderer::Quad {
                bounds: data.bounds,
                border: core::Border {
                    color,
                    width: 1.0,
                    radius: 0.0.into(),
                },
                snap: true,
                ..Default::default()
            },
            Color::TRANSPARENT,
        );

        let gap = 4.0;
        let text_position = Point::new(data.bounds.x, data.bounds.y - text_size - gap);

        renderer.fill_text(
            core::Text {
                content: format!("Layer #{:#?}", id.as_u64()),
                bounds: data.bounds.size(),
                size: 14.into(),
                line_height: core::text::LineHeight::default(),
                font: Default::default(),
                align_x: core::text::Alignment::Left,
                align_y: core::alignment::Vertical::Top,
                shaping: core::text::Shaping::Basic,
                wrapping: core::text::Wrapping::None,
                ellipsis: core::text::Ellipsis::None,
                hint_factor: None,
            },
            text_position,
            color,
            label_clip,
        );

        let start = renderer.compose_index.partition_point(|(p, _)| *p < id);
        let end = renderer.compose_index.partition_point(|(p, _)| *p <= id);

        let children: Vec<usize> = renderer.compose_index[start..end]
            .iter()
            .map(|(_, i)| *i)
            .collect();

        for ci in children {
            compose_outline_one(renderer, slots, ci, depth + 1);
        }
    };

    // Wrap in the slot's transform, then optionally in its clip.
    // `with_layer` must be outermost so the clip survives the
    // transform (clip bounds are absolute, not transform-relative).
    renderer.with_layer(Rectangle::INFINITE, move |r| {
        r.with_transformation(data.transform, paint);
    });
}

impl core::Renderer for Renderer {
    fn start_layer(&mut self, bounds: Rectangle) {
        self.layers.push_clip(bounds);
    }

    fn end_layer(&mut self) {
        self.layers.pop_clip();
    }

    fn start_transformation(&mut self, transformation: Transformation) {
        self.layers.push_transformation(transformation);
    }

    fn end_transformation(&mut self) {
        self.layers.pop_transformation();
    }

    fn fill_quad(&mut self, quad: core::renderer::Quad, background: impl Into<Background>) {
        let (layer, transformation) = self.layers.current_mut();
        layer.draw_quad(quad, background.into(), transformation);
    }

    fn start_recording_texture(
        &mut self,
        mode: TextureRecordMode,
        cache: &TextureCache,
        size: Size<u32>,
        scale_factor: f32,
    ) -> bool {
        let id = cache.id().as_u64();
        let invalidated = cache.take_invalidated();

        let physical_size = Size::new(
            ((size.width as f32) * scale_factor).round() as u32,
            ((size.height as f32) * scale_factor).round() as u32,
        );

        let needs_redraw = invalidated
            || match self.texture_cache.entries.get(&id) {
                Some(e) => {
                    e.size != size
                        || e.physical_size != physical_size
                        || (e.scale_factor - scale_factor).abs() > f32::EPSILON
                }
                None => true,
            };

        let keep = match mode {
            TextureRecordMode::Flush => {
                if !needs_redraw {
                    return false;
                }

                true
            }
            TextureRecordMode::TraverseOnly => needs_redraw,
        };

        if keep {
            self.ensure_texture_cache_entry(id, size, physical_size, scale_factor);
        }

        let bounds = Rectangle::with_size(Size::new(size.width as f32, size.height as f32));
        let mut new_stack = layer::Stack::new();
        new_stack.reset(bounds);

        let saved = std::mem::replace(&mut self.layers, new_stack);
        self.texture_cache.recording_stack.push((id, saved, keep));

        true
    }

    fn end_recording_texture(&mut self) {
        let Some((id, saved, keep)) = self.texture_cache.recording_stack.pop() else {
            return;
        };

        let captured = std::mem::replace(&mut self.layers, saved);

        if keep {
            let _ = self.texture_cache.pending.insert(id, captured);
        }
    }

    fn draw_cached_texture(&mut self, cache: &TextureCache, bounds: Rectangle) {
        let id = cache.id().as_u64();
        if !self.texture_cache.entries.contains_key(&id) {
            return;
        }

        let (layer, transformation) = self.layers.current_mut();
        layer.draw_cached_texture(id, bounds, transformation);
    }

    fn compose_layers(&mut self, registry: &LayerRegistry, debug_outline: bool) {
        let slots = registry.registered();
        if slots.is_empty() {
            return;
        }

        // Build a sorted child index: (parent_id, slot_index). For N<~50,
        // sorted `Vec` + `partition_point` is faster than a HashMap and
        // reuses pre-allocated capacity.
        self.compose_index.clear();
        for (i, s) in slots.iter().enumerate() {
            if let Some(p) = s.read().parent_id {
                self.compose_index.push((p, i));
            }
        }
        self.compose_index.sort_by_key(|(p, _)| *p);

        // Walk roots in registration order. Skipping the linear scan here
        // would require a second sorted index; for typical layer counts
        // the scan is cheaper than that.
        for i in 0..slots.len() {
            if slots[i].read().parent_id.is_none() {
                compose_one(self, slots, i, 0);

                if debug_outline {
                    compose_outline_one(self, slots, i, 0);
                }
            }
        }
    }

    fn allocate_image(
        &mut self,
        _handle: &core::image::Handle,
        _callback: impl FnOnce(Result<core::image::Allocation, core::image::Error>) + Send + 'static,
    ) {
        #[cfg(feature = "image")]
        self.image_cache
            .get_mut()
            .allocate_image(_handle, _callback);
    }

    fn hint(&mut self, scale_factor: f32) {
        self.scale_factor = Some(scale_factor);
    }

    fn scale_factor(&self) -> Option<f32> {
        Some(self.scale_factor? * self.layers.transformation().scale_factor())
    }

    fn tick(&mut self) {
        #[cfg(feature = "image")]
        self.image_cache.get_mut().receive();
    }

    fn reset(&mut self, new_bounds: Rectangle) {
        self.layers.reset(new_bounds);
    }
}

impl core::text::Renderer for Renderer {
    type Font = Font;
    type Paragraph = Paragraph;
    type Editor = Editor;

    const ICON_FONT: Font = Font::new("Iced-Icons");
    const CHECKMARK_ICON: char = '\u{f00c}';
    const ARROW_DOWN_ICON: char = '\u{e800}';
    const ICED_LOGO: char = '\u{e801}';
    const SCROLL_UP_ICON: char = '\u{e802}';
    const SCROLL_DOWN_ICON: char = '\u{e803}';
    const SCROLL_LEFT_ICON: char = '\u{e804}';
    const SCROLL_RIGHT_ICON: char = '\u{e805}';

    fn default_font(&self) -> Self::Font {
        self.settings.default_font
    }

    fn default_size(&self) -> Pixels {
        self.settings.default_text_size
    }

    fn fill_paragraph(
        &mut self,
        text: &Self::Paragraph,
        position: Point,
        color: Color,
        clip_bounds: Rectangle,
    ) {
        let (layer, transformation) = self.layers.current_mut();

        layer.draw_paragraph(text, position, color, clip_bounds, transformation);
    }

    fn fill_editor(
        &mut self,
        editor: &Self::Editor,
        position: Point,
        color: Color,
        clip_bounds: Rectangle,
    ) {
        let (layer, transformation) = self.layers.current_mut();
        layer.draw_editor(editor, position, color, clip_bounds, transformation);
    }

    fn fill_text(
        &mut self,
        text: core::Text,
        position: Point,
        color: Color,
        clip_bounds: Rectangle,
    ) {
        let (layer, transformation) = self.layers.current_mut();
        layer.draw_text(text, position, color, clip_bounds, transformation);
    }
}

impl graphics::text::Renderer for Renderer {
    fn fill_raw(&mut self, raw: graphics::text::Raw) {
        let (layer, transformation) = self.layers.current_mut();
        layer.draw_text_raw(raw, transformation);
    }
}

#[cfg(feature = "image")]
impl core::image::Renderer for Renderer {
    type Handle = core::image::Handle;

    fn load_image(
        &self,
        handle: &Self::Handle,
    ) -> Result<core::image::Allocation, core::image::Error> {
        self.image_cache
            .borrow_mut()
            .load_image(&self.engine.device, &self.engine.queue, handle)
    }

    fn measure_image(&self, handle: &Self::Handle) -> Option<core::Size<u32>> {
        self.image_cache.borrow_mut().measure_image(handle)
    }

    fn draw_image(&mut self, image: core::Image, bounds: Rectangle, clip_bounds: Rectangle) {
        let (layer, transformation) = self.layers.current_mut();
        layer.draw_raster(image, bounds, clip_bounds, transformation);
    }
}

#[cfg(feature = "svg")]
impl core::svg::Renderer for Renderer {
    fn measure_svg(&self, handle: &core::svg::Handle) -> core::Size<u32> {
        self.image_cache.borrow_mut().measure_svg(handle)
    }

    fn draw_svg(&mut self, svg: core::Svg, bounds: Rectangle, clip_bounds: Rectangle) {
        let (layer, transformation) = self.layers.current_mut();
        layer.draw_svg(svg, bounds, clip_bounds, transformation);
    }
}

impl graphics::mesh::Renderer for Renderer {
    fn draw_mesh(&mut self, mesh: graphics::Mesh) {
        debug_assert!(
            !mesh.indices().is_empty(),
            "Mesh must not have empty indices"
        );

        debug_assert!(
            mesh.indices().len().is_multiple_of(3),
            "Mesh indices length must be a multiple of 3"
        );

        let (layer, transformation) = self.layers.current_mut();
        layer.draw_mesh(mesh, transformation);
    }

    fn draw_mesh_cache(&mut self, cache: mesh::Cache) {
        let (layer, transformation) = self.layers.current_mut();
        layer.draw_mesh_cache(cache, transformation);
    }
}

#[cfg(feature = "geometry")]
impl graphics::geometry::Renderer for Renderer {
    type Geometry = Geometry;
    type Frame = geometry::Frame;

    fn new_frame(&self, bounds: Rectangle) -> Self::Frame {
        geometry::Frame::new(bounds)
    }

    fn draw_geometry(&mut self, geometry: Self::Geometry) {
        let (layer, transformation) = self.layers.current_mut();

        match geometry {
            Geometry::Live {
                meshes,
                images,
                text,
            } => {
                layer.draw_mesh_group(meshes, transformation);

                for image in images {
                    layer.draw_image(image, transformation);
                }

                layer.draw_text_group(text, transformation);
            }
            Geometry::Cached(cache) => {
                if let Some(meshes) = cache.meshes {
                    layer.draw_mesh_cache(meshes, transformation);
                }

                if let Some(images) = cache.images {
                    for image in images.iter().cloned() {
                        layer.draw_image(image, transformation);
                    }
                }

                if let Some(text) = cache.text {
                    layer.draw_text_cache(text, transformation);
                }
            }
        }
    }
}

impl primitive::Renderer for Renderer {
    fn draw_primitive(&mut self, bounds: Rectangle, primitive: impl Primitive) {
        let (layer, transformation) = self.layers.current_mut();
        layer.draw_primitive(bounds, primitive, transformation);
    }
}

impl graphics::compositor::Default for crate::Renderer {
    type Compositor = window::Compositor;
}

impl renderer::Headless for Renderer {
    async fn new(settings: renderer::Settings, backend: Option<&str>) -> Option<Self> {
        if backend.is_some_and(|backend| backend != "wgpu") {
            return None;
        }

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::from_env().unwrap_or(wgpu::Backends::PRIMARY),
            flags: wgpu::InstanceFlags::empty(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .ok()?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("iced_wgpu [headless]"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    max_bind_groups: 2,
                    ..wgpu::Limits::default()
                },
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            })
            .await
            .ok()?;

        let engine = Engine::new(
            &adapter,
            device,
            queue,
            if graphics::color::GAMMA_CORRECTION {
                wgpu::TextureFormat::Rgba8UnormSrgb
            } else {
                wgpu::TextureFormat::Rgba8Unorm
            },
            Some(graphics::Antialiasing::MSAAx4),
            Shell::headless(),
        );

        Some(Self::new(engine, settings))
    }

    fn name(&self) -> String {
        "wgpu".to_owned()
    }

    fn screenshot(
        &mut self,
        size: Size<u32>,
        scale_factor: f32,
        background_color: Color,
    ) -> Vec<u8> {
        self.screenshot(
            &Viewport::with_physical_size(size, scale_factor),
            background_color,
        )
    }
}
