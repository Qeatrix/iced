#![allow(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
pub mod window;

mod engine;
mod layer;
mod primitive;
mod text;
mod texture_cache;

#[cfg(feature = "image")]
mod raster;

#[cfg(feature = "svg")]
mod vector;

#[cfg(feature = "geometry")]
pub mod geometry;

use iced_debug as debug;
pub use iced_graphics as graphics;
pub use iced_graphics::core;

pub use layer::Layer;
pub use primitive::Primitive;

#[cfg(feature = "geometry")]
pub use geometry::Geometry;

use crate::core::renderer;
use crate::core::{
    Background, Color, Font, Pixels, Point, Rectangle, Size, TextureCache, Transformation,
};
use crate::engine::Engine;
use crate::graphics::Viewport;
use crate::graphics::compositor;
use crate::graphics::text::{Editor, Paragraph};

/// A [`tiny-skia`] graphics renderer for [`iced`].
///
/// [`tiny-skia`]: https://github.com/RazrFalcon/tiny-skia
/// [`iced`]: https://github.com/iced-rs/iced
pub struct Renderer {
    settings: renderer::Settings,
    layers: layer::Stack,
    engine: Engine, // TODO: Shared engine
    texture_cache: texture_cache::Storage,
}

impl std::fmt::Debug for Renderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Renderer")
            .field("settings", &self.settings)
            .field("layers", &self.layers)
            .field("engine", &self.engine)
            .finish()
    }
}

impl Renderer {
    pub fn new(settings: renderer::Settings) -> Self {
        Self {
            settings,
            layers: layer::Stack::new(),
            engine: Engine::new(),
            texture_cache: texture_cache::Storage::new(),
        }
    }

    pub fn layers(&mut self) -> &[Layer] {
        self.layers.flush();
        self.layers.as_slice()
    }

    pub fn draw(
        &mut self,
        pixels: &mut tiny_skia::PixmapMut<'_>,
        clip_mask: &mut tiny_skia::Mask,
        viewport: &Viewport,
        damage: &[Rectangle],
        background_color: Color,
    ) {
        let scale_factor = viewport.scale_factor();

        // 1. Flush pending texture caches into their backing pixmaps.
        if !self.texture_cache.pending.is_empty() {
            let pending: Vec<(u64, layer::Stack)> =
                self.texture_cache.pending.drain().collect();

            for (id, mut layers) in pending {
                let Some(mut entry) = self.texture_cache.entries.remove(&id) else {
                    continue;
                };

                layers.flush();

                // Clear the cache pixmap, then render the recorded layers
                // into it with a fresh clip mask sized to the pixmap.
                entry.pixmap.fill(tiny_skia::Color::TRANSPARENT);

                let mut cache_clip_mask = tiny_skia::Mask::new(
                    entry.physical_size.width.max(1),
                    entry.physical_size.height.max(1),
                )
                .expect("allocate texture-cache clip mask");

                let cache_logical_size = Size::new(
                    entry.size.width as f32,
                    entry.size.height as f32,
                );
                let cache_damage = [Rectangle::with_size(cache_logical_size)];

                Self::render_stack(
                    &mut self.engine,
                    &layers,
                    &mut entry.pixmap.as_mut(),
                    &mut cache_clip_mask,
                    entry.scale_factor,
                    &cache_damage,
                    Color::TRANSPARENT,
                    &self.texture_cache.entries,
                );

                let _ = self.texture_cache.entries.insert(id, entry);
            }
        }

        // 2. Render the main scene.
        self.layers.flush();
        Self::render_stack(
            &mut self.engine,
            &self.layers,
            pixels,
            clip_mask,
            scale_factor,
            damage,
            background_color,
            &self.texture_cache.entries,
        );

        self.engine.trim();
    }

    fn render_stack(
        engine: &mut Engine,
        layers: &layer::Stack,
        pixels: &mut tiny_skia::PixmapMut<'_>,
        clip_mask: &mut tiny_skia::Mask,
        scale_factor: f32,
        damage: &[Rectangle],
        background_color: Color,
        texture_cache_entries: &rustc_hash::FxHashMap<u64, texture_cache::Entry>,
    ) {
        for &damage_bounds in damage {
            let damage_bounds = damage_bounds * scale_factor;

            let path = tiny_skia::PathBuilder::from_rect(
                tiny_skia::Rect::from_xywh(
                    damage_bounds.x,
                    damage_bounds.y,
                    damage_bounds.width,
                    damage_bounds.height,
                )
                .expect("Create damage rectangle"),
            );

            pixels.fill_path(
                &path,
                &tiny_skia::Paint {
                    shader: tiny_skia::Shader::SolidColor(engine::into_color(background_color)),
                    anti_alias: false,
                    blend_mode: tiny_skia::BlendMode::Source,
                    ..Default::default()
                },
                tiny_skia::FillRule::default(),
                tiny_skia::Transform::identity(),
                None,
            );

            for layer in layers.iter() {
                let Some(layer_bounds) = damage_bounds.intersection(&(layer.bounds * scale_factor))
                else {
                    continue;
                };

                engine::adjust_clip_mask(clip_mask, layer_bounds);

                if !layer.quads.is_empty() {
                    let render_span = debug::render(debug::Primitive::Quad);
                    for (quad, background) in &layer.quads {
                        engine.draw_quad(
                            quad,
                            background,
                            Transformation::scale(scale_factor),
                            pixels,
                            clip_mask,
                            layer_bounds,
                        );
                    }
                    render_span.finish();
                }

                if !layer.primitives.is_empty() {
                    let render_span = debug::render(debug::Primitive::Triangle);

                    for group in &layer.primitives {
                        let Some(group_bounds) =
                            (group.clip_bounds() * scale_factor).intersection(&layer_bounds)
                        else {
                            continue;
                        };

                        engine::adjust_clip_mask(clip_mask, group_bounds);

                        for primitive in group.as_slice() {
                            engine.draw_primitive(
                                primitive,
                                Transformation::scale(scale_factor) * group.transformation(),
                                pixels,
                                clip_mask,
                                group_bounds,
                            );
                        }

                        engine::adjust_clip_mask(clip_mask, layer_bounds);
                    }

                    render_span.finish();
                }

                if !layer.images.is_empty() {
                    let render_span = debug::render(debug::Primitive::Image);

                    for image in &layer.images {
                        engine.draw_image(
                            image,
                            Transformation::scale(scale_factor),
                            pixels,
                            clip_mask,
                            layer_bounds,
                        );
                    }

                    render_span.finish();
                }

                if !layer.text.is_empty() {
                    let render_span = debug::render(debug::Primitive::Image);

                    for group in &layer.text {
                        for text in group.as_slice() {
                            engine.draw_text(
                                text,
                                Transformation::scale(scale_factor) * group.transformation(),
                                pixels,
                                clip_mask,
                                layer_bounds,
                            );
                        }
                    }

                    render_span.finish();
                }

                if !layer.cached_textures.is_empty() {
                    let render_span = debug::render(debug::Primitive::Image);

                    for instance in &layer.cached_textures {
                        let Some(entry) = texture_cache_entries.get(&instance.cache_id)
                        else {
                            continue;
                        };

                        let dst_bounds = instance.bounds
                            * Transformation::scale(scale_factor);

                        let src_w = entry.physical_size.width.max(1) as f32;
                        let src_h = entry.physical_size.height.max(1) as f32;
                        let scale_x = dst_bounds.width / src_w;
                        let scale_y = dst_bounds.height / src_h;

                        let _ = pixels.draw_pixmap(
                            dst_bounds.x as i32,
                            dst_bounds.y as i32,
                            entry.pixmap.as_ref(),
                            // Bilinear so a supersampled (ss > 1) pixmap is
                            // antialiased on downscale; at ss == 1 the scale is
                            // 1.0 at an integer destination, so this is exact.
                            &tiny_skia::PixmapPaint {
                                quality: tiny_skia::FilterQuality::Bilinear,
                                ..Default::default()
                            },
                            tiny_skia::Transform::from_scale(scale_x, scale_y),
                            Some(clip_mask),
                        );
                    }

                    render_span.finish();
                }
            }
        }
    }
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

    fn fill_quad(&mut self, quad: renderer::Quad, background: impl Into<Background>) {
        let (layer, transformation) = self.layers.current_mut();
        layer.draw_quad(quad, background.into(), transformation);
    }

    fn start_recording_texture(
        &mut self,
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

        if !needs_redraw {
            return false;
        }

        let needs_alloc = match self.texture_cache.entries.get(&id) {
            Some(e) => e.physical_size != physical_size,
            None => true,
        };

        if needs_alloc {
            let pixmap = tiny_skia::Pixmap::new(
                physical_size.width.max(1),
                physical_size.height.max(1),
            )
            .expect("allocate texture-cache pixmap");

            let _ = self.texture_cache.entries.insert(
                id,
                texture_cache::Entry {
                    pixmap,
                    size,
                    physical_size,
                    scale_factor,
                },
            );
        } else if let Some(entry) = self.texture_cache.entries.get_mut(&id) {
            entry.size = size;
            entry.scale_factor = scale_factor;
        }

        let bounds = Rectangle::with_size(Size::new(size.width as f32, size.height as f32));
        let mut new_stack = layer::Stack::new();
        new_stack.reset(bounds);

        let saved = std::mem::replace(&mut self.layers, new_stack);
        self.texture_cache.recording_stack.push((id, saved));

        true
    }

    fn end_recording_texture(&mut self) {
        let Some((id, saved)) = self.texture_cache.recording_stack.pop() else {
            return;
        };

        let captured = std::mem::replace(&mut self.layers, saved);
        let _ = self.texture_cache.pending.insert(id, captured);
    }

    fn draw_cached_texture(&mut self, cache: &TextureCache, bounds: Rectangle) {
        let id = cache.id().as_u64();
        if !self.texture_cache.entries.contains_key(&id) {
            return;
        }

        let generation = cache.generation();
        let (layer, transformation) = self.layers.current_mut();
        layer.draw_cached_texture(id, generation, bounds, transformation);
    }

    fn allocate_image(
        &mut self,
        _handle: &core::image::Handle,
        callback: impl FnOnce(Result<core::image::Allocation, core::image::Error>) + Send + 'static,
    ) {
        #[cfg(feature = "image")]
        #[allow(unsafe_code)]
        // TODO: Concurrency
        callback(self.engine.raster_pipeline.load(_handle));

        #[cfg(not(feature = "image"))]
        callback(Err(core::image::Error::Unsupported));
    }

    fn hint(&mut self, _scale_factor: f32) {
        // TODO: No hinting supported
        // We'll replace `tiny-skia` with `vello_cpu` soon
    }

    fn scale_factor(&self) -> Option<f32> {
        None
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
                primitives,
                images,
                text,
                clip_bounds,
            } => {
                layer.draw_primitive_group(primitives, clip_bounds, transformation);

                for image in images {
                    layer.draw_image(image, transformation);
                }

                layer.draw_text_group(text, clip_bounds, transformation);
            }
            Geometry::Cache(cache) => {
                layer.draw_primitive_cache(cache.primitives, cache.clip_bounds, transformation);

                for image in cache.images.iter() {
                    layer.draw_image(image.clone(), transformation);
                }

                layer.draw_text_cache(cache.text, cache.clip_bounds, transformation);
            }
        }
    }
}

impl graphics::mesh::Renderer for Renderer {
    fn draw_mesh(&mut self, _mesh: graphics::Mesh) {
        log::warn!("iced_tiny_skia does not support drawing meshes");
    }

    fn draw_mesh_cache(&mut self, _cache: iced_graphics::mesh::Cache) {
        log::warn!("iced_tiny_skia does not support drawing meshes");
    }
}

#[cfg(feature = "image")]
impl core::image::Renderer for Renderer {
    type Handle = core::image::Handle;

    fn load_image(
        &self,
        handle: &Self::Handle,
    ) -> Result<core::image::Allocation, core::image::Error> {
        self.engine.raster_pipeline.load(handle)
    }

    fn measure_image(&self, handle: &Self::Handle) -> Option<crate::core::Size<u32>> {
        self.engine.raster_pipeline.dimensions(handle)
    }

    fn draw_image(&mut self, image: core::Image, bounds: Rectangle, clip_bounds: Rectangle) {
        let (layer, transformation) = self.layers.current_mut();
        layer.draw_raster(image, bounds, clip_bounds, transformation);
    }
}

#[cfg(feature = "svg")]
impl core::svg::Renderer for Renderer {
    fn measure_svg(&self, handle: &core::svg::Handle) -> crate::core::Size<u32> {
        self.engine.vector_pipeline.viewport_dimensions(handle)
    }

    fn draw_svg(&mut self, svg: core::Svg, bounds: Rectangle, clip_bounds: Rectangle) {
        let (layer, transformation) = self.layers.current_mut();
        layer.draw_svg(svg, bounds, clip_bounds, transformation);
    }
}

impl compositor::Default for Renderer {
    type Compositor = window::Compositor;
}

impl renderer::Headless for Renderer {
    async fn new(settings: renderer::Settings, backend: Option<&str>) -> Option<Self> {
        if backend.is_some_and(|backend| !["tiny-skia", "tiny_skia", "software"].contains(&backend))
        {
            return None;
        }

        Some(Self::new(settings))
    }

    fn name(&self) -> String {
        "tiny-skia".to_owned()
    }

    fn screenshot(
        &mut self,
        size: Size<u32>,
        scale_factor: f32,
        background_color: Color,
    ) -> Vec<u8> {
        let viewport = Viewport::with_physical_size(size, scale_factor);

        window::compositor::screenshot(self, &viewport, background_color)
    }
}
