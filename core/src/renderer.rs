//! Write your own renderer.
#[cfg(debug_assertions)]
mod null;

use crate::image;
use crate::{
    Background, Border, Color, Font, Pixels, Rectangle, Shadow, Size, TextureCache, Transformation,
    Vector,
};

/// Whether anti-aliasing should be avoided by snapping primitive coordinates to the
/// pixel grid.
pub const CRISP: bool = cfg!(feature = "crisp");

/// A component that can be used by widgets to draw themselves on a screen.
pub trait Renderer {
    /// Starts recording a new layer.
    fn start_layer(&mut self, bounds: Rectangle);

    /// Ends recording a new layer.
    ///
    /// The new layer will clip its contents to the provided `bounds`.
    fn end_layer(&mut self);

    /// Draws the primitives recorded in the given closure in a new layer.
    ///
    /// The layer will clip its contents to the provided `bounds`.
    fn with_layer(&mut self, bounds: Rectangle, f: impl FnOnce(&mut Self)) {
        self.start_layer(bounds);
        f(self);
        self.end_layer();
    }

    /// Starts recording with a new [`Transformation`].
    fn start_transformation(&mut self, transformation: Transformation);

    /// Ends recording a new layer.
    ///
    /// The new layer will clip its contents to the provided `bounds`.
    fn end_transformation(&mut self);

    /// Applies a [`Transformation`] to the primitives recorded in the given closure.
    fn with_transformation(&mut self, transformation: Transformation, f: impl FnOnce(&mut Self)) {
        self.start_transformation(transformation);
        f(self);
        self.end_transformation();
    }

    /// Applies a translation to the primitives recorded in the given closure.
    fn with_translation(&mut self, translation: Vector, f: impl FnOnce(&mut Self)) {
        self.with_transformation(Transformation::translate(translation.x, translation.y), f);
    }

    /// Fills a [`Quad`] with the provided [`Background`].
    fn fill_quad(&mut self, quad: Quad, background: impl Into<Background>);

    /// Begins recording draw operations into the backing store keyed by the
    /// given [`TextureCache`].
    ///
    /// Returns `true` if the renderer is now in recording mode and the caller
    /// should issue the draw operations that should be cached. Returns
    /// `false` if the cache is still fresh — in which case the caller must
    /// skip recording and just call [`draw_cached_texture`] later.
    ///
    /// Callers must pair every `true` result with a matching call to
    /// [`end_recording_texture`]. See [`draw_to_texture`] for the
    /// recommended high-level wrapper.
    ///
    /// [`draw_cached_texture`]: Self::draw_cached_texture
    /// [`end_recording_texture`]: Self::end_recording_texture
    /// [`draw_to_texture`]: Self::draw_to_texture
    fn start_recording_texture(
        &mut self,
        cache: &TextureCache,
        size: Size<u32>,
        scale_factor: f32,
    ) -> bool;

    /// Ends a recording session started with [`start_recording_texture`].
    ///
    /// [`start_recording_texture`]: Self::start_recording_texture
    fn end_recording_texture(&mut self);

    /// Records the drawing operations performed in the given closure into a
    /// persistent backing store keyed by the [`TextureCache`] handle.
    ///
    /// If the cache is not invalidated and its existing backing store matches
    /// `size` and `scale_factor`, the closure is skipped entirely and the
    /// existing backing store is reused. Use [`draw_cached_texture`]
    /// afterwards to composite the cache into the current frame.
    ///
    /// [`draw_cached_texture`]: Self::draw_cached_texture
    fn draw_to_texture(
        &mut self,
        cache: &TextureCache,
        size: Size<u32>,
        scale_factor: f32,
        f: impl FnOnce(&mut Self),
    ) {
        if self.start_recording_texture(cache, size, scale_factor) {
            f(self);
            self.end_recording_texture();
        }
    }

    /// Composites a previously recorded [`TextureCache`] into the current
    /// frame at the given `bounds`.
    ///
    /// The current transformation stack is honored, so wrap the call with
    /// [`with_transformation`] to animate translate / scale / rotate of the
    /// cached contents without re-rasterizing them.
    ///
    /// If the cache has not been populated yet (no successful prior
    /// [`draw_to_texture`] call), this is a no-op.
    ///
    /// [`draw_to_texture`]: Self::draw_to_texture
    /// [`with_transformation`]: Self::with_transformation
    fn draw_cached_texture(&mut self, cache: &TextureCache, bounds: Rectangle);

    /// Creates an [`image::Allocation`] for the given [`image::Handle`] and calls the given callback with it.
    fn allocate_image(
        &mut self,
        handle: &image::Handle,
        callback: impl FnOnce(Result<image::Allocation, image::Error>) + Send + 'static,
    );

    /// Provides hints to the [`Renderer`] about the rendering target.
    ///
    /// This may be used internally by the [`Renderer`] to perform optimizations
    /// and/or improve rendering quality.
    ///
    /// For instance, providing a `scale_factor` may be used by some renderers to
    /// perform metrics hinting internally in physical coordinates while keeping
    /// layout coordinates logical and, therefore, maintain linearity.
    fn hint(&mut self, scale_factor: f32);

    /// Returns the last scale factor provided as a [`hint`](Self::hint).
    fn scale_factor(&self) -> Option<f32>;

    /// Resets the [`Renderer`] to start drawing in the `new_bounds` from scratch.
    fn reset(&mut self, new_bounds: Rectangle);

    /// Polls any concurrent computations that may be pending in the [`Renderer`].
    ///
    /// By default, it does nothing.
    fn tick(&mut self) {}
}

/// A polygon with four sides.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quad {
    /// The bounds of the [`Quad`].
    pub bounds: Rectangle,

    /// The [`Border`] of the [`Quad`]. The border is drawn on the inside of the [`Quad`].
    pub border: Border,

    /// The [`Shadow`] of the [`Quad`].
    pub shadow: Shadow,

    /// Whether the [`Quad`] should be snapped to the pixel grid.
    pub snap: bool,
}

impl Default for Quad {
    fn default() -> Self {
        Self {
            bounds: Rectangle::with_size(Size::ZERO),
            border: Border::default(),
            shadow: Shadow::default(),
            snap: CRISP,
        }
    }
}

/// The styling attributes of a [`Renderer`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Style {
    /// The text color
    pub text_color: Color,
}

impl Default for Style {
    fn default() -> Self {
        Style {
            text_color: Color::BLACK,
        }
    }
}

/// A headless renderer is a renderer that can render offscreen without
/// a window nor a compositor.
pub trait Headless {
    /// Creates a new [`Headless`] renderer;
    fn new(settings: Settings, backend: Option<&str>) -> impl Future<Output = Option<Self>>
    where
        Self: Sized;

    /// Returns the unique name of the renderer.
    ///
    /// This name may be used by testing libraries to uniquely identify
    /// snapshots.
    fn name(&self) -> String;

    /// Draws offscreen into a screenshot, returning a collection of
    /// bytes representing the rendered pixels in RGBA order.
    fn screenshot(
        &mut self,
        size: Size<u32>,
        scale_factor: f32,
        background_color: Color,
    ) -> Vec<u8>;
}

/// The settings of a [`Renderer`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Settings {
    /// The default [`Font`] to use.
    pub default_font: Font,

    /// The default size of text.
    ///
    /// By default, it will be set to `16.0`.
    pub default_text_size: Pixels,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            default_font: Font::DEFAULT,
            default_text_size: Pixels(16.0),
        }
    }
}
