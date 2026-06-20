//! Tiny-skia backing-store storage for [`core::TextureCache`].
//!
//! Each cached widget renders into its own [`tiny_skia::Pixmap`]; the main
//! `draw()` pass blits the pixmap into the frame's [`tiny_skia::PixmapMut`]
//! honoring the recording-time [`Transformation`].
use std::sync::Weak;

use rustc_hash::FxHashMap;
use tiny_skia;

use crate::core::Size;
use crate::layer;

pub struct Entry {
    pub pixmap: tiny_skia::Pixmap,
    pub size: Size<u32>,
    pub physical_size: Size<u32>,
    pub scale_factor: f32,
    pub liveness: Weak<()>,
}

#[derive(Default)]
pub struct Storage {
    pub entries: FxHashMap<u64, Entry>,
    pub pending: Vec<(u64, layer::Stack)>,
    pub recording_stack: Vec<(u64, layer::Stack, bool)>,
}

impl Storage {
    pub fn new() -> Self {
        Self::default()
    }
}
