//! Cache the rasterization of widget contents into a `wgpu::Texture` and
//! composite it cheaply each frame under a [`Transformation`].
//!
//! See [`core::TextureCache`] for the public widget-facing handle.
use std::borrow::Cow;
use std::mem;

use rustc_hash::FxHashMap;

use crate::core::{Rectangle, Size, Transformation};
use crate::layer;

use bytemuck::{Pod, Zeroable};

/// A single texture-cache draw instance recorded into a [`crate::layer::Layer`].
#[derive(Debug, Clone)]
pub struct Instance {
    /// The id of the [`crate::core::TextureCache`] backing store to sample from.
    pub cache_id: u64,
    /// The destination bounds (in logical pixels) where the cache should be
    /// composited, with the recording-time transformation already baked in.
    pub bounds: Rectangle,
}

/// A batch of texture-cache draws within a single [`crate::layer::Layer`].
pub type Batch = Vec<Instance>;

/// A persistent backing-store entry for a [`crate::core::TextureCache`].
///
/// Holds the GPU texture and the per-cache rendering state. The dedicated
/// state instances ensure that the cache's render pass does not alias the
/// main renderer's per-frame GPU buffer slots (which would corrupt either
/// the cache or the main scene before `queue.submit` runs).
pub struct Entry {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub texture_bind_group: wgpu::BindGroup,
    pub size: Size<u32>,
    pub physical_size: Size<u32>,
    pub scale_factor: f32,

    pub quad: crate::quad::State,
    pub triangle: crate::triangle::State,
    pub text: crate::text::State,
    pub text_viewport: crate::text::Viewport,
    #[cfg(any(feature = "image", feature = "svg"))]
    pub image: crate::image::State,
}

/// Internal storage of all texture caches owned by a single `Renderer`.
pub struct Storage {
    /// Persistent backing stores keyed by `TextureCache::id`.
    pub entries: FxHashMap<u64, Entry>,
    /// Recorded layer stacks awaiting a flush into their cache's texture
    /// at the start of the next frame's draw.
    pub pending: FxHashMap<u64, layer::Stack>,
    /// Stack of saved `Renderer.layers` values, paired with the cache id
    /// that triggered the swap. Supports nested `draw_to_texture` calls.
    pub recording_stack: Vec<(u64, layer::Stack)>,
    /// Lazily initialized rendering pipeline for compositing caches.
    pub pipeline: Option<Pipeline>,
}

impl Default for Storage {
    fn default() -> Self {
        Self {
            entries: FxHashMap::default(),
            pending: FxHashMap::default(),
            recording_stack: Vec::new(),
            pipeline: None,
        }
    }
}

impl Storage {
    pub fn new() -> Self {
        Self::default()
    }
}

/// The GPU pipeline that samples a cached texture into a textured quad.
#[derive(Debug)]
pub struct Pipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub constant_layout: wgpu::BindGroupLayout,
    pub texture_layout: wgpu::BindGroupLayout,
    pub sampler: wgpu::Sampler,
    pub uniform_alignment: wgpu::BufferAddress,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Uniforms {
    pub transform: [f32; 16],
    pub bounds: [f32; 4],
    pub scale: f32,
    pub _pad: [f32; 3],
}

impl Pipeline {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("iced_wgpu.texture_cache.sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..wgpu::SamplerDescriptor::default()
        });

        let uniform_alignment = device
            .limits()
            .min_uniform_buffer_offset_alignment
            .max(mem::size_of::<Uniforms>() as u32)
            as wgpu::BufferAddress;

        let constant_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("iced_wgpu.texture_cache.constant_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(mem::size_of::<Uniforms>() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("iced_wgpu.texture_cache.texture_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("iced_wgpu.texture_cache.pipeline_layout"),
            bind_group_layouts: &[Some(&constant_layout), Some(&texture_layout)],
            immediate_size: 0,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("iced_wgpu.texture_cache.shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "shader/texture_cache.wgsl"
            ))),
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("iced_wgpu.texture_cache.pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Cw,
                ..wgpu::PrimitiveState::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            constant_layout,
            texture_layout,
            sampler,
            uniform_alignment,
        }
    }
}

/// Per-renderer frame state for compositing cached textures.
#[derive(Default)]
pub struct FrameState {
    pub layers: Vec<LayerState>,
    pub prepare_layer: usize,
}

impl FrameState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn trim(&mut self) {
        self.prepare_layer = 0;
    }
}

/// Per-layer rendering state for compositing cached textures into the
/// current frame.
pub struct LayerState {
    pub uniform_buffer: Option<wgpu::Buffer>,
    pub constant_bind_group: Option<wgpu::BindGroup>,
    pub capacity: u32,
}

impl Default for LayerState {
    fn default() -> Self {
        Self {
            uniform_buffer: None,
            constant_bind_group: None,
            capacity: 0,
        }
    }
}

impl LayerState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Ensures the uniform buffer is large enough for `count` instances.
    pub fn ensure_capacity(&mut self, device: &wgpu::Device, pipeline: &Pipeline, count: u32) {
        if count == 0 {
            return;
        }

        if self.uniform_buffer.is_some() && self.capacity >= count {
            return;
        }

        let size = pipeline.uniform_alignment * count as u64;

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("iced_wgpu.texture_cache.uniform_buffer"),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("iced_wgpu.texture_cache.constant_bind_group"),
            layout: &pipeline.constant_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &buffer,
                        offset: 0,
                        size: wgpu::BufferSize::new(mem::size_of::<Uniforms>() as u64),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&pipeline.sampler),
                },
            ],
        });

        self.uniform_buffer = Some(buffer);
        self.constant_bind_group = Some(bind_group);
        self.capacity = count;
    }

    pub fn write_instance(
        &self,
        belt: &mut wgpu::util::StagingBelt,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        pipeline: &Pipeline,
        index: u32,
        instance: &Instance,
        projection: Transformation,
        scale: f32,
    ) {
        let buffer = self
            .uniform_buffer
            .as_ref()
            .expect("uniform buffer allocated");

        let offset = pipeline.uniform_alignment * index as u64;

        let uniforms = Uniforms {
            transform: *projection.as_ref(),
            bounds: [
                instance.bounds.x,
                instance.bounds.y,
                instance.bounds.width,
                instance.bounds.height,
            ],
            scale,
            _pad: [0.0; 3],
        };

        let bytes = bytemuck::bytes_of(&uniforms);
        let size = wgpu::BufferSize::new(bytes.len() as u64).expect("non-zero size");

        let _ = device;
        belt.write_buffer(encoder, buffer, offset, size)
            .copy_from_slice(bytes);
    }

    pub fn render<'a>(
        &'a self,
        pipeline: &'a Pipeline,
        instances: &[(u32, &'a wgpu::BindGroup)],
        scissor_rect: Rectangle<u32>,
        render_pass: &mut wgpu::RenderPass<'a>,
    ) {
        let Some(constant_bg) = self.constant_bind_group.as_ref() else {
            return;
        };

        render_pass.set_scissor_rect(
            scissor_rect.x,
            scissor_rect.y,
            scissor_rect.width,
            scissor_rect.height,
        );
        render_pass.set_pipeline(&pipeline.pipeline);

        for (index, texture_bg) in instances {
            let dyn_offset = pipeline.uniform_alignment as u32 * index;
            render_pass.set_bind_group(0, constant_bg, &[dyn_offset]);
            render_pass.set_bind_group(1, *texture_bg, &[]);
            render_pass.draw(0..6, 0..1);
        }
    }
}
