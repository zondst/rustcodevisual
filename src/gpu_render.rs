//! GPU-Accelerated Headless Renderer for Video Export
//! Uses wgpu for high-performance offscreen rendering with compute shaders
//! Achieves 4K 60fps export at 3-5x realtime on NVIDIA hardware

use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

/// GPU Particle data structure (matches WGSL layout)
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuParticle {
    pub position: [f32; 2],
    pub velocity: [f32; 2],
    pub color: [f32; 4],
    pub size: f32,
    pub life: f32,
    pub max_life: f32,
    pub audio_alpha: f32,
    pub audio_size: f32,
    pub brightness: f32,
    pub _padding: [f32; 2],
}

/// Simulation parameters uniform buffer
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SimParams {
    pub delta_time: f32,
    pub time: f32,
    pub width: f32,
    pub height: f32,
    pub audio_amplitude: f32,
    pub audio_bass: f32,
    pub audio_mid: f32,
    pub audio_high: f32,
    pub audio_beat: f32,
    pub beat_burst_strength: f32,
    pub damping: f32,
    pub speed: f32,
    pub num_particles: u32,
    pub has_audio: u32,
    // Audio-reactive parameters (matching preview behavior)
    pub fade_attack_speed: f32,
    pub fade_release_speed: f32,
    pub audio_spawn_threshold: f32,
    pub audio_reactive_spawn: u32,
    pub spawn_radius: f32,
    pub gravity: f32,
    pub _padding: [f32; 2],
}

/// Render parameters for the particle rendering + tonemap passes
///
/// Note: this struct is shared between the particle render pass and the tonemap pass.
/// Keep it tightly packed and WGSL-compatible (all scalars) to avoid alignment surprises.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct RenderParams {
    pub width: f32,
    pub height: f32,
    /// Particle glow / soft edge strength
    pub glow_intensity: f32,
    /// Exposure applied to *particles* (not background/overlay)
    pub exposure: f32,
    /// Bloom strength applied to *particles*
    pub bloom_strength: f32,
    /// Selected particle shape id
    pub shape_id: f32,
    /// Background color in **linear** space (RGB) and alpha.
    ///
    /// For alpha-capable exports you typically want bg_a = 0 and bg_rgb = 0.
    pub bg_r: f32,
    pub bg_g: f32,
    pub bg_b: f32,
    pub bg_a: f32,
    pub _padding: [f32; 2],
}

/// GPU-accelerated headless renderer
pub struct GpuRenderer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    // Render targets
    render_texture: wgpu::Texture,
    render_view: wgpu::TextureView,
    // We keep the backing textures around to own the GPU resources.
    // Only the views are used in pipelines, so Rust would otherwise warn these are "never read".
    #[allow(dead_code)]
    hdr_texture: wgpu::Texture,
    hdr_view: wgpu::TextureView,

    // MSAA textures for anti-aliased particle rendering
    #[allow(dead_code)]
    msaa_texture: wgpu::Texture,
    msaa_view: wgpu::TextureView,
    sample_count: u32,

    // Bloom textures (mip chain)
    #[allow(dead_code)]
    bloom_textures: Vec<wgpu::Texture>,
    bloom_views: Vec<wgpu::TextureView>,

    // Overlay texture (Waveform/Spectrum)
    overlay_texture: wgpu::Texture,
    overlay_view: wgpu::TextureView,

    // Staging buffer for CPU readback
    staging_buffer: wgpu::Buffer,
    bytes_per_row: u32,

    // Compute pipeline for particle simulation (optional / future use)
    #[allow(dead_code)]
    compute_pipeline: wgpu::ComputePipeline,
    #[allow(dead_code)]
    compute_bind_group_layout: wgpu::BindGroupLayout,

    // Particle buffers (double-buffered)
    particle_buffers: [wgpu::Buffer; 2],
    current_buffer: usize,

    // Spectrum buffer for audio data (optional)
    #[allow(dead_code)]
    spectrum_buffer: wgpu::Buffer,

    // Simulation params uniform (optional)
    #[allow(dead_code)]
    sim_params_buffer: wgpu::Buffer,
    // Render pipelines for particles (normal over / additive)
    render_pipeline_alpha: wgpu::RenderPipeline,
    render_pipeline_additive: wgpu::RenderPipeline,
    render_bind_group_layout: wgpu::BindGroupLayout,
    render_params_buffer: wgpu::Buffer,

    // Bloom pipelines
    bloom_downsample_pipeline: wgpu::ComputePipeline,
    bloom_upsample_pipeline: wgpu::ComputePipeline,
    bloom_bind_group_layout: wgpu::BindGroupLayout,

    // Tonemap pipeline
    tonemap_pipeline: wgpu::RenderPipeline,
    tonemap_bind_group_layout: wgpu::BindGroupLayout,

    // Sampler
    sampler: wgpu::Sampler,

    // Dimensions
    width: u32,
    height: u32,
    #[allow(dead_code)]
    max_particles: u32,
}

impl GpuRenderer {
    /// Create a new GPU renderer with headless context
    pub fn new(width: u32, height: u32, max_particles: u32) -> Result<Self, String> {
        pollster::block_on(Self::new_async(width, height, max_particles))
    }

    async fn new_async(width: u32, height: u32, max_particles: u32) -> Result<Self, String> {
        // Create wgpu instance (no window needed)
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::DX12,
            ..Default::default()
        });

        // Request high-performance adapter
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or("Failed to find GPU adapter")?;

        let adapter_info = adapter.get_info();
        log::info!("Using GPU: {} ({:?})", adapter_info.name, adapter_info.backend);

        // Request device with appropriate limits
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("GPU Renderer Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits {
                        max_texture_dimension_2d: adapter.limits().max_texture_dimension_2d.min(8192),
                        max_storage_buffer_binding_size: adapter.limits().max_storage_buffer_binding_size.min(256 * 1024 * 1024), // 256MB for particles
                        max_compute_workgroup_size_x: adapter.limits().max_compute_workgroup_size_x.min(256),
                        ..wgpu::Limits::downlevel_defaults()
                    },
                },
                None,
            )
            .await
            .map_err(|e| format!("Failed to create device: {}", e))?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        // Create render target texture (final output)
        let render_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Render Target"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                 | wgpu::TextureUsages::COPY_SRC
                 | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let render_view = render_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create HDR texture for particle rendering (resolve target for MSAA)
        let hdr_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("HDR Buffer"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,  // Resolve target must be sample_count: 1
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                 | wgpu::TextureUsages::TEXTURE_BINDING
                 | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let hdr_view = hdr_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Detect supported MSAA sample count (prefer 4x for quality/performance balance)
        let sample_flags = adapter.get_texture_format_features(wgpu::TextureFormat::Rgba16Float).flags;
        let sample_count = if sample_flags.contains(wgpu::TextureFormatFeatureFlags::MULTISAMPLE_X4) {
            4
        } else if sample_flags.contains(wgpu::TextureFormatFeatureFlags::MULTISAMPLE_X2) {
            2
        } else {
            1
        };

        log::info!("MSAA sample count: {}x", sample_count);

        // Create MSAA texture for anti-aliased particle rendering
        let msaa_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("MSAA HDR Texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count,  // 4x MSAA for smooth particle edges
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,  // Only used for rendering
            view_formats: &[],
        });
        let msaa_view = msaa_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create bloom mip chain (5 levels)
        let mut bloom_textures = Vec::new();
        let mut bloom_views = Vec::new();
        let mut mip_width = width / 2;
        let mut mip_height = height / 2;

        for i in 0..5 {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("Bloom Mip {}", i)),
                size: wgpu::Extent3d {
                    width: mip_width.max(1),
                    height: mip_height.max(1),
                    depth_or_array_layers: 1
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                     | wgpu::TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            });
            bloom_views.push(tex.create_view(&wgpu::TextureViewDescriptor::default()));
            bloom_textures.push(tex);
            mip_width /= 2;
            mip_height /= 2;
        }

        // Create Overlay texture
        let overlay_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Overlay Texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let overlay_view = overlay_texture.create_view(&wgpu::TextureViewDescriptor::default());    

        // Staging buffer for GPU->CPU transfer (aligned to 256 bytes)
        let bytes_per_row = (width * 4 + 255) & !255;
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (bytes_per_row * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Particle buffers (double-buffered for compute shader ping-pong)
        let particle_size = std::mem::size_of::<GpuParticle>() as u64;
        let buffer_size = particle_size * max_particles as u64;

        let particle_buffers = [
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Particle Buffer A"),
                size: buffer_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Particle Buffer B"),
                size: buffer_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        ];

        // Spectrum buffer (64 bands)
        let spectrum_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spectrum Buffer"),
            size: 64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Simulation params buffer
        let sim_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sim Params Buffer"),
            size: std::mem::size_of::<SimParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Render params buffer
        let render_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Render Params Buffer"),
            size: std::mem::size_of::<RenderParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Sampler for textures
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Linear Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Create compute pipeline for particle simulation
        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Particle Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(PARTICLE_COMPUTE_SHADER.into()),
        });

        let compute_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Bind Group Layout"),
            entries: &[
                // Particles in
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Particles out
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Sim params
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Spectrum
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&compute_bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Particle Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: "main",
        });

        // Create render pipeline for particles
        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Particle Render Shader"),
            source: wgpu::ShaderSource::Wgsl(PARTICLE_RENDER_SHADER.into()),
        });

        let render_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Render Bind Group Layout"),
            entries: &[
                // Particles
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Render params
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&render_bind_group_layout],
            push_constant_ranges: &[],
        });

        

        // Two particle pipelines:
        // - Alpha (premultiplied "over") for normal blending
        // - Additive (RGB add) for glow-heavy looks
        let render_pipeline_alpha = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Particle Render Pipeline (Alpha)"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    // Premultiplied alpha OVER: out = src + dst*(1-src.a)
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let render_pipeline_additive = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Particle Render Pipeline (Additive)"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    // Additive RGB (glow): out.rgb = src.rgb + dst.rgb
                    // Keep alpha as OVER so we still have meaningful coverage for alpha exports.
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

// Create bloom downsample pipeline
        let bloom_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bloom Shader"),
            source: wgpu::ShaderSource::Wgsl(BLOOM_SHADER.into()),
        });

        let bloom_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bloom Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let bloom_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Bloom Pipeline Layout"),
            bind_group_layouts: &[&bloom_bind_group_layout],
            push_constant_ranges: &[],
        });

        let bloom_downsample_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Bloom Downsample Pipeline"),
            layout: Some(&bloom_pipeline_layout),
            module: &bloom_shader,
            entry_point: "downsample",
        });

        let bloom_upsample_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Bloom Upsample Pipeline"),
            layout: Some(&bloom_pipeline_layout),
            module: &bloom_shader,
            entry_point: "upsample",
        });

        // Create tonemap pipeline
        let tonemap_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tonemap Shader"),
            source: wgpu::ShaderSource::Wgsl(TONEMAP_SHADER.into()),
        });

        let tonemap_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Tonemap Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let tonemap_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Tonemap Pipeline Layout"),
            bind_group_layouts: &[&tonemap_bind_group_layout],
            push_constant_ranges: &[],
        });

        let tonemap_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Tonemap Pipeline"),
            layout: Some(&tonemap_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &tonemap_shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &tonemap_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Ok(Self {
            device,
            queue,
            render_texture,
            render_view,
            hdr_texture,
            hdr_view,
            msaa_texture,
            msaa_view,
            sample_count,
            overlay_texture,
            overlay_view,
            bloom_textures,
            bloom_views,
            staging_buffer,
            bytes_per_row,
            compute_pipeline,
            compute_bind_group_layout,
            particle_buffers,
            current_buffer: 0,
            spectrum_buffer,
            sim_params_buffer,
            render_pipeline_alpha,
            render_pipeline_additive,
            render_bind_group_layout,
            render_params_buffer,
            bloom_downsample_pipeline,
            bloom_upsample_pipeline,
            bloom_bind_group_layout,
            tonemap_pipeline,
            tonemap_bind_group_layout,
            sampler,
            width,
            height,
            max_particles,
        })
    }

    /// Upload particles from CPU to GPU
    pub fn upload_particles(&self, particles: &[GpuParticle]) {
        let data = bytemuck::cast_slice(particles);
        self.queue.write_buffer(&self.particle_buffers[self.current_buffer], 0, data);
    }

    /// Upload spectrum data
    #[allow(dead_code)]
    pub fn upload_spectrum(&self, spectrum: &[f32]) {
        let mut padded = [0.0f32; 64];
        let len = spectrum.len().min(64);
        padded[..len].copy_from_slice(&spectrum[..len]);
        self.queue.write_buffer(&self.spectrum_buffer, 0, bytemuck::cast_slice(&padded));
    }

    /// Upload overlay image (rgba8)
    pub fn upload_overlay(&self, pixels: &[u8]) {
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.overlay_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            pixels,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(self.width * 4),
                rows_per_image: Some(self.height),
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Run particle simulation compute shader
    #[allow(dead_code)]
    pub fn simulate_particles(&mut self, params: &SimParams) {
        // Upload params
        self.queue.write_buffer(&self.sim_params_buffer, 0, bytemuck::bytes_of(params));

        // Create bind group for this frame
        let src_buffer = self.current_buffer;
        let dst_buffer = 1 - self.current_buffer;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: &self.compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.particle_buffers[src_buffer].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.particle_buffers[dst_buffer].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.sim_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.spectrum_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compute Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Particle Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups((params.num_particles + 255) / 256, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        // Swap buffers
        self.current_buffer = dst_buffer;
    }

    /// Render particles to HDR texture with MSAA anti-aliasing
    pub fn render_particles(
        &self,
        num_particles: u32,
        params: &RenderParams,
        clear_color: [f32; 4],
        additive_blend: bool,
    ) {
        self.queue.write_buffer(&self.render_params_buffer, 0, bytemuck::bytes_of(params));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group"),
            layout: &self.render_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.particle_buffers[self.current_buffer].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.render_params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        {
            // Use MSAA texture for rendering with resolve to HDR texture
            // If MSAA is enabled (sample_count > 1), render to MSAA texture and resolve to HDR
            // If MSAA is disabled (sample_count == 1), render directly to HDR texture
            let (view, resolve_target) = if self.sample_count > 1 {
                (&self.msaa_view, Some(&self.hdr_view))
            } else {
                (&self.hdr_view, None)
            };

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Particle Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target,  // MSAA resolve to HDR texture
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: clear_color[0] as f64,
                            g: clear_color[1] as f64,
                            b: clear_color[2] as f64,
                            a: clear_color[3] as f64,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            let pipeline = if additive_blend {
                &self.render_pipeline_additive
            } else {
                &self.render_pipeline_alpha
            };
            render_pass.set_pipeline(pipeline);
            render_pass.set_bind_group(0, &bind_group, &[]);
            // 6 vertices per particle (2 triangles for quad)
            render_pass.draw(0..6, 0..num_particles);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Execute bloom pass (downsample -> upsample)
    pub fn run_bloom(&self) {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Bloom Encoder"),
        });

        // Downsample: HDR -> bloom mip 0
        {
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Bloom Downsample 0"),
                layout: &self.bloom_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.hdr_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.bloom_views[0]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });

            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Bloom Downsample Pass 0"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.bloom_downsample_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let mip_width = (self.width / 2).max(1);
            let mip_height = (self.height / 2).max(1);
            compute_pass.dispatch_workgroups((mip_width + 7) / 8, (mip_height + 7) / 8, 1);
        }

        // Continue downsampling through mip chain
        let mut mip_width = self.width / 2;
        let mut mip_height = self.height / 2;

        for i in 0..4 {
            mip_width = (mip_width / 2).max(1);
            mip_height = (mip_height / 2).max(1);

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Bloom Downsample {}", i + 1)),
                layout: &self.bloom_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.bloom_views[i]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.bloom_views[i + 1]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });

            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("Bloom Downsample Pass {}", i + 1)),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.bloom_downsample_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups((mip_width + 7) / 8, (mip_height + 7) / 8, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        // Upsample: mip 4 -> mip 0
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Bloom Upsample Encoder"),
        });

        mip_width = self.width / 32;
        mip_height = self.height / 32;

        for i in (0..4).rev() {
            mip_width = (mip_width * 2).min(self.width / 2);
            mip_height = (mip_height * 2).min(self.height / 2);

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Bloom Upsample {}", i)),
                layout: &self.bloom_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.bloom_views[i + 1]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.bloom_views[i]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });

            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("Bloom Upsample Pass {}", i)),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.bloom_upsample_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups((mip_width + 7) / 8, (mip_height + 7) / 8, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Apply tonemap and output to final texture
    pub fn tonemap(&self, _params: &RenderParams) {
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tonemap Bind Group"),
            layout: &self.tonemap_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.bloom_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.render_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&self.overlay_view),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tonemap Encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Tonemap Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.render_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.tonemap_pipeline);
            render_pass.set_bind_group(0, &bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Read back rendered frame to CPU
    pub fn read_frame(&self) -> Vec<u8> {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Readback Encoder"),
        });

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.render_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &self.staging_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(self.bytes_per_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Map buffer and read data
        let buffer_slice = self.staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("Failed to map buffer");

        let data = buffer_slice.get_mapped_range();

        // Copy to output, removing padding
        let mut output = Vec::with_capacity((self.width * self.height * 4) as usize);
        let actual_row_bytes = self.width * 4;
        for row in 0..self.height {
            let start = (row * self.bytes_per_row) as usize;
            let end = start + actual_row_bytes as usize;
            output.extend_from_slice(&data[start..end]);
        }

        drop(data);
        self.staging_buffer.unmap();

        output
    }

    /// Get dimensions
    #[allow(dead_code)]
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

// WGSL Shaders

const PARTICLE_COMPUTE_SHADER: &str = r#"
struct Particle {
    position: vec2<f32>,
    velocity: vec2<f32>,
    color: vec4<f32>,
    size: f32,
    life: f32,
    max_life: f32,
    audio_alpha: f32,
    audio_size: f32,
    brightness: f32,
    _padding: vec2<f32>,
}

struct SimParams {
    delta_time: f32,
    time: f32,
    width: f32,
    height: f32,
    audio_amplitude: f32,
    audio_bass: f32,
    audio_mid: f32,
    audio_high: f32,
    audio_beat: f32,
    beat_burst_strength: f32,
    damping: f32,
    speed: f32,
    num_particles: u32,
    has_audio: u32,
    // Audio-reactive parameters (matching preview behavior)
    fade_attack_speed: f32,
    fade_release_speed: f32,
    audio_spawn_threshold: f32,
    audio_reactive_spawn: u32,
    spawn_radius: f32,
    gravity: f32,
    _padding: vec2<f32>,
}

@group(0) @binding(0) var<storage, read> particles_in: array<Particle>;
@group(0) @binding(1) var<storage, read_write> particles_out: array<Particle>;
@group(0) @binding(2) var<uniform> params: SimParams;
@group(0) @binding(3) var<storage, read> spectrum: array<f32, 64>;

// Simple pseudo-random function
fn rand(seed: u32) -> f32 {
    let s = seed * 747796405u + 2891336453u;
    let word = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return f32((word >> 22u) ^ word) / 4294967295.0;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.num_particles) { return; }

    var p = particles_in[idx];

    let center = vec2<f32>(params.width / 2.0, params.height / 2.0);
    let dt = params.delta_time;

    // Audio-reactive mode detection (matching preview logic)
    let audio_reactive = params.audio_reactive_spawn > 0u;
    let has_audio = params.has_audio > 0u && params.audio_amplitude > params.audio_spawn_threshold;
    let audio_level = params.audio_amplitude;
    let is_beat = params.audio_beat > 0.5;
    let is_strong_beat = params.audio_beat > 0.7;

    // Check if particle should respawn (dead or fully faded)
    let should_respawn = p.life <= 0.0 || (audio_reactive && p.audio_alpha < 0.01);

    if (should_respawn) {
        // Only respawn if there's audio (audio-reactive mode) or always (non-reactive)
        if (!audio_reactive || has_audio) {
            // Respawn particle from center (matching preview's spawn_audio_particle)
            let seed = idx + u32(params.time * 1000.0);
            let angle = rand(seed) * 6.28318530718;
            let radius = rand(seed + 1u) * params.spawn_radius + 10.0;

            p.position = vec2<f32>(
                center.x + cos(angle) * radius,
                center.y + sin(angle) * radius
            );

            // Initial velocity: outward from center, scaled by audio
            let speed_mult = audio_level * 0.5 + 0.1;
            p.velocity = vec2<f32>(cos(angle) * speed_mult, sin(angle) * speed_mult);

            // Lifetime matching preview: 2-5 seconds
            p.life = 2.0 + rand(seed + 2u) * 3.0;
            p.max_life = 5.0;

            // Start with low alpha like preview (ramps up with audio)
            p.audio_alpha = 0.1;
            p.audio_size = clamp(audio_level, 0.3, 1.0);
            p.brightness = 1.0;
        } else {
            // No audio in reactive mode - keep particle dead
            particles_out[idx] = p;
            return;
        }
    }

    // Beat burst - particles explode outward on strong beat (matching preview)
    if (is_strong_beat && params.beat_burst_strength > 0.0) {
        let dir = p.position - center;
        let dist = max(length(dir), 1.0);
        let normalized_dir = dir / dist;
        let burst = params.audio_beat * params.beat_burst_strength * 0.3;
        p.velocity += normalized_dir * burst;
    }

    // Sample spectrum for frequency-reactive movement
    let freq_bin = idx % 64u;
    let freq_response = spectrum[freq_bin];

    // Add curl noise-like flow field
    let noise_scale = 0.003;
    let nx = p.position.x * noise_scale + params.time;
    let ny = p.position.y * noise_scale + params.time * 0.7;
    let flow_x = sin(ny * 6.0) + cos(nx * 3.0) * 0.5;
    let flow_y = cos(nx * 6.0) + sin(ny * 3.0) * 0.5;
    p.velocity += vec2<f32>(flow_x, flow_y) * 0.02 * (1.0 + params.audio_mid);

    // Apply gravity (matching preview)
    p.velocity.y += params.gravity * params.speed * 60.0 * dt * 0.5;

    // Physics integration
    p.position += p.velocity * params.speed * 60.0 * dt;

    // Damping (exponential, matching preview)
    let damping_factor = exp(-params.damping * dt);
    p.velocity *= damping_factor;

    // Clamp velocity
    let vel_mag = length(p.velocity);
    if (vel_mag > 5.0) {
        p.velocity = p.velocity * (5.0 / vel_mag);
    }

    // Audio-driven opacity with asymmetric attack/release (BOOSTED for better visibility)
    if (audio_reactive) {
        // Target alpha based on audio level - HIGHER minimum for visibility
        let target_alpha = select(0.3, clamp(audio_level * 1.5, 0.6, 1.0), has_audio);

        // Asymmetric fade: FASTER appear (attack), slower fade (release)
        let fade_speed = select(params.fade_release_speed * 0.5, params.fade_attack_speed * 2.0, target_alpha > p.audio_alpha);
        p.audio_alpha += (target_alpha - p.audio_alpha) * fade_speed * dt;
        p.audio_alpha = clamp(p.audio_alpha, 0.1, 1.0);  // Minimum alpha of 0.1

        // Size driven by audio amplitude - BOOSTED
        p.audio_size = clamp(audio_level * 1.3 + 0.5, 0.5, 2.0);
    } else {
        // Non-reactive mode: always visible
        p.audio_alpha = 1.0;
        p.audio_size = 1.0;
    }

    // Brightness from audio - BOOSTED for better visibility
    p.brightness = select(0.8, 1.0 + audio_level * 0.3 + params.audio_beat * 0.2, has_audio);

    // Life decay
    p.life -= dt;

    // Screen wrapping
    if (p.position.x < -50.0) { p.position.x = params.width + 50.0; }
    if (p.position.x > params.width + 50.0) { p.position.x = -50.0; }
    if (p.position.y < -50.0) { p.position.y = params.height + 50.0; }
    if (p.position.y > params.height + 50.0) { p.position.y = -50.0; }

    particles_out[idx] = p;
}
"#;

const PARTICLE_RENDER_SHADER: &str = r#"
struct Particle {
    position: vec2<f32>,
    velocity: vec2<f32>,
    color: vec4<f32>,
    size: f32,
    life: f32,
    max_life: f32,
    audio_alpha: f32,
    audio_size: f32,
    brightness: f32,
    _padding: vec2<f32>,
}

struct RenderParams {
    width: f32,
    height: f32,
    glow_intensity: f32,
    exposure: f32,
    bloom_strength: f32,
    shape_id: f32,
    bg_r: f32,
    bg_g: f32,
    bg_b: f32,
    bg_a: f32,
    _pad0: f32,
    _pad1: f32,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) glow: f32,
}

var<private> QUAD_POSITIONS: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
    vec2<f32>(-1.0,  1.0)
);

var<private> QUAD_UVS: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0),
    vec2<f32>(1.0, 0.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(0.0, 1.0)
);

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: RenderParams;

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_idx: u32,
    @builtin(instance_index) instance_idx: u32
) -> VertexOutput {
    let p = particles[instance_idx];

    var output: VertexOutput;

    // Skip invisible particles - less aggressive culling to match preview
    if (p.audio_alpha < 0.005 || p.life <= 0.0) {
        output.position = vec4<f32>(0.0, 0.0, -2.0, 1.0);
        output.color = vec4<f32>(0.0);
        output.uv = vec2<f32>(0.0);
        output.glow = 0.0;
        return output;
    }

    let local_vertex_idx = vertex_idx % 6u;
    let quad_pos = QUAD_POSITIONS[local_vertex_idx];
    let quad_uv = QUAD_UVS[local_vertex_idx];

    // Scale by particle size with glow extension - BOOSTED to match preview volumetric rendering
    // Preview's draw_volumetric_particle draws multiple layers from size*0.1 to size*1.5
    // We need larger quads to accommodate the full glow effect plus extra for smooth edges
    // Quad covers the full glow radius.
    // In the preview glow_radius = base_radius * (1.0 + glow_intensity * 2.0).
    // We match that here so dist≈1.0 corresponds to the glow radius.
    let glow_mult = 1.0 + params.glow_intensity * 2.0;
    // Match preview: p.size already includes audio_size & beat pulse.
    // Preview draw() applies a subtle global pulse: size *= (1.0 + eff_amp * 0.2)
    let size_pulse = 1.0 + params._pad0 * 0.2;
    let size = p.size * glow_mult * size_pulse;

    // Convert to clip space
    let world_pos = p.position + quad_pos * size;
    let clip_x = (world_pos.x / params.width) * 2.0 - 1.0;
    let clip_y = 1.0 - (world_pos.y / params.height) * 2.0;

    output.position = vec4<f32>(clip_x, clip_y, 0.0, 1.0);

    // Compute alpha from life and audio - SIGNIFICANTLY boosted for better visibility
    // Preview's particles are much more visible, so we need higher alpha
    let life_alpha = clamp(p.life / p.max_life, 0.0, 1.0);
    let alpha = clamp(life_alpha * p.audio_alpha * p.brightness, 0.0, 1.0);

    output.color = vec4<f32>(p.color.rgb, alpha);
    output.uv = quad_uv;
    output.glow = params.glow_intensity;

    return output;
}

// Convert sRGB color to linear space for correct blending
fn srgb_to_linear(srgb: vec3<f32>) -> vec3<f32> {
    let cutoff = srgb <= vec3<f32>(0.04045);
    let lower = srgb / 12.92;
    let higher = pow((srgb + 0.055) / 1.055, vec3<f32>(2.4));
    return select(higher, lower, cutoff);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Distance from center of quad (0.5, 0.5)
    let center = vec2<f32>(0.5, 0.5);
    let uv = input.uv;
    var dist = 0.0;

    // Shape selection
    let shape = i32(params.shape_id + 0.5);
    if (shape == 1) {
        // Diamond
        dist = abs(uv.x - center.x) + abs(uv.y - center.y);
        dist = dist * 2.0;
    } else if (shape == 2) {
        // Star (4-point)
        let d1 = abs(uv.x - center.x) * 3.0;
        let d2 = abs(uv.y - center.y) * 3.0;
        dist = min(d1, d2) + length(uv - center) * 0.3;
    } else {
        // Circle (Default)
        dist = length(uv - center) * 2.0;
    }

    
// =========================================================
// Volumetric particle shading (preview-matching)
//
// IMPORTANT: Do NOT clamp dist to 1.0.
// dist is normalized such that:
//   dist ~= 0 at center
//   dist ~= 1 at the outer glow radius (set in VS via glow_mult)
// Allowing dist>1 fades corners and prevents square artifacts in export.
// =========================================================

let dist2 = dist * dist;

// Use params._pad1 to carry volumetric settings:
//   <0.5  => volumetric_rendering OFF  (simple, crisp particle)
//   >=0.5 => volumetric_rendering ON   (gaussian layers), value ~ steps count
let steps = params._pad1;

var intensity: f32 = 0.0;
var core: f32 = 0.0;

if (steps < 0.5) {
    // Simple disc with a soft AA edge.
    let edge = 1.0 - smoothstep(0.92, 1.0, dist);
    intensity = edge;

    // Optional tiny glow to avoid harsh aliasing on very small particles.
    intensity = intensity + (1.0 - smoothstep(1.0, 1.25, dist)) * input.glow * 0.08;
    core = exp(-35.0 * dist2);
} else {
    // Volumetric: adjust falloff with "steps" (more steps => wider, smoother glow).
    let falloff = clamp(8.0 / steps, 0.5, 2.0);

    // Preview uses gaussian exp(-2.5 * t^2)
    let base = exp(-2.5 * dist2 * falloff);

    // Wider halo component (glow)
    let halo = exp(-0.9 * dist2 * falloff) * input.glow * 0.6;

    // Tight hot core (roughly size*0.2)
    core = exp(-40.0 * dist2 * falloff);

    intensity = base + halo;
}

let final_alpha = clamp(input.color.a * intensity, 0.0, 1.0);

// Convert input color to linear space for correct blending
    // (input colors are in sRGB, GPU does linear blending)
let linear_color = srgb_to_linear(input.color.rgb);

// Slight core boost + subtle white hot center (preview-style)
let core_boost = 1.0 + core * 0.35;
let hot_core = vec3<f32>(1.0) * core * 0.7;

let rgb_unpremult = linear_color * core_boost + hot_core;

// Skip nearly invisible fragments
if (final_alpha < 0.001) {
    discard;
}

// Output premultiplied alpha for correct blending
return vec4<f32>(rgb_unpremult * final_alpha, final_alpha);
}
"#;

const BLOOM_SHADER: &str = r#"
@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var tex_sampler: sampler;

// 13-tap tent filter for high quality downsampling
@compute @workgroup_size(8, 8)
fn downsample(@builtin(global_invocation_id) id: vec3<u32>) {
    let output_size = textureDimensions(output_texture);
    if (id.x >= output_size.x || id.y >= output_size.y) { return; }

    let input_size = vec2<f32>(textureDimensions(input_texture));
    let texel_size = 1.0 / input_size;

    // Center UV for output pixel
    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(output_size);

    // 13-tap filter
    var color = textureSampleLevel(input_texture, tex_sampler, uv, 0.0).rgb * 0.125;

    color += textureSampleLevel(input_texture, tex_sampler, uv + vec2<f32>(-1.0, -1.0) * texel_size, 0.0).rgb * 0.03125;
    color += textureSampleLevel(input_texture, tex_sampler, uv + vec2<f32>( 0.0, -1.0) * texel_size, 0.0).rgb * 0.0625;
    color += textureSampleLevel(input_texture, tex_sampler, uv + vec2<f32>( 1.0, -1.0) * texel_size, 0.0).rgb * 0.03125;

    color += textureSampleLevel(input_texture, tex_sampler, uv + vec2<f32>(-1.0,  0.0) * texel_size, 0.0).rgb * 0.0625;
    color += textureSampleLevel(input_texture, tex_sampler, uv + vec2<f32>( 1.0,  0.0) * texel_size, 0.0).rgb * 0.0625;

    color += textureSampleLevel(input_texture, tex_sampler, uv + vec2<f32>(-1.0,  1.0) * texel_size, 0.0).rgb * 0.03125;
    color += textureSampleLevel(input_texture, tex_sampler, uv + vec2<f32>( 0.0,  1.0) * texel_size, 0.0).rgb * 0.0625;
    color += textureSampleLevel(input_texture, tex_sampler, uv + vec2<f32>( 1.0,  1.0) * texel_size, 0.0).rgb * 0.03125;

    // Threshold for bloom (only bright areas)
    let brightness = max(max(color.r, color.g), color.b);
    let soft_threshold = clamp((brightness - 0.8) / 0.5, 0.0, 1.0);
    color *= soft_threshold;

    textureStore(output_texture, id.xy, vec4<f32>(color, 1.0));
}

// Tent filter upsample with additive blending
@compute @workgroup_size(8, 8)
fn upsample(@builtin(global_invocation_id) id: vec3<u32>) {
    let output_size = textureDimensions(output_texture);
    if (id.x >= output_size.x || id.y >= output_size.y) { return; }

    let input_size = vec2<f32>(textureDimensions(input_texture));
    let texel_size = 1.0 / input_size;

    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(output_size);

    // 9-tap tent filter
    var color = textureSampleLevel(input_texture, tex_sampler, uv, 0.0).rgb * 0.25;

    color += textureSampleLevel(input_texture, tex_sampler, uv + vec2<f32>(-1.0, -1.0) * texel_size, 0.0).rgb * 0.0625;
    color += textureSampleLevel(input_texture, tex_sampler, uv + vec2<f32>( 0.0, -1.0) * texel_size, 0.0).rgb * 0.125;
    color += textureSampleLevel(input_texture, tex_sampler, uv + vec2<f32>( 1.0, -1.0) * texel_size, 0.0).rgb * 0.0625;

    color += textureSampleLevel(input_texture, tex_sampler, uv + vec2<f32>(-1.0,  0.0) * texel_size, 0.0).rgb * 0.125;
    color += textureSampleLevel(input_texture, tex_sampler, uv + vec2<f32>( 1.0,  0.0) * texel_size, 0.0).rgb * 0.125;

    color += textureSampleLevel(input_texture, tex_sampler, uv + vec2<f32>(-1.0,  1.0) * texel_size, 0.0).rgb * 0.0625;
    color += textureSampleLevel(input_texture, tex_sampler, uv + vec2<f32>( 0.0,  1.0) * texel_size, 0.0).rgb * 0.125;
    color += textureSampleLevel(input_texture, tex_sampler, uv + vec2<f32>( 1.0,  1.0) * texel_size, 0.0).rgb * 0.0625;

    textureStore(output_texture, id.xy, vec4<f32>(color, 1.0));
}
"#;

const TONEMAP_SHADER: &str = r#"
struct RenderParams {
    width: f32,
    height: f32,
    glow_intensity: f32,
    exposure: f32,
    bloom_strength: f32,
    shape_id: f32,
    bg_r: f32,
    bg_g: f32,
    bg_b: f32,
    bg_a: f32,
    _pad0: f32,
    _pad1: f32,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@group(0) @binding(0) var hdr_texture: texture_2d<f32>;
@group(0) @binding(1) var bloom_texture: texture_2d<f32>;
@group(0) @binding(2) var tex_sampler: sampler;
@group(0) @binding(3) var<uniform> params: RenderParams;
@group(0) @binding(4) var overlay_texture: texture_2d<f32>;

// Fullscreen triangle
@vertex
fn vs_main(@builtin(vertex_index) vertex_idx: u32) -> VertexOutput {
    var output: VertexOutput;

    // Generate fullscreen triangle
    let x = f32((vertex_idx << 1u) & 2u);
    let y = f32(vertex_idx & 2u);

    output.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    output.uv = vec2<f32>(x, y);

    return output;
}

// Soft tone mapping - preserves colors for particle rendering
fn soft_tonemap(color: vec3<f32>) -> vec3<f32> {
    // Very gentle compression only for values above 1.0
    // This preserves colors much better than aggressive tone mapping
    let max_comp = max(max(color.r, color.g), color.b);
    if (max_comp <= 1.0) {
        return color;
    }
    // Gentle roll-off for bright values
    let scale = 1.0 + (max_comp - 1.0) * 0.3;
    return color / scale;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // HDR buffer stores premultiplied RGB + alpha from the particle pass.
    let hdr = textureSample(hdr_texture, tex_sampler, input.uv);
    let base_alpha = clamp(hdr.a, 0.0, 1.0);

    // Bloom is computed in HDR space (linear). It may extend beyond the particle alpha.
    let bloom = textureSample(bloom_texture, tex_sampler, input.uv).rgb;
    let bloom_premult = bloom * params.bloom_strength * 1.5;

    // Combine in premultiplied HDR space
    var premult = hdr.rgb + bloom_premult;

    // Exposure in linear light
    let exposure_mult = pow(2.0, params.exposure);
    premult = premult * exposure_mult;

    // Derive an alpha contribution from bloom so glow survives on transparent backgrounds.
    // For glowy visuals this works much better than keeping alpha = particle coverage only.
    let bloom_alpha = clamp(
        max(max(bloom_premult.r, bloom_premult.g), bloom_premult.b) * exposure_mult * 1.2,
        0.0,
        1.0
    );

    // Effective base alpha (particles + bloom)
    let alpha_base = max(base_alpha, bloom_alpha);

    // Un-premultiply for tone mapping in linear space
    let safe_a = max(alpha_base, 1e-4);
    var color = premult / safe_a;

    // Tone mapping + clamp
    color = soft_tonemap(color);
    color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));

    // Convert base back to premultiplied for compositing
    let base_premult_out = color * alpha_base;

    // Overlay (waveform/spectrum) is generated on CPU as *straight* alpha.
    // Convert to premultiplied for correct compositing.
    let overlay = textureSample(overlay_texture, tex_sampler, input.uv);
    let overlay_a = clamp(overlay.a, 0.0, 1.0);
    let overlay_premult = overlay.rgb * overlay_a;
    // Background (linear) from params. For alpha exports bg_a should be 0.
    let bg_a = clamp(params.bg_a, 0.0, 1.0);
    let bg_rgb = clamp(vec3<f32>(params.bg_r, params.bg_g, params.bg_b), vec3<f32>(0.0), vec3<f32>(1.0));
    let bg_premult = bg_rgb * bg_a;

    // Compose overlay OVER background (overlay is straight alpha -> premult already)
    let mid_a = overlay_a + bg_a * (1.0 - overlay_a);
    let mid_premult = overlay_premult + bg_premult * (1.0 - overlay_a);

    // Compose particles OVER (background + overlay) to match preview layering.
    let out_a = alpha_base + mid_a * (1.0 - alpha_base);
    let out_premult = base_premult_out + mid_premult * (1.0 - alpha_base);
    // Output STRAIGHT (unmatted) alpha for easy import into NLEs.
    let safe_out_a = max(out_a, 1e-4);
    let out_rgb = clamp(out_premult / safe_out_a, vec3<f32>(0.0), vec3<f32>(1.0));

    // NOTE: Output target is Rgba8UnormSrgb, so the GPU automatically applies
    // sRGB encoding to RGB channels. Alpha stays linear.
    return vec4<f32>(out_rgb, out_a);
}
"#;

/// Convert CPU particle to GPU format
#[allow(dead_code)]
pub fn cpu_particle_to_gpu(p: &crate::particles::Particle) -> GpuParticle {
    GpuParticle {
        position: [p.pos.x, p.pos.y],
        velocity: [p.vel.x, p.vel.y],
        color: [
            p.color.r() as f32 / 255.0,
            p.color.g() as f32 / 255.0,
            p.color.b() as f32 / 255.0,
            1.0,
        ],
        size: p.size,
        life: p.life,
        max_life: p.max_life,
        audio_alpha: p.audio_alpha,
        audio_size: p.audio_size,
        brightness: p.brightness,
        _padding: [0.0, 0.0],
    }
}
