//! GPU compute context for wgpu/Metal-backed linear projections.
//!
//! `GpuContext` owns the device, queue, and compiled compute pipeline.
//! `GpuLinear` uploads a weight matrix once at construction; each `apply()`
//! call streams the input vector to the GPU, dispatches the workgroup-parallel
//! matvec kernel, and reads results back — blocking until done.
//!
//! On Apple Silicon (M-series) all memory is physically unified, so
//! write_buffer / map_read involve no PCIe transfers.

use wgpu::util::DeviceExt;

// ── GPU pipeline context ──────────────────────────────────────────────────────

pub struct GpuContext {
    pub device:       wgpu::Device,
    pub queue:        wgpu::Queue,
    pub(crate) pipeline: wgpu::ComputePipeline,
    pub(crate) bgl:      wgpu::BindGroupLayout,
}

impl GpuContext {
    /// Synchronously initialise wgpu, compile the WGSL kernel and return a
    /// ready context, or `None` if no GPU adapter is available.
    pub fn new() -> Option<Self> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Option<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok()?;

        let required_limits = adapter.limits(); // use Metal/Vulkan native caps
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits,
                ..wgpu::DeviceDescriptor::default()
            })
            .await
            .ok()?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matvec"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/matvec.wgsl").into(),
            ),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });

        let pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("matvec"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        Some(Self { device, queue, pipeline, bgl })
    }
}

// ── Per-weight GPU linear layer ───────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ShaderDims {
    rows: u32,
    cols: u32,
}

/// A weight matrix permanently resident in GPU memory.
/// Input vectors are written per call; output is read back synchronously.
pub struct GpuLinear {
    pub rows: usize,
    pub cols: usize,
    // kept alive so the bind_group reference stays valid
    _weight_buf: wgpu::Buffer,
    _dims_buf:   wgpu::Buffer,
    x_buf:       wgpu::Buffer,
    output_buf:  wgpu::Buffer,
    staging_buf: wgpu::Buffer,
    bind_group:  wgpu::BindGroup,
}

impl GpuLinear {
    /// Upload `weight` (rows × cols f32 values) to a GPU storage buffer and
    /// prepare per-call buffers.  `cols` must be divisible by 4.
    pub fn new(ctx: &GpuContext, rows: usize, cols: usize, weight: &[f32]) -> Self {
        assert!(cols % 4 == 0, "cols must be divisible by 4 for vec4 matvec shader");

        let weight_buf =
            ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gpu_linear_weight"),
                contents: bytemuck::cast_slice(weight),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let x_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_linear_x"),
            size: (cols * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let output_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_linear_output"),
            size: (rows * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_linear_staging"),
            size: (rows * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let dims = ShaderDims { rows: rows as u32, cols: cols as u32 };
        let dims_buf =
            ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gpu_linear_dims"),
                contents: bytemuck::bytes_of(&dims),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &ctx.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: weight_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dims_buf.as_entire_binding(),
                },
            ],
        });

        Self {
            rows,
            cols,
            _weight_buf: weight_buf,
            _dims_buf: dims_buf,
            x_buf,
            output_buf,
            staging_buf,
            bind_group,
        }
    }

    /// Compute `out = weight × x` on the GPU, blocking until the result is
    /// available in `out`.
    pub fn apply(&self, ctx: &GpuContext, out: &mut [f32], x: &[f32]) {
        // Stream the input vector.
        ctx.queue.write_buffer(&self.x_buf, 0, bytemuck::cast_slice(x));

        // Dispatch: one workgroup per output row, 256 threads per workgroup.
        let mut encoder = ctx.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor::default(),
        );
        {
            let mut pass = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                },
            );
            pass.set_pipeline(&ctx.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(self.rows as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &self.output_buf, 0,
            &self.staging_buf, 0,
            (self.rows * 4) as u64,
        );
        ctx.queue.submit(std::iter::once(encoder.finish()));

        // Synchronous readback.
        let slice = self.staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        ctx.device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
            .ok();
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        out.copy_from_slice(bytemuck::cast_slice(&data));
        drop(data);
        self.staging_buf.unmap();
    }
}
