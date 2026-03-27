use paged_infer::math::{matvec_f32_weight_transposed_parallel, pack_bf16_to_f32};
use std::time::Instant;

fn main() {
    pollster::block_on(run());
}

async fn run() {
    // ── synthetic data (same size as CPU benchmark) ──────────────────────────
    let rows: usize = 2048;
    let cols: usize = 2048;
    let iters: usize = 20;
    let warmup: usize = 5;

    let mut w_bf16 = vec![0u8; rows * cols * 2];
    for i in 0..rows * cols {
        let b = half::bf16::from_f32(((i % 97) as f32) * 0.001).to_le_bytes();
        w_bf16[i * 2] = b[0];
        w_bf16[i * 2 + 1] = b[1];
    }
    let weight_f32 = pack_bf16_to_f32(&w_bf16);
    let x: Vec<f32> = (0..cols).map(|i| i as f32 * 0.001).collect();

    // ── CPU reference ────────────────────────────────────────────────────────
    let mut cpu_out = vec![0.0f32; rows];
    // warmup
    for _ in 0..warmup {
        matvec_f32_weight_transposed_parallel(&mut cpu_out, &x, &weight_f32, rows, cols);
    }
    let t_cpu = Instant::now();
    for _ in 0..iters {
        matvec_f32_weight_transposed_parallel(&mut cpu_out, &x, &weight_f32, rows, cols);
    }
    let cpu_total = t_cpu.elapsed().as_secs_f64();

    // ── wgpu init ────────────────────────────────────────────────────────────
    let instance = wgpu::Instance::default();
    let adapter = match instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
    {
        Ok(a) => a,
        Err(_) => {
            println!("No GPU adapter found — skipping GPU benchmark.");
            println!("(Run on a machine with a GPU / Metal / Vulkan to see GPU results.)");
            println!();
            println!(
                "CPU packed+parallel reference: {:.4}s / {:.2}ms per iter",
                cpu_total,
                cpu_total * 1000.0 / iters as f64
            );
            return;
        }
    };

    let info = adapter.get_info();
    println!("GPU Matvec Benchmark");
    println!("====================");
    println!("Adapter : {} ({:?})", info.name, info.backend);
    println!(
        "Matrix  : {}×{} f32, {} warm-up + {} timed iters",
        rows, cols, warmup, iters
    );
    println!();

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await
        .expect("failed to create device");

    // ── buffers ───────────────────────────────────────────────────────────────
    use wgpu::util::DeviceExt;

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct Dims {
        rows: u32,
        cols: u32,
    }

    let dims = Dims {
        rows: rows as u32,
        cols: cols as u32,
    };

    let weight_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("weight"),
        contents: bytemuck::cast_slice(&weight_f32),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let x_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("x"),
        contents: bytemuck::cast_slice(&x),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output"),
        size: (rows * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: (rows * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let dims_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("dims"),
        contents: bytemuck::bytes_of(&dims),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // ── shader + pipeline ────────────────────────────────────────────────────
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("matvec"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/matvec.wgsl").into()),
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

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("matvec"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
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

    // ── warmup ────────────────────────────────────────────────────────────────
    for _ in 0..warmup {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(rows as u32, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
    }
    device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    }).ok();

    // ── timed GPU iterations ──────────────────────────────────────────────────
    let t_gpu = Instant::now();
    for _ in 0..iters {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(rows as u32, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
    }
    device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    }).ok();
    let gpu_compute_total = t_gpu.elapsed().as_secs_f64();

    // ── one final run with readback for correctness check ────────────────────
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(rows as u32, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, (rows * 4) as u64);
    queue.submit(std::iter::once(encoder.finish()));

    let slice = staging_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    }).ok();
    rx.recv().unwrap().unwrap();
    let data = slice.get_mapped_range();
    let gpu_out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging_buf.unmap();

    // ── report ────────────────────────────────────────────────────────────────
    let gpu_ms_per = gpu_compute_total * 1000.0 / iters as f64;
    let cpu_ms_per = cpu_total * 1000.0 / iters as f64;
    let speedup = cpu_total / gpu_compute_total;

    let max_err = cpu_out
        .iter()
        .zip(gpu_out.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0_f32, f32::max);

    println!("Results");
    println!("-------");
    println!(
        "CPU packed+parallel : {:.4}s total  ({:.3}ms / iter)",
        cpu_total, cpu_ms_per
    );
    println!(
        "GPU wgpu/Metal      : {:.4}s total  ({:.3}ms / iter)",
        gpu_compute_total, gpu_ms_per
    );
    println!("GPU speedup         : {:.1}x", speedup);
    println!();
    println!(
        "Correctness: max |CPU - GPU| = {:.2e}  {}",
        max_err,
        if max_err < 1e-2 { "✓" } else { "✗ (check kernel)" }
    );
    println!();
    println!("Note: GPU time excludes readback (weights stay on GPU in production).");
    println!("      M3 Unified Memory means no PCIe transfer — CPU/GPU share physical memory.");
}
