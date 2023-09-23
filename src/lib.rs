use std::borrow::Cow;
use wasm_bindgen::prelude::*;
use wgpu::util::DeviceExt;

async fn run(tid: u16, sid: u16) -> Option<String> {
    let results = execute_gpu((tid as u32) | (sid as u32) << 16).await?;

    let count = results[0] as usize;
    let seeds = results[1..count + 1]
        .iter()
        .map(|&n| n.to_string())
        .collect::<Vec<String>>()
        .join(",");

    Some(seeds)
}

async fn execute_gpu(id32: u32) -> Option<Vec<u32>> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await.expect("GPU failed to initialize. Is WebGPU supported by your browser?");
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        )
        .await
        .unwrap();

    execute_search(&device, &queue, id32).await
}

async fn execute_search(device: &wgpu::Device, queue: &wgpu::Queue, id32: u32) -> Option<Vec<u32>> {
    let input = &[id32];
    // 10 results is a generous overestimate
    let output = &[0u32; 10];
    let output_size = std::mem::size_of_val(output) as wgpu::BufferAddress;
    let host_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: output_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(input),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let output_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(output),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
    });
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader_module,
        entry_point: "main",
    });

    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut compute_pass =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(1024 / 4, 1024 / 4, 4096 / 16);
    }

    encoder.copy_buffer_to_buffer(&output_buffer, 0, &host_buffer, 0, output_size);

    queue.submit(Some(encoder.finish()));

    let buffer_slice = host_buffer.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    device.poll(wgpu::Maintain::Wait);

    if let Some(Ok(())) = receiver.receive().await {
        let data = buffer_slice.get_mapped_range();
        let result = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        host_buffer.unmap();
        Some(result)
    } else {
        panic!("failed to run compute on gpu!")
    }
}

#[wasm_bindgen]
pub fn init_gen4ids() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init().expect("could not initialize logger");
}

#[wasm_bindgen]
pub async fn search(tid: u16, sid: u16) -> String {
    let result = run(tid, sid).await;
    if result.is_none() {
        panic!("Search failure")
    } else {
        result.unwrap()
    }
}
