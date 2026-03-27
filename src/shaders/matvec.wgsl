// Parallel row matvec: output[row] = sum(weight[row,:] * x[:])
// Uses vec4<f32> loads + dot() for 4x higher memory bandwidth utilization.
// Each workgroup (256 threads) handles ONE output row.
// Threads stride over vec4 groups, then a tree reduction collapses 256 partial sums.
//
// Requires: dims.cols divisible by 4 (true for all TinyLlama/Llama projection dims).

struct Dims {
    rows: u32,
    cols: u32,
}

@group(0) @binding(0) var<storage, read> weight: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> x_vec: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

var<workgroup> partial_sums: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let row: u32 = wg_id.x;
    let lid: u32 = local_id.x;
    // Number of vec4 groups per row (cols / 4)
    let cols4: u32 = dims.cols / 4u;

    var acc: f32 = 0.0;

    // Each thread strides through vec4 groups — 4x fewer iterations than scalar.
    // dot(vec4, vec4) is a single MAD-4 instruction on Metal.
    var c: u32 = lid;
    loop {
        if c >= cols4 { break; }
        acc = acc + dot(weight[row * cols4 + c], x_vec[c]);
        c = c + 256u;
    }

    partial_sums[lid] = acc;
    workgroupBarrier();

    // Tree reduction: 256 → 1
    if lid < 128u { partial_sums[lid] = partial_sums[lid] + partial_sums[lid + 128u]; }
    workgroupBarrier();
    if lid < 64u  { partial_sums[lid] = partial_sums[lid] + partial_sums[lid + 64u]; }
    workgroupBarrier();
    if lid < 32u  { partial_sums[lid] = partial_sums[lid] + partial_sums[lid + 32u]; }
    workgroupBarrier();
    if lid < 16u  { partial_sums[lid] = partial_sums[lid] + partial_sums[lid + 16u]; }
    workgroupBarrier();
    if lid < 8u   { partial_sums[lid] = partial_sums[lid] + partial_sums[lid + 8u]; }
    workgroupBarrier();
    if lid < 4u   { partial_sums[lid] = partial_sums[lid] + partial_sums[lid + 4u]; }
    workgroupBarrier();
    if lid < 2u   { partial_sums[lid] = partial_sums[lid] + partial_sums[lid + 2u]; }
    workgroupBarrier();
    if lid < 1u   { partial_sums[lid] = partial_sums[lid] + partial_sums[lid + 1u]; }
    workgroupBarrier();

    if lid == 0u && row < dims.rows {
        output[row] = partial_sums[0];
    }
}
