// Parallel row matvec: output[row] = sum(weight[row,:] * x[:])
// Each workgroup (256 threads) handles ONE output row.
// Threads cooperatively compute a partial dot product then reduce.

struct Dims {
    rows: u32,
    cols: u32,
}

@group(0) @binding(0) var<storage, read> weight: array<f32>;
@group(0) @binding(1) var<storage, read> x_vec: array<f32>;
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
    let cols: u32 = dims.cols;

    var acc: f32 = 0.0;

    // Each thread strides through its columns
    var c: u32 = lid;
    loop {
        if c >= cols { break; }
        acc = acc + weight[row * cols + c] * x_vec[c];
        c = c + 256u;
    }

    partial_sums[lid] = acc;
    workgroupBarrier();

    // Tree reduction: 256 -> 1
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
