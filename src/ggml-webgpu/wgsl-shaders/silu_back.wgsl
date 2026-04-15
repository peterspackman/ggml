// SILU_BACK f32: backward of SiLU.
// dst[i] = grad[i] * sigmoid(x[i]) * (1 + x[i] * (1 - sigmoid(x[i])))
//   src0 = grad (dy)
//   src1 = x   (forward input)
//   dst  = dx

struct Params {
    n:           u32,
    offset_grad: u32,
    offset_x:    u32,
    offset_dst:  u32,
};

@group(0) @binding(0) var<storage, read_write> grad: array<f32>;
@group(0) @binding(1) var<storage, read_write> xv:   array<f32>;
@group(0) @binding(2) var<storage, read_write> dst:  array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.n) {
        return;
    }
    let x  = xv[params.offset_x + gid.x];
    let dy = grad[params.offset_grad + gid.x];
    let s  = 1.0 / (1.0 + exp(-x));
    dst[params.offset_dst + gid.x] = dy * s * (1.0 + x * (1.0 - s));
}
