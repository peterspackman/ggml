// SOFT_MAX_BACK f32: backward of softmax.
//   src0 = dy [nc, nr_rows]
//   src1 = y  [nc, nr_rows]   (forward output of softmax)
//   dst  = dx [nc, nr_rows]
// Per row: dot = sum_i(y[i] * dy[i]); dx[i] = scale * (dy[i] - dot) * y[i].
//
// One workgroup per row; threads in the workgroup stride over columns.

struct Params {
    nc:          u32,                   // row length
    n_rows:      u32,                   // total rows = ne1 * ne2 * ne3
    offset_dy:   u32,
    offset_y:    u32,
    offset_dst:  u32,
    nb_dy_row:   u32,                   // dy row stride (in f32 elements)
    nb_y_row:    u32,                   // y  row stride
    nb_dst_row:  u32,                   // dst row stride
    scale:       f32,
};

@group(0) @binding(0) var<storage, read_write> dy:  array<f32>;
@group(0) @binding(1) var<storage, read_write> yv:  array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> scratch: array<f32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x;
    if (row >= params.n_rows) {
        return;
    }
    let dy_base  = params.offset_dy  + row * params.nb_dy_row;
    let y_base   = params.offset_y   + row * params.nb_y_row;
    let dst_base = params.offset_dst + row * params.nb_dst_row;

    let elems = (params.nc + WG_SIZE - 1) / WG_SIZE;

    // Reduction: dot = sum(y * dy)
    var partial = 0.0;
    var col = lid.x;
    for (var j: u32 = 0u; j < elems; j = j + 1u) {
        if (col < params.nc) {
            partial = partial + yv[y_base + col] * dy[dy_base + col];
        }
        col = col + WG_SIZE;
    }
    scratch[lid.x] = partial;
    workgroupBarrier();
    var off: u32 = WG_SIZE / 2u;
    while (off > 0u) {
        if (lid.x < off) {
            scratch[lid.x] = scratch[lid.x] + scratch[lid.x + off];
        }
        off = off / 2u;
        workgroupBarrier();
    }
    let dot = scratch[0];

    // Apply: dx = scale * (dy - dot) * y
    col = lid.x;
    for (var j: u32 = 0u; j < elems; j = j + 1u) {
        if (col < params.nc) {
            let dyc = dy[dy_base + col];
            let yc  = yv[y_base + col];
            dst[dst_base + col] = params.scale * (dyc - dot) * yc;
        }
        col = col + WG_SIZE;
    }
}
