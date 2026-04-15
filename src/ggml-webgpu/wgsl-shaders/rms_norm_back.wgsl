// RMS_NORM_BACK f32: backward of rms_norm.
//   src0 = dz (grad from output) [nc, n_rows]
//   src1 = x  (forward input)    [nc, n_rows]
//   dst  = dx                    [nc, n_rows]
//
// Math (per row):
//   sum_xx  = sum(x*x);   sum_xdz = sum(x*dz)
//   mean_eps = sum_xx / N + eps;  sum_eps = sum_xx + eps*N
//   rrms = 1 / sqrt(mean_eps)
//   dx[i] = (x[i] * (-sum_xdz / sum_eps) + dz[i]) * rrms
//
// One workgroup per row; threads stride over the contracting dim with a
// workgroup-shared two-way reduction.

struct Params {
    nc:          u32,
    n_rows:      u32,
    offset_dz:   u32,
    offset_x:    u32,
    offset_dst:  u32,
    nb_dz_row:   u32,
    nb_x_row:    u32,
    nb_dst_row:  u32,
    eps:         f32,
};

@group(0) @binding(0) var<storage, read_write> dz:  array<f32>;
@group(0) @binding(1) var<storage, read_write> xv:  array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> sxx:  array<f32, WG_SIZE>;
var<workgroup> sxdz: array<f32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x;
    if (row >= params.n_rows) {
        return;
    }
    let dz_base  = params.offset_dz  + row * params.nb_dz_row;
    let x_base   = params.offset_x   + row * params.nb_x_row;
    let dst_base = params.offset_dst + row * params.nb_dst_row;

    let elems = (params.nc + WG_SIZE - 1u) / WG_SIZE;

    var partial_xx  = 0.0;
    var partial_xdz = 0.0;
    var col = lid.x;
    for (var j: u32 = 0u; j < elems; j = j + 1u) {
        if (col < params.nc) {
            let xc  = xv[x_base + col];
            let dzc = dz[dz_base + col];
            partial_xx  = partial_xx  + xc * xc;
            partial_xdz = partial_xdz + xc * dzc;
        }
        col = col + WG_SIZE;
    }
    sxx[lid.x]  = partial_xx;
    sxdz[lid.x] = partial_xdz;
    workgroupBarrier();
    var off: u32 = WG_SIZE / 2u;
    while (off > 0u) {
        if (lid.x < off) {
            sxx[lid.x]  = sxx[lid.x]  + sxx[lid.x  + off];
            sxdz[lid.x] = sxdz[lid.x] + sxdz[lid.x + off];
        }
        off = off / 2u;
        workgroupBarrier();
    }
    let sum_xx  = sxx[0];
    let sum_xdz = sxdz[0];

    let n        = f32(params.nc);
    let sum_eps  = sum_xx + params.eps * n;
    let mean_eps = sum_xx / n + params.eps;
    let rrms     = 1.0 / sqrt(mean_eps);
    let coeff    = -sum_xdz / sum_eps;

    col = lid.x;
    for (var j: u32 = 0u; j < elems; j = j + 1u) {
        if (col < params.nc) {
            dst[dst_base + col] = (xv[x_base + col] * coeff + dz[dz_base + col]) * rrms;
        }
        col = col + WG_SIZE;
    }
}
