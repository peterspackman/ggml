// GET_ROWS_BACK f32: scatter-add gradient rows back to original positions.
//   src0: gradient for selected rows, treated as flat [nr, nc] with stride nb01
//   src1: row indices (i32), nelements = nr; may be a non-contig view (use 2D strides)
//   dst : accumulated gradient [nc, n_dst_rows] (pre-zeroed by caller)
//
// WGSL has no atomic<f32>, so we accumulate via CAS loop over u32-aliased dst.
// Dispatch: one thread per element of src0 (nc * nr elements), 1D.

struct Params {
    offset_src0: u32,        // in f32 elements
    offset_idx:  u32,        // in i32 elements
    offset_dst:  u32,        // in f32 elements

    nc:          u32,        // row length (columns)
    nr:          u32,        // total number of source rows = nelements(src1)
    n_dst_rows:  u32,
    nb01:        u32,        // src0 row stride  (in f32 elements)
    nb1:         u32,        // dst  row stride  (in f32 elements)

    idx_ne0:     u32,        // src1 ne[0]
    idx_nb0:     u32,        // src1 nb[0] (in i32 elements)
    idx_nb1:     u32,        // src1 nb[1] (in i32 elements)
};

@group(0) @binding(0) var<storage, read_write> src0:    array<f32>;
@group(0) @binding(1) var<storage, read_write> indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> dst:     array<atomic<u32>>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.nc * params.nr;
    if (gid.x >= n) {
        return;
    }
    let col     = gid.x % params.nc;
    let src_row = gid.x / params.nc;

    // Decompose src_row into (i0, i1) within the (possibly viewed) indices tensor.
    let i1 = src_row / params.idx_ne0;
    let i0 = src_row % params.idx_ne0;
    let dst_row = indices[params.offset_idx + i0 * params.idx_nb0 + i1 * params.idx_nb1];
    if (dst_row < 0 || u32(dst_row) >= params.n_dst_rows) {
        return;
    }

    let val = src0[params.offset_src0 + col + src_row * params.nb01];
    if (val == 0.0) {
        return;
    }

    let ptr = params.offset_dst + col + u32(dst_row) * params.nb1;

    loop {
        let old_bits = atomicLoad(&dst[ptr]);
        let new_bits = bitcast<u32>(bitcast<f32>(old_bits) + val);
        let res = atomicCompareExchangeWeak(&dst[ptr], old_bits, new_bits);
        if (res.exchanged) {
            break;
        }
    }
}
