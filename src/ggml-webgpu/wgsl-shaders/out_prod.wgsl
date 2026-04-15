// OUT_PROD f32:
//   src0: [ne00, ne01, ne02, ne03]
//   src1: [ne10, ne11, ne12, ne13]   (ne01 == ne11 is the contracting dim)
//   dst : [ne0,  ne1,  ne2,  ne3 ]   = [ne00, ne10, ne12, ne13]
//   broadcast: src0 dims 2,3 broadcast over dst dims 2,3 by ne2/ne02 and ne3/ne03
//
// One workgroup per (i1, i2, i3) row of dst. Threads in the workgroup
// stride over i0.

struct Params {
    offset_src0: u32,   // in elements
    offset_src1: u32,
    offset_dst:  u32,

    nb00: u32, nb01: u32, nb02: u32, nb03: u32,   // src0 strides (in elements)
    nb10: u32, nb11: u32, nb12: u32, nb13: u32,   // src1 strides
    nb1:  u32, nb2:  u32, nb3:  u32,              // dst  strides

    ne0: u32, ne1: u32, ne2: u32, ne3: u32,
    ne01: u32,                                    // contracting dim length
    ne02: u32, ne03: u32,                         // src0 outer dims (broadcast)
};

@group(0) @binding(0) var<storage, read_write> src0: array<f32>;
@group(0) @binding(1) var<storage, read_write> src1: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst:  array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    // wid.x encodes (i1, i2, i3) flattened: i1 + i2*ne1 + i3*(ne1*ne2)
    var w = wid.x;
    let i1 = w % params.ne1;
    w = w / params.ne1;
    let i2 = w % params.ne2;
    let i3 = w / params.ne2;

    if (i3 >= params.ne3) {
        return;
    }

    let dps2 = params.ne2 / params.ne02;
    let dps3 = params.ne3 / params.ne03;
    let i02  = i2 / dps2;
    let i03  = i3 / dps3;

    let dst_row_base = params.offset_dst + i1 * params.nb1 + i2 * params.nb2 + i3 * params.nb3;
    let src0_base    = params.offset_src0 + i02 * params.nb02 + i03 * params.nb03;
    let src1_base    = params.offset_src1 + i1 * params.nb10 + i2 * params.nb12 + i3 * params.nb13;

    var i0: u32 = lid.x;
    while (i0 < params.ne0) {
        var sum = 0.0;
        for (var k: u32 = 0u; k < params.ne01; k = k + 1u) {
            let a = src0[src0_base + i0 * params.nb00 + k * params.nb01];
            let b = src1[src1_base + k * params.nb11];
            sum = sum + a * b;
        }
        dst[dst_row_base + i0] = sum;
        i0 = i0 + WG_SIZE;
    }
}
