// REPEAT_BACK f32: backward pass of repeat. Sum contributions from a
// tiled (larger) src0 back into a smaller dst.
//   src0: [ne00, ne01, ne02, ne03]   (larger)
//   dst : [ne0,  ne1,  ne2,  ne3 ]   (smaller, ne0i = nri * nei)
//
// One workgroup per (k1, k2, k3) row of dst. Threads stride over k0.

struct Params {
    offset_src0: u32,
    offset_dst:  u32,

    nb00: u32, nb01: u32, nb02: u32, nb03: u32,   // src0 strides (in elements)
    nb0:  u32, nb1:  u32, nb2:  u32, nb3:  u32,   // dst  strides

    ne00: u32, ne01: u32, ne02: u32, ne03: u32,   // src0 shape
    ne0:  u32, ne1:  u32, ne2:  u32, ne3:  u32,   // dst  shape
};

@group(0) @binding(0) var<storage, read_write> src0: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst:  array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    var w  = wid.x;
    let k1 = w % params.ne1; w = w / params.ne1;
    let k2 = w % params.ne2;
    let k3 = w / params.ne2;

    if (k3 >= params.ne3) {
        return;
    }

    let nr0 = params.ne00 / params.ne0;
    let nr1 = params.ne01 / params.ne1;
    let nr2 = params.ne02 / params.ne2;
    let nr3 = params.ne03 / params.ne3;

    var k0: u32 = lid.x;
    while (k0 < params.ne0) {
        var sum = 0.0;
        for (var i3: u32 = 0u; i3 < nr3; i3 = i3 + 1u) {
            for (var i2: u32 = 0u; i2 < nr2; i2 = i2 + 1u) {
                for (var i1: u32 = 0u; i1 < nr1; i1 = i1 + 1u) {
                    for (var i0: u32 = 0u; i0 < nr0; i0 = i0 + 1u) {
                        let idx = params.offset_src0
                                + (i3 * params.ne3 + k3) * params.nb03
                                + (i2 * params.ne2 + k2) * params.nb02
                                + (i1 * params.ne1 + k1) * params.nb01
                                + (i0 * params.ne0 + k0) * params.nb00;
                        sum = sum + src0[idx];
                    }
                }
            }
        }
        let dst_idx = params.offset_dst
                    + k3 * params.nb3 + k2 * params.nb2
                    + k1 * params.nb1 + k0 * params.nb0;
        dst[dst_idx] = sum;
        k0 = k0 + WG_SIZE;
    }
}
