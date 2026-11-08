struct Params {
    width: u32,
    height: u32,
    quad_count: u32,
    pad: u32,
}

struct RefStats {
    sb: vec4<f32>,
    bb: vec4<f32>,
    vb: vec4<f32>,
    q_sb: array<vec4<f32>, 4>,
    q_bb: array<vec4<f32>, 4>,
    q_vb: array<vec4<f32>, 4>,
}

struct Quad {
    p0: vec2<f32>,
    p1: vec2<f32>,
    p2: vec2<f32>,
    p3: vec2<f32>,
}

struct Result {
    valid: f32,
    rotation: f32,
    score: f32,
    margin: f32,
    votes: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<uniform> stats: RefStats;
@group(0) @binding(2) var<storage, read> quads: array<Quad>;
@group(0) @binding(3) var<storage, read_write> results: array<Result>;
@group(0) @binding(4) var t_image: texture_2d<f32>;
@group(0) @binding(5) var s_image: sampler;
@group(0) @binding(6) var<storage, read> refs: array<vec4<f32>, 256>;

@compute @workgroup_size(64, 1, 1)
fn classify_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.quad_count) {
        return;
    }

    let quad = quads[idx];
    let x0 = quad.p0.x;
    let y0 = quad.p0.y;
    let x1 = quad.p1.x;
    let y1 = quad.p1.y;
    let x2 = quad.p2.x;
    let y2 = quad.p2.y;
    let x3 = quad.p3.x;
    let y3 = quad.p3.y;

    let dx1 = x1 - x2;
    let dx2 = x3 - x2;
    let dx3 = x0 - x1 + x2 - x3;
    let dy1 = y1 - y2;
    let dy2 = y3 - y2;
    let dy3 = y0 - y1 + y2 - y3;

    let det = dx1 * dy2 - dx2 * dy1;
    if (abs(det) < 1e-7) {
        results[idx] = Result(0.0, 0.0, 0.0, 0.0, 0.0);
        return;
    }

    let g = ((dx3 * dy2 - dx2 * dy3) / det) / 16.0;
    let h = ((dx1 * dy3 - dx3 * dy1) / det) / 16.0;
    let a = (x1 - x0) / 16.0 + g * x1;
    let b = (x3 - x0) / 16.0 + h * x3;
    let c = x0;
    let d = (y1 - y0) / 16.0 + g * y1;
    let e = (y3 - y0) / 16.0 + h * y3;
    let f = y0;

    var q_sa = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    var q_aa = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    var q_ab = array<vec4<f32>, 4>(
        vec4<f32>(0.0),
        vec4<f32>(0.0),
        vec4<f32>(0.0),
        vec4<f32>(0.0)
    );

    let img_w = f32(params.width);
    let img_h = f32(params.height);

    for (var j = 1u; j < 15u; j = j + 1u) {
        let v = f32(j) + 0.5;
        for (var i = 1u; i < 15u; i = i + 1u) {
            let u = f32(i) + 0.5;
            let denom = g * u + h * v + 1.0;
            if (abs(denom) < 1e-7) {
                results[idx] = Result(0.0, 0.0, 0.0, 0.0, 0.0);
                return;
            }

            let px = (a * u + b * v + c) / denom;
            let py = (d * u + e * v + f) / denom;

            if (px < 0.0 || py < 0.0 || px >= img_w - 1.0 || py >= img_h - 1.0) {
                results[idx] = Result(0.0, 0.0, 0.0, 0.0, 0.0);
                return;
            }

            let uv = vec2<f32>((px + 0.5) / img_w, (py + 0.5) / img_h);
            let sample_val = textureSampleLevel(t_image, s_image, uv, 0.0).r * 255.0;

            let q = select(0u, 1u, i >= 8u) + select(0u, 2u, j >= 8u);
            let ref_val = refs[j * 16u + i];

            q_sa[q] = q_sa[q] + sample_val;
            q_aa[q] = q_aa[q] + sample_val * sample_val;
            q_ab[q] = q_ab[q] + sample_val * ref_val;
        }
    }

    let sa = q_sa[0] + q_sa[1] + q_sa[2] + q_sa[3];
    let aa = q_aa[0] + q_aa[1] + q_aa[2] + q_aa[3];
    let ab = q_ab[0] + q_ab[1] + q_ab[2] + q_ab[3];

    let va = aa - (sa * sa) / 196.0;

    var sc = vec4<f32>(0.0);
    if (va > 1e-7) {
        for (var k = 0u; k < 4u; k = k + 1u) {
            let vb_k = stats.vb[k];
            if (vb_k > 1e-7) {
                let cov_k = ab[k] - (sa * stats.sb[k]) / 196.0;
                sc[k] = cov_k / sqrt(va * vb_k);
            }
        }
    }

    var best = 0u;
    for (var k = 1u; k < 4u; k = k + 1u) {
        if (sc[k] > sc[best]) {
            best = k;
        }
    }

    var second = select(0u, 1u, best == 0u);
    for (var k = 0u; k < 4u; k = k + 1u) {
        if (k != best && sc[k] > sc[second]) {
            second = k;
        }
    }

    let score = sc[best];
    let margin = sc[best] - sc[second];

    var votes = 0u;
    for (var q = 0u; q < 4u; q = q + 1u) {
        let q_sa_val = q_sa[q];
        let q_aa_val = q_aa[q];
        let q_ab_val = q_ab[q];
        let q_va = q_aa_val - (q_sa_val * q_sa_val) / 49.0;

        let q_sb_val = stats.q_sb[q];
        let q_vb_val = stats.q_vb[q];

        var q_sc = vec4<f32>(0.0);
        if (q_va > 1e-7) {
            for (var k = 0u; k < 4u; k = k + 1u) {
                let vb_k = q_vb_val[k];
                if (vb_k > 1e-7) {
                    let cov_k = q_ab_val[k] - (q_sa_val * q_sb_val[k]) / 49.0;
                    q_sc[k] = cov_k / sqrt(q_va * vb_k);
                }
            }
        }

        var q_best = 0u;
        for (var k = 1u; k < 4u; k = k + 1u) {
            if (q_sc[k] > q_sc[q_best]) {
                q_best = k;
            }
        }

        if (q_best == best) {
            votes = votes + 1u;
        }
    }

    results[idx] = Result(1.0, f32(best), score, margin, f32(votes));
}
