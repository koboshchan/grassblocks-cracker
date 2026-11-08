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
@group(0) @binding(2) var<storage, read_write> quads: array<Quad>;
@group(0) @binding(3) var<storage, read_write> results: array<Result>;
@group(0) @binding(4) var t_image: texture_2d<f32>;
@group(0) @binding(5) var s_image: sampler;
@group(0) @binding(6) var<storage, read> refs: array<vec4<f32>, 256>;

struct Transform {
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    e: f32,
    f: f32,
    g: f32,
    h: f32,
    valid: bool,
}

fn get_transform(quad: Quad) -> Transform {
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
        return Transform(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, false);
    }

    let g = ((dx3 * dy2 - dx2 * dy3) / det) / 16.0;
    let h = ((dx1 * dy3 - dx3 * dy1) / det) / 16.0;
    let a = (x1 - x0) / 16.0 + g * x1;
    let b = (x3 - x0) / 16.0 + h * x3;
    let c = x0;
    let d = (y1 - y0) / 16.0 + g * y1;
    let e = (y3 - y0) / 16.0 + h * y3;
    let f = y0;

    return Transform(a, b, c, d, e, f, g, h, true);
}

fn eval_quad_score(quad: Quad) -> f32 {
    let t = get_transform(quad);
    if (!t.valid) {
        return -1.0;
    }

    let img_w = f32(params.width);
    let img_h = f32(params.height);

    var sa = 0.0;
    var aa = 0.0;
    var ab = vec4<f32>(0.0);

    for (var j = 1u; j < 15u; j = j + 1u) {
        let v = f32(j) + 0.5;
        for (var i = 1u; i < 15u; i = i + 1u) {
            let u = f32(i) + 0.5;
            let denom = t.g * u + t.h * v + 1.0;
            if (abs(denom) < 1e-7) {
                return -1.0;
            }

            let px = (t.a * u + t.b * v + t.c) / denom;
            let py = (t.d * u + t.e * v + t.f) / denom;

            if (px < 0.0 || py < 0.0 || px >= img_w - 1.0 || py >= img_h - 1.0) {
                return -1.0;
            }

            let uv = vec2<f32>((px + 0.5) / img_w, (py + 0.5) / img_h);
            let sample_val = textureSampleLevel(t_image, s_image, uv, 0.0).r * 255.0;
            let ref_val = refs[j * 16u + i];

            sa = sa + sample_val;
            aa = aa + sample_val * sample_val;
            ab = ab + sample_val * ref_val;
        }
    }

    let va = aa - (sa * sa) / 196.0;
    if (va <= 1e-7) {
        return -1.0;
    }

    var best_sc = -1.0;
    for (var k = 0u; k < 4u; k = k + 1u) {
        let vb_k = stats.vb[k];
        if (vb_k > 1e-7) {
            let cov_k = ab[k] - (sa * stats.sb[k]) / 196.0;
            let sc_k = cov_k / sqrt(va * vb_k);
            if (sc_k > best_sc) {
                best_sc = sc_k;
            }
        }
    }

    return best_sc;
}

fn eval_full_result(quad: Quad) -> Result {
    let t = get_transform(quad);
    if (!t.valid) {
        return Result(0.0, 0.0, 0.0, 0.0, 0.0);
    }

    let img_w = f32(params.width);
    let img_h = f32(params.height);

    var q_sa = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    var q_aa = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    var q_ab = array<vec4<f32>, 4>(
        vec4<f32>(0.0),
        vec4<f32>(0.0),
        vec4<f32>(0.0),
        vec4<f32>(0.0)
    );

    for (var j = 1u; j < 15u; j = j + 1u) {
        let v = f32(j) + 0.5;
        for (var i = 1u; i < 15u; i = i + 1u) {
            let u = f32(i) + 0.5;
            let denom = t.g * u + t.h * v + 1.0;
            if (abs(denom) < 1e-7) {
                return Result(0.0, 0.0, 0.0, 0.0, 0.0);
            }

            let px = (t.a * u + t.b * v + t.c) / denom;
            let py = (t.d * u + t.e * v + t.f) / denom;

            if (px < 0.0 || py < 0.0 || px >= img_w - 1.0 || py >= img_h - 1.0) {
                return Result(0.0, 0.0, 0.0, 0.0, 0.0);
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

    return Result(1.0, f32(best), score, margin, f32(votes));
}

fn get_corner(q: Quad, c: u32) -> vec2<f32> {
    if (c == 0u) { return q.p0; }
    if (c == 1u) { return q.p1; }
    if (c == 2u) { return q.p2; }
    return q.p3;
}

fn with_corner(q: Quad, c: u32, p: vec2<f32>) -> Quad {
    var res = q;
    if (c == 0u) { res.p0 = p; }
    else if (c == 1u) { res.p1 = p; }
    else if (c == 2u) { res.p2 = p; }
    else { res.p3 = p; }
    return res;
}

@compute @workgroup_size(64, 1, 1)
fn classify_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.quad_count) {
        return;
    }

    results[idx] = eval_full_result(quads[idx]);
}

@compute @workgroup_size(64, 1, 1)
fn refine_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.quad_count) {
        return;
    }

    var best_quad = quads[idx];
    var best_score = eval_quad_score(best_quad);

    if (best_score > -0.9) {
        for (var c = 0u; c < 4u; c = c + 1u) {
            let p = get_corner(best_quad, c);

            // Sample small subpixel offsets (-0.5, 0, +0.5) around corner coordinates
            let q_xm = with_corner(best_quad, c, vec2<f32>(p.x - 0.5, p.y));
            let s_xm = eval_quad_score(q_xm);

            let q_xp = with_corner(best_quad, c, vec2<f32>(p.x + 0.5, p.y));
            let s_xp = eval_quad_score(q_xp);

            let q_ym = with_corner(best_quad, c, vec2<f32>(p.x, p.y - 0.5));
            let s_ym = eval_quad_score(q_ym);

            let q_yp = with_corner(best_quad, c, vec2<f32>(p.x, p.y + 0.5));
            let s_yp = eval_quad_score(q_yp);

            // Compute local gradient along x and y
            let grad_x = s_xp - s_xm;
            let grad_y = s_yp - s_ym;

            // Second differences (curvature)
            let d2x = s_xp - 2.0 * best_score + s_xm;
            let d2y = s_yp - 2.0 * best_score + s_ym;

            var step_x = 0.0;
            if (d2x < -1e-6) {
                step_x = clamp(-0.5 * grad_x / (2.0 * d2x), -0.5, 0.5);
            } else if (abs(grad_x) > 1e-6) {
                step_x = select(-0.5, 0.5, grad_x > 0.0);
            }

            var step_y = 0.0;
            if (d2y < -1e-6) {
                step_y = clamp(-0.5 * grad_y / (2.0 * d2y), -0.5, 0.5);
            } else if (abs(grad_y) > 1e-6) {
                step_y = select(-0.5, 0.5, grad_y > 0.0);
            }

            // Adjust corner position along local gradient to maximize 4-way correlation score
            if (abs(step_x) > 1e-6 || abs(step_y) > 1e-6) {
                let q_grad = with_corner(best_quad, c, vec2<f32>(p.x + step_x, p.y + step_y));
                let s_grad = eval_quad_score(q_grad);
                if (s_grad > best_score) {
                    best_score = s_grad;
                    best_quad = q_grad;
                }
            }

            // Also test discrete sampled offsets
            if (s_xp > best_score) {
                best_score = s_xp;
                best_quad = q_xp;
            }
            if (s_xm > best_score) {
                best_score = s_xm;
                best_quad = q_xm;
            }
            if (s_yp > best_score) {
                best_score = s_yp;
                best_quad = q_yp;
            }
            if (s_ym > best_score) {
                best_score = s_ym;
                best_quad = q_ym;
            }
        }
    }

    quads[idx] = best_quad;
    results[idx] = eval_full_result(best_quad);
}
