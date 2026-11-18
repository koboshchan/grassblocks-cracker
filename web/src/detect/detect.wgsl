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

    let inv_16 = 0.0625;
    let inv_16_det = inv_16 / det;

    let g = (dx3 * dy2 - dx2 * dy3) * inv_16_det;
    let h = (dx1 * dy3 - dx3 * dy1) * inv_16_det;
    let a = (x1 - x0) * inv_16 + g * x1;
    let b = (x3 - x0) * inv_16 + h * x3;
    let c = x0;
    let d = (y1 - y0) * inv_16 + g * y1;
    let e = (y3 - y0) * inv_16 + h * y3;
    let f = y0;

    return Transform(a, b, c, d, e, f, g, h, true);
}

fn check_bounds(t: Transform, wm1: f32, hm1: f32) -> u32 {
    let cx = array<f32, 4>(1.5, 14.5, 1.5, 14.5);
    let cy = array<f32, 4>(1.5, 1.5, 14.5, 14.5);
    var fast = true;
    for (var k = 0u; k < 4u; k = k + 1u) {
        let d = t.g * cx[k] + t.h * cy[k] + 1.0;
        if (abs(d) < 1e-7) {
            return 0u;
        }
        let inv_d = 1.0 / d;
        let px = (t.a * cx[k] + t.b * cy[k] + t.c) * inv_d;
        let py = (t.d * cx[k] + t.e * cy[k] + t.f) * inv_d;
        if (px < -1e-4 || py < -1e-4 || px >= wm1 + 1e-4 || py >= hm1 + 1e-4) {
            return 0u;
        }
        if (d <= 1e-4 || px < 0.01 || py < 0.01 || px >= wm1 - 0.01 || py >= hm1 - 0.01) {
            fast = false;
        }
    }
    return select(1u, 2u, fast);
}

// Corrected centered two-pass Chan/Golub/LeVeque covariance with 4-corner early-out
// and algebraic 4-quadrant parallel variance/covariance merging (halves `refs` storage reads
// and eliminates the duplicate 196-pixel full-patch two-pass loops).
fn eval_quad_score(quad: Quad) -> f32 {
    let t = get_transform(quad);
    if (!t.valid) {
        return -1.0;
    }

    let img_w = f32(params.width);
    let img_h = f32(params.height);
    let wm1 = img_w - 1.0;
    let hm1 = img_h - 1.0;
    let bmode = check_bounds(t, wm1, hm1);
    if (bmode == 0u) {
        return -1.0;
    }
    let inv_w = 1.0 / img_w;
    let inv_h = 1.0 / img_h;

    var sampled_patch: array<f32, 196>;
    var sum_a = 0.0;

    if (bmode == 2u) {
        for (var j = 1u; j < 15u; j = j + 1u) {
            let v = f32(j) + 0.5;
            var nx = t.a * 1.5 + t.b * v + t.c;
            var ny = t.d * 1.5 + t.e * v + t.f;
            var d = t.g * 1.5 + t.h * v + 1.0;
            let pb = (j - 1u) * 14u;
            for (var i = 0u; i < 14u; i = i + 1u) {
                let inv_d = 1.0 / d;
                let val = textureSampleLevel(t_image, s_image, vec2<f32>((nx * inv_d + 0.5) * inv_w, (ny * inv_d + 0.5) * inv_h), 0.0).r * 255.0;
                sampled_patch[pb + i] = val;
                sum_a += val;
                nx += t.a;
                ny += t.d;
                d += t.g;
            }
        }
    } else {
        for (var j = 1u; j < 15u; j = j + 1u) {
            let v = f32(j) + 0.5;
            var nx = t.a * 1.5 + t.b * v + t.c;
            var ny = t.d * 1.5 + t.e * v + t.f;
            var d = t.g * 1.5 + t.h * v + 1.0;
            let pb = (j - 1u) * 14u;
            for (var i = 0u; i < 14u; i = i + 1u) {
                if (abs(d) < 1e-7) { return -1.0; }
                let px = nx / d;
                let py = ny / d;
                if (px < 0.0 || py < 0.0 || px >= wm1 || py >= hm1) { return -1.0; }
                let val = textureSampleLevel(t_image, s_image, vec2<f32>((px + 0.5) * inv_w, (py + 0.5) * inv_h), 0.0).r * 255.0;
                sampled_patch[pb + i] = val;
                sum_a += val;
                nx += t.a;
                ny += t.d;
                d += t.g;
            }
        }
    }

    let inv_196 = 1.0 / 196.0;
    let mean_a = sum_a * inv_196;
    let mean_b = stats.sb * inv_196;
    var sq_a = 0.0;
    var dev_a = 0.0;
    var cov = vec4<f32>(0.0);
    for (var j = 1u; j < 15u; j = j + 1u) {
        let pb = (j - 1u) * 14u;
        let rb = j * 16u;
        for (var i = 1u; i < 15u; i = i + 1u) {
            let da = sampled_patch[pb + (i - 1u)] - mean_a;
            let db = refs[rb + i] - mean_b;
            sq_a += da * da;
            dev_a += da;
            cov += da * db;
        }
    }
    let va = sq_a - dev_a * dev_a * inv_196;
    if (va <= 1e-7) {
        return -1.0;
    }
    let sc = select(vec4<f32>(0.0), cov * inverseSqrt(va * stats.vb), stats.vb > vec4<f32>(1e-7));
    return max(max(sc.x, sc.y), max(sc.z, sc.w));
}

fn eval_full_result(quad: Quad) -> Result {
    let t = get_transform(quad);
    if (!t.valid) {
        return Result(0.0, 0.0, 0.0, 0.0, 0.0);
    }

    let img_w = f32(params.width);
    let img_h = f32(params.height);
    let wm1 = img_w - 1.0;
    let hm1 = img_h - 1.0;
    let bmode = check_bounds(t, wm1, hm1);
    if (bmode == 0u) {
        return Result(0.0, 0.0, 0.0, 0.0, 0.0);
    }
    let inv_w = 1.0 / img_w;
    let inv_h = 1.0 / img_h;

    var sampled_patch: array<f32, 196>;
    var q_sums = vec4<f32>(0.0);

    if (bmode == 2u) {
        for (var j = 1u; j < 15u; j = j + 1u) {
            let v = f32(j) + 0.5;
            var nx = t.a * 1.5 + t.b * v + t.c;
            var ny = t.d * 1.5 + t.e * v + t.f;
            var d = t.g * 1.5 + t.h * v + 1.0;
            let pb = (j - 1u) * 14u;
            var sL = 0.0;
            var sR = 0.0;
            for (var i = 0u; i < 7u; i = i + 1u) {
                let inv_d = 1.0 / d;
                let val = textureSampleLevel(t_image, s_image, vec2<f32>((nx * inv_d + 0.5) * inv_w, (ny * inv_d + 0.5) * inv_h), 0.0).r * 255.0;
                sampled_patch[pb + i] = val;
                sL += val;
                nx += t.a;
                ny += t.d;
                d += t.g;
            }
            for (var i = 7u; i < 14u; i = i + 1u) {
                let inv_d = 1.0 / d;
                let val = textureSampleLevel(t_image, s_image, vec2<f32>((nx * inv_d + 0.5) * inv_w, (ny * inv_d + 0.5) * inv_h), 0.0).r * 255.0;
                sampled_patch[pb + i] = val;
                sR += val;
                nx += t.a;
                ny += t.d;
                d += t.g;
            }
            if (j < 8u) {
                q_sums.x += sL;
                q_sums.y += sR;
            } else {
                q_sums.z += sL;
                q_sums.w += sR;
            }
        }
    } else {
        for (var j = 1u; j < 15u; j = j + 1u) {
            let v = f32(j) + 0.5;
            var nx = t.a * 1.5 + t.b * v + t.c;
            var ny = t.d * 1.5 + t.e * v + t.f;
            var d = t.g * 1.5 + t.h * v + 1.0;
            let pb = (j - 1u) * 14u;
            for (var i = 0u; i < 14u; i = i + 1u) {
                if (abs(d) < 1e-7) {
                    return Result(0.0, 0.0, 0.0, 0.0, 0.0);
                }
                let px = nx / d;
                let py = ny / d;
                if (px < 0.0 || py < 0.0 || px >= wm1 || py >= hm1) {
                    return Result(0.0, 0.0, 0.0, 0.0, 0.0);
                }
                let val = textureSampleLevel(t_image, s_image, vec2<f32>((px + 0.5) * inv_w, (py + 0.5) * inv_h), 0.0).r * 255.0;
                sampled_patch[pb + i] = val;
                let q = select(0u, 1u, i >= 7u) + select(0u, 2u, j >= 8u);
                q_sums[q] += val;
                nx += t.a;
                ny += t.d;
                d += t.g;
            }
        }
    }

    let inv_49 = 1.0 / 49.0;
    var q_means = q_sums * inv_49;
    var va = 0.0;
    var cov = vec4<f32>(0.0);
    var q_vas: array<f32, 4>;
    var q_covs: array<vec4<f32>, 4>;

    for (var q = 0u; q < 4u; q = q + 1u) {
        let js = select(1u, 8u, q >= 2u);
        let is0 = select(1u, 8u, (q & 1u) == 1u);
        let qma = q_means[q];
        let qmb = stats.q_sb[q] * inv_49;
        var qsq = 0.0;
        var qdev = 0.0;
        var qp = vec4<f32>(0.0);
        for (var j = js; j < js + 7u; j = j + 1u) {
            let pb = (j - 1u) * 14u;
            let rb = j * 16u;
            for (var i = is0; i < is0 + 7u; i = i + 1u) {
                let da = sampled_patch[pb + (i - 1u)] - qma;
                let db = refs[rb + i] - qmb;
                qsq += da * da;
                qdev += da;
                qp += da * db;
            }
        }
        let qva = qsq - qdev * qdev * inv_49;
        q_means[q] = qma + qdev * inv_49;
        q_vas[q] = qva;
        q_covs[q] = qp;
        va += qva;
        cov += qp;
    }

    let mean_a = 0.25 * ((q_means.x + q_means.y) + (q_means.z + q_means.w));
    let mean_b = stats.sb * (1.0 / 196.0);
    for (var q = 0u; q < 4u; q = q + 1u) {
        let da_q = q_means[q] - mean_a;
        let db_q = stats.q_sb[q] * inv_49 - mean_b;
        va += 49.0 * da_q * da_q;
        cov += (49.0 * da_q) * db_q;
    }

    var sc = vec4<f32>(0.0);
    if (va > 1e-7) {
        sc = select(vec4<f32>(0.0), cov * inverseSqrt(va * stats.vb), stats.vb > vec4<f32>(1e-7));
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

    var votes = 0u;
    for (var q = 0u; q < 4u; q = q + 1u) {
        let qva = q_vas[q];
        var qsc = vec4<f32>(0.0);
        if (qva > 1e-7) {
            qsc = select(vec4<f32>(0.0), q_covs[q] * inverseSqrt(qva * stats.q_vb[q]), stats.q_vb[q] > vec4<f32>(1e-7));
        }
        var qb = 0u;
        for (var k = 1u; k < 4u; k = k + 1u) {
            if (qsc[k] > qsc[qb]) {
                qb = k;
            }
        }
        if (qb == best) {
            votes = votes + 1u;
        }
    }

    return Result(1.0, f32(best), sc[best], sc[best] - sc[second], f32(votes));
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

            let q_xm = with_corner(best_quad, c, vec2<f32>(p.x - 0.5, p.y));
            let s_xm = eval_quad_score(q_xm);

            let q_xp = with_corner(best_quad, c, vec2<f32>(p.x + 0.5, p.y));
            let s_xp = eval_quad_score(q_xp);

            let q_ym = with_corner(best_quad, c, vec2<f32>(p.x, p.y - 0.5));
            let s_ym = eval_quad_score(q_ym);

            let q_yp = with_corner(best_quad, c, vec2<f32>(p.x, p.y + 0.5));
            let s_yp = eval_quad_score(q_yp);

            let grad_x = s_xp - s_xm;
            let grad_y = s_yp - s_ym;

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

            if (abs(step_x) > 1e-6 || abs(step_y) > 1e-6) {
                let q_grad = with_corner(best_quad, c, vec2<f32>(p.x + step_x, p.y + step_y));
                let s_grad = eval_quad_score(q_grad);
                if (s_grad > best_score) {
                    best_score = s_grad;
                    best_quad = q_grad;
                }
            }

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
