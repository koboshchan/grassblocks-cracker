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

// Corrected centered two-pass covariance (Chan/Golub/LeVeque), with
// Kahan compensated reductions. No sampling, geometry, threshold or CPU changes.
// References: https://www.osti.gov/servlets/purl/1028931
// https://math.pku.edu.cn/teachers/litj/notes/numer_anal/AmerStat_37_242_Chan_Variance.pdf
struct Compensated4 { value:vec4<f32>, correction:vec4<f32> }
fn compensated_add(acc:Compensated4, term:vec4<f32>) -> Compensated4 {
    let residual = term - acc.correction;
    let total = acc.value + residual;
    return Compensated4(total, (total - acc.value) - residual);
}
struct CenteredCorrelation { scores:vec4<f32>, variance:f32 }
// region=4 means all196 pixels; regions0..3 are the unchanged49-pixel votes.
fn centered_correlation(sampled_patch:ptr<function,array<f32,256>>, region:u32) -> CenteredCorrelation {
    var sum_a = Compensated4(vec4<f32>(0.0),vec4<f32>(0.0));
    let count = select(49.0,196.0,region==4u);
    for(var j=1u;j<15u;j=j+1u){for(var i=1u;i<15u;i=i+1u){
        let q=select(0u,1u,i>=8u)+select(0u,2u,j>=8u);
        if(region==4u||region==q){
            sum_a=compensated_add(sum_a,vec4<f32>((*sampled_patch)[j*16u+i]));
        }
    }}
    let mean_a=sum_a.value/count;
    // Existing reference sums are exact uint8 integer sums, not raw variances.
    var reference_sum=stats.sb;
    if(region<4u){reference_sum=stats.q_sb[region];}
    let mean_b=reference_sum/count;
    var square_a=Compensated4(vec4<f32>(0.0),vec4<f32>(0.0));
    var square_b=square_a;var product=square_a;
    var deviation_a=square_a;var deviation_b=square_a;
    for(var j=1u;j<15u;j=j+1u){for(var i=1u;i<15u;i=i+1u){
        let q=select(0u,1u,i>=8u)+select(0u,2u,j>=8u);
        if(region==4u||region==q){
            let da=vec4<f32>((*sampled_patch)[j*16u+i])-mean_a;
            let db=refs[j*16u+i]-mean_b;
            square_a=compensated_add(square_a,da*da);
            square_b=compensated_add(square_b,db*db);
            product=compensated_add(product,da*db);
            deviation_a=compensated_add(deviation_a,da);
            deviation_b=compensated_add(deviation_b,db);
        }
    }}
    // Correct mean-rounding error using centered residual sums, not large raw moments.
    let va=square_a.value-deviation_a.value*deviation_a.value/count;
    let vb=square_b.value-deviation_b.value*deviation_b.value/count;
    let covariance=product.value-deviation_a.value*deviation_b.value/count;
    var scores=vec4<f32>(0.0);
    if(va.x>1e-7){for(var k=0u;k<4u;k=k+1u){
        if(vb[k]>1e-7){scores[k]=covariance[k]/sqrt(va.x*vb[k]);}
    }}
    return CenteredCorrelation(scores,va.x);
}

fn eval_quad_score(quad: Quad) -> f32 {
    let t = get_transform(quad);
    if (!t.valid) {
        return -1.0;
    }

    let img_w = f32(params.width);
    let img_h = f32(params.height);

    var sampled_patch:array<f32,256>;

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

            sampled_patch[j*16u+i] = sample_val;
        }
    }

    let centered=centered_correlation(&sampled_patch,4u);
    if(centered.variance<=1e-7){return -1.0;}
    var best_sc=-1.0;
    for(var k=0u;k<4u;k=k+1u){if(centered.scores[k]>best_sc){best_sc=centered.scores[k];}}

    return best_sc;
}

fn eval_full_result(quad: Quad) -> Result {
    let t = get_transform(quad);
    if (!t.valid) {
        return Result(0.0, 0.0, 0.0, 0.0, 0.0);
    }

    let img_w = f32(params.width);
    let img_h = f32(params.height);

    var sampled_patch:array<f32,256>;

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

            sampled_patch[j*16u+i] = sample_val;
        }
    }

    let sc=centered_correlation(&sampled_patch,4u).scores;

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
        let q_sc=centered_correlation(&sampled_patch,q).scores;

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
