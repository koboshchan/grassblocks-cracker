struct NccParams {
    width: u32,
    height: u32,
    step_size: u32,
    num_scales: u32,
    threshold: f32,
    max_candidates: u32,
    pad0: u32,
    pad1: u32,
}

struct SeedCandidate {
    x: f32,
    y: f32,
    size: f32,
    rotation: f32,
    score: f32,
    pad: vec3<f32>,
}

struct Candidates {
    candidate_count: atomic<u32>,
    candidates: array<SeedCandidate>,
}

@group(0) @binding(0) var<uniform> params: NccParams;
@group(0) @binding(1) var t_image: texture_2d<f32>;
@group(0) @binding(2) var s_image: sampler;
@group(0) @binding(3) var<storage, read> refs: array<vec4<f32>, 256>;
@group(0) @binding(4) var<storage, read_write> candidate_output: Candidates;

@compute @workgroup_size(16, 16, 1)
fn ncc_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x * params.step_size;
    let y = id.y * params.step_size;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let fx = f32(x);
    let fy = f32(y);
    let img_w = f32(params.width);
    let img_h = f32(params.height);

    var scales = array<f32, 5>(16.0, 24.0, 32.0, 48.0, 64.0);
    let num_scales = select(min(params.num_scales, 5u), 5u, params.num_scales == 0u);

    for (var s = 0u; s < num_scales; s = s + 1u) {
        let size = scales[s];

        if (fx + size > img_w || fy + size > img_h) {
            continue;
        }

        var sa = 0.0;
        var aa = 0.0;
        var sb = vec4<f32>(0.0);
        var bb = vec4<f32>(0.0);
        var ab = vec4<f32>(0.0);

        for (var j = 0u; j < 16u; j = j + 1u) {
            let v = (fy + (f32(j) + 0.5) / 16.0 * size + 0.5) / img_h;
            for (var i = 0u; i < 16u; i = i + 1u) {
                let u = (fx + (f32(i) + 0.5) / 16.0 * size + 0.5) / img_w;
                let sample_val = textureSampleLevel(t_image, s_image, vec2<f32>(u, v), 0.0).r;
                let ref_val = refs[j * 16u + i];

                sa = sa + sample_val;
                aa = aa + sample_val * sample_val;
                sb = sb + ref_val;
                bb = bb + ref_val * ref_val;
                ab = ab + sample_val * ref_val;
            }
        }

        let n = 256.0;
        let va = aa - (sa * sa) / n;
        let vb = bb - (sb * sb) / n;

        var sc = vec4<f32>(-1.0, -1.0, -1.0, -1.0);
        if (va > 1e-7) {
            let cov = ab - (sa * sb) / n;
            for (var k = 0u; k < 4u; k = k + 1u) {
                if (vb[k] > 1e-7) {
                    sc[k] = cov[k] / sqrt(va * vb[k]);
                }
            }
        }

        var best_rot = 0u;
        var best_score = sc[0];
        for (var k = 1u; k < 4u; k = k + 1u) {
            if (sc[k] > best_score) {
                best_score = sc[k];
                best_rot = k;
            }
        }

        if (best_score > params.threshold) {
            let slot = atomicAdd(&candidate_output.candidate_count, 1u);
            if (slot < params.max_candidates) {
                candidate_output.candidates[slot] = SeedCandidate(
                    fx,
                    fy,
                    size,
                    f32(best_rot),
                    best_score,
                    vec3<f32>(0.0, 0.0, 0.0)
                );
            }
        }
    }
}
