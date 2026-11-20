// u64 represented as vec2<u32>(low,high). All arithmetic wraps mod 2^64.
fn hi32_116129781(a: u32) -> u32 {
  let al = a & 65535u;
  let ah = a >> 16u;
  let p0 = al * 65525u;
  let mid = al * 1771u + (p0 >> 16u);
  let mid2 = (mid & 65535u) + ah * 65525u;
  return ah * 1771u + (mid >> 16u) + (mid2 >> 16u);
}

fn mulc_z(z: i32) -> vec2<u32> {
  let zu = u32(z);
  let zh = u32(z >> 31u);
  return vec2<u32>(zu * 116129781u, hi32_116129781(zu) + zh * 116129781u);
}

fn rotation_xz_zp(x: i32, zp: vec2<u32>) -> vec2<u32> {
  let xp = x * 3129871;
  return vec2<u32>(u32(xp) ^ zp.x, u32(xp >> 31u) ^ zp.y);
}

fn rotation_xz(x: i32, z: i32) -> vec2<u32> {
  let xp = x * 3129871;
  let zp = mulc_z(z);
  return vec2<u32>(u32(xp) ^ zp.x, u32(xp >> 31u) ^ zp.y);
}

// Finishes (p_lo >> 16, p_hi) -> step_hi with ((-expected - 1) & 3) << 14 folded in.
// Uses 16-bit constants 8468 = 65536 - 0xdeec and 6547 = 65536 - 0xe66d so all multiplies
// fit in fast 16-bit GPU MUL/IMAD paths with zero overflow (Q in [89.7M, 1.07B] < 2^31).
fn finish_p_step_hi_mid(rx_hi16: u32, ry: u32, neg_shift_m1: u32) -> u32 {
  let al = rx_hi16 ^ 0xe66du;
  let ah = (ry & 65535u) ^ 0xdeecu;
  let sy = (ry >> 16u) ^ 5u;
  let p0 = al * 0xe66du + 11u;
  let q = (0x40000000u - al * 8468u) - ah * 6547u + (p0 >> 16u);
  return (q >> 16u) + ah * 0xdeedu + sy * 0xe66du + al * 6u + neg_shift_m1;
}

fn finish_p_step_hi(rx: u32, ry: u32, neg_shift_m1: u32) -> u32 {
  return finish_p_step_hi_mid(rx >> 16u, ry, neg_shift_m1);
}

// Factored Y hash: p = lx * (42317861 * lx + 11 + 2^32 * k1) + 2^32 * k0 (mod 2^64).
// Folds +11 into p0_c, merges mid_c without overflow (max 3.13e9 < 2^32),
// and reuses mid2_m & 65535 == p_lo >> 16 to skip c_lo, a_lo, and p_lo completely.
fn rotation_y_lh_raw(xz_lo: u32, k1: u32, k0: u32, y_u: u32, neg_shift_m1: u32) -> u32 {
  let lx = xz_lo ^ y_u;
  let lx_l = lx & 65535u;
  let lx_h = lx >> 16u;

  let p0_c = lx_l * 47141u + 11u;
  let al = p0_c & 65535u;
  let mid_c = lx_l * 645u + lx_h * 47141u + (p0_c >> 16u);
  let ah = mid_c & 65535u;
  let a_hi = lx_h * 645u + (mid_c >> 16u) + k1;

  let p0_m = lx_l * al;
  let mid_m = lx_l * ah + (p0_m >> 16u);
  let mid2_m = (mid_m & 65535u) + lx_h * al;
  let p_hi = lx_h * ah + (mid_m >> 16u) + (mid2_m >> 16u) + lx * a_hi + k0;
  return finish_p_step_hi_mid(mid2_m & 65535u, p_hi, neg_shift_m1);
}

// Fast Y hash when 0 <= y_u <= 65535 with zero extra register cost:
// lxh_645_k1 = (xz_lo >> 16) * 645 + k1 is pre-folded into k1 at column setup.
fn rotation_y_u16_k1_raw(xz_lo: u32, lxh_645_k1: u32, k0: u32, y_u: u32, neg_shift_m1: u32) -> u32 {
  let lx = xz_lo ^ y_u;
  let lx_l = lx & 65535u;
  let lx_h = lx >> 16u;

  let p0_c = lx_l * 47141u + 11u;
  let al = p0_c & 65535u;
  let mid_c = lx_l * 645u + lx_h * 47141u + (p0_c >> 16u);
  let ah = mid_c & 65535u;
  let a_hi = (mid_c >> 16u) + lxh_645_k1;

  let p0_m = lx_l * al;
  let mid_m = lx_l * ah + (p0_m >> 16u);
  let mid2_m = (mid_m & 65535u) + lx_h * al;
  let p_hi = lx_h * ah + (mid_m >> 16u) + (mid2_m >> 16u) + lx * a_hi + k0;
  return finish_p_step_hi_mid(mid2_m & 65535u, p_hi, neg_shift_m1);
}

// Full hoisted Y hash for scalar single-north when 0 <= y_u <= 65535.
fn rotation_y_u16_raw(xz_lo: u32, lx_h: u32, lxh_47141: u32, lxh_645_k1: u32, k0: u32, y_u: u32, neg_shift_m1: u32) -> u32 {
  let lx = xz_lo ^ y_u;
  let lx_l = lx & 65535u;

  let p0_c = lx_l * 47141u + 11u;
  let al = p0_c & 65535u;
  let mid_c = lx_l * 645u + lxh_47141 + (p0_c >> 16u);
  let ah = mid_c & 65535u;
  let a_hi = (mid_c >> 16u) + lxh_645_k1;

  let p0_m = lx_l * al;
  let mid_m = lx_l * ah + (p0_m >> 16u);
  let mid2_m = (mid_m & 65535u) + lx_h * al;
  let p_hi = lx_h * ah + (mid_m >> 16u) + (mid2_m >> 16u) + lx * a_hi + k0;
  return finish_p_step_hi_mid(mid2_m & 65535u, p_hi, neg_shift_m1);
}

fn rotation_y_lh(xz_lo: u32, k1: u32, k0: u32, y_u: u32) -> u32 {
  return (rotation_y_lh_raw(xz_lo, k1, k0, y_u, 0xc000u) >> 14u) & 3u;
}

fn rotation_y(xz: vec2<u32>, y: i32) -> u32 {
  let ly = xz.y ^ u32(y >> 31u);
  return rotation_y_lh(xz.x, ly * 84635722u, ly * 11u, u32(y));
}

fn rotation(x: i32, y: i32, z: i32) -> u32 {
  return rotation_y(rotation_xz(x, z), y);
}

// 2nd-order finite-difference state for stepping lx = B, B+1, ..., B+(2^k - 1).
struct FdState { p_lo: u32, p_hi: u32, d_lo: u32, d_hi: u32 }

fn fd_init(B: u32, k1: u32, k0: u32) -> FdState {
  let bl = B & 65535u;
  let bh = B >> 16u;

  let p0_c = bl * 47141u;
  let mid_c = bl * 645u + bh * 47141u + (p0_c >> 16u);
  let c_lo = (p0_c & 65535u) | (mid_c << 16u);
  let c_hi = bh * 645u + (mid_c >> 16u);

  let a_lo = c_lo + 11u;
  let a_hi = c_hi + k1 + select(0u, 1u, a_lo < 11u);

  let al = a_lo & 65535u;
  let ah = a_lo >> 16u;
  let p0_m = bl * al;
  let mid_m = bl * ah + (p0_m >> 16u);
  let mid2_m = (mid_m & 65535u) + bh * al;
  let p_lo = B * a_lo;
  let p_hi = bh * ah + (mid_m >> 16u) + (mid2_m >> 16u) + B * a_hi + k0;

  let c2_lo = c_lo << 1u;
  let c2_hi = (c_hi << 1u) | (c_lo >> 31u);
  let d_lo = c2_lo + 42317872u;
  let d_hi = c2_hi + k1 + select(0u, 1u, d_lo < 42317872u);
  return FdState(p_lo, p_hi, d_lo, d_hi);
}

struct Block { x: vec4<i32>, z: vec4<i32>, r: vec4<i32>, y: i32, side: i32 }
struct Params { xmin: i32, zmin: i32, ymin: i32, ymax: i32, width: u32, offset: u32, count: u32, blocks: u32 }
struct Results { count: atomic<u32>, xyz: array<vec4<i32>> }

@group(0) @binding(0) var<storage, read> blocks: array<Block>;
@group(0) @binding(1) var<uniform> p: Params;
@group(0) @binding(2) var<storage, read_write> results: Results;

override WG_SIZE: u32 = 128u;
override ITEMS: u32 = 1u;
override SINGLE_NORTH: i32 = -1;
override ALL_FOUR: bool = false;
override NONNEG_Y: bool = false;
override ANCHOR_SIDE: bool = false;
override MANY_BLOCKS: bool = false;

fn neg_shift_m1_i32(r: i32) -> u32 {
  return (u32(-r - 1) & 3u) << 14u;
}

fn select4b(v: vec4<u32>, b0: bool, b1: bool) -> u32 {
  return select(select(v.x, v.y, b0), select(v.z, v.w, b0), b1);
}

fn check_from(x: i32, y: i32, z: i32, n: u32, start_i: u32) -> bool {
  for (var i = start_i; i < p.blocks; i++) {
    let b = blocks[i];
    let xz = rotation_xz(x + b.x[n], z + b.z[n]);
    let yb = y + b.y;
    let ly = xz.y ^ u32(yb >> 31u);
    let ms = select(0xc000u, 0x4000u, b.side != 0);
    if ((rotation_y_lh_raw(xz.x, ly * 84635722u, ly * 11u, u32(yb), neg_shift_m1_i32(b.r[n])) & ms) != 0u) {
      return false;
    }
  }
  return true;
}

fn check_rest_scalar(x: i32, y: i32, z: i32, n: u32, xz1: vec2<u32>, b1_y: i32, mask1_s: u32) -> bool {
  if (p.blocks > 1u) {
    let y1 = y + b1_y;
    let ly1 = select(xz1.y ^ u32(y1 >> 31u), xz1.y, NONNEG_Y);
    if ((rotation_y_lh_raw(xz1.x, ly1 * 84635722u, ly1 * 11u, u32(y1), neg_shift_m1_i32(blocks[1].r[n])) & mask1_s) != 0u) {
      return false;
    }
  }
  return check_from(x, y, z, n, 2u);
}

fn emit_result(x: i32, y: i32, z: i32, n: u32) {
  let idx = atomicAdd(&results.count, 1u);
  if (idx < 4096u) { results.xyz[idx] = vec4<i32>(x, y, z, i32(n)); }
}

@compute @workgroup_size(WG_SIZE)
fn search(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (p.blocks == 0u) { return; }
  let base_index = gid.x * ITEMS;
  if (base_index >= p.count) { return; }

  let base_pos = p.offset + base_index;
  var qz = base_pos / p.width;
  var rx = base_pos - qz * p.width;

  let b0_r = blocks[0].r;
  let anchor_r = select(select(select(b0_r.w - 3, b0_r.z - 2, b0_r.z >= 0), b0_r.y - 1, b0_r.y >= 0), b0_r.x, b0_r.x >= 0) & 3;
  var b0_mask = 0u;
  if (!ALL_FOUR && SINGLE_NORTH < 0) {
    if (b0_r.x >= 0) { b0_mask |= 1u; }
    if (b0_r.y >= 0) { b0_mask |= 2u; }
    if (b0_r.z >= 0) { b0_mask |= 4u; }
    if (b0_r.w >= 0) { b0_mask |= 8u; }
  }

  var b1_y = 0; var mask1_s = 0u; var neg_r1_base_m1 = 0u;
  if (p.blocks > 1u) {
    let b1r = blocks[1].r;
    b1_y = blocks[1].y;
    mask1_s = select(0xc000u, 0x4000u, blocks[1].side != 0);
    let r1_base = select(select(select(b1r.w - 3, b1r.z - 2, b1r.z >= 0), b1r.y - 1, b1r.y >= 0), b1r.x, b1r.x >= 0) & 3;
    neg_r1_base_m1 = neg_shift_m1_i32(r1_base);
  }
  var b2_y = 0; var mask2_s = 0u; var neg_r2_base_m1 = 0u;
  if (p.blocks > 2u) {
    let b2r = blocks[2].r;
    b2_y = blocks[2].y;
    mask2_s = select(0xc000u, 0x4000u, blocks[2].side != 0);
    let r2_base = select(select(select(b2r.w - 3, b2r.z - 2, b2r.z >= 0), b2r.y - 1, b2r.y >= 0), b2r.x, b2r.x >= 0) & 3;
    neg_r2_base_m1 = neg_shift_m1_i32(r2_base);
  }

  var zp0 = mulc_z(p.zmin + i32(qz));
  let seg_start = select(0u, 1u, NONNEG_Y);

  for (var item = 0u; item < ITEMS; item++) {
    let index = base_index + item;
    if (index >= p.count) { break; }
    let x = p.xmin + i32(rx);
    let z = p.zmin + i32(qz);
    let xz0 = rotation_xz_zp(x, zp0);

    if (SINGLE_NORTH >= 0) {
      let sn = u32(SINGLE_NORTH);
      let neg_r0_m1 = neg_shift_m1_i32(b0_r[sn]);
      let target0_mask = select(0xc000u, 0x4000u, ANCHOR_SIDE);

      var xz1 = vec2<u32>(0u);
      var xz1_lxh = 0u; var xz1_47141 = 0u; var xz1_645_k1 = 0u; var xz1_k0 = 0u;
      if (p.blocks > 1u) {
        let bx1 = blocks[1].x[sn];
        let bz1 = blocks[1].z[sn];
        if (bx1 == 0 && bz1 == 0) { xz1 = xz0; }
        else { xz1 = rotation_xz(x + bx1, z + bz1); }
        if (NONNEG_Y) {
          xz1_lxh = xz1.x >> 16u;
          xz1_47141 = xz1_lxh * 47141u;
          xz1_645_k1 = xz1_lxh * 645u + xz1.y * 84635722u;
          xz1_k0 = xz1.y * 11u;
        }
      }
      var xz2 = vec2<u32>(0u);
      var xz2_lxh = 0u; var xz2_47141 = 0u; var xz2_645_k1 = 0u; var xz2_k0 = 0u;
      if (MANY_BLOCKS) {
        let bx2 = blocks[2].x[sn];
        let bz2 = blocks[2].z[sn];
        if (bx2 == 0 && bz2 == 0) { xz2 = xz0; }
        else { xz2 = rotation_xz(x + bx2, z + bz2); }
        if (NONNEG_Y) {
          xz2_lxh = xz2.x >> 16u;
          xz2_47141 = xz2_lxh * 47141u;
          xz2_645_k1 = xz2_lxh * 645u + xz2.y * 84635722u;
          xz2_k0 = xz2.y * 11u;
        }
      }

      var neg1_m1 = 0u;
      if (p.blocks > 1u) { neg1_m1 = neg_shift_m1_i32(blocks[1].r[sn]); }
      var neg2_m1 = 0u;
      if (MANY_BLOCKS) { neg2_m1 = neg_shift_m1_i32(blocks[2].r[sn]); }

      for (var seg = seg_start; seg < 2u; seg++) {
        var ya = p.ymin;
        var yb = p.ymax;
        if (seg == 0u) { yb = min(p.ymax, -1); } else { ya = max(p.ymin, 0); }
        if (ya > yb) { continue; }
        let lh = select(xz0.y, ~xz0.y, seg == 0u);
        let k1 = lh * 84635722u;
        let k0 = lh * 11u;

        if (MANY_BLOCKS) {
          var y = ya;
          // On signed/negative ranges (e.g. -63..-1), align down to 64-multiple if close and clip cy >= ya.
          let y64_base = i32(u32(ya) & 0xffffffc0u);
          if (!NONNEG_Y && (u32(ya) & 63u) > 0u && (u32(ya) & 63u) <= 20u && y64_base + 63 <= yb) {
            y = y64_base;
          } else {
            while (y <= yb && (u32(y) & 31u) != 0u) {
              if ((rotation_y_lh_raw(xz0.x, k1, k0, u32(y), neg_r0_m1) & target0_mask) == 0u) {
                if (check_rest_scalar(x, y, z, sn, xz1, b1_y, mask1_s)) {
                  emit_result(x, y, z, sn);
                }
              }
              y++;
            }
          }
          // Unified 64/32-step FD + branchless joint drain (single fd_init, zero alignment waste).
          while (y + 31 <= yb) {
            let do64 = ((u32(y) & 63u) == 0u) && (y + 63 <= yb);
            let perm = select(xz0.x & 31u, xz0.x & 63u, do64);
            let B = (xz0.x ^ u32(y)) & select(0xffffffe0u, 0xffffffc0u, do64);
            let st0 = fd_init(B, k1, k0);
            var p_lo = st0.p_lo; var p_hi = st0.p_hi;
            var d_lo = st0.d_lo; var d_hi = st0.d_hi;
            var hm_a = 0u;
            var hm_b = 0u;

            for (var k = 0u; k < 32u; k++) {
              hm_a |= select(0u, 1u << k, (finish_p_step_hi(p_lo, p_hi, neg_r0_m1) & target0_mask) == 0u);
              let np = p_lo + d_lo;
              p_hi += d_hi + select(0u, 1u, np < p_lo);
              p_lo = np;
              let nd = d_lo + 84635722u;
              d_hi += select(0u, 1u, nd < d_lo);
              d_lo = nd;
            }
            if (do64) {
              for (var k = 0u; k < 32u; k++) {
                hm_b |= select(0u, 1u << k, (finish_p_step_hi(p_lo, p_hi, neg_r0_m1) & target0_mask) == 0u);
                let np = p_lo + d_lo;
                p_hi += d_hi + select(0u, 1u, np < p_lo);
                p_lo = np;
                let nd = d_lo + 84635722u;
                d_hi += select(0u, 1u, nd < d_lo);
                d_lo = nd;
              }
            }

            // Branchless joint drain of (hm_a, hm_b) -> Block 1 -> (hm2_a, hm2_b)
            var hm2_a = 0u;
            var hm2_b = 0u;
            let y1_base = u32(y + b1_y);
            let clip_lo = !NONNEG_Y && (y < ya);
            while ((hm_a | hm_b) != 0u) {
              let use_a = hm_a != 0u;
              let m = select(hm_b, hm_a, use_a);
              let bit = countTrailingZeros(m);
              let cleared = m & (m - 1u);
              hm_a = select(hm_a, cleared, use_a);
              hm_b = select(cleared, hm_b, use_a);
              let dy = (bit | select(32u, 0u, use_a)) ^ perm;
              if (clip_lo && y + i32(dy) < ya) { continue; }
              let y1_u = y1_base + dy;
              var sh1: u32;
              if (NONNEG_Y) { sh1 = rotation_y_u16_raw(xz1.x, xz1_lxh, xz1_47141, xz1_645_k1, xz1_k0, y1_u, neg1_m1); }
              else { let ly1 = xz1.y ^ u32(i32(y1_u) >> 31u); sh1 = rotation_y_lh_raw(xz1.x, ly1 * 84635722u, ly1 * 11u, y1_u, neg1_m1); }
              let ok1 = select(0u, 1u << bit, (sh1 & mask1_s) == 0u);
              hm2_a |= select(0u, ok1, use_a);
              hm2_b |= select(ok1, 0u, use_a);
            }

            // Branchless joint drain of (hm2_a, hm2_b) -> Block 2 -> (hm3_a, hm3_b)
            var hm3_a = 0u;
            var hm3_b = 0u;
            let y2_base = u32(y + b2_y);
            while ((hm2_a | hm2_b) != 0u) {
              let use_a = hm2_a != 0u;
              let m = select(hm2_b, hm2_a, use_a);
              let bit = countTrailingZeros(m);
              let cleared = m & (m - 1u);
              hm2_a = select(hm2_a, cleared, use_a);
              hm2_b = select(cleared, hm2_b, use_a);
              let dy = (bit | select(32u, 0u, use_a)) ^ perm;
              let y2_u = y2_base + dy;
              var sh2: u32;
              if (NONNEG_Y) { sh2 = rotation_y_u16_raw(xz2.x, xz2_lxh, xz2_47141, xz2_645_k1, xz2_k0, y2_u, neg2_m1); }
              else { let ly2 = xz2.y ^ u32(i32(y2_u) >> 31u); sh2 = rotation_y_lh_raw(xz2.x, ly2 * 84635722u, ly2 * 11u, y2_u, neg2_m1); }
              let ok2 = select(0u, 1u << bit, (sh2 & mask2_s) == 0u);
              hm3_a |= select(0u, ok2, use_a);
              hm3_b |= select(ok2, 0u, use_a);
            }

            // Joint drain of (hm3_a, hm3_b) -> Block 3+
            while ((hm3_a | hm3_b) != 0u) {
              let use_a = hm3_a != 0u;
              let m = select(hm3_b, hm3_a, use_a);
              let bit = countTrailingZeros(m);
              let cleared = m & (m - 1u);
              hm3_a = select(hm3_a, cleared, use_a);
              hm3_b = select(cleared, hm3_b, use_a);
              let dy = (bit | select(32u, 0u, use_a)) ^ perm;
              let cy = y + i32(dy);
              if (check_from(x, cy, z, sn, 3u)) {
                emit_result(x, cy, z, sn);
              }
            }
            y += select(32, 64, do64);
          }
          while (y <= yb) {
            if ((rotation_y_lh_raw(xz0.x, k1, k0, u32(y), neg_r0_m1) & target0_mask) == 0u) {
              if (check_rest_scalar(x, y, z, sn, xz1, b1_y, mask1_s)) {
                emit_result(x, y, z, sn);
              }
            }
            y++;
          }
        } else {
          for (var y = ya; y <= yb; y++) {
            if ((rotation_y_lh_raw(xz0.x, k1, k0, u32(y), neg_r0_m1) & target0_mask) == 0u) {
              if (check_rest_scalar(x, y, z, sn, xz1, b1_y, mask1_s)) {
                emit_result(x, y, z, sn);
              }
            }
          }
        }
      }
    } else {
      // Multi-north (ALL_FOUR or subset north mask).
      let neg_rA_m1 = neg_shift_m1_i32(anchor_r);
      var xz1_lo = vec4<u32>(0u);
      var xz1_hi = vec4<u32>(0u);
      var xz1_k1 = vec4<u32>(0u);
      var xz1_k0 = vec4<u32>(0u);
      if (p.blocks > 1u) {
        let b1x = blocks[1].x;
        let b1z = blocks[1].z;
        if (ALL_FOUR || b0_r.x >= 0) {
          let t = select(rotation_xz(x + b1x.x, z + b1z.x), xz0, b1x.x == 0 && b1z.x == 0);
          xz1_lo.x = t.x; xz1_hi.x = t.y;
        }
        if (ALL_FOUR || b0_r.y >= 0) {
          let t = select(rotation_xz(x + b1x.y, z + b1z.y), xz0, b1x.y == 0 && b1z.y == 0);
          xz1_lo.y = t.x; xz1_hi.y = t.y;
        }
        if (ALL_FOUR || b0_r.z >= 0) {
          let t = select(rotation_xz(x + b1x.z, z + b1z.z), xz0, b1x.z == 0 && b1z.z == 0);
          xz1_lo.z = t.x; xz1_hi.z = t.y;
        }
        if (ALL_FOUR || b0_r.w >= 0) {
          let t = select(rotation_xz(x + b1x.w, z + b1z.w), xz0, b1x.w == 0 && b1z.w == 0);
          xz1_lo.w = t.x; xz1_hi.w = t.y;
        }
        if (NONNEG_Y) {
          xz1_k1 = (xz1_lo >> vec4<u32>(16u)) * 645u + xz1_hi * 84635722u;
          xz1_k0 = xz1_hi * 11u;
        }
      }

      var xz2_lo = vec4<u32>(0u);
      var xz2_hi = vec4<u32>(0u);
      var xz2_k1 = vec4<u32>(0u);
      var xz2_k0 = vec4<u32>(0u);
      if (MANY_BLOCKS && !ANCHOR_SIDE) {
        let b2x = blocks[2].x;
        let b2z = blocks[2].z;
        if (ALL_FOUR || b0_r.x >= 0) {
          let t = select(rotation_xz(x + b2x.x, z + b2z.x), xz0, b2x.x == 0 && b2z.x == 0);
          xz2_lo.x = t.x; xz2_hi.x = t.y;
        }
        if (ALL_FOUR || b0_r.y >= 0) {
          let t = select(rotation_xz(x + b2x.y, z + b2z.y), xz0, b2x.y == 0 && b2z.y == 0);
          xz2_lo.y = t.x; xz2_hi.y = t.y;
        }
        if (ALL_FOUR || b0_r.z >= 0) {
          let t = select(rotation_xz(x + b2x.z, z + b2z.z), xz0, b2x.z == 0 && b2z.z == 0);
          xz2_lo.z = t.x; xz2_hi.z = t.y;
        }
        if (ALL_FOUR || b0_r.w >= 0) {
          let t = select(rotation_xz(x + b2x.w, z + b2z.w), xz0, b2x.w == 0 && b2z.w == 0);
          xz2_lo.w = t.x; xz2_hi.w = t.y;
        }
        if (NONNEG_Y) {
          xz2_k1 = (xz2_lo >> vec4<u32>(16u)) * 645u + xz2_hi * 84635722u;
          xz2_k0 = xz2_hi * 11u;
        }
      }

      for (var seg = seg_start; seg < 2u; seg++) {
        var ya = p.ymin;
        var yb = p.ymax;
        if (seg == 0u) { yb = min(p.ymax, -1); } else { ya = max(p.ymin, 0); }
        if (ya > yb) { continue; }
        let lh = select(xz0.y, ~xz0.y, seg == 0u);
        let k1 = lh * 84635722u;
        let k0 = lh * 11u;

        if (MANY_BLOCKS && !ANCHOR_SIDE) {
          var y = ya;
          let y64_base = i32(u32(ya) & 0xffffffc0u);
          if (!NONNEG_Y && (u32(ya) & 63u) > 0u && (u32(ya) & 63u) <= 20u && y64_base + 63 <= yb) {
            y = y64_base;
          } else {
            while (y <= yb && (u32(y) & 31u) != 0u) {
              let sh0 = rotation_y_lh_raw(xz0.x, k1, k0, u32(y), neg_rA_m1);
              let n = (sh0 >> 14u) & 3u;
              if (ALL_FOUR || ((b0_mask >> n) & 1u) != 0u) {
                let nb0 = (sh0 & 0x4000u) != 0u;
                let nb1 = (sh0 & 0x8000u) != 0u;
                let y1 = y + b1_y;
                var sh1: u32;
                if (NONNEG_Y) { sh1 = rotation_y_u16_k1_raw(select4b(xz1_lo, nb0, nb1), select4b(xz1_k1, nb0, nb1), select4b(xz1_k0, nb0, nb1), u32(y1), neg_r1_base_m1); }
                else { let ly1 = select4b(xz1_hi, nb0, nb1) ^ u32(y1 >> 31u); sh1 = rotation_y_lh_raw(select4b(xz1_lo, nb0, nb1), ly1 * 84635722u, ly1 * 11u, u32(y1), neg_r1_base_m1); }
                if (((sh0 ^ sh1) & mask1_s) == 0u) {
                  let y2 = y + b2_y;
                  var sh2: u32;
                  if (NONNEG_Y) { sh2 = rotation_y_u16_k1_raw(select4b(xz2_lo, nb0, nb1), select4b(xz2_k1, nb0, nb1), select4b(xz2_k0, nb0, nb1), u32(y2), neg_r2_base_m1); }
                  else { let ly2 = select4b(xz2_hi, nb0, nb1) ^ u32(y2 >> 31u); sh2 = rotation_y_lh_raw(select4b(xz2_lo, nb0, nb1), ly2 * 84635722u, ly2 * 11u, u32(y2), neg_r2_base_m1); }
                  if (((sh0 ^ sh2) & mask2_s) == 0u && check_from(x, y, z, n, 3u)) {
                    emit_result(x, y, z, n);
                  }
                }
              }
              y++;
            }
          }
          // Unified 64/32-step FD + Block 1 -> branchless joint drain -> Block 2 -> Block 3+.
          while (y + 31 <= yb) {
            let do64 = ((u32(y) & 63u) == 0u) && (y + 63 <= yb);
            let perm = select(xz0.x & 31u, xz0.x & 63u, do64);
            let B = (xz0.x ^ u32(y)) & select(0xffffffe0u, 0xffffffc0u, do64);
            let st0 = fd_init(B, k1, k0);
            var p_lo = st0.p_lo; var p_hi = st0.p_hi;
            var d_lo = st0.d_lo; var d_hi = st0.d_hi;
            let y1_base = u32(y + b1_y);
            var hm1_a = 0u; var hm1_b = 0u;
            var na_lo = 0u; var na_hi = 0u;
            var nb_lo = 0u; var nb_hi = 0u;

            for (var k = 0u; k < 16u; k++) {
              let sh0 = finish_p_step_hi(p_lo, p_hi, neg_rA_m1);
              let np = p_lo + d_lo; p_hi += d_hi + select(0u, 1u, np < p_lo); p_lo = np;
              let nd = d_lo + 84635722u; d_hi += select(0u, 1u, nd < d_lo); d_lo = nd;
              let nb0 = (sh0 & 0x4000u) != 0u;
              let nb1 = (sh0 & 0x8000u) != 0u;
              if (ALL_FOUR || ((b0_mask >> ((sh0 >> 14u) & 3u)) & 1u) != 0u) {
                let y1_u = y1_base + (k ^ perm);
                var sh1: u32;
                if (NONNEG_Y) { sh1 = rotation_y_u16_k1_raw(select4b(xz1_lo, nb0, nb1), select4b(xz1_k1, nb0, nb1), select4b(xz1_k0, nb0, nb1), y1_u, neg_r1_base_m1); }
                else { let ly1 = select4b(xz1_hi, nb0, nb1) ^ u32(i32(y1_u) >> 31u); sh1 = rotation_y_lh_raw(select4b(xz1_lo, nb0, nb1), ly1 * 84635722u, ly1 * 11u, y1_u, neg_r1_base_m1); }
                if (((sh0 ^ sh1) & mask1_s) == 0u) {
                  hm1_a |= (1u << k);
                  na_lo |= (((sh0 >> 14u) & 3u) << (k << 1u));
                }
              }
            }
            for (var k = 16u; k < 32u; k++) {
              let sh0 = finish_p_step_hi(p_lo, p_hi, neg_rA_m1);
              let np = p_lo + d_lo; p_hi += d_hi + select(0u, 1u, np < p_lo); p_lo = np;
              let nd = d_lo + 84635722u; d_hi += select(0u, 1u, nd < d_lo); d_lo = nd;
              let nb0 = (sh0 & 0x4000u) != 0u;
              let nb1 = (sh0 & 0x8000u) != 0u;
              if (ALL_FOUR || ((b0_mask >> ((sh0 >> 14u) & 3u)) & 1u) != 0u) {
                let y1_u = y1_base + (k ^ perm);
                var sh1: u32;
                if (NONNEG_Y) { sh1 = rotation_y_u16_k1_raw(select4b(xz1_lo, nb0, nb1), select4b(xz1_k1, nb0, nb1), select4b(xz1_k0, nb0, nb1), y1_u, neg_r1_base_m1); }
                else { let ly1 = select4b(xz1_hi, nb0, nb1) ^ u32(i32(y1_u) >> 31u); sh1 = rotation_y_lh_raw(select4b(xz1_lo, nb0, nb1), ly1 * 84635722u, ly1 * 11u, y1_u, neg_r1_base_m1); }
                if (((sh0 ^ sh1) & mask1_s) == 0u) {
                  hm1_a |= (1u << k);
                  na_hi |= (((sh0 >> 14u) & 3u) << ((k - 16u) << 1u));
                }
              }
            }
            if (do64) {
              for (var k = 0u; k < 16u; k++) {
                let sh0 = finish_p_step_hi(p_lo, p_hi, neg_rA_m1);
                let np = p_lo + d_lo; p_hi += d_hi + select(0u, 1u, np < p_lo); p_lo = np;
                let nd = d_lo + 84635722u; d_hi += select(0u, 1u, nd < d_lo); d_lo = nd;
                let nb0 = (sh0 & 0x4000u) != 0u;
                let nb1 = (sh0 & 0x8000u) != 0u;
                if (ALL_FOUR || ((b0_mask >> ((sh0 >> 14u) & 3u)) & 1u) != 0u) {
                  let y1_u = y1_base + ((k | 32u) ^ perm);
                  var sh1: u32;
                  if (NONNEG_Y) { sh1 = rotation_y_u16_k1_raw(select4b(xz1_lo, nb0, nb1), select4b(xz1_k1, nb0, nb1), select4b(xz1_k0, nb0, nb1), y1_u, neg_r1_base_m1); }
                  else { let ly1 = select4b(xz1_hi, nb0, nb1) ^ u32(i32(y1_u) >> 31u); sh1 = rotation_y_lh_raw(select4b(xz1_lo, nb0, nb1), ly1 * 84635722u, ly1 * 11u, y1_u, neg_r1_base_m1); }
                  if (((sh0 ^ sh1) & mask1_s) == 0u) {
                    hm1_b |= (1u << k);
                    nb_lo |= (((sh0 >> 14u) & 3u) << (k << 1u));
                  }
                }
              }
              for (var k = 16u; k < 32u; k++) {
                let sh0 = finish_p_step_hi(p_lo, p_hi, neg_rA_m1);
                let np = p_lo + d_lo; p_hi += d_hi + select(0u, 1u, np < p_lo); p_lo = np;
                let nd = d_lo + 84635722u; d_hi += select(0u, 1u, nd < d_lo); d_lo = nd;
                let nb0 = (sh0 & 0x4000u) != 0u;
                let nb1 = (sh0 & 0x8000u) != 0u;
                if (ALL_FOUR || ((b0_mask >> ((sh0 >> 14u) & 3u)) & 1u) != 0u) {
                  let y1_u = y1_base + ((k | 32u) ^ perm);
                  var sh1: u32;
                  if (NONNEG_Y) { sh1 = rotation_y_u16_k1_raw(select4b(xz1_lo, nb0, nb1), select4b(xz1_k1, nb0, nb1), select4b(xz1_k0, nb0, nb1), y1_u, neg_r1_base_m1); }
                  else { let ly1 = select4b(xz1_hi, nb0, nb1) ^ u32(i32(y1_u) >> 31u); sh1 = rotation_y_lh_raw(select4b(xz1_lo, nb0, nb1), ly1 * 84635722u, ly1 * 11u, y1_u, neg_r1_base_m1); }
                  if (((sh0 ^ sh1) & mask1_s) == 0u) {
                    hm1_b |= (1u << k);
                    nb_hi |= (((sh0 >> 14u) & 3u) << ((k - 16u) << 1u));
                  }
                }
              }
            }

            // Branchless joint drain of (hm1_a, hm1_b) -> Block 2 -> (hm2_a, hm2_b)
            var hm2_a = 0u;
            var hm2_b = 0u;
            let y2_base = u32(y + b2_y);
            let clip_lo = !NONNEG_Y && (y < ya);
            while ((hm1_a | hm1_b) != 0u) {
              let use_a = hm1_a != 0u;
              let m = select(hm1_b, hm1_a, use_a);
              let bit = countTrailingZeros(m);
              let cleared = m & (m - 1u);
              hm1_a = select(hm1_a, cleared, use_a);
              hm1_b = select(cleared, hm1_b, use_a);
              let dy = (bit | select(32u, 0u, use_a)) ^ perm;
              if (clip_lo && y + i32(dy) < ya) { continue; }
              let lo16 = bit < 16u;
              let nw = select(select(nb_hi, nb_lo, lo16), select(na_hi, na_lo, lo16), use_a);
              let pn = (nw >> ((bit & 15u) << 1u)) & 3u;
              let pb0 = (pn & 1u) != 0u;
              let pb1 = (pn & 2u) != 0u;
              let y2_u = y2_base + dy;
              var sh2: u32;
              if (NONNEG_Y) { sh2 = rotation_y_u16_k1_raw(select4b(xz2_lo, pb0, pb1), select4b(xz2_k1, pb0, pb1), select4b(xz2_k0, pb0, pb1), y2_u, neg_r2_base_m1); }
              else { let ly2 = select4b(xz2_hi, pb0, pb1) ^ u32(i32(y2_u) >> 31u); sh2 = rotation_y_lh_raw(select4b(xz2_lo, pb0, pb1), ly2 * 84635722u, ly2 * 11u, y2_u, neg_r2_base_m1); }
              let ok2 = select(0u, 1u << bit, ((sh2 ^ (pn << 14u)) & mask2_s) == 0u);
              hm2_a |= select(0u, ok2, use_a);
              hm2_b |= select(ok2, 0u, use_a);
            }

            // Joint drain of (hm2_a, hm2_b) -> Block 3+
            while ((hm2_a | hm2_b) != 0u) {
              let use_a = hm2_a != 0u;
              let m = select(hm2_b, hm2_a, use_a);
              let bit = countTrailingZeros(m);
              let cleared = m & (m - 1u);
              hm2_a = select(hm2_a, cleared, use_a);
              hm2_b = select(cleared, hm2_b, use_a);
              let lo16 = bit < 16u;
              let nw = select(select(nb_hi, nb_lo, lo16), select(na_hi, na_lo, lo16), use_a);
              let pn = (nw >> ((bit & 15u) << 1u)) & 3u;
              let dy = (bit | select(32u, 0u, use_a)) ^ perm;
              let cy = y + i32(dy);
              if (check_from(x, cy, z, pn, 3u)) {
                emit_result(x, cy, z, pn);
              }
            }
            y += select(32, 64, do64);
          }
          while (y <= yb) {
            let sh0 = rotation_y_lh_raw(xz0.x, k1, k0, u32(y), neg_rA_m1);
            let n = (sh0 >> 14u) & 3u;
            if (ALL_FOUR || ((b0_mask >> n) & 1u) != 0u) {
              let nb0 = (sh0 & 0x4000u) != 0u;
              let nb1 = (sh0 & 0x8000u) != 0u;
              let y1 = y + b1_y;
              var sh1: u32;
              if (NONNEG_Y) { sh1 = rotation_y_u16_k1_raw(select4b(xz1_lo, nb0, nb1), select4b(xz1_k1, nb0, nb1), select4b(xz1_k0, nb0, nb1), u32(y1), neg_r1_base_m1); }
              else { let ly1 = select4b(xz1_hi, nb0, nb1) ^ u32(y1 >> 31u); sh1 = rotation_y_lh_raw(select4b(xz1_lo, nb0, nb1), ly1 * 84635722u, ly1 * 11u, u32(y1), neg_r1_base_m1); }
              if (((sh0 ^ sh1) & mask1_s) == 0u) {
                let y2 = y + b2_y;
                var sh2: u32;
                if (NONNEG_Y) { sh2 = rotation_y_u16_k1_raw(select4b(xz2_lo, nb0, nb1), select4b(xz2_k1, nb0, nb1), select4b(xz2_k0, nb0, nb1), u32(y2), neg_r2_base_m1); }
                else { let ly2 = select4b(xz2_hi, nb0, nb1) ^ u32(y2 >> 31u); sh2 = rotation_y_lh_raw(select4b(xz2_lo, nb0, nb1), ly2 * 84635722u, ly2 * 11u, u32(y2), neg_r2_base_m1); }
                if (((sh0 ^ sh2) & mask2_s) == 0u && check_from(x, y, z, n, 3u)) {
                  emit_result(x, y, z, n);
                }
              }
            }
            y++;
          }
        } else {
          if (!ANCHOR_SIDE) {
            for (var y = ya; y <= yb; y++) {
              let sh0 = rotation_y_lh_raw(xz0.x, k1, k0, u32(y), neg_rA_m1);
              let n = (sh0 >> 14u) & 3u;
              if (!ALL_FOUR && ((b0_mask >> n) & 1u) == 0u) { continue; }
              if (check_rest_scalar(x, y, z, n, vec2<u32>(xz1_lo[n], xz1_hi[n]), b1_y, mask1_s)) {
                emit_result(x, y, z, n);
              }
            }
          } else {
            for (var y = ya; y <= yb; y++) {
              let sh0 = rotation_y_lh_raw(xz0.x, k1, k0, u32(y), neg_rA_m1);
              let p0 = (sh0 >> 14u) & 1u;
              for (var s = 0u; s < 2u; s++) {
                let n = p0 + (s << 1u);
                if (!ALL_FOUR && ((b0_mask >> n) & 1u) == 0u) { continue; }
                if (check_rest_scalar(x, y, z, n, vec2<u32>(xz1_lo[n], xz1_hi[n]), b1_y, mask1_s)) {
                  emit_result(x, y, z, n);
                }
              }
            }
          }
        }
      }
    }

    rx++;
    if (rx == p.width) {
      rx = 0u;
      qz++;
      zp0 = mulc_z(p.zmin + i32(qz));
    }
  }
}
