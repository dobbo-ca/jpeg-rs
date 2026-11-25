//! NEON kernels, the same shapes as `x86.rs` with 8 `i32` lanes as two `int32x4_t`. All are safe
//! `#[target_feature(enable = "neon")]` functions with no `unsafe`: they reach memory through the
//! helpers below, which turn a slice and an offset into an array reference (`super::arr`) and
//! pass it to `mem`, where the one load or store is. `dispatch` calls the kernels, one safe
//! function each, which takes the proof that `Tier::Neon` was detected.
use crate::encoder::Quant;
use crate::simd::{arr, arr_mut};
use std::arch::aarch64::*;

// The `unused_unsafe` allow: NEON is part of the aarch64 baseline, so the compiler may count the
// kernels as callable without `unsafe`.
#[allow(unsafe_code, unused_unsafe)]
pub(super) mod dispatch;
#[allow(unsafe_code)]
mod mem;

#[derive(Clone, Copy)]
struct V(int32x4_t, int32x4_t);

#[inline]
#[target_feature(enable = "neon")]
fn add(a: V, b: V) -> V {
    V(vaddq_s32(a.0, b.0), vaddq_s32(a.1, b.1))
}
#[inline]
#[target_feature(enable = "neon")]
fn sub(a: V, b: V) -> V {
    V(vsubq_s32(a.0, b.0), vsubq_s32(a.1, b.1))
}
#[inline]
#[target_feature(enable = "neon")]
fn mulk(a: V, k: i32) -> V {
    V(vmulq_n_s32(a.0, k), vmulq_n_s32(a.1, k))
}
#[inline]
#[target_feature(enable = "neon")]
fn shl<const N: i32>(a: V) -> V {
    V(vshlq_n_s32::<N>(a.0), vshlq_n_s32::<N>(a.1))
}
#[inline]
#[target_feature(enable = "neon")]
fn sra<const N: i32>(a: V) -> V {
    V(vshrq_n_s32::<N>(a.0), vshrq_n_s32::<N>(a.1))
}
#[inline]
#[target_feature(enable = "neon")]
fn and(a: V, b: V) -> V {
    V(vandq_s32(a.0, b.0), vandq_s32(a.1, b.1))
}
#[inline]
#[target_feature(enable = "neon")]
fn splat(k: i32) -> V {
    V(vdupq_n_s32(k), vdupq_n_s32(k))
}
#[inline]
#[target_feature(enable = "neon")]
fn dc_rows(a: V, b: V) -> (V, V) {
    (shl::<2>(add(a, b)), shl::<2>(sub(a, b)))
}
#[inline]
#[target_feature(enable = "neon")]
fn dc_cols(a: V, b: V) -> (V, V) {
    let two = splat(2);
    (sra::<2>(add(add(a, b), two)), sra::<2>(add(sub(a, b), two)))
}

// ---- slice access: each helper bounds-checks its slice (`arr`), then one access (`mem`) ----

#[inline]
#[target_feature(enable = "neon")]
fn ld8(s: &[u8], at: usize) -> uint8x8_t {
    mem::ld8(arr(s, at))
}
#[inline]
#[target_feature(enable = "neon")]
fn ld16(s: &[u8], at: usize) -> uint8x16_t {
    mem::ld16(arr(s, at))
}
#[inline]
#[target_feature(enable = "neon")]
fn ld4x8(s: &[u8], at: usize) -> uint8x8x4_t {
    mem::ld4x8(arr(s, at))
}
#[inline]
#[target_feature(enable = "neon")]
fn ld_i16x8(s: &[i16], at: usize) -> int16x8_t {
    mem::ld_i16x8(arr(s, at))
}
#[inline]
#[target_feature(enable = "neon")]
fn ld_u16x8(s: &[u16], at: usize) -> uint16x8_t {
    mem::ld_u16x8(arr(s, at))
}
#[inline]
#[target_feature(enable = "neon")]
fn ld_i32x8(s: &[i32], at: usize) -> V {
    let (lo, hi) = mem::ld_i32x8(arr(s, at));
    V(lo, hi)
}
#[inline]
#[target_feature(enable = "neon")]
fn st8(s: &mut [u8], at: usize, v: uint8x8_t) {
    mem::st8(arr_mut(s, at), v)
}
#[inline]
#[target_feature(enable = "neon")]
fn st16(s: &mut [u8], at: usize, v: uint8x16_t) {
    mem::st16(arr_mut(s, at), v)
}
#[inline]
#[target_feature(enable = "neon")]
fn st_i16x8(s: &mut [i16], at: usize, v: int16x8_t) {
    mem::st_i16x8(arr_mut(s, at), v)
}

/// Eight `i32` lanes to eight bytes, clamped to 0..=255.
#[inline]
#[target_feature(enable = "neon")]
fn narrow_u8(v: V) -> uint8x8_t {
    vqmovun_s16(vcombine_s16(vqmovn_s32(v.0), vqmovn_s32(v.1)))
}

/// Eight bytes to eight `i32` lanes.
#[inline]
#[target_feature(enable = "neon")]
fn widen(b: uint8x8_t) -> V {
    let w = vmovl_u8(b);
    V(
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(w))),
        vreinterpretq_s32_u32(vmovl_high_u16(w)),
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn clamp255(v: V) -> V {
    let (z, m) = (vdupq_n_s32(0), vdupq_n_s32(255));
    V(
        vminq_s32(vmaxq_s32(v.0, z), m),
        vminq_s32(vmaxq_s32(v.1, z), m),
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn or(a: V, b: V) -> V {
    V(vorrq_s32(a.0, b.0), vorrq_s32(a.1, b.1))
}

/// `m` raised to the largest absolute lane of `v`, as unsigned (`i32::MIN` counts as 2^31).
#[inline]
#[target_feature(enable = "neon")]
fn max_abs(m: uint32x4_t, v: V) -> uint32x4_t {
    let a = vmaxq_u32(
        vreinterpretq_u32_s32(vabsq_s32(v.0)),
        vreinterpretq_u32_s32(vabsq_s32(v.1)),
    );
    vmaxq_u32(m, a)
}

#[inline]
#[target_feature(enable = "neon")]
fn tr4(a: int32x4_t, b: int32x4_t, c: int32x4_t, d: int32x4_t) -> [int32x4_t; 4] {
    let t0 = vtrnq_s32(a, b);
    let t1 = vtrnq_s32(c, d);
    [
        vcombine_s32(vget_low_s32(t0.0), vget_low_s32(t1.0)),
        vcombine_s32(vget_low_s32(t0.1), vget_low_s32(t1.1)),
        vcombine_s32(vget_high_s32(t0.0), vget_high_s32(t1.0)),
        vcombine_s32(vget_high_s32(t0.1), vget_high_s32(t1.1)),
    ]
}

#[inline]
#[target_feature(enable = "neon")]
fn transpose(r: [V; 8]) -> [V; 8] {
    let tl = tr4(r[0].0, r[1].0, r[2].0, r[3].0);
    let bl = tr4(r[4].0, r[5].0, r[6].0, r[7].0);
    let tr = tr4(r[0].1, r[1].1, r[2].1, r[3].1);
    let br = tr4(r[4].1, r[5].1, r[6].1, r[7].1);
    [
        V(tl[0], bl[0]),
        V(tl[1], bl[1]),
        V(tl[2], bl[2]),
        V(tl[3], bl[3]),
        V(tr[0], br[0]),
        V(tr[1], br[1]),
        V(tr[2], br[2]),
        V(tr[3], br[3]),
    ]
}

/// `dct::idct8` in 32-bit lanes, exact for every `i16` coefficient times `u16` quantizer. Pass 1
/// is one pass while no sum can reach 2^31 (`max |dq| <= IDCT8_MAX_DQ`) and split in a high and a
/// low half (`idct_1d_split`) otherwise. Pass 2 wraps mod 2^32, which is exact: the scalar range
/// limit reads only bits 18..=27 of its sum, which the shift pair below keeps. Always returns
/// true, no block is refused.
#[target_feature(enable = "neon")]
fn idct8(coef: &[i16; 64], quant: &[u16; 64], dst: &mut [u8], stride: usize) -> bool {
    let mut v = [splat(0); 8];
    let mut mx = vdupq_n_u32(0);
    for r in 0..8 {
        let c = ld_i16x8(coef, 8 * r);
        let q = ld_u16x8(quant, 8 * r);
        let d = V(
            vmulq_s32(
                vmovl_s16(vget_low_s16(c)),
                vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(q))),
            ),
            vmulq_s32(vmovl_high_s16(c), vreinterpretq_s32_u32(vmovl_high_u16(q))),
        );
        mx = max_abs(mx, d);
        v[r] = d;
    }
    let ws = if vmaxvq_u32(mx) <= super::IDCT8_MAX_DQ {
        idct_1d!(v, 11)
    } else {
        idct_1d_split!(v, 11)
    };
    let o = transpose(idct_1d!(transpose(ws), 18));
    for r in 0..8 {
        // Bits 18..=27 of the sum, sign-extended: the scalar range limit's `x & 1023` wrap,
        // which `narrow_u8` then clamps.
        let wrapped = sra::<22>(shl::<22>(o[r]));
        st8(dst, r * stride, narrow_u8(add(wrapped, splat(128))));
    }
    true
}

/// `dct::idct4`, as `x86::idct4`: pass 1 over the columns, a transpose, pass 2 over the rows, a
/// transpose back and one 4-byte store per row. Pass 1 is one pass while `max |dq| <=
/// IDCT4_MAX_DQ`, split at 12 bits otherwise; pass 2 keeps bits 19..=28. `dst` must hold
/// `3 * stride + 4` bytes.
// ponytail: two full 8x8 transposes; pack two rows per register if the dec-rgba2 bench asks.
#[target_feature(enable = "neon")]
fn idct4(coef: &[i16; 64], quant: &[u16; 64], dst: &mut [u8], stride: usize) {
    let mut v = [splat(0); 8];
    let mut mx = vdupq_n_u32(0);
    for r in 0..8 {
        if r == 4 {
            continue;
        }
        let c = ld_i16x8(coef, 8 * r);
        let q = ld_u16x8(quant, 8 * r);
        let d = V(
            vmulq_s32(
                vmovl_s16(vget_low_s16(c)),
                vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(q))),
            ),
            vmulq_s32(vmovl_high_s16(c), vreinterpretq_s32_u32(vmovl_high_u16(q))),
        );
        mx = max_abs(mx, d);
        v[r] = d;
    }
    let ws = if vmaxvq_u32(mx) <= super::IDCT4_MAX_DQ {
        idct4_1d!(v, 12)
    } else {
        idct4_1d_split!(v, 12)
    };
    let z = splat(0);
    let t = transpose([ws[0], ws[1], ws[2], ws[3], z, z, z, z]);
    let q = idct4_1d!(t, 19);
    let o = transpose([q[0], q[1], q[2], q[3], z, z, z, z]);
    for r in 0..4 {
        let wrapped = sra::<22>(shl::<22>(o[r]));
        let px = vget_lane_u32::<0>(vreinterpret_u32_u8(narrow_u8(add(wrapped, splat(128)))));
        dst[r * stride..r * stride + 4].copy_from_slice(&px.to_le_bytes());
    }
}

/// `sample::color_row` for the first `y.len() & !7` pixels; returns how many it did.
#[target_feature(enable = "neon")]
fn color_row(dst: &mut [u8], y: &[u8], cb: &[u8], cr: &[u8]) -> usize {
    let n = y.len() & !7;
    let half = splat(32768);
    let alpha = splat(0xFF00_0000u32 as i32);
    let mut i = 0;
    while i < n {
        let yy = widen(ld8(y, i));
        let xb = sub(widen(ld8(cb, i)), splat(128));
        let xr = sub(widen(ld8(cr, i)), splat(128));
        let r = add(yy, sra::<16>(add(mulk(xr, 91881), half)));
        let g = add(
            yy,
            sra::<16>(add(add(mulk(xb, -22554), mulk(xr, -46802)), half)),
        );
        let b = add(yy, sra::<16>(add(mulk(xb, 116130), half)));
        let px = or(
            or(clamp255(r), shl::<8>(clamp255(g))),
            or(shl::<16>(clamp255(b)), alpha),
        );
        st16(dst, 4 * i, vreinterpretq_u8_s32(px.0));
        st16(dst, 4 * i + 16, vreinterpretq_u8_s32(px.1));
        i += 8;
    }
    n
}

/// `sample::h2v1_row`'s interior columns `start..end` in blocks of 8; returns the first column
/// not done. Needs `row[..=end]` and `dst[..2 * end]`.
#[target_feature(enable = "neon")]
fn h2v1_mid(dst: &mut [u8], row: &[u8], start: usize, end: usize) -> usize {
    let mut c = start;
    while c + 8 <= end {
        let prev = vmovl_u8(ld8(row, c - 1));
        let cur = vmovl_u8(ld8(row, c));
        let next = vmovl_u8(ld8(row, c + 1));
        let t = vmulq_n_u16(cur, 3);
        let even = vshrq_n_u16::<2>(vaddq_u16(vaddq_u16(t, prev), vdupq_n_u16(1)));
        let odd = vshrq_n_u16::<2>(vaddq_u16(vaddq_u16(t, next), vdupq_n_u16(2)));
        let comb = vorrq_u16(even, vshlq_n_u16::<8>(odd));
        st16(dst, 2 * c, vreinterpretq_u8_u16(comb));
        c += 8;
    }
    c
}

#[inline]
#[target_feature(enable = "neon")]
fn colsum(cur: &[u8], other: &[u8], at: usize) -> uint16x8_t {
    vaddq_u16(
        vmulq_n_u16(vmovl_u8(ld8(cur, at)), 3),
        vmovl_u8(ld8(other, at)),
    )
}

/// `sample::h2v2_row`'s interior columns, as `h2v1_mid`; `cur` and `other` need `[..=end]`.
#[target_feature(enable = "neon")]
fn h2v2_mid(dst: &mut [u8], cur: &[u8], other: &[u8], start: usize, end: usize) -> usize {
    let mut c = start;
    while c + 8 <= end {
        let prev = colsum(cur, other, c - 1);
        let this = colsum(cur, other, c);
        let next = colsum(cur, other, c + 1);
        let t = vmulq_n_u16(this, 3);
        let even = vshrq_n_u16::<4>(vaddq_u16(vaddq_u16(t, prev), vdupq_n_u16(8)));
        let odd = vshrq_n_u16::<4>(vaddq_u16(vaddq_u16(t, next), vdupq_n_u16(7)));
        let comb = vorrq_u16(even, vshlq_n_u16::<8>(odd));
        st16(dst, 2 * c, vreinterpretq_u8_u16(comb));
        c += 8;
    }
    c
}

/// floor((|x| + 4q) / 8q) with x's sign, four lanes (see `x86::fdct_quant_rows`).
#[inline]
#[target_feature(enable = "neon")]
fn quant4(x: int32x4_t, q: int32x4_t) -> int32x4_t {
    let a = vaddq_s32(vabsq_s32(x), vshlq_n_s32::<2>(q));
    let f = vdivq_f32(vcvtq_f32_s32(a), vcvtq_f32_s32(vshlq_n_s32::<3>(q)));
    let r = vcvtq_s32_f32(f);
    vbslq_s32(vcltq_s32(x, vdupq_n_s32(0)), vnegq_s32(r), r)
}

/// `encoder::fdct_quant_rows`. `pix` must hold `off + 7 * stride + 8` bytes.
#[target_feature(enable = "neon")]
fn fdct_quant_rows(pix: &[u8], off: usize, stride: usize, quant: &Quant, coef: &mut [i16; 64]) {
    let mut s = [splat(0); 8];
    for r in 0..8 {
        s[r] = sub(widen(ld8(pix, off + r * stride)), splat(128));
    }
    let rows = transpose(fdct_1d!(transpose(s), 11, dc_rows));
    let o = fdct_1d!(rows, 15, dc_cols);
    for k in 0..8 {
        let q = ld_i32x8(&quant.q, 8 * k);
        let lo = quant4(o[k].0, q.0);
        let hi = quant4(o[k].1, q.1);
        st_i16x8(coef, 8 * k, vcombine_s16(vqmovn_s32(lo), vqmovn_s32(hi)));
    }
}

/// `encoder::rgba_to_ycbcr_row` for the first `y.len() & !7` pixels; returns how many.
#[target_feature(enable = "neon")]
fn rgba_to_ycbcr_row(src: &[u8], y: &mut [u8], cb: &mut [u8], cr: &mut [u8]) -> usize {
    let n = y.len() & !7;
    let (yb, cbr) = (splat(32768), splat((128 << 16) + 32767));
    let mut i = 0;
    while i < n {
        let p = ld4x8(src, 4 * i);
        let (r, g, b) = (widen(p.0), widen(p.1), widen(p.2));
        let yv = sra::<16>(add(
            add(add(mulk(r, 19595), mulk(g, 38470)), mulk(b, 7471)),
            yb,
        ));
        let cbv = sra::<16>(add(
            add(add(mulk(r, -11059), mulk(g, -21709)), mulk(b, 32768)),
            cbr,
        ));
        let crv = sra::<16>(add(
            add(add(mulk(r, 32768), mulk(g, -27439)), mulk(b, -5329)),
            cbr,
        ));
        st8(y, i, narrow_u8(yv));
        st8(cb, i, narrow_u8(cbv));
        st8(cr, i, narrow_u8(crv));
        i += 8;
    }
    n
}

/// `encoder::downsample_h2v2_row` for the first `dst.len() & !7` outputs; returns how many.
#[target_feature(enable = "neon")]
fn downsample_h2v2_row(dst: &mut [u8], r0: &[u8], r1: &[u8]) -> usize {
    let n = dst.len() & !7;
    let bias = vreinterpretq_u16_u32(vdupq_n_u32(0x0002_0001)); // u16 lanes 1, 2, 1, 2, ...
    let mut i = 0;
    while i < n {
        let a = vpaddlq_u8(ld16(r0, 2 * i));
        let b = vpaddlq_u8(ld16(r1, 2 * i));
        st8(dst, i, vshrn_n_u16::<2>(vaddq_u16(vaddq_u16(a, b), bias)));
        i += 8;
    }
    n
}
