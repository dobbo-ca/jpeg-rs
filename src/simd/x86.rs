//! AVX2 kernels, all safe `#[target_feature]` functions: `avx2` for most, `avx2` with
//! `bmi1,bmi2,lzcnt,popcnt` for `write_block`, `bmi1,bmi2,lzcnt` alone for `decode_mcu_fast`, and
//! `avx512f,avx512bw` besides `avx2` for the `_512` kernels and `fdct_quant_pair`. A kernel has no
//! `unsafe`: it reaches memory through the helpers below, which take a slice and an offset, turn
//! them into an array reference (`super::arr`) and pass that to `mem`, where the one unaligned
//! load or store is. A wrong length panics instead of reading past the slice; the dispatchers
//! check lengths first and never reach that. `dispatch` calls the kernels, one safe function
//! each, which takes the proof that the features were detected.
use crate::dct::Block;
use crate::encoder::{BitWriter, HuffCode, Quant, ZZ_PERM, code_block, prep_coef, zigzag_mask};
use crate::huff::{BitReader, DerivedTable};
use crate::scan::decode_mcu_fast_impl;
use crate::simd::{arr, arr_mut};
use std::arch::x86_64::*;

#[allow(unsafe_code)]
pub(super) mod dispatch;
#[allow(unsafe_code)]
mod mem;

type V = __m256i;

#[inline]
#[target_feature(enable = "avx2")]
fn add(a: V, b: V) -> V {
    _mm256_add_epi32(a, b)
}
#[inline]
#[target_feature(enable = "avx2")]
fn sub(a: V, b: V) -> V {
    _mm256_sub_epi32(a, b)
}
#[inline]
#[target_feature(enable = "avx2")]
fn mulk(a: V, k: i32) -> V {
    _mm256_mullo_epi32(a, _mm256_set1_epi32(k))
}
#[inline]
#[target_feature(enable = "avx2")]
fn shl<const N: i32>(a: V) -> V {
    _mm256_slli_epi32::<N>(a)
}
#[inline]
#[target_feature(enable = "avx2")]
fn sra<const N: i32>(a: V) -> V {
    _mm256_srai_epi32::<N>(a)
}
#[inline]
#[target_feature(enable = "avx2")]
fn and(a: V, b: V) -> V {
    _mm256_and_si256(a, b)
}
#[inline]
#[target_feature(enable = "avx2")]
fn splat(k: i32) -> V {
    _mm256_set1_epi32(k)
}
// ---- slice access: each helper bounds-checks its slice (`arr`), then one unaligned access (`mem`) ----

#[inline]
#[target_feature(enable = "avx2")]
fn ld8(s: &[u8], at: usize) -> __m128i {
    mem::ld8(arr(s, at))
}
#[inline]
#[target_feature(enable = "avx2")]
fn ld16(s: &[u8], at: usize) -> __m128i {
    mem::ld16(arr(s, at))
}
#[inline]
#[target_feature(enable = "avx2")]
fn ld32(s: &[u8], at: usize) -> V {
    mem::ld32(arr(s, at))
}
#[inline]
#[target_feature(enable = "avx2")]
fn ld_i16x8(s: &[i16], at: usize) -> __m128i {
    mem::ld_i16x8(arr(s, at))
}
#[inline]
#[target_feature(enable = "avx2")]
fn ld_u16x8(s: &[u16], at: usize) -> __m128i {
    mem::ld_u16x8(arr(s, at))
}
#[inline]
#[target_feature(enable = "avx2")]
fn st8(s: &mut [u8], at: usize, v: __m128i) {
    mem::st8(arr_mut(s, at), v)
}
#[inline]
#[target_feature(enable = "avx2")]
fn sth8(s: &mut [u8], at: usize, v: __m128i) {
    mem::sth8(arr_mut(s, at), v)
}
#[inline]
#[target_feature(enable = "avx2")]
fn st16(s: &mut [u8], at: usize, v: __m128i) {
    mem::st16(arr_mut(s, at), v)
}
#[inline]
#[target_feature(enable = "avx2")]
fn st32(s: &mut [u8], at: usize, v: V) {
    mem::st32(arr_mut(s, at), v)
}
/// `3 * cur + other` over the 16 columns at `at`, as u16 lanes.
#[inline]
#[target_feature(enable = "avx2")]
fn colsum(cur: &[u8], other: &[u8], at: usize) -> V {
    let a = _mm256_cvtepu8_epi16(ld16(cur, at));
    let b = _mm256_cvtepu8_epi16(ld16(other, at));
    _mm256_add_epi16(_mm256_mullo_epi16(a, _mm256_set1_epi16(3)), b)
}

/// Eight `i32` lanes to eight bytes, clamped to 0..=255.
#[inline]
#[target_feature(enable = "avx2")]
fn narrow_u8(v: V) -> __m128i {
    let w = _mm_packs_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
    _mm_packus_epi16(w, w)
}

/// Whether every unsigned lane of `mx` is at most `lim`.
#[inline]
#[target_feature(enable = "avx2")]
fn all_le(mx: V, lim: u32) -> bool {
    let l = _mm256_set1_epi32(lim as i32);
    _mm256_movemask_epi8(_mm256_cmpeq_epi32(_mm256_max_epu32(mx, l), l)) == -1
}

#[inline]
#[target_feature(enable = "avx2")]
fn transpose(r: [V; 8]) -> [V; 8] {
    let t0 = _mm256_unpacklo_epi32(r[0], r[1]);
    let t1 = _mm256_unpackhi_epi32(r[0], r[1]);
    let t2 = _mm256_unpacklo_epi32(r[2], r[3]);
    let t3 = _mm256_unpackhi_epi32(r[2], r[3]);
    let t4 = _mm256_unpacklo_epi32(r[4], r[5]);
    let t5 = _mm256_unpackhi_epi32(r[4], r[5]);
    let t6 = _mm256_unpacklo_epi32(r[6], r[7]);
    let t7 = _mm256_unpackhi_epi32(r[6], r[7]);
    let u0 = _mm256_unpacklo_epi64(t0, t2);
    let u1 = _mm256_unpackhi_epi64(t0, t2);
    let u2 = _mm256_unpacklo_epi64(t1, t3);
    let u3 = _mm256_unpackhi_epi64(t1, t3);
    let u4 = _mm256_unpacklo_epi64(t4, t6);
    let u5 = _mm256_unpackhi_epi64(t4, t6);
    let u6 = _mm256_unpacklo_epi64(t5, t7);
    let u7 = _mm256_unpackhi_epi64(t5, t7);
    [
        _mm256_permute2x128_si256::<0x20>(u0, u4),
        _mm256_permute2x128_si256::<0x20>(u1, u5),
        _mm256_permute2x128_si256::<0x20>(u2, u6),
        _mm256_permute2x128_si256::<0x20>(u3, u7),
        _mm256_permute2x128_si256::<0x31>(u0, u4),
        _mm256_permute2x128_si256::<0x31>(u1, u5),
        _mm256_permute2x128_si256::<0x31>(u2, u6),
        _mm256_permute2x128_si256::<0x31>(u3, u7),
    ]
}

/// `dct::idct8`, as Go's `idct8AVX2Asm`: when every `coef * quant` fits an `i16` (and no quantizer
/// has bit 15 set, so the 16-bit product is the true one), pass 1 is a 16-bit dequantize
/// (`mullo`) and `madd` on row pairs, one dword per column, exact in 32 bits (the sums stay under
/// 2.01e9). Pass 2 wraps in 32 bits, which is exact too: the scalar range limit reads only bits
/// 18..=27 of its 64-bit sum, and those are in the low 32. So the result is the scalar's for every
/// such block, with no further check. Any other block goes to [`idct8_wide`]. Returns true: no
/// block is refused. `dst` must hold `7 * stride + 8` bytes.
#[target_feature(enable = "avx2")]
fn idct8(coef: &[i16; 64], quant: &[u16; 64], dst: &mut [u8], stride: usize) -> bool {
    let mut d = [splat(0); 4];
    let (mut over, mut qs) = (splat(0), splat(0));
    for k in 0..4 {
        let c = ld_i16x16(coef, 16 * k);
        let q = ld_u16x16(quant, 16 * k);
        let lo = _mm256_mullo_epi16(c, q);
        // The signed high word is the sign of the low word exactly when the product fits.
        let hi = _mm256_mulhi_epi16(c, q);
        over = _mm256_or_si256(over, _mm256_xor_si256(hi, _mm256_srai_epi16::<15>(lo)));
        qs = _mm256_or_si256(qs, q);
        d[k] = lo;
    }
    over = _mm256_or_si256(over, _mm256_and_si256(qs, _mm256_set1_epi16(i16::MIN)));
    if _mm256_testz_si256(over, over) == 0 {
        return idct8_wide(coef, quant, dst, stride);
    }
    // d[k] holds rows 2k and 2k+1; after the permute each lane has columns 0-3 (low) or 4-7
    // (high) of both rows, so the unpacks give one dword per column: (row 0, 4), (1, 5), (2, 6), (3, 7).
    let d0 = _mm256_permute4x64_epi64::<0xd8>(d[0]);
    let d1 = _mm256_permute4x64_epi64::<0xd8>(d[1]);
    let d2 = _mm256_permute4x64_epi64::<0xd8>(d[2]);
    let d3 = _mm256_permute4x64_epi64::<0xd8>(d[3]);
    let p04 = _mm256_unpacklo_epi16(d0, d2);
    let p15 = _mm256_unpackhi_epi16(d0, d2);
    let p26 = _mm256_unpacklo_epi16(d1, d3);
    let p37 = _mm256_unpackhi_epi16(d1, d3);
    // The word pairs (first row, second row) of each `madd`: the jidctint.c sums multiplied out
    // per coefficient, so no 16-bit value is ever added.
    macro_rules! kw {
        ($d:literal) => {
            splat($d as u32 as i32)
        };
    }
    macro_rules! odd {
        ($a:literal, $b:literal) => {
            add(
                _mm256_madd_epi16(p15, kw!($a)),
                _mm256_madd_epi16(p37, kw!($b)),
            )
        };
    }
    let bias = splat(1 << 10);
    let tmp0 = add(_mm256_madd_epi16(p04, kw!(0x2000_2000)), bias);
    let tmp1 = add(_mm256_madd_epi16(p04, kw!(0xE000_2000)), bias);
    let tmp2 = _mm256_madd_epi16(p26, kw!(0xD630_1151));
    let tmp3 = _mm256_madd_epi16(p26, kw!(0x1151_29CF));
    let t10 = add(tmp0, tmp3);
    let t13 = sub(tmp0, tmp3);
    let t11 = add(tmp1, tmp2);
    let t12 = sub(tmp1, tmp2);
    let o0 = odd!(0x25A1_08D4, 0xD39D_E6DC);
    let o1 = odd!(0x08D5_1925, 0x25A1_D39E);
    let o2 = odd!(0xD39E_25A1, 0xE6DC_F72D);
    let o3 = odd!(0x1925_2C63, 0x08D4_25A1);
    let ws = [
        sra::<11>(add(t10, o3)),
        sra::<11>(add(t11, o2)),
        sra::<11>(add(t12, o1)),
        sra::<11>(add(t13, o0)),
        sra::<11>(sub(t13, o0)),
        sra::<11>(sub(t12, o1)),
        sra::<11>(sub(t11, o2)),
        sra::<11>(sub(t10, o3)),
    ];
    // Pass 2 of column `x` of the output is `o[x]`, one lane per row. The output is not
    // transposed back: the bytes are packed and then transposed, as Go's `idct8AVX2Asm`.
    // Every constant is scaled by 16 (wrapping in 32 bits), so bits 22..=31 of a sum are bits
    // 18..=27 of the scalar sum, and `sra::<22>` is the scalar range limit's `x & 1023` wrap with
    // its sign extension; the rounding constant 2^17 becomes the `16` added to `v[0]` and then
    // shifted by 17. The signed saturating packs clamp to -128..=127 and flipping bit 7 adds
    // 128, which is the clamp to 0..=255 of `packus(x + 128)` (the sign-extended value lies in
    // -512..=511, so the i32 to i16 saturation of `packs` changes nothing). The bytes are the
    // old kernel's.
    let v = transpose(ws);
    let z1 = mulk(add(v[2], v[6]), 4433 * 16);
    let tmp2 = add(z1, mulk(v[6], -15137 * 16));
    let tmp3 = add(z1, mulk(v[2], 6270 * 16));
    let w0 = add(v[0], splat(16));
    let tmp0 = shl::<17>(add(w0, v[4]));
    let tmp1 = shl::<17>(sub(w0, v[4]));
    let tmp10 = add(tmp0, tmp3);
    let tmp13 = sub(tmp0, tmp3);
    let tmp11 = add(tmp1, tmp2);
    let tmp12 = sub(tmp1, tmp2);
    let (t0, t1, t2, t3) = (v[7], v[5], v[3], v[1]);
    let z1 = add(t0, t3);
    let z2 = add(t1, t2);
    let z3 = add(t0, t2);
    let z4 = add(t1, t3);
    let z5 = mulk(add(z3, z4), 9633 * 16);
    let t0 = mulk(t0, 2446 * 16);
    let t1 = mulk(t1, 16819 * 16);
    let t2 = mulk(t2, 25172 * 16);
    let t3 = mulk(t3, 12299 * 16);
    let z1 = mulk(z1, -7373 * 16);
    let z2 = mulk(z2, -20995 * 16);
    let z3 = add(mulk(z3, -16069 * 16), z5);
    let z4 = add(mulk(z4, -3196 * 16), z5);
    let t0 = add(t0, add(z1, z3));
    let t1 = add(t1, add(z2, z4));
    let t2 = add(t2, add(z2, z3));
    let t3 = add(t3, add(z1, z4));
    let o = [
        sra::<22>(add(tmp10, t3)),
        sra::<22>(add(tmp11, t2)),
        sra::<22>(add(tmp12, t1)),
        sra::<22>(add(tmp13, t0)),
        sra::<22>(sub(tmp13, t0)),
        sra::<22>(sub(tmp12, t1)),
        sra::<22>(sub(tmp11, t2)),
        sra::<22>(sub(tmp10, t3)),
    ];
    let flip = _mm256_set1_epi8(i8::MIN);
    // Lane 0 of `a` is columns 0..=3 of rows 0..=3 (byte `4 * col + row`), lane 1 rows 4..=7;
    // `b` is columns 4..=7.
    let a = _mm256_xor_si256(
        _mm256_packs_epi16(
            _mm256_packs_epi32(o[0], o[1]),
            _mm256_packs_epi32(o[2], o[3]),
        ),
        flip,
    );
    let b = _mm256_xor_si256(
        _mm256_packs_epi16(
            _mm256_packs_epi32(o[4], o[5]),
            _mm256_packs_epi32(o[6], o[7]),
        ),
        flip,
    );
    // Byte `4 * row + col` of each lane.
    let sh = _mm256_setr_epi8(
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10,
        14, 3, 7, 11, 15,
    );
    let a = _mm256_shuffle_epi8(a, sh);
    let b = _mm256_shuffle_epi8(b, sh);
    // `lo` holds rows 0, 1 (lane 0) and 4, 5 (lane 1); `hi` rows 2, 3 and 6, 7.
    let lo = _mm256_unpacklo_epi32(a, b);
    let hi = _mm256_unpackhi_epi32(a, b);
    let rows = [
        _mm256_castsi256_si128(lo),
        _mm256_castsi256_si128(hi),
        _mm256_extracti128_si256::<1>(lo),
        _mm256_extracti128_si256::<1>(hi),
    ];
    // One bounds check for the eight rows (as Go's `_ = dst[7*stride+7]`): every row below
    // lies inside this slice, so the per-row checks of `arr_mut` fold away. The dispatcher has
    // checked the length, so the slice cannot be short.
    let dst = &mut dst[..7 * stride + 8];
    for (i, v) in rows.into_iter().enumerate() {
        st8(dst, 2 * i * stride, v);
        sth8(dst, (2 * i + 1) * stride, v);
    }
    true
}

/// [`idct8`] for the blocks it cannot take (a product past 16 bits, or a quantizer with bit 15
/// set): 32-bit lanes throughout, exact for every `i16` coefficient times `u16` quantizer. Pass 1
/// is one pass while no sum can reach 2^31 (`max |dq| <= IDCT8_MAX_DQ`) and split in a high and a
/// low half otherwise (`idct_1d_split`). Pass 2 wraps mod 2^32, which is exact: the scalar range
/// limit reads only bits 18..=27 of its sum, so the output stage is [`idct8`]'s. Always returns
/// true.
#[target_feature(enable = "avx2")]
fn idct8_wide(coef: &[i16; 64], quant: &[u16; 64], dst: &mut [u8], stride: usize) -> bool {
    let mut v = [splat(0); 8];
    let mut mx = splat(0);
    for r in 0..8 {
        let c = _mm256_cvtepi16_epi32(ld_i16x8(coef, 8 * r));
        let q = _mm256_cvtepu16_epi32(ld_u16x8(quant, 8 * r));
        let d = _mm256_mullo_epi32(c, q);
        mx = _mm256_max_epu32(mx, _mm256_abs_epi32(d));
        v[r] = d;
    }
    let ws = if all_le(mx, super::IDCT8_MAX_DQ) {
        idct_1d!(v, 11)
    } else {
        idct_1d_split!(v, 11)
    };
    let o = transpose(idct_1d!(transpose(ws), 18));
    for r in 0..8 {
        let wrapped = sra::<22>(shl::<22>(o[r]));
        st8(dst, r * stride, narrow_u8(add(wrapped, splat(128))));
    }
    true
}

/// `dct::idct4`, 32-bit lanes: pass 1 over the columns (lane = column, 4 output rows; row 4 and
/// column 4 are not inputs), a transpose, pass 2 over the rows, a transpose back and one 4-byte
/// store per row. Pass 1 is one pass while `max |dq| <= IDCT4_MAX_DQ` and split at 12 bits
/// otherwise; pass 2 wraps mod 2^32 and keeps bits 19..=28 as `idct8` keeps 18..=27. Exact for
/// every `i16` coefficient times `u16` quantizer. `dst` must hold `3 * stride + 4` bytes.
// ponytail: two full 8x8 transposes; pack two rows per register if the dec-rgba2 bench asks.
#[target_feature(enable = "avx2")]
fn idct4(coef: &[i16; 64], quant: &[u16; 64], dst: &mut [u8], stride: usize) {
    let mut v = [splat(0); 8];
    let mut mx = splat(0);
    for r in 0..8 {
        if r == 4 {
            continue;
        }
        let c = _mm256_cvtepi16_epi32(ld_i16x8(coef, 8 * r));
        let q = _mm256_cvtepu16_epi32(ld_u16x8(quant, 8 * r));
        let d = _mm256_mullo_epi32(c, q);
        mx = _mm256_max_epu32(mx, _mm256_abs_epi32(d));
        v[r] = d;
    }
    let ws = if all_le(mx, super::IDCT4_MAX_DQ) {
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
        let px = _mm_cvtsi128_si32(narrow_u8(add(wrapped, splat(128))));
        dst[r * stride..r * stride + 4].copy_from_slice(&px.to_le_bytes());
    }
}

/// `sample::color_row` over the first `y.len() & !15` pixels (16 per step, in 16-bit lanes, as Go's
/// `colorRowAVX2Asm`), exact by splitting the constants: `(91881x + 32768) >> 16 =
/// x + ((26345x + 32768) >> 16)` (the product's high word plus its rounding bit),
/// `(116130x + 32768) >> 16 = 2x + ((32768 - 14942x) >> 16)` (`mulhrs` by half the constant) and G's
/// `>> 16 = -xr + ((-22554xb + 18734xr + 32768) >> 16)` (`madd` on word pairs). The sums are in
/// -285..=538, so the unsigned pack clamps to 0..=255.
#[target_feature(enable = "avx2")]
fn color_row(dst: &mut [u8], y: &[u8], cb: &[u8], cr: &[u8]) -> usize {
    let n = y.len() & !15;
    let c128 = _mm256_set1_epi16(128);
    let c_rr = _mm256_set1_epi16(26345);
    let c_bb = _mm256_set1_epi16(-7471);
    // The word pair [-22554, 18734] (low word first) in every dword.
    let c_g = _mm256_set1_epi32(18734 << 16 | (-22554i32 as u16 as i32));
    let bias = _mm256_set1_epi32(32768);
    let c255 = _mm256_set1_epi16(255);
    // Interleaves the low and high 8 bytes of each lane.
    let zip = _mm256_setr_epi8(
        0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15, 0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5,
        13, 6, 14, 7, 15,
    );
    let mut i = 0;
    while i < n {
        let yy = _mm256_cvtepu8_epi16(ld16(y, i));
        let xb = _mm256_sub_epi16(_mm256_cvtepu8_epi16(ld16(cb, i)), c128);
        let xr = _mm256_sub_epi16(_mm256_cvtepu8_epi16(ld16(cr, i)), c128);
        let rnd = _mm256_srli_epi16::<15>(_mm256_mullo_epi16(xr, c_rr));
        let r = _mm256_add_epi16(
            _mm256_add_epi16(_mm256_mulhi_epi16(xr, c_rr), rnd),
            _mm256_add_epi16(xr, yy),
        );
        let b = _mm256_add_epi16(
            _mm256_add_epi16(_mm256_mulhrs_epi16(xb, c_bb), _mm256_add_epi16(xb, xb)),
            yy,
        );
        let gl = _mm256_madd_epi16(_mm256_unpacklo_epi16(xb, xr), c_g);
        let gh = _mm256_madd_epi16(_mm256_unpackhi_epi16(xb, xr), c_g);
        let g = _mm256_add_epi16(
            _mm256_packs_epi32(
                _mm256_srai_epi32::<16>(_mm256_add_epi32(gl, bias)),
                _mm256_srai_epi32::<16>(_mm256_add_epi32(gh, bias)),
            ),
            _mm256_sub_epi16(yy, xr),
        );
        // Per lane: R0-7 G0-7 and B0-7 A0-7, zipped to R0 G0 R1 G1 ... and B0 A0 ..., then
        // to RGBA pixels, restored to column order across the lanes.
        let rg = _mm256_shuffle_epi8(_mm256_packus_epi16(r, g), zip);
        let ba = _mm256_shuffle_epi8(_mm256_packus_epi16(b, c255), zip);
        let p0 = _mm256_unpacklo_epi16(rg, ba);
        let p1 = _mm256_unpackhi_epi16(rg, ba);
        st32(dst, 4 * i, _mm256_permute2x128_si256::<0x20>(p0, p1));
        st32(dst, 4 * i + 32, _mm256_permute2x128_si256::<0x31>(p0, p1));
        i += 16;
    }
    n
}

/// 32 interior columns at `c` of `sample::h2v1_row` in byte lanes, as Go's `h2v1FancyRowAVX2Asm`:
/// `(3b + a + 1) >> 2 = floor((ceil((a + b) / 2) + b) / 2)` and `(3b + c + 2) >> 2 =
/// ceil((floor((c + b) / 2) + b) / 2)`, with `avg_epu8` the rounding average and the floor average
/// `NOT(avg(NOT x, NOT y))`. Reads `row[c - 1..c + 33]`, writes `dst[2 * c..2 * c + 64]`.
#[inline]
#[target_feature(enable = "avx2")]
fn h2v1_32(dst: &mut [u8], row: &[u8], c: usize) {
    let ones = _mm256_set1_epi8(-1);
    let a = ld32(row, c - 1);
    let b = ld32(row, c);
    let n = ld32(row, c + 1);
    let nb = _mm256_xor_si256(b, ones);
    let t = _mm256_xor_si256(_mm256_avg_epu8(b, a), ones);
    let left = _mm256_xor_si256(_mm256_avg_epu8(nb, t), ones);
    let f = _mm256_xor_si256(_mm256_avg_epu8(nb, _mm256_xor_si256(n, ones)), ones);
    let right = _mm256_avg_epu8(b, f);
    // Per lane: columns 0-7 | 16-23 and 8-15 | 24-31, restored to column order across the lanes.
    let lo = _mm256_unpacklo_epi8(left, right);
    let hi = _mm256_unpackhi_epi8(left, right);
    st32(dst, 2 * c, _mm256_permute2x128_si256::<0x20>(lo, hi));
    st32(dst, 2 * c + 32, _mm256_permute2x128_si256::<0x31>(lo, hi));
}

/// 16 interior columns at `c` of `sample::h2v1_row`, in 16-bit lanes, for what is left under 32.
#[inline]
#[target_feature(enable = "avx2")]
fn h2v1_16(dst: &mut [u8], row: &[u8], c: usize) {
    let (one, two, three) = (
        _mm256_set1_epi16(1),
        _mm256_set1_epi16(2),
        _mm256_set1_epi16(3),
    );
    let prev = _mm256_cvtepu8_epi16(ld16(row, c - 1));
    let cur = _mm256_cvtepu8_epi16(ld16(row, c));
    let next = _mm256_cvtepu8_epi16(ld16(row, c + 1));
    let t = _mm256_mullo_epi16(cur, three);
    let even = _mm256_srli_epi16::<2>(_mm256_add_epi16(_mm256_add_epi16(t, prev), one));
    let odd = _mm256_srli_epi16::<2>(_mm256_add_epi16(_mm256_add_epi16(t, next), two));
    st32(
        dst,
        2 * c,
        _mm256_or_si256(even, _mm256_slli_epi16::<8>(odd)),
    );
}

/// `sample::h2v1_row`'s interior columns `start..end`, 32 at a time (then 16 when fewer than 32
/// are asked for); the last block is redone ending at `end` (the overlap rewrites equal bytes), so
/// it returns `end`, or `start` when fewer than 16 columns are asked for. Needs `row[..=end]` and
/// `dst[..2 * end]`.
#[target_feature(enable = "avx2")]
fn h2v1_mid(dst: &mut [u8], row: &[u8], start: usize, end: usize) -> usize {
    let mut c = start;
    let n = end.saturating_sub(start);
    if n >= 32 {
        while c + 32 <= end {
            h2v1_32(dst, row, c);
            c += 32;
        }
        if c < end {
            h2v1_32(dst, row, end - 32);
        }
        return end;
    }
    if n >= 16 {
        h2v1_16(dst, row, start);
        if start + 16 < end {
            h2v1_16(dst, row, end - 16);
        }
        return end;
    }
    start
}

/// 16 interior columns at `c` of `sample::h2v2_row`: `prev` has `colsum(c - 1)` in its last word.
/// Per column, the colsums of the columns before and after come from the colsums at `c` and
/// `c + 1` and from `prev`, shifted by one word as Go's `h2v2FancyRowAVX2Asm` does. Returns the
/// colsums at `c`, the next call's `prev`.
#[inline]
#[target_feature(enable = "avx2")]
fn h2v2_16(dst: &mut [u8], cur: &[u8], other: &[u8], c: usize, prev: V) -> V {
    let (seven, eight, three) = (
        _mm256_set1_epi16(7),
        _mm256_set1_epi16(8),
        _mm256_set1_epi16(3),
    );
    let this = colsum(cur, other, c);
    let next = colsum(cur, other, c + 1);
    // [prev's high lane | this's low lane], then one word later: colsum(c - 1 + k) in word k.
    let before = _mm256_alignr_epi8::<14>(this, _mm256_permute2x128_si256::<0x21>(prev, this));
    let t = _mm256_mullo_epi16(this, three);
    let even = _mm256_srli_epi16::<4>(_mm256_add_epi16(_mm256_add_epi16(t, before), eight));
    let odd = _mm256_srli_epi16::<4>(_mm256_add_epi16(_mm256_add_epi16(t, next), seven));
    st32(
        dst,
        2 * c,
        _mm256_or_si256(even, _mm256_slli_epi16::<8>(odd)),
    );
    this
}

/// `3 * cur[c] + other[c]` in every word, the seed of [`h2v2_16`]'s `prev`.
#[inline]
#[target_feature(enable = "avx2")]
fn colsum_seed(cur: &[u8], other: &[u8], c: usize) -> V {
    _mm256_set1_epi16((3 * u16::from(cur[c]) + u16::from(other[c])) as i16)
}

/// `sample::h2v2_row`'s interior columns, as `h2v1_mid`: blocks of 16, the last redone ending at
/// `end`; `cur` and `other` need `[..=end]`.
#[target_feature(enable = "avx2")]
fn h2v2_mid(dst: &mut [u8], cur: &[u8], other: &[u8], start: usize, end: usize) -> usize {
    if end.saturating_sub(start) < 16 {
        return start;
    }
    let mut c = start;
    let mut prev = colsum_seed(cur, other, c - 1);
    while c + 16 <= end {
        prev = h2v2_16(dst, cur, other, c, prev);
        c += 16;
    }
    if c < end {
        c = end - 16;
        prev = colsum_seed(cur, other, c - 1);
        h2v2_16(dst, cur, other, c, prev);
    }
    end
}

/// Two 8-pixel rows (`at0`, `at1`) as 16 words, the first in the low lane.
#[inline]
#[target_feature(enable = "avx2")]
fn load_rows(pix: &[u8], at0: usize, at1: usize) -> V {
    _mm256_cvtepu8_epi16(_mm_unpacklo_epi64(ld8(pix, at0), ld8(pix, at1)))
}

/// The word pairs `[lo, hi]` of a `madd` constant: the dword `d0` in the low lane, `d1` in the high.
#[inline]
#[target_feature(enable = "avx2")]
fn pairs(d0: u32, d1: u32) -> V {
    _mm256_setr_epi32(
        d0 as i32, d0 as i32, d0 as i32, d0 as i32, d1 as i32, d1 as i32, d1 as i32, d1 as i32,
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn ld_u16x16(s: &[u16], at: usize) -> V {
    mem::ld_u16x16(arr(s, at))
}

#[inline]
#[target_feature(enable = "avx2")]
fn st_i16x16(s: &mut [i16], at: usize, v: V) {
    mem::st_i16x16(arr_mut(s, at), v)
}

/// libjpeg-turbo's quantizer (`jcquant-avx2.asm`) on 16 words: `((|x| + corr) * recip >> 16) * scale
/// >> 16` with the sign of `x`.
#[inline]
#[target_feature(enable = "avx2")]
fn quant16(x: V, quant: &Quant, at: usize) -> V {
    let a = _mm256_add_epi16(_mm256_abs_epi16(x), ld_u16x16(&quant.corr, at));
    let a = _mm256_mulhi_epu16(a, ld_u16x16(&quant.recip, at));
    let a = _mm256_mulhi_epu16(a, ld_u16x16(&quant.scale, at));
    _mm256_sign_epi16(a, x)
}

/// `encoder::fdct_quant_rows`: libjpeg-turbo's `jfdctint-avx2.asm` and quantizer fused, in 16-bit
/// lanes with two rows to a register, instruction for instruction as Go's `fdctQuantAsm`. The
/// macros are its `DOTRANSPOSE`, `DODCT_A` and `DODCT_B`; a register is a variable. `pix` must hold
/// `off + 7 * stride + 8` bytes.
#[target_feature(enable = "avx2")]
#[allow(unused_assignments)] // the last values of the macros' scratch registers
fn fdct_quant_rows(pix: &[u8], off: usize, stride: usize, quant: &Quant, coef: &mut [i16; 64]) {
    let kf130 = pairs(0x1151_29cf, 0x1151_d630);
    let kf078 = pairs(0x25a1_e6dc, 0x25a1_1925);
    let kf060 = pairs(0xe333_ecc1, 0xadfd_efb0);
    let kf050 = pairs(0xadfd_1051, 0xe333_133e);
    let (kpd1, kpd2) = (splat(1024), splat(16384));
    let kp2x = _mm256_set1_epi16(2);
    let k1n1 = _mm256_setr_epi16(1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1);
    let km128 = _mm256_set1_epi16(-128);

    macro_rules! transpose4 {
        ($a1:ident, $a2:ident, $a3:ident, $a4:ident, $a5:ident, $a6:ident, $a7:ident, $a8:ident) => {
            $a5 = _mm256_unpacklo_epi16($a1, $a2);
            $a6 = _mm256_unpackhi_epi16($a1, $a2);
            $a7 = _mm256_unpacklo_epi16($a3, $a4);
            $a8 = _mm256_unpackhi_epi16($a3, $a4);
            $a1 = _mm256_unpacklo_epi32($a5, $a7);
            $a2 = _mm256_unpackhi_epi32($a5, $a7);
            $a3 = _mm256_unpacklo_epi32($a6, $a8);
            $a4 = _mm256_unpackhi_epi32($a6, $a8);
            $a1 = _mm256_permute4x64_epi64::<0x8d>($a1);
            $a2 = _mm256_permute4x64_epi64::<0x8d>($a2);
            $a3 = _mm256_permute4x64_epi64::<0xd8>($a3);
            $a4 = _mm256_permute4x64_epi64::<0xd8>($a4);
        };
    }
    macro_rules! dct_a {
        ($a1:ident, $a2:ident, $a3:ident, $a4:ident, $a5:ident, $a6:ident, $a7:ident, $a8:ident) => {
            $a5 = _mm256_sub_epi16($a1, $a4);
            $a6 = _mm256_add_epi16($a1, $a4);
            $a7 = _mm256_add_epi16($a2, $a3);
            $a8 = _mm256_sub_epi16($a2, $a3);
            $a6 = _mm256_permute2x128_si256::<0x01>($a6, $a6);
            $a1 = _mm256_add_epi16($a6, $a7);
            $a6 = _mm256_sub_epi16($a6, $a7);
            $a7 = _mm256_permute2x128_si256::<0x01>($a1, $a1);
            $a1 = _mm256_sign_epi16($a1, k1n1);
            $a7 = _mm256_add_epi16($a7, $a1);
        };
    }
    // `$k` rounds, `$n` shifts: one butterfly's rotations by madd.
    macro_rules! dct_b {
        ($a2:ident, $a3:ident, $a4:ident, $a5:ident, $a6:ident, $a7:ident, $a8:ident, $kpd:ident, $n:literal) => {
            $a7 = _mm256_permute2x128_si256::<0x01>($a6, $a6);
            $a2 = _mm256_unpacklo_epi16($a6, $a7);
            $a6 = _mm256_unpackhi_epi16($a6, $a7);
            $a2 = _mm256_madd_epi16($a2, kf130);
            $a6 = _mm256_madd_epi16($a6, kf130);
            $a2 = _mm256_srai_epi32::<$n>(_mm256_add_epi32($a2, $kpd));
            $a6 = _mm256_srai_epi32::<$n>(_mm256_add_epi32($a6, $kpd));
            $a3 = _mm256_packs_epi32($a2, $a6);
            $a7 = _mm256_add_epi16($a8, $a5);
            $a2 = _mm256_permute2x128_si256::<0x01>($a7, $a7);
            $a6 = _mm256_unpacklo_epi16($a7, $a2);
            $a7 = _mm256_unpackhi_epi16($a7, $a2);
            $a6 = _mm256_madd_epi16($a6, kf078);
            $a7 = _mm256_madd_epi16($a7, kf078);
            $a4 = _mm256_permute2x128_si256::<0x01>($a5, $a5);
            $a2 = _mm256_unpacklo_epi16($a8, $a4);
            $a4 = _mm256_unpackhi_epi16($a8, $a4);
            $a2 = _mm256_madd_epi16($a2, kf060);
            $a4 = _mm256_madd_epi16($a4, kf060);
            $a2 = _mm256_add_epi32($a2, $a6);
            $a4 = _mm256_add_epi32($a4, $a7);
            $a2 = _mm256_srai_epi32::<$n>(_mm256_add_epi32($a2, $kpd));
            $a4 = _mm256_srai_epi32::<$n>(_mm256_add_epi32($a4, $kpd));
            $a4 = _mm256_packs_epi32($a2, $a4);
            $a2 = _mm256_permute2x128_si256::<0x01>($a8, $a8);
            $a8 = _mm256_unpacklo_epi16($a5, $a2);
            $a5 = _mm256_unpackhi_epi16($a5, $a2);
            $a8 = _mm256_madd_epi16($a8, kf050);
            $a5 = _mm256_madd_epi16($a5, kf050);
            $a8 = _mm256_add_epi32($a8, $a6);
            $a5 = _mm256_add_epi32($a5, $a7);
            $a8 = _mm256_srai_epi32::<$n>(_mm256_add_epi32($a8, $kpd));
            $a5 = _mm256_srai_epi32::<$n>(_mm256_add_epi32($a5, $kpd));
            $a2 = _mm256_packs_epi32($a8, $a5);
        };
    }

    // Rows 0|4, 1|5, 2|6, 3|7, level shifted.
    let s = 3 * stride;
    let mut y0 = _mm256_add_epi16(load_rows(pix, off, off + 4 * stride), km128);
    let mut y1 = _mm256_add_epi16(load_rows(pix, off + stride, off + 5 * stride), km128);
    let mut y2 = _mm256_add_epi16(load_rows(pix, off + 2 * stride, off + 6 * stride), km128);
    let mut y3 = _mm256_add_epi16(load_rows(pix, off + s, off + 7 * stride), km128);
    let (mut y4, mut y5, mut y6, mut y7) = (y0, y0, y0, y0);

    // Pass 1: rows.
    transpose4!(y0, y1, y2, y3, y4, y5, y6, y7);
    dct_a!(y0, y1, y2, y3, y4, y5, y6, y7);
    y0 = _mm256_slli_epi16::<2>(y6);
    // The macro's a1 is only an input of the first half of DODCT_B (DC): y0 is not rewritten.
    dct_b!(y1, y2, y3, y4, y5, y6, y7, kpd1, 11);
    // (y0 = DC, y2 = 2|6, y3 = 3|7?, y1 = 1|5: the registers as Go's.)
    y4 = _mm256_permute2x128_si256::<0x20>(y1, y3);
    y1 = _mm256_permute2x128_si256::<0x31>(y1, y3);

    // Pass 2: columns.
    transpose4!(y0, y1, y2, y4, y3, y5, y6, y7);
    dct_a!(y0, y1, y2, y4, y3, y5, y6, y7);
    y6 = _mm256_add_epi16(y6, kp2x);
    y0 = _mm256_srai_epi16::<2>(y6);
    dct_b!(y1, y2, y4, y3, y5, y6, y7, kpd2, 15);

    let x0 = _mm256_permute2x128_si256::<0x30>(y0, y1);
    let x1 = _mm256_permute2x128_si256::<0x20>(y2, y1);
    let x2 = _mm256_permute2x128_si256::<0x31>(y0, y4);
    let x3 = _mm256_permute2x128_si256::<0x21>(y2, y4);
    st_i16x16(coef, 0, quant16(x0, quant, 0));
    st_i16x16(coef, 16, quant16(x1, quant, 16));
    st_i16x16(coef, 32, quant16(x2, quant, 32));
    st_i16x16(coef, 48, quant16(x3, quant, 48));
}

/// The lanes of 8 pixels (R, G, B, A bytes each) to their Y, Cb and Cr as dwords in 0..=255, as
/// Go's `YCC8`: `pshufb` pairs R with G and G with B as words for `madd` (38470 and 32768 do not
/// fit a signed word, so Y uses 19595R + 19235G and 19235G + 7471B, Cb and Cr add 32768 * B or R
/// as a shift).
#[inline]
#[target_feature(enable = "avx2")]
fn ycc8(p: V, k: &YccConsts) -> (V, V, V) {
    let rg = _mm256_shuffle_epi8(p, k.shuf_rg);
    let gb = _mm256_shuffle_epi8(p, k.shuf_gb);
    let y = _mm256_srli_epi32::<16>(add(
        add(_mm256_madd_epi16(rg, k.y_rg), _mm256_madd_epi16(gb, k.y_gb)),
        k.half,
    ));
    let b = _mm256_slli_epi32::<15>(_mm256_shuffle_epi8(p, k.shuf_b));
    let cb = _mm256_srli_epi32::<16>(add(add(_mm256_madd_epi16(rg, k.cb), b), k.bias));
    let r = _mm256_slli_epi32::<15>(_mm256_shuffle_epi8(p, k.shuf_r));
    let cr = _mm256_srli_epi32::<16>(add(add(_mm256_madd_epi16(gb, k.cr), r), k.bias));
    (y, cb, cr)
}

struct YccConsts {
    shuf_rg: V,
    shuf_gb: V,
    shuf_r: V,
    shuf_b: V,
    y_rg: V,
    y_gb: V,
    cb: V,
    cr: V,
    half: V,
    bias: V,
}

/// The word pair `[lo, hi]` in every dword, for `madd`.
#[inline]
#[target_feature(enable = "avx2")]
fn word_pair(lo: i32, hi: i32) -> V {
    _mm256_set1_epi32(hi << 16 | (lo & 0xffff))
}

/// `m` in both 128-bit lanes.
#[inline]
#[target_feature(enable = "avx2")]
fn lane(m: __m128i) -> V {
    _mm256_broadcastsi128_si256(m)
}

/// 16 dwords in 0..=255 (`a` then `b`) to 16 bytes, in order.
#[inline]
#[target_feature(enable = "avx2")]
fn pack16(a: V, b: V) -> __m128i {
    let w = _mm256_packus_epi32(a, b); // a0-3 b0-3 | a4-7 b4-7, as words
    let bytes = _mm_packus_epi16(_mm256_castsi256_si128(w), _mm256_extracti128_si256::<1>(w));
    _mm_shuffle_epi32::<0xd8>(bytes) // dwords 0, 2, 1, 3: a0-7 b0-7
}

/// `encoder::rgba_to_ycbcr_row` for the first `y.len() & !15` pixels (16 per step); returns
/// how many.
#[target_feature(enable = "avx2")]
fn rgba_to_ycbcr_row(src: &[u8], y: &mut [u8], cb: &mut [u8], cr: &mut [u8]) -> usize {
    let n = y.len() & !15;
    let k = YccConsts {
        shuf_rg: lane(_mm_setr_epi8(
            0, -128, 1, -128, 4, -128, 5, -128, 8, -128, 9, -128, 12, -128, 13, -128,
        )),
        shuf_gb: lane(_mm_setr_epi8(
            1, -128, 2, -128, 5, -128, 6, -128, 9, -128, 10, -128, 13, -128, 14, -128,
        )),
        shuf_r: lane(_mm_setr_epi8(
            0, -128, -128, -128, 4, -128, -128, -128, 8, -128, -128, -128, 12, -128, -128, -128,
        )),
        shuf_b: lane(_mm_setr_epi8(
            2, -128, -128, -128, 6, -128, -128, -128, 10, -128, -128, -128, 14, -128, -128, -128,
        )),
        y_rg: word_pair(19595, 19235),
        y_gb: word_pair(19235, 7471),
        cb: word_pair(-11059, -21709),
        cr: word_pair(-27439, -5329),
        half: splat(32768),
        bias: splat((128 << 16) + 32767),
    };
    let mut i = 0;
    while i < n {
        let (y0, b0, r0) = ycc8(ld32(src, 4 * i), &k);
        let (y1, b1, r1) = ycc8(ld32(src, 4 * i + 32), &k);
        st16(y, i, pack16(y0, y1));
        st16(cb, i, pack16(b0, b1));
        st16(cr, i, pack16(r0, r1));
        i += 16;
    }
    n
}

/// `encoder::downsample_h2v2_row` for the first `dst.len() & !15` outputs; returns how many.
#[target_feature(enable = "avx2")]
fn downsample_h2v2_row(dst: &mut [u8], r0: &[u8], r1: &[u8]) -> usize {
    let n = dst.len() & !15;
    let ones = _mm256_set1_epi8(1);
    let bias = _mm256_set1_epi32(0x0002_0001); // u16 lanes 1, 2, 1, 2, ...
    let mut i = 0;
    while i < n {
        let a = _mm256_maddubs_epi16(ld32(r0, 2 * i), ones);
        let b = _mm256_maddubs_epi16(ld32(r1, 2 * i), ones);
        let s = _mm256_srli_epi16::<2>(_mm256_add_epi16(_mm256_add_epi16(a, b), bias));
        let p = _mm_packus_epi16(_mm256_castsi256_si128(s), _mm256_extracti128_si256::<1>(s));
        st16(dst, i, p);
        i += 16;
    }
    n
}

#[inline]
#[target_feature(enable = "avx2")]
fn ld_i16x16(s: &[i16], at: usize) -> V {
    mem::ld_i16x16(arr(s, at))
}
#[inline]
#[target_feature(enable = "avx2")]
fn st_u16x16(s: &mut [u16], at: usize, v: V) {
    mem::st_u16x16(arr_mut(s, at), v)
}

/// The bit length of each non-negative 32-bit lane (up to 2^24): the exponent of its f32, which
/// for 0 is negative.
#[inline]
#[target_feature(enable = "avx2")]
fn bit_len(x: V) -> V {
    let e = _mm256_srli_epi32::<23>(_mm256_castps_si256(_mm256_cvtepi32_ps(x)));
    _mm256_sub_epi32(e, _mm256_set1_epi32(126))
}

/// The 32 lanes of `a` then `b` (16-bit, each all ones or all zeros) as 32 bits, in order.
#[inline]
#[target_feature(enable = "avx2")]
fn lane_mask(a: V, b: V) -> u64 {
    // The pack works within 128-bit lanes: put the quadwords back in order.
    let p = _mm256_permute4x64_epi64::<0xd8>(_mm256_packs_epi16(a, b));
    _mm256_movemask_epi8(p) as u32 as u64
}

/// The value bits (`v + (v >> 31)` masked to the bit length) and the bit length of each 32-bit
/// lane; the length of 0 is negative, which the caller's unsigned saturation turns into 0.
#[inline]
#[target_feature(enable = "avx2")]
fn prep32(x: V) -> (V, V) {
    let one = splat(1);
    let nb = bit_len(_mm256_abs_epi32(x));
    // A shift count over 31 gives 0, so a zero coefficient gets an all-ones mask and value 0.
    let mask = sub(_mm256_sllv_epi32(one, nb), one);
    let vals = _mm256_and_si256(add(x, _mm256_srai_epi32::<31>(x)), mask);
    (vals, nb)
}

/// The mask of the nonzero coefficients, natural order.
#[inline]
#[target_feature(enable = "avx2")]
fn nonzero_mask(blk: &[i16; 64]) -> u64 {
    let zero = _mm256_setzero_si256();
    let z0 = _mm256_cmpeq_epi16(ld_i16x16(blk, 0), zero);
    let z1 = _mm256_cmpeq_epi16(ld_i16x16(blk, 16), zero);
    let z2 = _mm256_cmpeq_epi16(ld_i16x16(blk, 32), zero);
    let z3 = _mm256_cmpeq_epi16(ld_i16x16(blk, 48), zero);
    !(lane_mask(z0, z1) | lane_mask(z2, z3) << 32)
}

/// `encoder::prep_coef` of all 64 coefficients. The entries of zero coefficients are not used.
#[inline]
#[target_feature(enable = "avx2")]
fn prep_vectors(blk: &[i16; 64], vals: &mut [u16; 64], nb: &mut [u16; 64]) {
    for i in 0..4 {
        let v = ld_i16x16(blk, 16 * i);
        // The 16 coefficients as two vectors of sign-extended 32-bit lanes; packing the results
        // back restores the order.
        let sign = _mm256_srai_epi16::<15>(v);
        let (vl, nl) = prep32(_mm256_unpacklo_epi16(v, sign));
        let (vh, nh) = prep32(_mm256_unpackhi_epi16(v, sign));
        st_u16x16(vals, 16 * i, _mm256_packus_epi32(vl, vh));
        st_u16x16(nb, 16 * i, _mm256_packus_epi32(nl, nh));
    }
}

/// `encoder::write_block_scalar`: the AVX2 prep (for a block with few coefficients, the scalar one
/// of each), and the coding loop built with BMI1, BMI2, LZCNT and POPCNT (`tzcnt`, `blsr`, `bzhi`,
/// `shlx`).
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
fn write_block(
    bw: &mut BitWriter,
    blk: &[i16; 64],
    last_dc: &mut i32,
    dct: &[HuffCode; 256],
    act: &[HuffCode; 256],
) {
    let (mut vals, mut nb) = ([0u16; 64], [0u16; 64]);
    let nat = nonzero_mask(blk);
    if nat.count_ones() > 8 {
        prep_vectors(blk, &mut vals, &mut nb);
    } else {
        // Few coefficients: prepare just those.
        let mut m = nat;
        while m != 0 {
            let n = m.trailing_zeros() as usize & 63;
            m &= m - 1;
            (vals[n], nb[n]) = prep_coef(i32::from(blk[n]));
        }
    }
    code_block::<true>(
        bw,
        blk[0],
        last_dc,
        dct,
        act,
        zigzag_mask(nat) & !1,
        &vals,
        &nb,
    );
}

/// `scan::decode_mcu_fast_impl` built with BMI1, BMI2 and LZCNT: variable shifts without `cl`.
#[target_feature(enable = "bmi1,bmi2,lzcnt")]
fn decode_mcu_fast<const STORE: bool>(
    r: &mut BitReader,
    dc: &[DerivedTable; 4],
    ac: &[DerivedTable; 4],
    last_dc: &mut [i32; 4],
    blks: &mut [Block],
    ks: &[usize],
    has_ac: Option<&mut [bool]>,
) -> bool {
    decode_mcu_fast_impl::<STORE>(r, dc, ac, last_dc, blks, ks, has_ac)
}

// ---- AVX-512 (F and BW): the wide kernels (the encoder's, as Go's `fdctQuantPairAsm` and the wide loop of
// `rgbaToYCbCrAsm`, and the decoder's colour and upsample, which Go has in AVX2 only); `super` runs
// them only when `has_avx512` ----

type Z = __m512i;

#[inline]
#[target_feature(enable = "avx512f")]
fn ld64(s: &[u8], at: usize) -> Z {
    mem::ld64(arr(s, at))
}

#[inline]
#[target_feature(enable = "avx512f")]
fn st64(s: &mut [u8], at: usize, v: Z) {
    mem::st64(arr_mut(s, at), v)
}

#[inline]
#[target_feature(enable = "avx512f")]
fn ld_u16x32(s: &[u16], at: usize) -> Z {
    mem::ld_u16x32(arr(s, at))
}

/// The word pairs of [`pairs`] in both 256-bit halves.
#[inline]
#[target_feature(enable = "avx512f")]
fn pairs512(d0: u32, d1: u32) -> Z {
    let (a, b) = (d0 as i32, d1 as i32);
    _mm512_setr_epi32(a, a, a, a, b, b, b, b, a, a, a, a, b, b, b, b)
}

#[inline]
#[target_feature(enable = "avx512f")]
fn word_pair512(lo: i32, hi: i32) -> Z {
    _mm512_set1_epi32(hi << 16 | (lo & 0xffff))
}

#[inline]
#[target_feature(enable = "avx512f")]
fn lane512(m: __m128i) -> Z {
    _mm512_broadcast_i32x4(m)
}

/// Rows `a0` and `a1` of block a (low 256 bits) and `b0` and `b1` of block b (high), as 32 words.
#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw")]
fn load_rows2(pa: &[u8], a0: usize, a1: usize, pb: &[u8], b0: usize, b1: usize) -> Z {
    let xa = _mm_unpacklo_epi64(ld8(pa, a0), ld8(pa, a1));
    let xb = _mm_unpacklo_epi64(ld8(pb, b0), ld8(pb, b1));
    _mm512_cvtepu8_epi16(_mm256_inserti128_si256::<1>(_mm256_castsi128_si256(xa), xb))
}

/// [`quant16`] on two blocks, with `Quant`'s pair tables.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn quant16z(x: Z, quant: &Quant, at: usize) -> Z {
    let a = _mm512_add_epi16(_mm512_abs_epi16(x), ld_u16x32(&quant.pair_corr, at));
    let a = _mm512_mulhi_epu16(a, ld_u16x32(&quant.pair_recip, at));
    let a = _mm512_mulhi_epu16(a, ld_u16x32(&quant.pair_scale, at));
    _mm512_mask_sub_epi16(a, _mm512_movepi16_mask(x), _mm512_setzero_si512(), a)
}

/// `encoder::fdct_quant_rows` of blocks `a` and `b` (each `(pix, offset, stride)`) that share
/// `quant`, at once: [`fdct_quant_rows`] with block a in the low and block b in the high 256 bits
/// of each register, instruction for instruction as Go's `fdctQuantPairAsm` (`VPERM2I128` is
/// `permutex_epi64::<0x4e>` or `permutex2var` per half, `VPSIGNW` a masked negate). Each block
/// must hold `off + 7 * stride + 8` bytes.
#[target_feature(enable = "avx2,avx512f,avx512bw")]
#[allow(unused_assignments)] // the last values of the macros' scratch registers
fn fdct_quant_pair(
    a: (&[u8], usize, usize),
    b: (&[u8], usize, usize),
    quant: &Quant,
    ca: &mut [i16; 64],
    cb: &mut [i16; 64],
) {
    let kf130 = pairs512(0x1151_29cf, 0x1151_d630);
    let kf078 = pairs512(0x25a1_e6dc, 0x25a1_1925);
    let kf060 = pairs512(0xe333_ecc1, 0xadfd_efb0);
    let kf050 = pairs512(0xadfd_1051, 0xe333_133e);
    let (kpd1, kpd2) = (_mm512_set1_epi32(1024), _mm512_set1_epi32(16384));
    let kp2x = _mm512_set1_epi16(2);
    let km128 = _mm512_set1_epi16(-128);
    let zero = _mm512_setzero_si512();
    // Words 8-15 of each 256-bit half: the lanes DODCT_A negates.
    let k1: __mmask32 = 0xff00_ff00;
    // `permutex2var` selectors for VPERM2I128 0x20, 0x31, 0x30 and 0x21 in each 256-bit half.
    let pidx20 = _mm512_setr_epi64(0, 1, 8, 9, 4, 5, 12, 13);
    let pidx31 = _mm512_setr_epi64(2, 3, 10, 11, 6, 7, 14, 15);
    let pidx30 = _mm512_setr_epi64(0, 1, 10, 11, 4, 5, 14, 15);
    let pidx21 = _mm512_setr_epi64(2, 3, 8, 9, 6, 7, 12, 13);

    macro_rules! transpose4 {
        ($a1:ident, $a2:ident, $a3:ident, $a4:ident, $a5:ident, $a6:ident, $a7:ident, $a8:ident) => {
            $a5 = _mm512_unpacklo_epi16($a1, $a2);
            $a6 = _mm512_unpackhi_epi16($a1, $a2);
            $a7 = _mm512_unpacklo_epi16($a3, $a4);
            $a8 = _mm512_unpackhi_epi16($a3, $a4);
            $a1 = _mm512_unpacklo_epi32($a5, $a7);
            $a2 = _mm512_unpackhi_epi32($a5, $a7);
            $a3 = _mm512_unpacklo_epi32($a6, $a8);
            $a4 = _mm512_unpackhi_epi32($a6, $a8);
            $a1 = _mm512_permutex_epi64::<0x8d>($a1);
            $a2 = _mm512_permutex_epi64::<0x8d>($a2);
            $a3 = _mm512_permutex_epi64::<0xd8>($a3);
            $a4 = _mm512_permutex_epi64::<0xd8>($a4);
        };
    }
    macro_rules! dct_a {
        ($a1:ident, $a2:ident, $a3:ident, $a4:ident, $a5:ident, $a6:ident, $a7:ident, $a8:ident) => {
            $a5 = _mm512_sub_epi16($a1, $a4);
            $a6 = _mm512_add_epi16($a1, $a4);
            $a7 = _mm512_add_epi16($a2, $a3);
            $a8 = _mm512_sub_epi16($a2, $a3);
            $a6 = _mm512_permutex_epi64::<0x4e>($a6);
            $a1 = _mm512_add_epi16($a6, $a7);
            $a6 = _mm512_sub_epi16($a6, $a7);
            $a7 = _mm512_permutex_epi64::<0x4e>($a1);
            $a1 = _mm512_mask_sub_epi16($a1, k1, zero, $a1);
            $a7 = _mm512_add_epi16($a7, $a1);
        };
    }
    macro_rules! dct_b {
        ($a2:ident, $a3:ident, $a4:ident, $a5:ident, $a6:ident, $a7:ident, $a8:ident, $kpd:ident, $n:literal) => {
            $a7 = _mm512_permutex_epi64::<0x4e>($a6);
            $a2 = _mm512_unpacklo_epi16($a6, $a7);
            $a6 = _mm512_unpackhi_epi16($a6, $a7);
            $a2 = _mm512_madd_epi16($a2, kf130);
            $a6 = _mm512_madd_epi16($a6, kf130);
            $a2 = _mm512_srai_epi32::<$n>(_mm512_add_epi32($a2, $kpd));
            $a6 = _mm512_srai_epi32::<$n>(_mm512_add_epi32($a6, $kpd));
            $a3 = _mm512_packs_epi32($a2, $a6);
            $a7 = _mm512_add_epi16($a8, $a5);
            $a2 = _mm512_permutex_epi64::<0x4e>($a7);
            $a6 = _mm512_unpacklo_epi16($a7, $a2);
            $a7 = _mm512_unpackhi_epi16($a7, $a2);
            $a6 = _mm512_madd_epi16($a6, kf078);
            $a7 = _mm512_madd_epi16($a7, kf078);
            $a4 = _mm512_permutex_epi64::<0x4e>($a5);
            $a2 = _mm512_unpacklo_epi16($a8, $a4);
            $a4 = _mm512_unpackhi_epi16($a8, $a4);
            $a2 = _mm512_madd_epi16($a2, kf060);
            $a4 = _mm512_madd_epi16($a4, kf060);
            $a2 = _mm512_add_epi32($a2, $a6);
            $a4 = _mm512_add_epi32($a4, $a7);
            $a2 = _mm512_srai_epi32::<$n>(_mm512_add_epi32($a2, $kpd));
            $a4 = _mm512_srai_epi32::<$n>(_mm512_add_epi32($a4, $kpd));
            $a4 = _mm512_packs_epi32($a2, $a4);
            $a2 = _mm512_permutex_epi64::<0x4e>($a8);
            $a8 = _mm512_unpacklo_epi16($a5, $a2);
            $a5 = _mm512_unpackhi_epi16($a5, $a2);
            $a8 = _mm512_madd_epi16($a8, kf050);
            $a5 = _mm512_madd_epi16($a5, kf050);
            $a8 = _mm512_add_epi32($a8, $a6);
            $a5 = _mm512_add_epi32($a5, $a7);
            $a8 = _mm512_srai_epi32::<$n>(_mm512_add_epi32($a8, $kpd));
            $a5 = _mm512_srai_epi32::<$n>(_mm512_add_epi32($a5, $kpd));
            $a2 = _mm512_packs_epi32($a8, $a5);
        };
    }

    let ((pa, oa, sa), (pb, ob, sb)) = (a, b);
    let mut y0 = _mm512_add_epi16(load_rows2(pa, oa, oa + 4 * sa, pb, ob, ob + 4 * sb), km128);
    let mut y1 = _mm512_add_epi16(
        load_rows2(pa, oa + sa, oa + 5 * sa, pb, ob + sb, ob + 5 * sb),
        km128,
    );
    let mut y2 = _mm512_add_epi16(
        load_rows2(pa, oa + 2 * sa, oa + 6 * sa, pb, ob + 2 * sb, ob + 6 * sb),
        km128,
    );
    let mut y3 = _mm512_add_epi16(
        load_rows2(pa, oa + 3 * sa, oa + 7 * sa, pb, ob + 3 * sb, ob + 7 * sb),
        km128,
    );
    let (mut y4, mut y5, mut y6, mut y7) = (y0, y0, y0, y0);

    // Pass 1: rows.
    transpose4!(y0, y1, y2, y3, y4, y5, y6, y7);
    dct_a!(y0, y1, y2, y3, y4, y5, y6, y7);
    y0 = _mm512_slli_epi16::<2>(y6);
    dct_b!(y1, y2, y3, y4, y5, y6, y7, kpd1, 11);
    y4 = _mm512_permutex2var_epi64(y1, pidx20, y3);
    y1 = _mm512_permutex2var_epi64(y1, pidx31, y3);

    // Pass 2: columns.
    transpose4!(y0, y1, y2, y4, y3, y5, y6, y7);
    dct_a!(y0, y1, y2, y4, y3, y5, y6, y7);
    y6 = _mm512_add_epi16(y6, kp2x);
    y0 = _mm512_srai_epi16::<2>(y6);
    dct_b!(y1, y2, y4, y3, y5, y6, y7, kpd2, 15);

    let x0 = _mm512_permutex2var_epi64(y0, pidx30, y1);
    let x1 = _mm512_permutex2var_epi64(y2, pidx20, y1);
    let x2 = _mm512_permutex2var_epi64(y0, pidx31, y4);
    let x3 = _mm512_permutex2var_epi64(y2, pidx21, y4);
    for (k, x) in [x0, x1, x2, x3].into_iter().enumerate() {
        let q = quant16z(x, quant, 32 * k);
        st_i16x16(ca, 16 * k, _mm512_castsi512_si256(q));
        st_i16x16(cb, 16 * k, _mm512_extracti64x4_epi64::<1>(q));
    }
}

/// `rgba_to_ycbcr_row` with 16 pixels to a register: `YCC8` on all four 128-bit lanes, and the
/// dwords truncated to bytes (every one is 0..=255).
#[target_feature(enable = "avx2,avx512f,avx512bw")]
fn rgba_to_ycbcr_row_512(src: &[u8], y: &mut [u8], cb: &mut [u8], cr: &mut [u8]) -> usize {
    let n = y.len() & !15;
    let shuf_rg = lane512(_mm_setr_epi8(
        0, -128, 1, -128, 4, -128, 5, -128, 8, -128, 9, -128, 12, -128, 13, -128,
    ));
    let shuf_gb = lane512(_mm_setr_epi8(
        1, -128, 2, -128, 5, -128, 6, -128, 9, -128, 10, -128, 13, -128, 14, -128,
    ));
    let shuf_r = lane512(_mm_setr_epi8(
        0, -128, -128, -128, 4, -128, -128, -128, 8, -128, -128, -128, 12, -128, -128, -128,
    ));
    let shuf_b = lane512(_mm_setr_epi8(
        2, -128, -128, -128, 6, -128, -128, -128, 10, -128, -128, -128, 14, -128, -128, -128,
    ));
    let (y_rg, y_gb) = (word_pair512(19595, 19235), word_pair512(19235, 7471));
    let (k_cb, k_cr) = (word_pair512(-11059, -21709), word_pair512(-27439, -5329));
    let (half, bias) = (
        _mm512_set1_epi32(32768),
        _mm512_set1_epi32((128 << 16) + 32767),
    );
    let mut i = 0;
    while i < n {
        let p = ld64(src, 4 * i);
        let rg = _mm512_shuffle_epi8(p, shuf_rg);
        let gb = _mm512_shuffle_epi8(p, shuf_gb);
        let yv = _mm512_srli_epi32::<16>(_mm512_add_epi32(
            _mm512_add_epi32(_mm512_madd_epi16(rg, y_rg), _mm512_madd_epi16(gb, y_gb)),
            half,
        ));
        let b = _mm512_slli_epi32::<15>(_mm512_shuffle_epi8(p, shuf_b));
        let cbv = _mm512_srli_epi32::<16>(_mm512_add_epi32(
            _mm512_add_epi32(_mm512_madd_epi16(rg, k_cb), b),
            bias,
        ));
        let r = _mm512_slli_epi32::<15>(_mm512_shuffle_epi8(p, shuf_r));
        let crv = _mm512_srli_epi32::<16>(_mm512_add_epi32(
            _mm512_add_epi32(_mm512_madd_epi16(gb, k_cr), r),
            bias,
        ));
        st16(y, i, _mm512_cvtepi32_epi8(yv));
        st16(cb, i, _mm512_cvtepi32_epi8(cbv));
        st16(cr, i, _mm512_cvtepi32_epi8(crv));
        i += 16;
    }
    n
}

/// [`color_row`] with 32 pixels to a register, op for op in 512-bit lanes; the pack and unpack
/// steps work per 128-bit lane, so two `permutex2var` restore the pixels to column order before the
/// two 64-byte stores. The 16 pixels left under 32 go to [`color_row`]; returns `y.len() & !15` too.
#[target_feature(enable = "avx2,avx512f,avx512bw")]
fn color_row_512(dst: &mut [u8], y: &[u8], cb: &[u8], cr: &[u8]) -> usize {
    let n = y.len() & !15;
    let c128 = _mm512_set1_epi16(128);
    let c_rr = _mm512_set1_epi16(26345);
    let c_bb = _mm512_set1_epi16(-7471);
    let c_g = word_pair512(-22554, 18734);
    let bias = _mm512_set1_epi32(32768);
    let c255 = _mm512_set1_epi16(255);
    let zip = lane512(_mm_setr_epi8(
        0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15,
    ));
    // Pixels 0-3 and 4-7 of each lane of `p0` and `p1`: the first 16 pixels, then the last 16.
    let pidx_lo = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
    let pidx_hi = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);
    let mut i = 0;
    while i + 32 <= n {
        let yy = _mm512_cvtepu8_epi16(ld32(y, i));
        let xb = _mm512_sub_epi16(_mm512_cvtepu8_epi16(ld32(cb, i)), c128);
        let xr = _mm512_sub_epi16(_mm512_cvtepu8_epi16(ld32(cr, i)), c128);
        let rnd = _mm512_srli_epi16::<15>(_mm512_mullo_epi16(xr, c_rr));
        let r = _mm512_add_epi16(
            _mm512_add_epi16(_mm512_mulhi_epi16(xr, c_rr), rnd),
            _mm512_add_epi16(xr, yy),
        );
        let b = _mm512_add_epi16(
            _mm512_add_epi16(_mm512_mulhrs_epi16(xb, c_bb), _mm512_add_epi16(xb, xb)),
            yy,
        );
        let gl = _mm512_madd_epi16(_mm512_unpacklo_epi16(xb, xr), c_g);
        let gh = _mm512_madd_epi16(_mm512_unpackhi_epi16(xb, xr), c_g);
        let g = _mm512_add_epi16(
            _mm512_packs_epi32(
                _mm512_srai_epi32::<16>(_mm512_add_epi32(gl, bias)),
                _mm512_srai_epi32::<16>(_mm512_add_epi32(gh, bias)),
            ),
            _mm512_sub_epi16(yy, xr),
        );
        let rg = _mm512_shuffle_epi8(_mm512_packus_epi16(r, g), zip);
        let ba = _mm512_shuffle_epi8(_mm512_packus_epi16(b, c255), zip);
        let p0 = _mm512_unpacklo_epi16(rg, ba);
        let p1 = _mm512_unpackhi_epi16(rg, ba);
        st64(dst, 4 * i, _mm512_permutex2var_epi64(p0, pidx_lo, p1));
        st64(dst, 4 * i + 64, _mm512_permutex2var_epi64(p0, pidx_hi, p1));
        i += 32;
    }
    if i < n {
        color_row(&mut dst[4 * i..], &y[i..], &cb[i..], &cr[i..]);
    }
    n
}

/// 64 interior columns at `c` of `sample::h2v1_row`: [`h2v1_32`] in 512-bit lanes, the 128-bit
/// lanes restored to column order by two `permutex2var`. Reads `row[c - 1..c + 65]`, writes
/// `dst[2 * c..2 * c + 128]`.
#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw")]
fn h2v1_64(dst: &mut [u8], row: &[u8], c: usize) {
    let ones = _mm512_set1_epi8(-1);
    let a = ld64(row, c - 1);
    let b = ld64(row, c);
    let n = ld64(row, c + 1);
    let nb = _mm512_xor_si512(b, ones);
    let t = _mm512_xor_si512(_mm512_avg_epu8(b, a), ones);
    let left = _mm512_xor_si512(_mm512_avg_epu8(nb, t), ones);
    let f = _mm512_xor_si512(_mm512_avg_epu8(nb, _mm512_xor_si512(n, ones)), ones);
    let right = _mm512_avg_epu8(b, f);
    let lo = _mm512_unpacklo_epi8(left, right);
    let hi = _mm512_unpackhi_epi8(left, right);
    let pidx_lo = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
    let pidx_hi = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);
    st64(dst, 2 * c, _mm512_permutex2var_epi64(lo, pidx_lo, hi));
    st64(dst, 2 * c + 64, _mm512_permutex2var_epi64(lo, pidx_hi, hi));
}

/// [`h2v1_mid`] with 64 columns a step when 64 or more are asked for (the last block redone
/// ending at `end`), else the AVX2 kernel.
#[target_feature(enable = "avx2,avx512f,avx512bw")]
fn h2v1_mid_512(dst: &mut [u8], row: &[u8], start: usize, end: usize) -> usize {
    if end.saturating_sub(start) < 64 {
        return h2v1_mid(dst, row, start, end);
    }
    let mut c = start;
    while c + 64 <= end {
        h2v1_64(dst, row, c);
        c += 64;
    }
    if c < end {
        h2v1_64(dst, row, end - 64);
    }
    end
}

/// `3 * cur + other` over the 32 columns at `at`, as u16 lanes.
#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw")]
fn colsum512(cur: &[u8], other: &[u8], at: usize) -> Z {
    let a = _mm512_cvtepu8_epi16(ld32(cur, at));
    let b = _mm512_cvtepu8_epi16(ld32(other, at));
    _mm512_add_epi16(_mm512_mullo_epi16(a, _mm512_set1_epi16(3)), b)
}

/// [`h2v2_mid`] with 32 columns a step when 32 or more are asked for: the colsums before, at and
/// after the columns come from three loads, so no state is carried, and the last block is redone
/// ending at `end`; else the AVX2 kernel.
#[target_feature(enable = "avx2,avx512f,avx512bw")]
fn h2v2_mid_512(dst: &mut [u8], cur: &[u8], other: &[u8], start: usize, end: usize) -> usize {
    if end.saturating_sub(start) < 32 {
        return h2v2_mid(dst, cur, other, start, end);
    }
    let (seven, eight, three) = (
        _mm512_set1_epi16(7),
        _mm512_set1_epi16(8),
        _mm512_set1_epi16(3),
    );
    let mut c = start;
    loop {
        let before = colsum512(cur, other, c - 1);
        let this = colsum512(cur, other, c);
        let next = colsum512(cur, other, c + 1);
        let t = _mm512_mullo_epi16(this, three);
        let even = _mm512_srli_epi16::<4>(_mm512_add_epi16(_mm512_add_epi16(t, before), eight));
        let odd = _mm512_srli_epi16::<4>(_mm512_add_epi16(_mm512_add_epi16(t, next), seven));
        st64(
            dst,
            2 * c,
            _mm512_or_si512(even, _mm512_slli_epi16::<8>(odd)),
        );
        if c + 32 >= end {
            return end;
        }
        c = (c + 32).min(end - 32);
    }
}

/// The value bits and bit length of 16 sign-extended coefficients in 32-bit lanes, as [`prep32`].
#[inline]
#[target_feature(enable = "avx512f")]
fn prep512(x: Z) -> (Z, Z) {
    let one = _mm512_set1_epi32(1);
    let e = _mm512_srli_epi32::<23>(_mm512_castps_si512(_mm512_cvtepi32_ps(_mm512_abs_epi32(x))));
    let nb = _mm512_sub_epi32(e, _mm512_set1_epi32(126));
    // A shift count over 31 gives 0, so a zero coefficient gets an all-ones mask and value 0.
    let mask = _mm512_sub_epi32(_mm512_sllv_epi32(one, nb), one);
    let vals = _mm512_and_si512(_mm512_add_epi32(x, _mm512_srai_epi32::<31>(x)), mask);
    (vals, nb)
}

/// `encoder::write_block_scalar` with the prep in zigzag order by AVX-512: two `permutex2var`
/// put the 64 coefficients in the order the coder walks, a word mask gives the nonzero ones, and
/// 16 coefficients at a time give the value bits and lengths, narrowed to words. The coding loop
/// then reads them by position, as Go's `writeBlockAsm` does with its inline AVX-512 prep.
#[target_feature(enable = "avx2,avx512f,avx512bw,bmi1,bmi2,lzcnt,popcnt")]
fn write_block_512(
    bw: &mut BitWriter,
    blk: &[i16; 64],
    last_dc: &mut i32,
    dct: &[HuffCode; 256],
    act: &[HuffCode; 256],
) {
    let c0 = ld_i16x32(blk, 0);
    let c1 = ld_i16x32(blk, 32);
    // Selectors 0..32 pick from the first register, 32..64 from the second.
    let z0 = _mm512_permutex2var_epi16(c0, ld_i16x32(&ZZ_PERM[0], 0), c1);
    let z1 = _mm512_permutex2var_epi16(c0, ld_i16x32(&ZZ_PERM[1], 0), c1);
    let zz =
        u64::from(_mm512_test_epi16_mask(z0, z0)) | u64::from(_mm512_test_epi16_mask(z1, z1)) << 32;

    let (mut vals, mut nb) = ([0u16; 64], [0u16; 64]);
    for (half, z) in [z0, z1].into_iter().enumerate() {
        let lo32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(z));
        let hi32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(z));
        let (vl, nl) = prep512(lo32);
        let (vh, nh) = prep512(hi32);
        // The values are under 2^16 and the lengths at most 16 (negative for 0): the truncating
        // narrow is exact, and the length of 0 is not used.
        st_u16x16(&mut vals, 32 * half, _mm512_cvtepi32_epi16(vl));
        st_u16x16(&mut vals, 32 * half + 16, _mm512_cvtepi32_epi16(vh));
        st_u16x16(&mut nb, 32 * half, _mm512_cvtepi32_epi16(nl));
        st_u16x16(&mut nb, 32 * half + 16, _mm512_cvtepi32_epi16(nh));
    }
    code_block::<false>(bw, blk[0], last_dc, dct, act, zz & !1, &vals, &nb);
}

#[inline]
#[target_feature(enable = "avx512f")]
fn ld_i16x32(s: &[i16], at: usize) -> Z {
    mem::ld_i16x32(arr(s, at))
}
