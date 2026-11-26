//! SIMD tiers: AVX2 on x86_64, NEON on aarch64, and the scalar code beside them, which is the
//! oracle and the fallback. The tier is detected once; `JPEG_SIMD=scalar` forces the scalar one.
//! Each dispatcher takes the tier so the tests can run every available tier against scalar; the
//! rest of the crate calls the `tier()`-resolved wrappers.
//!
//! Every kernel is bit-exact with its scalar twin on every input, so the output never depends on
//! the tier. That holds for the IDCTs too: their 32-bit lanes are exact mod 2^32, the scalar range
//! limit reads only the low bits of its sum, and a first pass that could pass 2^31 is split into
//! a high and a low half (see `IDCT8_MAX_DQ`). No kernel refuses a block.
//!
//! This module has no `unsafe`. The kernels (`x86.rs`, `neon.rs`) are safe `#[target_feature]`
//! functions; `unsafe` is only in each tier's `mem` (the loads and stores, on array references)
//! and `dispatch` (the one call of each kernel, which takes the proof that its features were
//! detected: [`Detected`], [`Bmi`], [`Avx512`]).
#[macro_use]
mod lanes;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "x86_64")]
mod x86;

#[cfg(test)]
mod tests;

use crate::dct::Block;
use crate::encoder::{BitWriter, HuffCode, Quant};
use crate::huff::{BitReader, DerivedTable};
use std::sync::OnceLock;

/// Largest dequantized coefficient (`|coef * quant|`) for which the 32-bit `idct8` kernels run
/// pass 1 as one pass. A pass-1 sum is a weighted sum of the 8 inputs of a column; the largest sum
/// of absolute weights over any output is 61214 (with the rounding term 1024), so no sum reaches
/// 2^31 while `|dq| <= (2^31 - 1 - 1024) / 61214 = 35081`. Above it pass 1 is split in a high
/// and a low half (`idct_1d_split`), exact for every `i32`. Pass 2 needs no bound: it wraps mod
/// 2^32 and the scalar range limit reads only bits 18..=27 of its sum.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const IDCT8_MAX_DQ: u32 = 35081;

/// `IDCT8_MAX_DQ` for `idct4`: the largest sum of absolute weights is 77910 (rounding term
/// 2048), so the bound is `(2^31 - 1 - 2048) / 77910 = 27563`; pass 1 splits at 12 bits above it.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const IDCT4_MAX_DQ: u32 = 27563;

/// Proof that a feature was detected: only this module (`hardware_tier`) can build one.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Detected(());

/// Proof that BMI1, BMI2, LZCNT and POPCNT were detected ([`has_bmi`]): only [`bmi`] and `decode_mcu_fast` (which checks `bmi_decode`, implying `has_bmi`) build one.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
pub(crate) struct Bmi(());

/// Proof that AVX2 and AVX-512 F and BW were detected ([`has_avx512`]): only [`avx512`] builds one.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
pub(crate) struct Avx512(());

#[cfg(target_arch = "x86_64")]
fn bmi() -> Option<Bmi> {
    has_bmi().then_some(Bmi(()))
}

#[cfg(target_arch = "x86_64")]
fn avx512() -> Option<Avx512> {
    has_avx512().then_some(Avx512(()))
}

/// `s[at..at + N]` as an array reference, which is what the kernels' load and store helpers take
/// and what proves their bounds. A slice too short panics, as indexing does; the dispatchers
/// check the lengths first.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline(always)]
#[allow(clippy::unwrap_used)] // the slice has exactly N elements
fn arr<T, const N: usize>(s: &[T], at: usize) -> &[T; N] {
    <&[T; N]>::try_from(&s[at..at + N]).unwrap()
}

/// [`arr`] for a store.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline(always)]
#[allow(clippy::unwrap_used)] // the slice has exactly N elements
fn arr_mut<T, const N: usize>(s: &mut [T], at: usize) -> &mut [T; N] {
    <&mut [T; N]>::try_from(&mut s[at..at + N]).unwrap()
}

/// An instruction-set tier. A tier other than `Scalar` carries a [`Detected`], so it exists only
/// after its feature was detected, which is what makes the `unsafe` kernel calls in `dispatch` sound.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Tier {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2(Detected),
    #[cfg(target_arch = "aarch64")]
    Neon(Detected),
}

/// The best tier this CPU has, ignoring `JPEG_SIMD`.
pub(crate) fn hardware_tier() -> Tier {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return Tier::Avx2(Detected(()));
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return Tier::Neon(Detected(()));
        }
    }
    Tier::Scalar
}

/// The tier in use: the hardware's, or `Scalar` under `JPEG_SIMD=scalar`. Resolved once.
/// `JPEG_SIMD=avx512` requires the AVX-512 kernels (for CI under an emulator): it panics on a CPU
/// without them, so a test run cannot fall back to AVX2 without a word.
pub(crate) fn tier() -> Tier {
    static TIER: OnceLock<Tier> = OnceLock::new();
    *TIER.get_or_init(|| match std::env::var("JPEG_SIMD") {
        Ok(v) if v == "scalar" => Tier::Scalar,
        Ok(v) if v == "avx512" => {
            assert!(
                avx512_kernels(),
                "JPEG_SIMD=avx512, but this CPU lacks AVX-512 F and BW"
            );
            hardware_tier()
        }
        _ => hardware_tier(),
    })
}

/// Whether the AVX-512 kernels run here: false on a target without them.
fn avx512_kernels() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        has_avx512()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// The SIMD tier this process uses: `"avx2"`, `"neon"` or `"scalar"`. The kernels give the same
/// output in every tier; set the environment variable `JPEG_SIMD=scalar` to force the scalar one.
/// Infallible, takes no limits and spawns no thread.
pub fn simd_tier() -> &'static str {
    match tier() {
        Tier::Scalar => "scalar",
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2(_) => "avx2",
        #[cfg(target_arch = "aarch64")]
        Tier::Neon(_) => "neon",
    }
}

/// The islow IDCT of a block with AC coefficients, `dst` an 8x8 block at `stride`.
#[inline]
pub(crate) fn idct8(coef: &[i16; 64], quant: &[u16; 64], dst: &mut [u8], stride: usize) {
    idct8_with(tier(), coef, quant, dst, stride)
}

#[inline]
pub(crate) fn idct8_with(
    t: Tier,
    coef: &[i16; 64],
    quant: &[u16; 64],
    dst: &mut [u8],
    stride: usize,
) {
    if !idct8_try(t, coef, quant, dst, stride) {
        crate::dct::idct8(coef, quant, dst, stride);
    }
}

/// The SIMD IDCT alone: false, with `dst` untouched, when `t` is scalar or `dst` is short.
#[inline]
pub(crate) fn idct8_try(
    t: Tier,
    coef: &[i16; 64],
    quant: &[u16; 64],
    dst: &mut [u8],
    stride: usize,
) -> bool {
    let fits = stride
        .checked_mul(7)
        .and_then(|v| v.checked_add(8))
        .is_some_and(|need| need <= dst.len());
    if !fits {
        return false;
    }
    match t {
        Tier::Scalar => false,
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2(d) => x86::dispatch::idct8(d, coef, quant, dst, stride),
        #[cfg(target_arch = "aarch64")]
        Tier::Neon(d) => neon::dispatch::idct8(d, coef, quant, dst, stride),
    }
}

/// The reduced 4x4 IDCT of a block with AC coefficients, `dst` a 4x4 block at `stride`.
#[inline]
pub(crate) fn idct4(coef: &[i16; 64], quant: &[u16; 64], dst: &mut [u8], stride: usize) {
    idct4_with(tier(), coef, quant, dst, stride)
}

#[inline]
pub(crate) fn idct4_with(
    t: Tier,
    coef: &[i16; 64],
    quant: &[u16; 64],
    dst: &mut [u8],
    stride: usize,
) {
    if !idct4_try(t, coef, quant, dst, stride) {
        crate::dct::idct4(coef, quant, dst, stride);
    }
}

/// `idct8_try` for the reduced 4x4 IDCT; `dst` must hold `3 * stride + 4` bytes.
#[inline]
pub(crate) fn idct4_try(
    t: Tier,
    coef: &[i16; 64],
    quant: &[u16; 64],
    dst: &mut [u8],
    stride: usize,
) -> bool {
    let fits = stride
        .checked_mul(3)
        .and_then(|v| v.checked_add(4))
        .is_some_and(|need| need <= dst.len());
    if !fits {
        return false;
    }
    match t {
        Tier::Scalar => false,
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2(d) => {
            x86::dispatch::idct4(d, coef, quant, dst, stride);
            true
        }
        #[cfg(target_arch = "aarch64")]
        Tier::Neon(d) => {
            neon::dispatch::idct4(d, coef, quant, dst, stride);
            true
        }
    }
}

/// The 4-byte-per-pixel `ycc_rgb_convert` kernel over the first pixels of the row: returns how
/// many it converted (a multiple of its block; 0 for scalar), the caller does the rest.
#[inline]
pub(crate) fn color_row(t: Tier, dst: &mut [u8], y: &[u8], cb: &[u8], cr: &[u8]) -> usize {
    let n = y.len();
    if cb.len() < n || cr.len() < n || dst.len() / 4 < n {
        return 0;
    }
    match t {
        Tier::Scalar => 0,
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2(d) => match avx512() {
            Some(w) => x86::dispatch::color_row_512(d, w, dst, y, cb, cr),
            None => x86::dispatch::color_row(d, dst, y, cb, cr),
        },
        #[cfg(target_arch = "aarch64")]
        Tier::Neon(d) => neon::dispatch::color_row(d, dst, y, cb, cr),
    }
}

/// The h2v1 fancy upsample of interior columns `start..end` (each has both neighbours): returns
/// the first column left for the scalar loop. `row` holds `end + 1` samples, `dst` `2 * end`.
#[inline]
pub(crate) fn h2v1_mid(t: Tier, dst: &mut [u8], row: &[u8], start: usize, end: usize) -> usize {
    if start == 0 || row.len() <= end || dst.len() < 2 * end {
        return start;
    }
    match t {
        Tier::Scalar => start,
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2(d) => match avx512() {
            Some(w) => x86::dispatch::h2v1_mid_512(d, w, dst, row, start, end),
            None => x86::dispatch::h2v1_mid(d, dst, row, start, end),
        },
        #[cfg(target_arch = "aarch64")]
        Tier::Neon(d) => neon::dispatch::h2v1_mid(d, dst, row, start, end),
    }
}

/// As [`h2v1_mid`] for the h2v2 fancy upsample; `cur` and `other` hold `end + 1` samples.
#[inline]
pub(crate) fn h2v2_mid(
    t: Tier,
    dst: &mut [u8],
    cur: &[u8],
    other: &[u8],
    start: usize,
    end: usize,
) -> usize {
    if start == 0 || cur.len() <= end || other.len() <= end || dst.len() < 2 * end {
        return start;
    }
    match t {
        Tier::Scalar => start,
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2(d) => match avx512() {
            Some(w) => x86::dispatch::h2v2_mid_512(d, w, dst, cur, other, start, end),
            None => x86::dispatch::h2v2_mid(d, dst, cur, other, start, end),
        },
        #[cfg(target_arch = "aarch64")]
        Tier::Neon(d) => neon::dispatch::h2v2_mid(d, dst, cur, other, start, end),
    }
}

/// `encoder::fdct_quant_rows`, SIMD when `t` has it.
pub(crate) fn fdct_quant_rows(
    t: Tier,
    pix: &[u8],
    off: usize,
    stride: usize,
    quant: &Quant,
    coef: &mut [i16; 64],
) {
    let fits = stride
        .checked_mul(7)
        .and_then(|v| v.checked_add(off))
        .and_then(|v| v.checked_add(8))
        .is_some_and(|need| need <= pix.len());
    if fits {
        match t {
            Tier::Scalar => {}
            #[cfg(target_arch = "x86_64")]
            Tier::Avx2(d) => {
                x86::dispatch::fdct_quant_rows(d, pix, off, stride, quant, coef);
                return;
            }
            #[cfg(target_arch = "aarch64")]
            Tier::Neon(d) => {
                neon::dispatch::fdct_quant_rows(d, pix, off, stride, quant, coef);
                return;
            }
        }
    }
    crate::encoder::fdct_quant_rows(pix, off, stride, quant, coef);
}

/// The first pixels of `encoder::rgba_to_ycbcr_row` in SIMD: returns how many, the caller does
/// the rest.
pub(crate) fn rgba_to_ycbcr_row(
    t: Tier,
    src: &[u8],
    y: &mut [u8],
    cb: &mut [u8],
    cr: &mut [u8],
) -> usize {
    let n = y.len();
    if src.len() / 4 < n || cb.len() < n || cr.len() < n {
        return 0;
    }
    match t {
        Tier::Scalar => 0,
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2(d) => match avx512() {
            Some(w) => x86::dispatch::rgba_to_ycbcr_row_512(d, w, src, y, cb, cr),
            None => x86::dispatch::rgba_to_ycbcr_row(d, src, y, cb, cr),
        },
        #[cfg(target_arch = "aarch64")]
        Tier::Neon(d) => neon::dispatch::rgba_to_ycbcr_row(d, src, y, cb, cr),
    }
}

/// The first outputs of `encoder::downsample_h2v2_row` in SIMD: returns how many.
pub(crate) fn downsample_h2v2_row(t: Tier, dst: &mut [u8], r0: &[u8], r1: &[u8]) -> usize {
    if r0.len() / 2 < dst.len() || r1.len() / 2 < dst.len() {
        return 0;
    }
    match t {
        Tier::Scalar => 0,
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2(d) => x86::dispatch::downsample_h2v2_row(d, dst, r0, r1),
        #[cfg(target_arch = "aarch64")]
        Tier::Neon(d) => neon::dispatch::downsample_h2v2_row(d, dst, r0, r1),
    }
}

/// Whether this CPU has what the BMI builds (`x86::write_block`, `x86::decode_mcu_fast`) need besides
/// AVX2.
#[cfg(target_arch = "x86_64")]
fn has_bmi() -> bool {
    static B: OnceLock<bool> = OnceLock::new();
    *B.get_or_init(|| {
        is_x86_feature_detected!("bmi1")
            && is_x86_feature_detected!("bmi2")
            && is_x86_feature_detected!("lzcnt")
            && is_x86_feature_detected!("popcnt")
    })
}

/// `rgba_to_ycbcr_row` through one x86 kernel on its own, for the differential checks: the AVX2
/// one, or with `wide` the AVX-512 one (the dispatcher runs only the wider where there is one).
/// Returns how many pixels it did, or `None` when this CPU lacks the kernel or a slice is short.
#[cfg(target_arch = "x86_64")]
pub(crate) fn rgba_to_ycbcr_row_x86(
    wide: bool,
    src: &[u8],
    y: &mut [u8],
    cb: &mut [u8],
    cr: &mut [u8],
) -> Option<usize> {
    let n = y.len();
    let Tier::Avx2(d) = hardware_tier() else {
        return None;
    };
    if src.len() / 4 < n || cb.len() < n || cr.len() < n {
        return None;
    }
    if wide {
        let w = avx512()?;
        return Some(x86::dispatch::rgba_to_ycbcr_row_512(d, w, src, y, cb, cr));
    }
    Some(x86::dispatch::rgba_to_ycbcr_row(d, src, y, cb, cr))
}

/// [`color_row`] through one x86 kernel on its own, as [`rgba_to_ycbcr_row_x86`]: the AVX2 one, or
/// with `wide` the AVX-512 one. Returns how many pixels it did, or `None` when this CPU lacks the
/// kernel or a slice is short.
#[cfg(target_arch = "x86_64")]
pub(crate) fn color_row_x86(
    wide: bool,
    dst: &mut [u8],
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
) -> Option<usize> {
    let n = y.len();
    let Tier::Avx2(d) = hardware_tier() else {
        return None;
    };
    if cb.len() < n || cr.len() < n || dst.len() / 4 < n {
        return None;
    }
    if wide {
        let w = avx512()?;
        return Some(x86::dispatch::color_row_512(d, w, dst, y, cb, cr));
    }
    Some(x86::dispatch::color_row(d, dst, y, cb, cr))
}

/// [`h2v1_mid`] through one x86 kernel on its own, as [`color_row_x86`]: `None` also when `start`
/// is 0 or a slice is short.
#[cfg(target_arch = "x86_64")]
pub(crate) fn h2v1_mid_x86(
    wide: bool,
    dst: &mut [u8],
    row: &[u8],
    start: usize,
    end: usize,
) -> Option<usize> {
    let Tier::Avx2(d) = hardware_tier() else {
        return None;
    };
    if start == 0 || row.len() <= end || dst.len() < 2 * end {
        return None;
    }
    if wide {
        let w = avx512()?;
        return Some(x86::dispatch::h2v1_mid_512(d, w, dst, row, start, end));
    }
    Some(x86::dispatch::h2v1_mid(d, dst, row, start, end))
}

/// [`h2v2_mid`] through one x86 kernel on its own, as [`h2v1_mid_x86`].
#[cfg(target_arch = "x86_64")]
pub(crate) fn h2v2_mid_x86(
    wide: bool,
    dst: &mut [u8],
    cur: &[u8],
    other: &[u8],
    start: usize,
    end: usize,
) -> Option<usize> {
    let Tier::Avx2(d) = hardware_tier() else {
        return None;
    };
    if start == 0 || cur.len() <= end || other.len() <= end || dst.len() < 2 * end {
        return None;
    }
    if wide {
        let w = avx512()?;
        return Some(x86::dispatch::h2v2_mid_512(
            d, w, dst, cur, other, start, end,
        ));
    }
    Some(x86::dispatch::h2v2_mid(d, dst, cur, other, start, end))
}

/// `write_block` through one x86 kernel on its own, as [`rgba_to_ycbcr_row_x86`]: false, having
/// written nothing, when this CPU lacks the kernel.
#[cfg(target_arch = "x86_64")]
pub(crate) fn write_block_x86(
    wide: bool,
    bw: &mut BitWriter,
    blk: &[i16; 64],
    last_dc: &mut i32,
    dct: &[HuffCode; 256],
    act: &[HuffCode; 256],
) -> bool {
    let (Tier::Avx2(d), Some(b)) = (hardware_tier(), bmi()) else {
        return false;
    };
    if wide {
        let Some(w) = avx512() else {
            return false;
        };
        x86::dispatch::write_block_512(d, b, w, bw, blk, last_dc, dct, act);
    } else {
        x86::dispatch::write_block(d, b, bw, blk, last_dc, dct, act);
    }
    true
}

/// Whether this CPU runs the AVX-512 encoder kernels beside AVX2.
#[cfg(target_arch = "x86_64")]
pub(crate) fn has_wide() -> bool {
    has_avx512()
}

/// `encoder::write_block_scalar` in SIMD: false, having written nothing, when `t` has no kernel
/// (scalar, NEON) or this CPU lacks BMI; the caller then runs the scalar code.
#[cfg_attr(not(target_arch = "x86_64"), allow(unused_variables))]
pub(crate) fn write_block(
    t: Tier,
    bw: &mut BitWriter,
    blk: &[i16; 64],
    last_dc: &mut i32,
    dct: &[HuffCode; 256],
    act: &[HuffCode; 256],
) -> bool {
    match t {
        Tier::Scalar => false,
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2(d) => {
            let Some(b) = bmi() else {
                return false;
            };
            match avx512() {
                Some(w) => x86::dispatch::write_block_512(d, b, w, bw, blk, last_dc, dct, act),
                None => x86::dispatch::write_block(d, b, bw, blk, last_dc, dct, act),
            }
            true
        }
        #[cfg(target_arch = "aarch64")]
        Tier::Neon(_) => false,
    }
}

/// Whether the decoder's MCU loop runs in its BMI build: this tier is AVX2 and the CPU has BMI1,
/// BMI2 and LZCNT. Resolved once.
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn bmi_decode() -> bool {
    static B: OnceLock<bool> = OnceLock::new();
    *B.get_or_init(|| matches!(tier(), Tier::Avx2(_)) && has_bmi())
}

/// [`decode_mcu_fast`] through the BMI build on its own, whatever `JPEG_SIMD` says, for the
/// differential checks: `None`, having read nothing, when this CPU lacks BMI.
#[cfg(target_arch = "x86_64")]
pub(crate) fn decode_mcu_fast_x86<const STORE: bool>(
    r: &mut BitReader,
    dc: &[DerivedTable; 4],
    ac: &[DerivedTable; 4],
    last_dc: &mut [i32; 4],
    blks: &mut [Block],
    ks: &[usize],
    has_ac: Option<&mut [bool]>,
) -> Option<bool> {
    let b = bmi()?;
    Some(x86::dispatch::decode_mcu_fast::<STORE>(
        b, r, dc, ac, last_dc, blks, ks, has_ac,
    ))
}

/// `scan::decode_mcu_fast_impl`, built with BMI1, BMI2 and LZCNT (`shlx`, `shrx`, `tzcnt`) when
/// [`bmi_decode`].
#[inline]
pub(crate) fn decode_mcu_fast<const STORE: bool>(
    r: &mut BitReader,
    dc: &[DerivedTable; 4],
    ac: &[DerivedTable; 4],
    last_dc: &mut [i32; 4],
    blks: &mut [Block],
    ks: &[usize],
    has_ac: Option<&mut [bool]>,
) -> bool {
    #[cfg(target_arch = "x86_64")]
    if bmi_decode() {
        // SAFETY (proof token): bmi_decode() implies has_bmi(), which is what `Bmi` proves.
        return x86::dispatch::decode_mcu_fast::<STORE>(
            Bmi(()),
            r,
            dc,
            ac,
            last_dc,
            blks,
            ks,
            has_ac,
        );
    }
    crate::scan::decode_mcu_fast_impl::<STORE>(r, dc, ac, last_dc, blks, ks, has_ac)
}

/// Whether this CPU has the AVX-512 subset the wide kernels use.
#[cfg(target_arch = "x86_64")]
fn has_avx512() -> bool {
    static B: OnceLock<bool> = OnceLock::new();
    *B.get_or_init(|| {
        is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
    })
}

/// `encoder::fdct_quant_rows` of two blocks that share `quant`, `a` and `b` each `(pix, offset,
/// stride)`, in one AVX-512 kernel: false, having done nothing, when `t` has none, this CPU lacks
/// AVX-512 or a block is out of the slice.
#[cfg_attr(not(target_arch = "x86_64"), allow(unused_variables))]
pub(crate) fn fdct_quant_pair(
    t: Tier,
    a: (&[u8], usize, usize),
    b: (&[u8], usize, usize),
    quant: &Quant,
    ca: &mut [i16; 64],
    cb: &mut [i16; 64],
) -> bool {
    let fits = |(pix, off, stride): (&[u8], usize, usize)| {
        stride
            .checked_mul(7)
            .and_then(|v| v.checked_add(off))
            .and_then(|v| v.checked_add(8))
            .is_some_and(|need| need <= pix.len())
    };
    if !fits(a) || !fits(b) {
        return false;
    }
    match t {
        Tier::Scalar => false,
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2(d) => {
            let Some(w) = avx512() else {
                return false;
            };
            x86::dispatch::fdct_quant_pair(d, w, a, b, quant, ca, cb);
            true
        }
        #[cfg(target_arch = "aarch64")]
        Tier::Neon(_) => false,
    }
}
