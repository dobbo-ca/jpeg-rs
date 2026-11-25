//! The calls of the AVX2 kernels, the other place in the x86 tier with `unsafe`. A safe
//! `#[target_feature]` function is unsafe to call from code without its features, so each kernel
//! has one safe function here that makes the call. It takes the proof that the features were
//! detected ([`Detected`] for AVX2, [`Bmi`], [`Avx512`]), which `simd` builds only after
//! `is_x86_feature_detected!` said so. The kernels reach memory through bounds-checked helpers, so
//! a short slice panics and never reads out of bounds; the lengths are no part of these proofs.
use crate::dct::Block;
use crate::encoder::{BitWriter, HuffCode, Quant};
use crate::huff::{BitReader, DerivedTable};
use crate::simd::{Avx512, Bmi, Detected};

/// `x86::idct8`: always true, no block is out of the kernel's range.
#[inline]
pub(in crate::simd) fn idct8(
    _: Detected,
    coef: &[i16; 64],
    quant: &[u16; 64],
    dst: &mut [u8],
    stride: usize,
) -> bool {
    // SAFETY: `Detected` is built only by `hardware_tier` after detecting AVX2, the kernel's feature.
    unsafe { super::idct8(coef, quant, dst, stride) }
}

#[inline]
pub(in crate::simd) fn idct4(
    _: Detected,
    coef: &[i16; 64],
    quant: &[u16; 64],
    dst: &mut [u8],
    stride: usize,
) {
    // SAFETY: `Detected` is built only by `hardware_tier` after detecting AVX2, the kernel's feature.
    unsafe { super::idct4(coef, quant, dst, stride) }
}

#[inline]
pub(in crate::simd) fn color_row(
    _: Detected,
    dst: &mut [u8],
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
) -> usize {
    // SAFETY: `Detected` is built only by `hardware_tier` after detecting AVX2, the kernel's feature.
    unsafe { super::color_row(dst, y, cb, cr) }
}

#[inline]
pub(in crate::simd) fn h2v1_mid(
    _: Detected,
    dst: &mut [u8],
    row: &[u8],
    start: usize,
    end: usize,
) -> usize {
    // SAFETY: `Detected` is built only by `hardware_tier` after detecting AVX2, the kernel's feature.
    unsafe { super::h2v1_mid(dst, row, start, end) }
}

#[inline]
pub(in crate::simd) fn h2v2_mid(
    _: Detected,
    dst: &mut [u8],
    cur: &[u8],
    other: &[u8],
    start: usize,
    end: usize,
) -> usize {
    // SAFETY: `Detected` is built only by `hardware_tier` after detecting AVX2, the kernel's feature.
    unsafe { super::h2v2_mid(dst, cur, other, start, end) }
}

#[inline]
pub(in crate::simd) fn color_row_512(
    _: Detected,
    _: Avx512,
    dst: &mut [u8],
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
) -> usize {
    // SAFETY: `Detected` proves AVX2 (`hardware_tier`) and `Avx512` AVX-512 F and BW (`avx512`
    // after `has_avx512`), the kernel's features.
    unsafe { super::color_row_512(dst, y, cb, cr) }
}

#[inline]
pub(in crate::simd) fn h2v1_mid_512(
    _: Detected,
    _: Avx512,
    dst: &mut [u8],
    row: &[u8],
    start: usize,
    end: usize,
) -> usize {
    // SAFETY: `Detected` proves AVX2 (`hardware_tier`) and `Avx512` AVX-512 F and BW (`avx512`
    // after `has_avx512`), the kernel's features.
    unsafe { super::h2v1_mid_512(dst, row, start, end) }
}

#[inline]
pub(in crate::simd) fn h2v2_mid_512(
    _: Detected,
    _: Avx512,
    dst: &mut [u8],
    cur: &[u8],
    other: &[u8],
    start: usize,
    end: usize,
) -> usize {
    // SAFETY: `Detected` proves AVX2 (`hardware_tier`) and `Avx512` AVX-512 F and BW (`avx512`
    // after `has_avx512`), the kernel's features.
    unsafe { super::h2v2_mid_512(dst, cur, other, start, end) }
}

pub(in crate::simd) fn fdct_quant_rows(
    _: Detected,
    pix: &[u8],
    off: usize,
    stride: usize,
    quant: &Quant,
    coef: &mut [i16; 64],
) {
    // SAFETY: `Detected` is built only by `hardware_tier` after detecting AVX2, the kernel's feature.
    unsafe { super::fdct_quant_rows(pix, off, stride, quant, coef) }
}

pub(in crate::simd) fn rgba_to_ycbcr_row(
    _: Detected,
    src: &[u8],
    y: &mut [u8],
    cb: &mut [u8],
    cr: &mut [u8],
) -> usize {
    // SAFETY: `Detected` is built only by `hardware_tier` after detecting AVX2, the kernel's feature.
    unsafe { super::rgba_to_ycbcr_row(src, y, cb, cr) }
}

pub(in crate::simd) fn rgba_to_ycbcr_row_512(
    _: Detected,
    _: Avx512,
    src: &[u8],
    y: &mut [u8],
    cb: &mut [u8],
    cr: &mut [u8],
) -> usize {
    // SAFETY: `Detected` proves AVX2 (`hardware_tier`) and `Avx512` AVX-512 F and BW (`avx512`
    // after `has_avx512`), the kernel's features.
    unsafe { super::rgba_to_ycbcr_row_512(src, y, cb, cr) }
}

pub(in crate::simd) fn downsample_h2v2_row(
    _: Detected,
    dst: &mut [u8],
    r0: &[u8],
    r1: &[u8],
) -> usize {
    // SAFETY: `Detected` is built only by `hardware_tier` after detecting AVX2, the kernel's feature.
    unsafe { super::downsample_h2v2_row(dst, r0, r1) }
}

pub(in crate::simd) fn write_block(
    _: Detected,
    _: Bmi,
    bw: &mut BitWriter,
    blk: &[i16; 64],
    last_dc: &mut i32,
    dct: &[HuffCode; 256],
    act: &[HuffCode; 256],
) {
    // SAFETY: `Detected` proves AVX2 (`hardware_tier`) and `Bmi` BMI1, BMI2, LZCNT and POPCNT
    // (`bmi` after `has_bmi`), the kernel's features.
    unsafe { super::write_block(bw, blk, last_dc, dct, act) }
}

pub(in crate::simd) fn write_block_512(
    _: Detected,
    _: Bmi,
    _: Avx512,
    bw: &mut BitWriter,
    blk: &[i16; 64],
    last_dc: &mut i32,
    dct: &[HuffCode; 256],
    act: &[HuffCode; 256],
) {
    // SAFETY: `Detected` proves AVX2 (`hardware_tier`), `Bmi` BMI1, BMI2, LZCNT and POPCNT and
    // `Avx512` AVX-512 F and BW (`bmi`, `avx512`), the kernel's features.
    unsafe { super::write_block_512(bw, blk, last_dc, dct, act) }
}

#[inline]
pub(in crate::simd) fn decode_mcu_fast<const STORE: bool>(
    _: Bmi,
    r: &mut BitReader,
    dc: &[DerivedTable; 4],
    ac: &[DerivedTable; 4],
    last_dc: &mut [i32; 4],
    blks: &mut [Block],
    ks: &[usize],
    has_ac: Option<&mut [bool]>,
) -> bool {
    // SAFETY: `Bmi` proves BMI1, BMI2 and LZCNT were detected (`has_bmi`), the kernel's features.
    unsafe { super::decode_mcu_fast::<STORE>(r, dc, ac, last_dc, blks, ks, has_ac) }
}

pub(in crate::simd) fn fdct_quant_pair(
    _: Detected,
    _: Avx512,
    a: (&[u8], usize, usize),
    b: (&[u8], usize, usize),
    quant: &Quant,
    ca: &mut [i16; 64],
    cb: &mut [i16; 64],
) {
    // SAFETY: `Detected` proves AVX2 (`hardware_tier`) and `Avx512` AVX-512 F and BW (`avx512`
    // after `has_avx512`), the kernel's features.
    unsafe { super::fdct_quant_pair(a, b, quant, ca, cb) }
}
