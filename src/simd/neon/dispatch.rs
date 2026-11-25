//! The calls of the NEON kernels, the other place in the NEON tier with `unsafe`: one safe function
//! per kernel, which takes the [`Detected`] that `hardware_tier` builds only after
//! `is_aarch64_feature_detected!("neon")` said so. (NEON is also in the aarch64 baseline, so the
//! compiler may let the kernels be called without `unsafe`; the `unused_unsafe` allow on this
//! module covers that.) The kernels reach memory through bounds-checked helpers, so a short slice
//! panics and never reads out of bounds.
use crate::encoder::Quant;
use crate::simd::Detected;

#[inline]
pub(in crate::simd) fn idct8(
    _: Detected,
    coef: &[i16; 64],
    quant: &[u16; 64],
    dst: &mut [u8],
    stride: usize,
) -> bool {
    // SAFETY: `Detected` is built only by `hardware_tier` after detecting NEON, the kernel's feature.
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
    // SAFETY: `Detected` is built only by `hardware_tier` after detecting NEON, the kernel's feature.
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
    // SAFETY: `Detected` is built only by `hardware_tier` after detecting NEON, the kernel's feature.
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
    // SAFETY: `Detected` is built only by `hardware_tier` after detecting NEON, the kernel's feature.
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
    // SAFETY: `Detected` is built only by `hardware_tier` after detecting NEON, the kernel's feature.
    unsafe { super::h2v2_mid(dst, cur, other, start, end) }
}

pub(in crate::simd) fn fdct_quant_rows(
    _: Detected,
    pix: &[u8],
    off: usize,
    stride: usize,
    quant: &Quant,
    coef: &mut [i16; 64],
) {
    // SAFETY: `Detected` is built only by `hardware_tier` after detecting NEON, the kernel's feature.
    unsafe { super::fdct_quant_rows(pix, off, stride, quant, coef) }
}

pub(in crate::simd) fn rgba_to_ycbcr_row(
    _: Detected,
    src: &[u8],
    y: &mut [u8],
    cb: &mut [u8],
    cr: &mut [u8],
) -> usize {
    // SAFETY: `Detected` is built only by `hardware_tier` after detecting NEON, the kernel's feature.
    unsafe { super::rgba_to_ycbcr_row(src, y, cb, cr) }
}

pub(in crate::simd) fn downsample_h2v2_row(
    _: Detected,
    dst: &mut [u8],
    r0: &[u8],
    r1: &[u8],
) -> usize {
    // SAFETY: `Detected` is built only by `hardware_tier` after detecting NEON, the kernel's feature.
    unsafe { super::downsample_h2v2_row(dst, r0, r1) }
}
