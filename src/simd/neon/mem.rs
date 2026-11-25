//! The loads and stores of the NEON kernels, on array references: the array type is the proof that
//! the access is in bounds (`vld1` and `vst1` need no alignment), so the `unsafe` in a helper is
//! only the intrinsic. With `dispatch`, the only `unsafe` in the NEON tier.
use std::arch::aarch64::*;

#[inline]
#[target_feature(enable = "neon")]
pub(super) fn ld8(a: &[u8; 8]) -> uint8x8_t {
    // SAFETY: `a` is 8 readable bytes; vld1 reads exactly 8.
    unsafe { vld1_u8(a.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
pub(super) fn ld16(a: &[u8; 16]) -> uint8x16_t {
    // SAFETY: `a` is 16 readable bytes; vld1q reads exactly 16.
    unsafe { vld1q_u8(a.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
pub(super) fn ld4x8(a: &[u8; 32]) -> uint8x8x4_t {
    // SAFETY: `a` is 32 readable bytes; vld4 reads exactly 8 pixels of 4 bytes.
    unsafe { vld4_u8(a.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
pub(super) fn ld_i16x8(a: &[i16; 8]) -> int16x8_t {
    // SAFETY: `a` is 8 readable i16; vld1q reads exactly those.
    unsafe { vld1q_s16(a.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
pub(super) fn ld_u16x8(a: &[u16; 8]) -> uint16x8_t {
    // SAFETY: `a` is 8 readable u16; vld1q reads exactly those.
    unsafe { vld1q_u16(a.as_ptr()) }
}

/// The two halves of 8 `i32`.
#[inline]
#[target_feature(enable = "neon")]
pub(super) fn ld_i32x8(a: &[i32; 8]) -> (int32x4_t, int32x4_t) {
    let (lo, hi) = a.split_at(4);
    // SAFETY: `lo` and `hi` are 4 readable i32 each; vld1q reads exactly those.
    unsafe { (vld1q_s32(lo.as_ptr()), vld1q_s32(hi.as_ptr())) }
}

#[inline]
#[target_feature(enable = "neon")]
pub(super) fn st8(a: &mut [u8; 8], v: uint8x8_t) {
    // SAFETY: `a` is 8 writable bytes; vst1 writes exactly 8.
    unsafe { vst1_u8(a.as_mut_ptr(), v) }
}

#[inline]
#[target_feature(enable = "neon")]
pub(super) fn st16(a: &mut [u8; 16], v: uint8x16_t) {
    // SAFETY: `a` is 16 writable bytes; vst1q writes exactly 16.
    unsafe { vst1q_u8(a.as_mut_ptr(), v) }
}

#[inline]
#[target_feature(enable = "neon")]
pub(super) fn st_i16x8(a: &mut [i16; 8], v: int16x8_t) {
    // SAFETY: `a` is 8 writable i16; vst1q writes exactly those.
    unsafe { vst1q_s16(a.as_mut_ptr(), v) }
}
