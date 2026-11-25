//! The loads and stores of the AVX2 and AVX-512 kernels, on array references: the array type is the
//! proof that the access is in bounds, and each access is unaligned, so the `unsafe` in a helper
//! is only the intrinsic. With `dispatch`, the only `unsafe` in the x86 tier.
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2")]
pub(super) fn ld8(a: &[u8; 8]) -> __m128i {
    // SAFETY: `a` is 8 readable bytes; the load is unaligned and reads exactly 8.
    unsafe { _mm_loadl_epi64(a.as_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "avx2")]
pub(super) fn ld16(a: &[u8; 16]) -> __m128i {
    // SAFETY: `a` is 16 readable bytes; the load is unaligned and reads exactly 16.
    unsafe { _mm_loadu_si128(a.as_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "avx2")]
pub(super) fn ld32(a: &[u8; 32]) -> __m256i {
    // SAFETY: `a` is 32 readable bytes; the load is unaligned and reads exactly 32.
    unsafe { _mm256_loadu_si256(a.as_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "avx2")]
pub(super) fn ld_i16x8(a: &[i16; 8]) -> __m128i {
    // SAFETY: `a` is 8 readable i16 (16 bytes); the load is unaligned.
    unsafe { _mm_loadu_si128(a.as_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "avx2")]
pub(super) fn ld_u16x8(a: &[u16; 8]) -> __m128i {
    // SAFETY: `a` is 8 readable u16 (16 bytes); the load is unaligned.
    unsafe { _mm_loadu_si128(a.as_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "avx2")]
pub(super) fn ld_u16x16(a: &[u16; 16]) -> __m256i {
    // SAFETY: `a` is 16 readable u16 (32 bytes); the load is unaligned.
    unsafe { _mm256_loadu_si256(a.as_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "avx2")]
pub(super) fn ld_i16x16(a: &[i16; 16]) -> __m256i {
    // SAFETY: `a` is 16 readable i16 (32 bytes); the load is unaligned.
    unsafe { _mm256_loadu_si256(a.as_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "avx512f")]
pub(super) fn ld64(a: &[u8; 64]) -> __m512i {
    // SAFETY: `a` is 64 readable bytes; the load is unaligned and reads exactly 64.
    unsafe { _mm512_loadu_si512(a.as_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "avx512f")]
pub(super) fn st64(a: &mut [u8; 64], v: __m512i) {
    // SAFETY: `a` is 64 writable bytes; the store is unaligned and writes exactly 64.
    unsafe { _mm512_storeu_si512(a.as_mut_ptr().cast(), v) }
}

#[inline]
#[target_feature(enable = "avx512f")]
pub(super) fn ld_u16x32(a: &[u16; 32]) -> __m512i {
    // SAFETY: `a` is 32 readable u16 (64 bytes); the load is unaligned.
    unsafe { _mm512_loadu_si512(a.as_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "avx512f")]
pub(super) fn ld_i16x32(a: &[i16; 32]) -> __m512i {
    // SAFETY: `a` is 32 readable i16 (64 bytes); the load is unaligned.
    unsafe { _mm512_loadu_si512(a.as_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "avx2")]
pub(super) fn st8(a: &mut [u8; 8], v: __m128i) {
    // SAFETY: `a` is 8 writable bytes; the store is unaligned and writes exactly 8.
    unsafe { _mm_storel_epi64(a.as_mut_ptr().cast(), v) }
}

/// The high 8 bytes of `v` into `a`.
#[inline]
#[target_feature(enable = "avx2")]
pub(super) fn sth8(a: &mut [u8; 8], v: __m128i) {
    // SAFETY: `a` is 8 writable bytes and `_mm_storel_epi64` copies them bytewise (no alignment
    // needed; `_mm_storeh_pd` is a typed `f64` write, which would need 8-byte alignment).
    unsafe { _mm_storel_epi64(a.as_mut_ptr().cast(), _mm_unpackhi_epi64(v, v)) }
}

#[inline]
#[target_feature(enable = "avx2")]
pub(super) fn st16(a: &mut [u8; 16], v: __m128i) {
    // SAFETY: `a` is 16 writable bytes; the store is unaligned and writes exactly 16.
    unsafe { _mm_storeu_si128(a.as_mut_ptr().cast(), v) }
}

#[inline]
#[target_feature(enable = "avx2")]
pub(super) fn st32(a: &mut [u8; 32], v: __m256i) {
    // SAFETY: `a` is 32 writable bytes; the store is unaligned and writes exactly 32.
    unsafe { _mm256_storeu_si256(a.as_mut_ptr().cast(), v) }
}

#[inline]
#[target_feature(enable = "avx2")]
pub(super) fn st_i16x16(a: &mut [i16; 16], v: __m256i) {
    // SAFETY: `a` is 16 writable i16 (32 bytes); the store is unaligned.
    unsafe { _mm256_storeu_si256(a.as_mut_ptr().cast(), v) }
}

#[inline]
#[target_feature(enable = "avx2")]
pub(super) fn st_u16x16(a: &mut [u16; 16], v: __m256i) {
    // SAFETY: `a` is 16 writable u16 (32 bytes); the store is unaligned.
    unsafe { _mm256_storeu_si256(a.as_mut_ptr().cast(), v) }
}
