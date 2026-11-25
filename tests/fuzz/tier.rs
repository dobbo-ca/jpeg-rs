//! The SIMD targets assert that this CPU has a SIMD tier (x86_64 CI runners have AVX2, aarch64
//! has NEON): without one the differential targets would fuzz scalar against scalar.

pub fn check() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let t = jpeg::kernel_check::hardware_tier();
        assert_ne!(t, "scalar", "no SIMD tier: scalar against scalar");
        let wide = jpeg::kernel_check::wide_kernels();
        eprintln!("SIMD tier fuzzed against scalar: {t} ({wide})");
        // A node without AVX-512 would skip the wide kernels without a word: JPEG_EXPECT_AVX512=1
        // makes that a failure.
        if std::env::var("JPEG_EXPECT_AVX512").as_deref() == Ok("1") && cfg!(target_arch = "x86_64")
        {
            assert_eq!(wide, "avx512", "JPEG_EXPECT_AVX512=1, but no AVX-512");
        }
    });
}
