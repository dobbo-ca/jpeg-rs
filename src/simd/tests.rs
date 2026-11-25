//! Cross-tier parity: every SIMD tier this CPU has against the scalar code, bit for bit, on
//! deterministic pseudo-random inputs and edge sizes.
use super::*;
use crate::encoder::Quant;
use crate::{dct, encoder, sample};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        (self.next() >> 11) % n
    }
    fn byte(&mut self) -> u8 {
        (self.next() >> 24) as u8
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.byte()).collect()
    }
}

/// The non-scalar tiers this CPU runs; x86_64 CI runners have AVX2 and aarch64 has NEON, so a
/// missing tier is a failure, not a skip.
fn tiers() -> Vec<Tier> {
    #[cfg(target_arch = "x86_64")]
    assert!(is_x86_feature_detected!("avx2"), "AVX2 expected on x86_64");
    #[cfg(target_arch = "aarch64")]
    assert!(
        std::arch::is_aarch64_feature_detected!("neon"),
        "NEON expected on aarch64"
    );
    let t = hardware_tier();
    assert_ne!(t, Tier::Scalar, "no SIMD tier detected");
    vec![t]
}

#[test]
fn tier_is_detected_or_forced() {
    let t = tiers()[0];
    if std::env::var("JPEG_SIMD").as_deref() == Ok("scalar") {
        assert_eq!(tier(), Tier::Scalar);
        assert_eq!(simd_tier(), "scalar");
    } else {
        assert_eq!(tier(), t);
        assert_ne!(simd_tier(), "scalar");
    }
}

const SIZES: [usize; 30] = [
    0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 33, 39, 40, 41, 47, 48, 49, 63, 64, 65,
    95, 96, 97, 127, 1000,
];

#[test]
fn idct8_matches_scalar() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for t in tiers() {
        for i in 0..20_000 {
            let mut quant = [0u16; 64];
            let flat = rng.below(2) == 0;
            let q0 = 1 + rng.below(255) as u16;
            for q in quant.iter_mut() {
                *q = if flat { q0 } else { 1 + rng.below(255) as u16 };
            }
            let mut coef = [0i16; 64];
            match i % 4 {
                0 => {
                    // A real block: random pixels through the encoder's FDCT and quantizer.
                    for q in quant.iter_mut() {
                        *q = 1 + (*q - 1) % 32;
                    }
                    let mut pix = [0u8; 64];
                    match rng.below(3) {
                        0 => pix.iter_mut().for_each(|p| *p = rng.byte()),
                        1 => {
                            let base = rng.byte();
                            pix.iter_mut()
                                .for_each(|p| *p = base.saturating_add(rng.byte() & 15));
                        }
                        _ => pix
                            .iter_mut()
                            .enumerate()
                            .for_each(|(k, p)| *p = if (k + k / 8) % 2 == 0 { 0 } else { 255 }),
                    }
                    let q32: [i32; 64] = quant.map(|q| q as i32);
                    encoder::fdct_quant_rows(&pix, 0, 8, &Quant::new(&q32), &mut coef);
                }
                1 => {
                    for q in quant.iter_mut() {
                        *q = 1 + (*q - 1) % 8;
                    }
                    for _ in 0..1 + rng.below(8) {
                        coef[rng.below(64) as usize] = rng.below(513) as i16 - 256;
                    }
                }
                2 => {
                    coef[0] = rng.below(2049) as i16 - 1024;
                    coef[1 + rng.below(63) as usize] = rng.below(129) as i16 - 64;
                }
                _ => coef.iter_mut().for_each(|c| *c = rng.next() as i16),
            }
            let stride = 8 + rng.below(13) as usize;
            let off = rng.below(4) as usize;
            // Half the blocks get the shortest buffer the kernel accepts, the rest 3 bytes more.
            let len = off + 7 * stride + 8 + if i % 2 == 0 { 3 } else { 0 };
            let mut want = vec![0x55u8; len];
            let mut got = want.clone();
            dct::idct8(&coef, &quant, &mut want[off..], stride);
            assert!(
                idct8_try(t, &coef, &quant, &mut got[off..], stride),
                "block {i} refused"
            );
            assert_eq!(got, want, "block {i}");
            let mut disp = vec![0x55u8; len];
            idct8_with(t, &coef, &quant, &mut disp[off..], stride);
            assert_eq!(disp, want, "dispatch, block {i}");
        }
    }
}

/// The 16-bit first pass at its edges: products at and just past the `i16` limits, quantizers up to
/// 65535 (bit 15 set), and blocks of one extreme value, which drive pass 1 to its largest sums.
/// Every block must be taken and equal the scalar IDCT.
#[test]
fn idct8_i16_edges_match_scalar() {
    let mut rng = Rng(0x5eed_1d07_0c0f_fee5);
    for t in tiers() {
        for i in 0..40_000 {
            let q = match i % 8 {
                0 => 1u16,
                1 => 2,
                2 => 255,
                3 => 257,
                4 => 32767,
                5 => 32768,
                6 => 65535,
                _ => 1 + rng.below(1000) as u16,
            };
            let mut quant = [q; 64];
            if i % 3 == 0 {
                quant.iter_mut().for_each(|x| {
                    if rng.below(4) == 0 {
                        *x = 1 + rng.below(300) as u16;
                    }
                });
            }
            let lim = (32767 / i32::from(q)).max(1);
            let mut coef = [0i16; 64];
            let sign = if rng.below(2) == 0 { 1 } else { -1 };
            for (k, c) in coef.iter_mut().enumerate() {
                let v = match (i / 8) % 4 {
                    // One extreme value everywhere (row 0's weights are all positive).
                    0 => sign * lim,
                    1 => sign * (lim + rng.below(2) as i32),
                    // Extremes at random, signs random, some zero.
                    2 => match rng.below(6) {
                        0 => lim,
                        1 => -lim,
                        2 => -(lim + 1),
                        3 => 0,
                        4 => lim + 1,
                        _ => i32::from(rng.next() as i16),
                    },
                    _ => {
                        if k % 8 == (i / 32) % 8 {
                            sign * lim
                        } else {
                            0
                        }
                    }
                };
                *c = v.clamp(-32768, 32767) as i16;
            }
            let stride = 8 + rng.below(5) as usize;
            let len = 7 * stride + 8;
            let mut want = vec![0x55u8; len];
            let mut got = want.clone();
            dct::idct8(&coef, &quant, &mut want, stride);
            assert!(
                idct8_try(t, &coef, &quant, &mut got, stride),
                "block {i} refused"
            );
            assert_eq!(got, want, "block {i}");
            let mut disp = vec![0x55u8; len];
            idct8_with(t, &coef, &quant, &mut disp, stride);
            assert_eq!(disp, want, "dispatch, block {i}");
        }
    }
}

/// The sign of each input's weight in each output of one pass-1 column of `idct8` (row = output,
/// column = input), which a worst-case block aligns its coefficients with.
const IDCT8_SIGNS: [[i32; 8]; 8] = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, -1, -1, -1, -1, -1],
    [1, 1, -1, -1, -1, 1, 1, 1],
    [1, 1, -1, -1, 1, 1, -1, -1],
    [1, -1, -1, 1, 1, -1, -1, 1],
    [1, -1, -1, 1, -1, -1, 1, -1],
    [1, -1, 1, 1, -1, 1, -1, 1],
    [1, -1, 1, -1, 1, -1, 1, -1],
];

/// The same for `idct4` (input 4 is not an input).
const IDCT4_SIGNS: [[i32; 8]; 4] = [
    [1, 1, 1, 1, 0, -1, -1, -1],
    [1, 1, -1, -1, 0, 1, 1, -1],
    [1, -1, -1, 1, 0, -1, 1, 1],
    [1, -1, 1, -1, 0, 1, -1, 1],
];

/// (quantizer, coefficient magnitude) pairs. The products 35080 and 35082 straddle the `idct8`
/// single-pass bound (35081) and 27563 and 27564 the `idct4` one; 17540 * 2 and 17541 * 2 are past
/// 16 bits, 32767 * 1 is not (the x86_64 16-bit kernel); the rest are the largest products: 255
/// with 2047, 65535 with 32767 and with -32768.
const WORST: [(u16, i32); 9] = [
    (2, 17540),
    (2, 17541),
    (1, 27563),
    (1, 27564),
    (1, 32767),
    (255, 2047),
    (255, 138),
    (65535, 32767),
    (65535, 32768),
];

/// A block whose rows of coefficients carry `signs` (one entry per row, 0 for none) times the
/// magnitude `m` at quantizer `q`: every column is then a worst case for pass 1 of its output row.
fn worst_block(signs: &[i32; 8], q: u16, m: i32) -> ([i16; 64], [u16; 64]) {
    let mut coef = [0i16; 64];
    for (k, c) in coef.iter_mut().enumerate() {
        *c = (signs[k / 8] * m).clamp(-32768, 32767) as i16;
    }
    (coef, [q; 64])
}

#[test]
fn idct8_tiers_match_scalar() {
    let mut rng = Rng(0x7157_0b1e_55ca_1e11);
    for t in tiers() {
        for (q, m) in WORST {
            for k in 0..8 {
                for sign in [1, -1] {
                    let signs = IDCT8_SIGNS[k].map(|s| s * sign);
                    let (mut coef, quant) = worst_block(&signs, q, m);
                    // Also a variant with a random output row's worst case in each column.
                    for variant in 0..2 {
                        if variant == 1 {
                            for c in 0..8 {
                                let (k2, s2) = (rng.below(8) as usize, 1 - 2 * rng.below(2) as i32);
                                for r in 0..8 {
                                    coef[8 * r + c] =
                                        (IDCT8_SIGNS[k2][r] * s2 * m).clamp(-32768, 32767) as i16;
                                }
                            }
                        }
                        let stride = 8 + rng.below(5) as usize;
                        let mut want = vec![0x55u8; 7 * stride + 8];
                        let mut got = want.clone();
                        dct::idct8(&coef, &quant, &mut want, stride);
                        assert!(idct8_try(t, &coef, &quant, &mut got, stride));
                        assert_eq!(got, want, "q {q} m {m} row {k} sign {sign} v {variant}");
                    }
                }
            }
        }
    }
}

/// Blocks whose products fit an `i16` (so the 16-bit kernel takes them) and whose pass-2 output
/// is far outside 0..=255 and often outside the 10-bit range, so the wrap of bits 18..=27 and the
/// pack to bytes are tested at their edges. A DC of one sign with a random few AC terms of the
/// same size, at small quantizers.
#[test]
fn idct8_wrapping_output_matches_scalar() {
    let mut rng = Rng(0x0dd5_7a11_ba5e_ba11);
    for t in tiers() {
        let mut taken = 0;
        for i in 0..20_000 {
            let q = 1 + rng.below(8) as u16;
            let quant = [q; 64];
            let lim = 32767 / i32::from(q);
            let sign = if rng.below(2) == 0 { 1 } else { -1 };
            let mut coef = [0i16; 64];
            coef[0] = (sign * (lim / 2 + rng.below(lim as u64 / 2 + 1) as i32)) as i16;
            for _ in 0..rng.below(10) {
                let v = rng.below(2 * lim as u64 + 1) as i32 - lim;
                coef[1 + rng.below(63) as usize] = v as i16;
            }
            let stride = 8 + rng.below(5) as usize;
            let len = 7 * stride + 8;
            let mut want = vec![0x55u8; len];
            let mut got = want.clone();
            dct::idct8(&coef, &quant, &mut want, stride);
            if idct8_try(t, &coef, &quant, &mut got, stride) {
                taken += 1;
                assert_eq!(got, want, "block {i}");
            } else {
                assert_eq!(got, vec![0x55u8; len], "refusal wrote, block {i}");
            }
            let mut disp = vec![0x55u8; len];
            idct8_with(t, &coef, &quant, &mut disp, stride);
            assert_eq!(disp, want, "dispatch, block {i}");
        }
        if cfg!(target_arch = "x86_64") {
            assert!(taken >= 15_000, "only {taken} blocks were taken");
        }
    }
}

#[test]
fn idct4_tiers_match_scalar() {
    let mut rng = Rng(0x4a11_ed06_e5ca_1e44);
    for t in tiers() {
        for (q, m) in WORST {
            for k in 0..4 {
                for sign in [1, -1] {
                    let signs = IDCT4_SIGNS[k].map(|s| s * sign);
                    let (mut coef, quant) = worst_block(&signs, q, m);
                    for variant in 0..2 {
                        if variant == 1 {
                            for c in 0..8 {
                                let (k2, s2) = (rng.below(4) as usize, 1 - 2 * rng.below(2) as i32);
                                for r in 0..8 {
                                    coef[8 * r + c] =
                                        (IDCT4_SIGNS[k2][r] * s2 * m).clamp(-32768, 32767) as i16;
                                }
                            }
                        }
                        let stride = 4 + rng.below(17) as usize;
                        let off = rng.below(4) as usize;
                        let len = off + 3 * stride + 4 + 3 * rng.below(2) as usize;
                        let mut want = vec![0x55u8; len];
                        let mut got = want.clone();
                        dct::idct4(&coef, &quant, &mut want[off..], stride);
                        assert!(idct4_try(t, &coef, &quant, &mut got[off..], stride));
                        assert_eq!(got, want, "q {q} m {m} row {k} sign {sign} v {variant}");
                        let mut disp = vec![0x55u8; len];
                        idct4_with(t, &coef, &quant, &mut disp[off..], stride);
                        assert_eq!(disp, want, "dispatch, q {q} m {m} row {k} sign {sign}");
                    }
                }
            }
        }
    }
}

#[test]
fn idct4_matches_scalar() {
    let mut rng = Rng(0x0d17_c4c4_9e37_79b9);
    for t in tiers() {
        for i in 0..30_000 {
            let mut coef = [0i16; 64];
            let mut quant = [0u16; 64];
            let q0 = 1 + rng.below(255) as u16;
            match i % 5 {
                0 => {
                    quant
                        .iter_mut()
                        .for_each(|q| *q = 1 + rng.below(255) as u16);
                    coef.iter_mut().for_each(|c| *c = rng.next() as i16);
                }
                1 => {
                    quant = [1 + (q0 - 1) % 8; 64];
                    for _ in 0..1 + rng.below(8) {
                        coef[rng.below(64) as usize] = rng.below(513) as i16 - 256;
                    }
                }
                2 => {
                    // Row 4 and column 4 are not inputs of the 4x4 IDCT: values there must not matter.
                    quant = [1 + (q0 - 1) % 32; 64];
                    for (k, c) in coef.iter_mut().enumerate() {
                        *c = if k / 8 == 4 || k % 8 == 4 {
                            rng.next() as i16
                        } else {
                            rng.below(129) as i16 - 64
                        };
                    }
                }
                3 => {
                    quant.iter_mut().for_each(|q| *q = rng.next() as u16);
                    coef.iter_mut().for_each(|c| {
                        *c = match rng.below(6) {
                            0 => i16::MIN,
                            1 => i16::MAX,
                            2 => 0,
                            3 => 1,
                            4 => -1,
                            _ => rng.next() as i16,
                        }
                    });
                }
                _ => {
                    quant = [1 + (q0 - 1) % 32; 64];
                    coef[0] = rng.below(2049) as i16 - 1024;
                    coef[1 + rng.below(63) as usize] = rng.below(129) as i16 - 64;
                }
            }
            let stride = 4 + rng.below(17) as usize;
            let off = rng.below(4) as usize;
            // Half the blocks get the shortest buffer the kernel accepts, the rest 3 bytes more.
            let len = off + 3 * stride + 4 + if i % 2 == 0 { 3 } else { 0 };
            let mut want = vec![0x55u8; len];
            let mut got = want.clone();
            dct::idct4(&coef, &quant, &mut want[off..], stride);
            assert!(
                idct4_try(t, &coef, &quant, &mut got[off..], stride),
                "block {i} refused"
            );
            assert_eq!(got, want, "block {i}");
            let mut disp = vec![0x55u8; len];
            idct4_with(t, &coef, &quant, &mut disp[off..], stride);
            assert_eq!(disp, want, "dispatch, block {i}");
        }
    }
}

/// A destination one byte short is left alone and refused, so the dispatcher runs scalar.
#[test]
fn idct4_short_dst_is_refused() {
    let coef = [1i16; 64];
    let quant = [3u16; 64];
    for t in tiers() {
        let mut short = vec![0x55u8; 3 * 9 + 3];
        assert!(!idct4_try(t, &coef, &quant, &mut short, 9));
        assert_eq!(short, vec![0x55u8; 3 * 9 + 3]);
    }
}

#[test]
fn color_row_matches_scalar() {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    for t in tiers() {
        for &n in &SIZES {
            for off in 0..3 {
                let (y, cb, cr) = (rng.bytes(n), rng.bytes(n + 2), rng.bytes(n));
                let mut want = vec![0x55u8; 4 * n + 6];
                let mut got = want.clone();
                sample::color_row_with(Tier::Scalar, &mut want[off..off + 4 * n], &y, &cb, &cr);
                sample::color_row_with(t, &mut got[off..off + 4 * n], &y, &cb, &cr);
                assert_eq!(got, want, "n {n} off {off}");
            }
        }
        // Every (cb, cr) pair at a spread of luma values.
        let cb: Vec<u8> = (0..65536u32).map(|i| (i >> 8) as u8).collect();
        let cr: Vec<u8> = (0..65536u32).map(|i| i as u8).collect();
        for yv in [0u8, 1, 2, 63, 127, 128, 129, 200, 253, 254, 255] {
            let y = vec![yv; 65536];
            let mut want = vec![0u8; 4 * 65536];
            let mut got = want.clone();
            sample::color_row_with(Tier::Scalar, &mut want, &y, &cb, &cr);
            sample::color_row_with(t, &mut got, &y, &cb, &cr);
            assert!(got == want, "y {yv}");
        }
    }
}

fn rows(rng: &mut Rng, n: usize) -> Vec<u8> {
    match rng.below(4) {
        0 => vec![255; n],
        1 => vec![0; n],
        2 => (0..n).map(|i| if i % 2 == 0 { 255 } else { 0 }).collect(),
        _ => rng.bytes(n),
    }
}

#[test]
fn fancy_upsample_matches_scalar() {
    let mut rng = Rng(0x0fed_cba9_8765_4321);
    for t in tiers() {
        for dsw in (2..=160).chain([200, 1000]) {
            for round in 0..32 {
                let w = 2 * dsw - rng.below(2) as usize;
                let extra = rng.below(3) as usize;
                // The rows start 0 to 3 bytes into their buffers.
                let k = round % 4;
                let (bc, bo) = (
                    rows(&mut rng, k + dsw + extra),
                    rows(&mut rng, k + dsw + extra),
                );
                let (cur, other) = (&bc[k..], &bo[k..]);
                let mut want = vec![0x55u8; w];
                let mut got = want.clone();
                sample::h2v1_row_with(Tier::Scalar, &mut want, cur, dsw, w);
                sample::h2v1_row_with(t, &mut got, cur, dsw, w);
                assert_eq!(got, want, "h2v1 dsw {dsw} w {w}");
                sample::h2v2_row_with(Tier::Scalar, &mut want, cur, other, dsw, w);
                sample::h2v2_row_with(t, &mut got, cur, other, dsw, w);
                assert_eq!(got, want, "h2v2 dsw {dsw} w {w}");
            }
        }
    }
}

#[test]
fn quantizer_reciprocal_is_exact() {
    // The kernel computes floor((|x| + 4q) / 8q) with libjpeg-turbo's reciprocal tables: exact
    // for every 16-bit magnitude (the FDCT output is under 2^14) and every quantizer.
    for q in 1..=255i32 {
        let t = Quant::new(&[q; 64]);
        let (r, c, s) = (
            u32::from(t.recip[0]),
            u32::from(t.corr[0]),
            u32::from(t.scale[0]),
        );
        for x in 0..1u32 << 15 {
            let got = ((((x + c) * r) >> 16) * s) >> 16;
            assert_eq!(got, (x + 4 * q as u32) / (8 * q as u32), "x {x} q {q}");
        }
    }
}

#[test]
fn fdct_quant_matches_scalar() {
    let mut rng = Rng(0x5555_aaaa_1357_9bdf);
    let mut pairs = 0;
    for t in tiers() {
        for i in 0..6_000 {
            let mut quant = [0i32; 64];
            match i % 3 {
                0 => quant
                    .iter_mut()
                    .for_each(|q| *q = 1 + rng.below(255) as i32),
                1 => quant.fill(1 + rng.below(255) as i32),
                _ => quant.iter_mut().for_each(|q| *q = 1 + rng.below(8) as i32),
            }
            let stride = 8 + rng.below(17) as usize;
            let off = rng.below(6) as usize;
            // Half the blocks get the shortest buffer that holds both blocks of the pair.
            let len = if i % 2 == 0 {
                off + 8 * stride + 16
            } else {
                off + 7 * stride + 16
            };
            let mut pix = rows(&mut rng, len);
            if i % 5 == 0 {
                // A block of extremes next to noise.
                for r in 0..8 {
                    for c in 0..8 {
                        pix[off + r * stride + c] = if (r + c) % 2 == 0 { 0 } else { 255 };
                    }
                }
            }
            let quant = Quant::new(&quant);
            let mut want = [0x1234i16; 64];
            let mut got = want;
            encoder::fdct_quant_rows(&pix, off, stride, &quant, &mut want);
            fdct_quant_rows(t, &pix, off, stride, &quant, &mut got);
            assert_eq!(got, want, "block {i}");
            // The block 8 columns on, and the two at once where the CPU has the wide kernel.
            let mut want2 = [0x1234i16; 64];
            let (mut got_a, mut got_b) = (want2, want2);
            encoder::fdct_quant_rows(&pix, off + 8, stride, &quant, &mut want2);
            if fdct_quant_pair(
                t,
                (&pix, off, stride),
                (&pix, off + 8, stride),
                &quant,
                &mut got_a,
                &mut got_b,
            ) {
                pairs += 1;
                assert_eq!(got_a, want, "pair a, block {i}");
                assert_eq!(got_b, want2, "pair b, block {i}");
            }
        }
    }
    #[cfg(target_arch = "x86_64")]
    assert_eq!(
        pairs > 0,
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw"),
        "the two-block kernel runs where AVX-512 is"
    );
    #[cfg(not(target_arch = "x86_64"))]
    assert_eq!(pairs, 0);
}

/// The AVX2 and the AVX-512 RGBA to YCbCr kernels, each on its own (the dispatcher runs only the
/// wider one where there is one).
#[cfg(target_arch = "x86_64")]
#[test]
fn x86_encoder_colour_kernels_match_scalar() {
    tiers(); // AVX2 is asserted before any kernel below runs
    let mut rng = Rng(0x0bad_cafe_f00d_1234);
    let mut inputs: Vec<Vec<u8>> = [16usize, 32, 48, 64, 160, 1008]
        .iter()
        .map(|&n| rows(&mut rng, 4 * n))
        .collect();
    // A grid over RGB, including every corner.
    let mut grid = Vec::new();
    for r in (0..=255u32).step_by(5).chain([255]) {
        for g in (0..=255u32).step_by(5).chain([255]) {
            for b in (0..=255u32).step_by(5).chain([255]) {
                grid.extend_from_slice(&[r as u8, g as u8, b as u8, 7]);
            }
        }
    }
    grid.truncate(grid.len() / 64 * 64);
    inputs.push(grid);
    let avx512 = is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw");
    for src in &inputs {
        let n = src.len() / 4;
        let mut want = (vec![0u8; n], vec![0u8; n], vec![0u8; n]);
        encoder::rgba_to_ycbcr_row_with(Tier::Scalar, src, &mut want.0, &mut want.1, &mut want.2);
        for wide in [false, true] {
            if wide && !avx512 {
                continue;
            }
            let mut got = (vec![0x55u8; n], vec![0x55u8; n], vec![0x55u8; n]);
            let done = rgba_to_ycbcr_row_x86(wide, src, &mut got.0, &mut got.1, &mut got.2)
                .expect("the kernel runs here");
            assert_eq!(done, n, "wide {wide}: the kernel covers whole steps of 16");
            assert!(got == want, "wide {wide}, n {n}");
        }
    }
}

/// The decode colour and upsample kernels, the AVX2 and the AVX-512 ones each on its own (the
/// dispatcher runs only the AVX-512 kernel where there is one): the kernel's pixels and columns
/// with the scalar tail equal scalar, for every size and a start 0 to 3 bytes in, and the colour
/// kernel over every (cb, cr) pair.
#[cfg(target_arch = "x86_64")]
#[test]
fn x86_decode_kernels_match_scalar() {
    tiers(); // AVX2 is asserted before any kernel below runs
    let avx512 = is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw");
    let mut rng = Rng(0x7a11_c0de_5eed_0042);
    for wide in [false, true] {
        if wide && !avx512 {
            continue;
        }
        for &n in &SIZES {
            for off in 0..4 {
                let (y, cb, cr) = (rng.bytes(n), rng.bytes(n + 2), rng.bytes(n));
                let mut want = vec![0x55u8; 4 * n + 6];
                let mut got = want.clone();
                sample::color_row_with(Tier::Scalar, &mut want[off..off + 4 * n], &y, &cb, &cr);
                let done = color_row_x86(wide, &mut got[off..off + 4 * n], &y, &cb, &cr)
                    .expect("the kernel runs here");
                assert_eq!(done, n & !15, "wide {wide} n {n}");
                sample::color_row_with(
                    Tier::Scalar,
                    &mut got[off + 4 * done..off + 4 * n],
                    &y[done..],
                    &cb[done..],
                    &cr[done..],
                );
                assert_eq!(got, want, "wide {wide} n {n} off {off}");
            }
        }
        // Every (cb, cr) pair at a spread of luma values.
        let cb: Vec<u8> = (0..65536u32).map(|i| (i >> 8) as u8).collect();
        let cr: Vec<u8> = (0..65536u32).map(|i| i as u8).collect();
        for yv in [0u8, 1, 2, 63, 127, 128, 129, 200, 253, 254, 255] {
            let y = vec![yv; 65536];
            let mut want = vec![0u8; 4 * 65536];
            let mut got = want.clone();
            sample::color_row_with(Tier::Scalar, &mut want, &y, &cb, &cr);
            let done = color_row_x86(wide, &mut got, &y, &cb, &cr).expect("the kernel runs here");
            assert_eq!(done, 65536);
            assert!(got == want, "wide {wide} y {yv}");
        }
        for dsw in (2..=200).chain([1000]) {
            for round in 0..16 {
                let k = round % 4;
                let (bc, bo) = (rows(&mut rng, k + dsw + 1), rows(&mut rng, k + dsw + 1));
                let (cur, other) = (&bc[k..], &bo[k..]);
                let mut want = vec![0x55u8; 2 * dsw];
                sample::h2v1_row_with(Tier::Scalar, &mut want, cur, dsw, 2 * dsw);
                let mut got = vec![0x55u8; 2 * dsw];
                let c0 =
                    h2v1_mid_x86(wide, &mut got, cur, 1, dsw - 1).expect("the kernel runs here");
                assert!(
                    got[..2].iter().all(|&b| b == 0x55),
                    "h2v1 wide {wide} dsw {dsw}"
                );
                assert!(
                    got[2 * c0..].iter().all(|&b| b == 0x55),
                    "h2v1 wide {wide} dsw {dsw}"
                );
                assert_eq!(
                    got[2..2 * c0],
                    want[2..2 * c0],
                    "h2v1 wide {wide} dsw {dsw}"
                );
                sample::h2v2_row_with(Tier::Scalar, &mut want, cur, other, dsw, 2 * dsw);
                let mut got = vec![0x55u8; 2 * dsw];
                let c0 = h2v2_mid_x86(wide, &mut got, cur, other, 1, dsw - 1)
                    .expect("the kernel runs here");
                assert!(
                    got[..2].iter().all(|&b| b == 0x55),
                    "h2v2 wide {wide} dsw {dsw}"
                );
                assert!(
                    got[2 * c0..].iter().all(|&b| b == 0x55),
                    "h2v2 wide {wide} dsw {dsw}"
                );
                assert_eq!(
                    got[2..2 * c0],
                    want[2..2 * c0],
                    "h2v2 wide {wide} dsw {dsw}"
                );
            }
        }
    }
}

/// Whether this CPU has AVX-512 is logged (`--nocapture`), and with `JPEG_EXPECT_AVX512=1` it is
/// required: the wide kernels and the two-block FDCT are otherwise skipped without a word.
#[cfg(target_arch = "x86_64")]
#[test]
fn avx512_where_expected() {
    let have = has_avx512();
    eprintln!("AVX-512 F and BW detected: {have}");
    if std::env::var("JPEG_EXPECT_AVX512").as_deref() == Ok("1") {
        assert!(have, "JPEG_EXPECT_AVX512=1, but this CPU has no AVX-512");
    }
}

/// The AVX2 colour kernel and the scalar tail that finishes a row after it, for every size and a
/// start 0 to 3 bytes in (the dispatcher runs only the AVX-512 kernel where there is one).
#[cfg(target_arch = "x86_64")]
#[test]
fn x86_avx2_colour_kernel_with_scalar_tail() {
    tiers();
    let mut rng = Rng(0x5eed_0a2b_3c4d_5e6f);
    for &n in &SIZES {
        for off in 0..4 {
            let buf = rows(&mut rng, 4 * n + 8);
            let src = &buf[off..];
            let mut want = (vec![0x55u8; n], vec![0x55u8; n], vec![0x55u8; n]);
            let mut got = want.clone();
            encoder::rgba_to_ycbcr_row_with(
                Tier::Scalar,
                src,
                &mut want.0,
                &mut want.1,
                &mut want.2,
            );
            let done = rgba_to_ycbcr_row_x86(false, src, &mut got.0, &mut got.1, &mut got.2)
                .expect("the AVX2 kernel runs here");
            assert!(done <= n, "n {n} off {off}");
            encoder::rgba_to_ycbcr_row_with(
                Tier::Scalar,
                &src[4 * done..],
                &mut got.0[done..],
                &mut got.1[done..],
                &mut got.2[done..],
            );
            assert!(got == want, "n {n} off {off}");
        }
    }
}

#[test]
fn rgba_to_ycbcr_matches_scalar() {
    let mut rng = Rng(0xdead_beef_cafe_f00d);
    for t in tiers() {
        for &n in &SIZES {
            for off in 0..4 {
                let buf = rows(&mut rng, 4 * n + 8);
                let src = &buf[off..];
                let mut want = (
                    vec![0x55u8; n + 3],
                    vec![0x55u8; n + 3],
                    vec![0x55u8; n + 3],
                );
                let mut got = want.clone();
                encoder::rgba_to_ycbcr_row_with(
                    Tier::Scalar,
                    src,
                    &mut want.0[..n],
                    &mut want.1[..n],
                    &mut want.2[..n],
                );
                encoder::rgba_to_ycbcr_row_with(
                    t,
                    src,
                    &mut got.0[..n],
                    &mut got.1[..n],
                    &mut got.2[..n],
                );
                assert!(got == want, "n {n} off {off}");
            }
        }
        // A grid over RGB, including every corner.
        let mut src = Vec::new();
        for r in (0..=255u32).step_by(5).chain([255]) {
            for g in (0..=255u32).step_by(5).chain([255]) {
                for b in (0..=255u32).step_by(5).chain([255]) {
                    src.extend_from_slice(&[r as u8, g as u8, b as u8, 7]);
                }
            }
        }
        let n = src.len() / 4;
        let mut want = (vec![0u8; n], vec![0u8; n], vec![0u8; n]);
        let mut got = want.clone();
        encoder::rgba_to_ycbcr_row_with(Tier::Scalar, &src, &mut want.0, &mut want.1, &mut want.2);
        encoder::rgba_to_ycbcr_row_with(t, &src, &mut got.0, &mut got.1, &mut got.2);
        assert!(got == want, "grid");
    }
}

#[test]
fn downsample_h2v2_matches_scalar() {
    let mut rng = Rng(0x0123_4567_89ab_cdef);
    for t in tiers() {
        for &n in &SIZES {
            for off in 0..4 {
                let (b0, b1) = (rows(&mut rng, 2 * n + 5), rows(&mut rng, 2 * n + 3));
                let (r0, r1) = (&b0[off..], &b1[off..]);
                let mut want = vec![0x55u8; n + 3];
                let mut got = want.clone();
                encoder::downsample_h2v2_row_with(Tier::Scalar, &mut want[..n], r0, r1);
                encoder::downsample_h2v2_row_with(t, &mut got[..n], r0, r1);
                assert_eq!(got, want, "n {n} off {off}");
            }
        }
    }
}

#[test]
#[cfg_attr(not(target_arch = "x86_64"), allow(unused_mut))]
fn write_block_matches_scalar() {
    use crate::encoder::{BitWriter, std_codes, write_block_scalar};
    let codes = std_codes();
    let mut rng = Rng(0xfeed_f00d_1234_5678);
    #[cfg(target_arch = "x86_64")]
    let wide = is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw");
    #[cfg(not(target_arch = "x86_64"))]
    let wide = false;
    for _t in tiers() {
        for round in 0..400 {
            let n = 1 + rng.below(40) as usize;
            let mut blocks = Vec::new();
            for _ in 0..n {
                let mut b = [0i16; 64];
                match rng.below(6) {
                    0 => {}                                    // all zero
                    1 => b[0] = rng.below(2001) as i16 - 1000, // DC only
                    2 => {
                        // sparse, low frequencies
                        b[0] = rng.below(401) as i16 - 200;
                        for _ in 0..1 + rng.below(8) {
                            b[rng.below(20) as usize] = rng.below(41) as i16 - 20;
                        }
                    }
                    3 => {
                        // dense
                        for c in b.iter_mut() {
                            *c = rng.below(63) as i16 - 31;
                        }
                    }
                    4 => {
                        // the far end: long zero runs, ZRL, the last coefficient
                        b[0] = rng.below(21) as i16 - 10;
                        b[63] = 1 + rng.below(1022) as i16;
                        if rng.below(2) == 0 {
                            b[40] = -(1 + rng.below(1022) as i16);
                        }
                    }
                    _ => {
                        // the extremes of baseline and beyond
                        for c in b.iter_mut() {
                            if rng.below(3) == 0 {
                                *c = [-32768, -1023, -1, 1, 1023, 32767, 255, -256]
                                    [rng.below(8) as usize];
                            }
                        }
                    }
                }
                blocks.push(b);
            }
            let (mut want, mut got, mut got_wide) = (
                BitWriter::new(Vec::new()),
                BitWriter::new(Vec::new()),
                BitWriter::new(Vec::new()),
            );
            let (mut lw, mut lg, mut lgw) = (0i32, 0i32, 0i32);
            for (i, b) in blocks.iter().enumerate() {
                let (dct, act) = if i % 3 == 0 {
                    (&codes.dc_luma, &codes.ac_luma)
                } else {
                    (&codes.dc_chroma, &codes.ac_chroma)
                };
                write_block_scalar(&mut want, b, &mut lw, dct, act);
                // Every kernel this CPU has, each on its own: on x86_64 the AVX2 prep and, with
                // AVX-512, the wide one (the dispatcher runs only the wider).
                #[cfg(target_arch = "x86_64")]
                {
                    assert!(has_bmi(), "no BMI on x86_64");
                    assert!(write_block_x86(false, &mut got, b, &mut lg, dct, act));
                    if wide {
                        assert!(write_block_x86(true, &mut got_wide, b, &mut lgw, dct, act));
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    assert!(!super::write_block(_t, &mut got, b, &mut lg, dct, act));
                    write_block_scalar(&mut got, b, &mut lg, dct, act);
                }
            }
            want.flush();
            got.flush();
            got_wide.flush();
            assert_eq!(got.out, want.out, "round {round}");
            assert_eq!(lg, lw);
            if wide {
                assert_eq!(got_wide.out, want.out, "wide, round {round}");
                assert_eq!(lgw, lw);
            }
        }
    }
}

/// The decoder's MCU loop runs in its BMI build on every x86_64 CI host (AVX2 and BMI), unless
/// `JPEG_SIMD=scalar` forces the plain one.
#[cfg(target_arch = "x86_64")]
#[test]
fn bmi_decode_where_expected() {
    tiers();
    assert!(has_bmi(), "no BMI on x86_64");
    let forced = std::env::var("JPEG_SIMD").as_deref() == Ok("scalar");
    assert_eq!(bmi_decode(), !forced);
}

/// The BMI build of the MCU decoder against `decode_mcu_fast_impl`: the same blocks, DC
/// predictions, AC flags and reader state after every MCU, on random entropy data (0xFF stuffed,
/// so the stream runs to its end) with the standard tables.
#[cfg(target_arch = "x86_64")]
#[test]
fn bmi_mcu_decode_matches_impl() {
    tiers();
    let mut rng = Rng(0x7b3d_91c5_2e48_a6f0);
    for round in 0..300 {
        let n = 1 + rng.below(if round % 10 == 0 { 2000 } else { 200 }) as usize;
        let mut data = Vec::new();
        for _ in 0..n {
            let b = rng.byte();
            data.push(b);
            if b == 0xff {
                data.push(0);
            }
        }
        assert!(
            crate::kernel_check::mcu_decode(&data).expect("the BMI build matches"),
            "no BMI build to compare"
        );
    }
}
