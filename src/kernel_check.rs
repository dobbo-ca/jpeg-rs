//! Differential checks of the SIMD kernels, for the fuzz targets (`tests/fuzz/fuzz_targets/simd_*.rs`):
//! each runs one kernel in this CPU's tier and in scalar on the inputs it is given and returns a
//! description of the first difference. They are not part of the API.
//!
//! None takes [`Limits`](crate::Limits): each works on one fixed-size block or one row the fuzz
//! target hands it, allocates nothing from a size in the input, polls no cancel flag and spawns no
//! thread, so a limit would check nothing. The `Err(String)` is the difference found, not a
//! resource error.
use crate::encoder::{BitWriter, Quant, write_block_scalar};
use crate::simd::{self, Tier};
use crate::{dct, encoder, sample};

/// This CPU's SIMD tier, ignoring `JPEG_SIMD`: `"avx2"`, `"neon"` or `"scalar"`.
pub fn hardware_tier() -> &'static str {
    match simd::hardware_tier() {
        Tier::Scalar => "scalar",
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2(_) => "avx2",
        #[cfg(target_arch = "aarch64")]
        Tier::Neon(_) => "neon",
    }
}

/// The wide x86 kernels this CPU runs beside the base tier, for the fuzz log: `"avx512"`,
/// `"avx2 only"` on x86_64 without AVX-512, `""` elsewhere.
pub fn wide_kernels() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if simd::has_wide() {
            "avx512"
        } else {
            "avx2 only"
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        ""
    }
}

fn same<T: PartialEq>(what: &str, got: T, want: T) -> Result<(), String> {
    if got == want {
        Ok(())
    } else {
        Err(format!("{what}: the SIMD output differs from scalar"))
    }
}

/// The islow IDCT of a block at `stride` (8 to 31) and a start offset `off` (0 to 3). No SIMD
/// tier refuses a block, so a refusal is a difference.
pub fn idct8(coef: &[i16; 64], quant: &[u16; 64], off: usize, stride: usize) -> Result<(), String> {
    let t = simd::hardware_tier();
    let (off, stride) = (off % 4, 8 + stride % 24);
    let len = off + 7 * stride + 8 + 3;
    let mut want = vec![0x55u8; len];
    let mut got = want.clone();
    dct::idct8(coef, quant, &mut want[off..], stride);
    if simd::idct8_try(t, coef, quant, &mut got[off..], stride) {
        same("idct8", &got, &want)?;
    } else if t != Tier::Scalar {
        return Err("idct8: the SIMD tier refused a block".into());
    }
    let mut disp = vec![0x55u8; len];
    simd::idct8_with(t, coef, quant, &mut disp[off..], stride);
    same("idct8 dispatch", &disp, &want)
}

/// The reduced 4x4 IDCT of a block at `stride` (4 to 31) and a start offset `off` (0 to 3), as
/// [`idct8`].
pub fn idct4(coef: &[i16; 64], quant: &[u16; 64], off: usize, stride: usize) -> Result<(), String> {
    let t = simd::hardware_tier();
    let (off, stride) = (off % 4, 4 + stride % 28);
    let len = off + 3 * stride + 4 + 3;
    let mut want = vec![0x55u8; len];
    let mut got = want.clone();
    dct::idct4(coef, quant, &mut want[off..], stride);
    if simd::idct4_try(t, coef, quant, &mut got[off..], stride) {
        same("idct4", &got, &want)?;
    } else if t != Tier::Scalar {
        return Err("idct4: the SIMD tier refused a block".into());
    }
    let mut disp = vec![0x55u8; len];
    simd::idct4_with(t, coef, quant, &mut disp[off..], stride);
    same("idct4 dispatch", &disp, &want)
}

/// YCbCr to RGBA over the shortest of the three rows, into a destination at offset `off` (0 to 2).
pub fn color_row(y: &[u8], cb: &[u8], cr: &[u8], off: usize) -> Result<(), String> {
    let n = y.len().min(cb.len()).min(cr.len());
    let (y, cb, cr, off) = (&y[..n], &cb[..n], &cr[..n], off % 3);
    let mut want = vec![0x55u8; 4 * n + 6];
    let mut got = want.clone();
    sample::color_row_with(Tier::Scalar, &mut want[off..off + 4 * n], y, cb, cr);
    sample::color_row_with(simd::hardware_tier(), &mut got[off..off + 4 * n], y, cb, cr);
    same("color_row", &got, &want)?;
    // Each x86 kernel on its own, with the scalar tail.
    #[cfg(target_arch = "x86_64")]
    for wide in [false, true] {
        let mut got = vec![0x55u8; 4 * n + 6];
        let Some(done) = simd::color_row_x86(wide, &mut got[off..off + 4 * n], y, cb, cr) else {
            continue;
        };
        sample::color_row_with(
            Tier::Scalar,
            &mut got[off + 4 * done..off + 4 * n],
            &y[done..],
            &cb[done..],
            &cr[done..],
        );
        same(
            if wide {
                "color_row avx512"
            } else {
                "color_row avx2"
            },
            &got,
            &want,
        )?;
    }
    Ok(())
}

/// The h2v1 fancy upsample of `row`, to `2 * row.len()` samples or one fewer when `odd`.
pub fn h2v1(row: &[u8], odd: bool) -> Result<(), String> {
    let dsw = row.len();
    if dsw < 2 {
        return Ok(());
    }
    let w = 2 * dsw - usize::from(odd);
    let mut want = vec![0x55u8; w];
    let mut got = want.clone();
    sample::h2v1_row_with(Tier::Scalar, &mut want, row, dsw, w);
    sample::h2v1_row_with(simd::hardware_tier(), &mut got, row, dsw, w);
    same("h2v1", &got, &want)?;
    // Each x86 kernel on its own: the columns it did equal scalar, the rest is untouched.
    #[cfg(target_arch = "x86_64")]
    for wide in [false, true] {
        let mut got = vec![0x55u8; w];
        let Some(c0) = simd::h2v1_mid_x86(wide, &mut got, row, 1, dsw - 1) else {
            continue;
        };
        same(
            if wide { "h2v1 avx512" } else { "h2v1 avx2" },
            &got[2..2 * c0],
            &want[2..2 * c0],
        )?;
        same(
            "h2v1 untouched",
            got[2 * c0..].iter().all(|&b| b == 0x55),
            true,
        )?;
    }
    Ok(())
}

/// The h2v2 fancy upsample of `cur` with its neighbour row `other`, as [`h2v1`].
pub fn h2v2(cur: &[u8], other: &[u8], odd: bool) -> Result<(), String> {
    let dsw = cur.len().min(other.len());
    if dsw < 2 {
        return Ok(());
    }
    let w = 2 * dsw - usize::from(odd);
    let mut want = vec![0x55u8; w];
    let mut got = want.clone();
    sample::h2v2_row_with(Tier::Scalar, &mut want, cur, other, dsw, w);
    sample::h2v2_row_with(simd::hardware_tier(), &mut got, cur, other, dsw, w);
    same("h2v2", &got, &want)?;
    // As [`h2v1`].
    #[cfg(target_arch = "x86_64")]
    for wide in [false, true] {
        let mut got = vec![0x55u8; w];
        let Some(c0) = simd::h2v2_mid_x86(wide, &mut got, cur, other, 1, dsw - 1) else {
            continue;
        };
        same(
            if wide { "h2v2 avx512" } else { "h2v2 avx2" },
            &got[2..2 * c0],
            &want[2..2 * c0],
        )?;
        same(
            "h2v2 untouched",
            got[2 * c0..].iter().all(|&b| b == 0x55),
            true,
        )?;
    }
    Ok(())
}

/// The encoder's FDCT and quantizer of the block of `pix` at `off` (0 to 5) and `stride` (8 to 24);
/// quantizers are 1 to 255. A `pix` too short for the block is skipped.
pub fn fdct_quant(pix: &[u8], off: usize, stride: usize, quant: &[u8; 64]) -> Result<(), String> {
    let (off, stride) = (off % 6, 8 + stride % 17);
    if pix.len() < off + 7 * stride + 8 {
        return Ok(());
    }
    let q = Quant::new(&quant.map(|v| 1 + i32::from(v) % 255));
    let mut want = [0x1234i16; 64];
    let mut got = want;
    encoder::fdct_quant_rows(pix, off, stride, &q, &mut want);
    simd::fdct_quant_rows(simd::hardware_tier(), pix, off, stride, &q, &mut got);
    same("fdct_quant", got, want)?;
    // The block 8 columns on, and both at once in the wide kernel where there is one.
    let mut want_b = [0x1234i16; 64];
    let (mut got_a, mut got_b) = (want_b, want_b);
    if pix.len() < off + 8 + 7 * stride + 8 {
        return Ok(());
    }
    encoder::fdct_quant_rows(pix, off + 8, stride, &q, &mut want_b);
    if simd::fdct_quant_pair(
        simd::hardware_tier(),
        (pix, off, stride),
        (pix, off + 8, stride),
        &q,
        &mut got_a,
        &mut got_b,
    ) {
        same("fdct_quant_pair a", got_a, want)?;
        same("fdct_quant_pair b", got_b, want_b)?;
    }
    Ok(())
}

/// The encoder's RGBA to YCbCr over `src.len() / 4` pixels, from a start 0 to 3 bytes in (the
/// first byte says how far); on x86_64 also through the AVX2 and AVX-512 kernels each on its own,
/// with the scalar tail.
pub fn rgba_to_ycbcr(src: &[u8]) -> Result<(), String> {
    let off = src.first().map_or(0, |&b| usize::from(b % 4));
    let src = &src[off.min(src.len())..];
    let n = src.len() / 4;
    let mut want = (vec![0x55u8; n], vec![0x55u8; n], vec![0x55u8; n]);
    let mut got = want.clone();
    encoder::rgba_to_ycbcr_row_with(Tier::Scalar, src, &mut want.0, &mut want.1, &mut want.2);
    encoder::rgba_to_ycbcr_row_with(
        simd::hardware_tier(),
        src,
        &mut got.0,
        &mut got.1,
        &mut got.2,
    );
    same("rgba_to_ycbcr", got, want.clone())?;
    #[cfg(target_arch = "x86_64")]
    for wide in [false, true] {
        let mut got = (vec![0x55u8; n], vec![0x55u8; n], vec![0x55u8; n]);
        let Some(done) = simd::rgba_to_ycbcr_row_x86(wide, src, &mut got.0, &mut got.1, &mut got.2)
        else {
            continue;
        };
        encoder::rgba_to_ycbcr_row_with(
            Tier::Scalar,
            &src[4 * done..],
            &mut got.0[done..],
            &mut got.1[done..],
            &mut got.2[done..],
        );
        same(
            if wide {
                "rgba_to_ycbcr avx512"
            } else {
                "rgba_to_ycbcr avx2"
            },
            got,
            want.clone(),
        )?;
    }
    Ok(())
}

/// The encoder's h2v2 downsample of two rows, to the half of the shorter.
pub fn downsample_h2v2(r0: &[u8], r1: &[u8]) -> Result<(), String> {
    let n = (r0.len() / 2).min(r1.len() / 2);
    let mut want = vec![0x55u8; n];
    let mut got = want.clone();
    encoder::downsample_h2v2_row_with(Tier::Scalar, &mut want, r0, r1);
    encoder::downsample_h2v2_row_with(simd::hardware_tier(), &mut got, r0, r1);
    same("downsample_h2v2", got, want)
}

/// The encoder's block coder over `blocks` (zero blocks, DC-only and full ones, three block kinds
/// by position) in this CPU's tier and in scalar: the same bytes after the flush.
pub fn write_blocks(blocks: &[[i16; 64]]) -> Result<(), String> {
    let codes = crate::encoder::std_codes();
    let t = simd::hardware_tier();
    let (mut want, mut got) = (BitWriter::new(Vec::new()), BitWriter::new(Vec::new()));
    let (mut lw, mut lg) = (0i32, 0i32);
    for (i, blk) in blocks.iter().enumerate() {
        let (dct, act) = if i % 3 == 0 {
            (&codes.dc_luma, &codes.ac_luma)
        } else {
            (&codes.dc_chroma, &codes.ac_chroma)
        };
        write_block_scalar(&mut want, blk, &mut lw, dct, act);
        if !simd::write_block(t, &mut got, blk, &mut lg, dct, act) {
            write_block_scalar(&mut got, blk, &mut lg, dct, act);
        }
    }
    want.flush();
    got.flush();
    same("write_block", got.out, want.out.clone())?;
    // On x86_64 each kernel on its own: AVX2, then AVX-512 where there is one.
    #[cfg(target_arch = "x86_64")]
    for wide in [false, true] {
        let (mut bw, mut last) = (BitWriter::new(Vec::new()), 0i32);
        let mut ran = !blocks.is_empty();
        for (i, blk) in blocks.iter().enumerate() {
            let (dct, act) = if i % 3 == 0 {
                (&codes.dc_luma, &codes.ac_luma)
            } else {
                (&codes.dc_chroma, &codes.ac_chroma)
            };
            if !simd::write_block_x86(wide, &mut bw, blk, &mut last, dct, act) {
                ran = false;
                break;
            }
        }
        if ran {
            bw.flush();
            same(
                if wide {
                    "write_block avx512"
                } else {
                    "write_block avx2"
                },
                bw.out,
                want.out.clone(),
            )?;
        }
    }
    Ok(())
}

/// The decoder's MCU loop in its BMI build (x86_64) and in the plain build, over MCUs of four
/// blocks (two luma, two chroma) read from `data` as entropy-coded bytes with the standard
/// tables: after every MCU the same blocks, AC flags, DC predictions, result and reader state.
/// `Ok(false)`, having compared nothing, when this is not x86_64 or the CPU has no BMI.
pub fn mcu_decode(data: &[u8]) -> Result<bool, String> {
    #[cfg(target_arch = "x86_64")]
    {
        use crate::huff::{BitReader, DerivedTable, std_tables};
        let std = std_tables();
        let build = |t, is_dc| DerivedTable::build(t, is_dc).map_err(|_| "standard table");
        let (dl, dch) = (build(&std.dc[0], true)?, build(&std.dc[1], true)?);
        let (al, ach) = (build(&std.ac[0], false)?, build(&std.ac[1], false)?);
        let (dc, ac) = ([dl, dch, dch, dch], [al, ach, ach, ach]);
        let ks = [0usize, 0, 1, 2];
        let mut a = BitReader::new(data, 0);
        let mut b = a;
        let (mut la, mut lb) = ([0i32; 4], [0i32; 4]);
        // The skip path (an MCU outside a band): its own readers, walking the same bits.
        let (mut sa, mut sb) = (BitReader::new(data, 0), BitReader::new(data, 0));
        let (mut sla, mut slb) = ([0i32; 4], [0i32; 4]);
        let key = |r: &BitReader| {
            (
                r.pos,
                r.buf,
                r.bits_left,
                r.marker,
                r.insufficient,
                r.lost,
                r.eof,
                r.decoded,
            )
        };
        for mcu in 0..=2 * data.len() {
            let (mut ba, mut bb) = ([[0i16; 64]; 4], [[0i16; 64]; 4]);
            let (mut ha, mut hb) = ([false; 4], [false; 4]);
            let Some(ra) = simd::decode_mcu_fast_x86::<true>(
                &mut a,
                &dc,
                &ac,
                &mut la,
                &mut ba[..],
                &ks,
                Some(&mut ha[..]),
            ) else {
                return Ok(false);
            };
            let rb = crate::scan::decode_mcu_fast_impl::<true>(
                &mut b,
                &dc,
                &ac,
                &mut lb,
                &mut bb[..],
                &ks,
                Some(&mut hb[..]),
            );
            same(
                &format!("mcu_decode {mcu}"),
                (ra, ba, ha, la, key(&a)),
                (rb, bb, hb, lb, key(&b)),
            )?;
            let (mut xa, mut xb) = ([false; 4], [false; 4]);
            let Some(rsa) = simd::decode_mcu_fast_x86::<false>(
                &mut sa,
                &dc,
                &ac,
                &mut sla,
                &mut [],
                &ks,
                Some(&mut xa[..]),
            ) else {
                return Ok(false);
            };
            let rsb = crate::scan::decode_mcu_fast_impl::<false>(
                &mut sb,
                &dc,
                &ac,
                &mut slb,
                &mut [],
                &ks,
                Some(&mut xb[..]),
            );
            same(
                &format!("mcu_decode skip {mcu}"),
                (rsa, xa, sla, key(&sa)),
                (rsb, xb, slb, key(&sb)),
            )?;
            if !ra {
                break;
            }
        }
        Ok(true)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = data;
        Ok(false)
    }
}
