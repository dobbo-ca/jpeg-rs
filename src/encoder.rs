//! The baseline encoder, ported from jpeg-go's `internal/jpeg/encoder.go` and
//! `internal/codec/enc_*.go`, `fdct.go`, `enccolor.go`. The output is byte for byte what
//! `cjpeg -quality Q` and `tj3Compress8` write. The uniform-tile fast path is a one-entry cache
//! of the last solid-colour output in [`Encoder`] (identical bytes, no `WriteUniformScan` port).

use crate::Error;
use crate::huff::{HuffTable, std_tables};
use crate::image::{Gray, Rect, Rgba, Subsample, YCbCr};
use crate::limits::{Ledger, Limits, check, filled};
use crate::marker::ZIGZAG_TO_NATURAL;
use crate::simd::{self, Tier};
use std::io::Write;
use std::sync::atomic::AtomicBool;

/// Quality and chroma sampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EncodeOptions {
    /// 1..=100, libjpeg scaling; 0 means 75.
    pub quality: i32,
    /// RGBA input only: 4:4:4 instead of the default 4:2:0 of `jpeg_set_defaults`.
    pub chroma444: bool,
}

/// What the encoder takes: [`Rgba`] (alpha is ignored, as Go's `*image.NRGBA`), [`Gray`] or
/// [`YCbCr`] (4:2:0 or 4:4:4 at an even origin, encoded as is; any other sampling goes through
/// RGBA with Go's `image/draw` conversion).
#[derive(Clone, Copy)]
pub enum EncodeImage<'a> {
    Gray(&'a Gray),
    Rgba(&'a Rgba),
    YCbCr(&'a YCbCr),
}

impl<'a> From<&'a crate::Image> for EncodeImage<'a> {
    fn from(i: &'a crate::Image) -> Self {
        match i {
            crate::Image::Gray(g) => EncodeImage::Gray(g),
            crate::Image::Rgba(r) => EncodeImage::Rgba(r),
            crate::Image::YCbCr(y) => EncodeImage::YCbCr(y),
        }
    }
}

/// Encodes `m` and returns the JPEG stream. `limits` bound the frame's pixels and the memory the
/// call holds (strips, the RGBA copy of a YCbCr image `image/draw` converts, the output).
///
/// # Errors
///
/// [`Error::BadImage`] for an image the encoder cannot take (zero size, a short plane);
/// [`Error::PixelLimit`] when the frame is over `limits.max_pixels`; [`Error::AllocLimit`] when
/// the strips, the copy and the output would pass `limits.max_alloc` (the output vector is
/// charged to the same ledger, and a row is refused before its worst case is reserved);
/// [`Error::OutOfMemory`]; and [`Error::Cancelled`] when `cancel` is set (every MCU row, and
/// every 16 rows of the YCbCr-to-RGBA copy). Serial: spawns no thread, so `limits.max_threads`
/// has no effect.
pub fn encode(
    m: EncodeImage,
    o: &EncodeOptions,
    limits: &Limits,
    cancel: &AtomicBool,
) -> Result<Vec<u8>, Error> {
    Encoder {
        one_shot: true,
        ..Encoder::default()
    }
    .encode(m, o, limits, cancel)
}

/// Solid-colour outputs above this size are not cached.
const UNIFORM_CACHE_MAX: usize = 32 << 10;

/// What a solid-colour image's output depends on (Go's `uniformKey`); `kind` keeps equal samples
/// of different colour spaces apart.
#[derive(Clone, Copy, PartialEq, Eq)]
struct UniformKey {
    kind: u8,
    c: [u8; 3],
    w: usize,
    h: usize,
    quality: i32,
    chroma444: bool,
}

/// Go's reusable `Encoder`: keeps the quantization tables of the last quality and the output of
/// the last solid-colour image, so repeated tiles skip their set-up (and a repeated solid tile
/// its whole encode). One entry each, so nothing grows. The output is the same as [`encode`]'s.
#[derive(Default)]
pub struct Encoder {
    quant: Option<(i32, Quant, Quant)>,
    uniform: Option<(UniformKey, Vec<u8>)>,
    /// The free function's one-shot use: no solid-colour scan, no cache.
    one_shot: bool,
}

impl Encoder {
    /// [`encode`] on this encoder's cached state.
    ///
    /// # Errors
    ///
    /// As [`encode`]: [`Error::BadImage`], `limits.max_pixels` ([`Error::PixelLimit`]),
    /// `limits.max_alloc` ([`Error::AllocLimit`]), [`Error::OutOfMemory`] and
    /// [`Error::Cancelled`]. A cached solid-colour output is still charged to `limits.max_alloc`.
    /// Serial: spawns no thread.
    pub fn encode(
        &mut self,
        m: EncodeImage,
        o: &EncodeOptions,
        limits: &Limits,
        cancel: &AtomicBool,
    ) -> Result<Vec<u8>, Error> {
        let quality = if o.quality != 0 { o.quality } else { 75 };
        let key = if self.one_shot {
            None
        } else {
            uniform_key(m, quality, o.chroma444, limits)
        };
        let hit = match (key, &self.uniform) {
            (Some(k), Some((ck, bytes))) if *ck == k => Some(bytes),
            _ => None,
        };
        if let Some(bytes) = hit {
            check(cancel)?;
            limits.check_bytes(bytes.len() as u64)?;
            let mut out = Vec::new();
            out.try_reserve_exact(bytes.len())
                .map_err(|_| Error::OutOfMemory)?;
            out.extend_from_slice(bytes);
            return Ok(out);
        }
        let (q, lq, cq) = match self.quant.take() {
            Some(t) if t.0 == quality => t,
            _ => {
                let (l, c) = quant_tables(quality);
                (quality, l, c)
            }
        };
        let ledger = Ledger::new(limits);
        let mut out = Vec::new();
        let st = State {
            limits,
            ledger: &ledger,
            cancel,
        };
        let r = st.encode(&mut out, m, quality, o.chroma444, &lq, &cq);
        self.quant = Some((q, lq, cq));
        r?;
        if let Some(k) = key {
            // The copy counts against the budget while both are alive.
            let held = (out.capacity() as u64).saturating_add(out.len() as u64);
            let fits = ledger.used().saturating_add(held) <= limits.max_alloc;
            let mut keep = Vec::new();
            if out.len() <= UNIFORM_CACHE_MAX && fits && keep.try_reserve_exact(out.len()).is_ok() {
                keep.extend_from_slice(&out);
                self.uniform = Some((k, keep));
            }
        }
        Ok(out)
    }
}

/// The bytes of `rows` rows of `width` bytes at `stride` when all are one repeated `bpp`-byte
/// pixel. `pix` holds them (`need_plane`).
fn solid(pix: &[u8], stride: usize, width: usize, rows: usize, bpp: usize) -> Option<&[u8]> {
    let first = pix.get(..bpp)?;
    for y in 0..rows {
        let row = pix.get(y * stride..)?.get(..width)?;
        // Periodic with period `bpp` and starting with `first`.
        if &row[..bpp] != first || row[bpp..] != row[..width - bpp] {
            return None;
        }
    }
    Some(first)
}

/// `Some` when `m` is one solid colour that the encode accepts (the checks that fail go on to the
/// general path, which reports them); the caller's `chroma444` counts for RGBA only.
fn uniform_key(
    m: EncodeImage,
    quality: i32,
    chroma444: bool,
    limits: &Limits,
) -> Option<UniformKey> {
    let rect = match m {
        EncodeImage::Gray(g) => g.rect,
        EncodeImage::Rgba(r) => r.rect,
        EncodeImage::YCbCr(y) => y.rect,
    };
    let (w, h) = rect_dims(rect).ok()?;
    if w == 0 || h == 0 || limits.check_pixels(w, h).is_err() {
        return None;
    }
    let (kind, c, chroma444) = match m {
        EncodeImage::Gray(g) => {
            need_plane(&g.pix, g.stride, w, h).ok()?;
            (0, [solid(&g.pix, g.stride, w, h, 1)?[0], 0, 0], false)
        }
        EncodeImage::Rgba(r) => {
            need_plane(&r.pix, r.stride, w * 4, h).ok()?;
            let p = solid(&r.pix, r.stride, w * 4, h, 4)?;
            (2, [p[0], p[1], p[2]], chroma444)
        }
        EncodeImage::YCbCr(img) => {
            if !direct_ycbcr(img) {
                return None;
            }
            let c444 = img.ratio == Subsample::S444;
            let (cw, ch) = if c444 {
                (w, h)
            } else {
                (w.div_ceil(2), h.div_ceil(2))
            };
            let y = img.y.get(img.y_offset(rect.min_x, rect.min_y)..)?;
            let co = img.c_offset(rect.min_x, rect.min_y);
            let (cb, cr) = (img.cb.get(co..)?, img.cr.get(co..)?);
            need_plane(y, img.y_stride, w, h).ok()?;
            need_plane(cb, img.c_stride, cw, ch).ok()?;
            need_plane(cr, img.c_stride, cw, ch).ok()?;
            let c = [
                solid(y, img.y_stride, w, h, 1)?[0],
                solid(cb, img.c_stride, cw, ch, 1)?[0],
                solid(cr, img.c_stride, cw, ch, 1)?[0],
            ];
            (1, c, c444)
        }
    };
    Some(UniformKey {
        kind,
        c,
        w,
        h,
        quality,
        chroma444,
    })
}

/// [`encode`] into `w` with one `write_all`, as Go's `Encode`.
///
/// # Errors
///
/// Those of [`encode`] (`limits.max_pixels`, `limits.max_alloc`, [`Error::Cancelled`],
/// [`Error::BadImage`], [`Error::OutOfMemory`]), and [`Error::Io`] with the kind of the error
/// `w` returned. The whole stream is held in memory (and charged to `max_alloc`) before the one
/// write. Serial: spawns no thread.
pub fn encode_to<W: Write>(
    w: &mut W,
    m: EncodeImage,
    o: &EncodeOptions,
    limits: &Limits,
    cancel: &AtomicBool,
) -> Result<(), Error> {
    let b = encode(m, o, limits, cancel)?;
    w.write_all(&b).map_err(|e| Error::Io(e.kind()))
}

// ---- quantization tables (jcparam.c) ----

/// Annex K.1 base tables, natural (row-major) order.
const STD_LUMA_QUANT: [i32; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
    92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];

const STD_CHROMA_QUANT: [i32; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
];

/// libjpeg's scale factor for quality 1..100 (`jpeg_quality_scaling`).
fn quality_scaling(quality: i32) -> i32 {
    let q = quality.clamp(1, 100);
    if q < 50 { 5000 / q } else { 200 - q * 2 }
}

/// `jpeg_add_quant_table`: scales `base` by quality, natural order. `force_baseline` is always
/// on, so entries clamp to 1..=255.
fn scale_quant_table(base: &[i32; 64], quality: i32) -> [u8; 64] {
    let scale = quality_scaling(quality);
    let mut out = [0u8; 64];
    for (i, &b) in base.iter().enumerate() {
        out[i] = ((b * scale + 50) / 100).clamp(1, 255) as u8;
    }
    out
}

/// A quantization table in natural order, with libjpeg-turbo's reciprocal tables for the divisors
/// `8q` (`compute_reciprocal`, as its SIMD quantizer uses them): `floor((|x| + 4q) / 8q)` is
/// `((((|x| + corr) * recip) >> 16) * scale) >> 16`, exact for 16-bit `|x| + corr`.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))] // the tables are the AVX2 kernel's
pub(crate) struct Quant {
    pub(crate) q: [i32; 64],
    pub(crate) recip: [u16; 64],
    pub(crate) corr: [u16; 64],
    pub(crate) scale: [u16; 64],
    /// The three tables laid out for two blocks at once: per 16 coefficients, the 16 twice.
    pub(crate) pair_recip: [u16; 128],
    pub(crate) pair_corr: [u16; 128],
    pub(crate) pair_scale: [u16; 128],
}

impl Quant {
    pub(crate) fn new(q: &[i32; 64]) -> Quant {
        let mut t = Quant {
            q: *q,
            recip: [0; 64],
            corr: [0; 64],
            scale: [0; 64],
            pair_recip: [0; 128],
            pair_corr: [0; 128],
            pair_scale: [0; 128],
        };
        for i in 0..64 {
            let d = 8 * q[i].max(1) as u32;
            let mut r = 16 + (31 - d.leading_zeros());
            let (mut fq, fr) = ((1u32 << r) / d, (1u32 << r) % d);
            let mut c = d / 2;
            if fr == 0 {
                // A power of two: fq is a bit too large for 16 bits.
                fq >>= 1;
                r -= 1;
            } else if fr <= d / 2 {
                c += 1;
            } else {
                fq += 1;
            }
            t.recip[i] = fq as u16;
            t.corr[i] = c as u16;
            t.scale[i] = (1u32 << (32 - r)) as u16;
        }
        for k in 0..4 {
            for half in 0..2 {
                let (src, dst) = (
                    16 * k..16 * k + 16,
                    32 * k + 16 * half..32 * k + 16 * half + 16,
                );
                t.pair_recip[dst.clone()].copy_from_slice(&t.recip[src.clone()]);
                t.pair_corr[dst.clone()].copy_from_slice(&t.corr[src.clone()]);
                t.pair_scale[dst].copy_from_slice(&t.scale[src]);
            }
        }
        t
    }
}

fn quant_natural(t: &[u8; 64]) -> [i32; 64] {
    let mut out = [0i32; 64];
    for (i, &v) in t.iter().enumerate() {
        out[i] = v as i32;
    }
    out
}

// ---- Huffman codes ----

#[derive(Clone, Copy, Default)]
pub(crate) struct HuffCode {
    pub(crate) code: u16,
    pub(crate) size: u8,
}

/// A table's canonical codes indexed by symbol value.
fn code_array(t: &HuffTable) -> [HuffCode; 256] {
    let mut out = [HuffCode::default(); 256];
    for k in 0..t.n {
        out[t.values[k] as usize] = HuffCode {
            code: t.code[k],
            size: t.length[k],
        };
    }
    out
}

pub(crate) struct StdCodes {
    pub(crate) dc_luma: [HuffCode; 256],
    pub(crate) ac_luma: [HuffCode; 256],
    pub(crate) dc_chroma: [HuffCode; 256],
    pub(crate) ac_chroma: [HuffCode; 256],
}

pub(crate) fn std_codes() -> &'static StdCodes {
    static C: std::sync::OnceLock<StdCodes> = std::sync::OnceLock::new();
    C.get_or_init(|| {
        let t = std_tables();
        StdCodes {
            dc_luma: code_array(&t.dc[0]),
            ac_luma: code_array(&t.ac[0]),
            dc_chroma: code_array(&t.dc[1]),
            ac_chroma: code_array(&t.ac[1]),
        }
    })
}

// ---- markers (jcmarker.c) ----

fn put16(out: &mut Vec<u8>, v: usize) {
    out.push((v >> 8) as u8);
    out.push(v as u8);
}

/// `emit_dqt`: one DQT marker, zigzag order.
fn write_dqt(out: &mut Vec<u8>, index: u8, table: &[u8; 64]) {
    out.extend_from_slice(&[0xff, 0xdb]);
    put16(out, 2 + 1 + 64);
    out.push(index);
    for &n in &ZIGZAG_TO_NATURAL {
        out.push(table[n]);
    }
}

/// `emit_dht`: one DHT marker.
fn write_dht(out: &mut Vec<u8>, index: u8, is_ac: bool, t: &HuffTable) {
    out.extend_from_slice(&[0xff, 0xc4]);
    let length: usize = t.bits.iter().map(|&b| b as usize).sum();
    put16(out, 2 + 1 + 16 + length);
    out.push(if is_ac { index + 0x10 } else { index });
    out.extend_from_slice(&t.bits);
    out.extend_from_slice(&t.values[..length]);
}

/// SOI, APP0 JFIF, DQT, SOF0, DHT and SOS in cjpeg's order: JFIF 1.01, density 0/1/1,
/// `force_baseline`, standard Huffman, no restarts. Sizes are written as 16 bits, truncated like
/// Go's.
fn write_header(
    out: &mut Vec<u8>,
    width: usize,
    height: usize,
    quality: i32,
    gray: bool,
    chroma444: bool,
) {
    out.extend_from_slice(&[0xff, 0xd8]);
    out.extend_from_slice(&[0xff, 0xe0]);
    put16(out, 16);
    out.extend_from_slice(b"JFIF\0");
    out.extend_from_slice(&[0x01, 0x01, 0x00]);
    put16(out, 1);
    put16(out, 1);
    out.extend_from_slice(&[0x00, 0x00]);
    write_dqt(out, 0, &scale_quant_table(&STD_LUMA_QUANT, quality));
    if !gray {
        write_dqt(out, 1, &scale_quant_table(&STD_CHROMA_QUANT, quality));
    }
    // (id, h, v, quant table, dc table, ac table)
    let comps: &[(u8, u8, u8, u8, u8, u8)] = if gray {
        &[(1, 1, 1, 0, 0, 0)]
    } else if chroma444 {
        &[(1, 1, 1, 0, 0, 0), (2, 1, 1, 1, 1, 1), (3, 1, 1, 1, 1, 1)]
    } else {
        &[(1, 2, 2, 0, 0, 0), (2, 1, 1, 1, 1, 1), (3, 1, 1, 1, 1, 1)]
    };
    out.extend_from_slice(&[0xff, 0xc0]);
    put16(out, 2 + 6 + 3 * comps.len());
    out.push(8);
    put16(out, height);
    put16(out, width);
    out.push(comps.len() as u8);
    for &(id, h, v, q, _, _) in comps {
        out.extend_from_slice(&[id, h << 4 | v, q]);
    }
    let t = std_tables();
    write_dht(out, 0, false, &t.dc[0]);
    write_dht(out, 0, true, &t.ac[0]);
    if !gray {
        write_dht(out, 1, false, &t.dc[1]);
        write_dht(out, 1, true, &t.ac[1]);
    }
    out.extend_from_slice(&[0xff, 0xda]);
    put16(out, 2 + 1 + 2 * comps.len() + 3);
    out.push(comps.len() as u8);
    for &(id, _, _, _, d, a) in comps {
        out.extend_from_slice(&[id, d << 4 | a]);
    }
    out.extend_from_slice(&[0x00, 0x3f, 0x00]);
}

// ---- FDCT and quantization (jfdctint.c, jcdctmgr.c) ----

const FIX_0_298631336: i32 = 2446;
const FIX_0_390180644: i32 = 3196;
const FIX_0_541196100: i32 = 4433;
const FIX_0_765366865: i32 = 6270;
const FIX_0_899976223: i32 = 7373;
const FIX_1_175875602: i32 = 9633;
const FIX_1_501321110: i32 = 12299;
const FIX_1_847759065: i32 = 15137;
const FIX_1_961570560: i32 = 16069;
const FIX_2_053119869: i32 = 16819;
const FIX_2_562915447: i32 = 20995;
const FIX_3_072711026: i32 = 25172;

const FDCT_CONST_BITS: u32 = 13;
const FDCT_PASS1_BITS: u32 = 2;

fn fdct_descale(x: i32, n: u32) -> i32 {
    (x + (1 << (n - 1))) >> n
}

/// turbo's `_jpeg_fdct_islow`, in place: pixel values minus 128 in, the true DCT scaled up by 8
/// out, both natural (row-major) order.
fn fdct_islow(d: &mut [i32; 64]) {
    for pass in 0..2 {
        // Pass 1 is rows, pass 2 columns.
        for k in 0..8 {
            let (base, step) = if pass == 0 { (k * 8, 1) } else { (k, 8) };
            let at = |j: usize| base + j * step;
            let (s0, s1, s2, s3) = (d[at(0)], d[at(1)], d[at(2)], d[at(3)]);
            let (s4, s5, s6, s7) = (d[at(4)], d[at(5)], d[at(6)], d[at(7)]);
            let tmp0 = s0 + s7;
            let tmp7 = s0 - s7;
            let tmp1 = s1 + s6;
            let tmp6 = s1 - s6;
            let tmp2 = s2 + s5;
            let tmp5 = s2 - s5;
            let tmp3 = s3 + s4;
            let tmp4 = s3 - s4;

            let tmp10 = tmp0 + tmp3;
            let tmp13 = tmp0 - tmp3;
            let tmp11 = tmp1 + tmp2;
            let tmp12 = tmp1 - tmp2;

            let (shift, s0o, s4o) = if pass == 0 {
                (
                    FDCT_CONST_BITS - FDCT_PASS1_BITS,
                    (tmp10 + tmp11) << FDCT_PASS1_BITS,
                    (tmp10 - tmp11) << FDCT_PASS1_BITS,
                )
            } else {
                (
                    FDCT_CONST_BITS + FDCT_PASS1_BITS,
                    fdct_descale(tmp10 + tmp11, FDCT_PASS1_BITS),
                    fdct_descale(tmp10 - tmp11, FDCT_PASS1_BITS),
                )
            };
            d[at(0)] = s0o;
            d[at(4)] = s4o;
            let z1 = (tmp12 + tmp13) * FIX_0_541196100;
            d[at(2)] = fdct_descale(z1 + tmp13 * FIX_0_765366865, shift);
            d[at(6)] = fdct_descale(z1 - tmp12 * FIX_1_847759065, shift);

            let mut z1o = tmp4 + tmp7;
            let mut z2o = tmp5 + tmp6;
            let mut z3o = tmp4 + tmp6;
            let mut z4o = tmp5 + tmp7;
            let z5o = (z3o + z4o) * FIX_1_175875602;
            let t4 = tmp4 * FIX_0_298631336;
            let t5 = tmp5 * FIX_2_053119869;
            let t6 = tmp6 * FIX_3_072711026;
            let t7 = tmp7 * FIX_1_501321110;
            z1o *= -FIX_0_899976223;
            z2o *= -FIX_2_562915447;
            z3o = -z3o * FIX_1_961570560 + z5o;
            z4o = -z4o * FIX_0_390180644 + z5o;
            d[at(7)] = fdct_descale(t4 + z1o + z3o, shift);
            d[at(5)] = fdct_descale(t5 + z2o + z4o, shift);
            d[at(3)] = fdct_descale(t6 + z2o + z3o, shift);
            d[at(1)] = fdct_descale(t7 + z1o + z4o, shift);
        }
    }
}

/// jcdctmgr.c's reciprocal-multiply quantizer, equal to `sign(x) * ((|x| + 4q) / (8q))`.
fn quantize_islow(x: i32, q: i32) -> i32 {
    if x < 0 {
        -((-x + 4 * q) / (8 * q))
    } else {
        (x + 4 * q) / (8 * q)
    }
}

/// FDCT then quantize 8 rows of 8 pixels at `pix[off..]`, `stride` apart, level-shifted by 128.
pub(crate) fn fdct_quant_rows(
    pix: &[u8],
    off: usize,
    stride: usize,
    quant: &Quant,
    coef: &mut [i16; 64],
) {
    let mut s = [0i32; 64];
    for r in 0..8 {
        for c in 0..8 {
            s[r * 8 + c] = pix[off + r * stride + c] as i32 - 128;
        }
    }
    fdct_islow(&mut s);
    for i in 0..64 {
        coef[i] = quantize_islow(s[i], quant.q[i]) as i16;
    }
}

/// All 64 samples equal: that value.
fn flat_block(pix: &[u8], off: usize, stride: usize) -> Option<u8> {
    let v = pix[off];
    let want = u64::from(v) * 0x0101_0101_0101_0101;
    for r in 0..8 {
        let p = &pix[off + r * stride..off + r * stride + 8];
        let row = u64::from_le_bytes([p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]]);
        if row != want {
            return None;
        }
    }
    Some(v)
}

/// A flat block's FDCT DC is 64 (v - 128).
fn flat_dc(v: u8, q: i32) -> i16 {
    quantize_islow((v as i32 - 128) * 64, q) as i16
}

// ---- colour (jccolor.c, jcsample.c) ----

/// `rgb_ycc_convert`'s weights.
fn rgb_to_ycbcr(r: i32, g: i32, b: i32) -> (u8, u8, u8) {
    let y = (19595 * r + 38470 * g + 7471 * b + 32768) >> 16;
    let cb = (-11059 * r - 21709 * g + 32768 * b + (128 << 16) + 32767) >> 16;
    let cr = (32768 * r - 27439 * g - 5329 * b + (128 << 16) + 32767) >> 16;
    (y as u8, cb as u8, cr as u8)
}

/// `rgb_ycc_convert` over `y.len()` RGB(A)-ordered 4-byte pixels, alpha ignored.
fn rgba_to_ycbcr_row(src: &[u8], y: &mut [u8], cb: &mut [u8], cr: &mut [u8]) {
    rgba_to_ycbcr_row_with(simd::tier(), src, y, cb, cr);
}

/// [`rgba_to_ycbcr_row`] in tier `t`: the SIMD kernel does the first pixels, the scalar loop the rest.
pub(crate) fn rgba_to_ycbcr_row_with(
    t: Tier,
    src: &[u8],
    y: &mut [u8],
    cb: &mut [u8],
    cr: &mut [u8],
) {
    let done = simd::rgba_to_ycbcr_row(t, src, y, cb, cr);
    rgba_to_ycbcr_row_scalar(
        &src[4 * done..],
        &mut y[done..],
        &mut cb[done..],
        &mut cr[done..],
    );
}

/// The scalar reference of [`rgba_to_ycbcr_row`].
fn rgba_to_ycbcr_row_scalar(src: &[u8], y: &mut [u8], cb: &mut [u8], cr: &mut [u8]) {
    for i in 0..y.len() {
        let p = &src[4 * i..4 * i + 4];
        (y[i], cb[i], cr[i]) = rgb_to_ycbcr(p[0] as i32, p[1] as i32, p[2] as i32);
    }
}

/// `h2v2_downsample` for one output row: `(2x2 sum + bias) >> 2`, bias 1, 2, 1, 2. `r0` and `r1`
/// hold `2 * dst.len()` samples, already edge-padded.
fn downsample_h2v2_row(dst: &mut [u8], r0: &[u8], r1: &[u8]) {
    downsample_h2v2_row_with(simd::tier(), dst, r0, r1);
}

/// [`downsample_h2v2_row`] in tier `t`.
pub(crate) fn downsample_h2v2_row_with(t: Tier, dst: &mut [u8], r0: &[u8], r1: &[u8]) {
    let done = simd::downsample_h2v2_row(t, dst, r0, r1);
    downsample_h2v2_row_scalar(&mut dst[done..], &r0[2 * done..], &r1[2 * done..]);
}

/// The scalar reference of [`downsample_h2v2_row`]; `dst` starts at an even output.
fn downsample_h2v2_row_scalar(dst: &mut [u8], r0: &[u8], r1: &[u8]) {
    let mut bias = 1i32;
    for i in 0..dst.len() {
        let s = r0[2 * i] as i32 + r0[2 * i + 1] as i32 + r1[2 * i] as i32 + r1[2 * i + 1] as i32;
        dst[i] = ((s + bias) >> 2) as u8;
        bias ^= 3;
    }
}

// ---- entropy coding (jchuff.c) ----

/// Packs Huffman-coded bits MSB first, stuffs 0xFF data bytes with 0x00, and pads the final byte
/// with 1-bits. `acc` holds the pending bits at its low end, `64 - free` of them; bits above are
/// stale and shifted out by the next spill (jchuff.c `put_buffer`, and Go's `bitWriter`).
pub(crate) struct BitWriter {
    pub(crate) out: Vec<u8>,
    acc: u64,
    /// 64 when nothing is pending.
    free: i32,
}

/// A full 64-bit word goes out when `code` of `size` bits does not fit: `acc`'s pending bits, then
/// the leading bits of `code`; `free` is already reduced by `size` and negative. Returns the new
/// `acc` (the code) and `free`. A word with no 0xFF byte is written as it is.
#[inline]
fn spill(out: &mut Vec<u8>, acc: u64, free: i32, code: u64, size: i32) -> (u64, i32) {
    let v = acc << ((size + free) & 63) | code >> (-free & 63);
    let y = !v;
    if y.wrapping_sub(0x0101_0101_0101_0101) & !y & 0x8080_8080_8080_8080 == 0 {
        out.extend_from_slice(&v.to_be_bytes());
    } else {
        for b in v.to_be_bytes() {
            out.push(b);
            if b == 0xff {
                out.push(0);
            }
        }
    }
    (code, free + 64)
}

impl BitWriter {
    pub(crate) fn new(out: Vec<u8>) -> BitWriter {
        BitWriter {
            out,
            acc: 0,
            free: 64,
        }
    }

    #[inline(always)]
    fn put(&mut self, code: u64, size: u32) {
        let size = size as i32;
        let free = self.free - size;
        if free >= 0 {
            self.acc = self.acc << (size & 63) | code;
            self.free = free;
        } else {
            (self.acc, self.free) = spill(&mut self.out, self.acc, free, code, size);
        }
    }

    pub(crate) fn flush(&mut self) {
        let mut n = (64 - self.free) as u32;
        while n >= 8 {
            n -= 8;
            self.byte((self.acc >> n) as u8);
        }
        if n > 0 {
            let pad = 8 - n;
            self.byte(((self.acc << pad) | ((1u64 << pad) - 1)) as u8);
        }
        self.acc = 0;
        self.free = 64;
    }

    fn byte(&mut self, b: u8) {
        self.out.push(b);
        if b == 0xff {
            self.out.push(0);
        }
    }
}

/// The low n bits; baseline value sizes are at most 16.
#[inline(always)]
fn low_mask(n: u32) -> u64 {
    (1u64 << n.min(16)) - 1
}

/// Codes a block whose AC coefficients are all zero.
fn write_dc_only(
    bw: &mut BitWriter,
    dc: i32,
    last_dc: &mut i32,
    dct: &[HuffCode; 256],
    act: &[HuffCode; 256],
) {
    let diff = dc - *last_dc;
    *last_dc = dc;
    let a = diff >> 31;
    let s = 32 - (((diff ^ a) - a) as u32).leading_zeros();
    let (c, e) = (dct[s as usize & 255], act[0]);
    let code = ((c.code as u64) << s | (((diff + a) as u32) as u64 & low_mask(s))) << e.size
        | e.code as u64;
    bw.put(code, c.size as u32 + s + e.size as u32);
}

/// `ZIGZAG_TO_NATURAL` as bytes, for the coding loop.
const ZZ_NAT: [u8; 64] = {
    let mut t = [0u8; 64];
    let mut k = 0;
    while k < 64 {
        t[k] = ZIGZAG_TO_NATURAL[k] as u8;
        k += 1;
    }
    t
};

/// The source coefficient (0..64) of each zigzag position, as the word selectors of the wide
/// kernel's two `permutex2var`: positions 0..32 and 32..64.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub(crate) const ZZ_PERM: [[i16; 32]; 2] = {
    let mut t = [[0i16; 32]; 2];
    let mut k = 0;
    while k < 64 {
        t[k / 32][k % 32] = ZIGZAG_TO_NATURAL[k] as i16;
        k += 1;
    }
    t
};

/// For each byte of a natural-order mask, the zigzag-order mask of its bits: eight lookups turn a
/// block's nonzero mask into the order the coder walks (`huffprep`'s job in jchuff-neon.c).
static MASK_PERM: [[u64; 256]; 8] = {
    let mut nat_to_zz = [0usize; 64];
    let mut k = 0;
    while k < 64 {
        nat_to_zz[ZIGZAG_TO_NATURAL[k]] = k;
        k += 1;
    }
    let mut t = [[0u64; 256]; 8];
    let mut byte = 0;
    while byte < 8 {
        let mut v = 0;
        while v < 256 {
            let mut m = 0u64;
            let mut bit = 0;
            while bit < 8 {
                if v >> bit & 1 != 0 {
                    m |= 1u64 << nat_to_zz[byte * 8 + bit];
                }
                bit += 1;
            }
            t[byte][v] = m;
            v += 1;
        }
        byte += 1;
    }
    t
};

/// The zigzag-order mask of a natural-order mask.
#[inline(always)]
pub(crate) fn zigzag_mask(nat: u64) -> u64 {
    let t = &MASK_PERM;
    t[0][nat as u8 as usize]
        | t[1][(nat >> 8) as u8 as usize]
        | t[2][(nat >> 16) as u8 as usize]
        | t[3][(nat >> 24) as u8 as usize]
        | t[4][(nat >> 32) as u8 as usize]
        | t[5][(nat >> 40) as u8 as usize]
        | t[6][(nat >> 48) as u8 as usize]
        | t[7][(nat >> 56) as u8 as usize]
}

/// The value bits (`v`, or `v - 1` when negative, masked to its bit length) and the bit length of a
/// nonzero coefficient.
#[inline(always)]
pub(crate) fn prep_coef(v: i32) -> (u16, u16) {
    let a = v >> 31;
    let bits = 32 - (((v ^ a) - a) as u32).leading_zeros();
    (((v + a) as u64 & low_mask(bits)) as u16, bits as u16)
}

/// The scalar prep of a block, natural order: bit `n` of the result is set when coefficient `n` is
/// nonzero, and then `vals[n]` and `nb[n]` are its [`prep_coef`]. Other entries are left as they
/// were.
pub(crate) fn huff_prep_scalar(blk: &[i16; 64], vals: &mut [u16; 64], nb: &mut [u16; 64]) -> u64 {
    let mut mask = 0u64;
    for n in 0..64 {
        let v = blk[n] as i32;
        if v != 0 {
            (vals[n], nb[n]) = prep_coef(v);
            mask |= 1 << n;
        }
    }
    mask
}

/// Codes a block from its prep: the DC difference against `last_dc`, then each nonzero AC
/// coefficient in zigzag order (`mask`, the zigzag mask without bit 0) with ZRL and EOB. `vals` and
/// `nb` are in natural order, or in zigzag order when `NATURAL` is false (a wide kernel permutes
/// them). Inlined into the kernels, so the SIMD ones are built with BMI.
#[inline(always)]
pub(crate) fn code_block<const NATURAL: bool>(
    bw: &mut BitWriter,
    dc: i16,
    last_dc: &mut i32,
    dct: &[HuffCode; 256],
    act: &[HuffCode; 256],
    mut mask: u64,
    vals: &[u16; 64],
    nb: &[u16; 64],
) {
    let (mut acc, mut free) = (bw.acc, bw.free);
    macro_rules! put {
        ($code:expr, $size:expr) => {{
            let (code, size) = ($code as u64, $size as i32);
            free -= size;
            if free >= 0 {
                acc = acc << (size & 63) | code;
            } else {
                (acc, free) = spill(&mut bw.out, acc, free, code, size);
            }
        }};
    }

    let diff = dc as i32 - *last_dc;
    *last_dc = dc as i32;
    let a = diff >> 31;
    let s = 32 - (((diff ^ a) - a) as u32).leading_zeros();
    let c = dct[s as usize & 255];
    put!(
        (c.code as u64) << s | (((diff + a) as u32) as u64 & low_mask(s)),
        c.size as u32 + s
    );

    let mut last = 0usize;
    while mask != 0 {
        let k = mask.trailing_zeros() as usize;
        mask &= mask - 1;
        let n = if NATURAL {
            ZZ_NAT[k & 63] as usize & 63
        } else {
            k & 63
        };
        let mut run = k - last - 1;
        last = k;
        while run >= 16 {
            let c = act[0xf0];
            put!(c.code, c.size);
            run -= 16;
        }
        let s = nb[n] as u32;
        let c = act[((run << 4) as u8 | s as u8) as usize];
        put!((c.code as u64) << s | vals[n] as u64, c.size as u32 + s);
    }
    if last != 63 {
        let c = act[0];
        put!(c.code, c.size);
    }
    bw.acc = acc;
    bw.free = free;
}

/// Entropy-codes one quantized block, natural order, in tier `t`: DC difference against
/// `last_dc`, then AC in zigzag order with ZRL and EOB.
#[inline]
fn write_block(
    t: Tier,
    bw: &mut BitWriter,
    blk: &[i16; 64],
    last_dc: &mut i32,
    dct: &[HuffCode; 256],
    act: &[HuffCode; 256],
) {
    if !simd::write_block(t, bw, blk, last_dc, dct, act) {
        write_block_scalar(bw, blk, last_dc, dct, act);
    }
}

/// The scalar reference of [`write_block`].
pub(crate) fn write_block_scalar(
    bw: &mut BitWriter,
    blk: &[i16; 64],
    last_dc: &mut i32,
    dct: &[HuffCode; 256],
    act: &[HuffCode; 256],
) {
    let (mut vals, mut nb) = ([0u16; 64], [0u16; 64]);
    let nat = huff_prep_scalar(blk, &mut vals, &mut nb);
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

// ---- the scan (jccoefct.c) ----

/// One component's samples, pre-block-padding. `w`, `h` are the true (unpadded) sample
/// dimensions; reads past them replicate the edge.
#[derive(Clone, Copy)]
struct EncPlane<'a> {
    pix: &'a [u8],
    stride: usize,
    w: usize,
    h: usize,
}

impl<'a> EncPlane<'a> {
    fn at_row(&self, y: usize) -> EncPlane<'a> {
        EncPlane {
            pix: self.pix.get(y * self.stride..).unwrap_or(&[]),
            stride: self.stride,
            w: self.w,
            h: self.h.saturating_sub(y),
        }
    }
}

/// One scan component: MCU sampling factors and entropy tables.
struct EncComponent<'a> {
    hsamp: usize,
    vsamp: usize,
    quant: &'a Quant,
    dc: &'a [HuffCode; 256],
    ac: &'a [HuffCode; 256],
}

const MAX_MCU_BLOCKS: usize = 10;
/// Bounds one coded block, stuffing included (jchuff.c BUFSIZE, DCTSIZE2 * 8).
const MAX_BLOCK_BYTES: usize = 64 * 8;

/// The most the bit writer's `flush` adds (8 bytes, each stuffed) and EOI.
const FLUSH_AND_EOI: usize = 16 + 2;

/// Makes room in `out` for `extra` more bytes so that its capacity never passes `budget`. It
/// grows to `max(2 * cap, len + extra)` but at most what the budget leaves besides the old buffer,
/// which a move keeps alive beside the new one. Fails, growing nothing, when `len + extra` does
/// not fit that.
fn reserve_capped(out: &mut Vec<u8>, extra: usize, budget: u64) -> Result<(), Error> {
    let (len, cap) = (out.len(), out.capacity());
    let need = len as u64 + extra as u64;
    if need <= cap as u64 {
        return Ok(());
    }
    let room = budget.saturating_sub(cap as u64);
    if need > room {
        return Err(Error::AllocLimit {
            got: need + cap as u64,
            max: budget,
        });
    }
    let new_cap = need.max(2 * cap as u64).min(room);
    let new_cap = usize::try_from(new_cap).map_err(|_| Error::OutOfMemory)?;
    out.try_reserve_exact(new_cap - len)
        .map_err(|_| Error::OutOfMemory)
}

struct Scan<'a> {
    bw: BitWriter,
    tier: Tier,
    comps: Vec<EncComponent<'a>>,
    last_dc: [i32; 4],
    w_blocks: [usize; 4],
    h_blocks: [usize; 4],
    mcu_x: usize,
    mcu_y: usize,
    row_bound: usize,
    blk: [[i16; 64]; MAX_MCU_BLOCKS],
    dc_only: [bool; MAX_MCU_BLOCKS],
    /// The last block's DC, for the dummy blocks of the next MCU.
    prev_dc: i16,
    edge: [u8; 64],
}

impl<'a> Scan<'a> {
    /// Begins `comps`' scan; `sizes` are each plane's true `(w, h)`.
    fn start(
        out: Vec<u8>,
        w: usize,
        h: usize,
        comps: Vec<EncComponent<'a>>,
        sizes: &[(usize, usize)],
    ) -> Scan<'a> {
        let (mut hmax, mut vmax) = (1, 1);
        let mut blocks = 0;
        for c in &comps {
            hmax = hmax.max(c.hsamp);
            vmax = vmax.max(c.vsamp);
            blocks += c.hsamp * c.vsamp;
        }
        let mcu_x = w.div_ceil(8 * hmax);
        let mcu_y = h.div_ceil(8 * vmax);
        let mut w_blocks = [0; 4];
        let mut h_blocks = [0; 4];
        for (ci, &(pw, ph)) in sizes.iter().enumerate() {
            w_blocks[ci] = pw.div_ceil(8);
            h_blocks[ci] = ph.div_ceil(8);
        }
        Scan {
            bw: BitWriter::new(out),
            tier: simd::tier(),
            comps,
            last_dc: [0; 4],
            w_blocks,
            h_blocks,
            mcu_x,
            mcu_y,
            row_bound: mcu_x * blocks * MAX_BLOCK_BYTES,
            blk: [[0; 64]; MAX_MCU_BLOCKS],
            dc_only: [true; MAX_MCU_BLOCKS],
            prev_dc: 0,
            edge: [0; 64],
        }
    }

    /// Codes MCU row `my`; `planes` start at that row's first sample row.
    fn row(&mut self, my: usize, planes: &[EncPlane], max_alloc: u64) -> Result<(), Error> {
        // Room for the whole row: it may fill `row_bound` bytes.
        reserve_capped(&mut self.bw.out, self.row_bound, max_alloc)?;
        if self.comps.len() == 1 {
            self.row_gray(&planes[0]);
            return Ok(());
        }
        for mx in 0..self.mcu_x {
            // FDCT the whole MCU, then code it. A block that needs the FDCT waits for the next one
            // that uses the same table, and the pair goes through the two-block kernel; anything
            // else flushes it first, so the blocks are done in order.
            let mut n = 0;
            let mut dc = self.prev_dc;
            let mut pending: Option<Pend> = None;
            for ci in 0..self.comps.len() {
                let c = &self.comps[ci];
                let p = &planes[ci];
                for dy in 0..c.vsamp {
                    for dx in 0..c.hsamp {
                        let (by, bx) = (my * c.vsamp + dy, mx * c.hsamp + dx);
                        self.dc_only[n] = true;
                        if by >= self.h_blocks[ci] || bx >= self.w_blocks[ci] {
                            // dummy: previous DC
                            if let Some(q) = pending.take() {
                                fdct_one(self.tier, &q, &mut self.blk, &mut self.dc_only);
                                dc = self.blk[q.n][0];
                            }
                            self.blk[n][0] = dc;
                            n += 1;
                            continue;
                        }
                        let (x, y) = (bx * 8, dy * 8);
                        if x + 8 <= p.w && y + 8 <= p.h {
                            let off = y * p.stride + x;
                            if let Some(v) = flat_block(p.pix, off, p.stride) {
                                if let Some(q) = pending.take() {
                                    fdct_one(self.tier, &q, &mut self.blk, &mut self.dc_only);
                                }
                                self.blk[n][0] = flat_dc(v, c.quant.q[0]);
                                dc = self.blk[n][0];
                            } else {
                                let cur = Pend {
                                    n,
                                    pix: p.pix,
                                    off,
                                    stride: p.stride,
                                    quant: c.quant,
                                };
                                match pending.take() {
                                    None => pending = Some(cur),
                                    Some(q) => {
                                        if std::ptr::eq(q.quant, cur.quant)
                                            && fdct_pair(
                                                self.tier,
                                                &q,
                                                &cur,
                                                &mut self.blk,
                                                &mut self.dc_only,
                                            )
                                        {
                                            dc = self.blk[n][0];
                                        } else {
                                            fdct_one(
                                                self.tier,
                                                &q,
                                                &mut self.blk,
                                                &mut self.dc_only,
                                            );
                                            pending = Some(cur);
                                        }
                                    }
                                }
                            }
                        } else {
                            // edge block: reads past w, h replicate the edge
                            if let Some(q) = pending.take() {
                                fdct_one(self.tier, &q, &mut self.blk, &mut self.dc_only);
                            }
                            edge_block(&mut self.edge, p, x, y);
                            if let Some(v) = flat_block(&self.edge, 0, 8) {
                                self.blk[n][0] = flat_dc(v, c.quant.q[0]);
                            } else {
                                simd::fdct_quant_rows(
                                    self.tier,
                                    &self.edge,
                                    0,
                                    8,
                                    c.quant,
                                    &mut self.blk[n],
                                );
                                self.dc_only[n] = false;
                            }
                            dc = self.blk[n][0];
                        }
                        n += 1;
                    }
                }
            }
            if let Some(q) = pending.take() {
                fdct_one(self.tier, &q, &mut self.blk, &mut self.dc_only);
                dc = self.blk[q.n][0];
            }
            n = 0;
            for ci in 0..self.comps.len() {
                let c = &self.comps[ci];
                for _ in 0..c.vsamp * c.hsamp {
                    if self.dc_only[n] {
                        write_dc_only(
                            &mut self.bw,
                            self.blk[n][0] as i32,
                            &mut self.last_dc[ci],
                            c.dc,
                            c.ac,
                        );
                    } else {
                        write_block(
                            self.tier,
                            &mut self.bw,
                            &self.blk[n],
                            &mut self.last_dc[ci],
                            c.dc,
                            c.ac,
                        );
                    }
                    n += 1;
                }
            }
            self.prev_dc = dc;
        }
        Ok(())
    }

    /// Codes the MCU row of a single-component scan (an MCU is one block, no dummies): two MCUs
    /// at a time, so two interior blocks go through the pair FDCT kernel. The blocks are coded in
    /// order, so the bytes are those of one block at a time. `p` starts at the row.
    fn row_gray(&mut self, p: &EncPlane) {
        let c = &self.comps[0];
        let mut mx = 0;
        while mx < self.mcu_x {
            let k = (self.mcu_x - mx).min(2);
            let mut offs = [None; 2];
            for i in 0..k {
                let x = (mx + i) * 8;
                self.dc_only[i] = true;
                if x + 8 <= p.w && 8 <= p.h {
                    if let Some(v) = flat_block(p.pix, x, p.stride) {
                        self.blk[i][0] = flat_dc(v, c.quant.q[0]);
                    } else {
                        offs[i] = Some(x);
                        self.dc_only[i] = false;
                    }
                } else {
                    edge_block(&mut self.edge, p, x, 0);
                    if let Some(v) = flat_block(&self.edge, 0, 8) {
                        self.blk[i][0] = flat_dc(v, c.quant.q[0]);
                    } else {
                        simd::fdct_quant_rows(
                            self.tier,
                            &self.edge,
                            0,
                            8,
                            c.quant,
                            &mut self.blk[i],
                        );
                        self.dc_only[i] = false;
                    }
                }
            }
            if let [Some(a), Some(b)] = offs {
                let (lo, hi) = self.blk.split_at_mut(1);
                let pair = (p.pix, a, p.stride);
                if simd::fdct_quant_pair(
                    self.tier,
                    pair,
                    (p.pix, b, p.stride),
                    c.quant,
                    &mut lo[0],
                    &mut hi[0],
                ) {
                    offs = [None; 2];
                }
            }
            for i in 0..k {
                if let Some(off) = offs[i] {
                    simd::fdct_quant_rows(
                        self.tier,
                        p.pix,
                        off,
                        p.stride,
                        c.quant,
                        &mut self.blk[i],
                    );
                }
            }
            for i in 0..k {
                if self.dc_only[i] {
                    let dc = self.blk[i][0] as i32;
                    write_dc_only(&mut self.bw, dc, &mut self.last_dc[0], c.dc, c.ac);
                } else {
                    let (t, b) = (self.tier, &self.blk[i]);
                    write_block(t, &mut self.bw, b, &mut self.last_dc[0], c.dc, c.ac);
                }
            }
            mx += k;
        }
    }
}

/// The 8x8 block at (`x`, `y`) of `p` into `edge`; reads past the plane replicate its edge.
fn edge_block(edge: &mut [u8; 64], p: &EncPlane, x: usize, y: usize) {
    let nn = 8usize.min(p.w - x);
    for r in 0..8 {
        let row = &p.pix[(y + r).min(p.h - 1) * p.stride + x..][..nn];
        let d = &mut edge[r * 8..r * 8 + 8];
        d[..nn].copy_from_slice(row);
        for cc in nn..8 {
            d[cc] = row[nn - 1];
        }
    }
}

/// A block waiting for its FDCT: block `n` of the MCU, at `off` in `pix`.
struct Pend<'a> {
    n: usize,
    pix: &'a [u8],
    off: usize,
    stride: usize,
    quant: &'a Quant,
}

/// The FDCT and quantizer of a waiting block alone.
fn fdct_one(
    t: Tier,
    p: &Pend,
    blk: &mut [[i16; 64]; MAX_MCU_BLOCKS],
    dc_only: &mut [bool; MAX_MCU_BLOCKS],
) {
    simd::fdct_quant_rows(t, p.pix, p.off, p.stride, p.quant, &mut blk[p.n]);
    dc_only[p.n] = false;
}

/// Two waiting blocks with the same table at once; false, having done nothing, when the tier has
/// no such kernel (`a` comes before `b`).
fn fdct_pair(
    t: Tier,
    a: &Pend,
    b: &Pend,
    blk: &mut [[i16; 64]; MAX_MCU_BLOCKS],
    dc_only: &mut [bool; MAX_MCU_BLOCKS],
) -> bool {
    let (lo, hi) = blk.split_at_mut(b.n);
    let ok = simd::fdct_quant_pair(
        t,
        (a.pix, a.off, a.stride),
        (b.pix, b.off, b.stride),
        a.quant,
        &mut lo[a.n],
        &mut hi[0],
    );
    if ok {
        dc_only[a.n] = false;
        dc_only[b.n] = false;
    }
    ok
}

// ---- the encoder (encoder.go) ----

struct State<'a> {
    limits: &'a Limits,
    ledger: &'a Ledger,
    cancel: &'a AtomicBool,
}

/// Odd or negative origins split chroma pairs.
fn direct_ycbcr(img: &YCbCr) -> bool {
    match img.ratio {
        Subsample::S444 => true,
        Subsample::S420 => {
            let (x, y) = (img.rect.min_x, img.rect.min_y);
            x >= 0 && y >= 0 && (x | y) & 1 == 0
        }
        _ => false,
    }
}

/// The rect's size; an inverted rect (`max` below `min`) is [`Error::BadImage`], and a side over
/// 65535, which the SOF cannot hold, is [`Error::TooBig`].
fn rect_dims(b: Rect) -> Result<(usize, usize), Error> {
    if b.max_x < b.min_x || b.max_y < b.min_y {
        return Err(Error::BadImage);
    }
    let (w, h) = (b.width() as usize, b.height() as usize);
    if w > 65535 || h > 65535 {
        return Err(Error::TooBig);
    }
    Ok((w, h))
}

/// Whether `img`'s planes hold every sample `draw_rgba` reads: the strides reach the last
/// column, and the last pixel's `y_offset` and `c_offset` (both grow with x and y) are inside
/// `y`, `cb` and `cr`. `img.rect` is not inverted and not empty.
fn ycbcr_planes_cover(img: &YCbCr) -> bool {
    let b = img.rect;
    let (lx, ly) = (b.max_x - 1, b.max_y - 1);
    let (yx, yy) = (lx as i128 - b.min_x as i128, ly as i128 - b.min_y as i128);
    let (cx, cy) = img.c_xy(lx, ly);
    let (cx, cy) = (cx as i128, cy as i128);
    img.y_stride as i128 > yx
        && img.c_stride as i128 > cx
        && yy * img.y_stride as i128 + yx < img.y.len() as i128
        && cy * img.c_stride as i128 + cx < img.cb.len().min(img.cr.len()) as i128
}

/// One `color.YCbCrToRGB` pixel, alpha 255.
#[inline]
fn push_rgba(pix: &mut Vec<u8>, y: u8, cb: u8, cr: u8) {
    let clamp = |v: i32| (v >> 16).clamp(0, 255) as u8;
    let yy1 = y as i32 * 0x10101;
    let (cb1, cr1) = (cb as i32 - 128, cr as i32 - 128);
    pix.extend_from_slice(&[
        clamp(yy1 + 91881 * cr1),
        clamp(yy1 - 22554 * cb1 - 46802 * cr1),
        clamp(yy1 + 116130 * cb1),
        255,
    ]);
}

/// `image/draw`'s `YCbCr` to RGBA conversion (`color.YCbCrToRGB`), alpha 255. The output is
/// charged to the ledger, then appended in order (no zero fill).
fn draw_rgba(ledger: &Ledger, cancel: &AtomicBool, img: &YCbCr) -> Result<Rgba, Error> {
    let b = img.rect;
    let (w, h) = rect_dims(b)?;
    if w > 0 && h > 0 && !ycbcr_planes_cover(img) {
        return Err(Error::BadImage);
    }
    let n = w
        .checked_mul(h)
        .and_then(|n| n.checked_mul(4))
        .ok_or(Error::BadImage)?;
    ledger.charge(n as u64)?;
    let mut pix = Vec::new();
    pix.try_reserve_exact(n).map_err(|_| Error::OutOfMemory)?;
    let out = Rgba {
        stride: 4 * w,
        rect: Rect::new(0, 0, w as i32, h as i32),
        pix: Vec::new(),
    };
    if n == 0 {
        return Ok(out);
    }
    // Chroma shifts per axis; with a non-negative origin the chroma index of column x is
    // (x + min_x % 2^sx) >> sx, as Go's `COffset` (its division truncates, so a negative origin
    // takes the per-pixel path).
    let (sx, sy) = match img.ratio {
        Subsample::S444 => (0, 0),
        Subsample::S422 => (1, 0),
        Subsample::S420 => (1, 1),
        Subsample::S440 => (0, 1),
        Subsample::S411 => (2, 0),
        Subsample::S410 => (2, 1),
    };
    let fast = b.min_x >= 0 && b.min_y >= 0;
    let (wx, wy) = (1usize << sx, 1usize << sy);
    let (rx, ry) = (b.min_x as usize % wx, b.min_y as usize % wy);
    for y in 0..h {
        check(cancel)?;
        if fast {
            let cn = (w + rx + wx - 1) >> sx; // chroma samples this row reads
            let cy = ((y + ry) >> sy) * img.c_stride;
            let yr = img.y.get(y * img.y_stride..).and_then(|r| r.get(..w));
            let cbr = img.cb.get(cy..).and_then(|r| r.get(..cn));
            let crr = img.cr.get(cy..).and_then(|r| r.get(..cn));
            let (Some(yr), Some(cbr), Some(crr)) = (yr, cbr, crr) else {
                return Err(Error::BadImage);
            };
            for (x, &yv) in yr.iter().enumerate() {
                let c = (x + rx) >> sx;
                push_rgba(&mut pix, yv, cbr[c], crr[c]);
            }
        } else {
            for x in 0..w {
                let (px, py) = (b.min_x + x as i32, b.min_y + y as i32);
                let c = img.c_offset(px, py);
                push_rgba(&mut pix, img.y[img.y_offset(px, py)], img.cb[c], img.cr[c]);
            }
        }
    }
    Ok(Rgba { pix, ..out })
}

/// The slice must hold `rows` rows of `width` samples at `stride`.
fn need_plane(pix: &[u8], stride: usize, width: usize, rows: usize) -> Result<(), Error> {
    if rows == 0 || width == 0 {
        return Ok(());
    }
    let need = (rows - 1)
        .checked_mul(stride)
        .and_then(|v| v.checked_add(width))
        .ok_or(Error::BadImage)?;
    if stride < width || pix.len() < need {
        return Err(Error::BadImage);
    }
    Ok(())
}

impl State<'_> {
    /// What the output may grow to: `max_alloc` less what the ledger already holds.
    fn out_budget(&self) -> u64 {
        self.limits.max_alloc.saturating_sub(self.ledger.used())
    }

    /// Appends EOI to the flushed output, refusing if the final bytes pass the budget.
    fn finish(&self, out: &mut Vec<u8>) -> Result<(), Error> {
        let (got, max) = (out.len() as u64 + 2, self.out_budget());
        if got > max {
            return Err(Error::AllocLimit { got, max });
        }
        out.extend_from_slice(&[0xff, 0xd9]); // EOI
        Ok(())
    }

    fn encode(
        &self,
        out: &mut Vec<u8>,
        m: EncodeImage,
        quality: i32,
        chroma444: bool,
        lq: &Quant,
        cq: &Quant,
    ) -> Result<(), Error> {
        match m {
            EncodeImage::YCbCr(img) => {
                let b = img.rect;
                let (w, h) = rect_dims(b)?;
                self.limits.check_pixels(w, h)?;
                if !direct_ycbcr(img) {
                    let rgba = draw_rgba(self.ledger, self.cancel, img)?;
                    return self.encode_4byte(
                        out,
                        &rgba.pix,
                        rgba.stride,
                        w,
                        h,
                        quality,
                        chroma444,
                        lq,
                        cq,
                    );
                }
                let c444 = img.ratio == Subsample::S444;
                let (cw, ch) = if c444 {
                    (w, h)
                } else {
                    (w.div_ceil(2), h.div_ceil(2))
                };
                let (yo, co) = (
                    img.y_offset(b.min_x, b.min_y),
                    img.c_offset(b.min_x, b.min_y),
                );
                need_plane(img.y.get(yo..).unwrap_or(&[]), img.y_stride, w, h)?;
                need_plane(img.cb.get(co..).unwrap_or(&[]), img.c_stride, cw, ch)?;
                need_plane(img.cr.get(co..).unwrap_or(&[]), img.c_stride, cw, ch)?;
                let y = EncPlane {
                    pix: img.y.get(yo..).unwrap_or(&[]),
                    stride: img.y_stride,
                    w,
                    h,
                };
                // Chroma width/height come from the real bounds, not the stride: a window's
                // stride can exceed its width.
                let cb = EncPlane {
                    pix: img.cb.get(co..).unwrap_or(&[]),
                    stride: img.c_stride,
                    w: cw,
                    h: ch,
                };
                let cr = EncPlane {
                    pix: img.cr.get(co..).unwrap_or(&[]),
                    stride: img.c_stride,
                    w: cw,
                    h: ch,
                };
                let comps = ycbcr_components(lq, cq, c444);
                write_header(out, w, h, quality, false, c444);
                self.write_scan(out, w, h, comps, [y, cb, cr].to_vec())
            }
            EncodeImage::Gray(img) => {
                let b = img.rect;
                let (w, h) = rect_dims(b)?;
                self.limits.check_pixels(w, h)?;
                need_plane(&img.pix, img.stride, w, h)?;
                let y = EncPlane {
                    pix: &img.pix,
                    stride: img.stride,
                    w,
                    h,
                };
                let c = std_codes();
                let comps = vec![EncComponent {
                    hsamp: 1,
                    vsamp: 1,
                    quant: lq,
                    dc: &c.dc_luma,
                    ac: &c.ac_luma,
                }];
                write_header(out, w, h, quality, true, false);
                self.write_scan(out, w, h, comps, vec![y])
            }
            EncodeImage::Rgba(img) => {
                let b = img.rect;
                let (w, h) = rect_dims(b)?;
                self.limits.check_pixels(w, h)?;
                need_plane(&img.pix, img.stride, w * 4, h)?;
                self.encode_4byte(out, &img.pix, img.stride, w, h, quality, chroma444, lq, cq)
            }
        }
    }

    /// Codes whole planes; polls the cancel flag.
    fn write_scan(
        &self,
        out: &mut Vec<u8>,
        w: usize,
        h: usize,
        comps: Vec<EncComponent>,
        planes: Vec<EncPlane>,
    ) -> Result<(), Error> {
        let sizes: Vec<(usize, usize)> = planes.iter().map(|p| (p.w, p.h)).collect();
        let vs: Vec<usize> = comps.iter().map(|c| c.vsamp).collect();
        let mut s = Scan::start(std::mem::take(out), w, h, comps, &sizes);
        let mut rows = Vec::new();
        let mut res = Ok(());
        for my in 0..s.mcu_y {
            if let Err(e) = check(self.cancel) {
                res = Err(e);
                break;
            }
            rows.clear();
            for (ci, p) in planes.iter().enumerate() {
                rows.push(p.at_row(my * 8 * vs[ci]));
            }
            if let Err(e) = s.row(my, &rows, self.out_budget()) {
                res = Err(e);
                break;
            }
        }
        if res.is_ok() {
            res = reserve_capped(&mut s.bw.out, FLUSH_AND_EOI, self.out_budget());
        }
        if res.is_ok() {
            s.bw.flush();
        }
        *out = std::mem::take(&mut s.bw.out);
        res?;
        self.finish(out)
    }

    /// Encodes RGB(A) input, converting per MCU row.
    fn encode_4byte(
        &self,
        out: &mut Vec<u8>,
        pix: &[u8],
        stride: usize,
        w: usize,
        h: usize,
        quality: i32,
        chroma444: bool,
        lq: &Quant,
        cq: &Quant,
    ) -> Result<(), Error> {
        let (cw, ch, rows) = if chroma444 {
            (w, h, 8)
        } else {
            (w.div_ceil(16) * 8, h.div_ceil(2), 16)
        };
        self.ledger.charge((rows * w + 16 * cw) as u64)?;
        let mut y_buf = filled(rows * w, 0u8)?;
        let mut cb_buf = filled(8 * cw, 0u8)?;
        let mut cr_buf = filled(8 * cw, 0u8)?;
        let fw = 2 * cw; // chroma padded to block edge
        let (mut full_cb, mut full_cr) = (Vec::new(), Vec::new());
        if !chroma444 {
            self.ledger.charge(4 * fw as u64)?;
            full_cb = filled(2 * fw, 0u8)?;
            full_cr = filled(2 * fw, 0u8)?;
        }
        let comps = ycbcr_components(lq, cq, chroma444);
        write_header(out, w, h, quality, false, chroma444);
        let mut s = Scan::start(
            std::mem::take(out),
            w,
            h,
            comps,
            &[(w, h), (cw, ch), (cw, ch)],
        );
        let mut res = Ok(());
        for my in 0..s.mcu_y {
            if let Err(e) = check(self.cancel) {
                res = Err(e);
                break;
            }
            let y0 = my * rows;
            if chroma444 {
                for r in y0..(y0 + rows).min(h) {
                    let o = (r - y0) * w;
                    rgba_to_ycbcr_row(
                        pix.get(r * stride..).unwrap_or(&[]),
                        &mut y_buf[o..o + w],
                        &mut cb_buf[o..o + w],
                        &mut cr_buf[o..o + w],
                    );
                }
            } else {
                strip_h2v2(
                    pix,
                    stride,
                    w,
                    h,
                    cw,
                    my,
                    &mut y_buf,
                    &mut cb_buf,
                    &mut cr_buf,
                    &mut full_cb,
                    &mut full_cr,
                );
            }
            let planes = [
                EncPlane {
                    pix: &y_buf,
                    stride: w,
                    w,
                    h: h - y0,
                },
                EncPlane {
                    pix: &cb_buf,
                    stride: cw,
                    w: cw,
                    h: ch - my * 8,
                },
                EncPlane {
                    pix: &cr_buf,
                    stride: cw,
                    w: cw,
                    h: ch - my * 8,
                },
            ];
            if let Err(e) = s.row(my, &planes, self.out_budget()) {
                res = Err(e);
                break;
            }
        }
        if res.is_ok() {
            res = reserve_capped(&mut s.bw.out, FLUSH_AND_EOI, self.out_budget());
        }
        if res.is_ok() {
            s.bw.flush();
        }
        *out = std::mem::take(&mut s.bw.out);
        res?;
        self.finish(out)
    }
}

fn quant_tables(quality: i32) -> (Quant, Quant) {
    (
        Quant::new(&quant_natural(&scale_quant_table(&STD_LUMA_QUANT, quality))),
        Quant::new(&quant_natural(&scale_quant_table(
            &STD_CHROMA_QUANT,
            quality,
        ))),
    )
}

/// The 3-component list (`jcparam.c:379-381`).
fn ycbcr_components<'a>(lq: &'a Quant, cq: &'a Quant, chroma444: bool) -> Vec<EncComponent<'a>> {
    let c = std_codes();
    let (yh, yv) = if chroma444 { (1, 1) } else { (2, 2) };
    vec![
        EncComponent {
            hsamp: yh,
            vsamp: yv,
            quant: lq,
            dc: &c.dc_luma,
            ac: &c.ac_luma,
        },
        EncComponent {
            hsamp: 1,
            vsamp: 1,
            quant: cq,
            dc: &c.dc_chroma,
            ac: &c.ac_chroma,
        },
        EncComponent {
            hsamp: 1,
            vsamp: 1,
            quant: cq,
            dc: &c.dc_chroma,
            ac: &c.ac_chroma,
        },
    ]
}

/// Fills the 4:2:0 strips of MCU row `my` (`jcsample.c`).
fn strip_h2v2(
    pix: &[u8],
    stride: usize,
    w: usize,
    h: usize,
    cw: usize,
    my: usize,
    y_buf: &mut [u8],
    cb_buf: &mut [u8],
    cr_buf: &mut [u8],
    full_cb: &mut [u8],
    full_cr: &mut [u8],
) {
    let fw = 2 * cw; // chroma padded to block edge
    let ch = h.div_ceil(2);
    for oy in my * 8..(my * 8 + 8).min(ch) {
        for k in 0..2 {
            let row = (2 * oy + k).min(h - 1); // odd last row repeats
            let o = (row - my * 16) * w;
            let fcb = &mut full_cb[k * fw..k * fw + fw];
            let fcr = &mut full_cr[k * fw..k * fw + fw];
            rgba_to_ycbcr_row(
                pix.get(row * stride..).unwrap_or(&[]),
                &mut y_buf[o..o + w],
                &mut fcb[..w],
                &mut fcr[..w],
            );
            for x in w..fw {
                fcb[x] = fcb[w - 1];
                fcr[x] = fcr[w - 1];
            }
        }
        let o = (oy - my * 8) * cw;
        downsample_h2v2_row(&mut cb_buf[o..o + cw], &full_cb[..fw], &full_cb[fw..]);
        downsample_h2v2_row(&mut cr_buf[o..o + cw], &full_cr[..fw], &full_cr[fw..]);
    }
}
