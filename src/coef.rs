//! `DecodeCoefficients`, ported from jpeg-go's `internal/jpeg/coef.go`.

use crate::Error;
use crate::decoder::{Config, config_of};
use crate::limits::{Ledger, Limits};
use crate::marker::Frame;
use crate::scan::{Ctx, Plane, decode_scan, max_sampling};
use std::sync::atomic::AtomicBool;

/// A frame's quantized DCT coefficients, libjpeg-turbo's `jpeg_read_coefficients` view. It lets
/// a caller run its own IDCT and colour step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Coefficients {
    pub config: Config,
    /// In frame component order.
    pub comp: Vec<ComponentCoefficients>,
}

/// One component's blocks, row-major and padded to whole MCUs: block (bx, by) is
/// `blocks[by * blocks_wide + bx]`. Blocks are in natural order and not dequantized; `quant` is
/// the natural-order table they were coded with, or zero for a component no scan reached (turbo's
/// NULL `quant_table`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComponentCoefficients {
    /// Sampling factors.
    pub h: usize,
    pub v: usize,
    pub blocks_wide: usize,
    pub blocks_high: usize,
    pub blocks: Vec<[i16; 64]>,
    pub quant: [u16; 64],
}

/// The coefficients with the error Go returns beside them: `error` is [`Error::Truncated`] when
/// entropy data ran out, had a bad Huffman code or restart markers out of sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedCoefficients {
    pub coefficients: Coefficients,
    pub error: Option<Error>,
}

/// Entropy-decodes `src` without IDCT, upsampling or colour conversion; `tables`, when given,
/// supplies DQT/DHT. The size is checked against `limits` from the SOF before any allocation.
/// Truncated entropy data gives the blocks libjpeg-turbo would, with [`Error::Truncated`] beside
/// them.
///
/// # Errors
///
/// [`Error::PixelLimit`] when the frame is over `limits.max_pixels` and [`Error::AllocLimit`]
/// when its blocks (128 bytes each, padded to whole MCUs) would pass `limits.max_alloc`, both
/// from the headers before any allocation; [`Error::OutOfMemory`]; [`Error::Cancelled`] when
/// `cancel` is set (checked once per MCU row); and the stream's own errors (the marker and decode
/// variants of [`Error`]). Serial: spawns no thread, whatever `limits.max_threads` is.
/// A 4-component frame is [`Error::Unsupported`], as in Go; see
/// [`decode_coefficients_any_components`].
pub fn decode_coefficients(
    src: &[u8],
    tables: Option<&[u8]>,
    limits: &Limits,
    cancel: &AtomicBool,
) -> Result<DecodedCoefficients, Error> {
    coefficients(src, tables, limits, cancel, false)
}

/// [`decode_coefficients`] that also accepts 4-component (CMYK/YCCK) 8-bit frames, which Go's
/// `DecodeCoefficients` rejects: `Coefficients::comp` then has four entries and
/// `Config::components` is 4 (`rgb` is false). Sampling factors are limited to 1 to 4 with at
/// most 10 blocks per MCU. For callers (pdf-rs) that run their own IDCT and colour conversion.
///
/// # Errors
///
/// As [`decode_coefficients`], with the same limits, cancel checks and serial execution.
pub fn decode_coefficients_any_components(
    src: &[u8],
    tables: Option<&[u8]>,
    limits: &Limits,
    cancel: &AtomicBool,
) -> Result<DecodedCoefficients, Error> {
    coefficients(src, tables, limits, cancel, true)
}

fn coefficients(
    src: &[u8],
    tables: Option<&[u8]>,
    limits: &Limits,
    cancel: &AtomicBool,
    allow_four: bool,
) -> Result<DecodedCoefficients, Error> {
    let mut f = Frame::new();
    f.allow_four = allow_four;
    f.parse(src, tables)?;
    limits.check_pixels(f.width, f.height)?;
    let (hmax, vmax) = max_sampling(&f);
    let mut need: u64 = 1 << 16; // tables and frame state, as decode_need
    for ci in 0..f.ncomp {
        let w = f.width.div_ceil(hmax * 8) * f.comp[ci].h as usize;
        let h = f.height.div_ceil(vmax * 8) * f.comp[ci].v as usize;
        need += w as u64 * h as u64 * 128;
    }
    limits.check_bytes(need)?;
    let ledger = Ledger::new(limits);
    let cx = Ctx::serial(cancel, &ledger);

    let mut planes = [Plane::new(), Plane::new(), Plane::new(), Plane::new()];
    for p in planes.iter_mut().take(f.ncomp) {
        p.coef = true;
    }
    let nc = f.ncomp;
    let (truncated, err) = decode_scan(&cx, &mut f, src, &mut planes[..nc]);
    if let Some(e) = err {
        return Err(e);
    }

    let mut comp = Vec::new();
    for (ci, p) in planes.iter_mut().enumerate().take(f.ncomp) {
        comp.push(ComponentCoefficients {
            h: f.comp[ci].h as usize,
            v: f.comp[ci].v as usize,
            blocks_wide: p.blocks_wide,
            blocks_high: p.blocks_high,
            blocks: std::mem::take(&mut p.blocks),
            quant: p.quant,
        });
    }
    Ok(DecodedCoefficients {
        coefficients: Coefficients {
            config: config_of(&f),
            comp,
        },
        error: truncated.then_some(Error::Truncated),
    })
}
