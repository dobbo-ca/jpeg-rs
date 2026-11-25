//! Shared by the fuzz targets: the limits, and the options a target reads from the front of its
//! input as jpeg-go's `FuzzDecode` does, so its seeds (four option bytes, then a stream) work here.
#![allow(dead_code)]

use jpeg::{Format, Gray, Image, Limits, Rect, Rgba, Subsample, YCbCr};
use std::sync::atomic::AtomicBool;

pub static NO: AtomicBool = AtomicBool::new(false);

/// Small, so a hostile frame cannot reach the fuzzer's malloc limit.
pub fn limits() -> Limits {
    Limits {
        max_pixels: 1 << 20,
        max_alloc: 1 << 27,
        max_threads: 1,
    }
}

pub struct Input<'a> {
    /// The four option bytes.
    pub b: [u8; 4],
    /// An abbreviated tables stream split off the front of the data.
    pub tables: Option<&'a [u8]>,
    pub data: &'a [u8],
}

impl Input<'_> {
    pub fn format(&self) -> Format {
        [Format::YCbCr, Format::Rgba, Format::Gray][usize::from(self.b[0] % 3)]
    }

    pub fn scale(&self) -> u8 {
        [1, 2, 4, 8][usize::from(self.b[1] % 4)]
    }
}

/// [format][scale][bit 0: split off tables][split position][data].
pub fn parse(src: &[u8]) -> Option<Input<'_>> {
    let (&[b0, b1, b2, b3], data) = src.split_first_chunk::<4>()? else {
        return None;
    };
    let (mut tables, mut data) = (None, data);
    if b2 & 1 == 1 && data.len() > 16 {
        let n = usize::from(b3) % (data.len() - 1);
        if n > 0 {
            tables = Some(&data[..n]);
            data = &data[n..];
        }
    }
    Some(Input {
        b: [b0, b1, b2, b3],
        tables,
        data,
    })
}

/// The samples of `reg` inside `r` equal those of `full` (a region is a crop of the whole decode).
pub fn same_samples(full: &Image, reg: &Image, r: Rect) -> bool {
    let w = r.width() as usize;
    match (full, reg) {
        (Image::Rgba(a), Image::Rgba(b)) => rows_equal(a, b, r, 4 * w),
        (Image::Gray(a), Image::Gray(b)) => gray_equal(a, b, r, w),
        (Image::YCbCr(a), Image::YCbCr(b)) => {
            a.ratio == b.ratio
                && (r.min_y..r.max_y).all(|y| {
                    (r.min_x..r.max_x).all(|x| {
                        let (ia, ib) = (a.y_offset(x, y), b.y_offset(x, y));
                        let (ca, cb) = (a.c_offset(x, y), b.c_offset(x, y));
                        a.y[ia] == b.y[ib] && a.cb[ca] == b.cb[cb] && a.cr[ca] == b.cr[cb]
                    })
                })
        }
        _ => false,
    }
}

fn rows_equal(a: &Rgba, b: &Rgba, r: Rect, n: usize) -> bool {
    (r.min_y..r.max_y).all(|y| {
        let (oa, ob) = (a.pix_offset(r.min_x, y), b.pix_offset(r.min_x, y));
        a.pix[oa..oa + n] == b.pix[ob..ob + n]
    })
}

fn gray_equal(a: &Gray, b: &Gray, r: Rect, n: usize) -> bool {
    (r.min_y..r.max_y).all(|y| {
        let (oa, ob) = (a.pix_offset(r.min_x, y), b.pix_offset(r.min_x, y));
        a.pix[oa..oa + n] == b.pix[ob..ob + n]
    })
}

/// The visible pixels of an RGBA image, row after row.
pub fn tight_rgba(m: &Rgba) -> Vec<u8> {
    let (w, h) = (m.rect.width() as usize, m.rect.height() as usize);
    (0..h)
        .flat_map(|y| m.pix[y * m.stride..y * m.stride + 4 * w].iter().copied())
        .collect()
}

pub const RATIOS: [Subsample; 6] = [
    Subsample::S444,
    Subsample::S422,
    Subsample::S420,
    Subsample::S440,
    Subsample::S411,
    Subsample::S410,
];
pub const RATIO_DIV: [(usize, usize); 6] = [(1, 1), (2, 1), (2, 2), (1, 2), (4, 1), (4, 2)];

/// An encoder input built from the bytes of a fuzz input, cycled.
pub enum Pixels {
    Rgba(Rgba),
    Gray(Gray),
    YCbCr(YCbCr),
}

/// The sub-image (ox, oy, ox+w, oy+h) of a parent image whose strides are `pad` past the width;
/// `kind` 0 is RGBA, 1 Gray, 2 to 7 YCbCr at the six ratios. The parent's samples are `bytes`,
/// cycled.
pub fn build(
    kind: u8,
    w: usize,
    h: usize,
    ox: usize,
    oy: usize,
    pad: usize,
    bytes: &[u8],
) -> Pixels {
    let mut i = 0usize;
    let mut next = |n: usize| -> Vec<u8> {
        (0..n)
            .map(|_| {
                i += 1;
                bytes[(i - 1) % bytes.len()]
            })
            .collect()
    };
    let (pw, ph) = (w + ox, h + oy);
    let rect = Rect::new(ox as i32, oy as i32, (ox + w) as i32, (oy + h) as i32);
    match kind {
        0 => {
            let stride = 4 * pw + pad;
            let pix = next(stride * ph);
            Pixels::Rgba(Rgba {
                pix: pix[oy * stride + 4 * ox..].to_vec(),
                stride,
                rect,
            })
        }
        1 => {
            let stride = pw + pad;
            let pix = next(stride * ph);
            Pixels::Gray(Gray {
                pix: pix[oy * stride + ox..].to_vec(),
                stride,
                rect,
            })
        }
        _ => {
            let r = usize::from(kind - 2);
            let (dx, dy) = RATIO_DIV[r];
            let (cw, ch) = (pw.div_ceil(dx), ph.div_ceil(dy));
            let (ys, cs) = (pw + pad, cw + pad);
            let (y, cb, cr) = (next(ys * ph), next(cs * ch), next(cs * ch));
            let (yi, ci) = (oy * ys + ox, (oy / dy) * cs + ox / dx);
            Pixels::YCbCr(YCbCr {
                y: y[yi..].to_vec(),
                cb: cb[ci..].to_vec(),
                cr: cr[ci..].to_vec(),
                y_stride: ys,
                c_stride: cs,
                ratio: RATIOS[r],
                rect,
            })
        }
    }
}
