//! `RowReader`, ported from jpeg-go's `internal/jpeg/rowreader.go` and
//! `internal/codec/rows.go`.

use crate::Error;
use crate::decoder::{ColorModel, Config, Geom, build_rgba, config_of};
use crate::image::{Rect, Rgba};
use crate::limits::{Ledger, Limits, filled};
use crate::marker::Frame;
use crate::progressive::{ProgState, RowView, Smoother, decode_prog_scans, idct_row, smoothing_ok};
use crate::scan::{Ctx, Plane, decode_scan_band, max_sampling};
use std::sync::atomic::AtomicBool;

/// The most one `RowReader` may need, whatever `Limits::max_alloc` says: Go's cap. `max_alloc`
/// can only lower it; the default `max_alloc` of 1 GiB already does, so a
/// caller that wants frames up to this cap sets `max_alloc` to it.
pub const ROW_READER_MAX_NEED: u64 = 1100 << 20;

/// What a frame's scans leave besides the blocks themselves: the progressive state `idct_rows`
/// needs to match a whole-frame decode.
struct Coefs {
    ps: ProgState,
    scanned: [bool; 4],
    prog: bool,
    smooth: bool,
}

/// Entropy-decodes every scan of `f` into `planes`' blocks and does no IDCT. Every plane must
/// have `coef` set. Returns whether data was truncated. A fatal marker after a single-scan
/// frame's last row is not an error, as `decode` keeps that image.
fn decode_coefs(
    cx: &Ctx,
    f: &mut Frame,
    src: &[u8],
    planes: &mut [Plane],
) -> Result<(Coefs, bool), Error> {
    if !f.progressive {
        let r = decode_scan_band(cx, f, src, planes, None);
        return match r.err {
            Some(e) if !matches!(e, Error::AfterScan { .. }) => Err(e),
            _ => Ok((
                Coefs {
                    ps: ProgState::new(),
                    scanned: [false; 4],
                    prog: false,
                    smooth: false,
                },
                r.truncated,
            )),
        };
    }
    let pr = decode_prog_scans(cx, f, src, planes);
    if let Some(e) = pr.err {
        return Err(e);
    }
    let smooth = smoothing_ok(&pr.ps, f, planes);
    Ok((
        Coefs {
            ps: pr.ps,
            scanned: pr.scanned,
            prog: true,
            smooth,
        },
        pr.truncated,
    ))
}

impl Coefs {
    /// Writes component `ci`'s block rows `by0..by1` of `p` (a plane `decode_coefs` filled) into
    /// `pix`, whose row 0 is block row `by0`, exactly as a whole-frame decode would. `pix` is
    /// first set to 128, the value of the blocks a decode never transforms.
    fn idct_rows(&self, f: &Frame, p: &Plane, ci: usize, by0: usize, by1: usize, pix: &mut [u8]) {
        let bs = p.bs();
        let pix = &mut pix[..(by1 - by0) * bs * p.stride()];
        pix.fill(128);
        let (mut width, mut high) = (p.blocks_wide, p.blocks_high);
        let mut sm = None;
        if self.prog {
            if !self.scanned[ci] {
                return;
            }
            width = p.real_wide;
            high = p.real_high;
            if self.smooth {
                let (_, vmax) = max_sampling(f);
                sm = Some(Smoother::new(
                    f,
                    &self.ps,
                    ci,
                    &p.quant,
                    p.real_high,
                    p.real_wide,
                    &p.blocks,
                    p.blocks_wide,
                    vmax,
                ));
            }
        }
        let view = RowView::of(p);
        for by in by0..by1.min(high) {
            idct_row(&view, &p.blocks, sm.as_mut(), by, width, pix, by0);
        }
    }
}

/// The MCU rows one band spans: with 3 components and vmax 2 the chroma is upsampled vertically,
/// and fancy upsampling reads one MCU row on each side.
fn row_reader_win(nc: usize, vmax: usize, mcu_rows: usize) -> usize {
    if nc == 3 && vmax == 2 {
        3usize.min(mcu_rows)
    } else {
        1
    }
}

/// The bytes a `RowReader` holds for a `w` x `h` frame: the blocks of every component, one band
/// of planes and RGBA, and the chroma rows. `ext` is the page rounding only the ledger counts.
/// `hy`, `vy` are the first component's sampling; the other two are 1x1.
fn row_reader_need_of(
    w: usize,
    h: usize,
    nc: usize,
    hy: usize,
    vy: usize,
    rgb: bool,
) -> (u64, u64) {
    let (hmax, vmax) = (hy, vy);
    let (mcus, mcu_rows) = (w.div_ceil(hmax * 8), h.div_ceil(vmax * 8));
    let win = row_reader_win(nc, vmax, mcu_rows);
    let mut need: u64 = 1 << 16;
    let mut ext = 0u64;
    for ci in 0..nc {
        let (ch, cv) = if ci == 0 { (hy, vy) } else { (1, 1) };
        let bw = (mcus * ch) as u64;
        need += bw * (mcu_rows * cv) as u64 * 128; // coefficients
        need += bw * 8 * (win * cv) as u64 * 8; // one band of samples
        ext += 2 * 8192;
    }
    need += 4 * w as u64 * (win * vmax * 8).min(h) as u64; // RGBA band
    ext += 8192;
    if nc == 3 && !rgb {
        need += 2 * w as u64; // upsampled chroma rows
        ext += 2 * 8192;
    }
    (need, ext)
}

/// The bytes [`RowReader::new`] holds for `src`'s frame, read from its headers alone (`tables`
/// as for `new`). The frame must be one `new` accepts, else it returns the parse error, or
/// [`Error::PixelLimit`] when it is over `limits.max_pixels`.
///
/// # Errors
///
/// [`Error::PixelLimit`] as above, and the stream's header errors (the marker variants of
/// [`Error`], e.g. [`Error::NoSof`]). It reads headers only: it allocates nothing from the
/// stream, does not compare the need with `limits.max_alloc` (the caller does, or
/// [`RowReader::new`] does) and spawns no thread.
pub fn row_reader_need(src: &[u8], tables: Option<&[u8]>, limits: &Limits) -> Result<u64, Error> {
    let mut f = Frame::new();
    f.parse(src, tables)?;
    limits.check_pixels(f.width, f.height)?;
    let (need, _) = row_reader_need_of(
        f.width,
        f.height,
        f.ncomp,
        f.comp[0].h as usize,
        f.comp[0].v as usize,
        f.rgb,
    );
    Ok(need)
}

/// The most `row_reader_need` can be for any frame of at most `limits.max_pixels` pixels (and at most
/// 65535 on a side), over every width at the tallest height that fits, every sampling the decoder
/// accepts, and 1 or 3 components.
///
/// # Errors
///
/// None: it is infallible and returns a number. It depends on `limits.max_pixels` only; it
/// allocates nothing, polls no cancel flag and spawns no thread.
pub fn max_row_reader_need(limits: &Limits) -> u64 {
    let max_pixels = limits.max_pixels;
    let mut worst = 0u64;
    let mut try_one = |w: usize, h: usize, nc: usize, hy: usize, vy: usize| {
        let (need, _) = row_reader_need_of(w, h, nc, hy, vy, false);
        worst = worst.max(need);
    };
    for w in 1..=65535usize {
        let h = (max_pixels / w as u64).min(65535) as usize;
        if h < 1 {
            continue;
        }
        for hy in 1..=4 {
            for vy in 1..=4 {
                try_one(w, h, 1, hy, vy);
            }
        }
        for (hy, vy) in [(1, 1), (2, 1), (2, 2), (1, 2), (4, 1), (4, 2)] {
            try_one(w, h, 3, hy, vy);
        }
    }
    worst
}

/// Hands out a frame's RGBA rows one MCU row at a time. It entropy-decodes every scan once,
/// keeping only the int16 coefficients (128 bytes per block, padded to whole MCUs), and
/// transforms, upsamples and colour-converts a band when [`next`](RowReader::next) asks for it, so
/// its peak is the coefficients plus one band. For frames `decode` would hold as coefficients,
/// planes and RGBA at once (progressive, multi-scan). Rows equal `decode`'s `Format::Rgba` output
/// byte for byte.
pub struct RowReader {
    frame: Frame,
    cp: [Plane; 3],
    band: [Plane; 3],
    coefs: Coefs,
    cfg: Config,
    out: Rgba,
    /// `need + ext`: the budget of each `next` call's own scratch rows.
    budget: u64,
    next: usize,
    mcu_rows: usize,
    /// MCU rows per band: up to 3 when chroma needs vertical context.
    win: usize,
    truncated: bool,
    open: bool,
}

impl RowReader {
    /// Parses `src` (`tables`, when given, supplies DQT/DHT), checks the frame's
    /// [`row_reader_need`] against `limits` and [`ROW_READER_MAX_NEED`] before any allocation, and
    /// entropy-decodes every scan. Any frame decodes (progressive, multi-scan or single-scan),
    /// always to RGBA at Scale 1.
    ///
    /// # Errors
    ///
    /// [`Error::PixelLimit`] when the frame is over `limits.max_pixels`; [`Error::AllocLimit`] when
    /// its need is over `limits.max_alloc` or over [`ROW_READER_MAX_NEED`] (raise `max_alloc` to
    /// read frames above 1 GiB), from the headers before any allocation; [`Error::OutOfMemory`];
    /// [`Error::Cancelled`] when `cancel` is set (once per MCU row of each scan); and the stream's
    /// own errors (the marker and decode variants of [`Error`]). Serial: spawns no thread,
    /// whatever `limits.max_threads` is.
    pub fn new(
        src: &[u8],
        tables: Option<&[u8]>,
        limits: &Limits,
        cancel: &AtomicBool,
    ) -> Result<RowReader, Error> {
        Self::with_color_model(src, tables, ColorModel::Auto, limits, cancel)
    }

    /// [`RowReader::new`] with a [`ColorModel`] for a 3-component frame
    /// ([`DecodeOptions::color_model`](crate::DecodeOptions::color_model)). [`row_reader_need`]
    /// is an upper bound of the need for either model.
    ///
    /// # Errors
    ///
    /// Those of [`RowReader::new`]; [`Error::Unsupported`] for [`ColorModel::Rgb`] on a frame
    /// whose components do not share one sampling.
    pub fn with_color_model(
        src: &[u8],
        tables: Option<&[u8]>,
        color_model: ColorModel,
        limits: &Limits,
        cancel: &AtomicBool,
    ) -> Result<RowReader, Error> {
        let mut f = Frame::new();
        f.parse(src, tables)?;
        f.set_color_model(color_model);
        limits.check_pixels(f.width, f.height)?;
        if f.ncomp == 3
            && f.rgb
            && (f.comp[1].h != f.comp[0].h
                || f.comp[1].v != f.comp[0].v
                || f.comp[2].h != f.comp[0].h
                || f.comp[2].v != f.comp[0].v)
        {
            return Err(Error::Unsupported); // non-uniform RGB sampling, as decode
        }
        let cfg = config_of(&f);
        let (need, ext) = row_reader_need_of(
            f.width,
            f.height,
            f.ncomp,
            f.comp[0].h as usize,
            f.comp[0].v as usize,
            f.rgb,
        );
        if need > ROW_READER_MAX_NEED {
            return Err(Error::AllocLimit {
                got: need,
                max: ROW_READER_MAX_NEED,
            });
        }
        limits.check_bytes(need + ext)?;

        // The ledger charges every allocation before it is made, up to need + ext.
        let ledger = Ledger::new(&Limits {
            max_alloc: need + ext,
            ..*limits
        });
        let cx = Ctx::serial(cancel, &ledger);
        let mut cp = [Plane::new(), Plane::new(), Plane::new()];
        for p in cp.iter_mut().take(f.ncomp) {
            p.coef = true;
        }
        let nc = f.ncomp;
        let (coefs, truncated) = decode_coefs(&cx, &mut f, src, &mut cp[..nc])?;
        let (_, vmax) = max_sampling(&f);
        let mcu_rows = f.height.div_ceil(vmax * 8);
        let win = row_reader_win(nc, vmax, mcu_rows);
        let mut band = [Plane::new(), Plane::new(), Plane::new()];
        for ci in 0..nc {
            let bw = cp[ci].blocks_wide;
            let n = bw * 8 * win * f.comp[ci].v as usize * 8;
            ledger.charge(n as u64)?;
            band[ci] = Plane {
                block_size: 8,
                blocks_wide: bw,
                pix: filled(n, 0u8)?,
                ..Plane::new()
            };
        }
        // One RGBA band, reused by every `next` (the previous band is gone before the next is
        // built), as `row_reader_need_of` counts it.
        let n = 4 * f.width * (win * vmax * 8).min(f.height);
        ledger.charge(n as u64)?;
        let mut pix = Vec::new();
        pix.try_reserve_exact(n).map_err(|_| Error::OutOfMemory)?;
        Ok(RowReader {
            frame: f,
            cp,
            band,
            coefs,
            cfg,
            out: Rgba {
                pix,
                stride: 0,
                rect: Rect::default(),
            },
            budget: need + ext,
            next: 0,
            mcu_rows,
            win,
            truncated,
            open: true,
        })
    }

    /// The frame's shape. Infallible.
    pub fn config(&self) -> Config {
        self.cfg
    }

    /// Whether entropy data ran out, had a bad Huffman code or restart markers were out of
    /// sequence: the rows are `decode`'s, which would return them with [`Error::Truncated`].
    pub fn truncated(&self) -> bool {
        self.truncated
    }

    /// Returns the next MCU row's pixels, up to 8 * vmax rows at full width, at their position in
    /// the frame (`rect.min_y`). The image is valid until the next call. It returns `Ok(None)`
    /// after the last row, and closes the reader.
    ///
    /// # Errors
    ///
    /// [`Error::RowReaderClosed`] after `close` or a previous error; [`Error::Cancelled`] when
    /// `cancel` is set (checked on entry; the reader is then closed); [`Error::AllocLimit`] when
    /// the band's scratch rows do not fit the budget fixed at `new` (the `max_alloc` of that call)
    /// and [`Error::OutOfMemory`]. Serial: spawns no thread.
    pub fn next(&mut self, cancel: &AtomicBool) -> Result<Option<&Rgba>, Error> {
        if !self.open {
            return Err(Error::RowReaderClosed);
        }
        if self.next >= self.mcu_rows {
            self.close();
            return Ok(None);
        }
        if let Err(e) = crate::limits::check(cancel) {
            self.close();
            return Err(e);
        }
        let f = &self.frame;
        let (hmax, vmax) = max_sampling(f);
        let m = self.next;
        let (mut lo, mut hi) = (m, m);
        if self.win > 1 {
            lo = m.saturating_sub(1);
            hi = (m + 1).min(self.mcu_rows - 1);
        }
        for ci in 0..f.ncomp {
            let v = f.comp[ci].v as usize;
            self.band[ci].blocks_high = (hi - lo + 1) * v;
            self.coefs.idct_rows(
                f,
                &self.cp[ci],
                ci,
                lo * v,
                (hi + 1) * v,
                &mut self.band[ci].pix,
            );
        }
        let out_step = vmax * 8;
        let (w, h) = (f.width, f.height);
        let ledger = Ledger::new(&Limits {
            max_pixels: u64::MAX,
            max_alloc: self.budget,
            max_threads: 1,
        });
        let cx = Ctx::serial(cancel, &ledger);
        let r = Rect::new(
            0,
            (lo * out_step) as i32,
            w as i32,
            ((hi + 1) * out_step).min(h) as i32,
        );
        let g = Geom {
            hmax,
            vmax,
            min_size: 8,
            prefilter: 0,
        };
        // Writes into the previous band's buffer; its capacity is the whole band.
        let mut pix = std::mem::take(&mut self.out.pix);
        pix.resize(4 * w * r.height() as usize, 0);
        let img = match build_rgba(
            &cx,
            f,
            &mut self.band,
            f.ncomp == 1,
            None,
            r,
            None,
            &g,
            Some((&mut pix[..], 4 * w)),
            Vec::new(),
            &mut Vec::new(),
        ) {
            Ok(i) => i,
            Err(e) => {
                self.close();
                return Err(e);
            }
        };
        let y0 = m * out_step;
        let rows = Rect::new(0, y0 as i32, w as i32, (y0 + out_step).min(h) as i32);
        let off = img.pix_offset(0, y0 as i32);
        pix.drain(..off);
        pix.truncate(rows.height() as usize * img.stride);
        self.out = Rgba {
            pix,
            stride: img.stride,
            rect: rows,
        };
        self.next += 1;
        Ok(Some(&self.out))
    }

    /// Drops the coefficients. It is idempotent and cannot fail; it spawns no thread.
    pub fn close(&mut self) {
        if self.open {
            self.open = false;
            self.cp = [Plane::new(), Plane::new(), Plane::new()];
            self.band = [Plane::new(), Plane::new(), Plane::new()];
            self.out = Rgba {
                pix: Vec::new(),
                stride: 0,
                rect: Rect::default(),
            };
        }
    }
}
