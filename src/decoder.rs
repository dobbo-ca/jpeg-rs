//! The decode entry points, ported from jpeg-go's `internal/jpeg/jpeg.go` and `limit.go`.

use crate::Error;
use crate::dct::Block;
use crate::image::{Gray, Image, Rect, Rgba, Subsample, YCbCr};
use crate::limits::{Ledger, Limits, check, refit};
use crate::marker::Frame;
use crate::par::{Gauge, ThreadPool};
use crate::sample::{
    YccGeom, YccPlanes, chroma_prefilter, gray_to_rgba, rgb_to_rgba, ycbcr_upsample_convert,
    ycbcr_upsample_convert_n,
};
use crate::scan::{
    Band, Ctx, Idct, IdctHook, Plane, component_block_size, decode_scan_band, downsampled_dims,
    max_sampling,
};
use std::sync::atomic::AtomicBool;

/// Region coordinates are bounded so pad sizes cannot overflow.
const MAX_REGION: i32 = 1 << 29;

/// The output type of a decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Format {
    /// [`Image::YCbCr`] at the stream's sampling, no conversion; [`Image::Gray`] for 1 component.
    #[default]
    YCbCr,
    /// [`Image::Rgba`], alpha 255.
    Rgba,
    /// [`Image::Gray`]: Y for YCbCr streams, turbo's `rgb_gray` for RGB streams.
    Gray,
}

/// How a 3-component frame's samples map to RGB.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorModel {
    /// From the stream, as libjpeg-turbo and Go read it: JFIF means YCbCr, else an Adobe APP14
    /// transform 0 means RGB, else component ids R, G, B mean RGB, else YCbCr.
    #[default]
    Auto,
    /// RGB as stored: no colour transform, whatever JFIF, Adobe APP14 or the component ids say.
    /// For TIFF tiles with PhotometricInterpretation RGB (Aperio SVS), as libtiff and OpenSlide
    /// decode them. The frame's components must share one sampling (else
    /// [`Error::Unsupported`], as for a stream marked RGB); no effect on 1 or 4 components.
    Rgb,
}

/// Replaces `src`'s upsampling and colour conversion: it writes `dst` from `dst`'s origin. `src`
/// keeps block padding: strides are block-padded and the samples past `src.rect` are real IDCT
/// output.
pub type ToRgbaHook = fn(dst: &mut Rgba, src: &YCbCr);

/// Replaces reconstruction steps so a caller can match another library bit for bit. `None`
/// fields keep the libjpeg-turbo defaults. A non-empty `Recon` with `scale != 1` or a region
/// returns [`Error::Unsupported`].
#[derive(Clone, Copy, Default)]
pub struct Recon {
    /// One block of dequantized coefficients in natural order in, its 8x8 samples, level shifted
    /// and clamped, out.
    pub idct: Option<IdctHook>,
    pub to_rgba: Option<ToRgbaHook>,
}

/// What a decode needs besides the stream.
#[derive(Clone, Copy, Default)]
pub struct DecodeOptions<'a> {
    pub format: Format,
    /// 1, 2, 4 or 8; 0 means 1. The output is `ceil(W / scale)` by `ceil(H / scale)`.
    pub scale: u8,
    /// In scaled output coordinates; `None` or empty means all. The result's `bounds()` equal the
    /// region, zero-filled past the image.
    pub region: Option<Rect>,
    /// An abbreviated tables stream (TIFF tag 347), read before `src`.
    pub tables: Option<&'a [u8]>,
    /// The offset in `src` of each restart interval's first entropy byte (NDPI tag 65426); `None`
    /// scans for RSTn. An offset counts when RST(k-1) sits just before it.
    pub restart: Option<&'a [usize]>,
    pub recon: Recon,
    /// With `region`, entropy-decodes every MCU of the frame so the error is the one a
    /// whole-frame decode returns. Pixels are unchanged.
    pub check_frame: bool,
    /// The worker threads this call's parallel passes run on, instead of the crate's shared
    /// pool; see [`ThreadPool`]. `Limits::max_threads` 1 still runs serially. A [`Decoder`]'s
    /// own pool is used when this is `None`.
    pub pool: Option<&'a ThreadPool>,
    /// DCT-domain pyramid level: with `scale` 2, 4 or 8, the output is `floor(W / scale)` by
    /// `floor(H / scale)`, and chroma that the IDCT leaves at the output's resolution (4:2:0 at
    /// scale 2 and 4) gets a small blur (`[1 6 1] / 8` at scale 2, `[1 14 1] / 16` at scale 4)
    /// so the colour approximates the full decode averaged over `scale` x `scale` blocks. Not
    /// bit-exact with libjpeg-turbo or Go; a lossy, opt-in mode. No effect at scale 1. With a
    /// scale, [`Error::Unsupported`] for a `region`, [`Format::YCbCr`] or a frame smaller than
    /// `scale` in either direction. Default `false`.
    pub pyramid: bool,
    /// How a 3-component frame's samples map to RGB. Default [`ColorModel::Auto`], the stream's
    /// own markers; [`Config::rgb`] from [`decode_config`] reports that reading, not this field.
    pub color_model: ColorModel,
}

/// An image with the error Go returns beside it. `error` is [`Error::Truncated`] or
/// [`Error::AfterScan`], and the image is still usable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decoded {
    pub image: Image,
    pub error: Option<Error>,
}

/// A JPEG frame's shape and colour space, from [`decode_config`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Config {
    pub width: usize,
    pub height: usize,
    /// 1 or 3; 4 (CMYK/YCCK) only from `decode_coefficients_any_components`.
    pub components: usize,
    /// 3 components only; [`Subsample::S444`] for gray.
    pub ratio: Subsample,
    /// Stored as RGB (Adobe transform 0 or ids R, G, B).
    pub rgb: bool,
    /// In MCUs, 0 for none.
    pub restart_interval: usize,
}

pub(crate) fn config_of(f: &Frame) -> Config {
    Config {
        width: f.width,
        height: f.height,
        components: f.ncomp,
        ratio: if f.ncomp == 3 {
            Subsample::from_sampling(f.comp[0].h, f.comp[0].v)
        } else {
            Subsample::S444
        },
        rgb: f.rgb,
        restart_interval: f.restart_interval,
    }
}

/// Parses `src` (an abbreviated tables stream, when given, supplies DQT/DHT) up to SOS and
/// returns the frame's shape and colour space. It allocates nothing from the stream.
///
/// # Errors
///
/// [`Error::PixelLimit`] when the frame is over `limits.max_pixels`, and the stream's own errors
/// (the marker variants of [`Error`], e.g. [`Error::NoSof`]). It allocates nothing from the
/// stream, so `max_alloc` and `max_threads` do not apply, and it takes no cancel flag. Serial:
/// spawns no thread.
pub fn decode_config(src: &[u8], tables: Option<&[u8]>, limits: &Limits) -> Result<Config, Error> {
    let mut f = Frame::new();
    f.parse(src, tables)?;
    limits.check_pixels(f.width, f.height)?;
    Ok(config_of(&f))
}

/// Decodes `src`. Truncated entropy data decodes like libjpeg-turbo: missing blocks are gray
/// until the next restart, and the image comes back with [`Error::Truncated`] in
/// [`Decoded::error`]. A restart marker out of sequence, which turbo also warns about, gives the
/// same error, as does a Huffman code no table matches. A fatal marker after the last row of a
/// single-scan frame gives the image with [`Error::AfterScan`], as turbo fails in
/// `jpeg_finish_decompress` with every row already out. A region that ends above the last row
/// gets no error, as turbo's crop skips those rows. A multi-scan frame gives no image, as turbo
/// reads to EOI first.
///
/// `limits.max_alloc` is the peak of the call and `limits.max_threads` its thread count; the
/// image and the error are the same for every `max_threads`.
///
/// # Errors
///
/// [`Error::PixelLimit`] when the frame is over `limits.max_pixels`; [`Error::AllocLimit`] when
/// the call would hold more than `limits.max_alloc` bytes at once, all threads together (checked
/// from the headers before any allocation, and again at each charge); [`Error::OutOfMemory`] when
/// the allocator refuses a size within the limits; [`Error::Cancelled`] when `cancel` is set
/// (polled once per MCU row, and every 16 rows in the colour and copy passes); [`Error::Unsupported`]
/// for a non-empty `o.recon` with a scale or region, and for `o.pyramid` with a scale and a
/// region, `Format::YCbCr` or a frame smaller than the scale; [`Error::Region`] and
/// [`Error::RegionEmpty`] for a bad region; and the stream's own errors (the marker and decode variants of [`Error`]).
/// `Truncated` and `AfterScan` come in [`Decoded::error`] beside the image instead.
///
/// # Threads
///
/// Uses up to `limits.max_threads` workers, the calling thread included (restart bands of a
/// baseline scan with a DRI, colour stripes), as many as the input is big enough to feed and
/// only when their scratch memory fits in `max_alloc`. They are the threads of `o.pool`, else of
/// the crate's shared pool (started on the first such call); the pool's thread stacks are the
/// pool's and are not charged to `max_alloc`. `max_threads = 1` runs on the calling thread and
/// never touches a pool.
pub fn decode(
    src: &[u8],
    o: &DecodeOptions,
    limits: &Limits,
    cancel: &AtomicBool,
) -> Result<Decoded, Error> {
    decode_impl(src, o, limits, cancel, None)
}

/// What one call used, for tests: see [`decode_probe`].
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Probe {
    /// The most bytes the call's ledger had booked at once, an upper bound of its heap peak.
    pub charged_peak: u64,
    /// The most threads the call had running at once, the calling thread included.
    pub threads_peak: usize,
}

/// [`decode`] that also reports what the call used, per call and with no global state.
/// `uncapped` lifts the size caps on the workers of a pass, so a test can thread a small frame.
///
/// # Errors
///
/// Those of [`decode`], in the first field; it may spawn threads as [`decode`] does.
#[doc(hidden)]
pub fn decode_probe(
    src: &[u8],
    o: &DecodeOptions,
    limits: &Limits,
    cancel: &AtomicBool,
    uncapped: bool,
) -> (Result<Decoded, Error>, Probe) {
    let ledger = Ledger::new(limits);
    let gauge = Gauge::new();
    let r = decode_with(
        src,
        o,
        limits,
        cancel,
        None,
        None,
        uncapped,
        &ledger,
        &gauge,
        &mut Scratch::default(),
    );
    let probe = Probe {
        charged_peak: ledger.peak(),
        threads_peak: gauge.peak(),
    };
    (r, probe)
}

/// A window of a caller's RGBA buffer: row `y` is `pix[y * stride..]`, of which the first
/// `4 * width` bytes are written.
pub struct RgbaWindow<'a> {
    pub pix: &'a mut [u8],
    pub stride: usize,
}

/// [`decode`] with `Format::Rgba` at scale 1, no region and no `Recon`, written straight into
/// `dst`, which may be a strided window of a larger buffer; the bytes equal [`decode`]'s. It must
/// match the frame's size (`stride >= 4 * width`, `pix` reaching the last row), else
/// [`Error::Unsupported`] before any write. The result is the error Go returns beside the image:
/// `Some(Error::Truncated)` or `Some(Error::AfterScan { .. })`.
///
/// # Errors
///
/// Those of [`decode`] (`max_pixels`, `max_alloc` as [`Error::AllocLimit`], [`Error::Cancelled`],
/// [`Error::OutOfMemory`], the stream's errors), and [`Error::Unsupported`] when `dst` does not
/// match the frame or `o` asks for a scale, region or `Recon`. It may spawn threads on the same
/// terms as [`decode`], up to `limits.max_threads`; `dst` is the caller's and is not charged.
pub fn decode_into_rgba(
    src: &[u8],
    o: &DecodeOptions,
    limits: &Limits,
    cancel: &AtomicBool,
    dst: RgbaWindow,
) -> Result<Option<Error>, Error> {
    decode_impl(src, o, limits, cancel, Some(dst)).map(|d| d.error)
}

fn decode_impl(
    src: &[u8],
    o: &DecodeOptions,
    limits: &Limits,
    cancel: &AtomicBool,
    window: Option<RgbaWindow>,
) -> Result<Decoded, Error> {
    decode_pooled(
        src,
        o,
        limits,
        cancel,
        window,
        &mut Scratch::default(),
        None,
    )
}

fn decode_pooled(
    src: &[u8],
    o: &DecodeOptions,
    limits: &Limits,
    cancel: &AtomicBool,
    window: Option<RgbaWindow>,
    pool: &mut Scratch,
    workers: Option<&ThreadPool>,
) -> Result<Decoded, Error> {
    let ledger = Ledger::new(limits);
    let gauge = Gauge::new();
    decode_with(
        src, o, limits, cancel, window, workers, false, &ledger, &gauge, pool,
    )
}

/// The most buffers a [`Decoder`] keeps: a frame's planes, its RGBA image and the last result.
const POOL_MAX: usize = 8;

/// What a [`Decoder`] keeps between calls: the byte buffers (planes, RGBA image, last result),
/// the coefficient blocks of progressive and multi-scan frames, and the colour rows of the
/// RGBA conversion.
#[derive(Default)]
pub(crate) struct Scratch {
    bufs: Vec<Vec<u8>>,
    blocks: Vec<Vec<Block>>,
    rows: Vec<u8>,
}

/// The buffer of `pool` that fits `n` elements best (the smallest that is large enough, else the
/// largest, which then grows), or an empty one.
fn take_buf<T>(pool: &mut Vec<Vec<T>>, n: usize) -> Vec<T> {
    let fit = pool
        .iter()
        .enumerate()
        .filter(|(_, b)| b.capacity() >= n)
        .min_by_key(|(_, b)| b.capacity())
        .or_else(|| pool.iter().enumerate().max_by_key(|(_, b)| b.capacity()))
        .map(|(i, _)| i);
    fit.map_or_else(Vec::new, |i| pool.swap_remove(i))
}

/// A decoder that keeps its buffers between calls (Go's `Decoder`): the planes and the RGBA image
/// of one call are the next call's, grown when the frame is larger and cut when it is smaller,
/// never cleared, since a decode writes every byte of them before it reads one. A call is
/// charged to its `Limits::max_alloc` for all it uses, reused or not, so the limits, the errors
/// and the output are those of the free functions; only the time differs. Hold one for a run of
/// decodes (a tile loop, a strip reader); the buffers stay until the `Decoder` drops. It is not
/// `Sync`; use one per thread.
///
/// It also keeps the coefficient blocks of a progressive or multi-scan frame and the colour rows
/// of the RGBA conversion. With `Limits::max_threads` 1, a call that decodes a frame of the
/// same shape as an earlier one on this `Decoder` allocates nothing from the heap (tests/reuse_alloc.rs),
/// except a region that reaches past the image, whose zero-padded copy is new each time. Threaded
/// calls still allocate a few small job lists and one boxed task per extra worker.
pub struct Decoder {
    pool: Scratch,
    last: Option<Decoded>,
    workers: Option<ThreadPool>,
}

impl Default for Decoder {
    /// The two lists of kept buffers get room for `POOL_MAX` each, so no later call grows one:
    /// the first `recycle` of a result whose planes were never pooled would otherwise allocate.
    fn default() -> Self {
        Self {
            pool: Scratch {
                bufs: Vec::with_capacity(POOL_MAX),
                blocks: Vec::with_capacity(POOL_MAX),
                rows: Vec::new(),
            },
            last: None,
            workers: None,
        }
    }
}

impl Decoder {
    /// An empty `Decoder`: it holds no buffers until its first call, and takes no limits, which
    /// each call brings.
    pub fn new() -> Self {
        Self::default()
    }

    /// A `Decoder` whose calls run their parallel passes on `pool` (see [`ThreadPool`]) instead
    /// of the crate's shared one, unless a call's `DecodeOptions::pool` names another.
    pub fn with_pool(pool: ThreadPool) -> Self {
        Self {
            workers: Some(pool),
            ..Self::default()
        }
    }

    /// [`decode`], returning the image by reference: it is valid until the next call on this
    /// `Decoder`, which takes its buffers back. Clone it to keep it.
    ///
    /// # Errors
    ///
    /// Those of [`decode`]: `limits.max_pixels` ([`Error::PixelLimit`]), `limits.max_alloc`
    /// ([`Error::AllocLimit`], charged for all the call uses, reused buffers included),
    /// [`Error::Cancelled`] and the stream's errors. It uses up to `limits.max_threads` workers
    /// on the same terms as [`decode`]; `max_threads = 1` never touches a pool.
    pub fn decode(
        &mut self,
        src: &[u8],
        o: &DecodeOptions,
        limits: &Limits,
        cancel: &AtomicBool,
    ) -> Result<&Decoded, Error> {
        self.recycle();
        let d = decode_pooled(
            src,
            o,
            limits,
            cancel,
            None,
            &mut self.pool,
            self.workers.as_ref(),
        )?;
        Ok(&*self.last.insert(d))
    }

    /// [`decode_into_rgba`] with this `Decoder`'s planes.
    ///
    /// # Errors
    ///
    /// Those of [`decode_into_rgba`]: `limits.max_pixels`, `limits.max_alloc`, [`Error::Cancelled`],
    /// [`Error::Unsupported`] for a `dst` that does not match, and the stream's errors. Threads
    /// as [`Decoder::decode`].
    pub fn decode_into_rgba(
        &mut self,
        src: &[u8],
        o: &DecodeOptions,
        limits: &Limits,
        cancel: &AtomicBool,
        dst: RgbaWindow,
    ) -> Result<Option<Error>, Error> {
        self.recycle();
        decode_pooled(
            src,
            o,
            limits,
            cancel,
            Some(dst),
            &mut self.pool,
            self.workers.as_ref(),
        )
        .map(|d| d.error)
    }

    /// Takes the buffers of the last result back.
    fn recycle(&mut self) {
        let Some(d) = self.last.take() else { return };
        let bufs = match d.image {
            Image::Gray(g) => [g.pix, Vec::new(), Vec::new()],
            Image::Rgba(g) => [g.pix, Vec::new(), Vec::new()],
            Image::YCbCr(g) => [g.y, g.cb, g.cr],
        };
        for b in bufs {
            if b.capacity() > 0 && self.pool.bufs.len() < POOL_MAX {
                self.pool.bufs.push(b);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn decode_with(
    src: &[u8],
    o: &DecodeOptions,
    limits: &Limits,
    cancel: &AtomicBool,
    window: Option<RgbaWindow>,
    workers: Option<&ThreadPool>,
    uncapped: bool,
    ledger: &Ledger,
    gauge: &Gauge,
    pool: &mut Scratch,
) -> Result<Decoded, Error> {
    let scale = if o.scale == 0 { 1 } else { o.scale as usize };
    if !matches!(scale, 1 | 2 | 4 | 8) {
        return Err(Error::Unsupported);
    }
    let pyramid = o.pyramid && scale != 1;
    let region = o.region.filter(|r| !r.is_empty());
    if pyramid && (region.is_some() || o.format == Format::YCbCr) {
        return Err(Error::Unsupported);
    }
    if (o.recon.idct.is_some() || o.recon.to_rgba.is_some()) && (scale != 1 || region.is_some()) {
        return Err(Error::Unsupported); // Recon is 8x8 only, whole image
    }

    let mut frame = Frame::new();
    frame.parse(src, o.tables)?;
    frame.set_color_model(o.color_model);
    limits.check_pixels(frame.width, frame.height)?;
    let f = &mut frame;
    if pyramid && (f.width < scale || f.height < scale) {
        return Err(Error::Unsupported);
    }

    if f.ncomp == 3 && f.rgb {
        if o.format == Format::YCbCr {
            return Err(Error::Unsupported);
        }
        if f.comp[1].h != f.comp[0].h
            || f.comp[1].v != f.comp[0].v
            || f.comp[2].h != f.comp[0].h
            || f.comp[2].v != f.comp[0].v
        {
            return Err(Error::Unsupported); // non-uniform RGB sampling, not in the corpus
        }
    }

    let (hmax, vmax) = max_sampling(f);
    let min_size = 8 / scale;
    let raw = o.format == Format::YCbCr; // chroma uses Y's scaled IDCT

    let (w, h) = (f.width.div_ceil(scale), f.height.div_ceil(scale));
    let full = Rect::new(0, 0, w as i32, h as i32);
    if let Some(win) = &window
        && (o.format != Format::Rgba
            || scale != 1
            || region.is_some()
            || o.recon.idct.is_some()
            || o.recon.to_rgba.is_some()
            || win.stride < 4 * w
            || win.pix.len() < 4 * w
            || (h > 1 && win.stride > (win.pix.len() - 4 * w) / (h - 1)))
    {
        return Err(Error::Unsupported);
    }
    let mut eff = full;
    if let Some(r) = region {
        if r.min_x < -MAX_REGION
            || r.min_y < -MAX_REGION
            || r.max_x > MAX_REGION
            || r.max_y > MAX_REGION
        {
            return Err(Error::Region);
        }
        eff = r.intersect(full);
        if eff.is_empty() {
            return Err(Error::RegionEmpty);
        }
    }

    // Gray from YCbCr skips the chroma IDCT.
    let gray_only = f.ncomp == 1 || (o.format == Format::Gray && !f.rgb);

    // Mirrors the scan's own banding gate exactly: a frame whose first scan doesn't cover every
    // component decodes in full regardless of the region, so the Pix buffers stay full-size.
    let band_ok = !f.progressive && (f.ncomp == 1 || f.scan_n == f.ncomp);

    let restart = o.restart.unwrap_or(&[]);
    let mut band: Option<Band> = None;
    let mut band_rect = full;
    if region.is_some() && band_ok {
        let (b, rect) = plan_band(f, o, restart, eff, w, h, hmax, vmax, min_size);
        band = Some(b);
        band_rect = rect;
    }

    // need is computed before any allocation, from the same first/last rows the scan will use.
    let (mut first, mut last) = (0, f.height.div_ceil(vmax * 8) - 1);
    if let Some(b) = &band {
        first = b.first;
        last = b.last;
    }
    let (mut need, _, mut ext) = decode_need(
        f,
        hmax,
        vmax,
        min_size,
        raw,
        gray_only,
        first,
        last,
        o.format == Format::Rgba && window.is_none(),
        band_rect.width() as usize,
        band_rect.height() as usize,
    );
    if o.format == Format::Rgba {
        need += 2 * band_rect.width() as u64; // the cb and cr rows
        ext += 2 * band_rect.width() as u64;
    }
    let padded = region.is_some_and(|r| eff != r);
    if let (true, Some(r)) = (padded, region) {
        need += pad_need(o.format, f, gray_only, r);
    }
    limits.check_decode_bytes(need, ext, ledger)?;
    let cx = Ctx {
        cancel,
        ledger,
        threads: limits.threads(),
        pool: o.pool.or(workers),
        gauge: Some(gauge),
        uncapped,
    };

    let mut planes = [Plane::new(), Plane::new(), Plane::new()];
    for (ci, p) in planes.iter_mut().enumerate().take(f.ncomp) {
        let bs = if raw {
            min_size
        } else {
            component_block_size(
                f.comp[ci].h as usize,
                f.comp[ci].v as usize,
                hmax,
                vmax,
                min_size,
            )
        };
        p.skip = gray_only && ci > 0;
        if !p.skip {
            // The scan sizes it again; this only picks the buffer that fits.
            let c = f.comp[ci];
            let blocks =
                f.width.div_ceil(hmax * 8) * c.h as usize * ((last - first + 1) * c.v as usize);
            p.pix = take_buf(&mut pool.bufs, blocks * bs * bs);
        }
        if f.progressive || f.scan_n < f.ncomp {
            // The scan sizes it again; this only picks the buffer that fits.
            let c = f.comp[ci];
            let nblk =
                f.width.div_ceil(hmax * 8) * c.h as usize * ((last - first + 1) * c.v as usize);
            p.blocks = take_buf(&mut pool.blocks, nblk);
        }
        if let Some(h) = o.recon.idct {
            p.idct = Idct::Hook(h);
        }
        p.block_size = bs;
    }
    let nc = f.ncomp;
    let sr = decode_scan_band(&cx, f, src, &mut planes[..nc], band.as_ref());
    let truncated = sr.truncated;
    let mut err = match sr.err {
        Some(e @ Error::AfterScan { .. }) => Some(e),
        Some(e) => return Err(e),
        None => None,
    };
    if err.is_some() && (eff.max_y as usize) < h && !o.check_frame {
        err = None; // turbo skips the rows below a crop, and never reads the marker
    }

    let g = Geom {
        hmax,
        vmax,
        min_size,
        prefilter: match (pyramid, scale) {
            (true, 2) => 3,
            (true, 4) => 4,
            _ => 0,
        },
    };
    let seed = if o.format == Format::Rgba && window.is_none() {
        take_buf(
            &mut pool.bufs,
            4 * band_rect.width() as usize * band_rect.height() as usize,
        )
    } else {
        Vec::new()
    };
    let mut img = match o.format {
        Format::Rgba => {
            let hook = o.recon.to_rgba;
            Image::Rgba(build_rgba(
                &cx,
                f,
                &mut planes,
                gray_only,
                hook,
                band_rect,
                band.as_ref(),
                &g,
                window.map(|w| (w.pix, w.stride)),
                seed,
                &mut pool.rows,
            )?)
        }
        _ if gray_only => Image::Gray(gray_image(&mut planes[0], band_rect)),
        _ if f.rgb => {
            rgb_gray(&mut planes, cx.cancel)?;
            Image::Gray(gray_image(&mut planes[0], band_rect))
        }
        _ => {
            if scale == 8 && band_rect.max_y as usize == h {
                patch_yuv_overread(
                    &mut planes,
                    f.comp[0].h as usize,
                    vmax,
                    w,
                    band_rect.height() as usize,
                    cx.cancel,
                )?;
            }
            Image::YCbCr(ycbcr_image(f, &mut planes, band_rect))
        }
    };

    for p in &mut planes {
        if p.pix.capacity() > 0 && pool.bufs.len() < POOL_MAX {
            pool.bufs.push(std::mem::take(&mut p.pix)); // the RGBA path leaves its planes behind
        }
        if p.blocks.capacity() > 0 && pool.blocks.len() < POOL_MAX {
            pool.blocks.push(std::mem::take(&mut p.blocks));
        }
    }

    if let Some(r) = region {
        img = crop(img, eff);
        if padded {
            img = pad(&cx, img, r)?;
        }
    }
    if pyramid {
        let (pw, ph) = ((f.width / scale) as i32, (f.height / scale) as i32);
        img = crop(img, Rect::new(0, 0, pw, ph));
    }

    let error = match (err, truncated) {
        (Some(Error::AfterScan { cause, .. }), t) => Some(Error::AfterScan {
            cause,
            truncated: t,
        }),
        (Some(e), _) => Some(e),
        (None, true) => Some(Error::Truncated),
        (None, false) => None,
    };
    Ok(Decoded { image: img, error })
}

#[derive(Clone, Copy)]
pub(crate) struct Geom {
    pub hmax: usize,
    pub vmax: usize,
    pub min_size: usize,
    /// The shift of the pyramid chroma blur (3 or 4), 0 for none.
    pub prefilter: u32,
}

/// Fills the band `decode_scan_band` needs to cover `eff`: the region's own rows and columns,
/// plus one chroma row/column of context on each side when fancy upsampling applies (RGBA,
/// h2v2/h2v1). The returned rect is that band's row bounds in the same scaled output coordinates
/// as `eff`, at full width. A column band skips transforming MCUs outside it (and, with a DRI,
/// entropy-decoding them); the RGBA path then narrows its own buffer to the band's columns, while
/// gray and YCbCr alias the full-width planes.
#[allow(clippy::too_many_arguments)]
fn plan_band<'a>(
    f: &Frame,
    o: &DecodeOptions,
    restart: &'a [usize],
    eff: Rect,
    w: usize,
    h: usize,
    hmax: usize,
    vmax: usize,
    min_size: usize,
) -> (Band<'a>, Rect) {
    let out_step = vmax * min_size;
    let mcu_rows = f.height.div_ceil(vmax * 8);
    let mut first = eff.min_y as usize / out_step;
    let mut last = (eff.max_y as usize - 1) / out_step;
    let fancy = o.format == Format::Rgba && min_size > 1;
    if fancy && vmax == 2 {
        first = first.saturating_sub(1);
        if last < mcu_rows - 1 {
            last += 1;
        }
    }

    let out_step_x = hmax * min_size;
    let mcus_per_row = f.width.div_ceil(hmax * 8);
    let mut col_first = eff.min_x as usize / out_step_x;
    let mut col_last = (eff.max_x as usize - 1) / out_step_x;
    if fancy && hmax > 1 {
        col_first = col_first.saturating_sub(1);
        if col_last < mcus_per_row - 1 {
            col_last += 1;
        }
    }
    // a no-op band is skipped entirely
    let cols = col_first > 0 || col_last < mcus_per_row - 1;

    let band = Band {
        first,
        last,
        cols,
        col_first,
        col_last,
        restart,
        check_frame: o.check_frame,
    };
    let rect = Rect::new(
        0,
        (first * out_step) as i32,
        w as i32,
        ((last + 1) * out_step).min(h) as i32,
    );
    (band, rect)
}

/// The bytes a decode of MCU rows `first..=last` holds, from the headers (the planes, their
/// coefficients, the RGBA image), computed before any allocation. `out` is the bytes of the
/// returned image's buffers; `ext` is the page rounding only the ledger counts.
#[allow(clippy::too_many_arguments)]
fn decode_need(
    f: &Frame,
    hmax: usize,
    vmax: usize,
    min_size: usize,
    raw: bool,
    gray_only: bool,
    first: usize,
    last: usize,
    want_rgba: bool,
    rgba_w: usize,
    rgba_h: usize,
) -> (u64, u64, u64) {
    let mut need: u64 = 1 << 16; // tables, frame state, scratch
    let (mut out, mut ext) = (0u64, 0u64);
    for ci in 0..f.ncomp {
        let c = f.comp[ci];
        if f.progressive && gray_only && ci > 0 {
            // Refine scans read skipped planes.
            need += (f.width.div_ceil(hmax * 8) * c.h as usize) as u64
                * ((last - first + 1) * c.v as usize) as u64
                * 128;
        }
        if gray_only && ci > 0 {
            continue;
        }
        let bs = if raw {
            min_size
        } else {
            component_block_size(c.h as usize, c.v as usize, hmax, vmax, min_size)
        };
        let w = (f.width.div_ceil(hmax * 8) * c.h as usize) as u64;
        let h = ((last - first + 1) * c.v as usize) as u64;
        ext += 2 * 8192; // the plane and its coefficients each round up to whole 8 KiB pages
        need += 2 * 8192;
        out += w * h * (bs * bs) as u64;
        need += w * h * (bs * bs) as u64;
        if f.scan_n < f.ncomp || f.progressive {
            need += w * h * 128; // coefficients kept across scans
        }
    }
    if want_rgba {
        out = rgba_w as u64 * rgba_h as u64 * 4;
        need += out;
    }
    (need, out, ext)
}

/// The size of `pad`'s own buffer, mirroring its branches.
fn pad_need(format: Format, f: &Frame, gray_only: bool, r: Rect) -> u64 {
    let (w, h) = (r.width() as u64, r.height() as u64);
    if format == Format::Rgba {
        4 * w * h
    } else if gray_only || f.rgb {
        w * h
    } else {
        let (cw, ch) = chroma_rect(Subsample::from_sampling(f.comp[0].h, f.comp[0].v), r);
        w * h + 2 * cw as u64 * ch as u64
    }
}

/// `image.NewYCbCr`'s chroma plane size for `r`.
fn chroma_rect(ratio: Subsample, r: Rect) -> (usize, usize) {
    let (mut cw, mut ch) = (r.width(), r.height());
    match ratio {
        Subsample::S422 | Subsample::S420 => cw = (r.max_x + 1) / 2 - r.min_x / 2,
        Subsample::S411 | Subsample::S410 => cw = (r.max_x + 3) / 4 - r.min_x / 4,
        _ => {}
    }
    if matches!(ratio, Subsample::S420 | Subsample::S440 | Subsample::S410) {
        ch = (r.max_y + 1) / 2 - r.min_y / 2;
    }
    (cw as usize, ch as usize)
}

fn gray_image(p: &mut Plane, r: Rect) -> Gray {
    Gray {
        stride: p.stride(),
        pix: std::mem::take(&mut p.pix),
        rect: r,
    }
}

/// Wraps the IDCT'd planes as a YCbCr image bounded by `r`. Strides are block-padded and the
/// padding holds real IDCT output.
fn ycbcr_image(f: &Frame, planes: &mut [Plane; 3], r: Rect) -> YCbCr {
    YCbCr {
        y_stride: planes[0].stride(),
        c_stride: planes[1].stride(),
        y: std::mem::take(&mut planes[0].pix),
        cb: std::mem::take(&mut planes[1].pix),
        cr: std::mem::take(&mut planes[2].pix),
        ratio: Subsample::from_sampling(f.comp[0].h, f.comp[0].v),
        rect: r,
    }
}

/// Reproduces two `tj3DecompressToYUVPlanes8` quirks at Scale 8 for odd Y dimensions in
/// 4:2:0/4:2:2 (`turbojpeg.c:2251-2328`): a 4:2:0 iMCU row beyond Y's real height is never
/// refilled, so that dummy row holds whatever the previous iMCU row's same slot held; and Y's row
/// is exactly the unpadded width, one short of the YUV plane's padded width, for odd width, so
/// the memcpy of the padded width spills into the next row, or (the last Y row of an iMCU row)
/// into Cb's row for that iMCU row, both at column 0.
fn patch_yuv_overread(
    planes: &mut [Plane; 3],
    hsub: usize,
    vmax: usize,
    w: usize,
    h: usize,
    cancel: &AtomicBool,
) -> Result<(), Error> {
    let ys = planes[0].stride();
    let cs = planes[1].stride();
    let lh = h.div_ceil(vmax) * vmax;
    if lh > h && h > vmax {
        let (a, b) = ((h - vmax) * ys, h * ys);
        planes[0].pix.copy_within(a..a + w, b);
    }
    if hsub != 2 || w.is_multiple_of(2) {
        return Ok(());
    }
    for row in 0..lh {
        if row % 16 == 0 {
            check(cancel)?;
        }
        let (g, j) = (row / vmax, row % vmax);
        let v = if j < vmax - 1 {
            planes[0].pix[(row + 1) * ys]
        } else {
            planes[1].pix[g * cs]
        };
        planes[0].pix[row * ys + w] = v;
    }
    Ok(())
}

/// Combines an RGB stream's planes (uniform sampling, checked above) in place with turbo's
/// `rgb_gray_convert` weights.
fn rgb_gray(planes: &mut [Plane; 3], cancel: &AtomicBool) -> Result<(), Error> {
    let (y, rest) = planes.split_at_mut(1);
    let (g, b) = (&rest[0].pix, &rest[1].pix); // left in place, so the pool gets them back
    let band = 16 * y[0].stride().max(1); // polled every 16 rows
    for (k, chunk) in y[0].pix.chunks_mut(band).enumerate() {
        check(cancel)?;
        for (j, rp) in chunk.iter_mut().enumerate() {
            let i = k * band + j;
            *rp = ((*rp as i32 * 19595 + g[i] as i32 * 38470 + b[i] as i32 * 7471 + 32768) >> 16)
                as u8;
        }
    }
    Ok(())
}

/// Builds the RGBA image bounded by `r` from the IDCT'd planes: gray replication for a
/// 1-component stream, a null convert for an RGB stream, or upsampling plus colour conversion for
/// YCbCr. At Scale 1 a `to_rgba` hook, when set, takes the YCbCr seam.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_rgba(
    cx: &Ctx,
    f: &Frame,
    planes: &mut [Plane; 3],
    gray_only: bool,
    hook: Option<ToRgbaHook>,
    mut r: Rect,
    band: Option<&Band>,
    g: &Geom,
    window: Option<(&mut [u8], usize)>,
    seed: Vec<u8>,
    rows: &mut Vec<u8>,
) -> Result<Rgba, Error> {
    let Geom {
        hmax,
        vmax,
        min_size,
        prefilter,
    } = *g;
    let hooked = min_size == 8 && hook.is_some();
    let mut x0 = 0usize;
    if let Some(b) = band
        && b.cols
        && !hooked
    {
        // Only the band's MCU columns were transformed.
        let step = hmax * min_size;
        x0 = b.col_first * step;
        r.min_x = x0 as i32;
        r.max_x = ((b.col_last + 1) * step).min(r.max_x as usize) as i32;
    }
    let (w, h) = (r.width() as usize, r.height() as usize);
    let stride = w * 4;
    let n = stride * h;
    let mut rgba = Rgba {
        pix: Vec::new(),
        stride,
        rect: r,
    };
    if window.is_none() {
        cx.ledger.charge(n as u64)?;
        // A reused buffer keeps its bytes: the conversion writes all of them, but a caller's hook
        // may not.
        rgba.pix = refit(seed, n, hooked)?;
    }
    if hooked
        && !gray_only
        && !f.rgb
        && let Some(hk) = hook
    {
        cx.ledger.charge(2 * (w as u64 + 1))?; // the hook's scratch rows
        check(cx.cancel)?;
        let img = ycbcr_image(f, planes, r);
        hk(&mut rgba, &img);
        return Ok(rgba);
    }
    let (dst, dst_stride): (&mut [u8], usize) = match window {
        Some((p, s)) => (p, s),
        None => (&mut rgba.pix[..], stride),
    };
    if gray_only {
        let p = &planes[0];
        gray_to_rgba(dst, dst_stride, &p.pix[x0..], p.stride(), w, h, cx.cancel)?;
    } else if f.rgb {
        let st = planes[0].stride();
        rgb_to_rgba(
            dst,
            dst_stride,
            &planes[0].pix[x0..],
            &planes[1].pix[x0..],
            &planes[2].pix[x0..],
            st,
            w,
            h,
            cx.cancel,
        )?;
    } else {
        let bs1 = planes[1].bs();
        let (dsw, dsh) = downsampled_dims(f, 1, hmax, vmax, bs1);
        // A band starts partway into the chroma plane; dsh above is the full image's chroma
        // height, so shift it to band-local rows.
        let first = r.min_y as usize / (vmax * min_size);
        let dsh = dsh
            .saturating_sub(first * f.comp[1].v as usize * bs1)
            .min(planes[1].blocks_high * bs1);
        // The two conversion rows, then (with the prefilter) one of the chroma width for it.
        let pre = if prefilter > 0 { dsw } else { 0 };
        cx.ledger.charge(2 * w as u64 + pre as u64)?;
        *rows = refit(std::mem::take(rows), 2 * w + pre, true)?;
        let (cb_row, rest) = rows.split_at_mut(w);
        let (cr_row, pre_row) = rest.split_at_mut(w);
        let c1 = f.comp[1];
        let h_exp = hmax * min_size / (c1.h as usize * bs1);
        let v_exp = vmax * min_size / (c1.v as usize * bs1);
        if prefilter > 0 {
            let axes = (
                h_exp == 1 && (c1.h as usize) < hmax,
                v_exp == 1 && (c1.v as usize) < vmax,
            );
            let cs = planes[1].stride();
            for p in &mut planes[1..] {
                chroma_prefilter(
                    &mut p.pix,
                    cs,
                    (dsw, dsh),
                    axes,
                    prefilter,
                    pre_row,
                    cx.cancel,
                )?;
            }
        }
        // turbo decides on the whole width; h1v2 has no width test
        let fancy = min_size > 1 && (dsw > 2 || h_exp == 1);
        let cx0 = x0 * c1.h as usize * bs1 / (hmax * min_size);
        let cw = dsw.saturating_sub(cx0).min(w.div_ceil(h_exp));
        let (ys, cs) = (planes[0].stride(), planes[1].stride());
        let ycc = YccPlanes {
            y: &planes[0].pix[x0..],
            y_stride: ys,
            cb: &planes[1].pix[cx0..],
            cr: &planes[2].pix[cx0..],
            c_stride: cs,
        };
        let geom = YccGeom {
            w,
            h,
            dsw: cw,
            dsh,
            h_exp,
            v_exp,
            allow_fancy: fancy,
        };
        if cx.threads > 1 {
            ycbcr_upsample_convert_n(cx, dst, dst_stride, &ycc, &geom, cb_row, cr_row)?;
        } else {
            ycbcr_upsample_convert(dst, dst_stride, &ycc, &geom, cb_row, cr_row, cx.cancel)?;
        }
    }
    Ok(rgba)
}

/// `SubImage`'s logic: the part of `img` inside `eff`, the buffers' leading bytes dropped.
fn crop(img: Image, eff: Rect) -> Image {
    match img {
        Image::Gray(mut g) => {
            let off = g.pix_offset(eff.min_x, eff.min_y);
            g.pix.drain(..off);
            g.rect = eff;
            Image::Gray(g)
        }
        Image::Rgba(mut g) => {
            let off = g.pix_offset(eff.min_x, eff.min_y);
            g.pix.drain(..off);
            g.rect = eff;
            Image::Rgba(g)
        }
        Image::YCbCr(mut g) => {
            let (yo, co) = (
                g.y_offset(eff.min_x, eff.min_y),
                g.c_offset(eff.min_x, eff.min_y),
            );
            g.y.drain(..yo);
            g.cb.drain(..co);
            g.cr.drain(..co);
            g.rect = eff;
            Image::YCbCr(g)
        }
    }
}

/// Copies a cropped image into zeroed memory bounded by `r`, so `bounds()` equal the region even
/// where it leaves the image.
fn pad(cx: &Ctx, img: Image, r: Rect) -> Result<Image, Error> {
    let (w, h) = (r.width() as usize, r.height() as usize);
    match img {
        Image::Gray(im) => {
            let in_ = im.rect;
            let mut p = Gray {
                pix: cx.ledger.bytes(w * h)?,
                stride: w,
                rect: r,
            };
            for y in in_.min_y..in_.max_y {
                check(cx.cancel)?;
                let n = in_.width() as usize;
                let (so, dof) = (im.pix_offset(in_.min_x, y), p.pix_offset(in_.min_x, y));
                p.pix[dof..dof + n].copy_from_slice(&im.pix[so..so + n]);
            }
            Ok(Image::Gray(p))
        }
        Image::Rgba(im) => {
            let in_ = im.rect;
            let mut p = Rgba {
                pix: cx.ledger.bytes(4 * w * h)?,
                stride: 4 * w,
                rect: r,
            };
            for y in in_.min_y..in_.max_y {
                check(cx.cancel)?;
                let n = 4 * in_.width() as usize;
                let (so, dof) = (im.pix_offset(in_.min_x, y), p.pix_offset(in_.min_x, y));
                p.pix[dof..dof + n].copy_from_slice(&im.pix[so..so + n]);
            }
            Ok(Image::Rgba(p))
        }
        Image::YCbCr(im) => {
            let in_ = im.rect;
            let (cw, ch) = chroma_rect(im.ratio, r);
            let mut p = YCbCr {
                y: cx.ledger.bytes(w * h)?,
                cb: cx.ledger.bytes(cw * ch)?,
                cr: cx.ledger.bytes(cw * ch)?,
                y_stride: w,
                c_stride: cw,
                ratio: im.ratio,
                rect: r,
            };
            for y in in_.min_y..in_.max_y {
                check(cx.cancel)?;
                let n = in_.width() as usize;
                let (so, dof) = (im.y_offset(in_.min_x, y), p.y_offset(in_.min_x, y));
                p.y[dof..dof + n].copy_from_slice(&im.y[so..so + n]);
                for x in in_.min_x..in_.max_x {
                    let (sc, dc) = (im.c_offset(x, y), p.c_offset(x, y));
                    p.cb[dc] = im.cb[sc];
                    p.cr[dc] = im.cr[sc];
                }
            }
            Ok(Image::YCbCr(p))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn planes(rows: usize) -> [Plane; 3] {
        std::array::from_fn(|_| {
            let mut p = Plane::new();
            p.blocks_wide = 1; // stride 8
            p.pix = vec![0; 8 * rows];
            p
        })
    }

    /// The full-plane passes poll the cancel flag.
    #[test]
    fn plane_passes_poll_cancel() {
        let set = AtomicBool::new(true);
        let mut p = planes(40);
        assert_eq!(rgb_gray(&mut p, &set), Err(Error::Cancelled));
        let mut p = planes(41);
        assert_eq!(
            patch_yuv_overread(&mut p, 2, 2, 3, 40, &set),
            Err(Error::Cancelled)
        );
        let clear = AtomicBool::new(false);
        let mut p = planes(40);
        assert_eq!(rgb_gray(&mut p, &clear), Ok(()));
        assert_eq!(p[0].pix.len(), 8 * 40);
    }
}
