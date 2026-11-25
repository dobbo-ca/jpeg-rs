//! Entropy-decoding of sequential scans, ported from jpeg-go's `internal/codec/decode.go` (turbo
//! `jdhuff.c`, `jdinput.c`, `jdcoefct.c`). The Go port's assembly MCU decoder is not ported; the
//! output is the same. Its restart-interval bands are: a baseline scan with a DRI decodes in
//! bands on a thread pool (see `try_bands`), byte for byte the serial decode.

use crate::Error;
use crate::dct::{Block, dequantize, idct_bs};
use crate::huff::{BitReader, COMB_BITS, DerivedTable, huff_extend};
use crate::limits::{Ledger, check};
use crate::marker::{Frame, MARKER_EOI, ZIGZAG_TO_NATURAL};
use crate::par::{self, Gauge, ThreadPool};
use crate::simd;
use std::sync::atomic::AtomicBool;

/// turbo's `D_MAX_BLOCKS_IN_MCU`.
pub(crate) const MAX_MCU_BLOCKS: usize = 10;

/// The per-call context of a scan: the cancel flag, the allocation ledger and the threads the
/// call may use.
#[derive(Clone, Copy)]
pub(crate) struct Ctx<'a> {
    pub cancel: &'a AtomicBool,
    pub ledger: &'a Ledger,
    /// At most this many threads, the calling thread included; 1 never spawns.
    pub threads: usize,
    /// The caller's pool; `None` uses the crate's shared one when a pass is worth threading.
    pub pool: Option<&'a ThreadPool>,
    /// Counts the threads a call has running at once, when a caller wants to know.
    pub gauge: Option<&'a Gauge>,
    /// Lifts the size caps on the threads of a pass, so a test can thread a small frame.
    pub uncapped: bool,
}

impl<'a> Ctx<'a> {
    /// A context that never spawns a thread.
    pub(crate) fn serial(cancel: &'a AtomicBool, ledger: &'a Ledger) -> Self {
        Ctx {
            cancel,
            ledger,
            threads: 1,
            pool: None,
            gauge: None,
            uncapped: false,
        }
    }
}

/// A callback that replaces the IDCT: it gets one block of dequantized coefficients in natural
/// order and writes its 8x8 samples, level shifted and clamped, to `dst` at `stride`.
pub type IdctHook = fn(dst: &mut [u8], stride: usize, coef: &[i32; 64]);

/// How a plane's blocks are transformed.
#[derive(Clone, Copy)]
pub(crate) enum Idct {
    Default,
    Hook(IdctHook),
}

/// One frame component's output, padded to a whole number of MCUs. Each block is transformed
/// into `pix` as it is decoded (stride `blocks_wide * block_size`), unless `skip` is set (entropy
/// decode only) or `coef` (keep the raw coefficients in `blocks` instead).
pub(crate) struct Plane {
    pub skip: bool,
    pub coef: bool,
    pub idct: Idct,
    /// The IDCT output size (8, 4, 2 or 1); 0 means 8.
    pub block_size: usize,
    pub pix: Vec<u8>,
    pub blocks: Vec<Block>,
    pub blocks_wide: usize,
    pub blocks_high: usize,
    pub real_wide: usize,
    pub real_high: usize,
    pub quant: [u16; 64],
    /// quant copied at the first scan (turbo `latch_quant_tables`).
    pub latched: bool,
}

impl Plane {
    pub(crate) fn new() -> Plane {
        Plane {
            skip: false,
            coef: false,
            idct: Idct::Default,
            block_size: 0,
            pix: Vec::new(),
            blocks: Vec::new(),
            blocks_wide: 0,
            blocks_high: 0,
            real_wide: 0,
            real_high: 0,
            quant: [0; 64],
            latched: false,
        }
    }

    pub(crate) fn at(&self, bx: usize, by: usize) -> usize {
        by * self.blocks_wide + bx
    }

    pub(crate) fn bs(&self) -> usize {
        if self.block_size == 0 {
            8
        } else {
            self.block_size
        }
    }

    pub(crate) fn stride(&self) -> usize {
        self.blocks_wide * self.bs()
    }
}

/// Max horizontal/vertical sampling factors.
pub(crate) fn max_sampling(f: &Frame) -> (usize, usize) {
    let (mut hmax, mut vmax) = (1, 1);
    for c in &f.comp[..f.ncomp] {
        hmax = hmax.max(c.h as usize);
        vmax = vmax.max(c.v as usize);
    }
    (hmax, vmax)
}

/// A component's IDCT output size given the frame's `min_DCT_scaled_size`: a component sampled
/// below hmax/vmax gets a larger IDCT so its plane stays proportional to the full-size output, up
/// to 8x8.
pub(crate) fn component_block_size(
    h: usize,
    v: usize,
    hmax: usize,
    vmax: usize,
    min_size: usize,
) -> usize {
    let mut ssize = min_size;
    while ssize < 8 && h * ssize * 2 <= hmax * min_size && v * ssize * 2 <= vmax * min_size {
        ssize *= 2;
    }
    ssize
}

/// A component's real pixel width and height at block size `bs` (`downsampled_width/height`
/// under IDCT scaling).
pub(crate) fn downsampled_dims(
    f: &Frame,
    ci: usize,
    hmax: usize,
    vmax: usize,
    bs: usize,
) -> (usize, usize) {
    let c = f.comp[ci];
    let w = (f.width * c.h as usize * bs).div_ceil(hmax * 8);
    let h = (f.height * c.v as usize * bs).div_ceil(vmax * 8);
    (w, h)
}

/// Restricts decoding to MCU rows `first..=last` of the frame's max-sampling grid, growing planes
/// to only that band (region decode). When `cols` is set, MCUs outside columns
/// `col_first..=col_last` are not transformed. Without a DRI they are still entropy-decoded, for
/// DC prediction; with a DRI, restart intervals holding no needed MCU are skipped. `restart` is
/// the absolute `src` offset of each restart interval's first entropy byte (NDPI tag 65426); empty,
/// or an offset not just past its RSTn, means scan the entropy stream for RSTn instead.
pub(crate) struct Band<'a> {
    pub first: usize,
    pub last: usize,
    pub cols: bool,
    pub col_first: usize,
    pub col_last: usize,
    pub restart: &'a [usize],
    /// Entropy-decode every MCU, for whole-frame errors.
    pub check_frame: bool,
}

/// What a scan decode returns: Go's `(truncated, decoded, scanned, err)`.
pub(crate) struct ScanResult {
    pub truncated: bool,
    /// Go's region-read tests use the byte counts; no caller here does yet.
    #[allow(dead_code)]
    pub decoded: usize,
    #[allow(dead_code)]
    pub scanned: usize,
    pub err: Option<Error>,
}

impl ScanResult {
    pub(crate) fn fail(truncated: bool, err: Error) -> ScanResult {
        ScanResult {
            truncated,
            decoded: 0,
            scanned: 0,
            err: Some(err),
        }
    }
}

/// `jpeg_natural_order` padded to 128 entries, its 64..127 range mapped to 63 like the 16 real
/// overflow entries (`jdhuff.c:618`).
pub(crate) const NATURAL_ORDER_FAST: [u8; 128] = {
    let mut t = [63u8; 128];
    let mut i = 0;
    while i < 64 {
        t[i] = ZIGZAG_TO_NATURAL[i] as u8;
        i += 1;
    }
    t
};

/// turbo `latch_quant_tables`.
pub(crate) fn latch_quant(f: &Frame, planes: &mut [Plane]) -> Result<(), Error> {
    for k in 0..f.scan_n {
        let ci = f.scan_comp[k];
        let p = &mut planes[ci];
        if !p.latched {
            let tq = f.comp[ci].tq as usize;
            if tq > 3 || !f.has_quant[tq] {
                return Err(Error::NoQuantTable);
            }
            p.quant = f.quant[tq];
            p.latched = true;
        }
    }
    Ok(())
}

/// Sets the pixels past the real (unpadded) `rw` x `rh` extent to 128, the IDCT of the zero blocks
/// turbo keeps there; a non-interleaved scan never visits them.
pub(crate) fn fill_padding(p: &mut Plane, rw: usize, rh: usize) {
    let bs = p.bs();
    let stride = p.stride();
    for y in 0..p.blocks_high * bs {
        let from = if y >= rh { 0 } else { rw };
        p.pix[y * stride + from..(y + 1) * stride].fill(128);
    }
}

#[derive(Clone, Copy, Default)]
struct ScanComp {
    ci: usize,
    mcu_w: usize,
    mcu_h: usize,
}

#[derive(Clone, Copy, Default)]
struct McuSlot {
    ci: usize,
    /// From the MCU's top-left sample.
    off: usize,
    /// Per MCU column and row.
    col_step: usize,
    row_step: usize,
    /// The block's column and row inside its component's part of the MCU.
    xx: usize,
    yy: usize,
}

/// Sizes `planes` for a sequential scan decode and returns whether blocks persist across scans.
#[allow(clippy::too_many_arguments)]
fn prepare_planes(
    cx: &Ctx,
    f: &Frame,
    planes: &mut [Plane],
    hmax: usize,
    vmax: usize,
    first: usize,
    last: usize,
    multi: bool,
    zero: bool,
) -> Result<(), Error> {
    for ci in 0..planes.len() {
        let c = f.comp[ci];
        let w = f.width.div_ceil(hmax * 8) * c.h as usize;
        let h = (last - first + 1) * c.v as usize;
        let p = &mut planes[ci];
        let bs = p.bs();
        // Sizes in whole usize, checked before anything is charged (a 32-bit target can have a
        // budget over 4 GiB).
        let nblk = w.checked_mul(h).ok_or(Error::OutOfMemory)?;
        if p.coef || multi && !p.skip {
            cx.ledger.charge(nblk as u64 * 128)?;
            p.blocks = crate::limits::refill(std::mem::take(&mut p.blocks), nblk, [0i16; 64])?;
        }
        if !p.coef && !p.skip {
            let n = nblk.checked_mul(bs * bs).ok_or(Error::OutOfMemory)?;
            cx.ledger.charge(n as u64)?;
            // A reused buffer keeps its bytes: the scan writes every one (the IDCT of each block,
            // `fill_padding`, or the 128 fill of an unscanned plane) before the call returns.
            p.pix = crate::limits::refit(std::mem::take(&mut p.pix), n, zero)?;
        }
        p.blocks_wide = w;
        p.blocks_high = h;
        p.real_wide = (f.width * c.h as usize).div_ceil(hmax * 8);
        let full_real_high = (f.height * c.v as usize).div_ceil(vmax * 8);
        p.real_high = full_real_high.saturating_sub(first * c.v as usize).min(h);
        p.latched = false;
        if !p.skip && !p.coef {
            // Real pixel extent scales with bs (downsampled_width/height), not real_wide * bs:
            // rounding differs from the 8x8 block count when the image size isn't a multiple of
            // bs * hmax.
            let (rw, rh) = downsampled_dims(f, ci, hmax, vmax, bs);
            let rh = rh.saturating_sub(first * c.v as usize * bs).min(h * bs);
            fill_padding(p, rw, rh);
        }
    }
    Ok(())
}

/// Entropy-decodes `f`'s scan(s) for coefficient output (`planes` have `coef` set), reading to
/// EOI first as `jpeg_read_coefficients` does: a fatal marker after the last scan is the marker's
/// own error, not `AfterScan`.
pub(crate) fn decode_scan(
    cx: &Ctx,
    f: &mut Frame,
    src: &[u8],
    planes: &mut [Plane],
) -> (bool, Option<Error>) {
    if f.progressive {
        let r = crate::progressive::decode_progressive(cx, f, src, planes);
        return (r.truncated, r.err);
    }
    let r = decode_scan_inner(cx, f, src, planes, None);
    let err = match r.err {
        Some(Error::AfterScan { cause, .. }) => Some(*cause),
        e => e,
    };
    (r.truncated || f.overran, err)
}

/// `decode_scan` restricted to `band` (or every row when `None`). For the last scan, `decoded`
/// counts the entropy bytes the Huffman bit reader read and `scanned` the bytes searched for
/// markers.
pub(crate) fn decode_scan_band(
    cx: &Ctx,
    f: &mut Frame,
    src: &[u8],
    planes: &mut [Plane],
    band: Option<&Band>,
) -> ScanResult {
    if f.progressive {
        return crate::progressive::decode_progressive(cx, f, src, planes);
    }
    let mut r = decode_scan_inner(cx, f, src, planes, band);
    r.truncated = r.truncated || f.overran;
    r
}

fn decode_scan_inner(
    cx: &Ctx,
    f: &mut Frame,
    src: &[u8],
    planes: &mut [Plane],
    band: Option<&Band>,
) -> ScanResult {
    let (hmax, vmax) = max_sampling(f);
    let mcu_rows = f.height.div_ceil(vmax * 8);
    let (mut first, mut last) = (0, mcu_rows - 1);
    // Only a single scan covering every frame component (interleaved, or the lone scan of a
    // 1-component frame) shares one MCU-row grid; any other multi-scan frame is decoded in full
    // rather than banded on the first scan's grid alone.
    let banded = band.is_some() && (f.ncomp == 1 || f.scan_n == f.ncomp);
    let multi = f.scan_n < f.ncomp; // has_multiple_scans: blocks persist
    if banded && let Some(b) = band {
        first = b.first;
        last = b.last;
    }
    // A region decode leaves the MCUs outside its columns untouched, and a hook may not write
    // every byte of its block: those buffers start zeroed.
    let zero = band.is_some() || planes.iter().any(|p| !matches!(p.idct, Idct::Default));
    if let Err(e) = prepare_planes(cx, f, planes, hmax, vmax, first, last, multi, zero) {
        return ScanResult::fail(false, e);
    }
    f.install_std_huff_tables();

    let mut scanned = [false; 4];
    let mut offset = f.scan_offset;
    let mut truncated = false;
    let mut r;
    let mut first_scan = true;
    loop {
        // turbo's JWRN_NOT_SEQUENTIAL, then decodes as usual
        if f.scan_ss != 0 || f.scan_se != 63 || f.scan_ah != 0 || f.scan_al != 0 {
            truncated = true;
        }
        let scan = match Scan::new(f, hmax, vmax, planes, multi) {
            Ok(s) => s,
            Err(e) => return ScanResult::fail(true, e),
        };
        // Go tries bands on the first scan only; a frame that bands has just that one scan.
        let bands = if first_scan && band.is_none() {
            match try_bands(cx, src, &scan, planes) {
                Ok(b) => b.map(|b| (b, false)),
                Err(e) => return ScanResult::fail(true, e),
            }
        } else if first_scan
            && banded
            && let Some(b) = band
        {
            match try_region_bands(cx, src, &scan, planes, b) {
                Ok(b) => b,
                Err(e) => return ScanResult::fail(true, e),
            }
        } else {
            None
        };
        first_scan = false;
        let stopped = if let Some((b, stopped)) = bands {
            r = b;
            stopped
        } else {
            r = BitReader::new(src, offset);
            let mut views = plane_views(planes);
            match scan.run(cx, src, &mut views, &mut r, banded, first, last, band, None) {
                Ok(s) => s,
                Err(e) => return ScanResult::fail(true, e),
            }
        };
        for k in 0..f.scan_n {
            scanned[f.scan_comp[k]] = true;
        }
        if stopped {
            // Region done early: report data lost so far.
            truncated = truncated || r.eof || r.lost;
            break;
        }
        if !r.marker {
            // Last MCU fit in buffered bits (jdmarker.c next_marker).
            let (c, ok) = next_marker(&mut r);
            r.marker = ok;
            r.marker_byte = c;
        }
        truncated = truncated || r.insufficient || r.eof || r.lost;

        // A further SOS means more non-interleaved scans follow; anything else (EOI, or no marker
        // found) ends the frame.
        if !r.marker || r.marker_byte == MARKER_EOI {
            break;
        }
        match f.consume_from_marker(src, r.marker_byte, r.pos, true) {
            Err(e) => {
                // Not just end of frame: turbo's next_marker ERREXITs fatally on a marker it
                // doesn't recognize (jdmarker.c:742).
                let e = if !multi {
                    Error::AfterScan {
                        cause: Box::new(e),
                        truncated: false,
                    }
                } else {
                    e
                };
                return ScanResult {
                    truncated,
                    decoded: r.decoded,
                    scanned: r.scanned,
                    err: Some(e),
                };
            }
            Ok((false, _)) => break,
            Ok((true, next)) => {
                if !multi {
                    return ScanResult {
                        truncated,
                        decoded: r.decoded,
                        scanned: r.scanned,
                        err: Some(Error::AfterScan {
                            cause: Box::new(Error::EoiExpected),
                            truncated: false,
                        }),
                    };
                }
                offset = next;
            }
        }
    }
    for ci in 0..f.ncomp {
        if scanned[ci] {
            continue;
        }
        let p = &mut planes[ci];
        if !p.skip && !p.coef {
            p.pix.fill(128); // IDCT of an all-zero block
        }
    }
    ScanResult {
        truncated,
        decoded: r.decoded,
        scanned: r.scanned,
        err: None,
    }
}

/// What a plane contributes to a scan, copied out of its [`Plane`] so that a thread decoding a
/// band needs no `&Plane`.
#[derive(Clone, Copy)]
struct PlaneMeta {
    skip: bool,
    coef: bool,
    idct: Idct,
    bs: usize,
    stride: usize,
    blocks_wide: usize,
    quant: [u16; 64],
}

impl PlaneMeta {
    fn of(p: &Plane) -> PlaneMeta {
        PlaneMeta {
            skip: p.skip,
            coef: p.coef,
            idct: p.idct,
            bs: p.bs(),
            stride: p.stride(),
            blocks_wide: p.blocks_wide,
            quant: p.quant,
        }
    }

    fn at(&self, bx: usize, by: usize) -> usize {
        by * self.blocks_wide + bx
    }
}

/// The parts of the planes a scan writes: pixels, and the kept blocks of a coefficient or
/// multi-scan decode. A band thread gets only its own rows of `pix` and no blocks.
struct Views<'p> {
    pix: [&'p mut [u8]; 4],
    blocks: [&'p mut [Block]; 4],
}

fn plane_views(planes: &mut [Plane]) -> Views<'_> {
    let mut v = Views {
        pix: [&mut [], &mut [], &mut [], &mut []],
        blocks: [&mut [], &mut [], &mut [], &mut []],
    };
    for (i, p) in planes.iter_mut().enumerate() {
        v.pix[i] = p.pix.as_mut_slice();
        v.blocks[i] = p.blocks.as_mut_slice();
    }
    v
}

/// The scan currently described by `f.scan_*`, set up: quant tables latched, Huffman tables
/// derived, and the MCU's block layout. It is read-only, so band threads share one.
struct Scan<'f> {
    f: &'f Frame,
    interleaved: bool,
    comps: [ScanComp; 4],
    mcus_per_row: usize,
    mcu_rows: usize,
    dc: [DerivedTable; 4],
    ac: [DerivedTable; 4],
    meta: [PlaneMeta; 4],
    multi: bool,
    /// Every block is a scratch block (`decompress_onepass`).
    plain: bool,
    /// Each component's `McuSlot::row_step`: its plane's bytes per MCU row.
    row_step: [usize; 4],
    /// `comps[blk_k[i]]` is MCU block `i`'s component, in the order the loop fills them; the
    /// sampling pattern is the same for every MCU.
    blk_k: [usize; MAX_MCU_BLOCKS],
    slots: [McuSlot; MAX_MCU_BLOCKS],
    n_blk: usize,
}

impl<'f> Scan<'f> {
    fn new(
        f: &'f Frame,
        hmax: usize,
        vmax: usize,
        planes: &mut [Plane],
        multi: bool,
    ) -> Result<Scan<'f>, Error> {
        let interleaved = f.scan_n > 1;
        let mut comps = [ScanComp::default(); 4];
        for k in 0..f.scan_n {
            let ci = f.scan_comp[k];
            comps[k] = if interleaved {
                ScanComp {
                    ci,
                    mcu_w: f.comp[ci].h as usize,
                    mcu_h: f.comp[ci].v as usize,
                }
            } else {
                ScanComp {
                    ci,
                    mcu_w: 1,
                    mcu_h: 1,
                }
            };
        }
        let (mcus_per_row, mcu_rows) = if interleaved {
            (f.width.div_ceil(hmax * 8), f.height.div_ceil(vmax * 8))
        } else {
            let c = f.comp[comps[0].ci];
            (
                (f.width * c.h as usize).div_ceil(hmax * 8),
                (f.height * c.v as usize).div_ceil(vmax * 8),
            )
        };

        // turbo start_input_pass: latch quant, then Huffman tables.
        crate::scan::latch_quant(f, planes)?;
        let mut dc = [DerivedTable::EMPTY; 4];
        let mut ac = [DerivedTable::EMPTY; 4];
        for k in 0..f.scan_n {
            if !f.has_dc[f.scan_dc[k] as usize] || !f.has_ac[f.scan_ac[k] as usize] {
                return Err(Error::NoHuffTable);
            }
            // Same selector, same table.
            let mut j = 0;
            while j < k && f.scan_dc[j] != f.scan_dc[k] {
                j += 1;
            }
            if j < k {
                dc[k] = dc[j];
            } else {
                dc[k] = DerivedTable::build(&f.dc[f.scan_dc[k] as usize], true)?;
            }
            j = 0;
            while j < k && f.scan_ac[j] != f.scan_ac[k] {
                j += 1;
            }
            if j < k {
                ac[k] = ac[j];
            } else {
                ac[k] = DerivedTable::build(&f.ac[f.scan_ac[k] as usize], false)?;
            }
        }

        let meta: [PlaneMeta; 4] = std::array::from_fn(|i| {
            planes
                .get(i)
                .map_or_else(|| PlaneMeta::of(&Plane::new()), PlaneMeta::of)
        });
        let mut blk_k = [0usize; MAX_MCU_BLOCKS];
        let mut slots = [McuSlot::default(); MAX_MCU_BLOCKS];
        let mut row_step = [0usize; 4];
        let mut plain = !multi;
        let mut n_blk = 0;
        for (k, sc) in comps.iter().enumerate().take(f.scan_n) {
            let m = &meta[sc.ci];
            plain = plain && !m.coef;
            let (bs, stride) = (m.bs, m.stride);
            row_step[sc.ci] = sc.mcu_h * bs * stride;
            for yy in 0..sc.mcu_h {
                for xx in 0..sc.mcu_w {
                    blk_k[n_blk] = k;
                    slots[n_blk] = McuSlot {
                        ci: sc.ci,
                        off: yy * bs * stride + xx * bs,
                        col_step: sc.mcu_w * bs,
                        row_step: sc.mcu_h * bs * stride,
                        xx,
                        yy,
                    };
                    n_blk += 1;
                }
            }
        }
        Ok(Scan {
            f,
            interleaved,
            comps,
            mcus_per_row,
            mcu_rows,
            dc,
            ac,
            meta,
            multi,
            plain,
            row_step,
            blk_k,
            slots,
            n_blk,
        })
    }

    /// Whether this scan can be decoded in restart-interval bands (Go's `bandable`): one
    /// baseline scan over every component, with a DRI, writing only 8x8 `pix` blocks, which
    /// never touch a neighbouring MCU's bytes.
    fn bandable(&self) -> bool {
        let f = self.f;
        !f.progressive
            && f.scan_n == f.ncomp
            && f.restart_interval != 0
            && f.scan_ss == 0
            && f.scan_se == 63
            && f.scan_ah == 0
            && f.scan_al == 0
            && self
                .meta
                .iter()
                .take(f.ncomp)
                .all(|m| !m.coef && matches!(m.idct, Idct::Default) && (m.skip || m.bs == 8))
    }

    /// Decodes the scan from `r` (positioned at its first entropy byte by the caller), writing
    /// into `views` indexed by frame component. When `banded`, `first`/`last` are the MCU-row
    /// band in this scan's own grid: MCUs outside the band are entropy decoded but not
    /// transformed (DC prediction must stay correct), or skipped a restart interval at a time
    /// with a DRI, and decoding stops after the band's last needed MCU. With `span` (restart
    /// intervals `k0..k1`, aligned to MCU rows, `r` at the first one's entropy byte, `views`
    /// holding just those rows) it decodes only those intervals. Returns whether it ended before
    /// the scan's last MCU.
    fn run<'a>(
        &self,
        cx: &Ctx,
        src: &'a [u8],
        views: &mut Views,
        r: &mut BitReader<'a>,
        banded: bool,
        mut first: usize,
        mut last: usize,
        band: Option<&Band>,
        span: Option<(usize, usize)>,
    ) -> Result<bool, Error> {
        let f = self.f;
        let (mcus_per_row, mcu_rows) = (self.mcus_per_row, self.mcu_rows);
        let (interleaved, comps, multi, plain) =
            (self.interleaved, &self.comps, self.multi, self.plain);
        let (dc, ac, slots, blk_k, n_blk) =
            (&self.dc, &self.ac, &self.slots, &self.blk_k, self.n_blk);

        // band.first/last are frame-grid MCU-row indices (each spanning vmax component block-rows),
        // but a non-interleaved scan's own "MCU" is a single component block: a single-component
        // frame (the only case reaching here banded) still has hmax == vmax == that component's own
        // H/V, so converting by its V factor maps the frame-grid band onto this scan's block rows.
        let mut col_first = 0usize;
        let mut col_last: Option<usize> = None; // None: no column restriction
        if banded
            && let Some(b) = band
            && b.cols
        {
            col_first = b.col_first;
            col_last = Some(b.col_last);
        }
        if banded && !interleaved {
            let v = f.comp[comps[0].ci].v as usize;
            first *= v;
            last = (last + 1) * v - 1;
            if let Some(cl) = col_last {
                let h = f.comp[comps[0].ci].h as usize;
                col_first *= h;
                col_last = Some((cl + 1) * h - 1);
            }
        }

        let mut last_dc = [0i32; 4];
        let mut scratch = [[0i16; 64]; MAX_MCU_BLOCKS];
        let mut saved = [[0i16; 64]; MAX_MCU_BLOCKS];
        let mut has_ac = [false; MAX_MCU_BLOCKS];

        let ri = f.restart_interval;
        let mut restarts_to_go = ri;
        let mut next_restart = 0usize;
        let mut interval = 0usize; // restarts processed
        let mut mk: i64 = -1; // nextBail from r.pos, cached
        let mut mk_real = false;

        let mut last_row = mcu_rows - 1;
        if banded {
            last_row = last.min(mcu_rows - 1);
        }
        let mut end = (last_row + 1) * mcus_per_row;
        let check_frame = banded && band.is_some_and(|b| b.check_frame);
        if check_frame {
            end = mcu_rows * mcus_per_row;
        } else if let Some(cl) = col_last
            && cl < mcus_per_row - 1
            && last_row < mcu_rows - 1
        {
            // turbo's crop decodes all of the last row, then the marker after it
            end = last_row * mcus_per_row + cl + 1;
        }
        let restart: &[usize] = match band {
            Some(b) => b.restart,
            None => &[],
        };

        let mut polled = usize::MAX;
        let mut tried = usize::MAX;
        let mut m = 0usize;
        if let Some((k0, k1)) = span {
            // Lands as `skip_to` does: interval `k0`'s first MCU, its RSTn already consumed.
            m = k0 * ri;
            end = end.min(k1 * ri);
            interval = k0;
            next_restart = k0 & 7;
        }
        // The MCU row `views` starts at.
        let row0 = first + span.map_or(0, |(k0, _)| k0 * ri / mcus_per_row);
        // `m` as its MCU row and column, kept in step so the loop does no division.
        let (mut my, mut mx) = (m / mcus_per_row, m % mcus_per_row);
        // Where each block of a written MCU persists in its plane (Coef planes and multi-scan
        // frames keep blocks across MCUs and scans), else `usize::MAX`: a scratch block, zeroed
        // per MCU. Set for every written MCU of a scan that is not `plain`.
        let mut persist = [usize::MAX; MAX_MCU_BLOCKS];
        while m < end {
            if my != polled {
                polled = my;
                check(cx.cancel)?;
            }
            let write = !banded
                || my >= first
                    && my <= last_row
                    && col_last.is_none_or(|cl| mx >= col_first && mx <= cl);
            if !write && ri != 0 && !check_frame {
                let mut next = my.max(first) * mcus_per_row + col_first;
                if my >= first && mx > col_first {
                    next += mcus_per_row; // right of the band: next row
                }
                let t = next / ri;
                if t > interval && next != tried && next < end {
                    tried = next;
                    if let Some(k) = skip_to(src, r, restart, interval, t) {
                        interval = k;
                        next_restart = k & 7;
                        restarts_to_go = ri;
                        last_dc = [0; 4];
                        mk = -1;
                        m = k * ri;
                        (my, mx) = (m / mcus_per_row, m % mcus_per_row);
                        continue;
                    }
                }
            }
            if ri != 0 && restarts_to_go == 0 {
                process_restart(r, &mut last_dc[..f.scan_n], &mut next_restart);
                restarts_to_go = ri;
                interval += 1;
            }
            let n = n_blk;
            if write && !plain {
                for i in 0..n {
                    let sc = comps[blk_k[i]];
                    let pm = &self.meta[sc.ci];
                    if pm.coef || multi && !pm.skip {
                        let idx = pm.at(
                            mx * sc.mcu_w + slots[i].xx,
                            (my - first) * sc.mcu_h + slots[i].yy,
                        );
                        persist[i] = idx;
                        scratch[i] = views.blocks[sc.ci][idx];
                    } else {
                        persist[i] = usize::MAX;
                        scratch[i] = [0; 64]; // jdcoefct.c decompress_onepass zeroes each MCU
                    }
                }
            }
            // turbo's 512-bytes-per-block margin keeps short tails exact.
            let mut fast = !r.marker && src.len() as i64 - r.pos as i64 >= 512 * n as i64;
            let mut undo = false; // try fast, roll back on bail
            if ri != 0 {
                fast = false;
                if !r.marker {
                    if r.pos as i64 > mk {
                        let (a, b) = next_bail(src, r.pos);
                        mk = a as i64;
                        mk_real = b;
                    }
                    // Never bails here, so equals slow.
                    fast = mk_real && mk - r.pos as i64 >= 512 * n as i64;
                    // Real marker ahead: eof stays unset. Last MCU almost always bails: go slow.
                    undo = mk_real && !fast && restarts_to_go != 1;
                }
            } else if !fast && !r.marker {
                // The tail of a scan with no restarts: a real marker ahead makes the fast path
                // exact until it bails, as between two restarts; with none (a truncated stream)
                // the slow path decodes.
                if r.pos as i64 > mk {
                    let (a, b) = next_bail(src, r.pos);
                    mk = a as i64;
                    mk_real = b;
                }
                undo = mk_real;
            }
            if write {
                if !r.insufficient {
                    // else turbo leaves the MCU unchanged
                    let blks = &mut scratch[..n];
                    let ha = &mut has_ac[..n];
                    if undo {
                        let (rs, dcs) = (*r, last_dc);
                        if !plain {
                            saved[..n].copy_from_slice(blks);
                        }
                        if !decode_mcu_fast(
                            r,
                            dc,
                            ac,
                            &mut last_dc,
                            Some(&mut *blks),
                            &blk_k[..n],
                            Some(&mut *ha),
                        ) {
                            *r = rs;
                            last_dc = dcs;
                            for i in 0..n {
                                blks[i] = if plain { [0; 64] } else { saved[i] };
                            }
                            decode_mcu_slow(
                                r,
                                dc,
                                ac,
                                &mut last_dc,
                                Some(&mut *blks),
                                &blk_k[..n],
                                Some(&mut *ha),
                            );
                        }
                    } else if !fast {
                        decode_mcu_slow(
                            r,
                            dc,
                            ac,
                            &mut last_dc,
                            Some(&mut *blks),
                            &blk_k[..n],
                            Some(&mut *ha),
                        );
                    } else if !decode_mcu_fast(
                        r,
                        dc,
                        ac,
                        &mut last_dc,
                        Some(&mut *blks),
                        &blk_k[..n],
                        Some(&mut *ha),
                    ) {
                        decode_mcu_slow(
                            r,
                            dc,
                            ac,
                            &mut last_dc,
                            Some(&mut *blks),
                            &blk_k[..n],
                            Some(&mut *ha),
                        );
                        ha.fill(true); // the failed attempt's stores stay
                    }
                }
            } else if !r.insufficient {
                // No block in this MCU is ever read: decode with no coefficient store at all.
                if undo {
                    let (rs, dcs) = (*r, last_dc);
                    if !decode_mcu_fast(r, dc, ac, &mut last_dc, None, &blk_k[..n], None) {
                        *r = rs;
                        last_dc = dcs;
                        decode_mcu_slow(r, dc, ac, &mut last_dc, None, &blk_k[..n], None);
                    }
                } else if !fast
                    || !decode_mcu_fast(r, dc, ac, &mut last_dc, None, &blk_k[..n], None)
                {
                    decode_mcu_slow(r, dc, ac, &mut last_dc, None, &blk_k[..n], None);
                }
            }
            if write {
                idct_mcu(
                    &self.meta,
                    &mut views.pix,
                    slots,
                    &scratch,
                    &has_ac,
                    n,
                    mx,
                    my - row0,
                    multi,
                );
                if !plain {
                    for i in 0..n {
                        if persist[i] != usize::MAX {
                            views.blocks[slots[i].ci][persist[i]] = scratch[i];
                        }
                    }
                }
                if plain {
                    // Restore zero blocks for the next MCU; an AC-free block only holds DC.
                    for i in 0..n {
                        if has_ac[i] {
                            scratch[i] = [0; 64];
                        } else {
                            scratch[i][0] = 0;
                        }
                    }
                }
            }
            restarts_to_go = restarts_to_go.wrapping_sub(1);
            m += 1;
            mx += 1;
            if mx == mcus_per_row {
                mx = 0;
                my += 1;
            }
        }
        let stopped = end < mcu_rows * mcus_per_row;
        if stopped && ri != 0 && (interval + 1) * ri < mcu_rows * mcus_per_row {
            // Turbo reads this RSTn next; a resync there warns.
            let mut s = *r;
            let mut want = next_restart;
            if read_restart_marker(&mut s, &mut want) {
                r.lost = true;
            }
            r.scanned = s.scanned;
        }
        Ok(stopped)
    }
}

/// Transforms MCU (`mx`, `my`)'s decoded blocks into their planes. Stale `has_ac` flags are safe:
/// onepass blocks start zeroed.
#[allow(clippy::too_many_arguments)]
fn idct_mcu(
    meta: &[PlaneMeta; 4],
    pix: &mut [&mut [u8]; 4],
    slots: &[McuSlot; MAX_MCU_BLOCKS],
    mcu: &[Block; MAX_MCU_BLOCKS],
    has_ac: &[bool; MAX_MCU_BLOCKS],
    n: usize,
    mx: usize,
    my: usize,
    multi: bool,
) {
    for i in 0..n {
        let s = &slots[i];
        let m = &meta[s.ci];
        if m.skip || m.coef {
            continue;
        }
        let off = my * s.row_step + mx * s.col_step + s.off;
        let dst = &mut pix[s.ci][off..];
        match m.idct {
            Idct::Hook(h) => {
                let mut deq = [0i32; 64];
                for (j, &c) in mcu[i].iter().enumerate() {
                    deq[j] = dequantize(c, m.quant[j]) as i32;
                }
                h(dst, m.stride, &deq);
            }
            Idct::Default => idct_bs(m.bs, &mcu[i], has_ac[i] || multi, &m.quant, dst, m.stride),
        }
    }
}

/// The first entropy byte of each of the `n` restart intervals from `off`, or `None` unless the
/// `n - 1` markers between them are RST0..7 in order. The bit reader cannot pass a marker and
/// lexes FF00 and FF fill as `scan_marker` does, so a serial decode meets these same markers
/// with no resync, and leaves interval `k` in the state a band starts from (DC zero, no bits,
/// `insufficient` clear: `process_restart`).
// ponytail: serial scan; split it across the pool only if a CI timing of it on strip_rst is at
// least 5% of the threaded dec-gray time (the lexer's marker test is local, so chunks merge).
fn restart_starts(src: &[u8], off: usize, n: usize) -> Option<Vec<usize>> {
    if n < 2 || off > src.len() {
        return None;
    }
    let mut starts = Vec::new();
    starts.try_reserve_exact(n).ok()?;
    starts.push(off);
    let mut pos = off;
    for k in 1..n {
        loop {
            pos += find_ff(&src[pos..])? + 1;
            while pos < src.len() && src[pos] == 0xff {
                pos += 1;
            }
            let c = *src.get(pos)?;
            pos += 1;
            if c == 0 {
                continue;
            }
            if c != 0xd0u8.wrapping_add(((k - 1) & 7) as u8) {
                return None;
            }
            break;
        }
        starts.push(pos);
    }
    Some(starts)
}

/// Go's `decodeBands`: decodes a bandable scan's restart intervals in up to `cx.threads`
/// contiguous bands of about equal entropy bytes on the call's pool (the calling thread works
/// too), and returns the last band's reader. `None` means decode serially: the scan has too few
/// bytes for a second worker, a band cannot be cut, the memory for the bands does not fit the
/// ledger, or any band lost data or did not end on its RSTn (the serial decode then rewrites
/// every MCU). The memory is held only for this call, so every later charge sees the ledger a
/// single-threaded call would.
///
/// A band is a run of whole MCU rows, so its rows of each plane are one contiguous slice. Cuts
/// go at the first interval at or after Go's byte-balanced cut that starts an MCU row. Any cut
/// gives the same bytes, since each band starts from the state a restart resets.
fn try_bands<'a>(
    cx: &Ctx,
    src: &'a [u8],
    scan: &Scan,
    planes: &mut [Plane],
) -> Result<Option<BitReader<'a>>, Error> {
    let f = scan.f;
    if cx.threads < 2 || !scan.bandable() {
        return Ok(None);
    }
    let (ri, mpr) = (f.restart_interval, scan.mcus_per_row);
    let n = (mpr * scan.mcu_rows).div_ceil(ri);
    let cap = if cx.uncapped {
        usize::MAX
    } else {
        par::band_threads(src.len().saturating_sub(f.scan_offset))
    };
    if cx.threads.min(n).min(cap) < 2 {
        return Ok(None);
    }
    // The pool starts here, once the scan is known to be worth threading.
    let Some((pool, nw)) = par::stage(cx) else {
        return Ok(None);
    };
    let k = nw.min(n).min(cap);
    if k < 2 {
        return Ok(None);
    }
    // A pool's thread stacks are the pool's: only the call's own scratch is booked.
    let Some(_held) = cx.ledger.hold(8 * n as u64) else {
        return Ok(None);
    };
    let Some(starts) = restart_starts(src, f.scan_offset, n) else {
        return Ok(None);
    };
    let mut cuts = vec![0usize];
    let mut prev = 0;
    for b in 1..k {
        let at = starts[0] + (src.len() - starts[0]) / k * b;
        let mut c = starts.partition_point(|&s| s < at);
        while c < n && (c * ri) % mpr != 0 {
            c += 1;
        }
        if c > prev && c < n {
            cuts.push(c);
            prev = c;
        }
    }
    cuts.push(n);
    let nb = cuts.len() - 1;
    if nb < 2 {
        return Ok(None);
    }

    let rows: Vec<usize> = cuts
        .windows(2)
        .map(|w| w[1] * ri / mpr - w[0] * ri / mpr)
        .collect();
    let jobs: Vec<_> = split_rows(planes, scan, &rows)
        .into_iter()
        .enumerate()
        .map(|(i, pix)| (cuts[i], cuts[i + 1], pix))
        .collect();
    let done = par::run(pool, cx.gauge, k, jobs, |(k0, k1, pix)| {
        let mut r = BitReader::new(src, starts[k0]);
        let mut views = Views {
            pix,
            blocks: [&mut [], &mut [], &mut [], &mut []],
        };
        scan.run(
            cx,
            src,
            &mut views,
            &mut r,
            false,
            0,
            0,
            None,
            Some((k0, k1)),
        )
        .map(|_| r)
    });
    let (mut ok, mut last) = (true, None);
    for r in done {
        let r = r?;
        ok = ok && !r.lost && !r.insufficient && !r.eof;
        last = Some(r);
    }
    Ok(if ok { last } else { None })
}

/// Cuts each plane's rows into one slice per band, `rows[i]` MCU rows (of the scan's own grid)
/// for band `i`; the last band takes whatever is left.
fn split_rows<'p>(planes: &'p mut [Plane], scan: &Scan, rows: &[usize]) -> Vec<[&'p mut [u8]; 4]> {
    let nb = rows.len();
    let mut rest: [&mut [u8]; 4] = [&mut [], &mut [], &mut [], &mut []];
    for (ci, p) in planes.iter_mut().enumerate() {
        if !scan.meta[ci].skip {
            rest[ci] = p.pix.as_mut_slice();
        }
    }
    let mut out = Vec::with_capacity(nb);
    for (i, &n) in rows.iter().enumerate() {
        let mut pix: [&mut [u8]; 4] = [&mut [], &mut [], &mut [], &mut []];
        for ci in 0..4 {
            let cur = std::mem::take(&mut rest[ci]);
            if i + 1 < nb {
                let (head, tail) = cur.split_at_mut((n * scan.row_step[ci]).min(cur.len()));
                pix[ci] = head;
                rest[ci] = tail;
            } else {
                pix[ci] = cur;
            }
        }
        out.push(pix);
    }
    out
}

/// Bands for a region decode of a stream with restart `offsets` (`Band::restart`), the same
/// threading as [`try_bands`] but for the rows `b` selects. Returns the last band's reader and
/// whether it stopped before the scan's end, or `None` to decode serially: the region is as wide
/// as the frame, no row can be cut, or any band lost data or ended short
/// (the serial decode then redoes every MCU).
///
/// A cut at row `r` is exact when the serial decode would jump to interval `t` at that point
/// (`t` is the interval of the row's first needed MCU; it jumps because the MCUs left of the
/// columns, or right of them in the row above, are skipped) and that jump lands on `offsets[t]`
/// (`landing_ok`). A band that starts at the scan's first byte with `first = r` makes the same
/// jump from interval 0 and lands on the same byte. A cut where `t` is still the interval the
/// serial decode is in is not taken. Full-width regions are not banded: the serial decode
/// reads their restart markers from the stream rather than jumping, which equals an offset
/// only if the offset is truthful.
fn try_region_bands<'a>(
    cx: &Ctx,
    src: &'a [u8],
    scan: &Scan,
    planes: &mut [Plane],
    b: &Band,
) -> Result<Option<(BitReader<'a>, bool)>, Error> {
    let f = scan.f;
    if cx.threads < 2 || !b.cols || b.check_frame || b.restart.is_empty() || !scan.bandable() {
        return Ok(None);
    }
    let (ri, mpr) = (f.restart_interval, scan.mcus_per_row);
    // The scan's own block rows and columns per frame MCU row and column: a lone component's
    // scan has one block per MCU (`run` scales `first`, `last` and the columns the same way).
    let (sv, sh) = if scan.interleaved {
        (1, 1)
    } else {
        let c = f.comp[scan.comps[0].ci];
        (c.v as usize, c.h as usize)
    };
    let cf = b.col_first.saturating_mul(sh);
    let cl = b
        .col_last
        .saturating_add(1)
        .saturating_mul(sh)
        .saturating_sub(1)
        .min(mpr - 1);
    let last = b.last.min((scan.mcu_rows - 1) / sv);
    if cf >= mpr || (cf == 0 && cl == mpr - 1) || b.first >= last {
        return Ok(None);
    }
    let off = |k: usize| b.restart.get(k).map_or(src.len(), |&o| o.min(src.len()));
    // The first interval a frame row needs, and the bytes of the intervals it spans.
    let t_of = |r: usize| (r * sv * mpr + cf) / ri;
    let row_bytes =
        |r: usize| off(((r * sv + sv - 1) * mpr + cl) / ri + 1).saturating_sub(off(t_of(r)));
    let total = (b.first..=last)
        .map(row_bytes)
        .fold(0usize, usize::saturating_add);
    let cap = if cx.uncapped {
        usize::MAX
    } else {
        par::band_threads(total)
    };
    if cx.threads.min(cap) < 2 {
        return Ok(None);
    }
    let Some((pool, nw)) = par::stage(cx) else {
        return Ok(None);
    };
    let k = nw.min(cap).min(last - b.first + 1);
    if k < 2 {
        return Ok(None);
    }
    let mut cuts = vec![b.first];
    let (mut cum, mut want) = (0usize, 1usize);
    for r in b.first..=last {
        if r > b.first && want < k && cum >= total / k * want {
            let t = t_of(r);
            if t > ((r * sv - 1) * mpr + cl) / ri && landing_ok(src, b.restart, t) {
                cuts.push(r);
                while want < k && cum >= total / k * want {
                    want += 1;
                }
            }
        }
        cum = cum.saturating_add(row_bytes(r));
    }
    let nb = cuts.len();
    if nb < 2 {
        return Ok(None);
    }
    cuts.push(last + 1);
    let rows: Vec<usize> = cuts.windows(2).map(|w| (w[1] - w[0]) * sv).collect();
    let jobs: Vec<_> = split_rows(planes, scan, &rows)
        .into_iter()
        .enumerate()
        .map(|(i, pix)| {
            let end = if i + 1 < nb { cuts[i + 1] - 1 } else { b.last };
            (cuts[i], end, pix)
        })
        .collect();
    let done = par::run(pool, cx.gauge, k, jobs, |(r0, r1, pix)| {
        let mut r = BitReader::new(src, f.scan_offset);
        let mut views = Views {
            pix,
            blocks: [&mut [], &mut [], &mut [], &mut []],
        };
        scan.run(cx, src, &mut views, &mut r, true, r0, r1, Some(b), None)
            .map(|stopped| (r, stopped))
    });
    let (mut ok, mut last_band) = (true, None);
    for d in done {
        let (r, stopped) = d?;
        ok = ok && !r.lost && !r.insufficient && !r.eof;
        last_band = Some((r, stopped));
    }
    Ok(if ok { last_band } else { None })
}

/// Whether `restart[k]` is a state `skip_to` can land on without decoding: RST(k-1) sits just
/// before the offset.
fn landing_ok(src: &[u8], restart: &[usize], k: usize) -> bool {
    restart.get(k).is_some_and(|&off| {
        (2..=src.len()).contains(&off)
            && src[off - 2] == 0xff
            && src[off - 1] == 0xd0u8.wrapping_add((k.wrapping_sub(1) & 7) as u8)
    })
}

/// Moves `r` from interval `cur` to the start of the latest interval `k` in `(cur, t]` whose
/// decoder state is known without decoding: its RSTn was consumed, so no marker is pending and
/// `process_restart` cleared `insufficient_data`. A pending marker keeps the previous interval's
/// flag, which only decoding reveals, so such an interval is not a landing point. A `restart`
/// offset counts when RST(k-1) sits just before it; otherwise turbo's own restart automaton is
/// replayed on the raw bytes from `r`, finding the same markers the full decode would, so any
/// resync it meets is one turbo warns about and marks `r` lost.
fn skip_to<'a>(
    src: &'a [u8],
    r: &mut BitReader<'a>,
    restart: &[usize],
    cur: usize,
    t: usize,
) -> Option<usize> {
    let mut k = t.min(restart.len().wrapping_sub(1));
    while !restart.is_empty() && k > cur {
        if landing_ok(src, restart, k) {
            let off = restart[k];
            let mut n = BitReader::new(src, off);
            n.decoded = r.decoded;
            n.scanned = r.scanned;
            n.lost = r.lost;
            *r = n;
            return Some(k);
        }
        k -= 1;
    }
    let mut s = *r;
    let (mut best, mut pos) = (0usize, 0usize);
    let mut expected = cur & 7;
    let mut resynced = false;
    for k in cur + 1..=t {
        if read_restart_marker(&mut s, &mut expected) {
            resynced = true;
        }
        if !s.marker {
            best = k;
            pos = s.pos;
        }
    }
    r.scanned = s.scanned;
    if best == 0 {
        return None;
    }
    let lost = r.lost || resynced;
    let mut n = BitReader::new(src, pos);
    n.decoded = r.decoded;
    n.scanned = r.scanned;
    n.lost = lost;
    *r = n;
    Some(best)
}

/// The index of the first 0xFF in `b`, eight bytes at a time (Go's `bytes.IndexByte`).
fn find_ff(b: &[u8]) -> Option<usize> {
    let (words, rest) = b.as_chunks::<8>();
    for (i, w) in words.iter().enumerate() {
        let y = !u64::from_le_bytes(*w);
        // A zero byte in y (an FF in the data); the test can flag bytes above the first, so the
        // exact index is looked up in the word.
        if y.wrapping_sub(0x0101010101010101) & !y & 0x8080808080808080 != 0 {
            return w.iter().position(|&x| x == 0xff).map(|j| 8 * i + j);
        }
    }
    rest.iter()
        .position(|&x| x == 0xff)
        .map(|j| 8 * words.len() + j)
}

/// The first FF at or after `pos` not followed by 00, where the fast path would bail, and
/// whether it starts a marker inside `src`: only then do both paths stop there with `eof` unset,
/// whatever their buffering.
fn next_bail(src: &[u8], mut pos: usize) -> (usize, bool) {
    while pos < src.len() {
        let Some(i) = find_ff(&src[pos..]) else {
            break;
        };
        pos += i + 1;
        if pos < src.len() && src[pos] == 0 {
            pos += 1;
            continue;
        }
        let mut j = pos;
        while j < src.len() && src[j] == 0xff {
            j += 1;
        }
        return (pos - 1, j < src.len() && src[j] != 0);
    }
    (src.len(), false)
}

fn shl(x: u64, n: i32) -> u64 {
    if (0..64).contains(&n) { x << n } else { 0 }
}

fn shr(x: u64, n: i32) -> u64 {
    if (0..64).contains(&n) { x >> n } else { 0 }
}

/// turbo's `FILL_BIT_BUFFER_FAST`: six bytes with no 0xFF among them, inline into the MCU loop;
/// anything else is `refill_slow`.
#[inline(always)]
fn refill_fast(
    data: &[u8],
    pos: usize,
    lb: u64,
    bits: i32,
    marker: bool,
) -> (u64, usize, i32, bool) {
    if pos + 8 <= data.len() {
        let d = &data[pos..pos + 8];
        let v = u64::from_be_bytes([d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]]);
        let x = !v | 0xffff; // SWAR 0xff test; bytes 6-7 unused
        if x.wrapping_sub(0x0101010101010101) & !x & 0x8080808080808080 == 0 {
            let sh = ((16 - bits) & 63) as u32;
            return (lb | (v >> 16) << sh, pos + 6, bits + 48, marker);
        }
    }
    refill_slow(data, pos, lb, bits, marker)
}

/// `refill_fast` byte by byte: a 0xFF in the window (a stuffed zero, or a marker), or the end of
/// the data.
#[cold]
#[inline(never)]
fn refill_slow(
    data: &[u8],
    pos: usize,
    lb: u64,
    bits: i32,
    mut marker: bool,
) -> (u64, usize, i32, bool) {
    let data_len = data.len();
    let sh = ((16 - bits) & 63) as u32;
    let mut acc = 0u64;
    let mut pos = pos;
    for _ in 0..6 {
        acc <<= 8;
        if pos + 1 < data_len {
            let (c0, c1) = (data[pos], data[pos + 1]);
            if c0 != 0xff {
                acc |= c0 as u64;
                pos += 1;
            } else if c1 == 0 {
                acc |= 0xff;
                pos += 2;
            } else {
                marker = true;
            }
        } else {
            marker = true;
        }
    }
    (lb | acc << sh, pos, bits + 48, marker)
}

/// The bit-reader state of the fast path, left-aligned in `lb`.
struct Fast {
    pos: usize,
    lb: u64,
    bits: i32,
    marker: bool,
    insufficient: bool,
    lost: bool,
    eof: bool,
}

impl Fast {
    #[inline(always)]
    fn refill(&mut self, data: &[u8]) {
        let (lb, pos, bits, marker) = refill_fast(data, self.pos, self.lb, self.bits, self.marker);
        self.lb = lb;
        self.pos = pos;
        self.bits = bits;
        self.marker = marker;
    }

    /// `slowHuff` on the left-aligned buffer, as Go's `SLOWHUFF` (huff_amd64.s): the caller
    /// left more than 16 bits buffered, so the code (up to the 17 bits of the `maxcode[17]`
    /// sentinel) is all in `lb` and nothing is filled.
    #[inline(always)]
    fn slow_huff(&mut self, d: &DerivedTable) -> u8 {
        let mut l = 9usize;
        let mut code = (self.lb >> 55) as i32;
        while code > d.maxcode[l] {
            l += 1;
            code = (self.lb >> (64 - l)) as i32;
        }
        self.lb <<= l;
        self.bits -= l as i32;
        if l > 16 {
            self.lost = true; // JWRN_HUFF_BAD_CODE
            return 0;
        }
        d.values[(code + d.valoffset[l]) as usize & 255]
    }

    /// One Huffman symbol through the 8-bit lookahead table.
    #[inline(always)]
    fn symbol(&mut self, d: &DerivedTable) -> u8 {
        let entry = d.lookup[(self.lb >> 56) as usize];
        let nb = (entry >> 8) as i32;
        if (1..=8).contains(&nb) {
            self.lb <<= nb & 63;
            self.bits -= nb;
            (entry & 0xff) as u8
        } else {
            self.slow_huff(d)
        }
    }

    /// The next `size` (1..=15) bits, extended.
    #[inline(always)]
    fn value(&mut self, data: &[u8], size: i32) -> i32 {
        if self.bits <= 16 {
            self.refill(data);
        }
        let v = (self.lb >> ((64 - size) & 63)) as i32;
        self.lb <<= size & 63;
        self.bits -= size;
        huff_extend(v, size)
    }
}

/// turbo's `decode_mcu_fast`; failed writes stay. `has_ac[i]` reports block i's first AC symbol
/// was not EOB. With `blks` `None` it walks the same bits but stores nothing (an MCU outside a
/// band).
#[inline]
fn decode_mcu_fast(
    r: &mut BitReader,
    dc: &[DerivedTable; 4],
    ac: &[DerivedTable; 4],
    last_dc: &mut [i32; 4],
    blks: Option<&mut [Block]>,
    ks: &[usize],
    has_ac: Option<&mut [bool]>,
) -> bool {
    match blks {
        Some(b) => simd::decode_mcu_fast::<true>(r, dc, ac, last_dc, b, ks, has_ac),
        None => simd::decode_mcu_fast::<false>(r, dc, ac, last_dc, &mut [], ks, has_ac),
    }
}

/// The body of [`decode_mcu_fast`]; `STORE` false stores nothing and `blks` is empty. Inlined into
/// the SIMD tier's build with BMI.
#[inline(always)]
pub(crate) fn decode_mcu_fast_impl<const STORE: bool>(
    r: &mut BitReader,
    dc: &[DerivedTable; 4],
    ac: &[DerivedTable; 4],
    last_dc: &mut [i32; 4],
    blks: &mut [Block],
    ks: &[usize],
    mut has_ac: Option<&mut [bool]>,
) -> bool {
    let data = r.data;
    let mut s = Fast {
        pos: r.pos,
        lb: shl(r.buf, 64 - r.bits_left),
        bits: r.bits_left,
        marker: r.marker,
        insufficient: r.insufficient,
        lost: r.lost,
        eof: r.eof,
    };
    let mut dcs = *last_dc;

    for (i, &k0) in ks.iter().enumerate() {
        let k = k0 & 3; // scan_n <= 4
        let (d, a) = (&dc[k], &ac[k]);

        if s.bits <= 16 {
            s.refill(data);
        }
        let ce = d.combined[(s.lb >> (64 - COMB_BITS)) as usize];
        let mut sym = if ce != 0 {
            // A code and its value in one lookup; turbo refills before the value (none to read
            // for a flagged entry).
            let fill = ce & 64 == 0 && s.bits - (ce >> 8 & 15) as i32 <= 16;
            s.lb <<= ce & 63;
            s.bits -= (ce & 63) as i32;
            if fill {
                s.refill(data);
            }
            (ce as i32) >> 16
        } else {
            let mut sym = s.symbol(d) as i32;
            if sym != 0 {
                sym = s.value(data, sym);
            }
            sym
        };
        sym = sym.wrapping_add(dcs[k]);
        if STORE {
            blks[i][0] = sym as i16;
        }
        dcs[k] = sym;

        let mut kk = 1usize;
        while kk < 64 {
            if s.bits <= 16 {
                s.refill(data);
            }
            let ce = a.combined[(s.lb >> (64 - COMB_BITS)) as usize];
            if ce != 0 {
                let fill = ce & 64 == 0 && s.bits - (ce >> 8 & 15) as i32 <= 16;
                s.lb <<= ce & 63;
                s.bits -= (ce & 63) as i32;
                if fill {
                    s.refill(data);
                }
                if ce & 64 != 0 {
                    // No value: EOB ends the block, ZRL skips 16.
                    if ce >> 12 & 15 != 15 {
                        break;
                    }
                    kk += 16;
                    continue;
                }
                kk += (ce >> 12 & 15) as usize;
                if STORE {
                    blks[i][NATURAL_ORDER_FAST[kk & 127] as usize & 63] =
                        ((ce as i32) >> 16) as i16;
                }
                kk += 1;
                continue;
            }
            let rs = s.symbol(a);
            let (run, size) = ((rs >> 4) as usize, (rs & 15) as i32);
            if size == 0 {
                if run != 15 {
                    break;
                }
                kk += 16;
                continue;
            }
            kk += run;
            let v = s.value(data, size);
            if STORE {
                blks[i][NATURAL_ORDER_FAST[kk & 127] as usize & 63] = v as i16;
            }
            kk += 1;
        }
        if let Some(h) = has_ac.as_deref_mut() {
            h[i] = kk != 1;
        }
    }

    if s.marker {
        return false;
    }
    r.decoded += s.pos - r.pos;
    r.pos = s.pos;
    r.buf = shr(s.lb, 64 - s.bits);
    r.bits_left = s.bits;
    r.marker = s.marker;
    r.insufficient = s.insufficient;
    r.lost = s.lost;
    r.eof = s.eof;
    *last_dc = dcs;
    true
}

/// turbo's `decode_mcu_slow` (`jdhuff.c:560-650`).
fn decode_mcu_slow(
    r: &mut BitReader,
    dc: &[DerivedTable; 4],
    ac: &[DerivedTable; 4],
    last_dc: &mut [i32; 4],
    mut blks: Option<&mut [Block]>,
    ks: &[usize],
    mut has_ac: Option<&mut [bool]>,
) {
    let mut buf = r.buf;
    let mut bits = r.bits_left;
    for (i, &k0) in ks.iter().enumerate() {
        let c = k0 & 3; // scan_n <= 4
        let mut t = &dc[c];
        let mut kk = 0usize;
        while kk < 64 {
            // HUFF_DECODE fills below the lookahead.
            if bits < 8 {
                r.buf = buf;
                r.bits_left = bits;
                r.fill(0);
                buf = r.buf;
                bits = r.bits_left;
            }
            let sym: i32;
            if bits < 8 {
                r.buf = buf;
                r.bits_left = bits;
                sym = r.slow_huff(t, 1) as i32;
                buf = r.buf;
                bits = r.bits_left;
            } else {
                let e = t.lookup[(shr(buf, bits - 8) & 255) as usize];
                if e >> 8 != 0 {
                    bits -= (e >> 8) as i32;
                    sym = (e & 0xff) as i32;
                } else {
                    r.buf = buf;
                    r.bits_left = bits;
                    sym = r.slow_huff(t, 9) as i32;
                    buf = r.buf;
                    bits = r.bits_left;
                }
            }
            let mut size = sym;
            if kk > 0 {
                size = sym & 15;
                if size == 0 {
                    if sym >> 4 != 15 {
                        break;
                    }
                    kk += 16;
                    continue;
                }
                kk += (sym >> 4) as usize;
            }
            let mut v = 0i32;
            if size != 0 {
                if bits < size {
                    // CHECK_BIT_BUFFER
                    r.buf = buf;
                    r.bits_left = bits;
                    r.fill(size);
                    buf = r.buf;
                    bits = r.bits_left;
                }
                bits -= size;
                v = huff_extend((buf >> (bits & 63)) as i32 & ((1 << (size & 31)) - 1), size);
            }
            if kk == 0 {
                v = v.wrapping_add(last_dc[c]);
                last_dc[c] = v;
                if let Some(b) = blks.as_deref_mut() {
                    b[i][0] = v as i16;
                }
                t = &ac[c];
            } else if let Some(b) = blks.as_deref_mut() {
                b[i][NATURAL_ORDER_FAST[kk & 127] as usize & 63] = v as i16;
            }
            kk += 1;
        }
        if let Some(h) = has_ac.as_deref_mut() {
            h[i] = kk != 1;
        }
    }
    r.buf = buf;
    r.bits_left = bits;
}

/// Advances past the expected RSTn marker and resets per-restart state (turbo `process_restart`,
/// `jdhuff.c:512-541`).
pub(crate) fn process_restart(r: &mut BitReader, last_dc: &mut [i32], expected: &mut usize) {
    r.bits_left = 0; // discard leftover bits, turbo jdhuff.c:515-517
    if read_restart_marker(r, expected) {
        r.lost = true; // turbo always warns on resync
    }
    last_dc.fill(0);
    if !r.marker {
        r.insufficient = false;
    }
}

/// Consumes the next RSTn marker, resyncing on mismatch (turbo `read_restart_marker`). Reports
/// whether the marker didn't match `expected` and had to be resynced.
fn read_restart_marker(r: &mut BitReader, expected: &mut usize) -> bool {
    if !r.marker {
        let (c, ok) = next_marker(r);
        r.marker = true;
        r.marker_byte = if ok { c } else { MARKER_EOI };
    }
    let mut resynced = false;
    if r.marker_byte == 0xd0u8.wrapping_add(*expected as u8) {
        r.marker = false;
    } else {
        resync_to_restart(r, *expected);
        resynced = true;
    }
    *expected = (*expected + 1) & 7;
    resynced
}

/// Scans for the next marker code, skipping stuffed zero bytes and padding FFs (turbo
/// `next_marker`).
pub(crate) fn next_marker(r: &mut BitReader) -> (u8, bool) {
    let p = r.pos;
    let res = scan_marker(r);
    r.scanned += r.pos - p;
    res
}

fn scan_marker(r: &mut BitReader) -> (u8, bool) {
    loop {
        if r.pos >= r.data.len() {
            r.eof = true;
            return (0, false);
        }
        let mut c = r.data[r.pos];
        r.pos += 1;
        if c != 0xff {
            continue;
        }
        while r.pos < r.data.len() && r.data[r.pos] == 0xff {
            r.pos += 1;
        }
        if r.pos >= r.data.len() {
            r.eof = true;
            return (0, false);
        }
        c = r.data[r.pos];
        r.pos += 1;
        if c != 0 {
            return (c, true);
        }
    }
}

/// turbo's default `jpeg_resync_to_restart`: tolerates a missing or off-by-one-or-two restart
/// marker.
fn resync_to_restart(r: &mut BitReader, desired: usize) {
    loop {
        let m = r.marker_byte as usize;
        let action = if m < 0xc0 {
            2 // invalid marker
        } else if !(0xd0..=0xd7).contains(&m)
            || m == 0xd0 + ((desired + 1) & 7)
            || m == 0xd0 + ((desired + 2) & 7)
        {
            3 // valid non-restart marker, or one of the next two expected restarts
        } else if m == 0xd0 + (desired.wrapping_sub(1) & 7)
            || m == 0xd0 + (desired.wrapping_sub(2) & 7)
        {
            2 // a prior restart, so advance
        } else {
            1 // desired restart or too far away
        };
        match action {
            1 => {
                r.marker = false;
                return;
            }
            2 => {
                let (c, ok) = next_marker(r);
                if !ok {
                    r.marker_byte = MARKER_EOI;
                    return;
                }
                r.marker_byte = c;
            }
            _ => return,
        }
    }
}
