//! Upsampling and colour conversion, ported from jpeg-go's `internal/codec/sample.go` (turbo
//! `jdsample.c`, `jdcolor.c`, `jdcolext.c`).

use crate::Error;
use crate::limits::{Limits, check, filled};
use crate::par;
use crate::scan::Ctx;
use crate::simd::{self, Tier};
use std::sync::atomic::AtomicBool;

/// Whole-image row loops poll the cancel flag when `y & POLL_ROWS == 0`: every 16 rows.
const POLL_ROWS: usize = 15;

const FIX_YCC_1_40200: i64 = 91881;
const FIX_YCC_1_77200: i64 = 116130;
const FIX_YCC_0_71414: i64 = 46802;
const FIX_YCC_0_34414: i64 = 22554;
const YCC_ONE_HALF: i64 = 1 << 15;

struct YccTables {
    cr_r: [i32; 256],
    cb_b: [i32; 256],
    cb_g: [i32; 256],
    cr_g: [i32; 256],
}

/// `build_ycc_rgb_table`, indexed directly by the sample byte (`x = i - 128` folded in).
const fn build_ycc_tables() -> YccTables {
    let mut t = YccTables {
        cr_r: [0; 256],
        cb_b: [0; 256],
        cb_g: [0; 256],
        cr_g: [0; 256],
    };
    let mut i = 0;
    while i < 256 {
        let x = i as i64 - 128;
        t.cr_r[i] = ((FIX_YCC_1_40200 * x + YCC_ONE_HALF) >> 16) as i32;
        t.cb_b[i] = ((FIX_YCC_1_77200 * x + YCC_ONE_HALF) >> 16) as i32;
        t.cr_g[i] = (-FIX_YCC_0_71414 * x) as i32;
        t.cb_g[i] = (-FIX_YCC_0_34414 * x + YCC_ONE_HALF) as i32;
        i += 1;
    }
    t
}

static YCC: YccTables = build_ycc_tables();

/// turbo's `sample_range_limit` as a table: a plain clamp. YCbCr to RGB sums stay within
/// [-384, 639], so the +384 bias and `& 1023` mask never wrap.
const fn build_range_limit_table() -> [u8; 1024] {
    let mut t = [0u8; 1024];
    let mut i = 0;
    while i < 1024 {
        let v = i as i32 - 384;
        t[i] = if v < 0 {
            0
        } else if v > 255 {
            255
        } else {
            v as u8
        };
        i += 1;
    }
    t
}

static RANGE_LIMIT_TABLE: [u8; 1024] = build_range_limit_table();

fn clamp_byte(v: i32) -> u8 {
    RANGE_LIMIT_TABLE[(v.wrapping_add(384) as u32 & 1023) as usize]
}

/// `ycc_rgb_convert` over one row of `y.len()` pixels.
pub(crate) fn color_row(dst: &mut [u8], y: &[u8], cb: &[u8], cr: &[u8]) {
    color_row_with(simd::tier(), dst, y, cb, cr);
}

/// [`color_row`] in tier `t`: the SIMD kernel does the pixels of its blocks and, on a row of 16 or
/// more, the last 16 again; the scalar loop does the rest.
pub(crate) fn color_row_with(t: Tier, dst: &mut [u8], y: &[u8], cb: &[u8], cr: &[u8]) {
    let mut done = simd::color_row(t, dst, y, cb, cr);
    if done > 0 && done < y.len() && y.len() >= 16 {
        // Redo the last block in SIMD, ending at the row's end: the overlap rewrites equal bytes.
        let c = y.len() - 16;
        simd::color_row(t, &mut dst[4 * c..], &y[c..], &cb[c..], &cr[c..]);
        done = y.len();
    }
    color_row_scalar(&mut dst[4 * done..], &y[done..], &cb[done..], &cr[done..]);
}

/// The scalar reference of [`color_row`].
fn color_row_scalar(dst: &mut [u8], y: &[u8], cb: &[u8], cr: &[u8]) {
    for (x, &yv) in y.iter().enumerate() {
        let yy = yv as i32;
        let (cbv, crv) = (cb[x] as usize, cr[x] as usize);
        let r = clamp_byte(yy + YCC.cr_r[crv]);
        let g = clamp_byte(yy + ((YCC.cb_g[cbv] + YCC.cr_g[crv]) >> 16));
        let b = clamp_byte(yy + YCC.cb_b[cbv]);
        dst[x * 4..x * 4 + 4].copy_from_slice(&[r, g, b, 255]);
    }
}

/// `h2v1_fancy_upsample` for a whole row: `dst` has room for `w` bytes, `w` being `2 * dsw` or
/// `2 * dsw - 1`; `dsw` is at least 2 (the decoder's fancy path gates on `dsw > 2`).
pub(crate) fn h2v1_row(dst: &mut [u8], row: &[u8], dsw: usize, w: usize) {
    h2v1_row_with(simd::tier(), dst, row, dsw, w);
}

/// [`h2v1_row`] in tier `t`.
pub(crate) fn h2v1_row_with(t: Tier, dst: &mut [u8], row: &[u8], dsw: usize, w: usize) {
    if dsw < 2 {
        box_row(dst, row, dsw, w);
        return;
    }
    let v = row[0] as i32;
    dst[0] = v as u8;
    dst[1] = ((v * 3 + row[1] as i32 + 2) >> 2) as u8;
    let c0 = simd::h2v1_mid(t, dst, row, 1, dsw - 1);
    let mut o = 2 * c0;
    for c in c0..dsw - 1 {
        let v = row[c] as i32 * 3;
        dst[o] = ((v + row[c - 1] as i32 + 1) >> 2) as u8;
        dst[o + 1] = ((v + row[c + 1] as i32 + 2) >> 2) as u8;
        o += 2;
    }
    let v = row[dsw - 1] as i32;
    dst[o] = ((v * 3 + row[dsw - 2] as i32 + 1) >> 2) as u8;
    if o + 1 < w {
        dst[o + 1] = v as u8;
    }
}

/// `h2v2_fancy_upsample` for a whole row: one sliding column sum per input column.
pub(crate) fn h2v2_row(dst: &mut [u8], row_cur: &[u8], row_other: &[u8], dsw: usize, w: usize) {
    h2v2_row_with(simd::tier(), dst, row_cur, row_other, dsw, w);
}

/// [`h2v2_row`] in tier `t`.
pub(crate) fn h2v2_row_with(
    t: Tier,
    dst: &mut [u8],
    row_cur: &[u8],
    row_other: &[u8],
    dsw: usize,
    w: usize,
) {
    if dsw < 2 {
        box_row(dst, row_cur, dsw, w);
        return;
    }
    let colsum = |c: usize| row_cur[c] as i32 * 3 + row_other[c] as i32;
    let first = colsum(0);
    let second = colsum(1);
    dst[0] = ((first * 4 + 8) >> 4) as u8;
    dst[1] = ((first * 3 + second + 7) >> 4) as u8;
    let c0 = simd::h2v2_mid(t, dst, row_cur, row_other, 1, dsw - 1);
    let mut lastcolsum = colsum(c0 - 1);
    let mut thiscolsum = colsum(c0);
    let mut o = 2 * c0;
    for c in c0..dsw - 1 {
        let nextcolsum = colsum(c + 1);
        dst[o] = ((thiscolsum * 3 + lastcolsum + 8) >> 4) as u8;
        dst[o + 1] = ((thiscolsum * 3 + nextcolsum + 7) >> 4) as u8;
        lastcolsum = thiscolsum;
        thiscolsum = nextcolsum;
        o += 2;
    }
    dst[o] = ((thiscolsum * 3 + lastcolsum + 8) >> 4) as u8;
    if o + 1 < w {
        dst[o + 1] = ((thiscolsum * 4 + 7) >> 4) as u8;
    }
}

/// `h2v1/h2v2_upsample`'s row body: plain duplication of each downsampled column into two output
/// columns.
fn box_row(dst: &mut [u8], row: &[u8], dsw: usize, w: usize) {
    let mut o = 0;
    let mut c = 0;
    while c < dsw && o < w {
        dst[o] = row[c];
        o += 1;
        if o < w {
            dst[o] = row[c];
            o += 1;
        }
        c += 1;
    }
}

fn upsample_row_h(dst: &mut [u8], row: &[u8], dsw: usize, w: usize, fancy: bool) {
    if fancy {
        h2v1_row(dst, row, dsw, w);
    } else {
        box_row(dst, row, dsw, w);
    }
}

/// `row_other` is the box filter's don't-care when not `fancy`.
fn upsample_row_v(dst: &mut [u8], cur: &[u8], other: &[u8], dsw: usize, w: usize, fancy: bool) {
    if fancy {
        h2v2_row(dst, cur, other, dsw, w);
    } else {
        box_row(dst, cur, dsw, w);
    }
}

/// `h1v2_fancy_upsample` for one output row: 3/4 of the nearest chroma row plus 1/4 of the one
/// beside it. `bias` is 1 for the even output row (neighbour above) and 2 for the odd (below).
fn h1v2_fancy_row(dst: &mut [u8], cur: &[u8], other: &[u8], bias: i32) {
    for x in 0..cur.len() {
        dst[x] = ((cur[x] as i32 * 3 + other[x] as i32 + bias) >> 2) as u8;
    }
}

/// `int_upsample`'s row body: each downsampled column repeated `h_exp` times, the 4:1:1 and
/// 4:1:0 case.
fn int_row(dst: &mut [u8], row: &[u8], dsw: usize, w: usize, h_exp: usize) {
    let mut o = 0;
    for c in 0..dsw {
        for _ in 0..h_exp {
            if o >= w {
                break;
            }
            dst[o] = row[c];
            o += 1;
        }
    }
}

/// The chroma row indices h2v2 needs for output row `y`: the nearest row and the row above (`y`
/// even) or below (`y` odd) it, clamped to the plane's real height.
fn v2_rows(y: usize, dsh: usize) -> (usize, usize) {
    let p = (y / 2) as i64;
    let other = if y % 2 == 1 { p + 1 } else { p - 1 };
    (clamp_idx(p, dsh), clamp_idx(other, dsh))
}

/// Duplicates row 0 or n-1 for an out-of-range index.
fn clamp_idx(i: i64, n: usize) -> usize {
    if i < 0 {
        0
    } else if i as usize >= n {
        n.saturating_sub(1)
    } else {
        i as usize
    }
}

/// The planes an upsample and colour conversion reads.
pub(crate) struct YccPlanes<'a> {
    pub y: &'a [u8],
    pub y_stride: usize,
    pub cb: &'a [u8],
    pub cr: &'a [u8],
    pub c_stride: usize,
}

/// Shape of one upsample and colour conversion: `w` x `h` output, chroma `dsw` x `dsh`, turbo's
/// `h_expand`/`v_expand`, and `allow_fancy` = `do_fancy_upsampling && min_DCT_scaled_size > 1`.
pub(crate) struct YccGeom {
    pub w: usize,
    pub h: usize,
    pub dsw: usize,
    pub dsh: usize,
    pub h_exp: usize,
    pub v_exp: usize,
    pub allow_fancy: bool,
}

/// Upsamples chroma and converts to RGBA, row by row. `cb_row` and `cr_row` are scratch rows of
/// at least `w` bytes.
pub(crate) fn ycbcr_upsample_convert(
    dst: &mut [u8],
    dst_stride: usize,
    p: &YccPlanes,
    g: &YccGeom,
    cb_row: &mut [u8],
    cr_row: &mut [u8],
    cancel: &AtomicBool,
) -> Result<(), Error> {
    ycbcr_rows(dst, dst_stride, p, g, 0, g.h, cb_row, cr_row, cancel)
}

/// [`ycbcr_upsample_convert`] on up to `cx.threads` workers of the call's pool, each converting
/// a stripe of rows with its own scratch rows (Go's `YCbCrUpsampleConvertN`). An output row
/// reads only the finished planes and its own absolute row, so every stripe cut gives the same
/// bytes. Goes serial, with the same result, when the frame has too few pixels for a second
/// worker, there are under 16 rows a stripe, or the extra rows do not fit the ledger.
pub(crate) fn ycbcr_upsample_convert_n(
    cx: &Ctx,
    dst: &mut [u8],
    dst_stride: usize,
    p: &YccPlanes,
    g: &YccGeom,
    cb_row: &mut [u8],
    cr_row: &mut [u8],
) -> Result<(), Error> {
    let cap = if cx.uncapped {
        usize::MAX
    } else {
        par::stripe_threads(g.w * g.h)
    };
    let serial = |dst: &mut [u8], cb: &mut [u8], cr: &mut [u8]| {
        ycbcr_upsample_convert(dst, dst_stride, p, g, cb, cr, cx.cancel)
    };
    if cx.threads.min(g.h / 16).min(cap) < 2 {
        return serial(dst, cb_row, cr_row);
    }
    // The pool starts here, once the frame is known to be worth threading.
    let Some((pool, nw)) = par::stage(cx) else {
        return serial(dst, cb_row, cr_row);
    };
    let k = nw.min(g.h / 16).min(cap);
    if k < 2 {
        return serial(dst, cb_row, cr_row);
    }
    let extra = k - 1;
    let Some(_held) = cx.ledger.hold(extra as u64 * 2 * g.w as u64) else {
        return serial(dst, cb_row, cr_row);
    };
    let (mut cbs, mut crs) = (Vec::new(), Vec::new());
    for _ in 0..extra {
        match (filled(g.w, 0u8), filled(g.w, 0u8)) {
            (Ok(cb), Ok(cr)) => {
                cbs.push(cb);
                crs.push(cr);
            }
            _ => return serial(dst, cb_row, cr_row),
        }
    }
    let mut rows: Vec<(&mut [u8], &mut [u8])> = vec![(cb_row, cr_row)];
    rows.extend(
        cbs.iter_mut()
            .zip(crs.iter_mut())
            .map(|(cb, cr)| (cb.as_mut_slice(), cr.as_mut_slice())),
    );
    let mut jobs = Vec::with_capacity(k);
    let mut rest: &mut [u8] = dst;
    for (i, (cb, cr)) in rows.into_iter().enumerate() {
        let (y0, y1) = (g.h * i / k, g.h * (i + 1) / k);
        let cur = std::mem::take(&mut rest);
        let head = if i + 1 < k {
            let (head, tail) = cur.split_at_mut((y1 - y0) * dst_stride);
            rest = tail;
            head
        } else {
            cur
        };
        jobs.push((y0, y1, head, cb, cr));
    }
    let done = par::run(pool, cx.gauge, k, jobs, |(y0, y1, head, cb, cr)| {
        ycbcr_rows(head, dst_stride, p, g, y0, y1, cb, cr, cx.cancel)
    });
    done.into_iter().collect()
}

/// Rows `y0..y1` of [`ycbcr_upsample_convert`], written to `dst` from its first row. Everything
/// that depends on the row (the chroma neighbours, the h1v2 bias, the cancel poll) uses the
/// absolute `y`.
fn ycbcr_rows(
    dst: &mut [u8],
    dst_stride: usize,
    p: &YccPlanes,
    g: &YccGeom,
    y0: usize,
    y1: usize,
    cb_row: &mut [u8],
    cr_row: &mut [u8],
    cancel: &AtomicBool,
) -> Result<(), Error> {
    let (w, dsw, dsh) = (g.w, g.dsw, g.dsh);
    let fancy = g.allow_fancy && dsw > 2; // h2 only; h1v2 fancy has no width test
    let cs = p.c_stride;
    for y in y0..y1 {
        if y & POLL_ROWS == 0 {
            check(cancel)?;
        }
        let y_row = &p.y[y * p.y_stride..y * p.y_stride + w];
        let o = (y - y0) * dst_stride;
        let out_row = &mut dst[o..o + w * 4];
        match (g.h_exp, g.v_exp) {
            (1, 1) => {
                color_row(
                    out_row,
                    y_row,
                    &p.cb[y * cs..y * cs + w],
                    &p.cr[y * cs..y * cs + w],
                );
            }
            (2, 1) => {
                upsample_row_h(cb_row, &p.cb[y * cs..y * cs + dsw], dsw, w, fancy);
                upsample_row_h(cr_row, &p.cr[y * cs..y * cs + dsw], dsw, w, fancy);
                color_row(out_row, y_row, cb_row, cr_row);
            }
            (2, 2) => {
                let (cur, other) = v2_rows(y, dsh);
                upsample_row_v(
                    cb_row,
                    &p.cb[cur * cs..cur * cs + dsw],
                    &p.cb[other * cs..other * cs + dsw],
                    dsw,
                    w,
                    fancy,
                );
                upsample_row_v(
                    cr_row,
                    &p.cr[cur * cs..cur * cs + dsw],
                    &p.cr[other * cs..other * cs + dsw],
                    dsw,
                    w,
                    fancy,
                );
                color_row(out_row, y_row, cb_row, cr_row);
            }
            (1, 2) => {
                let (cur, other) = v2_rows(y, dsh);
                let cb_cur = &p.cb[cur * cs..cur * cs + w];
                let cr_cur = &p.cr[cur * cs..cur * cs + w];
                if g.allow_fancy {
                    let bias = 1 + (y & 1) as i32;
                    h1v2_fancy_row(cb_row, cb_cur, &p.cb[other * cs..other * cs + w], bias);
                    h1v2_fancy_row(cr_row, cr_cur, &p.cr[other * cs..other * cs + w], bias);
                    color_row(out_row, y_row, &cb_row[..w], &cr_row[..w]);
                } else {
                    color_row(out_row, y_row, cb_cur, cr_cur);
                }
            }
            _ => {
                // int_upsample: 4:1:1, 4:1:0 and their scaled forms
                let r = y / g.v_exp;
                int_row(cb_row, &p.cb[r * cs..r * cs + dsw], dsw, w, g.h_exp);
                int_row(cr_row, &p.cr[r * cs..r * cs + dsw], dsw, w, g.h_exp);
                color_row(out_row, y_row, cb_row, cr_row);
            }
        }
    }
    Ok(())
}

/// `rgb_rgb_convert`: a null colour convert for Adobe/ids-RGB streams, alpha forced opaque.
pub(crate) fn rgb_to_rgba(
    dst: &mut [u8],
    dst_stride: usize,
    r: &[u8],
    g: &[u8],
    b: &[u8],
    stride: usize,
    w: usize,
    h: usize,
    cancel: &AtomicBool,
) -> Result<(), Error> {
    for y in 0..h {
        if y & POLL_ROWS == 0 {
            check(cancel)?;
        }
        let (rr, gr, br) = (&r[y * stride..], &g[y * stride..], &b[y * stride..]);
        let out = &mut dst[y * dst_stride..];
        for x in 0..w {
            out[x * 4..x * 4 + 4].copy_from_slice(&[rr[x], gr[x], br[x], 255]);
        }
    }
    Ok(())
}

/// `gray_rgb_convert`: replicate Y into R, G, B.
pub(crate) fn gray_to_rgba(
    dst: &mut [u8],
    dst_stride: usize,
    gray: &[u8],
    stride: usize,
    w: usize,
    h: usize,
    cancel: &AtomicBool,
) -> Result<(), Error> {
    for y in 0..h {
        if y & POLL_ROWS == 0 {
            check(cancel)?;
        }
        let gr = &gray[y * stride..];
        let out = &mut dst[y * dst_stride..];
        for x in 0..w {
            let v = gr[x];
            out[x * 4..x * 4 + 4].copy_from_slice(&[v, v, v, 255]);
        }
    }
    Ok(())
}

/// jdsample.c's `h2v1_fancy_upsample` for one chroma row of `n = row.len()` samples (`n >= 2`),
/// for callers that decode with `Format::YCbCr` and need 4:4:4. `dst` must be `2n` or `2n - 1`
/// bytes (a longer one is filled to `2n`). The output equals libjpeg-turbo's for `n > 2`;
/// at `n == 2` libjpeg-turbo and `decode` use box upsampling, while this still applies the fancy
/// formula. Go panics on a bad length; this returns [`Error::InvalidArgument`], or
/// [`Error::AllocLimit`] when `2 * n` is over `limits.max_alloc`.
///
/// # Errors
///
/// [`Error::InvalidArgument`] for `n < 2` or a `dst` shorter than `2n - 1`, and
/// [`Error::AllocLimit`] when `2 * n` is over `limits.max_alloc`. `max_pixels` does not apply (there
/// is no frame), nothing is allocated, and it spawns no thread.
pub fn h2v1_fancy_row(dst: &mut [u8], row: &[u8], limits: &Limits) -> Result<(), Error> {
    let n = row.len();
    limits.check_bytes(2 * n as u64)?;
    let w = dst.len().min(2 * n);
    if n < 2 || w + 1 < 2 * n {
        return Err(Error::InvalidArgument(
            "h2v1_fancy_row: row under 2 samples or dst too short",
        ));
    }
    h2v1_row(&mut dst[..w], row, n, w);
    Ok(())
}

/// The h2v2 fancy upsample formula (`h2v2_fancy_upsample`) for one output row, from its chroma
/// row `cur` and the nearer neighbour row `other`; `other.len() >= cur.len() = n`. The `n`,
/// `dst` length and `n == 2` rules of [`h2v1_fancy_row`] apply.
///
/// # Errors
///
/// As [`h2v1_fancy_row`]: [`Error::InvalidArgument`] (also for `other` shorter than `cur`) and
/// [`Error::AllocLimit`] against `limits.max_alloc`. It spawns no thread.
pub fn h2v2_fancy_row(
    dst: &mut [u8],
    cur: &[u8],
    other: &[u8],
    limits: &Limits,
) -> Result<(), Error> {
    let n = cur.len();
    limits.check_bytes(2 * n as u64)?;
    let w = dst.len().min(2 * n);
    if n < 2 || other.len() < n || w + 1 < 2 * n {
        return Err(Error::InvalidArgument(
            "h2v2_fancy_row: rows under 2 samples or dst too short",
        ));
    }
    h2v2_row(&mut dst[..w], cur, other, n, w);
    Ok(())
}

/// The pyramid mode's chroma blur, in place on the real `w` x `h` samples of a plane: `[1, m-2,
/// 1] / m` with `m = 1 << shift` on each axis asked for (rows first, then columns), rounded,
/// with the edge sample repeated, as fancy upsampling treats the edge. At scale 2 (`shift` 3) it
/// is the exact mean of the four fancy-upsampled samples around a chroma sample; at scale 4
/// (`shift` 4) it removes most of the blur left at the output's Nyquist. Both passes walk rows,
/// keeping one copy of the previous original row in `scratch` (at least `w` bytes; its contents
/// are overwritten), so it allocates nothing. Polls `cancel` every 16 rows.
pub(crate) fn chroma_prefilter(
    p: &mut [u8],
    stride: usize,
    (w, h): (usize, usize),
    (horiz, vert): (bool, bool),
    shift: u32,
    scratch: &mut [u8],
    cancel: &AtomicBool,
) -> Result<(), Error> {
    if w == 0 || h == 0 || stride < w || p.len() < (h - 1) * stride + w {
        return Ok(());
    }
    let (mid, half) = ((1u16 << shift) - 2, 1u16 << (shift - 1));
    let tap = |a: u8, c: u8, b: u8| ((a as u16 + mid * c as u16 + b as u16 + half) >> shift) as u8;
    let s = &mut scratch[..w];
    if horiz {
        for (y, row) in p.chunks_mut(stride).take(h).enumerate() {
            if y & POLL_ROWS == 0 {
                check(cancel)?;
            }
            let row = &mut row[..w];
            s.copy_from_slice(row);
            row[0] = tap(s[0], s[0], s[1.min(w - 1)]);
            row[w - 1] = tap(s[w.saturating_sub(2)], s[w - 1], s[w - 1]);
            if w > 2 {
                for (((o, &a), &c), &b) in row[1..w - 1]
                    .iter_mut()
                    .zip(&s[..])
                    .zip(&s[1..])
                    .zip(&s[2..])
                {
                    *o = tap(a, c, b);
                }
            }
        }
    }
    if vert {
        s.copy_from_slice(&p[..w]);
        for y in 0..h {
            if y & POLL_ROWS == 0 {
                check(cancel)?;
            }
            let (a, b) = p.split_at_mut((y + 1) * stride);
            let cur = &mut a[y * stride..y * stride + w];
            if y + 1 < h {
                for ((sv, c), &n) in s.iter_mut().zip(cur.iter_mut()).zip(&b[..w]) {
                    let old = *c;
                    *c = tap(*sv, old, n);
                    *sv = old;
                }
            } else {
                for (c, &sv) in cur.iter_mut().zip(s.iter()) {
                    *c = tap(sv, *c, *c);
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The per-sample algorithm the row-wise function replaced.
    fn reference(
        p: &mut [u8],
        stride: usize,
        (w, h): (usize, usize),
        (horiz, vert): (bool, bool),
        shift: u32,
    ) {
        let (mid, half) = ((1u32 << shift) - 2, 1u32 << (shift - 1));
        let tap =
            |a: u8, c: u8, b: u8| ((a as u32 + mid * c as u32 + b as u32 + half) >> shift) as u8;
        if horiz {
            for row in p.chunks_mut(stride).take(h) {
                let row = &mut row[..w];
                let mut prev = row[0];
                for x in 0..w {
                    let cur = row[x];
                    row[x] = tap(prev, cur, row[(x + 1).min(w - 1)]);
                    prev = cur;
                }
            }
        }
        if vert {
            for x in 0..w {
                let mut prev = p[x];
                for y in 0..h {
                    let cur = p[y * stride + x];
                    p[y * stride + x] = tap(prev, cur, p[(y + 1).min(h - 1) * stride + x]);
                    prev = cur;
                }
            }
        }
    }

    #[test]
    fn chroma_prefilter_matches_reference() {
        let cancel = AtomicBool::new(false);
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 32) as u8
        };
        for w in 1..=40 {
            for h in 1..=20 {
                let stride = w + 3;
                let plane: Vec<u8> = (0..stride * h).map(|_| next()).collect();
                for axes in [(false, false), (true, false), (false, true), (true, true)] {
                    for shift in [3, 4] {
                        let mut want = plane.clone();
                        reference(&mut want, stride, (w, h), axes, shift);
                        let mut got = plane.clone();
                        let mut scratch = vec![0u8; w];
                        chroma_prefilter(
                            &mut got,
                            stride,
                            (w, h),
                            axes,
                            shift,
                            &mut scratch,
                            &cancel,
                        )
                        .unwrap();
                        assert_eq!(got, want, "w {w} h {h} axes {axes:?} shift {shift}");
                    }
                }
            }
        }
    }
}
