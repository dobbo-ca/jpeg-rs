//! Progressive (SOF2) scans and block smoothing, ported from jpeg-go's
//! `internal/codec/progressive.go` (turbo `jdphuff.c`, `jdcoefct.c`).

use crate::Error;
use crate::dct::{Block, dequantize, idct_bs};
use crate::huff::{BitReader, DerivedTable, huff_extend};
use crate::limits::check;
use crate::marker::{Frame, MARKER_EOI};
use crate::scan::{
    Ctx, Idct, MAX_MCU_BLOCKS, NATURAL_ORDER_FAST, Plane, ScanResult, downsampled_dims,
    fill_padding, latch_quant, max_sampling, next_marker, process_restart,
};

/// turbo's cross-scan state.
#[derive(Clone, Copy)]
pub(crate) struct ProgState {
    pub coef_bits: [[i8; 64]; 4],
    pub prev_bits: [[i8; 10]; 4],
    /// `cinfo->input_scan_number`.
    pub scans: usize,
    /// `master->last_good_iMCU_row`.
    pub last_good: usize,
}

impl ProgState {
    pub(crate) fn new() -> ProgState {
        ProgState {
            coef_bits: [[-1; 64]; 4],
            prev_bits: [[0; 10]; 4],
            scans: 0,
            last_good: 0,
        }
    }
}

/// What every scan of a progressive frame leaves behind.
pub(crate) struct ProgScans {
    pub truncated: bool,
    pub decoded: usize,
    pub scanned_bytes: usize,
    pub ps: ProgState,
    pub scanned: [bool; 4],
    pub err: Option<Error>,
}

impl ProgScans {
    fn fail(truncated: bool, decoded: usize, scanned_bytes: usize, err: Error) -> ProgScans {
        ProgScans {
            truncated,
            decoded,
            scanned_bytes,
            ps: ProgState::new(),
            scanned: [false; 4],
            err: Some(err),
        }
    }
}

/// `decode_scan` for SOF2.
pub(crate) fn decode_progressive(
    cx: &Ctx,
    f: &mut Frame,
    src: &[u8],
    planes: &mut [Plane],
) -> ScanResult {
    let (_, vmax) = max_sampling(f);
    let pr = decode_prog_scans(cx, f, src, planes);
    if let Some(e) = pr.err {
        return ScanResult {
            truncated: pr.truncated,
            decoded: pr.decoded,
            scanned: pr.scanned_bytes,
            err: Some(e),
        };
    }
    if let Err(e) = idct_progressive(cx, f, planes, &pr.scanned, &pr.ps, vmax) {
        return ScanResult::fail(true, e);
    }
    for ci in 0..f.ncomp {
        if pr.scanned[ci] {
            continue;
        }
        let p = &mut planes[ci];
        if !p.skip && !p.coef {
            p.pix.fill(128);
        }
    }
    ScanResult {
        truncated: pr.truncated,
        decoded: pr.decoded,
        scanned: pr.scanned_bytes,
        err: None,
    }
}

/// Runs every scan of a SOF2 frame into `planes`' blocks, leaving the IDCT to the caller, with
/// the state smoothing needs.
pub(crate) fn decode_prog_scans(
    cx: &Ctx,
    f: &mut Frame,
    src: &[u8],
    planes: &mut [Plane],
) -> ProgScans {
    let (hmax, vmax) = max_sampling(f);
    let last = f.height.div_ceil(vmax * 8) - 1;
    for ci in 0..planes.len() {
        let c = f.comp[ci];
        let w = f.width.div_ceil(hmax * 8) * c.h as usize;
        let h = (last + 1) * c.v as usize;
        let p = &mut planes[ci];
        let bs = p.bs();
        // Sizes in whole usize, checked before anything is charged.
        let Some(nblk) = w.checked_mul(h) else {
            return ProgScans::fail(false, 0, 0, Error::OutOfMemory);
        };
        if let Err(e) = cx.ledger.charge(nblk as u64 * 128) {
            return ProgScans::fail(false, 0, 0, e);
        }
        match crate::limits::refill(std::mem::take(&mut p.blocks), nblk, [0i16; 64]) {
            Ok(b) => p.blocks = b,
            Err(e) => return ProgScans::fail(false, 0, 0, e),
        }
        if !p.coef && !p.skip {
            let Some(n) = nblk.checked_mul(bs * bs) else {
                return ProgScans::fail(false, 0, 0, Error::OutOfMemory);
            };
            if let Err(e) = cx.ledger.charge(n as u64) {
                return ProgScans::fail(false, 0, 0, e);
            }
            match crate::limits::refill(std::mem::take(&mut p.pix), n, 0u8) {
                Ok(b) => p.pix = b,
                Err(e) => return ProgScans::fail(false, 0, 0, e),
            }
        }
        p.blocks_wide = w;
        p.blocks_high = h;
        p.real_wide = (f.width * c.h as usize).div_ceil(hmax * 8);
        p.real_high = (f.height * c.v as usize).div_ceil(vmax * 8).min(h);
        p.latched = false;
        if !p.skip && !p.coef {
            let (rw, rh) = downsampled_dims(f, ci, hmax, vmax, bs);
            fill_padding(p, rw, rh.min(h * bs));
        }
    }

    let mut scanned = [false; 4];
    let mut ps = ProgState::new();
    let mut offset = f.scan_offset;
    let mut truncated = false;
    loop {
        let mut r = BitReader::new(src, offset);
        if let Err(e) = decode_prog_scan(cx, f, hmax, vmax, planes, &mut r, &mut ps) {
            return ProgScans::fail(true, 0, 0, e);
        }
        for k in 0..f.scan_n {
            scanned[f.scan_comp[k]] = true;
        }
        if !r.marker {
            let (c, ok) = next_marker(&mut r);
            r.marker = ok;
            r.marker_byte = c;
        }
        truncated = truncated || r.insufficient || r.eof || r.lost;
        if !r.marker || r.marker_byte == MARKER_EOI {
            return ProgScans {
                truncated,
                decoded: r.decoded,
                scanned_bytes: r.scanned,
                ps,
                scanned,
                err: None,
            };
        }
        match f.consume_from_marker(src, r.marker_byte, r.pos, true) {
            Err(e) => return ProgScans::fail(true, r.decoded, r.scanned, e),
            Ok((false, _)) => {
                truncated = truncated || f.overran;
                return ProgScans {
                    truncated,
                    decoded: r.decoded,
                    scanned_bytes: r.scanned,
                    ps,
                    scanned,
                    err: None,
                };
            }
            Ok((true, next)) => offset = next,
        }
    }
}

#[derive(Clone, Copy, Default)]
struct ScanComp {
    ci: usize,
    mcu_w: usize,
    mcu_h: usize,
}

/// Follows `jdphuff.c`, `jdcoefct.c`.
fn decode_prog_scan(
    cx: &Ctx,
    f: &Frame,
    hmax: usize,
    vmax: usize,
    planes: &mut [Plane],
    r: &mut BitReader,
    ps: &mut ProgState,
) -> Result<(), Error> {
    ps.scans += 1;
    latch_quant(f, planes)?;

    let (ss, se, ah, al) = (
        f.scan_ss as usize,
        f.scan_se as usize,
        f.scan_ah as usize,
        f.scan_al as usize,
    );
    let dc_band = ss == 0;
    let mut bad = al > 13 || ah != 0 && al != ah - 1;
    if dc_band {
        bad = bad || se != 0;
    } else {
        bad = bad || ss > se || se >= 64 || f.scan_n != 1;
    }
    if bad {
        return Err(Error::BadProgression);
    }
    for k in 0..f.scan_n {
        let ci = f.scan_comp[k];
        let (cb, pb) = (&mut ps.coef_bits[ci], &mut ps.prev_bits[ci]);
        if !dc_band && cb[0] < 0 {
            r.lost = true; // AC without a prior DC scan
        }
        for c in ss.min(1)..=9 {
            pb[c] = if ps.scans > 1 { cb[c] } else { 0 };
        }
        for c in ss..=se {
            if ah as i32 != (cb[c] as i32).max(0) {
                r.lost = true; // inconsistent progression
            }
            cb[c] = al as i8;
        }
    }

    let mut dc = [DerivedTable::EMPTY; 4];
    let mut ac = DerivedTable::EMPTY;
    if dc_band {
        if ah == 0 {
            // DC refinement needs no table
            for k in 0..f.scan_n {
                let s = f.scan_dc[k] as usize;
                if s > 3 || !f.has_dc[s] {
                    return Err(Error::NoHuffTable);
                }
                dc[k] = DerivedTable::build(&f.dc[s], true)?;
            }
        }
    } else {
        let s = f.scan_ac[0] as usize;
        if s > 3 || !f.has_ac[s] {
            return Err(Error::NoHuffTable);
        }
        ac = DerivedTable::build(&f.ac[s], false)?;
    }

    // Non-interleaved scans skip MCU padding.
    let interleaved = f.scan_n > 1;
    let mut comps = [ScanComp::default(); 4];
    for k in 0..f.scan_n {
        let ci = f.scan_comp[k];
        comps[k] = ScanComp {
            ci,
            mcu_w: 1,
            mcu_h: 1,
        };
        if interleaved {
            comps[k] = ScanComp {
                ci,
                mcu_w: f.comp[ci].h as usize,
                mcu_h: f.comp[ci].v as usize,
            };
        }
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
    let mut block_k = [0usize; MAX_MCU_BLOCKS];
    let mut block_x = [0usize; MAX_MCU_BLOCKS];
    let mut block_y = [0usize; MAX_MCU_BLOCKS];
    let mut n_blk = 0;
    for k in 0..f.scan_n {
        for yy in 0..comps[k].mcu_h {
            for xx in 0..comps[k].mcu_w {
                block_k[n_blk] = k;
                block_x[n_blk] = xx;
                block_y[n_blk] = yy;
                n_blk += 1;
            }
        }
    }

    let mut last_dc = [0i32; 4];
    let mut mcu = [(0usize, 0usize); MAX_MCU_BLOCKS]; // (plane, block index)
    let mut eobrun = 0u32;
    let ri = f.restart_interval;
    let (mut restarts_to_go, mut next_restart) = (ri, 0usize);
    let p1 = 1i16 << al;
    let m1 = -(1i16 << al);

    for my in 0..mcu_rows {
        check(cx.cancel)?;
        let imcu = if interleaved {
            my
        } else {
            my / f.comp[comps[0].ci].v as usize
        };
        for mx in 0..mcus_per_row {
            if !r.insufficient {
                ps.last_good = imcu;
            }
            if ri != 0 && restarts_to_go == 0 {
                process_restart(r, &mut last_dc[..f.scan_n], &mut next_restart);
                eobrun = 0;
                restarts_to_go = ri;
            }
            for i in 0..n_blk {
                let sc = comps[block_k[i]];
                let p = &planes[sc.ci];
                mcu[i] = (
                    sc.ci,
                    p.at(mx * sc.mcu_w + block_x[i], my * sc.mcu_h + block_y[i]),
                );
            }
            if dc_band && ah == 0 {
                if !r.insufficient {
                    dc_first(
                        r,
                        &dc,
                        &mut last_dc,
                        planes,
                        &mcu[..n_blk],
                        &block_k[..n_blk],
                        al,
                    )?;
                }
            } else if dc_band {
                for &(ci, idx) in &mcu[..n_blk] {
                    r.ensure(1);
                    if r.get_bits(1) != 0 {
                        planes[ci].blocks[idx][0] |= p1;
                    }
                }
            } else if ah == 0 {
                if !r.insufficient {
                    let (ci, idx) = mcu[0];
                    eobrun = ac_first(r, &ac, &mut planes[ci].blocks[idx], eobrun, ss, se, al);
                }
            } else if !r.insufficient {
                let (ci, idx) = mcu[0];
                eobrun = ac_refine(r, &ac, &mut planes[ci].blocks[idx], eobrun, ss, se, p1, m1);
            }
            restarts_to_go = restarts_to_go.wrapping_sub(1);
        }
    }
    Ok(())
}

/// `decode_mcu_DC_first`.
fn dc_first(
    r: &mut BitReader,
    dc: &[DerivedTable; 4],
    last_dc: &mut [i32; 4],
    planes: &mut [Plane],
    mcu: &[(usize, usize)],
    ks: &[usize],
    al: usize,
) -> Result<(), Error> {
    for (i, &(ci, idx)) in mcu.iter().enumerate() {
        let k = ks[i];
        let mut s = r.decode_huff(&dc[k]) as i32;
        if s != 0 {
            r.ensure(s);
            s = huff_extend(r.get_bits(s), s);
        }
        let l = last_dc[k];
        if l >= 0 && s > i32::MAX - l || l < 0 && s < i32::MIN - l {
            return Err(Error::BadDctCoef);
        }
        s += l;
        last_dc[k] = s;
        planes[ci].blocks[idx][0] = (s << al) as i16;
    }
    Ok(())
}

/// `decode_mcu_AC_first`.
fn ac_first(
    r: &mut BitReader,
    ac: &DerivedTable,
    blk: &mut Block,
    mut eobrun: u32,
    ss: usize,
    se: usize,
    al: usize,
) -> u32 {
    if eobrun > 0 {
        return eobrun - 1;
    }
    let mut k = ss;
    while k <= se {
        let sym = r.decode_huff(ac) as i32;
        let (run, s) = (sym >> 4, sym & 15);
        if s != 0 {
            k += run as usize;
            r.ensure(s);
            let v = huff_extend(r.get_bits(s), s);
            blk[NATURAL_ORDER_FAST[k & 127] as usize & 63] = (v << al) as i16;
        } else if run == 15 {
            k += 15;
        } else {
            eobrun = 1 << run;
            if run != 0 {
                r.ensure(run);
                eobrun += r.get_bits(run) as u32;
            }
            return eobrun - 1;
        }
        k += 1;
    }
    0
}

fn refine(r: &mut BitReader, c: &mut i16, p1: i16, m1: i16) {
    r.ensure(1);
    if r.get_bits(1) != 0 && *c & p1 == 0 {
        if *c >= 0 {
            *c = c.wrapping_add(p1);
        } else {
            *c = c.wrapping_add(m1);
        }
    }
}

/// `decode_mcu_AC_refine`.
#[allow(clippy::too_many_arguments)]
fn ac_refine(
    r: &mut BitReader,
    ac: &DerivedTable,
    blk: &mut Block,
    mut eobrun: u32,
    ss: usize,
    se: usize,
    p1: i16,
    m1: i16,
) -> u32 {
    let mut k = ss;
    if eobrun == 0 {
        while k <= se {
            let sym = r.decode_huff(ac) as i32;
            let (mut run, s) = (sym >> 4, sym & 15);
            let mut nv = 0i16;
            if s != 0 {
                if s != 1 {
                    r.lost = true; // new coefficient is always 1
                }
                r.ensure(1);
                nv = if r.get_bits(1) != 0 { p1 } else { m1 };
            } else if run != 15 {
                eobrun = 1 << run;
                if run != 0 {
                    r.ensure(run);
                    eobrun += r.get_bits(run) as u32;
                }
                break;
            }
            loop {
                let c = &mut blk[NATURAL_ORDER_FAST[k & 127] as usize & 63];
                if *c != 0 {
                    refine(r, c, p1, m1);
                } else {
                    run -= 1;
                    if run < 0 {
                        break;
                    }
                }
                k += 1;
                if k > se {
                    break;
                }
            }
            if nv != 0 {
                blk[NATURAL_ORDER_FAST[k & 127] as usize & 63] = nv;
            }
            k += 1;
        }
    }
    if eobrun > 0 {
        while k <= se {
            let c = &mut blk[NATURAL_ORDER_FAST[k & 127] as usize & 63];
            if *c != 0 {
                refine(r, c, p1, m1);
            }
            k += 1;
        }
        eobrun -= 1;
    }
    eobrun
}

/// Q01_POS..Q30_POS in natural order.
const SMOOTH_POS: [usize; 9] = [1, 8, 16, 9, 2, 3, 10, 17, 24];

/// Caps at `1 << al`.
fn smooth_pred(num: i64, q: i64, al: i32) -> i16 {
    let neg = num < 0;
    let num = num.abs();
    let mut pred = ((q << 7) + num).checked_div(q << 8).unwrap_or(0);
    if al > 0 && pred >= 1 << al {
        pred = (1 << al) - 1;
    }
    if neg {
        pred = -pred;
    }
    pred as i16
}

/// turbo `smoothing_ok`.
pub(crate) fn smoothing_ok(ps: &ProgState, f: &Frame, planes: &[Plane]) -> bool {
    let mut useful = false;
    for ci in 0..f.ncomp {
        let p = &planes[ci];
        if !p.latched {
            return false;
        }
        for pos in [0usize, 1, 8, 16, 9, 2, 3, 10, 17, 24] {
            if p.quant[pos] == 0 {
                return false;
            }
        }
        if ps.coef_bits[ci][0] < 0 {
            return false;
        }
        for c in 1..10 {
            if ps.coef_bits[ci][c] != 0 {
                useful = true;
            }
        }
    }
    useful
}

/// Waits for every scan, then IDCTs.
fn idct_progressive(
    cx: &Ctx,
    f: &Frame,
    planes: &mut [Plane],
    scanned: &[bool; 4],
    ps: &ProgState,
    vmax: usize,
) -> Result<(), Error> {
    let smooth = smoothing_ok(ps, f, planes);
    for ci in 0..f.ncomp {
        let p = &mut planes[ci];
        if p.skip || p.coef || !scanned[ci] {
            continue;
        }
        let view = RowView::of(p);
        let mut sm = if smooth {
            Some(Smoother::new(
                f,
                ps,
                ci,
                &p.quant,
                p.real_high,
                p.real_wide,
                &p.blocks,
                p.blocks_wide,
                vmax,
            ))
        } else {
            None
        };
        for by in 0..p.real_high {
            if by % f.comp[ci].v as usize == 0 {
                check(cx.cancel)?;
            }
            idct_row(
                &view,
                &p.blocks,
                sm.as_mut(),
                by,
                p.real_wide,
                &mut p.pix,
                0,
            );
        }
    }
    Ok(())
}

/// What `idct_row` needs of a plane besides its blocks and pixels.
pub(crate) struct RowView {
    bs: usize,
    stride: usize,
    quant: [u16; 64],
    idct: Idct,
}

impl RowView {
    pub(crate) fn of(p: &Plane) -> RowView {
        RowView {
            bs: p.bs(),
            stride: p.stride(),
            quant: p.quant,
            idct: p.idct,
        }
    }
}

/// IDCTs blocks `0..width` of block row `by` into `pix`, whose row 0 is block row `pix_by0`;
/// `sm`, when set, smooths them first.
pub(crate) fn idct_row(
    v: &RowView,
    blocks: &[Block],
    mut sm: Option<&mut Smoother>,
    by: usize,
    width: usize,
    pix: &mut [u8],
    pix_by0: usize,
) {
    let (bs, stride) = (v.bs, v.stride);
    if let Some(s) = sm.as_deref_mut() {
        s.start_row(by);
    }
    let blocks_wide = stride / bs;
    for bx in 0..width {
        let mut ws = blocks[by * blocks_wide + bx];
        if let Some(s) = sm.as_deref_mut() {
            s.estimate(&mut ws, bx);
        }
        let dst = &mut pix[(by - pix_by0) * bs * stride + bx * bs..];
        match v.idct {
            Idct::Hook(h) => {
                let mut deq = [0i32; 64];
                for (j, &c) in ws.iter().enumerate() {
                    deq[j] = dequantize(c, v.quant[j]) as i32;
                }
                h(dst, stride, &deq);
            }
            Idct::Default => idct_bs(bs, &ws, true, &v.quant, dst, stride),
        }
    }
}

/// Per-component turbo smoothing.
pub(crate) struct Smoother<'a> {
    q: [u16; 64],
    cur: [i8; 10],
    prv: [i8; 10], // this and previous scan
    last_good: usize,
    v: usize,
    total_rows: usize, // iMCU geometry
    high: usize,
    width: usize, // real size in blocks
    blocks: &'a [Block],
    stride: usize,

    bits: [i8; 10], // per-row precision
    change: bool,
    rows: [usize; 5], // index of block 0 of rows -2 to +2
    d: [i64; 25],     // 5x5 DC window
}

impl<'a> Smoother<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        f: &Frame,
        ps: &ProgState,
        ci: usize,
        quant: &[u16; 64],
        high: usize,
        width: usize,
        blocks: &'a [Block],
        stride: usize,
        vmax: usize,
    ) -> Smoother<'a> {
        let mut s = Smoother {
            q: *quant,
            cur: [0; 10],
            prv: [0; 10],
            last_good: ps.last_good,
            v: f.comp[ci].v as usize,
            total_rows: f.height.div_ceil(vmax * 8),
            high,
            width,
            blocks,
            stride,
            bits: [0; 10],
            change: false,
            rows: [0; 5],
            d: [0; 25],
        };
        for c in 0..10 {
            s.cur[c] = ps.coef_bits[ci][c];
            s.prv[c] = if ps.scans > 1 {
                ps.prev_bits[ci][c]
            } else {
                -1
            };
        }
        s
    }

    fn dc_at(&self, row: usize, bx: usize) -> i64 {
        self.blocks[self.rows[row] + bx][0] as i64
    }

    pub(crate) fn start_row(&mut self, by: usize) {
        let imcu = by / self.v;
        let mut block_rows = self.v;
        if imcu == self.total_rows - 1 {
            block_rows = self.high % self.v;
            if block_rows == 0 {
                block_rows = self.v;
            }
        }
        let ibr = imcu * block_rows + by % self.v;
        let ibrs = block_rows * self.total_rows;
        let (mut prev, mut next) = (by, by);
        if ibr > 0 {
            prev = by - 1;
        }
        if ibr + 1 < ibrs {
            next = by + 1;
        }
        let (mut pp, mut nn) = (prev, next);
        if ibr > 1 {
            pp = by - 2;
        }
        if ibr + 2 < ibrs {
            nn = by + 2;
        }
        for (i, row) in [pp, prev, by, next, nn].into_iter().enumerate() {
            self.rows[i] = row * self.stride;
            let v = self.dc_at(i, 0);
            for j in 0..5 {
                self.d[i * 5 + j] = v;
            }
        }
        self.bits = self.cur;
        if imcu > self.last_good {
            self.bits = self.prv;
        }
        self.change = true;
        for &b in &self.bits[1..] {
            self.change = self.change && b == -1;
        }
    }

    /// Slides the DC window.
    pub(crate) fn estimate(&mut self, ws: &mut Block, bx: usize) {
        for r in 0..5 {
            if bx == 0 && bx + 1 < self.width {
                let v = self.dc_at(r, 1);
                self.d[r * 5 + 3] = v;
                self.d[r * 5 + 4] = v;
            }
            if bx + 2 < self.width {
                self.d[r * 5 + 4] = self.dc_at(r, bx + 2);
            }
        }
        let q00 = self.q[0] as i64;
        let mut i = 0;
        while i < 9 && (i < 5 || self.change) {
            let pos = SMOOTH_POS[i];
            let al = self.bits[i + 1] as i32;
            if al != 0 && ws[pos] == 0 {
                ws[pos] = smooth_pred(
                    q00 * smooth_num(i, self.change, &self.d),
                    self.q[pos] as i64,
                    al,
                );
            }
            i += 1;
        }
        if self.change {
            ws[0] = smooth_pred(q00 * smooth_num(9, true, &self.d), q00, 0);
        }
        for r in (0..25).step_by(5) {
            self.d.copy_within(r + 1..r + 5, r);
        }
    }
}

/// The kernel; `i == 9` is DC.
fn smooth_num(i: usize, change: bool, d: &[i64; 25]) -> i64 {
    match i {
        0 => {
            if change {
                -d[0] - d[1] + d[3] + d[4] - 3 * d[5] + 13 * d[6] - 13 * d[8] + 3 * d[9] - 3 * d[10]
                    + 38 * d[11]
                    - 38 * d[13]
                    + 3 * d[14]
                    - 3 * d[15]
                    + 13 * d[16]
                    - 13 * d[18]
                    + 3 * d[19]
                    - d[20]
                    - d[21]
                    + d[23]
                    + d[24]
            } else {
                -7 * d[10] + 50 * d[11] - 50 * d[13] + 7 * d[14]
            }
        }
        1 => {
            if change {
                -d[0] - 3 * d[1] - 3 * d[2] - 3 * d[3] - d[4] - d[5]
                    + 13 * d[6]
                    + 38 * d[7]
                    + 13 * d[8]
                    - d[9]
                    + d[15]
                    - 13 * d[16]
                    - 38 * d[17]
                    - 13 * d[18]
                    + d[19]
                    + d[20]
                    + 3 * d[21]
                    + 3 * d[22]
                    + 3 * d[23]
                    + d[24]
            } else {
                -7 * d[2] + 50 * d[7] - 50 * d[17] + 7 * d[22]
            }
        }
        2 => {
            if change {
                d[2] + 2 * d[6] + 7 * d[7] + 2 * d[8] - 5 * d[11] - 14 * d[12] - 5 * d[13]
                    + 2 * d[16]
                    + 7 * d[17]
                    + 2 * d[18]
                    + d[22]
            } else {
                -d[2] + 13 * d[7] - 24 * d[12] + 13 * d[17] - d[22]
            }
        }
        3 => {
            if change {
                -d[0] + d[4] + 9 * d[6] - 9 * d[8] - 9 * d[16] + 9 * d[18] + d[20] - d[24]
            } else {
                d[9] + d[15] - 10 * d[16] + 10 * d[18] - d[1] - d[19] + d[21] - d[23] + d[3] - d[5]
                    + 10 * d[6]
                    - 10 * d[8]
            }
        }
        4 => {
            if change {
                2 * d[6] - 5 * d[7] + 2 * d[8] + d[10] + 7 * d[11] - 14 * d[12]
                    + 7 * d[13]
                    + d[14]
                    + 2 * d[16]
                    - 5 * d[17]
                    + 2 * d[18]
            } else {
                -d[10] + 13 * d[11] - 24 * d[12] + 13 * d[13] - d[14]
            }
        }
        5 => d[6] - d[8] + 2 * d[11] - 2 * d[13] + d[16] - d[18],
        6 => d[6] - 3 * d[7] + d[8] - d[16] + 3 * d[17] - d[18],
        7 => d[6] - d[8] - 3 * d[11] + 3 * d[13] + d[16] - d[18],
        8 => d[6] + 2 * d[7] + d[8] - d[16] - 2 * d[17] - d[18],
        _ => {
            -2 * d[0] - 6 * d[1] - 8 * d[2] - 6 * d[3] - 2 * d[4] - 6 * d[5]
                + 6 * d[6]
                + 42 * d[7]
                + 6 * d[8]
                - 6 * d[9]
                - 8 * d[10]
                + 42 * d[11]
                + 152 * d[12]
                + 42 * d[13]
                - 8 * d[14]
                - 6 * d[15]
                + 6 * d[16]
                + 42 * d[17]
                + 6 * d[18]
                - 6 * d[19]
                - 2 * d[20]
                - 6 * d[21]
                - 8 * d[22]
                - 6 * d[23]
                - 2 * d[24]
        }
    }
}
