//! Marker-segment parser, ported from jpeg-go's `internal/codec/marker.go` (turbo `jdmarker.c`).

use crate::Error;
use crate::huff::{HuffTable, std_tables};

pub(crate) const MARKER_SOI: u8 = 0xd8;
pub(crate) const MARKER_EOI: u8 = 0xd9;
const MARKER_SOS: u8 = 0xda;
const MARKER_DQT: u8 = 0xdb;
const MARKER_DHT: u8 = 0xc4;
const MARKER_DRI: u8 = 0xdd;
const MARKER_APP0: u8 = 0xe0;
const MARKER_APP14: u8 = 0xee;

/// A DQT/coefficient zigzag index to its natural (raster) position (turbo `jdmarker.c`).
pub(crate) const ZIGZAG_TO_NATURAL: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// One SOF component descriptor.
#[derive(Clone, Copy, Default)]
pub(crate) struct Component {
    pub id: u8,
    pub h: u8,
    pub v: u8,
    pub tq: u8,
}

/// Marker-segment state up to SOS (or EOI for a tables-only stream).
#[derive(Clone)]
pub(crate) struct Frame {
    pub width: usize,
    pub height: usize,
    pub ncomp: usize,
    pub comp: [Component; 4],
    pub restart_interval: usize,
    pub rgb: bool,
    pub has_sof: bool,
    /// Accept 4-component (CMYK/YCCK) frames; only `decode_coefficients` sets it.
    pub allow_four: bool,

    jfif: bool,
    adobe: bool,
    adobe_transform: u8,

    pub quant: [[u16; 64]; 4],
    pub has_quant: [bool; 4],
    pub progressive: bool,
    /// turbo's fake EOI warns.
    pub overran: bool,
    pub scan_ss: u8,
    pub scan_se: u8,
    pub scan_ah: u8,
    pub scan_al: u8,

    /// The current SOS: components in scan order (indices into `comp`) and each one's Huffman
    /// table selectors.
    pub scan_n: usize,
    pub scan_comp: [usize; 4],
    pub scan_dc: [u8; 4],
    pub scan_ac: [u8; 4],
    /// `src` offset of the first entropy byte after SOS.
    pub scan_offset: usize,

    pub dc: [HuffTable; 4],
    pub ac: [HuffTable; 4],
    pub has_dc: [bool; 4],
    pub has_ac: [bool; 4],
}

/// Where a marker walk ended.
struct Walk {
    sos: bool,
    offset: usize,
    done: bool,
}

impl Frame {
    pub(crate) fn new() -> Frame {
        Frame {
            width: 0,
            height: 0,
            ncomp: 0,
            comp: [Component::default(); 4],
            restart_interval: 0,
            rgb: false,
            has_sof: false,
            allow_four: false,
            jfif: false,
            adobe: false,
            adobe_transform: 0,
            quant: [[0; 64]; 4],
            has_quant: [false; 4],
            progressive: false,
            overran: false,
            scan_ss: 0,
            scan_se: 0,
            scan_ah: 0,
            scan_al: 0,
            scan_n: 0,
            scan_comp: [0; 4],
            scan_dc: [0; 4],
            scan_ac: [0; 4],
            scan_offset: 0,
            dc: [HuffTable::EMPTY; 4],
            ac: [HuffTable::EMPTY; 4],
            has_dc: [false; 4],
            has_ac: [false; 4],
        }
    }

    /// Reads an optional abbreviated tables stream, then `src`, stopping at SOS. `src` must
    /// contain a SOF; `tables` (when given) supplies only DQT and DHT segments.
    pub(crate) fn parse(&mut self, src: &[u8], tables: Option<&[u8]>) -> Result<(), Error> {
        if let Some(t) = tables {
            self.parse_tables(t)?;
        }
        self.parse_frame(src)
    }

    /// Reads an abbreviated tables stream (DQT/DHT only).
    pub(crate) fn parse_tables(&mut self, tables: &[u8]) -> Result<(), Error> {
        let w = self.parse_segments(tables, true)?;
        if w.sos || self.has_sof {
            return Err(Error::TablesStream);
        }
        Ok(())
    }

    /// Reads `src`, stopping at SOS. Any DQT/DHT already in `self` apply unless `src` carries its
    /// own.
    pub(crate) fn parse_frame(&mut self, src: &[u8]) -> Result<(), Error> {
        let w = self.parse_segments(src, false)?;
        self.scan_offset = w.offset;
        if !w.sos {
            return Err(if self.has_sof {
                Error::NoSos
            } else {
                Error::NoSof
            });
        }
        self.resolve_color();
        Ok(())
    }

    fn parse_segments(&mut self, b: &[u8], tables: bool) -> Result<Walk, Error> {
        if b.len() < 2 || b[0] != 0xff || b[1] != MARKER_SOI {
            return Err(Error::NoSoi);
        }
        // turbo get_soi resets these per SOI; has_sof too.
        self.restart_interval = 0;
        self.jfif = false;
        self.adobe = false;
        self.adobe_transform = 0;
        self.has_sof = false;
        self.overran = false;
        self.consume_from(b, 2, tables)
    }

    fn consume_from(&mut self, b: &[u8], i: usize, need_seg: bool) -> Result<Walk, Error> {
        match find_marker(b, i) {
            None => Ok(Walk {
                sos: false,
                offset: 0,
                done: true,
            }), // fake EOI
            Some((marker, i)) => self.consume_from_marker_walk(b, marker, i, need_seg),
        }
    }

    /// Processes `marker` (already identified, code position `i`) then keeps consuming further
    /// markers. Returns whether SOS was reached and the offset of the first entropy byte. Also
    /// serves a multi-scan continuation.
    pub(crate) fn consume_from_marker(
        &mut self,
        b: &[u8],
        marker: u8,
        i: usize,
        need_seg: bool,
    ) -> Result<(bool, usize), Error> {
        let w = self.consume_from_marker_walk(b, marker, i, need_seg)?;
        Ok((w.sos, w.offset))
    }

    fn consume_from_marker_walk(
        &mut self,
        b: &[u8],
        mut marker: u8,
        mut i: usize,
        need_seg: bool,
    ) -> Result<Walk, Error> {
        loop {
            let w = self.handle_marker(b, marker, i, need_seg)?;
            if w.done {
                return Ok(w);
            }
            match find_marker(b, w.offset) {
                None => {
                    self.overran = true;
                    return Ok(Walk {
                        sos: false,
                        offset: 0,
                        done: true,
                    }); // fake EOI
                }
                Some((m, j)) => {
                    marker = m;
                    i = j;
                }
            }
        }
    }

    /// Parses the segment following an already-identified marker. `done` is true once the header
    /// work for this marker is finished: SOS or EOI end the scan, anything else just moves `i`
    /// past its segment.
    fn handle_marker(
        &mut self,
        b: &[u8],
        marker: u8,
        mut i: usize,
        need_seg: bool,
    ) -> Result<Walk, Error> {
        let end = Walk {
            sos: false,
            offset: 0,
            done: true,
        };
        if marker == MARKER_EOI {
            return Ok(end);
        }
        if marker == MARKER_SOI {
            return Err(Error::DupSoi);
        }
        if marker == 0x01 || (0xd0..=0xd7).contains(&marker) {
            return Ok(Walk {
                sos: false,
                offset: i,
                done: false,
            }); // turbo: parameterless, skipped
        }
        let mut lb = [0u8; 2];
        fake_eoi(&mut lb, b, i);
        let mut length = (lb[0] as usize) << 8 | lb[1] as usize;
        if length < 2 && ((0xe0..=0xef).contains(&marker) || marker == 0xfe || marker == 0xdc) {
            length = 2; // turbo skip_variable ignores short lengths
        }
        if length < 2 {
            return Err(Error::BadLength);
        }
        if i + length > b.len() {
            return self.overrun(b, marker, i, length, need_seg);
        }
        let seg = &b[i + 2..i + length];
        i += length;
        if marker == MARKER_SOS {
            self.read_sos(seg)?;
            return Ok(Walk {
                sos: true,
                offset: i,
                done: true,
            });
        }
        self.read_segment(marker, seg)?;
        Ok(Walk {
            sos: false,
            offset: i,
            done: false,
        })
    }

    /// Reads a segment that runs past the data. Only fake EOI bytes follow, so the stream ends at
    /// EOI unless the segment is fatal. `need_seg` is false in a frame header, where that EOI is
    /// fatal anyway.
    fn overrun(
        &mut self,
        b: &[u8],
        marker: u8,
        i: usize,
        length: usize,
        need_seg: bool,
    ) -> Result<Walk, Error> {
        if marker != MARKER_SOS {
            self.overran = true;
        }
        let end = Walk {
            sos: false,
            offset: 0,
            done: true,
        };
        if marker == MARKER_SOS {
            // turbo get_sos: length is 6 + 2*Ns
            let mut buf = [0u8; 12];
            if length - 2 > buf.len() {
                return Err(Error::BadLength);
            }
            let seg = &mut buf[..length - 2];
            fake_eoi(seg, b, i + 2);
            // i+length, not len(b): the entropy scan's bit reader picks up the fake-EOI cycle
            // where this header read left it, which may be mid-cycle.
            self.read_sos(seg)?;
            return Ok(Walk {
                sos: true,
                offset: i + length,
                done: true,
            });
        }
        if marker == MARKER_DRI {
            if length != 4 {
                return Err(Error::BadLength);
            }
        } else if marker == MARKER_DHT || marker == MARKER_DQT || marker == 0xcc {
            if need_seg {
                let mut seg = crate::limits::filled(length - 2, 0u8)?; // hostile streams only
                fake_eoi(&mut seg, b, i + 2);
                self.read_segment(marker, &seg)?;
            }
        } else if (0xe0..=0xef).contains(&marker) || marker == 0xfe || marker == 0xdc {
        } else {
            self.read_segment(marker, &b[(i + 2).min(b.len())..])?;
            return Ok(end);
        }
        Ok(end)
    }

    fn read_segment(&mut self, marker: u8, seg: &[u8]) -> Result<(), Error> {
        match marker {
            MARKER_APP0 => {
                if seg.len() >= 14 && &seg[..5] == b"JFIF\0" {
                    self.jfif = true;
                }
            }
            MARKER_APP14 => {
                if seg.len() >= 12 && &seg[..5] == b"Adobe" {
                    self.adobe = true;
                    self.adobe_transform = seg[11];
                }
            }
            MARKER_DQT => return self.read_dqt(seg),
            MARKER_DHT => return self.read_dht(seg),
            MARKER_DRI => {
                if seg.len() != 2 {
                    return Err(Error::BadLength);
                }
                self.restart_interval = (seg[0] as usize) << 8 | seg[1] as usize;
            }
            0xc0 | 0xc1 => return self.read_sof(seg, false),
            0xc2 => return self.read_sof(seg, true),
            // Out-of-scope SOFn and JPG.
            0xc3 | 0xc5 | 0xc6 | 0xc7 | 0xc8 | 0xc9 | 0xca | 0xcb | 0xcd | 0xce | 0xcf => {
                return Err(Error::Unsupported);
            }
            0xcc => return read_dac(seg),
            0xdc | 0xfe => {} // DNL, COM: skip.
            _ => {
                if !(0xe1..=0xef).contains(&marker) {
                    return Err(Error::UnknownMarker); // turbo rejects DHP, EXP, JPGn, RESn
                }
            }
        }
        Ok(())
    }

    fn read_dqt(&mut self, mut seg: &[u8]) -> Result<(), Error> {
        while !seg.is_empty() {
            let pq = seg[0] >> 4;
            let tq = (seg[0] & 0xf) as usize;
            if tq > 3 {
                return Err(Error::DqtIndex);
            }
            seg = &seg[1..];
            if pq == 0 {
                if seg.len() < 64 {
                    return Err(Error::BadLength);
                }
                for i in 0..64 {
                    self.quant[tq][ZIGZAG_TO_NATURAL[i]] = seg[i] as u16;
                }
                seg = &seg[64..];
            } else {
                // turbo: any nonzero Pq is 16-bit
                if seg.len() < 128 {
                    return Err(Error::BadLength);
                }
                for i in 0..64 {
                    self.quant[tq][ZIGZAG_TO_NATURAL[i]] =
                        (seg[i * 2] as u16) << 8 | seg[i * 2 + 1] as u16;
                }
                seg = &seg[128..];
            }
            self.has_quant[tq] = true;
        }
        Ok(())
    }

    fn read_dht(&mut self, mut seg: &[u8]) -> Result<(), Error> {
        while !seg.is_empty() {
            if seg.len() < 17 {
                return Err(Error::BadLength);
            }
            let index = seg[0];
            if index & 0xef > 3 {
                return Err(Error::DhtIndex); // turbo jdmarker.c:488-492
            }
            let mut bits = [0u8; 16];
            bits.copy_from_slice(&seg[1..17]);
            let n: usize = bits.iter().map(|&c| c as usize).sum();
            seg = &seg[17..];
            if n > 256 || seg.len() < n {
                return Err(Error::HuffTable); // turbo jdmarker.c:475
            }
            let tbl = HuffTable::build(bits, &seg[..n]);
            seg = &seg[n..];
            if index & 0x10 == 0 {
                self.dc[index as usize] = tbl;
                self.has_dc[index as usize] = true;
            } else {
                self.ac[(index & 3) as usize] = tbl;
                self.has_ac[(index & 3) as usize] = true;
            }
        }
        Ok(())
    }

    /// Follows turbo `get_sof`, plus `initial_setup`'s size and sampling checks.
    fn read_sof(&mut self, seg: &[u8], progressive: bool) -> Result<(), Error> {
        if self.has_sof {
            return Err(Error::DupSof);
        }
        if seg.len() < 6 {
            return Err(Error::BadLength);
        }
        let height = (seg[1] as usize) << 8 | seg[2] as usize;
        let width = (seg[3] as usize) << 8 | seg[4] as usize;
        let nc = seg[5] as usize;
        if height == 0 || width == 0 || nc == 0 {
            return Err(Error::EmptyImage);
        }
        if seg.len() != 6 + nc * 3 {
            return Err(Error::BadLength);
        }
        if height > 65500 || width > 65500 {
            return Err(Error::TooBig);
        }
        if seg[0] != 8 || (nc != 1 && nc != 3 && !(nc == 4 && self.allow_four)) {
            return Err(Error::Unsupported); // only 8-bit gray or 3 components (4 for coefficients)
        }
        let mut comp = [Component::default(); 4];
        for (i, c) in comp.iter_mut().enumerate().take(nc) {
            let off = 6 + i * 3;
            *c = Component {
                id: seg[off],
                h: seg[off + 1] >> 4,
                v: seg[off + 1] & 0xf,
                tq: seg[off + 2],
            };
        }
        validate_sampling(nc, &comp)?;
        self.width = width;
        self.height = height;
        self.ncomp = nc;
        self.comp = comp;
        self.has_sof = true;
        self.progressive = progressive;
        Ok(())
    }

    /// Checks the scan header as turbo `get_sos` does.
    fn read_sos(&mut self, seg: &[u8]) -> Result<(), Error> {
        if !self.has_sof {
            return Err(Error::SosBeforeSof);
        }
        if seg.is_empty() {
            return Err(Error::BadLength);
        }
        let n = seg[0] as usize;
        if !(1..=4).contains(&n) || seg.len() != 4 + 2 * n {
            return Err(Error::BadLength);
        }
        let mut picked = [0usize; 4];
        for k in 0..n {
            let cc = seg[1 + 2 * k];
            // turbo quirk: ci < k never matches.
            let mut ci = 0;
            while ci < self.ncomp && (self.comp[ci].id != cc || ci < k) {
                ci += 1;
            }
            if ci == self.ncomp {
                return Err(Error::ComponentId);
            }
            if picked[..k].contains(&ci) {
                return Err(Error::ComponentId);
            }
            picked[k] = ci;

            let td = seg[2 + 2 * k] >> 4;
            let ta = seg[2 + 2 * k] & 0xf;
            // jdphuff.c rejects bad selectors late.
            if !self.progressive && (td > 3 || ta > 3) {
                return Err(Error::ScanSelector);
            }
            self.scan_comp[k] = ci;
            self.scan_dc[k] = td;
            self.scan_ac[k] = ta;
        }
        self.scan_n = n;
        self.scan_ss = seg[1 + 2 * n];
        self.scan_se = seg[2 + 2 * n];
        self.scan_ah = seg[3 + 2 * n] >> 4;
        self.scan_al = seg[3 + 2 * n] & 15;
        Ok(())
    }

    /// Applies a caller's colour model after the stream's own reading (`resolve_color`).
    pub(crate) fn set_color_model(&mut self, m: crate::ColorModel) {
        if m == crate::ColorModel::Rgb && self.ncomp == 3 {
            self.rgb = true;
        }
    }

    /// Follows turbo `jdapimin.c`: JFIF wins over Adobe, Adobe transform 0 means RGB, and with
    /// neither marker ids R,G,B mean RGB.
    fn resolve_color(&mut self) {
        self.rgb = if self.ncomp == 1 || self.ncomp == 4 || self.jfif {
            false
        } else if self.adobe {
            self.adobe_transform == 0
        } else {
            self.ncomp == 3
                && self.comp[0].id == b'R'
                && self.comp[1].id == b'G'
                && self.comp[2].id == b'B'
        };
    }

    /// turbo `jinit_huff_decoder`: fills undefined tables 0 and 1 with the standard ones.
    pub(crate) fn install_std_huff_tables(&mut self) {
        let std = std_tables();
        for i in 0..2 {
            if !self.has_dc[i] {
                self.dc[i] = std.dc[i];
                self.has_dc[i] = true;
            }
            if !self.has_ac[i] {
                self.ac[i] = std.ac[i];
                self.has_ac[i] = true;
            }
        }
    }
}

/// The next marker code in `b` at or after `i` and the index right after its 2-byte code,
/// skipping stuffed 0xff padding like turbo `next_marker`; TEM and RSTn (no length) are skipped
/// too. None at EOF.
pub(crate) fn find_marker(b: &[u8], mut i: usize) -> Option<(u8, usize)> {
    loop {
        let mut m = 0u8;
        while m == 0 {
            while i < b.len() && b[i] != 0xff {
                i += 1;
            }
            while i < b.len() && b[i] == 0xff {
                i += 1;
            }
            if i >= b.len() {
                return None;
            }
            m = b[i];
            i += 1;
        }
        if m == 0x01 || (0xd0..=0xd7).contains(&m) {
            continue; // TEM/RSTn carry no length
        }
        return Some((m, i));
    }
}

/// Fills `dst` from `b[i..]`, then with turbo's fake EOI bytes: past the data, `jdatasrc-tj.c`
/// returns FF D9 on every refill.
fn fake_eoi(dst: &mut [u8], b: &[u8], i: usize) {
    for (k, d) in dst.iter_mut().enumerate() {
        let j = i + k;
        *d = if j < b.len() {
            b[j]
        } else if (j - b.len()).is_multiple_of(2) {
            0xff
        } else {
            MARKER_EOI
        };
    }
}

/// Only validates, as turbo `get_dac` does.
fn read_dac(mut seg: &[u8]) -> Result<(), Error> {
    if !seg.len().is_multiple_of(2) {
        return Err(Error::BadLength);
    }
    while !seg.is_empty() {
        let (index, val) = (seg[0], seg[1]);
        if index >= 32 {
            return Err(Error::DacIndex);
        }
        if index < 16 && val & 0x0f > val >> 4 {
            return Err(Error::DacIndex);
        }
        seg = &seg[2..];
    }
    Ok(())
}

/// Accepts gray at any legal factor (1..4) and, for 3 components, 4:4:4, 4:2:2, 4:2:0, 4:4:0,
/// 4:1:1 and 4:1:0 (Y 1x2, 4x1, 4x2) with Cb/Cr unsubsampled. 4 components (coefficients only)
/// take any factors 1..4 within a 10-block MCU.
fn validate_sampling(nc: usize, c: &[Component; 4]) -> Result<(), Error> {
    if nc == 1 {
        if c[0].h < 1 || c[0].h > 4 || c[0].v < 1 || c[0].v > 4 {
            return Err(Error::Unsupported);
        }
        return Ok(());
    }
    if nc == 4 {
        // Every factor 1..4, and an interleaved MCU of at most 10 blocks (MAX_MCU_BLOCKS).
        let ok = c
            .iter()
            .all(|k| (1..=4).contains(&k.h) && (1..=4).contains(&k.v));
        let blocks: usize = c.iter().map(|k| k.h as usize * k.v as usize).sum();
        return if ok && blocks <= 10 {
            Ok(())
        } else {
            Err(Error::Unsupported)
        };
    }
    if c[1].h != 1 || c[1].v != 1 || c[2].h != 1 || c[2].v != 1 {
        return Err(Error::Unsupported);
    }
    match (c[0].h, c[0].v) {
        (1, 1) | (2, 1) | (2, 2) | (1, 2) | (4, 1) | (4, 2) => Ok(()),
        _ => Err(Error::Unsupported),
    }
}
