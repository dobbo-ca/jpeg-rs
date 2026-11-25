//! Huffman tables and the entropy bit reader, ported from jpeg-go's `internal/codec/huff.go`
//! (libjpeg-turbo `jdhuff.c`).

use crate::Error;
use std::sync::OnceLock;

/// A canonical Huffman table from a DHT segment: `bits` are the counts of codes of each length
/// 1..16, and `values[..n]` lists the symbols in code order. `code` and `length` give each
/// value's canonical code.
#[derive(Clone, Copy)]
pub(crate) struct HuffTable {
    pub bits: [u8; 16],
    pub n: usize,
    pub values: [u8; 256],
    pub code: [u16; 256],
    pub length: [u8; 256],
}

impl HuffTable {
    pub(crate) const EMPTY: HuffTable = HuffTable {
        bits: [0; 16],
        n: 0,
        values: [0; 256],
        code: [0; 256],
        length: [0; 256],
    };

    /// Assigns canonical Huffman codes to values in bit-length order (JPEG Annex C). `values`
    /// holds at most 256 symbols.
    pub(crate) fn build(bits: [u8; 16], values: &[u8]) -> HuffTable {
        let mut t = HuffTable::EMPTY;
        t.bits = bits;
        t.n = values.len();
        t.values[..values.len()].copy_from_slice(values);
        let (mut code, mut k) = (0u16, 0usize);
        for length in 1..=16usize {
            for _ in 0..bits[length - 1] {
                if k < 256 {
                    t.code[k] = code;
                    t.length[k] = length as u8;
                }
                code = code.wrapping_add(1);
                k += 1;
            }
            code <<= 1;
        }
        t
    }
}

/// Standard tables, JPEG Annex K.3 (turbo `jstdhuff.c`): index 0 luminance, 1 chrominance.
pub(crate) struct StdTables {
    pub dc: [HuffTable; 2],
    pub ac: [HuffTable; 2],
}

const STD_AC_LUMA: [u8; 162] = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
    0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,
];

const STD_AC_CHROMA: [u8; 162] = [
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
    0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
    0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
    0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
    0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
    0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,
];

pub(crate) fn std_tables() -> &'static StdTables {
    static T: OnceLock<StdTables> = OnceLock::new();
    T.get_or_init(|| {
        let dc_vals: Vec<u8> = (0..12).collect();
        StdTables {
            dc: [
                HuffTable::build([0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], &dc_vals),
                HuffTable::build([0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], &dc_vals),
            ],
            ac: [
                HuffTable::build(
                    [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d],
                    &STD_AC_LUMA,
                ),
                HuffTable::build(
                    [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77],
                    &STD_AC_CHROMA,
                ),
            ],
        }
    })
}

/// The bit-reader's lookahead width, matching turbo's `HUFF_LOOKAHEAD`: the fast path's refill
/// point, and so where it meets FF FF, depends on this exactly.
pub(crate) const LOOKAHEAD_BITS: i32 = 8;

/// Bits of the combined table's index: a code and its value bits that fit in them are one lookup.
pub(crate) const COMB_BITS: usize = 10;

/// `jpeg_make_d_derived_tbl`'s decode-side lookup: a lookahead table for short codes plus
/// maxcode/valoffset for the slow path, and the Go port's "combined" table, which gives a short
/// code and its value bits in one lookup (a speed-up with the same output).
#[derive(Clone, Copy)]
pub(crate) struct DerivedTable {
    /// `lookup[b]` packs `(length << 8) | symbol` for a code found within 8 bits; unfilled
    /// entries stay 0.
    pub lookup: [u16; 256],
    /// `combined[b]` is 0 for a miss (no code and value within `COMB_BITS` bits), else `length +
    /// size` in bits 0-5 (the bits to consume), `length` in bits 8-11, the run in bits 12-15 and the
    /// extended value in bits 16-31. Bit 6 flags a symbol with no value (EOB, ZRL, a DC difference
    /// of 0): then only `length` is consumed, in bits 0-5, and no refill follows.
    pub combined: [u32; 1 << COMB_BITS],
    pub maxcode: [i32; 18],
    pub valoffset: [i32; 18],
    pub values: [u8; 256],
}

impl DerivedTable {
    pub(crate) const EMPTY: DerivedTable = DerivedTable {
        lookup: [0; 256],
        combined: [0; 1 << COMB_BITS],
        maxcode: [0; 18],
        valoffset: [0; 18],
        values: [0; 256],
    };

    pub(crate) fn build(t: &HuffTable, is_dc: bool) -> Result<DerivedTable, Error> {
        let mut d = DerivedTable::EMPTY;
        d.values = t.values;
        let (mut code, mut p) = (0i32, 0i32);
        for length in 1..=16usize {
            let n = t.bits[length - 1] as i32;
            if n > 0 {
                d.valoffset[length] = p - code;
                code += n;
                p += n;
                d.maxcode[length] = code - 1;
            } else {
                d.maxcode[length] = -1;
            }
            // A code must still fit in length bits: no code is all ones.
            if code >= 1 << length {
                return Err(Error::HuffTable);
            }
            code <<= 1;
        }
        d.maxcode[17] = 1 << 30; // sentinel so the slow loop always terminates
        if is_dc && t.values[..t.n].iter().any(|&s| s > 15) {
            return Err(Error::HuffTable);
        }
        let mut idx = 0usize;
        for length in 1..=COMB_BITS {
            let n = t.bits[length - 1] as usize;
            for i in 0..n {
                let sym = t.values[idx + i];
                if length <= 8 {
                    let base = (t.code[idx + i] as usize) << (8 - length);
                    let entry = (length as u16) << 8 | sym as u16;
                    for f in 0..1usize << (8 - length) {
                        d.lookup[base + f] = entry;
                    }
                }
                let (run, size) = if is_dc {
                    (0, sym as usize)
                } else {
                    ((sym >> 4) as usize, (sym & 15) as usize)
                };
                if length + size > COMB_BITS {
                    continue;
                }
                let base = (t.code[idx + i] as usize) << (COMB_BITS - length);
                if size == 0 {
                    let ce = length as u32 | 64 | (length as u32) << 8 | (run as u32) << 12;
                    for f in 0..1usize << (COMB_BITS - length) {
                        d.combined[base + f] = ce;
                    }
                    continue;
                }
                let pad = COMB_BITS - length - size;
                for v in 0..1usize << size {
                    let val = huff_extend(v as i32, size as i32);
                    let ce = (length + size) as u32
                        | (length as u32) << 8
                        | (run as u32) << 12
                        | (val as i16 as u16 as u32) << 16;
                    for f in 0..1usize << pad {
                        d.combined[base + (v << pad) + f] = ce;
                    }
                }
            }
            idx += n;
        }
        Ok(d)
    }
}

/// `HUFF_EXTEND`: an s-bit magnitude-and-sign value to a signed int. `s` is at least 1.
#[inline]
pub(crate) fn huff_extend(v: i32, s: i32) -> i32 {
    let neg = (v - (1 << ((s - 1) & 31))) >> 31;
    v + (neg & ((-1i32 << (s & 31)) + 1))
}

/// turbo's `MIN_GET_BITS` for a 64-bit bit buffer.
const MIN_GET_BITS: i32 = 57;

/// The Huffman entropy bit reader: a 64-bit buffer with marker detection and byte unstuffing
/// (turbo `jpeg_fill_bit_buffer`).
#[derive(Clone, Copy)]
pub(crate) struct BitReader<'a> {
    pub data: &'a [u8],
    pub pos: usize,
    pub buf: u64,
    pub bits_left: i32,
    /// `cinfo->unread_marker != 0`: once set, no more real bytes are read and further fills
    /// zero-pad.
    pub marker: bool,
    pub marker_byte: u8,
    /// `entropy->insufficient_data`.
    pub insufficient: bool,
    /// Insufficient, a resync or a bad code ever seen; sticky.
    pub lost: bool,
    /// Ran off `data`; turbo fakes an EOI and warns.
    pub eof: bool,
    /// Bytes read by fill and `next_marker`.
    pub decoded: usize,
    pub scanned: usize,
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(data: &'a [u8], pos: usize) -> Self {
        BitReader {
            data,
            pos,
            buf: 0,
            bits_left: 0,
            marker: false,
            marker_byte: 0,
            insufficient: false,
            lost: false,
            eof: false,
            decoded: 0,
            scanned: 0,
        }
    }

    /// The byte at `pos`: real if inside `data`, else turbo's fake EOI repeating past it.
    fn virtual_byte(&self, pos: usize) -> u8 {
        if pos < self.data.len() {
            return self.data[pos];
        }
        if (pos - self.data.len()).is_multiple_of(2) {
            0xff
        } else {
            0xd9
        }
    }

    /// The next entropy-stream byte, unstuffing FF 00 and stopping (marker set) on a real marker
    /// or the fake EOI.
    fn next_logical_byte(&mut self) -> Option<u8> {
        if self.pos >= self.data.len() {
            self.eof = true;
        }
        let mut c = self.virtual_byte(self.pos);
        self.pos += 1;
        if c != 0xff {
            return Some(c);
        }
        loop {
            if self.pos >= self.data.len() {
                self.eof = true;
            }
            c = self.virtual_byte(self.pos);
            self.pos += 1;
            if c != 0xff {
                break;
            }
        }
        if c == 0 {
            return Some(0xff);
        }
        self.marker = true;
        self.marker_byte = c;
        None
    }

    /// Loads up to `MIN_GET_BITS` bits, or pads with zero bits once a marker is hit and `nbits`
    /// still exceeds what is buffered.
    #[inline]
    pub(crate) fn fill(&mut self, nbits: i32) {
        if !self.marker {
            let p = self.pos;
            while self.bits_left < MIN_GET_BITS {
                match self.next_logical_byte() {
                    Some(c) => {
                        self.buf = self.buf << 8 | c as u64;
                        self.bits_left += 8;
                    }
                    None => break,
                }
            }
            self.decoded += self.pos - p;
        }
        if nbits > self.bits_left {
            self.buf <<= (MIN_GET_BITS - self.bits_left) as u32;
            self.bits_left = MIN_GET_BITS;
            self.insufficient = true;
            self.lost = true;
        }
    }

    #[inline]
    pub(crate) fn ensure(&mut self, nbits: i32) {
        if self.bits_left < nbits {
            self.fill(nbits);
        }
    }

    #[inline]
    pub(crate) fn get_bits(&mut self, n: i32) -> i32 {
        self.bits_left -= n;
        ((self.buf >> (self.bits_left as u32 & 63)) as i32) & ((1i32 << (n as u32 & 31)) - 1)
    }

    fn peek_bits(&self, n: i32) -> i32 {
        ((self.buf >> ((self.bits_left - n) as u32 & 63)) as i32) & ((1i32 << (n as u32 & 31)) - 1)
    }

    /// The next Huffman symbol (turbo `HUFF_DECODE`).
    #[inline]
    pub(crate) fn decode_huff(&mut self, d: &DerivedTable) -> u8 {
        if self.bits_left < LOOKAHEAD_BITS {
            self.fill(0);
            if self.bits_left < LOOKAHEAD_BITS {
                return self.slow_huff(d, 1);
            }
        }
        let entry = d.lookup[self.peek_bits(LOOKAHEAD_BITS) as usize & 255];
        let nb = (entry >> LOOKAHEAD_BITS) as i32;
        if (1..=LOOKAHEAD_BITS).contains(&nb) {
            self.bits_left -= nb;
            return (entry & 0xff) as u8;
        }
        self.slow_huff(d, LOOKAHEAD_BITS + 1)
    }

    /// turbo's `jpeg_huff_decode`: fetch `min_bits`, then extend one bit at a time until the code
    /// matches a known length.
    pub(crate) fn slow_huff(&mut self, d: &DerivedTable, min_bits: i32) -> u8 {
        self.ensure(min_bits);
        let mut l = min_bits as usize;
        let mut code = self.get_bits(min_bits);
        while code > d.maxcode[l] {
            self.ensure(1);
            code = code << 1 | self.get_bits(1);
            l += 1;
        }
        if l > 16 {
            self.lost = true; // JWRN_HUFF_BAD_CODE; turbo's fast path skips it
            return 0; // turbo: fake a zero as the safest result
        }
        d.values[(code + d.valoffset[l]) as usize & 255]
    }
}
