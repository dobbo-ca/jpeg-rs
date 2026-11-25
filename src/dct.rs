//! The islow IDCT, ported from jpeg-go's `internal/codec/dct.go` (turbo `jidctint.c` and
//! `jidctred.c`). Intermediates are `i64` like turbo's `JLONG`; stores truncate like C.

pub(crate) type Block = [i16; 64];

const CONST_BITS: u32 = 13;
const PASS1_BITS: u32 = 2;

const FIX_0_298631336: i64 = 2446;
const FIX_0_390180644: i64 = 3196;
const FIX_0_541196100: i64 = 4433;
const FIX_0_765366865: i64 = 6270;
const FIX_0_899976223: i64 = 7373;
const FIX_1_175875602: i64 = 9633;
const FIX_1_501321110: i64 = 12299;
const FIX_1_847759065: i64 = 15137;
const FIX_1_961570560: i64 = 16069;
const FIX_2_053119869: i64 = 16819;
const FIX_2_562915447: i64 = 20995;
const FIX_3_072711026: i64 = 25172;

const FIX_0_211164243: i64 = 1730;
const FIX_0_509795579: i64 = 4176;
const FIX_0_601344887: i64 = 4926;
const FIX_0_720959822: i64 = 5906;
const FIX_0_850430095: i64 = 6967;
const FIX_1_061594337: i64 = 8697;
const FIX_1_272758580: i64 = 10426;
const FIX_1_451774981: i64 = 11893;
const FIX_2_172734803: i64 = 17799;
const FIX_3_624509785: i64 = 29692;

/// turbo's post-IDCT range-limit table, indexed by `x & 1023` so out-of-range values wrap the way
/// C does.
const fn build_range_limit() -> [u8; 1024] {
    let mut t = [0u8; 1024];
    let mut k = 0;
    while k < 1024 {
        t[k] = if k < 128 {
            (k + 128) as u8
        } else if k < 512 {
            255
        } else if k < 896 {
            0
        } else {
            (k - 896) as u8
        };
        k += 1;
    }
    t
}

static RANGE_LIMIT: [u8; 1024] = build_range_limit();

fn rl(x: i64) -> u8 {
    RANGE_LIMIT[(x as i32 & 1023) as usize]
}

/// `ISLOW_MULT_TYPE=int` dequantization: a 32-bit truncating multiply, widened to `i64`.
pub(crate) fn dequantize(coef: i16, quant: u16) -> i64 {
    (coef as i32).wrapping_mul(quant as i32) as i64
}

/// `DESCALE(x, n)`: round and arithmetic-shift.
fn descale(x: i64, n: u32) -> i64 {
    (x + (1 << (n - 1))) >> n
}

/// `_jpeg_idct_islow`. `coef` is natural order; `dst` gets 8x8 samples at `stride`.
pub(crate) fn idct8(coef: &Block, quant: &[u16; 64], dst: &mut [u8], stride: usize) {
    let mut ws = [0i32; 64];

    // Pass 1: columns.
    for col in 0..8 {
        if coef[8 + col] == 0
            && coef[16 + col] == 0
            && coef[24 + col] == 0
            && coef[32 + col] == 0
            && coef[40 + col] == 0
            && coef[48 + col] == 0
            && coef[56 + col] == 0
        {
            let dc = (dequantize(coef[col], quant[col]) << PASS1_BITS) as i32;
            for r in 0..8 {
                ws[r * 8 + col] = dc;
            }
            continue;
        }

        let mut z2 = dequantize(coef[16 + col], quant[16 + col]);
        let mut z3 = dequantize(coef[48 + col], quant[48 + col]);

        let mut z1 = (z2 + z3) * FIX_0_541196100;
        let mut tmp2 = z1 + z3 * (-FIX_1_847759065);
        let mut tmp3 = z1 + z2 * FIX_0_765366865;

        z2 = dequantize(coef[col], quant[col]);
        z3 = dequantize(coef[32 + col], quant[32 + col]);

        let mut tmp0 = (z2 + z3) << CONST_BITS;
        let mut tmp1 = (z2 - z3) << CONST_BITS;

        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;

        tmp0 = dequantize(coef[56 + col], quant[56 + col]);
        tmp1 = dequantize(coef[40 + col], quant[40 + col]);
        tmp2 = dequantize(coef[24 + col], quant[24 + col]);
        tmp3 = dequantize(coef[8 + col], quant[8 + col]);

        z1 = tmp0 + tmp3;
        z2 = tmp1 + tmp2;
        z3 = tmp0 + tmp2;
        let mut z4 = tmp1 + tmp3;
        let z5 = (z3 + z4) * FIX_1_175875602;

        tmp0 *= FIX_0_298631336;
        tmp1 *= FIX_2_053119869;
        tmp2 *= FIX_3_072711026;
        tmp3 *= FIX_1_501321110;
        z1 *= -FIX_0_899976223;
        z2 *= -FIX_2_562915447;
        z3 *= -FIX_1_961570560;
        z4 *= -FIX_0_390180644;

        z3 += z5;
        z4 += z5;

        tmp0 += z1 + z3;
        tmp1 += z2 + z4;
        tmp2 += z2 + z3;
        tmp3 += z1 + z4;

        const S: u32 = CONST_BITS - PASS1_BITS;
        ws[col] = descale(tmp10 + tmp3, S) as i32;
        ws[56 + col] = descale(tmp10 - tmp3, S) as i32;
        ws[8 + col] = descale(tmp11 + tmp2, S) as i32;
        ws[48 + col] = descale(tmp11 - tmp2, S) as i32;
        ws[16 + col] = descale(tmp12 + tmp1, S) as i32;
        ws[40 + col] = descale(tmp12 - tmp1, S) as i32;
        ws[24 + col] = descale(tmp13 + tmp0, S) as i32;
        ws[32 + col] = descale(tmp13 - tmp0, S) as i32;
    }

    // Pass 2: rows.
    for row in 0..8 {
        let base = 8 * row;
        let out = &mut dst[row * stride..row * stride + 8];

        if ws[base + 1] == 0
            && ws[base + 2] == 0
            && ws[base + 3] == 0
            && ws[base + 4] == 0
            && ws[base + 5] == 0
            && ws[base + 6] == 0
            && ws[base + 7] == 0
        {
            let dc = rl(descale(ws[base] as i64, PASS1_BITS + 3));
            out.fill(dc);
            continue;
        }

        let mut z2 = ws[base + 2] as i64;
        let mut z3 = ws[base + 6] as i64;

        let mut z1 = (z2 + z3) * FIX_0_541196100;
        let mut tmp2 = z1 + z3 * (-FIX_1_847759065);
        let mut tmp3 = z1 + z2 * FIX_0_765366865;

        let mut tmp0 = (ws[base] as i64 + ws[base + 4] as i64) << CONST_BITS;
        let mut tmp1 = (ws[base] as i64 - ws[base + 4] as i64) << CONST_BITS;

        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;

        tmp0 = ws[base + 7] as i64;
        tmp1 = ws[base + 5] as i64;
        tmp2 = ws[base + 3] as i64;
        tmp3 = ws[base + 1] as i64;

        z1 = tmp0 + tmp3;
        z2 = tmp1 + tmp2;
        z3 = tmp0 + tmp2;
        let mut z4 = tmp1 + tmp3;
        let z5 = (z3 + z4) * FIX_1_175875602;

        tmp0 *= FIX_0_298631336;
        tmp1 *= FIX_2_053119869;
        tmp2 *= FIX_3_072711026;
        tmp3 *= FIX_1_501321110;
        z1 *= -FIX_0_899976223;
        z2 *= -FIX_2_562915447;
        z3 *= -FIX_1_961570560;
        z4 *= -FIX_0_390180644;

        z3 += z5;
        z4 += z5;

        tmp0 += z1 + z3;
        tmp1 += z2 + z4;
        tmp2 += z2 + z3;
        tmp3 += z1 + z4;

        const SHIFT: u32 = CONST_BITS + PASS1_BITS + 3;
        out[0] = rl(descale(tmp10 + tmp3, SHIFT));
        out[7] = rl(descale(tmp10 - tmp3, SHIFT));
        out[1] = rl(descale(tmp11 + tmp2, SHIFT));
        out[6] = rl(descale(tmp11 - tmp2, SHIFT));
        out[2] = rl(descale(tmp12 + tmp1, SHIFT));
        out[5] = rl(descale(tmp12 - tmp1, SHIFT));
        out[3] = rl(descale(tmp13 + tmp0, SHIFT));
        out[4] = rl(descale(tmp13 - tmp0, SHIFT));
    }
}

/// `_jpeg_idct_4x4`. Column 4 is unused by the reduced output and is skipped.
pub(crate) fn idct4(coef: &Block, quant: &[u16; 64], dst: &mut [u8], stride: usize) {
    let mut ws = [0i32; 64];

    for col in 0..8 {
        if col == 4 {
            continue;
        }
        if coef[8 + col] == 0
            && coef[16 + col] == 0
            && coef[24 + col] == 0
            && coef[40 + col] == 0
            && coef[48 + col] == 0
            && coef[56 + col] == 0
        {
            let dc = (dequantize(coef[col], quant[col]) << PASS1_BITS) as i32;
            ws[col] = dc;
            ws[8 + col] = dc;
            ws[16 + col] = dc;
            ws[24 + col] = dc;
            continue;
        }

        let tmp0 = dequantize(coef[col], quant[col]) << (CONST_BITS + 1);

        let mut z2 = dequantize(coef[16 + col], quant[16 + col]);
        let mut z3 = dequantize(coef[48 + col], quant[48 + col]);
        let tmp2 = z2 * FIX_1_847759065 + z3 * (-FIX_0_765366865);

        let tmp10 = tmp0 + tmp2;
        let tmp12 = tmp0 - tmp2;

        let z1 = dequantize(coef[56 + col], quant[56 + col]);
        z2 = dequantize(coef[40 + col], quant[40 + col]);
        z3 = dequantize(coef[24 + col], quant[24 + col]);
        let z4 = dequantize(coef[8 + col], quant[8 + col]);

        let odd_a = z1 * (-FIX_0_211164243)
            + z2 * FIX_1_451774981
            + z3 * (-FIX_2_172734803)
            + z4 * FIX_1_061594337;
        let odd_b = z1 * (-FIX_0_509795579)
            + z2 * (-FIX_0_601344887)
            + z3 * FIX_0_899976223
            + z4 * FIX_2_562915447;

        const S: u32 = CONST_BITS - PASS1_BITS + 1;
        ws[col] = descale(tmp10 + odd_b, S) as i32;
        ws[24 + col] = descale(tmp10 - odd_b, S) as i32;
        ws[8 + col] = descale(tmp12 + odd_a, S) as i32;
        ws[16 + col] = descale(tmp12 - odd_a, S) as i32;
    }

    for row in 0..4 {
        let base = 8 * row;
        let out = &mut dst[row * stride..row * stride + 4];

        if ws[base + 1] == 0
            && ws[base + 2] == 0
            && ws[base + 3] == 0
            && ws[base + 5] == 0
            && ws[base + 6] == 0
            && ws[base + 7] == 0
        {
            let dc = rl(descale(ws[base] as i64, PASS1_BITS + 3));
            out.fill(dc);
            continue;
        }

        let tmp0 = (ws[base] as i64) << (CONST_BITS + 1);
        let tmp2 = ws[base + 2] as i64 * FIX_1_847759065 + ws[base + 6] as i64 * (-FIX_0_765366865);
        let tmp10 = tmp0 + tmp2;
        let tmp12 = tmp0 - tmp2;

        let z1 = ws[base + 7] as i64;
        let z2 = ws[base + 5] as i64;
        let z3 = ws[base + 3] as i64;
        let z4 = ws[base + 1] as i64;

        let odd_a = z1 * (-FIX_0_211164243)
            + z2 * FIX_1_451774981
            + z3 * (-FIX_2_172734803)
            + z4 * FIX_1_061594337;
        let odd_b = z1 * (-FIX_0_509795579)
            + z2 * (-FIX_0_601344887)
            + z3 * FIX_0_899976223
            + z4 * FIX_2_562915447;

        const SHIFT: u32 = CONST_BITS + PASS1_BITS + 3 + 1;
        out[0] = rl(descale(tmp10 + odd_b, SHIFT));
        out[3] = rl(descale(tmp10 - odd_b, SHIFT));
        out[1] = rl(descale(tmp12 + odd_a, SHIFT));
        out[2] = rl(descale(tmp12 - odd_a, SHIFT));
    }
}

/// `_jpeg_idct_2x2`. Only rows and columns 0, 1, 3, 5, 7 are used.
pub(crate) fn idct2(coef: &Block, quant: &[u16; 64], dst: &mut [u8], stride: usize) {
    let mut ws = [0i32; 16];

    for col in 0..8 {
        if col == 2 || col == 4 || col == 6 {
            continue;
        }
        if coef[8 + col] == 0 && coef[24 + col] == 0 && coef[40 + col] == 0 && coef[56 + col] == 0 {
            let dc = (dequantize(coef[col], quant[col]) << PASS1_BITS) as i32;
            ws[col] = dc;
            ws[8 + col] = dc;
            continue;
        }

        let tmp10 = dequantize(coef[col], quant[col]) << (CONST_BITS + 2);

        let mut tmp0 = dequantize(coef[56 + col], quant[56 + col]) * (-FIX_0_720959822);
        tmp0 += dequantize(coef[40 + col], quant[40 + col]) * FIX_0_850430095;
        tmp0 += dequantize(coef[24 + col], quant[24 + col]) * (-FIX_1_272758580);
        tmp0 += dequantize(coef[8 + col], quant[8 + col]) * FIX_3_624509785;

        const S: u32 = CONST_BITS - PASS1_BITS + 2;
        ws[col] = descale(tmp10 + tmp0, S) as i32;
        ws[8 + col] = descale(tmp10 - tmp0, S) as i32;
    }

    for row in 0..2 {
        let base = 8 * row;
        let out = &mut dst[row * stride..row * stride + 2];

        if ws[base + 1] == 0 && ws[base + 3] == 0 && ws[base + 5] == 0 && ws[base + 7] == 0 {
            let dc = rl(descale(ws[base] as i64, PASS1_BITS + 3));
            out.fill(dc);
            continue;
        }

        let tmp10 = (ws[base] as i64) << (CONST_BITS + 2);
        let tmp0 = ws[base + 7] as i64 * (-FIX_0_720959822)
            + ws[base + 5] as i64 * FIX_0_850430095
            + ws[base + 3] as i64 * (-FIX_1_272758580)
            + ws[base + 1] as i64 * FIX_3_624509785;

        const SHIFT: u32 = CONST_BITS + PASS1_BITS + 3 + 2;
        out[0] = rl(descale(tmp10 + tmp0, SHIFT));
        out[1] = rl(descale(tmp10 - tmp0, SHIFT));
    }
}

/// `_jpeg_idct_1x1`: the DC average.
pub(crate) fn idct1(coef: &Block, quant: &[u16; 64], dst: &mut [u8]) {
    let dc = descale(dequantize(coef[0], quant[0]), 3);
    dst[0] = rl(dc);
}

/// Whether `coef` has no AC coefficients.
#[inline]
pub(crate) fn dc_only(coef: &Block) -> bool {
    // The two lowest frequencies (natural order 1 and 8) are the likeliest AC, so a block with
    // AC mostly leaves here. The fold has no early exit, so it vectorizes.
    (coef[1] | coef[8]) == 0 && coef[1..].iter().fold(0i16, |a, &c| a | c) == 0
}

/// The 8/4/2 IDCTs' output for a DC-only block.
fn idct_dc_const(dc: i16, quant: u16) -> u8 {
    let v = (dequantize(dc, quant) << PASS1_BITS) as i32; // pass 1 stores int32
    rl(descale(v as i64, PASS1_BITS + 3))
}

/// A `bs` x `bs` block of `v` at `stride`.
fn fill_const(bs: usize, dst: &mut [u8], stride: usize, v: u8) {
    if bs == 8 {
        // One 8-byte store per row, as Go's `fillConst`.
        for r in 0..8 {
            dst[r * stride..r * stride + 8].copy_from_slice(&[v; 8]);
        }
        return;
    }
    for r in 0..bs {
        dst[r * stride..r * stride + bs].fill(v);
    }
}

/// The islow IDCT matching a component's scaled block size (Scale 2/4/8 selects the reduced
/// IDCTs). `may_ac` false means the Huffman decoder stored no AC in `coef`, so the DC-only scan is
/// skipped.
pub(crate) fn idct_bs(
    bs: usize,
    coef: &Block,
    may_ac: bool,
    quant: &[u16; 64],
    dst: &mut [u8],
    stride: usize,
) {
    if bs > 1 && (!may_ac || dc_only(coef)) {
        // idct1 is already DC only
        fill_const(bs, dst, stride, idct_dc_const(coef[0], quant[0]));
        return;
    }
    match bs {
        4 => crate::simd::idct4(coef, quant, dst, stride),
        2 => idct2(coef, quant, dst, stride),
        1 => idct1(coef, quant, dst),
        _ => crate::simd::idct8(coef, quant, dst, stride),
    }
}
