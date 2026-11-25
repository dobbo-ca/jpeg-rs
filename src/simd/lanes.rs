//! The islow IDCT and FDCT butterflies, written once over an abstract 8-lane `i32` vector. Each
//! SIMD module defines `V` and the helpers `add`, `sub`, `mulk` (wrapping multiply by a constant),
//! `shl::<N>`, `sra::<N>`, `and` and `splat`, then expands these macros. The arithmetic is the scalar
//! code's (`dct::idct8`, `encoder::fdct_islow`) lane by lane, in wrapping `i32`.

/// The 8 sums of one 1-D islow IDCT over `v[0..8]` (the 8 vectors are the 8 inputs of the 1-D
/// transform, each lane an independent transform), before rounding and the shift.
#[allow(unused_macros)]
macro_rules! idct_1d_sums {
    ($v:expr) => {{
        let v = $v;
        let z2 = v[2];
        let z3 = v[6];
        let z1 = mulk(add(z2, z3), 4433);
        let tmp2 = add(z1, mulk(z3, -15137));
        let tmp3 = add(z1, mulk(z2, 6270));
        let tmp0 = shl::<13>(add(v[0], v[4]));
        let tmp1 = shl::<13>(sub(v[0], v[4]));
        let tmp10 = add(tmp0, tmp3);
        let tmp13 = sub(tmp0, tmp3);
        let tmp11 = add(tmp1, tmp2);
        let tmp12 = sub(tmp1, tmp2);

        let (t0, t1, t2, t3) = (v[7], v[5], v[3], v[1]);
        let z1 = add(t0, t3);
        let z2 = add(t1, t2);
        let z3 = add(t0, t2);
        let z4 = add(t1, t3);
        let z5 = mulk(add(z3, z4), 9633);
        let t0 = mulk(t0, 2446);
        let t1 = mulk(t1, 16819);
        let t2 = mulk(t2, 25172);
        let t3 = mulk(t3, 12299);
        let z1 = mulk(z1, -7373);
        let z2 = mulk(z2, -20995);
        let z3 = add(mulk(z3, -16069), z5);
        let z4 = add(mulk(z4, -3196), z5);
        let t0 = add(t0, add(z1, z3));
        let t1 = add(t1, add(z2, z4));
        let t2 = add(t2, add(z2, z3));
        let t3 = add(t3, add(z1, z4));

        [
            add(tmp10, t3),
            add(tmp11, t2),
            add(tmp12, t1),
            add(tmp13, t0),
            sub(tmp13, t0),
            sub(tmp12, t1),
            sub(tmp11, t2),
            sub(tmp10, t3),
        ]
    }};
}

/// One 1-D islow IDCT over `v[0..8]`. Descales by `$shift` with rounding.
#[allow(unused_macros)]
macro_rules! idct_1d {
    ($v:expr, $shift:literal) => {{
        let s = idct_1d_sums!($v);
        let rnd = splat(1 << ($shift - 1));
        [
            sra::<$shift>(add(s[0], rnd)),
            sra::<$shift>(add(s[1], rnd)),
            sra::<$shift>(add(s[2], rnd)),
            sra::<$shift>(add(s[3], rnd)),
            sra::<$shift>(add(s[4], rnd)),
            sra::<$shift>(add(s[5], rnd)),
            sra::<$shift>(add(s[6], rnd)),
            sra::<$shift>(add(s[7], rnd)),
        ]
    }};
}

/// `idct_1d` for inputs whose weighted sums can pass 2^31 in `i32` (`$shift` is the same). Each
/// input is split into `h = v >> $shift` and `l = v & (2^$shift - 1)`; the transform is linear,
/// so `(S(v) + rnd) >> $shift = S(h) + ((S(l) + rnd) >> $shift)` exactly, where `S(h)` is taken
/// mod 2^32 (as the scalar code's `as i32` truncation does) and `S(l)` is under 2^31 for every
/// `i32` input.
#[allow(unused_macros)]
macro_rules! idct_1d_split {
    ($v:expr, $shift:literal) => {{
        let v = $v;
        let m = splat((1 << $shift) - 1);
        let h = [
            sra::<$shift>(v[0]),
            sra::<$shift>(v[1]),
            sra::<$shift>(v[2]),
            sra::<$shift>(v[3]),
            sra::<$shift>(v[4]),
            sra::<$shift>(v[5]),
            sra::<$shift>(v[6]),
            sra::<$shift>(v[7]),
        ];
        let l = [
            and(v[0], m),
            and(v[1], m),
            and(v[2], m),
            and(v[3], m),
            and(v[4], m),
            and(v[5], m),
            and(v[6], m),
            and(v[7], m),
        ];
        let hs = idct_1d_sums!(h);
        let ls = idct_1d!(l, $shift);
        [
            add(hs[0], ls[0]),
            add(hs[1], ls[1]),
            add(hs[2], ls[2]),
            add(hs[3], ls[3]),
            add(hs[4], ls[4]),
            add(hs[5], ls[5]),
            add(hs[6], ls[6]),
            add(hs[7], ls[7]),
        ]
    }};
}

/// The 4 sums of one 1-D reduced 4x4 IDCT (`dct::idct4`) over `v[0..8]`, before rounding and the
/// shift. It reads `v[0..=3]` and `v[5..=7]`; `v[4]` is not an input.
#[allow(unused_macros)]
macro_rules! idct4_1d_sums {
    ($v:expr) => {{
        let v = $v;
        let tmp0 = shl::<14>(v[0]);
        let tmp2 = add(mulk(v[2], 15137), mulk(v[6], -6270));
        let tmp10 = add(tmp0, tmp2);
        let tmp12 = sub(tmp0, tmp2);
        let (z1, z2, z3, z4) = (v[7], v[5], v[3], v[1]);
        let odd_a = add(
            add(mulk(z1, -1730), mulk(z2, 11893)),
            add(mulk(z3, -17799), mulk(z4, 8697)),
        );
        let odd_b = add(
            add(mulk(z1, -4176), mulk(z2, -4926)),
            add(mulk(z3, 7373), mulk(z4, 20995)),
        );
        [
            add(tmp10, odd_b),
            add(tmp12, odd_a),
            sub(tmp12, odd_a),
            sub(tmp10, odd_b),
        ]
    }};
}

/// One 1-D reduced 4x4 IDCT, descaled by `$shift` with rounding: `idct_1d` for `dct::idct4`.
#[allow(unused_macros)]
macro_rules! idct4_1d {
    ($v:expr, $shift:literal) => {{
        let s = idct4_1d_sums!($v);
        let rnd = splat(1 << ($shift - 1));
        [
            sra::<$shift>(add(s[0], rnd)),
            sra::<$shift>(add(s[1], rnd)),
            sra::<$shift>(add(s[2], rnd)),
            sra::<$shift>(add(s[3], rnd)),
        ]
    }};
}

/// `idct_1d_split` for `idct4_1d`.
#[allow(unused_macros)]
macro_rules! idct4_1d_split {
    ($v:expr, $shift:literal) => {{
        let v = $v;
        let m = splat((1 << $shift) - 1);
        let h = [
            sra::<$shift>(v[0]),
            sra::<$shift>(v[1]),
            sra::<$shift>(v[2]),
            sra::<$shift>(v[3]),
            sra::<$shift>(v[4]),
            sra::<$shift>(v[5]),
            sra::<$shift>(v[6]),
            sra::<$shift>(v[7]),
        ];
        let l = [
            and(v[0], m),
            and(v[1], m),
            and(v[2], m),
            and(v[3], m),
            and(v[4], m),
            and(v[5], m),
            and(v[6], m),
            and(v[7], m),
        ];
        let hs = idct4_1d_sums!(h);
        let ls = idct4_1d!(l, $shift);
        [
            add(hs[0], ls[0]),
            add(hs[1], ls[1]),
            add(hs[2], ls[2]),
            add(hs[3], ls[3]),
        ]
    }};
}

/// One 1-D islow FDCT over `s[0..8]`. The row pass has `$shift` 11 and `$dc` = `dc_rows` (DC
/// scaled up by 4), the column pass `$shift` 15 and `$dc` = `dc_cols` (DC descaled by 4); each
/// SIMD module defines both.
#[allow(unused_macros)]
macro_rules! fdct_1d {
    ($s:expr, $shift:literal, $dc:ident) => {{
        let s = $s;
        let tmp0 = add(s[0], s[7]);
        let tmp7 = sub(s[0], s[7]);
        let tmp1 = add(s[1], s[6]);
        let tmp6 = sub(s[1], s[6]);
        let tmp2 = add(s[2], s[5]);
        let tmp5 = sub(s[2], s[5]);
        let tmp3 = add(s[3], s[4]);
        let tmp4 = sub(s[3], s[4]);

        let tmp10 = add(tmp0, tmp3);
        let tmp13 = sub(tmp0, tmp3);
        let tmp11 = add(tmp1, tmp2);
        let tmp12 = sub(tmp1, tmp2);

        let rnd = splat(1 << ($shift - 1));
        let (d0, d4) = $dc(tmp10, tmp11);
        let z1 = mulk(add(tmp12, tmp13), 4433);
        let d2 = sra::<$shift>(add(add(z1, mulk(tmp13, 6270)), rnd));
        let d6 = sra::<$shift>(add(sub(z1, mulk(tmp12, 15137)), rnd));

        let z1o = mulk(add(tmp4, tmp7), -7373);
        let z2o = mulk(add(tmp5, tmp6), -20995);
        let z3o = add(tmp4, tmp6);
        let z4o = add(tmp5, tmp7);
        let z5o = mulk(add(z3o, z4o), 9633);
        let z3o = add(mulk(z3o, -16069), z5o);
        let z4o = add(mulk(z4o, -3196), z5o);
        let t4 = mulk(tmp4, 2446);
        let t5 = mulk(tmp5, 16819);
        let t6 = mulk(tmp6, 25172);
        let t7 = mulk(tmp7, 12299);
        let d7 = sra::<$shift>(add(add(add(t4, z1o), z3o), rnd));
        let d5 = sra::<$shift>(add(add(add(t5, z2o), z4o), rnd));
        let d3 = sra::<$shift>(add(add(add(t6, z2o), z3o), rnd));
        let d1 = sra::<$shift>(add(add(add(t7, z1o), z4o), rnd));
        [d0, d1, d2, d3, d4, d5, d6, d7]
    }};
}
