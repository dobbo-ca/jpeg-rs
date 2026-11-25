//! The fancy upsampling rows against `jdsample.c`'s formulas written out per output sample
//! (ported from jpeg-go's `fancy_test.go`).
#![allow(clippy::needless_range_loop)]

use jpeg::{Error, Limits};

fn ref_h2v1(dst: &mut [u8], inp: &[u8]) {
    let n = inp.len();
    let at = |i: i64| inp[i.clamp(0, n as i64 - 1) as usize] as i32;
    for o in 0..dst.len() {
        let c = (o / 2) as i64;
        dst[o] = if o == 0 {
            inp[0]
        } else if o == 2 * n - 1 {
            inp[n - 1]
        } else if o.is_multiple_of(2) {
            ((at(c) * 3 + at(c - 1) + 1) >> 2) as u8
        } else {
            ((at(c) * 3 + at(c + 1) + 2) >> 2) as u8
        };
    }
}

fn ref_h2v2(dst: &mut [u8], cur: &[u8], other: &[u8]) {
    let n = cur.len();
    let col = |i: usize| cur[i] as i32 * 3 + other[i] as i32;
    for o in 0..dst.len() {
        let c = o / 2;
        dst[o] = if o == 0 {
            ((col(0) * 4 + 8) >> 4) as u8
        } else if o == 2 * n - 1 {
            ((col(n - 1) * 4 + 7) >> 4) as u8
        } else if o.is_multiple_of(2) {
            ((col(c) * 3 + col(c - 1) + 8) >> 4) as u8
        } else {
            ((col(c) * 3 + col(c + 1) + 7) >> 4) as u8
        };
    }
}

fn row(rng: &mut u64, n: usize, kind: u32) -> Vec<u8> {
    (0..n)
        .map(|i| match kind {
            0 => {
                *rng ^= *rng << 13;
                *rng ^= *rng >> 7;
                *rng ^= *rng << 17;
                *rng as u8
            }
            1 => 255 * (i & 1) as u8,
            _ => 255,
        })
        .collect()
}

#[test]
fn fancy_rows_match_the_reference() {
    let mut rng = 7u64;
    for n in 2..=300usize {
        for kc in 0..3 {
            for ko in 0..3 {
                let cur = row(&mut rng, n, kc);
                let other = row(&mut rng, n + (n % 3), ko);
                for w in [2 * n, 2 * n - 1] {
                    let (mut want, mut got) = (vec![0u8; w], vec![0u8; w]);
                    ref_h2v1(&mut want, &cur);
                    jpeg::h2v1_fancy_row(&mut got, &cur, &Limits::default()).unwrap();
                    assert_eq!(got, want, "h2v1 n={n} w={w}");
                    ref_h2v2(&mut want, &cur, &other);
                    jpeg::h2v2_fancy_row(&mut got, &cur, &other, &Limits::default()).unwrap();
                    assert_eq!(got, want, "h2v2 n={n} w={w}");
                }
            }
        }
    }
}

#[test]
fn fancy_rows_refuse_bad_lengths() {
    let mut dst = [0u8; 4];
    assert!(matches!(
        jpeg::h2v1_fancy_row(&mut dst, &[1], &Limits::default()),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        jpeg::h2v1_fancy_row(&mut dst[..2], &[1, 2, 3], &Limits::default()),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        jpeg::h2v2_fancy_row(&mut dst, &[1, 2], &[1], &Limits::default()),
        Err(Error::InvalidArgument(_))
    ));
    // The row length is checked against the limits before anything is written.
    let tiny = Limits {
        max_alloc: 3,
        ..Limits::default()
    };
    assert!(matches!(
        jpeg::h2v1_fancy_row(&mut dst, &[1, 2], &tiny),
        Err(Error::AllocLimit { .. })
    ));
}
