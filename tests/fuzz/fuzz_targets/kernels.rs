#![no_main]
use libfuzzer_sys::fuzz_target;

fn ref_h2v1(dst: &mut [u8], src: &[u8]) {
    let n = src.len();
    let at = |i: isize| i32::from(src[i.clamp(0, n as isize - 1) as usize]);
    for (o, d) in dst.iter_mut().enumerate() {
        let c = (o / 2) as isize;
        *d = match o {
            0 => src[0],
            o if o == 2 * n - 1 => src[n - 1],
            o if o % 2 == 0 => ((at(c) * 3 + at(c - 1) + 1) >> 2) as u8,
            _ => ((at(c) * 3 + at(c + 1) + 2) >> 2) as u8,
        };
    }
}

fn ref_h2v2(dst: &mut [u8], cur: &[u8], other: &[u8]) {
    let n = cur.len();
    let col = |i: usize| i32::from(cur[i]) * 3 + i32::from(other[i]);
    for (o, d) in dst.iter_mut().enumerate() {
        let c = o / 2;
        *d = match o {
            0 => ((col(0) * 4 + 8) >> 4) as u8,
            o if o == 2 * n - 1 => ((col(n - 1) * 4 + 7) >> 4) as u8,
            o if o % 2 == 0 => ((col(c) * 3 + col(c - 1) + 8) >> 4) as u8,
            _ => ((col(c) * 3 + col(c + 1) + 7) >> 4) as u8,
        };
    }
}

// data = [op][n][dst length adjust][bytes]. The kernels that are not SIMD: the fancy upsample
// rows against the formula written out per sample (jpeg-go's refH2V1 and refH2V2), a bad length
// an error and not a panic; the BMI MCU decoder against the plain one.
fuzz_target!(|data: &[u8]| {
    let Some((&[op, n, adj], bytes)) = data.split_first_chunk::<3>() else {
        return;
    };
    let n = usize::from(n);
    let cyc = |i: usize| {
        if bytes.is_empty() {
            0
        } else {
            bytes[i % bytes.len()]
        }
    };
    match op % 3 {
        0 | 1 => {
            let row: Vec<u8> = (0..n).map(&cyc).collect();
            let other: Vec<u8> = (0..n + usize::from(adj % 3)).map(|i| cyc(i + 7)).collect();
            let dl = (2 * n + usize::from(adj >> 2) % 5).saturating_sub(usize::from(adj >> 5));
            let mut dst = vec![0xaau8; dl];
            let r = if op % 3 == 0 {
                jpeg::h2v1_fancy_row(&mut dst, &row, &jpeg::Limits::default())
            } else {
                jpeg::h2v2_fancy_row(&mut dst, &row, &other, &jpeg::Limits::default())
            };
            let w = dl.min(2 * n);
            if n < 2 || w + 1 < 2 * n {
                assert!(r.is_err(), "n {n} dst {dl}");
                return;
            }
            assert!(r.is_ok());
            let mut want = vec![0u8; w];
            if op % 3 == 0 {
                ref_h2v1(&mut want, &row);
            } else {
                ref_h2v2(&mut want, &row, &other);
            }
            assert!(dst[..w] == want[..], "n {n} dst {dl}");
            assert!(dst[w..].iter().all(|&b| b == 0xaa), "wrote past 2n");
        }
        _ => {
            // The BMI build of the MCU decoder against the plain one, on the bytes as entropy data.
            if let Err(e) = jpeg::kernel_check::mcu_decode(bytes) {
                panic!("{e}");
            }
        }
    }
});
