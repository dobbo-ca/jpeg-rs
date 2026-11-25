//! `DecodeOptions::pyramid`: floor-sized output, exact where the chroma blur
//! does not run, a quality floor against the full decode averaged in 2x2 blocks, the
//! `Unsupported` cases, every thread count, and a reused `Decoder`.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

mod common;
use common::*;
use jpeg::{
    DecodeOptions, Decoder, EncodeImage, EncodeOptions, Error, Format, Image, Limits, Recon, Rect,
    Rgba, decode, decode_probe,
};
use std::sync::atomic::AtomicBool;

fn noop_idct(_: &mut [u8], _: usize, _: &[i32; 64]) {}

fn pyr(format: Format, scale: u8) -> DecodeOptions<'static> {
    DecodeOptions {
        pyramid: true,
        ..opts(format, scale)
    }
}

/// The sample rows of an image, tightly packed: (width, height, channels, bytes). RGBA drops alpha.
fn packed(img: &Image) -> (usize, usize, usize, Vec<u8>) {
    match img {
        Image::Gray(g) => {
            let (w, h) = (g.rect.width() as usize, g.rect.height() as usize);
            let v = g
                .pix
                .chunks(g.stride)
                .take(h)
                .flat_map(|r| &r[..w])
                .copied();
            (w, h, 1, v.collect())
        }
        Image::Rgba(g) => {
            let (w, h) = (g.rect.width() as usize, g.rect.height() as usize);
            let mut v = Vec::with_capacity(3 * w * h);
            for row in g.pix.chunks(g.stride).take(h) {
                for px in row[..4 * w].as_chunks::<4>().0 {
                    v.extend_from_slice(&px[..3]);
                }
            }
            (w, h, 3, v)
        }
        Image::YCbCr(_) => panic!("not RGBA or gray"),
    }
}

/// The top-left `cw` x `ch` samples.
fn crop((w, _, c, p): &(usize, usize, usize, Vec<u8>), cw: usize, ch: usize) -> Vec<u8> {
    (0..ch)
        .flat_map(|y| p[y * w * c..(y * w + cw) * c].iter().copied())
        .collect()
}

/// The rounded mean of 2x2 blocks, an odd last row or column dropped.
fn halve((w, h, c, p): (usize, usize, usize, Vec<u8>)) -> (usize, usize, usize, Vec<u8>) {
    let (nw, nh) = (w / 2, h / 2);
    let mut o = vec![0u8; nw * nh * c];
    for y in 0..nh {
        for x in 0..nw {
            for k in 0..c {
                let a = (2 * y * w + 2 * x) * c + k;
                let s =
                    p[a] as u32 + p[a + c] as u32 + p[a + w * c] as u32 + p[a + w * c + c] as u32;
                o[(y * nw + x) * c + k] = ((s + 2) >> 2) as u8;
            }
        }
    }
    (nw, nh, c, o)
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let se: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (x as f64 - y as f64).powi(2))
        .sum();
    if se == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (255.0 * 255.0 * a.len() as f64 / se).log10()
}

fn width_height(b: &[u8]) -> (usize, usize) {
    let c = jpeg::decode_config(b, None, &Limits::default()).unwrap();
    (c.width, c.height)
}

#[test]
fn output_is_floor_sized() {
    for name in [
        "synth_420_203x157",
        "synth_420_17x9",
        "synth_420_15x1",
        "synth_422_203x157",
        "synth_444_227x149",
        "synth_gray_203x157",
        "prog_420_111x75",
        "prog_420_15x1",
    ] {
        let b = read(&format!("jpg/{name}.jpg"));
        let (w, h) = width_height(&b);
        for format in [Format::Rgba, Format::Gray] {
            for scale in [2u8, 4, 8] {
                let s = scale as usize;
                let r = dec(&b, &pyr(format, scale));
                if w < s || h < s {
                    assert!(
                        matches!(r, Err(Error::Unsupported)),
                        "{name} {format:?} 1/{s}"
                    );
                    continue;
                }
                let img = r.unwrap().image;
                let want = Rect::new(0, 0, (w / s) as i32, (h / s) as i32);
                assert_eq!(img.bounds(), want, "{name} {format:?} 1/{s}");
                let (pw, ph, c, p) = packed(&img);
                assert_eq!((pw, ph, p.len()), (w / s, h / s, w / s * (h / s) * c));
            }
        }
    }
}

/// Where no chroma blur runs the output is the plain scaled decode, cropped to the floor size.
#[test]
fn equals_cropped_plain_decode_where_the_filter_is_off() {
    const ALL: &[u8] = &[2, 4, 8];
    const EIGHT: &[u8] = &[8];
    let cases = [
        ("synth_gray_203x157", ALL),
        ("prog_gray_111x75", ALL),
        ("synth_444_203x157", ALL),
        ("prog_444_111x75", ALL),
        ("synth_422_203x157", ALL),
        ("synth_420_203x157", EIGHT),
        ("prog_420_111x75", EIGHT),
    ];
    for (name, scales) in cases {
        let b = read(&format!("jpg/{name}.jpg"));
        let (w, h) = width_height(&b);
        for &scale in scales {
            let s = scale as usize;
            for format in [Format::Rgba, Format::Gray] {
                let plain = packed(&dec(&b, &opts(format, scale)).unwrap().image);
                let got = packed(&dec(&b, &pyr(format, scale)).unwrap().image);
                assert_eq!(got.3, crop(&plain, w / s, h / s), "{name} {format:?} 1/{s}");
            }
        }
    }
}

/// `Format::Gray` of a subsampled YCbCr stream skips the chroma path, so no blur runs: it is the
/// plain scaled decode, cropped to the floor size.
#[test]
fn gray_of_a_subsampled_stream_equals_the_plain_decode() {
    for name in ["synth_420_203x157", "prog_420_111x75", "synth_422_203x157"] {
        let b = read(&format!("jpg/{name}.jpg"));
        let (w, h) = width_height(&b);
        for scale in [2u8, 4, 8] {
            let s = scale as usize;
            let plain = packed(&dec(&b, &opts(Format::Gray, scale)).unwrap().image);
            let got = packed(&dec(&b, &pyr(Format::Gray, scale)).unwrap().image);
            assert_eq!(got.3, crop(&plain, w / s, h / s), "{name} 1/{s}");
        }
    }
}

/// The pyramid scale 1 is the plain decode.
#[test]
fn scale_one_is_unchanged() {
    let b = read("jpg/synth_420_203x157.jpg");
    let plain = dec(&b, &opts(Format::Rgba, 1)).unwrap();
    assert_eq!(dec(&b, &pyr(Format::Rgba, 1)).unwrap(), plain);
    assert_eq!(dec(&b, &pyr(Format::Rgba, 0)).unwrap(), plain);
}

/// Pseudo-random RGBA noise, encoded at Q75 4:2:0: the worst case for the chroma of a 2x2 mean.
fn noise_jpeg(side: usize) -> Vec<u8> {
    let mut s = 0x2545_f491_4f6c_dd1du64;
    let mut pix = Vec::with_capacity(4 * side * side);
    for _ in 0..side * side {
        for _ in 0..3 {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            pix.push((s >> 56) as u8);
        }
        pix.push(255);
    }
    let img = Rgba {
        pix,
        stride: 4 * side,
        rect: Rect::new(0, 0, side as i32, side as i32),
    };
    let o = EncodeOptions {
        quality: 75,
        chroma444: false,
    };
    jpeg::encode(
        EncodeImage::Rgba(&img),
        &o,
        &Limits::default(),
        &AtomicBool::new(false),
    )
    .unwrap()
}

/// The reference path is the full decode averaged in 2x2 blocks `k` times. The floors are a
/// little under what the chroma blur reaches on this input (about 48 and 41 dB), and the blur
/// must not be worse than no blur.
#[test]
fn quality_against_the_reference_path() {
    let b = noise_jpeg(256);
    let mut reference = packed(&dec(&b, &opts(Format::Rgba, 1)).unwrap().image);
    for (k, floor) in [(1u32, 44.0), (2, 38.0)] {
        reference = halve(reference);
        let scale = 1u8 << k;
        let got = packed(&dec(&b, &pyr(Format::Rgba, scale)).unwrap().image);
        let plain = packed(&dec(&b, &opts(Format::Rgba, scale)).unwrap().image);
        assert_eq!((got.0, got.1), (reference.0, reference.1));
        let (pg, pp) = (psnr(&got.3, &reference.3), psnr(&plain.3, &reference.3));
        assert!(pg >= floor, "1/{scale}: {pg:.2} dB under {floor}");
        assert!(
            pg >= pp,
            "1/{scale}: pyramid {pg:.2} dB under plain {pp:.2}"
        );
    }
}

#[test]
fn unsupported_cases() {
    let b = read("jpg/synth_420_203x157.jpg");
    let region = Some(Rect::new(8, 8, 40, 40));
    let cases = [
        DecodeOptions {
            region,
            ..pyr(Format::Rgba, 2)
        },
        DecodeOptions {
            region,
            ..pyr(Format::Gray, 4)
        },
        pyr(Format::YCbCr, 2),
        DecodeOptions {
            recon: Recon {
                idct: Some(noop_idct),
                to_rgba: None,
            },
            ..pyr(Format::Rgba, 2)
        },
    ];
    for (i, o) in cases.iter().enumerate() {
        assert!(matches!(dec(&b, o), Err(Error::Unsupported)), "case {i}");
    }
}

#[test]
fn threads_give_the_serial_result() {
    for name in ["svs_420_512x512", "synth_420_227x149", "prog_420_111x75"] {
        let b = read(&format!("jpg/{name}.jpg"));
        for scale in [2u8, 4] {
            let o = pyr(Format::Rgba, scale);
            let run = |t: usize| {
                let l = Limits {
                    max_threads: t,
                    ..Limits::default()
                };
                decode_probe(&b, &o, &l, &AtomicBool::new(false), true)
            };
            let (one, p1) = run(1);
            assert!(one.is_ok(), "{name} 1/{scale}");
            assert_eq!(
                p1.threads_peak, 1,
                "{name} 1/{scale}: max_threads 1 spawned"
            );
            for t in [2, 4, 0] {
                let (r, p) = run(t);
                assert_eq!(r, one, "{name} 1/{scale} threads {t}");
                assert!(
                    p.threads_peak
                        <= if t == 0 {
                            Limits::default().max_threads
                        } else {
                            t
                        },
                    "{name} 1/{scale} threads {t}: {} running",
                    p.threads_peak
                );
            }
            // Under a cap every thread count gives the serial answer and stays under the cap
            // (the pyramid's extra chroma row is charged); half the serial peak is AllocLimit.
            let need = p1.charged_peak;
            for max_alloc in [need + (1 << 20), need, need / 2] {
                let capped = |t: usize| {
                    let l = Limits {
                        max_threads: t,
                        max_alloc,
                        ..Limits::default()
                    };
                    decode_probe(&b, &o, &l, &AtomicBool::new(false), true)
                };
                let (want, _) = capped(1);
                for t in [1, 2, 4] {
                    let (r, p) = capped(t);
                    assert_eq!(
                        r, want,
                        "{name} 1/{scale} threads {t} max_alloc {max_alloc}"
                    );
                    assert!(p.charged_peak <= max_alloc, "{name} 1/{scale} threads {t}");
                }
                if max_alloc == need / 2 {
                    assert!(
                        matches!(want, Err(Error::AllocLimit { .. })),
                        "{name} 1/{scale}"
                    );
                } else if max_alloc > need {
                    assert!(want.is_ok(), "{name} 1/{scale}");
                }
            }
        }
    }
}

#[test]
fn reused_decoder_equals_one_shot() {
    let go = AtomicBool::new(false);
    let l = Limits::default();
    let mut d = Decoder::new();
    for name in [
        "synth_420_203x157",
        "svs_420_512x512",
        "synth_422_17x9",
        "prog_420_111x75",
    ] {
        let b = read(&format!("jpg/{name}.jpg"));
        for scale in [2u8, 4, 8, 2] {
            let o = pyr(Format::Rgba, scale);
            let want = decode(&b, &o, &l, &go);
            let got = d.decode(&b, &o, &l, &go).cloned();
            assert_eq!(got, want, "{name} 1/{scale}");
        }
    }
}
