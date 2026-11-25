//! Encode against cjpeg and `tj3Compress8`: the md5 of the bytes at q 30/50/75/90/100 for 4:2:0,
//! 4:4:4 and gray, and for YCbCr input (ported from jpeg-go's `encoder_test.go`).
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

mod common;
use common::*;
use jpeg::{
    EncodeImage, EncodeOptions, Error, Format, Gray, Image, Limits, Rect, Rgba, Subsample, YCbCr,
};
use std::sync::atomic::AtomicBool;

const QUALS: [i32; 5] = [30, 50, 75, 90, 100];

fn enc(m: EncodeImage, o: &EncodeOptions) -> Vec<u8> {
    jpeg::encode(m, o, &Limits::default(), &AtomicBool::new(false)).unwrap()
}

/// Splits a binary P5/P6 header: width, height, and the offset of the samples.
fn pnm_header(b: &[u8]) -> (usize, usize, usize) {
    let mut fields = Vec::new();
    let mut i = 2;
    while fields.len() < 3 {
        while b[i].is_ascii_whitespace() {
            i += 1;
        }
        let st = i;
        while !b[i].is_ascii_whitespace() {
            i += 1;
        }
        fields.push(
            std::str::from_utf8(&b[st..i])
                .unwrap()
                .parse::<usize>()
                .unwrap(),
        );
    }
    (fields[0], fields[1], i + 1)
}

fn read_ppm(name: &str) -> Rgba {
    let b = read(&format!("ppm/{name}"));
    assert_eq!(&b[..2], b"P6");
    let (w, h, off) = pnm_header(&b);
    let mut pix = Vec::with_capacity(w * h * 4);
    for p in b[off..off + w * h * 3].chunks(3) {
        pix.extend_from_slice(&[p[0], p[1], p[2], 255]);
    }
    Rgba {
        pix,
        stride: w * 4,
        rect: Rect::new(0, 0, w as i32, h as i32),
    }
}

fn read_pgm(name: &str) -> Gray {
    let b = read(&format!("ppm/{name}"));
    assert_eq!(&b[..2], b"P5");
    let (w, h, off) = pnm_header(&b);
    Gray {
        pix: b[off..off + w * h].to_vec(),
        stride: w,
        rect: Rect::new(0, 0, w as i32, h as i32),
    }
}

fn rgb_to_ycbcr(r: i32, g: i32, b: i32) -> (u8, u8, u8) {
    let y = (19595 * r + 38470 * g + 7471 * b + 32768) >> 16;
    let cb = (-11059 * r - 21709 * g + 32768 * b + (128 << 16) + 32767) >> 16;
    let cr = (32768 * r - 27439 * g - 5329 * b + (128 << 16) + 32767) >> 16;
    (y as u8, cb as u8, cr as u8)
}

fn full_planes(img: &Rgba) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (w, h) = (img.rect.width() as usize, img.rect.height() as usize);
    let (mut y, mut cb, mut cr) = (vec![0; w * h], vec![0; w * h], vec![0; w * h]);
    for i in 0..w * h {
        let p = &img.pix[4 * i..4 * i + 4];
        (y[i], cb[i], cr[i]) = rgb_to_ycbcr(p[0] as i32, p[1] as i32, p[2] as i32);
    }
    (y, cb, cr)
}

fn gray_from_ppm(img: &Rgba) -> Gray {
    let (y, _, _) = full_planes(img);
    Gray {
        pix: y,
        stride: img.rect.width() as usize,
        rect: img.rect,
    }
}

/// A standard-stride YCbCr (`image.NewYCbCr`'s layout) from a PPM, through our own RGB to YCbCr
/// conversion and h2v2 downsample.
fn ycbcr_from_ppm(img: &Rgba, ratio: Subsample) -> YCbCr {
    let (w, h) = (img.rect.width() as usize, img.rect.height() as usize);
    let (y, cb, cr) = full_planes(img);
    if ratio == Subsample::S444 {
        return YCbCr {
            y,
            cb,
            cr,
            y_stride: w,
            c_stride: w,
            ratio,
            rect: img.rect,
        };
    }
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let (mut dcb, mut dcr) = (vec![0u8; cw * ch], vec![0u8; cw * ch]);
    for oy in 0..ch {
        for ox in 0..cw {
            let bias = if ox.is_multiple_of(2) { 1 } else { 2 };
            let at = |p: &Vec<u8>, dy: usize, dx: usize| {
                p[(2 * oy + dy).min(h - 1) * w + (2 * ox + dx).min(w - 1)] as i32
            };
            for (src, dst) in [(&cb, &mut dcb), (&cr, &mut dcr)] {
                let s = at(src, 0, 0) + at(src, 0, 1) + at(src, 1, 0) + at(src, 1, 1);
                dst[oy * cw + ox] = ((s + bias) >> 2) as u8;
            }
        }
    }
    YCbCr {
        y,
        cb: dcb,
        cr: dcr,
        y_stride: w,
        c_stride: cw,
        ratio,
        rect: img.rect,
    }
}

#[test]
fn golden_encode() {
    let g = golden();
    for name in g.get("encode").keys() {
        let img = read_ppm(name);
        for subsamp in ["420", "444"] {
            for q in QUALS {
                let want = g
                    .get("encode")
                    .get(name)
                    .get(subsamp)
                    .get(&format!("q{q}"))
                    .str();
                let o = EncodeOptions {
                    quality: q,
                    chroma444: subsamp == "444",
                };
                let got = md5(&enc(EncodeImage::Rgba(&img), &o));
                assert_eq!(got, want, "{name}/{subsamp}/q{q}");
            }
        }
    }
}

#[test]
fn golden_encode_gray() {
    let g = golden();
    for name in g.get("encode").keys() {
        assert!(g.get("encode_gray").has(name), "encode_gray lacks {name}");
    }
    for name in g.get("encode_gray").keys() {
        let img = if name.ends_with(".pgm") {
            read_pgm(name)
        } else {
            gray_from_ppm(&read_ppm(name))
        };
        for q in QUALS {
            let want = g.get("encode_gray").get(name).get(&format!("q{q}")).str();
            let o = EncodeOptions {
                quality: q,
                chroma444: false,
            };
            assert_eq!(md5(&enc(EncodeImage::Gray(&img), &o)), want, "{name}/q{q}");
        }
    }
}

#[test]
fn golden_encode_yuv() {
    let g = golden();
    for name in g.get("encode_yuv").keys() {
        let img = read_ppm(name);
        for (subsamp, ratio) in [("420", Subsample::S420), ("444", Subsample::S444)] {
            let ycc = ycbcr_from_ppm(&img, ratio);
            for q in QUALS {
                let want = g
                    .get("encode_yuv")
                    .get(name)
                    .get(subsamp)
                    .get(&format!("q{q}"))
                    .str();
                let o = EncodeOptions {
                    quality: q,
                    chroma444: false,
                };
                assert_eq!(
                    md5(&enc(EncodeImage::YCbCr(&ycc), &o)),
                    want,
                    "{name}/{subsamp}/q{q}"
                );
            }
        }
    }
}

#[test]
fn encode_then_decode_round_trip() {
    let img = read_ppm("tissue_512x512.ppm");
    for c444 in [false, true] {
        let o = EncodeOptions {
            quality: 90,
            chroma444: c444,
        };
        let bytes = enc(EncodeImage::Rgba(&img), &o);
        let d = dec(&bytes, &opts(Format::Rgba, 1)).unwrap();
        assert_eq!(d.error, None);
        let Image::Rgba(out) = d.image else {
            panic!("not RGBA")
        };
        assert_eq!(out.rect, img.rect);
        let err: u64 = out
            .pix
            .iter()
            .zip(&img.pix)
            .map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs() as u64)
            .sum();
        assert!(err / (out.pix.len() as u64) < 8, "mean abs error {err}");
    }
}

/// A YCbCr whose planes do not cover its rect, or an inverted rect, is `BadImage` at every ratio,
/// never a panic (the non-direct ratios go through `draw_rgba`).
#[test]
fn encode_refuses_bad_geometry() {
    let enc_err = |m: EncodeImage| {
        jpeg::encode(
            m,
            &EncodeOptions::default(),
            &Limits::default(),
            &AtomicBool::new(false),
        )
    };
    for ratio in [
        Subsample::S444,
        Subsample::S422,
        Subsample::S420,
        Subsample::S440,
        Subsample::S411,
        Subsample::S410,
    ] {
        let short = |y: usize, c: usize, ys: usize, cs: usize, rect: Rect| YCbCr {
            y: vec![0; y],
            cb: vec![0; c],
            cr: vec![0; c],
            y_stride: ys,
            c_stride: cs,
            ratio,
            rect,
        };
        let r16 = Rect::new(0, 0, 16, 16);
        for m in [
            short(1, 1, 1, 1, r16),
            short(0, 0, 16, 8, r16),
            short(256, 0, 16, 8, r16),
            short(256, 256, 16, 1, r16),
            short(256, 256, 1, 16, r16),
            short(256, 256, 16, 16, Rect::new(0, 0, -1, -1)),
            short(256, 256, 16, 16, Rect::new(3, 5, 1, 9)),
        ] {
            assert_eq!(
                enc_err(EncodeImage::YCbCr(&m)),
                Err(Error::BadImage),
                "{ratio:?}"
            );
        }
    }
    let gray = Gray {
        pix: vec![0; 16],
        stride: 16,
        rect: Rect::new(0, 0, -1, -1),
    };
    assert_eq!(enc_err(EncodeImage::Gray(&gray)), Err(Error::BadImage));
    let rgba = Rgba {
        pix: vec![0; 16],
        stride: 16,
        rect: Rect::new(5, 0, 0, 1),
    };
    assert_eq!(enc_err(EncodeImage::Rgba(&rgba)), Err(Error::BadImage));
    // A side the SOF cannot hold is refused, not written truncated.
    let wide = Gray {
        pix: vec![0; 70000 * 8],
        stride: 70000,
        rect: Rect::new(0, 0, 70000, 8),
    };
    assert_eq!(enc_err(EncodeImage::Gray(&wide)), Err(Error::TooBig));
}

/// A cancel flag already set stops a non-direct YCbCr encode before the whole-image copy ends.
#[test]
fn encode_draw_rgba_polls_cancel() {
    let m = YCbCr {
        y: vec![0; 64 * 64],
        cb: vec![0; 64 * 64],
        cr: vec![0; 64 * 64],
        y_stride: 64,
        c_stride: 32,
        ratio: Subsample::S422,
        rect: Rect::new(0, 0, 64, 64),
    };
    let r = jpeg::encode(
        EncodeImage::YCbCr(&m),
        &EncodeOptions::default(),
        &Limits::default(),
        &AtomicBool::new(true),
    );
    assert_eq!(r, Err(Error::Cancelled));
}

/// A zero-width or zero-height frame, as Go, writes a header-only stream and never reads a row:
/// it must not panic on the empty or short pixel slices that go with such a rect.
#[test]
fn encode_empty_frame_is_header_only() {
    let o = |chroma444| EncodeOptions {
        chroma444,
        ..Default::default()
    };
    let ok = |m: EncodeImage, chroma444| {
        let b = jpeg::encode(
            m,
            &o(chroma444),
            &Limits::default(),
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(&b[..2], &[0xff, 0xd8]);
        assert_eq!(&b[b.len() - 2..], &[0xff, 0xd9]);
    };
    for r in [
        Rect::new(0, 0, 0, 2),
        Rect::new(0, 0, 3, 0),
        Rect::new(0, 0, 0, 0),
    ] {
        for c444 in [true, false] {
            let rgba = Rgba {
                pix: vec![],
                stride: 1,
                rect: r,
            };
            ok(EncodeImage::Rgba(&rgba), c444);
            let gray = Gray {
                pix: vec![],
                stride: 1,
                rect: r,
            };
            ok(EncodeImage::Gray(&gray), c444);
        }
        for ratio in [Subsample::S444, Subsample::S420, Subsample::S422] {
            let m = YCbCr {
                y: vec![],
                cb: vec![],
                cr: vec![],
                y_stride: 1,
                c_stride: 1,
                ratio,
                rect: r,
            };
            ok(EncodeImage::YCbCr(&m), false);
        }
    }
}

/// A deterministic non-flat gray image of `w` x `h`, with a flat 8x8 block in the middle.
fn noisy_gray(w: usize, h: usize) -> Gray {
    let mut pix: Vec<u8> = (0..w * h)
        .map(|i| (i.wrapping_mul(2654435761) >> 11) as u8)
        .collect();
    for y in 8..16.min(h) {
        for x in 8..16.min(w) {
            pix[y * w + x] = 77;
        }
    }
    Gray {
        pix,
        stride: w,
        rect: Rect::new(0, 0, w as i32, h as i32),
    }
}

/// An `Encoder` writes the bytes of `encode`, across qualities, sizes (odd block counts and edge
/// blocks for the grey pair path) and repeated solid tiles (cache hits and misses).
#[test]
fn encoder_reuse_matches_one_shot() {
    let no = AtomicBool::new(false);
    let lim = Limits::default();
    let mut e = jpeg::Encoder::default();
    let mut check = |m: EncodeImage, q: i32, c444: bool| {
        let o = EncodeOptions {
            quality: q,
            chroma444: c444,
        };
        let want = jpeg::encode(m, &o, &lim, &no).unwrap();
        for _ in 0..2 {
            assert_eq!(e.encode(m, &o, &lim, &no).unwrap(), want);
        }
    };
    for (w, h) in [(8, 8), (13, 5), (17, 16), (24, 9), (64, 64), (130, 33)] {
        let g = noisy_gray(w, h);
        for q in [30, 75, 100] {
            check(EncodeImage::Gray(&g), q, false);
        }
        let solid = Gray {
            pix: vec![200; w * h],
            stride: w,
            rect: g.rect,
        };
        let rgba = Rgba {
            pix: [10, 200, 90, 255].repeat(w * h),
            stride: 4 * w,
            rect: g.rect,
        };
        for q in [50, 75, 50] {
            check(EncodeImage::Gray(&solid), q, false);
            check(EncodeImage::Rgba(&rgba), q, false);
            check(EncodeImage::Rgba(&rgba), q, true);
        }
        let ycc = YCbCr {
            y: vec![90; w * h],
            cb: vec![100; w * h],
            cr: vec![150; w * h],
            y_stride: w,
            c_stride: w,
            ratio: Subsample::S444,
            rect: g.rect,
        };
        check(EncodeImage::YCbCr(&ycc), 75, false);
        check(EncodeImage::YCbCr(&ycc), 75, true);
        // One sample off: not solid any more, and the cache must not answer.
        let mut off = ycc.clone();
        off.cr[w * h - 1] = 151;
        check(EncodeImage::YCbCr(&off), 75, false);
        check(EncodeImage::YCbCr(&ycc), 75, false);
    }
}

/// A solid-tile cache hit still obeys the pixel limit, the output limit and the cancel flag.
#[test]
fn encoder_cache_hit_obeys_limits() {
    let g = Gray {
        pix: vec![5; 64 * 64],
        stride: 64,
        rect: Rect::new(0, 0, 64, 64),
    };
    let m = EncodeImage::Gray(&g);
    let o = EncodeOptions::default();
    let no = AtomicBool::new(false);
    let mut e = jpeg::Encoder::default();
    let first = e.encode(m, &o, &Limits::default(), &no).unwrap();
    assert_eq!(e.encode(m, &o, &Limits::default(), &no), Ok(first.clone()));
    let tight = Limits {
        max_pixels: 100,
        ..Default::default()
    };
    assert!(matches!(
        e.encode(m, &o, &tight, &no),
        Err(Error::PixelLimit { .. })
    ));
    let small = Limits {
        max_alloc: first.len() as u64 - 1,
        ..Default::default()
    };
    assert!(matches!(
        e.encode(m, &o, &small, &no),
        Err(Error::AllocLimit { .. })
    ));
    assert_eq!(
        e.encode(m, &o, &Limits::default(), &AtomicBool::new(true)),
        Err(Error::Cancelled)
    );
}

/// `draw_rgba`'s per-row chroma indices: a 4:2:2, 4:4:0, 4:1:1 or 4:1:0 YCbCr (also at odd and at
/// negative origins) encodes as the RGBA that `color.YCbCrToRGB` gives pixel by pixel.
#[test]
fn encode_ycbcr_other_ratios_matches_rgba() {
    let clamp = |v: i32| (v >> 16).clamp(0, 255) as u8;
    for ratio in [
        Subsample::S422,
        Subsample::S440,
        Subsample::S411,
        Subsample::S410,
        Subsample::S420,
    ] {
        for (ox, oy) in [(0, 0), (1, 0), (3, 5), (-3, -2), (-1, 4)] {
            let (w, h) = (19i32, 14i32);
            let mut m = YCbCr {
                y: (0..400 * 40).map(|i| (i * 7 % 251) as u8).collect(),
                cb: (0..400 * 40).map(|i| (i * 13 % 241) as u8).collect(),
                cr: (0..400 * 40).map(|i| (i * 29 % 239) as u8).collect(),
                y_stride: 40,
                c_stride: 40,
                ratio,
                rect: Rect::new(ox, oy, ox + w, oy + h),
            };
            if ratio == Subsample::S420 && (ox | oy) & 1 == 0 && ox >= 0 && oy >= 0 {
                m.rect = Rect::new(ox + 1, oy, ox + 1 + w, oy + h); // keep it off the direct path
            }
            let mut pix = Vec::new();
            for y in 0..h {
                for x in 0..w {
                    let (px, py) = (m.rect.min_x + x, m.rect.min_y + y);
                    let yy1 = m.y[m.y_offset(px, py)] as i32 * 0x10101;
                    let c = m.c_offset(px, py);
                    let (cb, cr) = (m.cb[c] as i32 - 128, m.cr[c] as i32 - 128);
                    pix.extend_from_slice(&[
                        clamp(yy1 + 91881 * cr),
                        clamp(yy1 - 22554 * cb - 46802 * cr),
                        clamp(yy1 + 116130 * cb),
                        255,
                    ]);
                }
            }
            let rgba = Rgba {
                pix,
                stride: 4 * w as usize,
                rect: Rect::new(0, 0, w, h),
            };
            for c444 in [false, true] {
                let o = EncodeOptions {
                    quality: 80,
                    chroma444: c444,
                };
                assert_eq!(
                    enc(EncodeImage::YCbCr(&m), &o),
                    enc(EncodeImage::Rgba(&rgba), &o),
                    "{ratio:?} at ({ox},{oy})"
                );
            }
        }
    }
}

#[test]
fn encode_to_reports_the_writer_error_kind() {
    struct Broken;
    impl std::io::Write for Broken {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::from(std::io::ErrorKind::BrokenPipe))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let gray = Gray {
        pix: vec![128; 64],
        stride: 8,
        rect: Rect::new(0, 0, 8, 8),
    };
    let r = jpeg::encode_to(
        &mut Broken,
        EncodeImage::Gray(&gray),
        &EncodeOptions::default(),
        &Limits::default(),
        &AtomicBool::new(false),
    );
    assert_eq!(r, Err(Error::Io(std::io::ErrorKind::BrokenPipe)));
}
