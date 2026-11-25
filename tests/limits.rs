//! Limits, cancellation and the `Dst` window (ported from jpeg-go's `limit_call_test.go`,
//! `limit_ledger_test.go` and `dst_test.go`; the process-wide limiter has no counterpart).
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

mod common;
use common::*;
use jpeg::{
    DecodeOptions, EncodeImage, EncodeOptions, Error, Format, Gray, Image, Limits, Rect, Rgba,
    RgbaWindow,
};
use std::sync::atomic::{AtomicBool, Ordering};

fn go() -> AtomicBool {
    AtomicBool::new(false)
}

#[test]
fn pixel_limit_is_checked_from_the_headers() {
    let b = read("jpg/svs_420_512x512.jpg");
    let l = Limits {
        max_pixels: 512 * 512 - 1,
        ..Limits::default()
    };
    let want = Error::PixelLimit {
        got: 512 * 512,
        max: 512 * 512 - 1,
    };
    assert_eq!(jpeg::decode_config(&b, None, &l).unwrap_err(), want);
    assert_eq!(
        jpeg::decode(&b, &Default::default(), &l, &go()).unwrap_err(),
        want
    );
    assert_eq!(
        jpeg::decode_coefficients(&b, None, &l, &go()).unwrap_err(),
        want
    );
    assert_eq!(jpeg::RowReader::new(&b, None, &l, &go()).err(), Some(want));
    let ok = Limits {
        max_pixels: 512 * 512,
        ..Limits::default()
    };
    assert!(jpeg::decode_config(&b, None, &ok).is_ok());
}

#[test]
fn alloc_limit_refuses_before_allocating() {
    let b = read("jpg/svs_420_512x512.jpg");
    for max_alloc in [0u64, 1 << 16, 1 << 19] {
        let l = Limits {
            max_alloc,
            ..Limits::default()
        };
        assert!(
            matches!(
                jpeg::decode(&b, &opts(Format::Rgba, 1), &l, &go()),
                Err(Error::AllocLimit { .. })
            ),
            "decode {max_alloc}"
        );
        assert!(
            matches!(
                jpeg::decode_coefficients(&b, None, &l, &go()),
                Err(Error::AllocLimit { .. })
            ),
            "coefficients {max_alloc}"
        );
    }
    // The same call fits with room.
    let l = Limits {
        max_alloc: 1 << 26,
        ..Limits::default()
    };
    assert!(jpeg::decode(&b, &opts(Format::Rgba, 1), &l, &go()).is_ok());
}

#[test]
fn cancel_flag_stops_every_long_call() {
    let b = read("jpg/svs_420_512x512.jpg");
    let stop = AtomicBool::new(true);
    let l = Limits::default();
    assert_eq!(
        jpeg::decode(&b, &Default::default(), &l, &stop).unwrap_err(),
        Error::Cancelled
    );
    assert_eq!(
        jpeg::decode_coefficients(&b, None, &l, &stop).unwrap_err(),
        Error::Cancelled
    );
    assert_eq!(
        jpeg::RowReader::new(&b, None, &l, &stop).err(),
        Some(Error::Cancelled)
    );
    let img = Rgba {
        pix: vec![7; 64 * 64 * 4],
        stride: 256,
        rect: Rect::new(0, 0, 64, 64),
    };
    assert_eq!(
        jpeg::encode(
            EncodeImage::Rgba(&img),
            &EncodeOptions::default(),
            &l,
            &stop
        )
        .unwrap_err(),
        Error::Cancelled
    );
    // A reader stops between rows when the flag is set later.
    let flag = go();
    let mut r = jpeg::RowReader::new(&b, None, &l, &flag).unwrap();
    assert!(r.next(&flag).unwrap().is_some());
    flag.store(true, Ordering::Relaxed);
    assert_eq!(r.next(&flag).unwrap_err(), Error::Cancelled);
    assert_eq!(r.next(&flag).unwrap_err(), Error::RowReaderClosed);
}

#[test]
fn encode_limits() {
    let img = Rgba {
        pix: vec![9; 64 * 64 * 4],
        stride: 256,
        rect: Rect::new(0, 0, 64, 64),
    };
    let o = EncodeOptions::default();
    let tiny = Limits {
        max_pixels: 100,
        ..Limits::default()
    };
    assert!(matches!(
        jpeg::encode(EncodeImage::Rgba(&img), &o, &tiny, &go()),
        Err(Error::PixelLimit { .. })
    ));
    let none = Limits {
        max_alloc: 0,
        ..Limits::default()
    };
    assert!(matches!(
        jpeg::encode(EncodeImage::Rgba(&img), &o, &none, &go()),
        Err(Error::AllocLimit { .. })
    ));
    let short = Rgba {
        pix: vec![0; 10],
        ..img.clone()
    };
    assert_eq!(
        jpeg::encode(EncodeImage::Rgba(&short), &o, &Limits::default(), &go()).unwrap_err(),
        Error::BadImage
    );
}

#[test]
fn dst_window_equals_decode() {
    for name in [
        "synth_420_227x149.jpg",
        "synth_422_227x149.jpg",
        "synth_gray_227x149.jpg",
        "synth_rgb_227x149.jpg",
        "prog_420_111x75.jpg",
    ] {
        let b = read(&format!("jpg/{name}"));
        let full = dec(&b, &opts(Format::Rgba, 1)).unwrap();
        let Image::Rgba(full) = full.image else {
            panic!("not RGBA")
        };
        let (w, h) = (full.rect.width() as usize, full.rect.height() as usize);
        let stride = 4 * w + 12;
        let mut buf = vec![0xaau8; stride * (h - 1) + 4 * w];
        let err = jpeg::decode_into_rgba(
            &b,
            &opts(Format::Rgba, 1),
            &Limits::default(),
            &go(),
            RgbaWindow {
                pix: &mut buf,
                stride,
            },
        )
        .unwrap();
        assert_eq!(err, None, "{name}");
        for y in 0..h {
            assert_eq!(
                &buf[y * stride..y * stride + 4 * w],
                &full.pix[y * 4 * w..(y + 1) * 4 * w],
                "{name} row {y}"
            );
            if y + 1 < h {
                assert!(
                    buf[y * stride + 4 * w..(y + 1) * stride]
                        .iter()
                        .all(|&v| v == 0xaa),
                    "{name} padding {y}"
                );
            }
        }
    }
}

#[test]
fn dst_wrong_shape_is_unsupported_before_any_write() {
    let b = read("jpg/synth_420_227x149.jpg");
    let (w, h) = (227usize, 149usize);
    let l = Limits::default();
    let run = |o: &DecodeOptions, stride: usize, len: usize| {
        let mut buf = vec![0xaau8; len];
        let r = jpeg::decode_into_rgba(
            &b,
            o,
            &l,
            &go(),
            RgbaWindow {
                pix: &mut buf,
                stride,
            },
        );
        assert!(buf.iter().all(|&v| v == 0xaa), "wrote before refusing");
        r
    };
    let rgba = opts(Format::Rgba, 1);
    assert_eq!(
        run(&rgba, 4 * w - 1, 4 * w * h).unwrap_err(),
        Error::Unsupported
    );
    assert_eq!(
        run(&rgba, 4 * w, 4 * w * h - 1).unwrap_err(),
        Error::Unsupported
    );
    assert_eq!(
        run(&opts(Format::Gray, 1), 4 * w, 4 * w * h).unwrap_err(),
        Error::Unsupported
    );
    assert_eq!(
        run(&opts(Format::Rgba, 2), 4 * w, 4 * w * h).unwrap_err(),
        Error::Unsupported
    );
    let mut region = opts(Format::Rgba, 1);
    region.region = Some(Rect::new(0, 0, 10, 10));
    assert_eq!(
        run(&region, 4 * w, 4 * w * h).unwrap_err(),
        Error::Unsupported
    );
}

/// The encoder charges the output it may write, not only what it has: a row's reservation counts
/// against `max_alloc` before it is made.
#[test]
fn encode_output_is_charged_to_max_alloc() {
    let mut seed = 12345u32;
    let mut noise = |n: usize| -> Vec<u8> {
        (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (seed >> 24) as u8
            })
            .collect()
    };
    let small = Limits {
        max_alloc: 65536,
        ..Limits::default()
    };
    let gray = Gray {
        pix: noise(65535 * 8),
        stride: 65535,
        rect: Rect::new(0, 0, 65535, 8),
    };
    assert!(matches!(
        jpeg::encode(
            EncodeImage::Gray(&gray),
            &EncodeOptions::default(),
            &small,
            &go()
        ),
        Err(Error::AllocLimit { .. })
    ));
    let rgba = Rgba {
        pix: noise(4096 * 16 * 4),
        stride: 4096 * 4,
        rect: Rect::new(0, 0, 4096, 16),
    };
    for chroma444 in [false, true] {
        let o = EncodeOptions {
            chroma444,
            ..Default::default()
        };
        assert!(matches!(
            jpeg::encode(EncodeImage::Rgba(&rgba), &o, &small, &go()),
            Err(Error::AllocLimit { .. })
        ));
        assert!(jpeg::encode(EncodeImage::Rgba(&rgba), &o, &Limits::default(), &go()).is_ok());
    }
}

/// A header-only 1-component baseline stream (standard tables, no entropy data) of w x h.
fn header_only_gray(w: usize, h: usize) -> Vec<u8> {
    let mut b = vec![0xff, 0xd8, 0xff, 0xdb, 0, 67, 0];
    b.extend_from_slice(&[1; 64]);
    b.extend_from_slice(&[
        0xff,
        0xc0,
        0,
        11,
        8,
        (h >> 8) as u8,
        h as u8,
        (w >> 8) as u8,
        w as u8,
    ]);
    b.extend_from_slice(&[1, 1, 0x11, 0]);
    b.extend_from_slice(&[0xff, 0xda, 0, 8, 1, 1, 0, 0, 63, 0]);
    b
}

/// Under the default `Limits` only Go's 1 GiB cap applies, to the need less the page rounding
/// (`need - ext`): a frame over it is refused from the headers, before any allocation.
#[test]
fn default_limits_refuse_over_the_cap_from_the_headers() {
    // 16784 x 64000 after block padding, 1,074,176,000 B of plane.
    let b = header_only_gray(16777, 64000);
    let want = 1u64 << 30;
    match jpeg::decode(&b, &Default::default(), &Limits::default(), &go()) {
        Err(Error::AllocLimit { got, max }) => {
            assert_eq!(max, want);
            assert_eq!(got, 1_074_176_000 + (1 << 16)); // need less the pages
        }
        r => panic!("{:?}", r.map(|d| d.image.bounds())),
    }
    // A user-set max_alloc is Go's MaxBytes: the whole need counts against it.
    let l = Limits {
        max_alloc: want + 1,
        ..Limits::default()
    };
    match jpeg::decode(&b, &Default::default(), &l, &go()) {
        Err(Error::AllocLimit { got, max }) => {
            assert_eq!(max, want + 1);
            assert_eq!(got, 1_074_176_000 + (1 << 16) + 2 * 8192); // with the pages
        }
        r => panic!("{:?}", r.map(|d| d.image.bounds())),
    }
}
