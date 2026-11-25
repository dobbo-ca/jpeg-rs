//! `Decoder` (buffers kept between calls, never cleared) against the one-shot functions: the same
//! image, error and limits for a run of different inputs, a smaller frame after a larger one,
//! errors in between, a region, a hook that writes nothing, and every thread count.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

mod common;
use common::*;
use jpeg::{
    DecodeOptions, Decoder, Error, Format, Image, Limits, Recon, Rect, RgbaWindow, decode,
    decode_into_rgba,
};
use std::sync::atomic::AtomicBool;

/// The small fixtures of every directory, the largest first and then the smallest first, so a
/// small frame follows a large one and the other way round, with garbage between them.
fn sequence() -> Vec<Vec<u8>> {
    let mut v = Vec::new();
    for dir in ["jpg", "truncated", "subsample"] {
        for e in std::fs::read_dir(testdata(dir)).unwrap() {
            let n = e.unwrap().file_name().into_string().unwrap();
            if n.ends_with(".jpg") {
                let b = read(&format!("{dir}/{n}"));
                if b.len() < 100_000 {
                    v.push(b);
                }
            }
        }
    }
    v.sort_by_key(|b| std::cmp::Reverse(b.len()));
    let mut out = Vec::new();
    for b in v.iter().chain(v.iter().rev()) {
        out.push(b.clone());
        out.push(vec![0xff, 0xd8, 0xff, 0xdb, 0, 3, 0]); // a stream that fails to parse
    }
    out
}

fn limits(max_threads: usize) -> Limits {
    Limits {
        max_threads,
        ..Limits::default()
    }
}

fn noop(_: &mut [u8], _: usize, _: &[i32; 64]) {}

#[test]
fn reuse_equals_one_shot() {
    let seq = sequence();
    let go = AtomicBool::new(false);
    let region = Rect::new(9, 5, 38, 31);
    for threads in [1, 2, 0] {
        let l = limits(threads);
        for format in [Format::Rgba, Format::YCbCr, Format::Gray] {
            for (scale, region, hook) in [
                (1u8, None, false),
                (4, None, false),
                (1, Some(region), false),
                (1, None, true),
            ] {
                let o = DecodeOptions {
                    region,
                    recon: Recon {
                        idct: hook.then_some(noop as jpeg::IdctHook),
                        to_rgba: None,
                    },
                    ..opts(format, scale)
                };
                let mut dec = Decoder::new();
                for (i, b) in seq.iter().enumerate() {
                    let want = decode(b, &o, &l, &go);
                    let got = dec.decode(b, &o, &l, &go).cloned();
                    assert_eq!(
                        got,
                        want,
                        "input {i} {format:?} scale {scale} region {} hook {hook} threads {threads}",
                        region.is_some()
                    );
                }
            }
        }
    }
}

#[test]
fn reuse_into_a_window_equals_one_shot() {
    let seq = sequence();
    let go = AtomicBool::new(false);
    let o = opts(Format::Rgba, 1);
    for threads in [1, 2, 0] {
        let l = limits(threads);
        let mut dec = Decoder::new();
        // A window decode, then an image decode, on one Decoder.
        for (i, b) in seq.iter().enumerate() {
            let Ok(want) = decode(b, &o, &l, &go) else {
                assert!(dec.decode(b, &o, &l, &go).is_err(), "input {i}");
                continue;
            };
            let Image::Rgba(rgba) = &want.image else {
                panic!("input {i}: not RGBA")
            };
            let mut pix = vec![0xabu8; rgba.pix.len()];
            let w = RgbaWindow {
                pix: &mut pix,
                stride: rgba.stride,
            };
            let err = dec.decode_into_rgba(b, &o, &l, &go, w);
            assert_eq!(err, Ok(want.error.clone()), "input {i} threads {threads}");
            assert!(pix == rgba.pix, "input {i} threads {threads}: pixels");
            let w = RgbaWindow {
                pix: &mut pix,
                stride: rgba.stride,
            };
            assert_eq!(
                dec.decode(b, &o, &l, &go).cloned(),
                Ok(want.clone()),
                "input {i}"
            );
            assert_eq!(
                decode_into_rgba(b, &o, &l, &go, w),
                Ok(want.error.clone()),
                "input {i}"
            );
        }
    }
}

/// A reused `Decoder` charges what the call uses: the limit that stops a one-shot call stops it.
#[test]
fn reuse_keeps_the_limits() {
    let go = AtomicBool::new(false);
    let big = read("jpg/svs_420_512x512.jpg");
    let o = opts(Format::Rgba, 1);
    let mut dec = Decoder::new();
    assert!(dec.decode(&big, &o, &Limits::default(), &go).is_ok());
    let tight = Limits {
        max_alloc: 1 << 16,
        ..Limits::default()
    };
    let want = decode(&big, &o, &tight, &go);
    assert!(matches!(want, Err(Error::AllocLimit { .. })));
    assert_eq!(dec.decode(&big, &o, &tight, &go).cloned(), want);
    // and it still decodes after the failure
    let ok = decode(&big, &o, &Limits::default(), &go);
    assert_eq!(dec.decode(&big, &o, &Limits::default(), &go).cloned(), ok);
}
