//! Decode against libjpeg-turbo 3.2.0: the md5 of every fixture's output at every scale and
//! format, from `tests/data/golden.json` (ported from jpeg-go's `golden_test.go`, `decoder_test.go`
//! and `scaled_test.go`).
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

mod common;
use common::*;
use jpeg::{Error, Format, Limits, Subsample};

const SCALES: [(u8, &str); 4] = [(1, "1/1"), (2, "1/2"), (4, "1/4"), (8, "1/8")];

#[test]
fn golden_decode() {
    let g = golden();
    let names = fixtures();
    assert!(names.len() >= 60, "fixtures: {}", names.len());
    let mut failures = Vec::new();
    for name in &names {
        let b = read(&format!("jpg/{name}"));
        let cfg = jpeg::decode_config(&b, None, &Limits::default()).unwrap();
        let want = g.get("decode").get(name);
        for (scale, key) in SCALES {
            let d = dec(&b, &opts(Format::Rgba, scale))
                .unwrap_or_else(|e| panic!("{name} rgba {key}: {e}"));
            assert_eq!(d.error, None, "{name} rgba {key}");
            if rgba_md5(&d.image) != want.get("rgba").get(key).str() {
                failures.push(format!("{name} rgba {key}"));
            }
            let d = dec(&b, &opts(Format::Gray, scale))
                .unwrap_or_else(|e| panic!("{name} gray {key}: {e}"));
            if yuv_and_gray_md5(&d.image).1 != want.get("gray").get(key).str() {
                failures.push(format!("{name} gray {key}"));
            }
            if cfg.components == 3 && cfg.rgb {
                assert_eq!(
                    dec(&b, &opts(Format::YCbCr, scale)).unwrap_err(),
                    Error::Unsupported
                );
                continue;
            }
            if scale == 8 && cfg.ratio == Subsample::S420 && cfg.height <= 8 {
                continue; // turbo's YUV has an uninitialised Y row here
            }
            let d = dec(&b, &opts(Format::YCbCr, scale))
                .unwrap_or_else(|e| panic!("{name} yuv {key}: {e}"));
            if yuv_and_gray_md5(&d.image).0 != want.get("yuv").get(key).str() {
                failures.push(format!("{name} yuv {key}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} mismatches: {failures:#?}",
        failures.len()
    );
}

#[test]
fn golden_abbreviated() {
    let g = golden();
    let tables = read("abbrev/tables.bin");
    let image = read("abbrev/image.jpg");
    let want = g.get("abbrev_spliced");
    for (scale, key) in SCALES {
        let mut o = opts(Format::Rgba, scale);
        o.tables = Some(&tables);
        let d = dec(&image, &o).unwrap();
        assert_eq!(
            rgba_md5(&d.image),
            want.get("rgba").get(key).str(),
            "rgba {key}"
        );
        o.format = Format::Gray;
        let d = dec(&image, &o).unwrap();
        assert_eq!(
            yuv_and_gray_md5(&d.image).1,
            want.get("gray").get(key).str(),
            "gray {key}"
        );
        o.format = Format::YCbCr;
        let d = dec(&image, &o).unwrap();
        assert_eq!(
            yuv_and_gray_md5(&d.image).0,
            want.get("yuv").get(key).str(),
            "yuv {key}"
        );
    }
}

#[test]
fn scaled_output_size() {
    let b = read("jpg/synth_420_17x9.jpg");
    for (scale, w, h) in [(1u8, 17, 9), (2, 9, 5), (4, 5, 3), (8, 3, 2)] {
        let d = dec(&b, &opts(Format::Rgba, scale)).unwrap();
        let r = d.image.bounds();
        assert_eq!((r.width(), r.height()), (w, h), "scale {scale}");
    }
}

#[test]
fn scale_unsupported() {
    let b = read("jpg/synth_420_16x16.jpg");
    assert_eq!(
        dec(&b, &opts(Format::YCbCr, 3)).unwrap_err(),
        Error::Unsupported
    );
}
