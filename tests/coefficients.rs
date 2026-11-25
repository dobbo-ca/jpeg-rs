//! Entropy decode against libjpeg-turbo's `jpeg_read_coefficients`, the truncated and corrupt
//! streams' RGBA, and the row reader (ported from jpeg-go's `huff_test.go`, `coef_test.go`,
//! `progressive_test.go` and `rowreader_test.go`).
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

mod common;
use common::*;
use jpeg::{Error, Format, Image, Limits, RowReader};
use std::sync::atomic::AtomicBool;

fn coef_md5(c: &jpeg::Coefficients) -> String {
    let cfg = c.config;
    let (hmax, vmax) = c
        .comp
        .iter()
        .fold((1, 1), |(h, v), k| (h.max(k.h), v.max(k.v)));
    let mut buf = Vec::new();
    for k in &c.comp {
        let real_wide = (cfg.width * k.h).div_ceil(hmax * 8);
        let real_high = (cfg.height * k.v).div_ceil(vmax * 8);
        for by in 0..real_high {
            for bx in 0..real_wide {
                for v in &k.blocks[by * k.blocks_wide + bx] {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
    }
    md5(&buf)
}

fn coefficients(src: &[u8]) -> jpeg::DecodedCoefficients {
    jpeg::decode_coefficients(src, None, &Limits::default(), &AtomicBool::new(false)).unwrap()
}

fn names(dir: &str) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(testdata(dir))
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    v.sort();
    v
}

#[test]
fn golden_coefficients() {
    let g = golden();
    for name in fixtures() {
        let d = coefficients(&read(&format!("jpg/{name}")));
        assert_eq!(d.error, None, "{name}");
        assert_eq!(
            coef_md5(&d.coefficients),
            g.get("decode").get(&name).get("coef").str(),
            "{name}"
        );
    }
    let t = names("truncated");
    assert!(!t.is_empty());
    for name in t {
        let d = coefficients(&read(&format!("truncated/{name}")));
        assert_eq!(d.error, Some(Error::Truncated), "{name}");
        assert_eq!(
            coef_md5(&d.coefficients),
            g.get("truncated_coef").get(&name).str(),
            "{name}"
        );
    }
}

#[test]
fn golden_truncated_rgba() {
    let g = golden();
    for name in g.get("truncated_rgba").keys() {
        let b = read(&format!("truncated/{name}"));
        let d = dec(&b, &opts(Format::Rgba, 1)).unwrap();
        assert_eq!(d.error, Some(Error::Truncated), "{name}");
        assert_eq!(
            rgba_md5(&d.image),
            g.get("truncated_rgba").get(name).str(),
            "{name}"
        );
    }
}

#[test]
fn decode_truncated_returns_the_partial_image() {
    for name in [
        "synth_multiscan_16x16_cut.jpg",
        "synth_restart_227x149_truncated.jpg",
    ] {
        let b = read(&format!("truncated/{name}"));
        let d = dec(&b, &Default::default()).unwrap();
        assert_eq!(d.error, Some(Error::Truncated), "{name}");
    }
}

#[test]
fn config_of_fixtures() {
    let g = golden();
    for name in g.get("header").keys() {
        let want = g.get("header").get(name);
        let c =
            jpeg::decode_config(&read(&format!("jpg/{name}")), None, &Limits::default()).unwrap();
        let n = |k: &str| want.get(k).str().parse::<usize>().unwrap();
        assert_eq!(
            (c.width, c.height, c.components),
            (n("width"), n("height"), n("components")),
            "{name}"
        );
        assert_eq!(c.rgb, want.get("rgb").str() == "true", "{name}");
        assert_eq!(c.restart_interval, n("restart_interval"), "{name}");
    }
}

/// The row reader's rows equal a whole-frame RGBA decode, byte for byte.
#[test]
fn row_reader_equals_decode() {
    let cancel = AtomicBool::new(false);
    for name in fixtures() {
        let b = read(&format!("jpg/{name}"));
        let want = dec(&b, &opts(Format::Rgba, 1)).unwrap();
        let Image::Rgba(want) = want.image else {
            panic!("not RGBA")
        };
        let mut r = RowReader::new(&b, None, &Limits::default(), &cancel).unwrap();
        assert!(!r.truncated(), "{name}");
        let mut got = Vec::new();
        let mut y = 0;
        while let Some(rows) = r.next(&cancel).unwrap() {
            assert_eq!(rows.rect.min_y, y, "{name}");
            y = rows.rect.max_y;
            got.extend_from_slice(&rows.pix);
        }
        assert_eq!(got, want.pix, "{name}");
        assert_eq!(r.next(&cancel).unwrap_err(), Error::RowReaderClosed);
    }
}

#[test]
fn row_reader_need_is_bounded() {
    let b = read("jpg/svs_420_512x512.jpg");
    let need = jpeg::row_reader_need(&b, None, &Limits::default()).unwrap();
    assert!(need > 512 * 512 * 6 / 2 && need < 1 << 24, "{need}");
    let few = Limits {
        max_pixels: 10,
        ..Limits::default()
    };
    assert!(matches!(
        jpeg::row_reader_need(&b, None, &few),
        Err(Error::PixelLimit { .. })
    ));
    let tiny = Limits {
        max_alloc: 1 << 16,
        ..Limits::default()
    };
    assert!(matches!(
        RowReader::new(&b, None, &tiny, &AtomicBool::new(false)),
        Err(Error::AllocLimit { .. })
    ));
}

/// A 8x8 baseline frame of four 1x1 components whose every block is DC 0, EOB.
fn four_component_jpeg() -> Vec<u8> {
    let mut v = vec![0xff, 0xd8, 0xff, 0xdb, 0x00, 0x43, 0x00];
    v.extend([1u8; 64]);
    v.extend([0xff, 0xc0, 0x00, 0x14, 8, 0, 8, 0, 8, 4]);
    for id in 1..=4 {
        v.extend([id, 0x11, 0]);
    }
    for class in [0x00, 0x10] {
        v.extend([0xff, 0xc4, 0x00, 0x14, class, 1]);
        v.extend([0u8; 15]);
        v.push(0); // one 1-bit code: DC category 0, AC EOB
    }
    v.extend([0xff, 0xda, 0x00, 0x0e, 4]);
    for id in 1..=4 {
        v.extend([id, 0x00]);
    }
    v.extend([0, 63, 0, 0x00, 0xff, 0xd9]); // four blocks of two 0 bits
    v
}

#[test]
fn four_components_only_from_decode_coefficients_any_components() {
    let src = four_component_jpeg();
    let (l, cancel) = (Limits::default(), AtomicBool::new(false));
    let c = jpeg::decode_coefficients_any_components(&src, None, &l, &cancel).unwrap();
    assert!(c.error.is_none());
    assert_eq!(c.coefficients.config.components, 4);
    assert!(!c.coefficients.config.rgb);
    assert_eq!(c.coefficients.comp.len(), 4);
    for k in &c.coefficients.comp {
        assert_eq!(k.blocks, vec![[0i16; 64]]);
    }
    assert!(matches!(
        jpeg::decode_coefficients(&src, None, &l, &cancel),
        Err(Error::Unsupported)
    ));
    assert!(matches!(
        jpeg::decode(&src, &jpeg::DecodeOptions::default(), &l, &cancel),
        Err(Error::Unsupported)
    ));
    assert!(matches!(
        jpeg::decode_config(&src, None, &l),
        Err(Error::Unsupported)
    ));
}
