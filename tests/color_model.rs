//! `DecodeOptions::color_model`: `Rgb` reads a 3-component frame as RGB with no colour transform,
//! whatever the stream's markers say (TIFF PhotometricInterpretation RGB, Aperio SVS); `Auto`, the
//! default, is the stream's own reading and is unchanged.

use jpeg::{
    ColorModel, DecodeOptions, EncodeImage, EncodeOptions, Error, Format, Image, Limits, Rect,
    Rgba, RowReader,
};
use std::sync::atomic::AtomicBool;

const W: usize = 45;
const H: usize = 37;

/// A 4:4:4 (or 4:2:0) YCbCr stream with component ids 1, 2, 3 and no JFIF or Adobe segment (the
/// encoder's JFIF APP0 is cut out).
fn stream(chroma444: bool) -> Vec<u8> {
    let mut pix = vec![0u8; 4 * W * H];
    for y in 0..H {
        for x in 0..W {
            let o = 4 * (y * W + x);
            pix[o..o + 4].copy_from_slice(&[
                (x * 5 + y) as u8,
                (200 - 3 * y) as u8,
                ((x * y) % 251) as u8,
                255,
            ]);
        }
    }
    let m = Rgba {
        pix,
        stride: 4 * W,
        rect: Rect::new(0, 0, W as i32, H as i32),
    };
    let o = EncodeOptions {
        quality: 90,
        chroma444,
    };
    let mut b = jpeg::encode(
        EncodeImage::Rgba(&m),
        &o,
        &Limits::default(),
        &AtomicBool::new(false),
    )
    .unwrap();
    if b[2..4] == [0xff, 0xe0] {
        let len = usize::from(u16::from_be_bytes([b[4], b[5]]));
        b.drain(2..4 + len);
    }
    assert_ne!(b[2..4], [0xff, 0xe0]);
    b
}

/// `b` with an APPn segment `data` inserted after SOI.
fn with_app(b: &[u8], marker: u8, data: &[u8]) -> Vec<u8> {
    let mut v = vec![0xff, 0xd8, 0xff, marker];
    v.extend_from_slice(&((data.len() + 2) as u16).to_be_bytes());
    v.extend_from_slice(data);
    v.extend_from_slice(&b[2..]);
    v
}

fn adobe(transform: u8) -> Vec<u8> {
    let mut d = b"Adobe\0\x64\0\0\0\0".to_vec();
    d.push(transform);
    d
}

fn decode(b: &[u8], format: Format, color_model: ColorModel, threads: usize) -> Image {
    let o = DecodeOptions {
        format,
        color_model,
        ..Default::default()
    };
    let lim = Limits {
        max_threads: threads,
        ..Limits::default()
    };
    let d = jpeg::decode(b, &o, &lim, &AtomicBool::new(false)).unwrap();
    assert_eq!(d.error, None);
    d.image
}

fn rgba(img: Image) -> Rgba {
    match img {
        Image::Rgba(m) => m,
        _ => panic!("not RGBA"),
    }
}

/// The RGB samples of an RGBA image, row by row.
fn rgb_of(m: &Rgba) -> Vec<u8> {
    let mut v = Vec::new();
    for y in m.rect.min_y..m.rect.max_y {
        let o = m.pix_offset(m.rect.min_x, y);
        for x in 0..m.rect.width() as usize {
            v.extend_from_slice(&m.pix[o + 4 * x..o + 4 * x + 3]);
            assert_eq!(m.pix[o + 4 * x + 3], 255);
        }
    }
    v
}

// With `Rgb`, the output samples are the stored component samples: the Y, Cb and Cr planes that
// `Format::YCbCr` returns for the same 4:4:4 stream, untouched.
#[test]
fn rgb_is_the_stored_samples() {
    let b = stream(true);
    let Image::YCbCr(p) = decode(&b, Format::YCbCr, ColorModel::Auto, 1) else {
        panic!("not YCbCr");
    };
    let mut want = Vec::new();
    for y in 0..H as i32 {
        let (yo, co) = (p.y_offset(0, y), p.c_offset(0, y));
        for x in 0..W {
            want.extend_from_slice(&[p.y[yo + x], p.cb[co + x], p.cr[co + x]]);
        }
    }
    for threads in [1, 4] {
        let got = rgb_of(&rgba(decode(&b, Format::Rgba, ColorModel::Rgb, threads)));
        assert_eq!(got, want, "threads {threads}");
    }
    // Auto converts the same stream, so it differs.
    let auto = rgb_of(&rgba(decode(&b, Format::Rgba, ColorModel::Auto, 1)));
    assert_ne!(auto, want);
}

// `Rgb` ignores the markers: JFIF and an Adobe transform of 1 give the same samples as none, and
// `Auto` on an Adobe transform 0 stream (the marker that says RGB) gives them too.
#[test]
fn rgb_ignores_markers() {
    let b = stream(true);
    let want = rgb_of(&rgba(decode(&b, Format::Rgba, ColorModel::Rgb, 1)));
    let jfif = with_app(&b, 0xe0, b"JFIF\0\x01\x01\0\0\x01\0\x01\0\0");
    for (name, s) in [
        ("jfif", jfif),
        ("adobe 1", with_app(&b, 0xee, &adobe(1))),
        ("adobe 0", with_app(&b, 0xee, &adobe(0))),
    ] {
        let got = rgb_of(&rgba(decode(&s, Format::Rgba, ColorModel::Rgb, 1)));
        assert_eq!(got, want, "{name}");
    }
    let marked = with_app(&b, 0xee, &adobe(0));
    let auto = rgb_of(&rgba(decode(&marked, Format::Rgba, ColorModel::Auto, 1)));
    assert_eq!(auto, want, "auto on adobe 0");
}

// A region and `Format::Gray` follow the same reading; `Format::YCbCr` is refused, as for a
// stream marked RGB.
#[test]
fn rgb_region_gray_ycbcr() {
    let b = stream(true);
    let full = rgba(decode(&b, Format::Rgba, ColorModel::Rgb, 1));
    let r = Rect::new(3, 9, 40, 30);
    let o = DecodeOptions {
        format: Format::Rgba,
        region: Some(r),
        color_model: ColorModel::Rgb,
        ..Default::default()
    };
    let cancel = AtomicBool::new(false);
    let d = jpeg::decode(&b, &o, &Limits::default(), &cancel).unwrap();
    let part = rgba(d.image);
    for y in r.min_y..r.max_y {
        let (a, c) = (full.pix_offset(r.min_x, y), part.pix_offset(r.min_x, y));
        let n = 4 * r.width() as usize;
        assert_eq!(full.pix[a..a + n], part.pix[c..c + n], "row {y}");
    }
    let marked = with_app(&b, 0xee, &adobe(0));
    assert_eq!(
        decode(&b, Format::Gray, ColorModel::Rgb, 1),
        decode(&marked, Format::Gray, ColorModel::Auto, 1)
    );
    let o = DecodeOptions {
        format: Format::YCbCr,
        color_model: ColorModel::Rgb,
        ..Default::default()
    };
    assert_eq!(
        jpeg::decode(&b, &o, &Limits::default(), &cancel).unwrap_err(),
        Error::Unsupported
    );
}

// Components with different samplings cannot be RGB here (libtiff refuses them too); `Auto` still
// decodes the 4:2:0 stream as YCbCr.
#[test]
fn rgb_subsampled_is_unsupported() {
    let b = stream(false);
    let cancel = AtomicBool::new(false);
    let o = DecodeOptions {
        format: Format::Rgba,
        color_model: ColorModel::Rgb,
        ..Default::default()
    };
    assert_eq!(
        jpeg::decode(&b, &o, &Limits::default(), &cancel).unwrap_err(),
        Error::Unsupported
    );
    assert_eq!(
        RowReader::with_color_model(&b, None, ColorModel::Rgb, &Limits::default(), &cancel).err(),
        Some(Error::Unsupported)
    );
    decode(&b, Format::Rgba, ColorModel::Auto, 1);
}

// The row reader gives the rows of `decode`, for both models, and its need stays under
// `row_reader_need`.
#[test]
fn row_reader_color_model() {
    let b = stream(true);
    let cancel = AtomicBool::new(false);
    let need = jpeg::row_reader_need(&b, None, &Limits::default()).unwrap();
    for cm in [ColorModel::Auto, ColorModel::Rgb] {
        let want = rgb_of(&rgba(decode(&b, Format::Rgba, cm, 1)));
        let lim = Limits {
            max_alloc: need + (1 << 17),
            ..Limits::default()
        };
        let mut rr = RowReader::with_color_model(&b, None, cm, &lim, &cancel).unwrap();
        let mut got = Vec::new();
        while let Some(m) = rr.next(&cancel).unwrap() {
            got.extend(rgb_of(m));
        }
        assert_eq!(got, want, "{cm:?}");
    }
}

// One component: `Rgb` changes nothing.
#[test]
fn rgb_gray_frame_unchanged() {
    let g = jpeg::Gray {
        pix: (0..W * H).map(|i| (i * 7) as u8).collect(),
        stride: W,
        rect: Rect::new(0, 0, W as i32, H as i32),
    };
    let b = jpeg::encode(
        EncodeImage::Gray(&g),
        &EncodeOptions {
            quality: 90,
            chroma444: false,
        },
        &Limits::default(),
        &AtomicBool::new(false),
    )
    .unwrap();
    for f in [Format::Gray, Format::Rgba] {
        assert_eq!(
            decode(&b, f, ColorModel::Rgb, 1),
            decode(&b, f, ColorModel::Auto, 1)
        );
    }
}
