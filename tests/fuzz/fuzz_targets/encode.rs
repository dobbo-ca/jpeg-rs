#![no_main]
use jpeg::{DecodeOptions, EncodeImage, EncodeOptions, Format, Gray, Rect, Rgba, YCbCr};
use libfuzzer_sys::fuzz_target;

#[path = "../common.rs"]
mod common;
use common::*;

// data = [kind][w: u16 LE][h: u16 LE][quality][flags][offsets][samples]: an image of 1..=200 by
// 1..=200 pixels, the sub-image of a parent with strides past the width, its samples the rest of
// the input cycled. The encoder gives a stream that decodes to the image's size; the same input
// twice gives the same bytes; quality 0 is 75; and an RGBA or gray sub-image gives the bytes of a
// tight copy of its pixels. (The SIMD kernels have their own differential targets.) With bit 7 of
// `kind` the image is hostile instead: rect, strides and plane lengths from the input, which need
// not agree, and the encoder gives Ok or Err, never a panic.
fn hostile(kind: u8, d: &[u8]) {
    let mut at = 0usize;
    let mut take = |n: usize| -> i64 {
        let mut v = 0i64;
        for _ in 0..n {
            v = v << 8 | i64::from(d.get(at).copied().unwrap_or(0));
            at += 1;
        }
        v
    };
    let mut i16_ = || take(2) as i16 as i32;
    let rect = Rect::new(i16_(), i16_(), i16_(), i16_());
    let (ys, cs, yl, cl) = (
        take(2) as usize,
        take(2) as usize,
        take(2) as usize,
        take(2) as usize,
    );
    let o = EncodeOptions {
        quality: 50,
        chroma444: kind & 0x40 != 0,
    };
    let (gray, rgba, ycc);
    let img = match kind & 3 {
        1 => {
            gray = Gray {
                pix: vec![0x80; yl],
                stride: ys,
                rect,
            };
            EncodeImage::Gray(&gray)
        }
        2 => {
            rgba = Rgba {
                pix: vec![0x80; yl],
                stride: ys,
                rect,
            };
            EncodeImage::Rgba(&rgba)
        }
        _ => {
            ycc = YCbCr {
                y: vec![0x80; yl],
                cb: vec![0x80; cl],
                cr: vec![0x80; cl],
                y_stride: ys,
                c_stride: cs,
                ratio: RATIOS[usize::from(kind >> 2 & 7) % 6],
                rect,
            };
            EncodeImage::YCbCr(&ycc)
        }
    };
    let _ = jpeg::encode(img, &o, &limits(), &NO);
}

fuzz_target!(|data: &[u8]| {
    let Some((&[kind, wl, wh, hl, hh, q, flags, offs], bytes)) = data.split_first_chunk::<8>()
    else {
        return;
    };
    if bytes.is_empty() {
        return;
    }
    if kind & 0x80 != 0 {
        hostile(kind, bytes);
        return;
    }
    let (w, h) = (
        1 + usize::from(u16::from_le_bytes([wl, wh])) % 200,
        1 + usize::from(u16::from_le_bytes([hl, hh])) % 200,
    );
    let (ox, oy, pad) = (
        usize::from(offs & 3),
        usize::from(offs >> 2 & 3),
        usize::from(flags >> 4 & 7),
    );
    let pix = build(kind % 8, w, h, ox, oy, pad, bytes);
    let img = match &pix {
        Pixels::Rgba(m) => EncodeImage::Rgba(m),
        Pixels::Gray(m) => EncodeImage::Gray(m),
        Pixels::YCbCr(m) => EncodeImage::YCbCr(m),
    };
    let l = limits();
    let o = EncodeOptions {
        quality: i32::from(q) % 130 - 5,
        chroma444: flags & 1 == 1,
    };
    let j = jpeg::encode(img, &o, &l, &NO).expect("a valid image encodes");
    assert!(j.starts_with(&[0xff, 0xd8]) && j.ends_with(&[0xff, 0xd9]));
    for format in [Format::Rgba, Format::YCbCr] {
        let d = jpeg::decode(
            &j,
            &DecodeOptions {
                format,
                ..Default::default()
            },
            &l,
            &NO,
        )
        .expect("the encoder's stream decodes");
        assert!(d.error.is_none());
        let b = d.image.bounds();
        assert_eq!((b.width() as usize, b.height() as usize), (w, h));
    }
    assert_eq!(jpeg::encode(img, &o, &l, &NO).as_ref(), Ok(&j));
    let mut sink = Vec::new();
    assert_eq!(jpeg::encode_to(&mut sink, img, &o, &l, &NO), Ok(()));
    assert!(sink == j, "encode_to differs from encode");
    if o.quality == 0 {
        let o75 = EncodeOptions { quality: 75, ..o };
        assert_eq!(jpeg::encode(img, &o75, &l, &NO).as_ref(), Ok(&j));
    }
    // The same pixels with no origin and no padding.
    let tight = match &pix {
        Pixels::Rgba(m) => {
            let t = tight_rgba(m);
            let m = jpeg::Rgba {
                pix: t,
                stride: 4 * w,
                rect: jpeg::Rect::new(0, 0, w as i32, h as i32),
            };
            jpeg::encode(EncodeImage::Rgba(&m), &o, &l, &NO)
        }
        Pixels::Gray(m) => {
            let t: Vec<u8> = (0..h)
                .flat_map(|y| m.pix[y * m.stride..y * m.stride + w].iter().copied())
                .collect();
            let m = jpeg::Gray {
                pix: t,
                stride: w,
                rect: jpeg::Rect::new(0, 0, w as i32, h as i32),
            };
            jpeg::encode(EncodeImage::Gray(&m), &o, &l, &NO)
        }
        Pixels::YCbCr(_) => return,
    };
    assert_eq!(
        tight.as_ref(),
        Ok(&j),
        "a sub-image differs from a tight copy"
    );
});
