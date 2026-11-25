#![no_main]
use jpeg::{DecodeOptions, Decoded, Error, Format, Image, Limits, Rect, RgbaWindow};
use libfuzzer_sys::fuzz_target;

#[path = "../common.rs"]
mod common;
use common::*;

/// Whether every pixel of `img` outside `eff` is zero: a region past the image is zero-filled.
fn zero_outside(img: &Image, eff: Rect) -> bool {
    let b = img.bounds();
    (b.min_y..b.max_y).all(|y| {
        (b.min_x..b.max_x).all(|x| {
            if (eff.min_x..eff.max_x).contains(&x) && (eff.min_y..eff.max_y).contains(&y) {
                return true;
            }
            match img {
                Image::Rgba(m) => m.pix[m.pix_offset(x, y)..][..4] == [0; 4],
                Image::Gray(m) => m.pix[m.pix_offset(x, y)] == 0,
                Image::YCbCr(m) => m.y[m.y_offset(x, y)] == 0,
            }
        })
    })
}

/// What a container's metadata sets: a restart-offset list (NDPI tag 65426: the stream's real
/// RSTn offsets, extra or reordered ones), `check_frame`, and a region that starts at a negative or
/// odd origin and leaves the image. The tail of the input says which. Nothing panics; and when the
/// whole decode was clean and no restart list is given, the region is that decode cropped, with
/// zeros outside the image (or `Error::RegionEmpty` when it does not meet the image).
fn container_options(
    src: &[u8],
    inp: &Input,
    o: &DecodeOptions,
    l: &Limits,
    full: Option<&Decoded>,
) {
    let g = |i: usize| src.get(src.len().wrapping_sub(1 + i)).copied().unwrap_or(0);
    let mut starts: Vec<usize> = inp
        .data
        .windows(2)
        .enumerate()
        .filter(|(_, w)| w[0] == 0xff && (0xd0..=0xd7).contains(&w[1]))
        .map(|(i, _)| i + 2)
        .collect();
    for k in 0..usize::from(g(5) & 3) {
        let at = (k * 31 + usize::from(g(4))) % inp.data.len().max(1);
        starts.push(usize::from(inp.data.get(at).copied().unwrap_or(0)) * (1 + usize::from(g(6))));
    }
    if g(5) & 4 != 0 {
        starts.reverse();
    }
    let restart = (g(5) & 8 != 0).then_some(&starts[..]);
    let (x0, y0) = (
        i32::from(g(0) as i8) * 3 + i32::from(g(4) & 1),
        i32::from(g(1) as i8) * 3 + i32::from(g(4) >> 1 & 1),
    );
    let region = Rect::new(
        x0,
        y0,
        x0 + 1 + i32::from(g(2)) * 2,
        y0 + 1 + i32::from(g(3)) * 2,
    );
    let opts = DecodeOptions {
        region: Some(region),
        check_frame: g(5) & 16 != 0,
        restart,
        ..*o
    };
    let r = jpeg::decode(inp.data, &opts, l, &NO);
    // Threads never change a region decode either, with or without the offsets.
    let l4 = Limits {
        max_threads: 4,
        ..*l
    };
    assert_eq!(
        jpeg::decode_probe(inp.data, &opts, &l4, &NO, true).0,
        r,
        "region {region:?} with max_threads 4 differs from 1"
    );
    // The same list over the whole frame.
    let _ = jpeg::decode(inp.data, &DecodeOptions { restart, ..*o }, l, &NO);
    let (Some(full), None) = (full, restart) else {
        return;
    };
    let eff = region.intersect(full.image.bounds());
    match r {
        Err(Error::AllocLimit { .. } | Error::PixelLimit { .. }) => {}
        Err(Error::RegionEmpty) => assert!(
            eff.is_empty(),
            "region {region:?} refused but meets the image"
        ),
        Err(e) => panic!("region {region:?} failed after the full decode succeeded: {e}"),
        Ok(reg) => {
            assert!(
                !eff.is_empty(),
                "region {region:?} outside the image decoded"
            );
            assert!(reg.error.is_none(), "region {region:?}: {:?}", reg.error);
            assert_eq!(reg.image.bounds(), region);
            assert!(
                same_samples(&full.image, &reg.image, eff),
                "region {region:?} is not a crop"
            );
            assert!(
                zero_outside(&reg.image, eff),
                "region {region:?} is not zero outside the image"
            );
        }
    }
}

// data = [format][scale][bit 0: split off tables][split position][stream], as jpeg-go's
// FuzzDecode. A decode never panics; when it returns an image with no error, a region of it
// (taken from the option bytes) equals a crop of the whole decode, and RGBA at scale 1 written
// straight into a caller's buffer equals the decode. Then the options a container hands over:
// see `container_options`.
fuzz_target!(|src: &[u8]| {
    let Some(inp) = parse(src) else { return };
    let l = limits();
    let o = DecodeOptions {
        format: inp.format(),
        scale: inp.scale(),
        tables: inp.tables,
        ..Default::default()
    };
    let full = jpeg::decode(inp.data, &o, &l, &NO);
    // Threads never change the result: 4 threads, every frame that can be cut cut.
    let l4 = Limits {
        max_threads: 4,
        ..l
    };
    assert_eq!(
        jpeg::decode_probe(inp.data, &o, &l4, &NO, true).0,
        full,
        "max_threads 4 differs from 1"
    );
    container_options(
        src,
        &inp,
        &o,
        &l,
        full.as_ref().ok().filter(|d| d.error.is_none()),
    );
    let Ok(full) = full else {
        return;
    };
    if full.error.is_some() {
        return;
    }
    let b = full.image.bounds();
    let (w, h) = (b.width(), b.height());
    if w == 0 || h == 0 {
        return;
    }
    let x0 = i32::from(inp.b[3]) % w;
    let y0 = i32::from(inp.b[2]) % h;
    let x1 = x0 + 1 + i32::from(inp.b[1]) % (w - x0);
    let y1 = y0 + 1 + i32::from(inp.b[0]) % (h - y0);
    let region = Rect::new(x0, y0, x1, y1);
    let reg = jpeg::decode(
        inp.data,
        &DecodeOptions {
            region: Some(region),
            ..o
        },
        &l,
        &NO,
    )
    .unwrap_or_else(|e| panic!("region {region:?} failed after the full decode succeeded: {e}"));
    assert!(reg.error.is_none(), "region {region:?}: {:?}", reg.error);
    assert_eq!(reg.image.bounds(), region);
    assert!(
        same_samples(&full.image, &reg.image, region),
        "region {region:?} differs from a crop of the full decode"
    );

    if let (Format::Rgba, 1, Image::Rgba(m)) = (o.format, o.scale, &full.image) {
        let (w, h) = (w as usize, h as usize);
        // A window with a gap after each row: the rows are the decode, the gaps stay as they were.
        let stride = 4 * w + usize::from(inp.b[3]) % 17;
        let mut pix = vec![0xa5u8; stride * h];
        let r = jpeg::decode_into_rgba(
            inp.data,
            &o,
            &l,
            &NO,
            RgbaWindow {
                pix: &mut pix,
                stride,
            },
        );
        assert_eq!(r, Ok(None));
        let want = tight_rgba(m);
        for y in 0..h {
            let row = &pix[y * stride..(y + 1) * stride];
            assert!(
                row[..4 * w] == want[y * 4 * w..(y + 1) * 4 * w],
                "decode_into_rgba row {y} differs from decode"
            );
            assert!(
                row[4 * w..].iter().all(|&v| v == 0xa5),
                "decode_into_rgba wrote into the gap after row {y}"
            );
        }
        // A stride under the row, or a buffer one byte short, is refused before any write.
        for (len, stride) in [(4 * w * h - 1, 4 * w), (4 * w * h, 4 * w - 1)] {
            let mut bad = vec![0u8; len];
            let r = jpeg::decode_into_rgba(
                inp.data,
                &o,
                &l,
                &NO,
                RgbaWindow {
                    pix: &mut bad,
                    stride,
                },
            );
            assert_eq!(r, Err(Error::Unsupported));
        }
    }
});
