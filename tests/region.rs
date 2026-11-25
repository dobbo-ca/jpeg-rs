//! Region decode and restart-interval skipping: a region equals a crop of the full decode
//! (ported from jpeg-go's `jpeg_region_test.go`).
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

mod common;
use common::*;
use jpeg::{DecodeOptions, Error, Format, Image, Limits, Rect};

struct Rng(u64);

impl Rng {
    fn step(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: i32) -> i32 {
        (self.step() % n as u64) as i32
    }
}

fn rand_rect(rng: &mut Rng, w: i32, h: i32) -> Rect {
    let (x0, y0) = (rng.below(w), rng.below(h));
    let x1 = x0 + 1 + rng.below(w - x0);
    let y1 = y0 + 1 + rng.below(h - y0);
    Rect::new(x0, y0, x1, y1)
}

/// Every pixel of `r` is equal in both images, compared on the raw planes (a chroma-only
/// mismatch must not cancel out through a colour conversion).
fn equal_in(a: &Image, b: &Image, r: Rect) -> bool {
    match (a, b) {
        (Image::Gray(a), Image::Gray(b)) => (r.min_y..r.max_y).all(|y| {
            (r.min_x..r.max_x).all(|x| a.pix[a.pix_offset(x, y)] == b.pix[b.pix_offset(x, y)])
        }),
        (Image::Rgba(a), Image::Rgba(b)) => (r.min_y..r.max_y).all(|y| {
            (r.min_x..r.max_x).all(|x| {
                a.pix[a.pix_offset(x, y)..a.pix_offset(x, y) + 4]
                    == b.pix[b.pix_offset(x, y)..b.pix_offset(x, y) + 4]
            })
        }),
        (Image::YCbCr(a), Image::YCbCr(b)) => {
            a.ratio == b.ratio
                && (r.min_y..r.max_y).all(|y| {
                    (r.min_x..r.max_x).all(|x| {
                        a.y[a.y_offset(x, y)] == b.y[b.y_offset(x, y)]
                            && a.cb[a.c_offset(x, y)] == b.cb[b.c_offset(x, y)]
                            && a.cr[a.c_offset(x, y)] == b.cr[b.c_offset(x, y)]
                    })
                })
        }
        _ => false,
    }
}

fn load_starts() -> Vec<usize> {
    String::from_utf8(read("strip_rst.starts"))
        .unwrap()
        .split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect()
}

#[test]
fn region_is_crop() {
    let fixtures = [
        "synth_420_227x149.jpg",
        "synth_422_227x149.jpg",
        "synth_444_227x149.jpg",
        "synth_gray_227x149.jpg",
        "synth_rgb_227x149.jpg",
        "synth_restart_227x149.jpg",
        "synth_420_17x9.jpg",
        // non-interleaved single-component scan with 2x2 sampling
        "gray22.jpg",
        "gray22_rst.jpg",
        // the first scan interleaves 2 of 3 components: genuinely multi-scan
        "ms_ycb_cr_420.jpg",
        "ms_cbcr_y_420.jpg",
        // progressive: whole-frame decode
        "prog_420_111x75.jpg",
        "prog_422_111x75.jpg",
        "prog_gray_111x75.jpg",
        "prog_inc_ac_420_111x75.jpg",
        "prog_sa_rst_420_111x75.jpg",
    ];
    let mut rng = Rng(1);
    for name in fixtures {
        let b = read(&format!("jpg/{name}"));
        let cfg = jpeg::decode_config(&b, None, &Limits::default()).unwrap();
        let mut formats = vec![Format::Rgba, Format::Gray];
        if cfg.components != 3 || !cfg.rgb {
            formats.push(Format::YCbCr);
        }
        for scale in [1u8, 2, 4, 8] {
            let (w, h) = (
                cfg.width.div_ceil(scale as usize) as i32,
                cfg.height.div_ceil(scale as usize) as i32,
            );
            for &format in &formats {
                let full = dec(&b, &opts(format, scale)).unwrap();
                for _ in 0..12 {
                    let r = rand_rect(&mut rng, w, h);
                    let mut o = opts(format, scale);
                    o.region = Some(r);
                    let got = dec(&b, &o).unwrap();
                    assert_eq!(got.error, full.error, "{name} {scale} {format:?} {r:?}");
                    assert_eq!(got.image.bounds(), r, "{name} {scale} {format:?}");
                    assert!(
                        equal_in(&got.image, &full.image, r),
                        "{name} 1/{scale} {format:?} {r:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn ndpi_restart_offsets_and_scan_agree() {
    let b = read("jpg/strip_rst.jpg");
    let starts = load_starts();
    let full = dec(&b, &Default::default()).unwrap();
    assert_eq!(full.error, None);
    let region = Rect::new(137, 619, 803, 1027); // mid-image, off any MCU-row boundary
    for restart in [Some(&starts[..]), None] {
        let o = DecodeOptions {
            region: Some(region),
            restart,
            ..Default::default()
        };
        let got = dec(&b, &o).unwrap();
        assert!(
            equal_in(&got.image, &full.image, region),
            "restart {}",
            restart.is_some()
        );
    }
}

#[test]
fn region_partly_outside_is_zero_filled() {
    let b = read("jpg/synth_420_227x149.jpg");
    for scale in [1u8, 2] {
        let sc = scale as i32;
        let (w, h) = ((227 + sc - 1) / sc, (149 + sc - 1) / sc);
        let img = Rect::new(0, 0, w, h);
        for format in [Format::YCbCr, Format::Rgba, Format::Gray] {
            let full = dec(&b, &opts(format, scale)).unwrap();
            for r in [
                Rect::new(-10, -10, 50, 50),
                Rect::new(w - 21, h - 19, w + 10, h + 10),
                Rect::new(-3, 5, w + 3, 6),
            ] {
                let mut o = opts(format, scale);
                o.region = Some(r);
                let got = dec(&b, &o).unwrap();
                assert_eq!(got.image.bounds(), r);
                assert!(
                    equal_in(&got.image, &full.image, r.intersect(img)),
                    "{format:?} {r:?}"
                );
                for y in r.min_y..r.max_y {
                    for x in r.min_x..r.max_x {
                        if x >= 0 && y >= 0 && x < w && y < h {
                            continue;
                        }
                        let v: &[u8] = match &got.image {
                            Image::Gray(g) => &g.pix[g.pix_offset(x, y)..][..1],
                            Image::Rgba(g) => &g.pix[g.pix_offset(x, y)..][..4],
                            // Y only: chroma offsets truncate toward zero, so x=-1 and x=0 share a sample.
                            Image::YCbCr(g) => &g.y[g.y_offset(x, y)..][..1],
                        };
                        assert!(v.iter().all(|&c| c == 0), "{format:?} {r:?} ({x},{y})");
                    }
                }
            }
        }
    }
}

#[test]
fn region_outside_the_image_is_an_error() {
    let b = read("jpg/synth_420_227x149.jpg");
    let o = DecodeOptions {
        region: Some(Rect::new(1000, 1000, 1010, 1010)),
        ..Default::default()
    };
    assert_eq!(dec(&b, &o).unwrap_err(), Error::RegionEmpty);
    let o = DecodeOptions {
        region: Some(Rect::new(-(1 << 30), 0, 5, 5)),
        ..Default::default()
    };
    assert_eq!(dec(&b, &o).unwrap_err(), Error::Region);
}

/// With `check_frame` a region's error is the whole frame's, wherever the damage is, and its
/// pixels stay a crop.
#[test]
fn region_check_frame() {
    let mut streams: Vec<(String, Vec<u8>, Option<Vec<usize>>)> = Vec::new();
    let mut names: Vec<String> = std::fs::read_dir(testdata("truncated"))
        .unwrap()
        .map(|e| format!("truncated/{}", e.unwrap().file_name().to_str().unwrap()))
        .collect();
    names.sort();
    for n in names {
        let b = read(&n);
        streams.push((n, b, None));
    }
    let b = read("jpg/synth_420_203x157.jpg");
    let sof9 = [0xff, 0xc9, 0, 11, 8, 0, 16, 0, 16, 1, 1, 0x11, 0];
    let mut after = b[..b.len() - 2].to_vec();
    after.extend_from_slice(&sof9);
    after.extend_from_slice(&b[b.len() - 2..]);
    streams.push(("afterscan".into(), after, None));
    streams.push(("synth_420_203x157".into(), b, None));
    let b = read("jpg/strip_rst.jpg");
    let starts = load_starts();
    streams.push((
        "strip_rst cut".into(),
        b[..b.len() * 6 / 10].to_vec(),
        Some(starts.clone()),
    ));
    // One middle interval of bad codes: restart skipping would jump it.
    let mut bad = b.clone();
    let k = starts.len() / 2;
    let mut i = starts[k];
    while i + 1 < starts[k + 1] - 2 {
        bad[i] = 0xff;
        bad[i + 1] = 0x00;
        i += 2;
    }
    streams.push(("strip_rst badhuff".into(), bad, Some(starts)));

    let class = |d: &Result<jpeg::Decoded, Error>| match d {
        Ok(d) => format!("img err={:?}", d.error),
        Err(e) => format!("none err={e:?}"),
    };
    for (name, b, restart) in &streams {
        for format in [Format::Rgba, Format::YCbCr] {
            let Ok(full) = dec(b, &opts(format, 1)) else {
                continue;
            };
            let want = class(&Ok(full.clone()));
            let (w, h) = (full.image.bounds().width(), full.image.bounds().height());
            for r in [
                Rect::new(0, 0, 1, 1),
                Rect::new(0, 0, w, h / 4 + 1),
                Rect::new(w / 4, h / 3, w / 2 + 1, h / 2 + 1),
                Rect::new(w - 1, h - 1, w, h),
            ] {
                let offs: Vec<Option<&[usize]>> = match restart {
                    Some(s) => vec![None, Some(&s[..])],
                    None => vec![None],
                };
                for rs in offs {
                    let o = DecodeOptions {
                        format,
                        region: Some(r),
                        restart: rs,
                        check_frame: true,
                        ..Default::default()
                    };
                    let got = dec(b, &o);
                    assert_eq!(
                        class(&got),
                        want,
                        "{name} {format:?} {r:?} offsets={}",
                        rs.is_some()
                    );
                    if let Ok(g) = got {
                        assert!(
                            equal_in(&g.image, &full.image, r),
                            "{name} {format:?} {r:?}"
                        );
                    }
                }
            }
        }
    }
}
