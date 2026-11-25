//! Threading (docs/threading.md): the output and the error are the same for every
//! `max_threads`, bands and stripes really run on more than one thread, and the memory for the
//! threads never turns into an error a single-threaded call would not give. Ported in part from
//! jpeg-go's `workers_test.go` (`TestWorkersMatchDefault`, `TestWorkersAtLimit`).
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

mod common;
use common::*;
use jpeg::{DecodeOptions, Decoded, Error, Format, Limits, Probe, Rect, RgbaWindow};
use std::sync::atomic::AtomicBool;

fn go() -> AtomicBool {
    AtomicBool::new(false)
}

fn limits(max_threads: usize) -> Limits {
    Limits {
        max_threads,
        ..Limits::default()
    }
}

/// `decode` on `t` threads with every frame that can be cut cut (the size threshold off).
fn run(b: &[u8], o: &DecodeOptions, t: usize) -> (Result<Decoded, Error>, Probe) {
    jpeg::decode_probe(b, o, &limits(t), &go(), true)
}

/// The most workers a decode of `b` used in up to 30 tries: a pool task that wakes after the
/// caller has taken every job is not a worker the call used, so one try can read 1 on a busy host.
fn peak_of(b: &[u8], o: &DecodeOptions, t: usize, uncapped: bool) -> usize {
    let mut best = 1;
    for _ in 0..30 {
        let p = jpeg::decode_probe(b, o, &limits(t), &go(), uncapped).1;
        best = best.max(p.threads_peak);
        if best > 1 {
            break;
        }
    }
    best
}

/// The jpg, truncated and subsample fixtures: (name, bytes).
fn all_fixtures() -> Vec<(String, Vec<u8>)> {
    let mut v = Vec::new();
    for dir in ["jpg", "truncated", "subsample"] {
        let mut names: Vec<String> = std::fs::read_dir(testdata(dir))
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .filter(|n| n.ends_with(".jpg"))
            .collect();
        names.sort();
        for n in names {
            let b = read(&format!("{dir}/{n}"));
            v.push((format!("{dir}/{n}"), b));
        }
    }
    v
}

#[test]
fn every_thread_count_gives_the_serial_result() {
    for (name, b) in all_fixtures() {
        let big = b.len() > 400_000; // the sweep is a debug build: keep the big frames lean
        let (threads, scales): (&[usize], &[u8]) = if big {
            (&[2, 3, 8], &[1, 4])
        } else {
            (&[2, 3, 4, 5, 6, 7, 8], &[1, 2, 4, 8])
        };
        for format in [Format::Rgba, Format::YCbCr, Format::Gray] {
            for &scale in scales {
                let o = opts(format, scale);
                let (want, p1) = run(&b, &o, 1);
                assert_eq!(p1.threads_peak, 1, "{name}: one thread must not spawn");
                for &t in threads {
                    let (got, p) = run(&b, &o, t);
                    assert_eq!(got, want, "{name} {format:?} scale {scale} threads {t}");
                    assert!(p.threads_peak <= t, "{name}: {} > {t}", p.threads_peak);
                }
            }
        }
    }
}

/// A restart stream and a wide one use more than one thread, and `max_threads` caps them.
#[test]
fn restart_streams_use_threads() {
    for name in [
        "jpg/strip_rst.jpg",
        "jpg/synth_restart_227x149.jpg",
        "jpg/ndpi_wide_4096x32.jpg",
    ] {
        let b = read(name);
        for format in [Format::Rgba, Format::Gray] {
            let o = opts(format, 1);
            let (want, p1) = run(&b, &o, 1);
            assert_eq!(p1.threads_peak, 1);
            for t in [2usize, 3, 4] {
                let (got, p) = run(&b, &o, t);
                assert_eq!(got, want, "{name} {format:?} threads {t}");
                assert!(p.threads_peak <= t, "{name} {format:?} threads {t}");
                let peak = peak_of(&b, &o, t, true);
                assert!(
                    peak > 1 && peak <= t,
                    "{name} {format:?} threads {t}: peak {peak}"
                );
            }
            // 0 counts as 1; a huge value is capped (Go's maxWorkers) and cannot overflow.
            assert_eq!(run(&b, &o, 0).1.threads_peak, 1);
            let (got, p) = run(&b, &o, usize::MAX);
            assert_eq!(got, want);
            assert!(p.threads_peak <= 256);
        }
    }
}

/// A small frame now uses some workers at the default caps instead of running serial: a 512x512
/// 4:2:0 tile has two stripes' worth of pixels, and a dense restart stream is cut by its bytes.
#[test]
fn small_frames_use_some_workers_by_default() {
    let b = read("jpg/svs_420_512x512.jpg");
    let o = opts(Format::Rgba, 1);
    let peak = peak_of(&b, &o, 8, false);
    assert!((1..=8).contains(&peak), "{peak}");
    let got = jpeg::decode(&b, &o, &limits(8), &go());
    assert_eq!(got, run(&b, &o, 1).0);
    let b = read("jpg/ndpi_wide_4096x32.jpg");
    let o = opts(Format::Gray, 1);
    let peak = peak_of(&b, &o, 8, false);
    assert!((2..=8).contains(&peak), "{peak}");
    assert_eq!(
        jpeg::decode_probe(&b, &o, &limits(8), &go(), false).0,
        run(&b, &o, 1).0
    );
}

/// The caps size the workers: no more than the bytes or pixels pay for (a 4096x32 stream of
/// 165 KB gets 20 byte-sized shares, so `max_threads` is what binds).
#[test]
fn caps_never_exceed_max_threads() {
    let b = read("jpg/ndpi_wide_4096x32.jpg");
    let o = opts(Format::Gray, 1);
    for t in [2usize, 3, 5] {
        let p = jpeg::decode_probe(&b, &o, &limits(t), &go(), false).1;
        assert!(p.threads_peak <= t, "{}", p.threads_peak);
    }
}

/// Damage in the middle of the `strip_rst` stream, in a marker, and at the end: (what, bytes).
fn damaged_cases(b: &[u8]) -> Vec<(String, Vec<u8>)> {
    let sos = b.windows(2).position(|w| w == [0xff, 0xda]).unwrap();
    let rst = |n: u8| {
        b[sos..]
            .windows(2)
            .position(|w| w == [0xff, 0xd0 + n])
            .map(|i| sos + i)
            .unwrap()
    };
    let mut cases: Vec<(String, Vec<u8>)> = Vec::new();
    for (i, f) in [3usize, 5, 7, 11].iter().enumerate() {
        let mut m = b.to_vec();
        let at = sos + (b.len() - sos) * f / 13;
        m[at] ^= 0x5a + i as u8;
        cases.push((format!("flip at {at}"), m));
    }
    for f in [2usize, 3, 4] {
        cases.push((format!("truncated 1/{f}"), b[..b.len() / f].to_vec()));
    }
    let mut m = b.to_vec();
    m[rst(2) + 1] = 0xd5; // out of sequence
    cases.push(("RST2 became RST5".into(), m));
    let mut m = b.to_vec();
    let at = rst(4);
    m.drain(at..at + 2); // a restart marker missing
    cases.push(("RST4 removed".into(), m));
    let mut m = b.to_vec();
    let at = rst(6);
    m[at - 40..at].fill(0); // a run of zeros before a marker
    cases.push(("zeros before RST6".into(), m));
    let mut m = b.to_vec();
    let n = m.len();
    m.truncate(n - 2); // no EOI
    cases.push(("no EOI".into(), m));
    cases
}

/// Bands that lose data or end off their RSTn make Go decode the scan again serially; so do
/// these: damage in the middle of a restart stream, in a marker, and at the end.
#[test]
fn damaged_restart_streams_match_serial() {
    let b = read("jpg/strip_rst.jpg");
    for (what, m) in damaged_cases(&b) {
        for format in [Format::Rgba, Format::Gray] {
            let o = opts(format, 1);
            let (want, _) = run(&m, &o, 1);
            for t in [2usize, 4, 8] {
                assert_eq!(run(&m, &o, t).0, want, "{what} {format:?} threads {t}");
            }
        }
    }
}

/// `decode_into_rgba` (its stripes run on the caller's window) equals the serial one, here
/// through the default size threshold: the strip is over it.
#[test]
fn window_decode_matches_serial() {
    let b = read("jpg/strip_rst.jpg");
    let o = opts(Format::Rgba, 1);
    let (w, h) = (1024usize, 1120usize);
    let stride = 4 * w + 8;
    let mut want = vec![0xaau8; stride * (h - 1) + 4 * w];
    let e1 = jpeg::decode_into_rgba(
        &b,
        &o,
        &limits(1),
        &go(),
        RgbaWindow {
            pix: &mut want,
            stride,
        },
    );
    for t in [2usize, 4] {
        let mut got = vec![0xaau8; want.len()];
        let e = jpeg::decode_into_rgba(
            &b,
            &o,
            &limits(t),
            &go(),
            RgbaWindow {
                pix: &mut got,
                stride,
            },
        );
        assert_eq!(e, e1);
        assert!(got == want, "window differs at threads {t}");
    }
}

/// The cancel flag stops a threaded decode with the same error.
#[test]
fn cancel_stops_every_thread() {
    let b = read("jpg/strip_rst.jpg");
    let stop = AtomicBool::new(true);
    for t in [1usize, 4] {
        let r = jpeg::decode_probe(&b, &opts(Format::Rgba, 1), &limits(t), &stop, true).0;
        assert_eq!(r, Err(Error::Cancelled), "threads {t}");
    }
}

/// `TestWorkersAtLimit`: the thread count keeps a decode's limit outcome. At the smallest
/// `max_alloc` that decodes serially the threaded decode gives the same image, and one byte
/// under it the same error.
#[test]
fn threads_at_the_limit() {
    for name in ["jpg/synth_restart_227x149.jpg", "jpg/ndpi_wide_4096x32.jpg"] {
        let b = read(name);
        let o = opts(Format::Rgba, 1);
        let with = |t: usize, max_alloc: u64| {
            let l = Limits {
                max_alloc,
                ..limits(t)
            };
            jpeg::decode_probe(&b, &o, &l, &go(), true).0
        };
        let (mut lo, mut hi) = (0u64, 256u64 << 20); // lo fails, hi decodes
        assert!(with(1, hi).is_ok());
        assert!(matches!(with(1, lo), Err(Error::AllocLimit { .. })));
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            if with(1, mid).is_ok() {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        for t in [2usize, 8, usize::MAX] {
            for lim in [hi, hi - 1, hi + (1 << 20), hi + (8 << 20)] {
                assert_eq!(with(t, lim), with(1, lim), "{name} limit {lim} threads {t}");
            }
        }
    }
}

fn load_starts(name: &str) -> Vec<usize> {
    String::from_utf8(read(name))
        .unwrap()
        .split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect()
}

/// The fixtures a region decode with restart offsets runs on: (fixture, offsets file, regions).
fn region_cases() -> Vec<(&'static str, &'static str, Vec<Rect>)> {
    let r = Rect::new;
    vec![
        (
            "jpg/strip_rst.jpg",
            "strip_rst.starts",
            vec![
                r(128, 64, 640, 576),   // on MCU boundaries
                r(137, 619, 803, 1027), // off them
                r(0, 100, 700, 900),    // touching the left edge
                r(300, 40, 1024, 700),  // the right edge
                r(200, 500, 900, 1120), // the bottom edge
                r(250, 300, 762, 812),  // the dec-region bench
                r(50, 80, 400, 90),     // about one MCU row
                r(50, 79, 400, 97),     // across an interval and a row
                r(0, 0, 1024, 1120),    // the whole frame
                r(0, 200, 1024, 900),   // full width: never banded
            ],
        ),
        (
            "jpg/ndpi_wide_4096x32.jpg",
            "ndpi_wide_4096x32.starts",
            vec![
                r(100, 0, 3000, 32),
                r(0, 5, 2500, 29),
                r(1500, 0, 4096, 20),
                r(700, 9, 3900, 32),
            ],
        ),
    ]
}

/// A band region decode of a stream with restart offsets: the image and the error are the
/// `max_threads` 1 result for every thread count, and a region narrower than the frame uses more
/// than one thread.
#[test]
fn region_restart_bands_match_serial() {
    for (name, starts_file, regions) in region_cases() {
        let b = read(name);
        let starts = load_starts(starts_file);
        for region in regions {
            for format in [Format::Rgba, Format::Gray, Format::YCbCr] {
                let o = DecodeOptions {
                    format,
                    region: Some(region),
                    restart: Some(&starts[..]),
                    ..Default::default()
                };
                let (want, p1) = run(&b, &o, 1);
                assert_eq!(p1.threads_peak, 1);
                for t in [2usize, 3, 4, 5, 16] {
                    let (got, p) = run(&b, &o, t);
                    assert_eq!(got, want, "{name} {region:?} {format:?} threads {t}");
                    assert!(p.threads_peak <= t);
                }
            }
        }
    }
    // Gray has no colour stripes, so a peak over 1 is the entropy bands; a full-width region
    // has none.
    let b = read("jpg/strip_rst.jpg");
    let starts = load_starts("strip_rst.starts");
    let go_gray = |region: Rect| {
        let o = DecodeOptions {
            format: Format::Gray,
            region: Some(region),
            restart: Some(&starts[..]),
            ..Default::default()
        };
        peak_of(&b, &o, 4, true)
    };
    let peak = go_gray(Rect::new(250, 300, 762, 812));
    assert!(peak > 1 && peak <= 4, "{peak}");
    assert_eq!(go_gray(Rect::new(0, 200, 1024, 900)), 1);
}

/// The restart offsets of a one-scan stream, found by scanning its entropy data for RSTn markers:
/// the first interval starts after the SOS header, each next one after a marker.
fn derived_starts(b: &[u8]) -> Vec<usize> {
    let sos = b.windows(2).position(|w| w == [0xff, 0xda]).unwrap();
    let start = sos + 2 + usize::from(u16::from_be_bytes([b[sos + 2], b[sos + 3]]));
    let mut v = vec![start];
    v.extend(
        b[start..]
            .windows(2)
            .enumerate()
            .filter(|(_, w)| w[0] == 0xff && (0xd0..=0xd7).contains(&w[1]))
            .map(|(i, _)| start + i + 2),
    );
    v
}

/// Region bands on a lone-component scan (`gray22_rst`: one 2x2-sampled component, so the scan
/// maps a block to `v` and `h` frame MCUs) and with subsampled chroma (4:2:0 colour stripes split
/// across threads): every thread count gives the serial result, and truthful offsets give the
/// same result as none.
#[test]
fn region_restart_bands_lone_component_and_subsampled_chroma() {
    let r = Rect::new;
    for (name, regions) in [
        (
            "jpg/gray22_rst.jpg",
            vec![r(40, 20, 180, 120), r(13, 7, 150, 140), r(100, 0, 227, 149)],
        ),
        (
            "jpg/synth_restart_227x149.jpg",
            vec![r(40, 20, 180, 120), r(13, 7, 150, 140), r(100, 0, 227, 149)],
        ),
        (
            "jpg/synth_restart_203x157.jpg",
            vec![r(33, 17, 170, 130), r(0, 40, 120, 157)],
        ),
    ] {
        let b = read(name);
        let starts = derived_starts(&b);
        for region in regions {
            for format in [Format::Rgba, Format::Gray, Format::YCbCr] {
                let o = DecodeOptions {
                    format,
                    region: Some(region),
                    restart: Some(&starts[..]),
                    ..Default::default()
                };
                let plain = DecodeOptions { restart: None, ..o };
                let (want, _) = run(&b, &o, 1);
                assert_eq!(want, run(&b, &plain, 1).0, "{name} {region:?} {format:?}");
                for t in [2usize, 3, 5, 16] {
                    let (got, p) = run(&b, &o, t);
                    assert_eq!(got, want, "{name} {region:?} {format:?} threads {t}");
                    assert!(p.threads_peak <= t);
                }
            }
        }
    }
}

/// Offsets that lie (shifted by 8 intervals so each still follows the right RSTn, zeroed, out of
/// order, too few) and damaged streams with the true offsets: a band region decode gives the
/// serial result.
#[test]
fn region_restart_bands_with_bad_offsets_match_serial() {
    let b = read("jpg/strip_rst.jpg");
    let good = load_starts("strip_rst.starts");
    let n = good.len();
    let mut lists: Vec<(&str, Vec<usize>)> = vec![
        (
            "shifted by 8",
            (0..n).map(|k| good[(k + 8).min(n - 1)]).collect(),
        ),
        ("zeroed", vec![0; n]),
        (
            "every third zeroed",
            (0..n)
                .map(|k| if k % 3 == 1 { 0 } else { good[k] })
                .collect(),
        ),
        ("reversed", good.iter().rev().copied().collect()),
        ("too few", good[..n / 2].to_vec()),
        (
            "past the end",
            good.iter().map(|&o| o + (1 << 20)).collect(),
        ),
    ];
    let mut swapped = good.clone();
    for k in (1..n - 1).step_by(7) {
        swapped.swap(k, k + 1);
    }
    lists.push(("swapped pairs", swapped));
    let region = Rect::new(250, 300, 762, 812);
    for (what, starts) in &lists {
        for format in [Format::Rgba, Format::Gray] {
            let o = DecodeOptions {
                format,
                region: Some(region),
                restart: Some(&starts[..]),
                ..Default::default()
            };
            let want = run(&b, &o, 1).0;
            for t in [2usize, 4, 16] {
                assert_eq!(
                    run(&b, &o, t).0,
                    want,
                    "offsets {what} {format:?} threads {t}"
                );
            }
        }
    }
    for (what, m) in damaged_cases(&b) {
        for format in [Format::Rgba, Format::Gray] {
            let o = DecodeOptions {
                format,
                region: Some(region),
                restart: Some(&good[..]),
                ..Default::default()
            };
            let want = run(&m, &o, 1).0;
            for t in [2usize, 4, 16] {
                assert_eq!(run(&m, &o, t).0, want, "{what} {format:?} threads {t}");
            }
        }
    }
}
