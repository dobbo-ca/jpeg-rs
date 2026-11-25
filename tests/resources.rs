//! Resource controls (docs/limits.md): the heap peak of a call stays under `max_alloc` and
//! under what its ledger charged, and its threads under `max_threads`, for several of each,
//! `max_threads` 1 included. This binary counts every allocation, so it holds exactly one test:
//! another test running beside it would be counted too.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

mod common;
use common::*;
use jpeg::{
    DecodeOptions, Decoded, Decoder, EncodeImage, EncodeOptions, Encoder, Error, Format, Gray,
    Limits, Rect, Rgba, Subsample, YCbCr,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};

/// Counts live heap bytes and their peak.
struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grow(n: usize) {
    let now = LIVE.fetch_add(n, Relaxed) + n;
    PEAK.fetch_max(now, Relaxed);
}

// SAFETY: every method forwards to `System` with the caller's arguments, so System's contract
// carries over; the counters are atomics.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        // SAFETY: the caller upholds `alloc`'s contract, which `System.alloc` has too.
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            grow(l.size());
        }
        p
    }

    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        // SAFETY: as for `alloc`.
        let p = unsafe { System.alloc_zeroed(l) };
        if !p.is_null() {
            grow(l.size());
        }
        p
    }

    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: `p` came from this allocator with layout `l`, so from System with it.
        unsafe { System.dealloc(p, l) };
        LIVE.fetch_sub(l.size(), Relaxed);
    }

    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        // SAFETY: `p` came from this allocator with layout `l`, and `new` is valid for it.
        let q = unsafe { System.realloc(p, l, new) };
        if !q.is_null() {
            if new >= l.size() {
                grow(new - l.size());
            } else {
                LIVE.fetch_sub(l.size() - new, Relaxed);
            }
        }
        q
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// What a call may hold beyond its ledger: fixed state the ledger does not count (tables and
/// frame state under 64 KiB each), the spare capacity of vectors, and what spawning a thread
/// allocates.
const SLACK: u64 = 1 << 20;

/// One decode on `t` threads under `max_alloc`, threading every frame that can be cut; returns
/// the result, the heap peak above what was live before, and the probe.
fn measured(
    b: &[u8],
    format: Format,
    t: usize,
    max_alloc: u64,
) -> (Result<Decoded, Error>, u64, jpeg::Probe) {
    measured_with(b, &opts(format, 1), t, max_alloc)
}

/// `measured` with the options given.
fn measured_with(
    b: &[u8],
    o: &DecodeOptions,
    t: usize,
    max_alloc: u64,
) -> (Result<Decoded, Error>, u64, jpeg::Probe) {
    let l = Limits {
        max_alloc,
        max_threads: t,
        ..Limits::default()
    };
    let cancel = AtomicBool::new(false);
    let base = LIVE.load(Relaxed);
    PEAK.store(base, Relaxed);
    let (r, probe) = jpeg::decode_probe(b, o, &l, &cancel, true);
    let heap = PEAK.load(Relaxed).saturating_sub(base) as u64;
    (r, heap, probe)
}

/// The caps and thread counts of `peaks_stay_under_max_alloc_and_max_threads`, on one call with
/// options `o` (a region with restart offsets, the pyramid mode): the heap stays under the cap and
/// the charged peak, the charged peak under the cap, the threads under `max_threads`, and every
/// thread count gives the serial result. A `Decoder` warmed by one call then decodes under a cap
/// at its charged peak with the one-shot result (or `AllocLimit`) and the heap bound.
fn peak_under_caps(what: &str, b: &[u8], o: &DecodeOptions) {
    let default_threads = Limits::default().max_threads;
    let (want, _, p1) = measured_with(b, o, 1, 1 << 30);
    assert!(want.is_ok(), "{what}");
    let charged = p1.charged_peak;
    let caps = [
        1u64 << 30,
        charged + (1 << 20),
        charged,
        charged / 2,
        1 << 20,
    ];
    for t in [1usize, 2, default_threads] {
        for max_alloc in caps {
            let what = format!("{what} threads {t} max_alloc {max_alloc}");
            let (r, heap, p) = measured_with(b, o, t, max_alloc);
            if max_alloc != 1 << 30 {
                assert!(
                    heap <= max_alloc + SLACK,
                    "{what}: heap peak {heap} over max_alloc"
                );
                assert!(
                    p.charged_peak <= max_alloc,
                    "{what}: charged over max_alloc"
                );
            }
            assert!(
                heap <= p.charged_peak + SLACK,
                "{what}: heap peak {heap} over the charged peak {}",
                p.charged_peak
            );
            assert!(
                p.threads_peak <= t.max(1),
                "{what}: threads {}",
                p.threads_peak
            );
            if t == 1 {
                assert_eq!(p.threads_peak, 1, "{what}");
            }
            assert_eq!(r, measured_with(b, o, 1, max_alloc).0, "{what}");
        }
    }
    // A reused Decoder, warmed by one call.
    let mut dec = Decoder::new();
    let cancel = AtomicBool::new(false);
    dec.decode(b, o, &Limits::default(), &cancel).unwrap();
    for max_alloc in [charged, charged + (1 << 20)] {
        let l = Limits {
            max_alloc,
            max_threads: 1,
            ..Limits::default()
        };
        let base = LIVE.load(Relaxed);
        PEAK.store(base, Relaxed);
        let r = dec.decode(b, o, &l, &cancel).cloned();
        let heap = PEAK.load(Relaxed).saturating_sub(base) as u64;
        assert!(
            heap <= max_alloc + SLACK,
            "{what} reused, max_alloc {max_alloc}: heap peak {heap}"
        );
        // Reused buffers are charged too, so a tight cap may refuse; an answer is the one-shot one.
        match r {
            Ok(_) => assert_eq!(r, want, "{what} reused, max_alloc {max_alloc}"),
            Err(Error::AllocLimit { .. }) => {}
            Err(e) => panic!("{what} reused, max_alloc {max_alloc}: {e:?}"),
        }
    }
}

/// What an encode may hold beyond `max_alloc`: the stream header and the fixed tables, under
/// 64 KiB. Tighter than `SLACK`, so an output vector that grows past the budget is caught.
const ENCODE_SLACK: u64 = 64 << 10;

/// One encode under `max_alloc`, through `enc` when given; returns the result and the heap peak
/// above what was live before.
fn measured_encode(
    enc: Option<&mut Encoder>,
    m: EncodeImage,
    o: &EncodeOptions,
    max_alloc: u64,
) -> (Result<Vec<u8>, Error>, u64) {
    let l = Limits {
        max_alloc,
        max_threads: 1,
        ..Limits::default()
    };
    let cancel = AtomicBool::new(false);
    let base = LIVE.load(Relaxed);
    PEAK.store(base, Relaxed);
    let r = match enc {
        Some(e) => e.encode(m, o, &l, &cancel),
        None => jpeg::encode(m, o, &l, &cancel),
    };
    (r, PEAK.load(Relaxed).saturating_sub(base) as u64)
}

/// The encoder's output vector counts against `max_alloc` by its capacity: for Gray, RGBA at both
/// chroma ratios, a YCbCr converted through `draw_rgba` and a solid tile (one-shot, and through a
/// reused `Encoder` whose second encode is a cache hit), the heap peak stays under `max_alloc`
/// and the result is the unlimited one or `AllocLimit`.
fn encode_peaks_stay_under_max_alloc() {
    let mut seed = 99u32;
    let mut noise = |n: usize| -> Vec<u8> {
        (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (seed >> 24) as u8
            })
            .collect()
    };
    let gray = Gray {
        pix: noise(640 * 480),
        stride: 640,
        rect: Rect::new(0, 0, 640, 480),
    };
    let rgba = Rgba {
        pix: noise(320 * 240 * 4),
        stride: 320 * 4,
        rect: Rect::new(0, 0, 320, 240),
    };
    // 4:2:2 is not a direct input of the encoder: it goes through `draw_rgba`.
    let ycc = YCbCr {
        y: noise(320 * 240),
        cb: noise(160 * 240),
        cr: noise(160 * 240),
        y_stride: 320,
        c_stride: 160,
        ratio: Subsample::S422,
        rect: Rect::new(0, 0, 320, 240),
    };
    let solid = Gray {
        pix: vec![200; 64 * 64],
        stride: 64,
        rect: Rect::new(0, 0, 64, 64),
    };
    let images = [
        ("gray", EncodeImage::Gray(&gray)),
        ("rgba", EncodeImage::Rgba(&rgba)),
        ("ycbcr 4:2:2 via draw_rgba", EncodeImage::YCbCr(&ycc)),
        ("solid gray", EncodeImage::Gray(&solid)),
    ];
    for (name, img) in images {
        for chroma444 in [false, true] {
            let o = EncodeOptions {
                quality: 100,
                chroma444,
            };
            let want = measured_encode(None, img, &o, 1 << 30).0.unwrap();
            let n = want.len() as u64;
            let caps = [
                1u64 << 30,
                8 * n,
                3 * n,
                2 * n,
                n + (64 << 10),
                n,
                n / 2,
                1 << 12,
            ];
            for max_alloc in caps {
                let mut reused = Encoder::default();
                for reuse in [false, true] {
                    for call in 0..2 {
                        let what = format!(
                            "{name} 444 {chroma444} reuse {reuse} call {call} max_alloc {max_alloc}"
                        );
                        let enc = reuse.then_some(&mut reused);
                        let (r, heap) = measured_encode(enc, img, &o, max_alloc);
                        assert!(
                            heap <= max_alloc + ENCODE_SLACK,
                            "{what}: heap peak {heap} over max_alloc"
                        );
                        match r {
                            Ok(got) => assert!(got == want, "{what}: bytes differ"),
                            Err(Error::AllocLimit { .. }) => {
                                assert!(max_alloc != 1 << 30, "{what}: refused")
                            }
                            Err(e) => panic!("{what}: {e:?}"),
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn peaks_stay_under_max_alloc_and_max_threads() {
    encode_peaks_stay_under_max_alloc();
    let default_threads = Limits::default().max_threads;
    for name in [
        "jpg/strip_rst.jpg",
        "jpg/synth_restart_227x149.jpg",
        "jpg/ndpi_wide_4096x32.jpg",
        "jpg/svs_420_512x512.jpg",
    ] {
        let b = read(name);
        for format in [Format::Rgba, Format::Gray] {
            // What one serial decode charges, with no cap: the yardstick for the caps below.
            let (want, _, p1) = measured(&b, format, 1, 1 << 30);
            let charged = p1.charged_peak;
            assert!(want.is_ok(), "{name}");
            let caps = [
                1u64 << 30,
                64 << 20,
                charged + (1 << 20),
                charged + (8 << 20),
                charged,
                charged / 2,
                1 << 20,
            ];
            for t in [1usize, 2, default_threads] {
                for max_alloc in caps {
                    let what = format!("{name} {format:?} threads {t} max_alloc {max_alloc}");
                    let (r, heap, p) = measured(&b, format, t, max_alloc);
                    // Whatever the memory, a call cannot go above the cap or what it charged.
                    if max_alloc != 1 << 30 {
                        assert!(
                            heap <= max_alloc + SLACK,
                            "{what}: heap peak {heap} over max_alloc"
                        );
                        assert!(
                            p.charged_peak <= max_alloc,
                            "{what}: charged over max_alloc"
                        );
                    }
                    assert!(
                        heap <= p.charged_peak + SLACK,
                        "{what}: heap peak {heap} over the charged peak {}",
                        p.charged_peak
                    );
                    // Nor go above the threads, and 1 never spawns.
                    assert!(
                        p.threads_peak <= t.max(1),
                        "{what}: threads {}",
                        p.threads_peak
                    );
                    if t == 1 {
                        assert_eq!(p.threads_peak, 1, "{what}");
                    }
                    // The answer is the serial one, threads or not.
                    assert_eq!(r, measured(&b, format, 1, max_alloc).0, "{what}");
                }
            }
        }
    }
    // A region of a restart stream with its offsets (restart bands, colour stripes), the pyramid
    // mode and a reused Decoder.
    let starts: Vec<usize> = String::from_utf8(read("strip_rst.starts"))
        .unwrap()
        .split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect();
    let strip = read("jpg/strip_rst.jpg");
    for format in [Format::Rgba, Format::Gray, Format::YCbCr] {
        let o = DecodeOptions {
            region: Some(Rect::new(250, 300, 762, 812)),
            restart: Some(&starts[..]),
            ..opts(format, 1)
        };
        peak_under_caps(&format!("strip_rst {format:?} restart region"), &strip, &o);
    }
    let svs = read("jpg/svs_420_512x512.jpg");
    for scale in [2u8, 4] {
        let o = DecodeOptions {
            pyramid: true,
            ..opts(Format::Rgba, scale)
        };
        peak_under_caps(&format!("svs_420 pyramid 1/{scale}"), &svs, &o);
    }
    // With room to spare, a restart stream does use its threads.
    let b = read("jpg/strip_rst.jpg");
    // A pool task that wakes after the caller took every job is not a worker the call used, so
    // allow a few tries.
    let mut best = 0;
    for _ in 0..30 {
        let (r, _, p) = measured(&b, Format::Rgba, 2, 1 << 30);
        assert!(r.is_ok());
        best = best.max(p.threads_peak);
        if best == 2 {
            break;
        }
    }
    assert_eq!(best, 2);
}
