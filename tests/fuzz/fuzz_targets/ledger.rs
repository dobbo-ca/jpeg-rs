#![no_main]
use jpeg::{DecodeOptions, EncodeImage, EncodeOptions, Format, Limits, RgbaWindow, RowReader};
use libfuzzer_sys::fuzz_target;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

#[path = "../common.rs"]
mod common;
use common::*;

/// Counts live heap bytes and their peak, as jpeg-go's `FuzzLedger` counts allocation.
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

/// What a call may hold beyond `max_alloc`: fixed state the ledger does not count (tables and
/// frame state under 64 KiB each, the output vector's spare capacity, the fuzzer's own).
const SLACK: usize = 1 << 20;

/// What an encode may hold beyond `max_alloc`: the stream header and the fixed tables, under
/// 64 KiB. Tighter than `SLACK`, so an output vector that grows past the budget is caught.
const ENCODE_SLACK: usize = 64 << 10;

/// Runs `f` and asserts its peak live heap, above what was live before, is at most `max_alloc`
/// plus `SLACK`.
fn bounded<R>(what: &str, max_alloc: u64, f: impl FnOnce() -> R) -> R {
    bounded_by(what, max_alloc, SLACK, f)
}

/// `bounded` with the slack given.
fn bounded_by<R>(what: &str, max_alloc: u64, slack: usize, f: impl FnOnce() -> R) -> R {
    let base = LIVE.load(Relaxed);
    PEAK.store(base, Relaxed);
    let r = f();
    let peak = PEAK.load(Relaxed).saturating_sub(base);
    assert!(
        peak as u64 <= max_alloc.saturating_add(slack as u64),
        "{what}: peak {peak} bytes over max_alloc {max_alloc}"
    );
    r
}

// data = [max_alloc: u16 LE][shift][threads][the option bytes of `decode`, then a stream].
// max_alloc is the u16 shifted left by 0 to 11 bits (up to 128 MiB); max_threads is 1 to 8. Every
// entry point (decode, decode into a caller's buffer, the coefficients, the row reader, encode)
// runs under them, and none holds more than max_alloc plus a small slack at any moment, nor
// charges more than max_alloc; `decode` threads every frame it can.
fuzz_target!(|src: &[u8]| {
    let Some((&[a, b, shift, threads], rest)) = src.split_first_chunk::<4>() else {
        return;
    };
    let Some(inp) = parse(rest) else { return };
    let max_alloc = u64::from(u16::from_le_bytes([a, b])) << (shift % 12);
    let l = Limits {
        max_pixels: 1 << 22,
        max_alloc,
        max_threads: 1 + usize::from(threads % 8),
    };
    let o = DecodeOptions {
        format: inp.format(),
        scale: inp.scale(),
        tables: inp.tables,
        ..Default::default()
    };
    let probe = bounded("decode", max_alloc, || {
        jpeg::decode_probe(inp.data, &o, &l, &NO, true).1
    });
    assert!(
        probe.charged_peak <= max_alloc && probe.threads_peak <= l.max_threads,
        "decode: charged {} of {max_alloc}, {} of {} threads",
        probe.charged_peak,
        probe.threads_peak,
        l.max_threads
    );
    bounded("decode_coefficients", max_alloc, || {
        jpeg::decode_coefficients(inp.data, inp.tables, &l, &NO)
    });
    bounded("decode_coefficients_any_components", max_alloc, || {
        jpeg::decode_coefficients_any_components(inp.data, inp.tables, &l, &NO)
    });
    bounded("row_reader", max_alloc, || {
        if let Ok(mut r) = RowReader::new(inp.data, inp.tables, &l, &NO) {
            while let Ok(Some(_)) = r.next(&NO) {}
        }
    });
    if let Ok(c) = jpeg::decode_config(inp.data, inp.tables, &l)
        && c.width * c.height <= 1 << 22
    {
        // The caller's buffer is its own, so it is made outside the measured call.
        let mut pix = vec![0u8; 4 * c.width * c.height];
        let rgba = DecodeOptions {
            format: Format::Rgba,
            scale: 1,
            ..o
        };
        bounded("decode_into_rgba", max_alloc, || {
            jpeg::decode_into_rgba(
                inp.data,
                &rgba,
                &l,
                &NO,
                RgbaWindow {
                    pix: &mut pix,
                    stride: 4 * c.width,
                },
            )
        });
    }
    // Encode: an image from the stream's bytes, one-shot and through one reused `Encoder` (twice:
    // a solid image's second encode is a cache hit), at both chroma ratios, every kind (a YCbCr
    // that is not 4:4:4 or 4:2:0 goes through the RGBA conversion), under a tight slack.
    if !inp.data.is_empty() {
        let (w, h) = (1 + usize::from(inp.b[2]), 1 + usize::from(inp.b[3]));
        let pix = build(inp.b[0] % 8, w, h, 0, 0, 0, inp.data);
        let img = match &pix {
            Pixels::Rgba(m) => EncodeImage::Rgba(m),
            Pixels::Gray(m) => EncodeImage::Gray(m),
            Pixels::YCbCr(m) => EncodeImage::YCbCr(m),
        };
        let mut enc = jpeg::Encoder::default();
        for chroma444 in [false, true] {
            let eo = EncodeOptions {
                quality: i32::from(inp.b[1]) % 101,
                chroma444,
            };
            let want = bounded_by("encode", max_alloc, ENCODE_SLACK, || {
                jpeg::encode(img, &eo, &l, &NO)
            });
            for _ in 0..2 {
                let got = bounded_by("Encoder::encode", max_alloc, ENCODE_SLACK, || {
                    enc.encode(img, &eo, &l, &NO)
                });
                assert_eq!(got, want, "a reused Encoder differs from a fresh encode");
            }
        }
    }
});
