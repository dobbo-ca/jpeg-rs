//! Buffer reuse (docs/performance.md): a `Decoder` that has decoded a frame once decodes
//! it again, at `max_threads` 1, without one call to the allocator. This binary counts every
//! allocation (`alloc`, `alloc_zeroed`, `realloc`), so it holds exactly one test: another test
//! running beside it would be counted too. A threaded call on a caller's pool keeps a few
//! allocations that are not the decoder's buffers (the job lists, and one boxed task per extra
//! worker, which rayon's `spawn` allocates); its count is bounded, not zero.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

mod common;
use common::*;
use jpeg::{
    DecodeOptions, Decoded, Decoder, Error, Format, Image, Limits, Rect, RgbaWindow, ThreadPool,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};

/// Counts the calls that ask the allocator for memory.
struct Counting;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

/// How many allocator calls of a measured window `LOG` keeps.
const LOG_LEN: usize = 16;
/// `ALLOCS` when the window opened.
static BASE: AtomicUsize = AtomicUsize::new(0);
/// The first `LOG_LEN` calls of the window as `(size << 2) | kind`; atomics, so the allocator
/// itself never allocates to record them.
static LOG: [AtomicUsize; LOG_LEN] = [const { AtomicUsize::new(0) }; LOG_LEN];
const KINDS: [&str; 3] = ["alloc", "alloc_zeroed", "realloc"];

/// Counts one allocator call of `size` bytes (`kind` indexes `KINDS`) and files it in `LOG`.
fn note(size: usize, kind: usize) {
    let k = ALLOCS
        .fetch_add(1, Relaxed)
        .wrapping_sub(BASE.load(Relaxed));
    if k < LOG_LEN {
        LOG[k].store((size << 2) | kind, Relaxed);
    }
}

/// Opens a measured window and returns the count at its start.
fn open_window() -> usize {
    let now = ALLOCS.load(Relaxed);
    BASE.store(now, Relaxed);
    now
}

/// The first allocator calls of the window that just closed with `n` of them, by kind and size.
fn first_allocs(n: usize) -> String {
    (0..n.min(LOG_LEN))
        .map(|k| {
            let v = LOG[k].load(Relaxed);
            format!("{} {}", KINDS[v & 3], v >> 2)
        })
        .collect::<Vec<_>>()
        .join(", ")
}

// SAFETY: every method forwards to `System` with the caller's arguments, so System's contract
// carries over; the counter is an atomic.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        note(l.size(), 0);
        // SAFETY: the caller upholds `alloc`'s contract, which `System.alloc` has too.
        unsafe { System.alloc(l) }
    }

    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        note(l.size(), 1);
        // SAFETY: as for `alloc`.
        unsafe { System.alloc_zeroed(l) }
    }

    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: `p` came from this allocator with layout `l`, so from System with it.
        unsafe { System.dealloc(p, l) };
    }

    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        note(new, 2);
        // SAFETY: `p` came from this allocator with layout `l`, and `new` is valid for it.
        unsafe { System.realloc(p, l, new) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// The most allocator calls of one threaded call on a caller's pool. A restart-band decode makes
/// about a dozen (the band starts and cuts, the job list, `par::run`'s queue, results and one
/// boxed rayon task per extra worker) and the colour stripes about as many.
const THREADED_MAX: usize = 48;

fn limits(max_threads: usize) -> Limits {
    Limits {
        max_threads,
        ..Limits::default()
    }
}

/// One warmed-up call of `dec`: the allocator calls of its second decode of `src` (with the
/// first of them named by kind and size), and whether its result is the one-shot decode's.
/// `window` decodes through `decode_into_rgba` into a buffer made before the count starts.
fn second_call(
    dec: &mut Decoder,
    src: &[u8],
    o: &DecodeOptions,
    l: &Limits,
    window: bool,
) -> Result<(usize, bool, String), Error> {
    let cancel = AtomicBool::new(false);
    let want = jpeg::decode(src, o, l, &cancel)?;
    if window {
        let r = want.image.bounds();
        let (w, h) = (r.width() as usize, r.height() as usize);
        let mut dst = vec![0u8; 4 * w * h];
        let mut call = |dst: &mut [u8]| {
            dec.decode_into_rgba(
                src,
                o,
                l,
                &cancel,
                RgbaWindow {
                    pix: dst,
                    stride: 4 * w,
                },
            )
        };
        call(&mut dst)?;
        let before = open_window();
        let r = call(&mut dst);
        let n = ALLOCS.load(Relaxed) - before;
        r?;
        let same = matches!(&want.image, Image::Rgba(g) if g.pix == dst);
        return Ok((n, same, first_allocs(n)));
    }
    dec.decode(src, o, l, &cancel)?;
    let before = open_window();
    let r = dec.decode(src, o, l, &cancel);
    let n = ALLOCS.load(Relaxed) - before;
    let got: &Decoded = r?;
    Ok((n, got == &want, first_allocs(n)))
}

#[test]
fn second_decode_allocates_nothing() {
    let mut rows: Vec<(String, usize)> = Vec::new();
    let mut bad: Vec<String> = Vec::new();
    let mut record = |what: String, r: Result<(usize, bool, String), Error>, max: usize| match r {
        Ok((n, same, first)) => {
            if n > max || !same {
                bad.push(format!(
                    "{what}: {n} allocations (max {max}), same output {same}, first: [{first}]"
                ));
            }
            rows.push((what, n));
        }
        Err(e) => bad.push(format!("{what}: {e:?}")),
    };

    // A 4:2:0 SVS-like tile, a gray file, a restart file and a progressive file.
    let files: Vec<(&str, Vec<u8>)> = [
        "jpg/svs_420_512x512.jpg",
        "jpg/cmp_gray.jpg",
        "jpg/strip_rst.jpg",
        "jpg/cmp_prog.jpg",
    ]
    .into_iter()
    .map(|n| (n, read(n)))
    .collect();

    // Serial: no allocation at all, on a fresh Decoder warmed by one call.
    for (name, src) in &files {
        let cfg = jpeg::decode_config(src, None, &Limits::default()).unwrap();
        let (w, h) = (cfg.width as i32, cfg.height as i32);
        let inner = Rect::new(w / 4, h / 4, w - w / 4, h - h / 4);
        for format in [Format::Rgba, Format::Gray, Format::YCbCr] {
            for scale in [1u8, 2, 4, 8] {
                let o = opts(format, scale);
                let what = format!("{name} {format:?} scale {scale} decode");
                record(
                    what,
                    second_call(&mut Decoder::new(), src, &o, &limits(1), false),
                    0,
                );
            }
            let o = DecodeOptions {
                region: Some(inner),
                ..opts(format, 1)
            };
            let what = format!("{name} {format:?} region {inner:?} decode");
            record(
                what,
                second_call(&mut Decoder::new(), src, &o, &limits(1), false),
                0,
            );
        }
        // The pyramid mode: its chroma-prefilter row comes from the reused scratch.
        for scale in [2u8, 4, 8] {
            let o = DecodeOptions {
                pyramid: true,
                ..opts(Format::Rgba, scale)
            };
            let what = format!("{name} Rgba pyramid scale {scale} decode");
            record(
                what,
                second_call(&mut Decoder::new(), src, &o, &limits(1), false),
                0,
            );
        }
        let o = opts(Format::Rgba, 1);
        let what = format!("{name} Rgba decode_into_rgba");
        record(
            what,
            second_call(&mut Decoder::new(), src, &o, &limits(1), true),
            0,
        );
    }

    // A region of the NDPI-style strip with its restart offsets (the slide viewer's tile call).
    let starts: Vec<usize> = String::from_utf8(read("strip_rst.starts"))
        .unwrap()
        .split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect();
    let strip = read("jpg/strip_rst.jpg");
    let tile = |format| DecodeOptions {
        region: Some(Rect::new(250, 300, 762, 812)),
        restart: Some(&starts[..]),
        ..opts(format, 1)
    };
    for format in [Format::Rgba, Format::Gray, Format::YCbCr] {
        let what = format!("strip_rst {format:?} restart region decode");
        record(
            what,
            second_call(
                &mut Decoder::new(),
                &strip,
                &tile(format),
                &limits(1),
                false,
            ),
            0,
        );
    }

    // Threaded, on a caller's pool: a small fixed number of allocations, not zero.
    let pool = ThreadPool::new(3).unwrap();
    for (name, src) in &files {
        for format in [Format::Rgba, Format::Gray, Format::YCbCr] {
            let o = opts(format, 1);
            let what = format!("{name} {format:?} decode, max_threads 4 on a pool");
            let mut dec = Decoder::with_pool(pool.clone());
            record(
                what,
                second_call(&mut dec, src, &o, &limits(4), false),
                THREADED_MAX,
            );
        }
        let o = opts(Format::Rgba, 1);
        let what = format!("{name} Rgba decode_into_rgba, max_threads 4 on a pool");
        let mut dec = Decoder::with_pool(pool.clone());
        record(
            what,
            second_call(&mut dec, src, &o, &limits(4), true),
            THREADED_MAX,
        );
    }

    // The restart-region call threaded (restart bands, then colour stripes): bounded, not zero.
    for format in [Format::Rgba, Format::Gray, Format::YCbCr] {
        let what = format!("strip_rst {format:?} restart region decode, max_threads 4 on a pool");
        let mut dec = Decoder::with_pool(pool.clone());
        record(
            what,
            second_call(&mut dec, &strip, &tile(format), &limits(4), false),
            THREADED_MAX,
        );
    }

    for (what, n) in &rows {
        println!("{n:4} allocations  {what}");
    }
    assert!(bad.is_empty(), "\n{}\n", bad.join("\n"));
}
