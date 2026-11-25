//! Which threads a call runs on (docs/threading.md): `max_threads` 1 never starts the crate's
//! shared pool, a caller's pool is used instead of it, and a call that is worth threading starts
//! it. The shared pool is process state, so this binary holds one test and runs the steps in order.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

mod common;
use common::*;
use jpeg::{DecodeOptions, Decoder, Format, Limits, RgbaWindow, ThreadPool, shared_pool_started};
use std::sync::atomic::AtomicBool;

fn limits(max_threads: usize) -> Limits {
    Limits {
        max_threads,
        ..Limits::default()
    }
}

/// The most workers of a decode in up to 30 tries (a pool task that wakes after the caller took
/// every job is not a worker the call used).
fn peak(b: &[u8], o: &DecodeOptions, t: usize) -> usize {
    let cancel = AtomicBool::new(false);
    let mut best = 1;
    for _ in 0..30 {
        best = best.max(
            jpeg::decode_probe(b, o, &limits(t), &cancel, true)
                .1
                .threads_peak,
        );
        if best > 1 {
            break;
        }
    }
    best
}

#[test]
fn pools() {
    let b = read("jpg/strip_rst.jpg");
    let o = opts(Format::Rgba, 1);
    let cancel = AtomicBool::new(false);
    let want = jpeg::decode(&b, &o, &limits(1), &cancel).unwrap();

    // max_threads 1 never starts the shared pool, whatever the input.
    assert!(!shared_pool_started());
    assert_eq!(peak(&b, &o, 1), 1);
    assert!(!shared_pool_started());

    // A caller's pool is used instead of the shared one: through the options, and through a
    // Decoder that holds it.
    let pool = ThreadPool::new(3).unwrap();
    assert_eq!(pool.threads(), 3);
    let own = DecodeOptions {
        pool: Some(&pool),
        ..o
    };
    assert_eq!(jpeg::decode(&b, &own, &limits(4), &cancel).unwrap(), want);
    assert!(peak(&b, &own, 4) > 1);
    let mut d = Decoder::with_pool(pool.clone());
    assert_eq!(d.decode(&b, &o, &limits(4), &cancel).unwrap(), &want);
    let r = want.image.bounds();
    let (w, h) = (r.width() as usize, r.height() as usize);
    let mut dst = vec![0u8; 4 * w * h];
    let win = RgbaWindow {
        pix: &mut dst,
        stride: 4 * w,
    };
    assert_eq!(
        d.decode_into_rgba(&b, &o, &limits(4), &cancel, win)
            .unwrap(),
        want.error
    );
    assert!(!shared_pool_started(), "a caller's pool must win");

    // A pool of 1 thread gives a call two workers at most (itself and that one).
    let one = ThreadPool::new(1).unwrap();
    let o1 = DecodeOptions {
        pool: Some(&one),
        ..o
    };
    assert!(peak(&b, &o1, 8) <= 2);
    assert_eq!(jpeg::decode(&b, &o1, &limits(8), &cancel).unwrap(), want);
    assert!(!shared_pool_started());

    // No pool of the caller's and max_threads above 1: the shared pool starts, once.
    assert!(peak(&b, &o, 4) > 1);
    assert!(shared_pool_started());
    assert_eq!(jpeg::decode(&b, &o, &limits(4), &cancel).unwrap(), want);
}
