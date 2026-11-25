#![no_main]
use jpeg::{DecodeOptions, Decoder, Format, Limits, Rect};
use libfuzzer_sys::fuzz_target;

#[path = "../common.rs"]
mod common;
use common::{NO, limits, parse, same_samples};

// The pyramid mode (`DecodeOptions::pyramid`) on the front-of-input options of `common::parse`:
// byte 0 picks Rgba or Gray, byte 1 the scale (2, 4, 8, 2), byte 2 bits 1-2 the threads (2 to 5).
// An `Ok` result is floor-sized; the threaded decode and a reused (and first dirtied) `Decoder` equal the one-shot
// decode at 1 thread; and a gray or RGB stream, or any stream decoded to Gray (no chroma blur), equals the plain scaled decode.
fuzz_target!(|src: &[u8]| {
    let Some(inp) = parse(src) else { return };
    let l = limits();
    let scale = [2u8, 4, 8, 2][usize::from(inp.b[1] & 3)];
    let s = usize::from(scale);
    let o = DecodeOptions {
        format: [Format::Rgba, Format::Gray][usize::from(inp.b[0] & 1)],
        scale,
        tables: inp.tables,
        pyramid: true,
        ..Default::default()
    };
    let want = jpeg::decode(inp.data, &o, &l, &NO);

    if let (Ok(d), Ok(c)) = (&want, jpeg::decode_config(inp.data, inp.tables, &l)) {
        let floor = Rect::new(0, 0, (c.width / s) as i32, (c.height / s) as i32);
        assert_eq!(d.image.bounds(), floor, "not floor-sized");
        if c.components == 1 || c.rgb || o.format == Format::Gray {
            let plain = DecodeOptions {
                pyramid: false,
                ..o
            };
            let p = jpeg::decode(inp.data, &plain, &l, &NO)
                .expect("plain decode of a pyramid-ok frame");
            assert_eq!(p.error, d.error, "plain decode error differs");
            assert!(
                same_samples(&p.image, &d.image, floor),
                "not a crop of the plain decode"
            );
        }
    }

    let threads = 2 + usize::from((inp.b[2] >> 1) & 3);
    let ln = Limits {
        max_threads: threads,
        ..l
    };
    assert_eq!(
        jpeg::decode_probe(inp.data, &o, &ln, &NO, true).0,
        want,
        "{threads} threads differ from 1"
    );
    // A Decoder dirtied by a plain decode of the other format and a smaller scale (its pooled
    // planes and rows hold old bytes), then the pyramid decode twice: equal to the one-shot decode.
    let mut dec = Decoder::new();
    let _ = dec.decode(
        inp.data,
        &DecodeOptions {
            format: [Format::Gray, Format::Rgba][usize::from(inp.b[0] & 1)],
            scale: if scale == 2 { 1 } else { 2 },
            pyramid: false,
            ..o
        },
        &ln,
        &NO,
    );
    for _ in 0..2 {
        assert_eq!(
            dec.decode(inp.data, &o, &ln, &NO).cloned(),
            want,
            "a reused Decoder differs from the one-shot decode"
        );
    }
});
