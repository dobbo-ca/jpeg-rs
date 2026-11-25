#![no_main]
use jpeg::{DecodeOptions, Decoder, Format, Limits, Rect, RgbaWindow};
use libfuzzer_sys::fuzz_target;

#[path = "../common.rs"]
mod common;
use common::{NO, limits};

/// Large, dense frames that leave non-zero bytes in a `Decoder`'s buffers: 4:2:0, 4:4:4, 4:2:2,
/// progressive and gray.
const DIRTY: [&[u8]; 5] = [
    include_bytes!("../../data/jpg/svs_420_512x512.jpg"),
    include_bytes!("../../data/jpg/synth_444_227x149.jpg"),
    include_bytes!("../../data/jpg/synth_422_227x149.jpg"),
    include_bytes!("../../data/jpg/cmp_prog.jpg"),
    include_bytes!("../../data/jpg/synth_gray_227x149.jpg"),
];

fn be16(b: &[u8]) -> usize {
    usize::from(b[0]) << 8 | usize::from(b[1])
}

// data = 13 header bytes, then the streams:
//   0      format of the first stream (+1 for each next one), mod 3: YCbCr, Rgba, Gray
//   1      scale: 1, 2, 4, 8 (mod 4)
//   2      bits 0-2: max_threads - 1; bit 3: unused; bits 4-7: stride padding of a window
//   3      the dirty frame (bits 2-4), its format (bits 0-1) and scale 2 (bit 7)
//   4      bit 0: a region; bit 1: three streams, else two; bit 2: a small max_alloc, 2^(16 + bits
//          3-5); bit 6: check_frame
//   5, 6   length of stream 1; 7, 8: length of stream 2 (the rest is the last one)
//   9-12   the region
// A `Decoder` first decodes a large dense frame, so its planes and RGBA image hold data, then each
// stream in turn. Every result and error must equal a fresh one-shot decode of the same stream
// at max_threads 1 (a stale byte leaking out of a reused buffer, or a thread count changing the
// output, is a crash); so must the threaded one-shot decode of every frame that can be cut.
fuzz_target!(|src: &[u8]| {
    let Some((h, rest)) = src.split_first_chunk::<13>() else {
        return;
    };
    let (a, rest) = rest.split_at(be16(&h[5..7]).min(rest.len()));
    let (b, c) = rest.split_at(be16(&h[7..9]).min(rest.len()));
    let (three, two) = ([a, b, c], [a, rest]);
    let streams: &[&[u8]] = if h[4] & 2 != 0 { &three } else { &two };

    let mut l = limits();
    if h[4] & 4 != 0 {
        l.max_alloc = 1 << (16 + (h[4] >> 3 & 7));
    }
    let threads = 1 + usize::from(h[2] & 7);
    let ln = Limits {
        max_threads: threads,
        ..l
    };
    let region = (h[4] & 1 != 0).then(|| {
        let (x0, y0) = (i32::from(h[9] as i8) * 3, i32::from(h[10] as i8) * 3);
        Rect::new(
            x0,
            y0,
            x0 + 1 + i32::from(h[11]) * 2,
            y0 + 1 + i32::from(h[12]) * 2,
        )
    });

    let mut dec = Decoder::new();
    let dirty = DecodeOptions {
        format: [Format::Rgba, Format::YCbCr, Format::Gray][usize::from(h[3] & 3) % 3],
        scale: if h[3] & 0x80 != 0 { 2 } else { 1 },
        ..Default::default()
    };
    let big = Limits {
        max_threads: threads,
        ..limits()
    };
    let _ = dec.decode(
        DIRTY[usize::from(h[3] >> 2 & 7) % DIRTY.len()],
        &dirty,
        &big,
        &NO,
    );

    for (i, s) in streams.iter().enumerate() {
        let o = DecodeOptions {
            format: [Format::YCbCr, Format::Rgba, Format::Gray][(usize::from(h[0]) + i) % 3],
            scale: [1, 2, 4, 8][usize::from(h[1] & 3)],
            region,
            check_frame: h[4] & 64 != 0,
            // Unsupported combinations (YCbCr, a region, scale 1) are the same error on both sides.
            pyramid: h[4] & 128 != 0,
            ..Default::default()
        };
        let want = jpeg::decode(s, &o, &l, &NO);
        assert_eq!(
            dec.decode(s, &o, &ln, &NO).cloned(),
            want,
            "stream {i}: a reused Decoder at {threads} threads differs from a fresh decode"
        );
        assert_eq!(
            jpeg::decode_probe(s, &o, &ln, &NO, true).0,
            want,
            "stream {i}: {threads} threads differ from 1"
        );

        // The same into a caller's window with a gap after each row: bytes and result.
        if o.format != Format::Rgba || o.scale != 1 || region.is_some() || h[2] & 8 != 0 {
            continue;
        }
        let Ok(cfg) = jpeg::decode_config(s, None, &l) else {
            continue;
        };
        let stride = 4 * cfg.width + usize::from(h[2] >> 4);
        let (mut p1, mut p2) = (
            vec![0xa5u8; stride * cfg.height],
            vec![0xa5u8; stride * cfg.height],
        );
        let r1 = jpeg::decode_into_rgba(
            s,
            &o,
            &l,
            &NO,
            RgbaWindow {
                pix: &mut p1,
                stride,
            },
        );
        let r2 = dec.decode_into_rgba(
            s,
            &o,
            &ln,
            &NO,
            RgbaWindow {
                pix: &mut p2,
                stride,
            },
        );
        assert_eq!(r2, r1, "stream {i}: window result differs");
        assert!(p1 == p2, "stream {i}: window bytes differ");
    }
});
