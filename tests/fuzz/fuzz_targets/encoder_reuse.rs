#![no_main]
use jpeg::{EncodeImage, EncodeOptions, Encoder};
use libfuzzer_sys::fuzz_target;

#[path = "../common.rs"]
mod common;
use common::*;

// data = [count][limits][count descriptors of 6 bytes][samples]: one `Encoder` encodes up to six
// images in turn, each result and error equal to a fresh one-shot `jpeg::encode` of the same input
// (a stale quantization table, or a solid-tile cache hit answering for another image, is a crash).
//   count    1 + count % 6 images
//   limits   bit 0: a small max_alloc, 2^(11 + bits 1-3), else the fuzz default
//   a descriptor [kind][w][h][quality][flags][colour]: an image of 1..=64 by 1..=64 pixels of
//            `build`'s `kind`, quality as `encode`'s target; flags bit 0 chroma444, bit 1 a solid
//            image of `colour` (what the cache keeps), bit 2 the previous image and options again
//            (a cache hit), bits 4-6 the stride padding
//   samples  the rest of the input cycled, the pixels of an image that is not solid
fuzz_target!(|data: &[u8]| {
    let Some((&[count, lim], rest)) = data.split_first_chunk::<2>() else {
        return;
    };
    let mut l = limits();
    if lim & 1 != 0 {
        l.max_alloc = 1 << (11 + usize::from(lim >> 1 & 7));
    }
    let (descs, samples) = rest.split_at(rest.len().min(6 * (1 + usize::from(count % 6))));
    let samples = if samples.is_empty() { &[0x80] } else { samples };
    let mut enc = Encoder::default();
    let mut cur: Option<(Pixels, EncodeOptions)> = None;
    for d in descs.chunks_exact(6) {
        let &[kind, w, h, q, flags, colour] = d else {
            continue;
        };
        if flags & 4 == 0 || cur.is_none() {
            let (w, h) = (1 + usize::from(w) % 64, 1 + usize::from(h) % 64);
            let one = [colour];
            let bytes: &[u8] = if flags & 2 != 0 { &one } else { samples };
            let pix = build(kind % 8, w, h, 0, 0, usize::from(flags >> 4 & 7), bytes);
            let o = EncodeOptions {
                quality: i32::from(q) % 130 - 5,
                chroma444: flags & 1 == 1,
            };
            cur = Some((pix, o));
        }
        let Some((pix, o)) = &cur else { continue };
        let img = match pix {
            Pixels::Rgba(m) => EncodeImage::Rgba(m),
            Pixels::Gray(m) => EncodeImage::Gray(m),
            Pixels::YCbCr(m) => EncodeImage::YCbCr(m),
        };
        let want = jpeg::encode(img, o, &l, &NO);
        assert_eq!(
            enc.encode(img, o, &l, &NO),
            want,
            "a reused Encoder differs from a fresh encode"
        );
    }
});
