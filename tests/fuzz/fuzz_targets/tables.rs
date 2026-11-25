#![no_main]
use jpeg::{DecodeOptions, Format};
use libfuzzer_sys::fuzz_target;

#[path = "../common.rs"]
mod common;
use common::*;

/// The fixed abbreviated tile of jpeg-go's `FuzzTables`: only the shared tables vary.
static IMAGE: &[u8] = include_bytes!("../../data/abbrev/image.jpg");

// data = a tables stream (TIFF tag 347). Decoding the tile against it, as RGBA and as
// coefficients, never panics.
fuzz_target!(|tables: &[u8]| {
    let l = limits();
    let o = DecodeOptions {
        format: Format::Rgba,
        tables: Some(tables),
        ..Default::default()
    };
    let _ = jpeg::decode(IMAGE, &o, &l, &NO);
    let _ = jpeg::decode_coefficients(IMAGE, Some(tables), &l, &NO);
});
