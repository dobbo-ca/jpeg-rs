#![no_main]
use jpeg::DecodeOptions;
use libfuzzer_sys::fuzz_target;

#[path = "../common.rs"]
mod common;
use common::*;

// data = the option bytes of `decode`, then a stream. The header parse never panics, and when a
// decode returns an image its size is the one the header gave.
fuzz_target!(|src: &[u8]| {
    let Some(inp) = parse(src) else { return };
    let l = limits();
    let cfg = jpeg::decode_config(inp.data, inp.tables, &l);
    if let Ok(c) = &cfg {
        assert!(c.components == 1 || c.components == 3);
        assert!(c.width > 0 && c.height > 0 && c.width <= 65500 && c.height <= 65500);
    }
    let o = DecodeOptions {
        tables: inp.tables,
        ..Default::default()
    };
    if let Ok(d) = jpeg::decode(inp.data, &o, &l, &NO) {
        let c = cfg
            .as_ref()
            .expect("decode gave an image but decode_config failed");
        let b = d.image.bounds();
        assert_eq!(
            (b.width() as usize, b.height() as usize),
            (c.width, c.height)
        );
    }
    // row_reader_need with unlimited pixels parses what decode_config does with them unlimited.
    let any = jpeg::Limits {
        max_pixels: u64::MAX,
        ..l
    };
    if jpeg::row_reader_need(inp.data, inp.tables, &any).is_ok() {
        assert!(
            jpeg::decode_config(inp.data, inp.tables, &any).is_ok(),
            "row_reader_need parsed a stream decode_config refused"
        );
    }
});
