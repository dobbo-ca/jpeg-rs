#![no_main]
use jpeg::DecodeOptions;
use libfuzzer_sys::fuzz_target;

#[path = "../common.rs"]
mod common;
use common::*;

// data = the option bytes of `decode`, then a stream. The coefficient blocks are whole; and a
// stream whose decode gives an image with no error gives coefficients with no error.
fuzz_target!(|src: &[u8]| {
    let Some(inp) = parse(src) else { return };
    let l = limits();
    let coef = jpeg::decode_coefficients(inp.data, inp.tables, &l, &NO);
    if let Ok(d) = &coef {
        let c = &d.coefficients;
        assert_eq!(c.comp.len(), c.config.components);
        for cc in &c.comp {
            assert_eq!(cc.blocks.len(), cc.blocks_wide * cc.blocks_high);
        }
    }
    // The any-components entry point: never a panic, whole blocks, and the same as
    // `decode_coefficients` whenever the stream is not 4-component.
    let any = jpeg::decode_coefficients_any_components(inp.data, inp.tables, &l, &NO);
    if let Ok(d) = &any {
        for cc in &d.coefficients.comp {
            assert_eq!(cc.blocks.len(), cc.blocks_wide * cc.blocks_high);
        }
        if d.coefficients.config.components != 4 {
            assert_eq!(
                Ok(d),
                coef.as_ref(),
                "any_components differs from decode_coefficients"
            );
        }
    }
    let o = DecodeOptions {
        tables: inp.tables,
        ..Default::default()
    };
    if let Ok(d) = jpeg::decode(inp.data, &o, &l, &NO) {
        if d.error.is_none() {
            let c = coef.expect("decode succeeded but decode_coefficients failed");
            assert!(c.error.is_none(), "decode_coefficients: {:?}", c.error);
        }
    }
});
