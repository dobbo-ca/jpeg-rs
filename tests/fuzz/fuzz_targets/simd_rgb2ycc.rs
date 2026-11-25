#![no_main]
use libfuzzer_sys::fuzz_target;

#[path = "../tier.rs"]
mod tier;

// data = RGBA pixels. The encoder's RGBA to YCbCr row equals scalar.
fuzz_target!(|data: &[u8]| {
    tier::check();
    if let Err(e) = jpeg::kernel_check::rgba_to_ycbcr(data) {
        panic!("{e}");
    }
});
