#![no_main]
use libfuzzer_sys::fuzz_target;

#[path = "../tier.rs"]
mod tier;

// data = two rows of equal length. The encoder's h2v2 downsample equals scalar.
fuzz_target!(|data: &[u8]| {
    tier::check();
    let (r0, r1) = data.split_at(data.len() / 2);
    if let Err(e) = jpeg::kernel_check::downsample_h2v2(r0, r1) {
        panic!("{e}");
    }
});
