#![no_main]
use libfuzzer_sys::fuzz_target;

#[path = "../tier.rs"]
mod tier;

// data = [odd][row]. The h2v1 fancy upsample equals scalar, to 2n samples or 2n - 1.
fuzz_target!(|data: &[u8]| {
    tier::check();
    let Some((&odd, row)) = data.split_first() else {
        return;
    };
    if let Err(e) = jpeg::kernel_check::h2v1(row, odd & 1 == 1) {
        panic!("{e}");
    }
});
