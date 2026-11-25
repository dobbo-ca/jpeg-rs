#![no_main]
use libfuzzer_sys::fuzz_target;

#[path = "../tier.rs"]
mod tier;

// data = [odd][cur, other: two rows of equal length]. The h2v2 fancy upsample equals scalar.
fuzz_target!(|data: &[u8]| {
    tier::check();
    let Some((&odd, rest)) = data.split_first() else {
        return;
    };
    let (cur, other) = rest.split_at(rest.len() / 2);
    if let Err(e) = jpeg::kernel_check::h2v2(cur, other, odd & 1 == 1) {
        panic!("{e}");
    }
});
