#![no_main]
use libfuzzer_sys::fuzz_target;

#[path = "../tier.rs"]
mod tier;

// data = [offset][y, cb, cr: three rows of equal length]. YCbCr to RGBA equals scalar.
fuzz_target!(|data: &[u8]| {
    tier::check();
    let Some((&off, rest)) = data.split_first() else {
        return;
    };
    let n = rest.len() / 3;
    let (y, rest) = rest.split_at(n);
    let (cb, cr) = rest.split_at(n);
    if let Err(e) = jpeg::kernel_check::color_row(y, cb, cr, usize::from(off)) {
        panic!("{e}");
    }
});
