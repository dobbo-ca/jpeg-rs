#![no_main]
use libfuzzer_sys::fuzz_target;

#[path = "../tier.rs"]
mod tier;

// data = [offset][stride][quantizers: 64 bytes][pixels]. The encoder's FDCT and quantizer on a
// block of the pixels equals scalar, so the encoder's output stays byte-identical to Go's.
fuzz_target!(|data: &[u8]| {
    tier::check();
    let Some((&[off, stride], rest)) = data.split_first_chunk::<2>() else {
        return;
    };
    let Some((quant, pix)) = rest.split_first_chunk::<64>() else {
        return;
    };
    if let Err(e) =
        jpeg::kernel_check::fdct_quant(pix, usize::from(off), usize::from(stride), quant)
    {
        panic!("{e}");
    }
});
