#![no_main]
use libfuzzer_sys::fuzz_target;

#[path = "../tier.rs"]
mod tier;

// data = blocks of [mode][128 bytes]: 64 coefficients, raw, small, sparse or clamped to the
// baseline range by mode. The encoder's block coder (AVX2 prep and BMI coding loop) writes the
// bytes of the scalar one, flush included. (Only x86_64 has this kernel; NEON runs scalar.)
fuzz_target!(|data: &[u8]| {
    tier::check();
    let blocks: Vec<[i16; 64]> = data
        .chunks_exact(129)
        .map(|c| {
            let mode = c[0] & 3;
            std::array::from_fn(|i| {
                let raw = i16::from_le_bytes([c[1 + 2 * i], c[2 + 2 * i]]);
                match mode {
                    0 => raw,
                    1 => raw >> 8,
                    2 if i % 5 != 0 => 0,
                    2 => raw >> 6,
                    _ => raw.clamp(-1023, 1023),
                }
            })
        })
        .collect();
    if let Err(e) = jpeg::kernel_check::write_blocks(&blocks) {
        panic!("{e}");
    }
});
