#![no_main]
use libfuzzer_sys::fuzz_target;

#[path = "../tier.rs"]
mod tier;

// data = [offset][stride][mode][bytes]. The AVX2/NEON IDCT of a block, in whatever tier this CPU
// has, equals the scalar IDCT (no tier refuses a block); the dispatcher equals scalar.
// Mode bits 0-1 pick the coefficients (raw, small, sparse, medium), bit 2 the quantizer range,
// bit 3 raw 16-bit quantizers (0 and values with bit 15 set included).
fuzz_target!(|data: &[u8]| {
    tier::check();
    let Some((&[off, stride, mode], bytes)) = data.split_first_chunk::<3>() else {
        return;
    };
    if bytes.is_empty() {
        return;
    }
    let cyc = |i: usize| bytes[i % bytes.len()];
    let signed = |i: usize| i16::from(cyc(i) as i8);
    let coef: [i16; 64] = std::array::from_fn(|i| match mode & 3 {
        0 => i16::from_le_bytes([cyc(2 * i), cyc(2 * i + 1)]),
        1 => signed(i) * 4,
        2 => {
            if i < 3 {
                signed(i) * 8
            } else {
                0
            }
        }
        _ => signed(i) * 16,
    });
    let quant: [u16; 64] = std::array::from_fn(|i| {
        let b = u16::from(cyc(128 + i));
        if mode & 8 != 0 {
            u16::from_le_bytes([cyc(128 + 2 * i), cyc(129 + 2 * i)])
        } else if mode & 4 != 0 {
            1 + b % 255
        } else {
            1 + b % 16
        }
    });
    if let Err(e) = jpeg::kernel_check::idct8(&coef, &quant, usize::from(off), usize::from(stride))
    {
        panic!("{e}");
    }
});
