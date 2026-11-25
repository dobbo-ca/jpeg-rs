# Safety and SIMD

Hot loops have an AVX2 kernel (x86_64) and a NEON kernel (aarch64) beside the scalar code, which is
the oracle and the fallback. The tier is detected once (`is_x86_feature_detected!`,
`is_aarch64_feature_detected!`); `JPEG_SIMD=scalar` forces the scalar one, and `jpeg::simd_tier()`
reports the tier in use. `unsafe` is confined to `src/simd` (`#![deny(unsafe_code)]` elsewhere),
each `unsafe` block names its invariant, and a kernel only touches memory through bounds-checked
slices. The threaded decode passes (restart bands of a whole frame or of a region, colour stripes)
hand each worker its own `split_at_mut` slice of the planes and use no `unsafe`.

| Kernel | Go source | Output |
|---|---|---|
| islow IDCT 8x8 (`dct::idct8`) | `idct8AVX2Asm`, `idct8NEON` | equals scalar for every block; see below |
| reduced 4x4 IDCT (`dct::idct4`, scale 2) | `idct4AVX2Asm`, `idct4NEONAsm` | equals scalar for every block; see below |
| YCbCr to RGBA row | `colorRow*` | equals scalar for all 2^24 inputs (x86_64 with AVX-512: 32 pixels a step) |
| h2v1 and h2v2 fancy upsample rows | `h2v1FancyRow*`, `h2v2FancyRow*` | equals scalar (x86_64 with AVX-512: 64 and 32 columns a step) |
| encoder FDCT and quantize (`fdct_quant_rows`) | `fdctQuantAsm`, `fdctQuantPairAsm` | equals scalar (encoder output stays byte-identical); on x86_64 with AVX-512 (F, BW) two blocks that share a table go through one wide kernel (4:2:0 MCUs) |
| encoder RGBA to YCbCr row, h2v2 downsample row | `enccolor*` | equals scalar (the RGBA to YCbCr row has an AVX-512 loop on x86_64) |
| encoder block coder (`encoder::write_block`) | `huffPrepAsm`, `writeBlockAsm` | equals scalar (x86_64 only: AVX2 prep, or with AVX-512 a `permutex2var` to zigzag order; the coding loop is built with BMI1, BMI2, LZCNT and POPCNT) |

The IDCTs run in 32-bit lanes (the scalar code uses 64-bit sums) and no block is refused, so the
scalar IDCT never runs after a SIMD one gave up and the output does not depend on the tier. Pass 2
wraps mod 2^32, which is exact: the scalar range limit reads only bits 18 to 27 of its sum (bits 19
to 28 for the 4x4 IDCT), and those are in the low 32. Pass 1 needs the exact sum before its shift.
The largest sum of absolute weights is 61214 for the 8x8 IDCT, so one 32-bit pass is exact while
every dequantized coefficient is at most 35081 in magnitude (27563 for the 4x4, whose sum is
77910). A block with a larger one (a quantizer of 255 with a coefficient over 137, for one) runs
pass 1 on each value split into a high and a low half, which is exact for every `i32`. On x86_64
a block whose products all fit 16 bits takes the 16-bit `madd` kernel first, as Go's does. Go's
16-bit kernels give other bytes on such blocks (Go skips `cmp_evil_q255.jpg` under SIMD).
The 16-bit kernel's pass 2 is Go's: constants times 16, the rounding folded into the DC term and
a shift by 22, so the wrapped 32-bit lanes hold the scalar sum's bits 18 to 27 in their top bits;
signed saturating packs and an xor with 0x80 then equal turbo's range limit, and one `pshufb`
transposes the bytes. `src/simd/tests.rs` checks blocks whose output wraps against the scalar IDCT.

The AVX-512 kernels (F and BW) are a second x86_64 tier inside the AVX2 one: each is picked by its own
detection, `JPEG_SIMD=scalar` turns them off with the rest, and the tests run every kernel the CPU has
on its own against scalar. The decoder's colour and fancy upsample kernels have an AVX-512 version
too, although Go has none for decoding (its decode kernels are AVX2 only): here a wide tier is
added where it pays, not only where Go has one, and the benchmark decides whether it stays. Their
output is the same bytes as the AVX2 kernels, and
`jpeg::simd_tier()` still reports `avx2`. The decoder's MCU loop is also built a second time with BMI1, BMI2 and
LZCNT (`shlx`, `shrx`, `tzcnt`) and used where the CPU has them; it is the same Rust code, so no
differential target compares it with the plain loop: the `decode` fuzz target runs with
`JPEG_SIMD=scalar` as well, and the parity job decodes everything in both tiers.

Not ported to SIMD: the reduced IDCTs `idct2` and `idct1` (scale 4 and 8; Go has none either), and
the encoder's block coder and FDCT pair on aarch64 (NEON has the single-block FDCT).
Go's decoder Huffman assembly is not ported as assembly: its algorithm is, in Rust (the 10-bit
combined table of `decodeMCUFast`, the same bit-buffer refill rules).

Building with `-C target-cpu=x86-64-v4` (see [performance.md](performance.md)) changes only what the
compiler emits for the safe code; no `unsafe` is added, the run-time tier detection is unchanged, and
CI runs `tests/golden.rs` under that build to check the output is the same.

Tests: `src/simd/tests.rs` runs each kernel against scalar on deterministic pseudo-random inputs and
edge sizes (0, 1, lane-1, lane, lane+1, offset slices) and asserts that AVX2 (x86_64) or NEON
(aarch64) was detected. CI runs the whole suite on x86_64 with and without `JPEG_SIMD=scalar`, lints
the aarch64 target, and runs it natively on an aarch64 runner (job `arm64`), both tiers.

The pyramid mode's chroma blur (`sample::chroma_prefilter`) is scalar safe Rust with no SIMD tier and no
`unsafe`, so the output does not depend on the tier; the `pyramid` fuzz target runs under the default
and the `JPEG_SIMD=scalar` modes, which covers the kernels that run around it.
