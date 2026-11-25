# Differences from jpeg-go

## Where this differs from Go

- Idiomatic API: `&[u8]` in, owned images out (`Gray`, `Rgba`, `YCbCr` with Go's strides, block
  padding and `rect`). `Decoder` is Go's reusable `Decoder`: it keeps the planes and the output of
  one call for the next, cuts or grows them and never clears them (a decode writes every byte before
  it reads one; a region decode and a `Recon` hook, which may leave bytes alone, start from zeroes),
  and returns the image by reference, valid until its next call. It also keeps the coefficient
  blocks of a progressive or multi-scan frame and the colour rows of the RGBA conversion, which
  Go allocates per call; at `max_threads` 1 the second decode of a frame of the same shape makes no
  allocator call (`tests/reuse_alloc.rs`, [performance.md](performance.md)). The free functions are its one-shot
  form and have no global pool (Go's `Decode` pools its `Decoder`s). The charge against
  `Limits::max_alloc` is the same with or without reuse, so the limits and errors do not change. The
  parity job runs one reused `Decoder` and `Encoder` on each side (Go's and Rust's) over every input
  in the same order, each reused decode after a fixed dirty decode, and the outputs must be equal,
  so what a reused object keeps between inputs is checked on both; `tests/decoder.rs` and the
  `decoder_reuse` and `encoder_reuse` fuzz targets compare the Rust ones with fresh calls. A region is copied out of the band, and a region past the image edge is zero-filled, as
  Go's.
- A region decode with restart offsets runs in bands on several threads, and its colour conversion in
  stripes; Go decodes a region serially. The output is the same: a band starts only where the serial
  decode jumps to an offset itself, and any band that ends short sends the decode back to the serial
  path (see [threading.md](threading.md)).
- The process-wide memory limiter (`SetMemoryLimit`, `ErrMemoryLimit`, cgroup detection, the FIFO of
  waiting decodes) is not ported: a caller bounds its memory with `Limits::max_alloc`.
  `DecodeOptions.Workers` is `Limits::max_threads` (default `available_parallelism`, where Go's
  default is serial; one value for the whole call, not per decode); `Reserved` and `Owned` are not
  ported (they have no meaning without reuse and the limiter). `Dst` is
  `decode_into_rgba`, compared with Go's on every input by the parity job (`dst` and `dst0`).
- `DecodeOptions::color_model` and `RowReader::with_color_model` (Rust only, additive):
  `ColorModel::Rgb` decodes a 3-component frame as RGB with no colour transform whatever its
  markers say, as libtiff and OpenSlide do for TIFF tiles with PhotometricInterpretation RGB (see
  [usage.md](usage.md), Colour model). Go reads the markers only. The default, `ColorModel::Auto`,
  is Go's reading, so parity does not change.
- The encoder takes `EncodeImage::{Gray, Rgba, YCbCr}`; Go's `*image.NRGBA` is an `Rgba` (alpha is
  ignored either way), and other `image.Image` types have no Rust counterpart. `Encoder` is Go's
  reusable `Encoder`: it keeps the quantization tables of the last quality and the output of the
  last solid-colour image (up to 32 KiB; part of the fixed state in [limits.md](limits.md)), and `encode` is its
  one-shot form. Go's `WriteUniformScan` is not ported, only the cache: the speed changes, not
  the output.
- The "combined" Huffman lookup table of Go's decoder is ported (`src/huff.rs`, 10 bits); the Go
  tests that compare it with the plain table on random tables have their counterpart in the 400
  `rh~` streams of the parity job, which Go and Rust decode alike. The Huffman assembly
  (`decodeMCUAsm`, `writeBlockAsm`, `huffPrepAsm`) is not ported as assembly.
- The reference of the parity job is Go's scalar build (`JPEG_NOSIMD=1`), not its default SIMD one:
  Go's SIMD kernels give other bytes than libjpeg-turbo on `cmp_evil_q255.jpg` and streams like it,
  where this crate's 32-bit kernels give the scalar code's bytes. So "byte for byte with Go" holds for Go's scalar
  code; Go's SIMD output and its arm64 NEON paths are not compared (a deliberate deviation, found by
  review). This crate's output does not depend on its own tier.
- `decode_coefficients_any_components` (no Go counterpart) accepts 4-component (CMYK/YCCK) 8-bit frames,
  which Go's `DecodeCoefficients` and this crate's `decode_coefficients` reject with `ErrUnsupported`: `Coefficients::comp` then has four entries and `Config::components` is 4
  (`rgb` is false). It is limited to sampling factors 1 to 4 with at most 10 blocks per interleaved MCU.
  `decode`, `decode_config` and `RowReader` still return `Error::Unsupported` for 4 components, as Go does.
- The decoder's colour conversion and fancy upsampling have AVX-512 kernels on x86_64; Go's decode
  kernels are AVX2 only. The bytes are the same (the tests run every kernel against scalar); only
  the speed differs.

- `DecodeOptions::pyramid` has no Go counterpart: it is a lossy, opt-in mode for building a pyramid
  (floor-sized output at scale 2, 4 or 8, with a chroma blur; see [usage.md](usage.md)). With it off,
  the output is Go's. `tests/pyramid.rs` checks it against the full decode averaged in 2x2 blocks
  (a PSNR floor) instead of byte for byte; the parity job does not run it.

## Errors

One enum, `jpeg::Error`, with a variant for each Go error. `Unsupported`, `Truncated` and `AfterScan`
are `ErrUnsupported`, `ErrTruncated` and `ErrAfterScan`; the marker errors (`NoSoi`, `DupSoi`,
`BadLength`, `UnknownMarker`, `NoSof`, `NoSos`, `SosBeforeSof`, `DupSof`, `EmptyImage`, `TooBig`,
`ComponentId`, `DqtIndex`, `DhtIndex`, `HuffTable`, `TablesStream`, `DacIndex`, `ScanSelector`) and the
decode errors (`NoQuantTable`, `NoHuffTable`, `EoiExpected`, `BadProgression`, `BadDctCoef`) are the
package's unexported ones; `Region` is the two region errors. `Format`, `UnsupportedFeature` and
`UnexpectedEof` (Go `image/jpeg`'s `FormatError`, `UnsupportedError`, `io.ErrUnexpectedEOF`) are
not returned by this crate; they stay for dependents that report Go-style errors.

Go returns an image together with `ErrTruncated` (data ran out, a bad Huffman code, restarts out
of sequence) or `ErrAfterScan` (a fatal marker after the last row of a single-scan frame). Rust keeps
that: `Decoded { image, error }` has the image and `Some(error)` for those two, and `Err` only when
there is no image.

## Recon

`Recon` replaces the IDCT (`IdctHook`) and the colour step (`ToRgbaHook`) with the caller's own
functions, at `Format::Rgba`, scale 1 and no region; any other use is `Error::Unsupported`.
