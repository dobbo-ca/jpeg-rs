# Fixtures and tests

## Fixtures

`tests/data/` is the Go repo's `internal/jpeg/testdata` (the `.jpg` and `.ppm` files in git LFS, as
there). `golden.json` has the libjpeg-turbo 3.2.0 md5s. Real-slide fixtures (`tests/data/local`)
are never committed.

## Tests

`cargo test` runs: `tests/golden.rs` (every fixture, every scale, RGBA/Gray/YCbCr, against turbo's
md5s; the abbreviated tables pair), `tests/coefficients.rs` (`jpeg_read_coefficients` md5s, the
truncated and corrupt fixtures, `DecodeConfig`, the row reader against `decode`),
`tests/region.rs` (`TestRegionIsCrop`, NDPI restart offsets, regions past the edge, `CheckFrame`),
`tests/encode.rs` (encode md5s for 4:2:0, 4:4:4, gray and YCbCr input), `tests/threads.rs` (every
fixture, format and scale at `max_threads` 1 to 8 gives the same image and error; damaged restart
streams, which send the bands back to the serial decode; `TestWorkersAtLimit`),
`tests/pyramid.rs` (`DecodeOptions::pyramid`: floor-sized output, equal to the cropped plain decode where the chroma blur is off, a PSNR floor against the full decode averaged in 2x2 blocks, the `Unsupported` cases, every thread count, a reused `Decoder`), and
`tests/resources.rs` (its own binary with a counting allocator: heap peak against `max_alloc` and the
charged peak, thread peak against `max_threads`, for several of each), and `tests/reuse_alloc.rs`
(its own binary with a counting allocator: the second decode on a reused `Decoder` makes no allocator
call at `max_threads` 1, and a small bounded number on a caller's pool; see
[performance.md](performance.md)). The `ci` stage also runs `tests/golden.rs` and `tests/encode.rs`
under `-C target-cpu=x86-64-v4`, and only on a runner with AVX-512.
Go tests with no Rust counterpart here: the zero-alloc and race tests (Go runtime), the limiter
and cgroup tests, the SIMD kernel tests and `-tags purego`, and the tests that need libjpeg-turbo built
(`JPEG_GOLDEN_REPRO`, `scripts/golden.sh`).
