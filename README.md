# jpeg-rs

JPEG (ITU-T T.81) decoder and encoder for whole slide image tiles, in Rust. A port of the Go
library jpeg-go. The default decode matches libjpeg-turbo 3.2.0 (islow IDCT, fancy upsampling) bit
for bit, and the default encode matches `cjpeg -quality Q` and `tj3Compress8` byte for byte. Pure
Rust, one dependency (`rayon`, for the
worker pool: [docs/dependencies.md](docs/dependencies.md)); `unsafe` only in the SIMD kernel
modules (`src/simd`, [docs/safety.md](docs/safety.md)).

```toml
jpeg = { git = "https://github.com/dobbo-ca/jpeg-rs", rev = "<sha>" }
```

## Quick start
```rust
use jpeg::{DecodeOptions, EncodeImage, EncodeOptions, Format, Limits};
use std::sync::atomic::AtomicBool;

// At most 2 threads and 256 MiB for the whole call, threads included.
let limits = Limits { max_threads: 2, max_alloc: 256 << 20, ..Limits::default() };
let cancel = AtomicBool::new(false);

// `error` is Some(Truncated) or Some(AfterScan { .. }) when Go returns an image with an error.
let d = jpeg::decode(bytes, &DecodeOptions { format: Format::Rgba, ..Default::default() }, &limits, &cancel)?;

// A run of decodes on one Decoder reuses its buffers (after the first call, none are allocated).
let mut dec = jpeg::Decoder::new();
let d = dec.decode(bytes, &DecodeOptions { format: Format::Rgba, ..Default::default() }, &limits, &cancel)?;

let jpeg_bytes = jpeg::encode(EncodeImage::Rgba(&rgba), &EncodeOptions { quality: 75, chroma444: false }, &limits, &cancel)?;
```

More of the API (scale, region, restart offsets, tables, coefficients, `RowReader`):
[docs/usage.md](docs/usage.md).

## Resource limits

Every entry point that takes input of any size takes `&Limits`, and a `&AtomicBool` cancel flag
where it can run long (`Error::Cancelled`, polled once per MCU row).

| Field | Type, unit | Default | Over the limit |
|---|---|---|---|
| `max_pixels` | `u64`, pixels | 2^32 | `Error::PixelLimit { got, max }` |
| `max_alloc` | `u64`, bytes | 1 GiB | `Error::AllocLimit { got, max }` (`Error::OutOfMemory` if the allocator refuses) |
| `max_threads` | `usize`, threads, the caller included | `available_parallelism()` | none: fewer threads, same output |

- `max_alloc` is the enforced peak of the whole call, every thread together, charged before allocation.
- `max_threads = 1` runs on the calling thread and never touches a pool. A caller's
  `jpeg::ThreadPool` (`Decoder::with_pool`, `DecodeOptions::pool`) wins over the lazy shared pool.

Details: [docs/limits.md](docs/limits.md), [docs/threading.md](docs/threading.md).
## Docs

- [docs/go-differences.md](docs/go-differences.md): API differences, errors, Recon
- [docs/safety.md](docs/safety.md): `unsafe` and the SIMD kernels
- [docs/testing.md](docs/testing.md), [docs/parity.md](docs/parity.md) (Go parity job),
  [docs/fuzzing.md](docs/fuzzing.md): tests, fixtures, fuzz targets
- [docs/benchmarks.md](docs/benchmarks.md), [docs/performance.md](docs/performance.md): benchmarks;
  build profile, `target-cpu`, `Decoder` reuse, PGO

## Licence

Apache-2.0, see `LICENSE`. `NOTICE` has the libjpeg-turbo licences (IJG, BSD-3-Clause, zlib).
