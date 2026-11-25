# Performance

How to build and call this crate for speed. The measured numbers are in [benchmarks.md](benchmarks.md).

## Build profile

A dependent crate's own release profile applies to this code, so the profile that counts is the
consumer's. For a worker that decodes tiles:

```toml
[profile.release]
lto = "fat"
codegen-units = 1
panic = "abort"
```

```sh
RUSTFLAGS="-C target-cpu=x86-64-v4 -C target-feature=+false-deps-tzcnt,+false-deps-bls" cargo build --release
```

- `lto = "fat"` and `codegen-units = 1` let the kernels and the loops that call them inline across
  modules and across the crate boundary. With the default 16 units the Rust to Go ratios in
  [benchmarks.md](benchmarks.md) are somewhat higher. A dependent that only wants the one-unit build
  of this crate can set `[profile.release.package.jpeg] codegen-units = 1`.
- `panic = "abort"` removes the unwinding tables. It also changes one thing here: a job that panics on
  a pool thread is re-raised on the calling thread (`src/par.rs`), and with `abort` the process stops
  at the panic. The crate does not panic on any input (clippy denies `unwrap`, `expect` and `panic`
  in its non-test code), so this only matters for a bug.
- `-C target-cpu=x86-64-v4` lets the compiler use AVX-512 (F, BW, CD, DQ, VL) everywhere, not only in
  the kernels. **Use it only when every machine that runs the binary has AVX-512.** On a CPU without
  it the program dies with `SIGILL` (illegal instruction) at the first such instruction, with no
  message. Check `/proc/cpuinfo` for `avx512f`, `avx512bw` and `avx512vl` first; the
  default build needs no flag and runs everywhere. Do not set it for an aarch64 build.
- The default build already runs the AVX-512 kernels on a machine that has them: the SIMD tier is
  chosen at run time (`jpeg::simd_tier()`, [safety.md](safety.md)). The gain of the v4 build comes from
  the compiler vectorising the scalar paths (the entropy decoder's surroundings, the reduced-scale
  IDCTs, the copies) and from inlining with the wider baseline, not from new kernels. The output is
  the same: the golden test (`tests/golden.rs`, against libjpeg-turbo) runs under the v4 build in CI, only on a runner with
  AVX-512.

- `-C target-feature=+false-deps-tzcnt,+false-deps-bls` is part of the v4 recommendation, not an
  option. Zen 4 and Zen 5 make `tzcnt`, `blsr`, `blsi` and `blsmsk` wait for the
  last write to their output register. LLVM breaks that wait only under `target-cpu=znver4`/`znver5`
  tuning; the `x86-64-v4` tuning does not, and the Huffman coder's per-coefficient loop (one `tzcnt`
  and one `blsr` per nonzero coefficient) then runs 30 to 50% slower than in the default build.
  rustc prints a warning that it does not know these two features and passes them to LLVM, which
  does. `-C target-cpu=znver5` gives the same speed but runs only on Zen 5. The output bytes are the
  same in every build.

Encoder, Rust ms/MB single-threaded, one `bench.yml` run with seven builds of the bench example on
an AMD Ryzen 7 9800X3D, each timed three times interleaved, best kept:

| build | enc-gray | enc-rgba420 | enc-rgba444 | enc-ycc | cmp_evil_q255 |
|---|---|---|---|---|---|
| default | 4.556 | 5.268 | 5.414 | 4.823 | 2.568 |
| `x86-64-v3` | 4.655 | 5.264 | 5.506 | 4.941 | 2.617 |
| `x86-64-v4` | 6.950 | 7.462 | 7.815 | 7.441 | 4.529 |
| `x86-64-v4`, kernels `#[inline(never)]` | 6.860 | 7.356 | 7.784 | 7.362 | 4.526 |
| `x86-64-v4`, `-C no-vectorize-slp -C no-vectorize-loops` | 6.862 | 7.394 | 7.838 | 7.376 | 4.526 |
| `x86-64-v4` + `+false-deps-tzcnt,+false-deps-bls` | 4.644 | 5.236 | 5.351 | 4.838 | 2.587 |
| `znver5` | 4.638 | 5.204 | 5.331 | 4.873 | 2.592 |

Only the two builds that break the false dependency recover; inlining and vectorisation do not
matter. `bench.yml` builds its v4 column (`bench-v4`) and `ci` runs the golden test with the
recommended flags. Against Go's `GOAMD64=v4` build, the v4 column of [benchmarks.md](benchmarks.md).

## Reusing a `Decoder`

Keep one `Decoder` for each worker thread and decode every tile on it. It is not `Sync`; give each
thread its own. A `Decoder` keeps what a call allocates: the planes and the RGBA image, the
coefficient blocks of a progressive or multi-scan frame, and the colour rows of the RGBA conversion.
A later call of the same size does not allocate and does not zero those buffers (the decode writes
every byte before it reads it); a larger frame grows them once and a smaller one reuses a part. The
result of `Decoder::decode` is valid until the next call, so write it out (or clone it) first.

- With `Limits::max_threads` 1, the second decode of a frame of the same shape makes no call to the
  allocator at all: `decode` and `decode_into_rgba`, all three formats, scales 1, 2, 4 and 8, a
  region inside the image, sequential, restart and progressive streams. `tests/reuse_alloc.rs` counts
  the calls of the allocator and asserts zero.
- With more threads (`Decoder::with_pool(pool)` and `max_threads` above 1) a call that is worth
  threading still makes a small number of allocations that are not the decoder's buffers: the job
  lists of a restart-band decode and of the colour stripes, their result lists, the colour rows of the
  extra workers, and the one heap-allocated task rayon makes for each extra worker it spawns. That
  count is bounded and does not grow with the image; `tests/reuse_alloc.rs` asserts the bound and
  prints the count of each case. A frame too small to thread (under 8 KiB of entropy data in a
  restart stream, under 131072 pixels for the colour conversion) runs on the calling thread with none
  of them. `max_threads` 1 is the setting that guarantees no allocation.
- Every charge still goes through the ledger: `max_alloc` is charged for all a call uses, reused or
  not, so the limits, the errors and the output are those of the free functions.
- A region that reaches past the image edge is copied into a new zero-filled image, which allocates
  on every call.
- Memory stays with the `Decoder` until it drops: the planes and RGBA image of the largest frame it
  has seen (about 4 bytes a pixel of RGBA, 1.5 of 4:2:0 planes), and, after a progressive frame, 2
  bytes a pixel of coefficients for each component.

Out of scope: `RowReader`, `Encoder` and `decode_coefficients` still allocate on every call.
`RowReader::new` builds its band buffers each time, `Encoder::encode` returns a new `Vec` (and keeps
only the quantization tables and a solid-colour output between calls), and `decode_coefficients`
returns the blocks it allocates. A reuse API for them would add public API and is not part of this
round.

## PGO and BOLT

Profile-guided optimisation and BOLT apply to the final binary, so the consumer does them (the
worker that links this crate); the library does not change. `pgo.yml` (dispatch only) measures how
much they gain on the bench example: it builds the example with `cargo-pgo`, trains it on the
fixtures in `tests/data`, then times the default, the PGO and the PGO plus BOLT binaries with
`tests/parity/bench.sh` and prints the table in the job summary. BOLT is optional there: if the
package is not available the job reports PGO only.

PGO and PGO plus BOLT, `pgo.yml`, Rust ms/MB against the default
build, single-threaded then default threads:

| class | PGO (1 thread) | PGO + BOLT (1 thread) | PGO (default) | PGO + BOLT (default) |
|---|---|---|---|---|
| coef | -1.6% | -1.6% | -1.7% | -1.7% |
| dec-gray | -2.6% | -2.4% | -4.2% | -4.4% |
| dec-rgba | -2.8% | -2.7% | -4.4% | -4.8% |
| dec-rgba4 | -2.2% | -2.8% | -2.3% | -3.0% |
| enc-rgba420 | -6.8% | -7.1% | -5.9% | -6.7% |
| rowreader | -0.2% | -0.3% | +0.5% | +0.5% |
| all | -2.6% | -2.6% | -2.9% | -3.3% |

BOLT adds 0 to 0.4 points over PGO.

The gain is an upper bound for another workload: the training input is the benchmark's own, so the
profile fits it more closely than it would fit a real slide.
