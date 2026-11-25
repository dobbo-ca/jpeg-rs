# Fuzzing

`tests/fuzz/` (cargo-fuzz, the only crate with `libfuzzer-sys`) has a target for each entry point
(`decode`, `decode_config`, `tables`, `coefficients` (also `decode_coefficients_any_components`: whole blocks, and equal to `decode_coefficients` unless the stream has 4 components), `row_reader`, `pyramid` (floor-sized output, threads and a reused `Decoder`, dirtied by a plain decode first, equal the one-shot decode, a gray or RGB stream equals the plain scaled decode), `encode`, `ledger`, the plain kernels in
`kernels`, and `decoder_reuse` and `encoder_reuse`, which check a reused `Decoder` (`pyramid` included) or `Encoder`, with
any `max_threads`, against fresh one-shot calls) and a differential target per SIMD kernel (`simd_idct8`, `simd_idct4`, `simd_color`, `simd_h2v1`,
`simd_h2v2`, `simd_fdct`, `simd_rgb2ycc`, `simd_downsample`, `simd_huff`: this CPU's tier against scalar, through the
hidden module `jpeg::kernel_check`; on x86_64 `simd_rgb2ycc`, `simd_color`, `simd_h2v1`, `simd_h2v2` and `simd_huff` also run the AVX2 and the
AVX-512 kernel each on its own, and the log says whether the runner has AVX-512, which
`JPEG_EXPECT_AVX512=1` makes a requirement). `decode` checks a region against a crop of the whole decode and
`decode_into_rgba` against `decode`; `row_reader` is the stateful one (cancel mid-read, closed after an
error, rows equal to the decode); `encode` checks the stream decodes, is deterministic and equal for a
sub-image and its tight copy (and `encode_to` writes the same bytes), and (kind bit 7) that an image with a hostile rect, strides or plane
lengths gives `Ok` or `Err`, never a panic; `decode` also sets a restart-offset list, `check_frame` and
a region past the image from the input and checks the region against a crop with zeros outside; `ledger`
runs every entry point under a counting allocator and a `max_alloc` from the input, and asserts the peak
is at most `max_alloc` plus 1 MiB (jpeg-go's `FuzzLedger`). Seeds: jpeg-go's `testdata/fuzz` entries (`tests/fuzz/seeds`) and the fixtures. The
workflow `fuzz.yml` (weekly and `workflow_dispatch`, `max_total_time` per target, `-rss_limit_mb=2048
-malloc_limit_mb=512`) runs each target, the decode, decoder_reuse, row reader, pyramid and encode targets also with
`JPEG_SIMD=scalar`, and the `simd_*` targets also run natively on an arm64 runner, with ASan (the
corpus is cached between runs, per architecture).
