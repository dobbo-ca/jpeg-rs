# Parity with Go

CI job `parity` builds `tests/parity/go/upstream` (jpeg-go's source) and `tests/parity/go/oracle`, which
writes about 10,000 inputs (every fixture, mutations of each, random images, coefficient-level
streams, kernel vectors, ported from the Go tests' generators) and runs about 60 operations on them (about 220,000 output files)
(decode at every format and scale, regions, restart offsets, tables, coefficients, the row reader,
limits, cancellation, encode at eight qualities). `examples/oracle.rs` does the same with this crate and
the two output trees must be equal, in both SIMD tiers and with `oracle --threads 1`, the default and
`--threads 4` (with threads, every decode runs with the size threshold off); `parity-arm64` runs a
subset on a native arm64 runner. Where Go
returns an image with an error, both write the image too. This page has the inputs, the wire
formats, the error class table and the count of Go tests ported.

`tests/parity/go/upstream/` is the non-test source of the Go library jpeg-go (`jpeg.go`, `internal/codec`,
`internal/jpeg`, `internal/ledger`, `go.mod` and `LICENSE`;
jpeg-go has no NOTICE). `tests/parity/go/oracle/` is a Go program with its own `go.mod` that `replace`s the module
with `../upstream`. The CI job `parity` (`.github/ci/Dockerfile`, stages `go-oracle`, `go-run`, `parity`) runs:

1. `oracle -gen tests/data /inputs tests/fuzz/seeds` writes the inputs.
2. `oracle /inputs /go-out` runs every operation on every input with jpeg-go.
3. `cargo run --release --example oracle -- --threads 1 /inputs /rust-out` does the same with this
   crate.
4. `tests/parity/diff.sh /go-out /rust-out` must find no difference (`diff -rq`; a failure lists the count
   per operation and the first bytes of the first files). A difference is a bug in the Rust port,
   never in the oracle.

Steps 3 and 4 run three times against the same Go output, so the job makes three Rust runs: with
`--threads 1` (`max_threads` 1 on every call), with the default (`max_threads` is the runner's
`available_parallelism`), and with `JPEG_SIMD=scalar --threads 4` (the scalar tier, four threads).
With `max_threads` over 1 every decode goes through `decode_probe` with the size threshold off, so the
small inputs run on bands and stripes too.

Go runs with `JPEG_NOSIMD=1`: its SIMD kernels give other bytes than libjpeg-turbo on
`cmp_evil_q255.jpg` and on streams like it (the Go tests skip that fixture under SIMD), where the
Rust kernels give the scalar code's bytes. The Rust output is the same in every tier, so the reference is
Go's scalar code. This is a deliberate deviation: Go's own SIMD output (decode and encode) and its
arm64 NEON paths are not compared with Rust, and "identical to Go" means to Go's scalar build. The job `parity-arm64` builds the oracle example for aarch64 and runs it natively
on an aarch64 runner on a subset (every eighth input, the sidecar files and the dirty frame), once per tier, against Go on the
same subset.

The oracle decodes with a new `jpeg.Decoder` per call, not the package-level `jpeg.Decode`: that one
copies the image and its `MaxBytes` check counts the copy, which this crate, returning the image itself,
has not. The exceptions are the ops that test reuse: `sd1`, `sy1`, `sr` decode through one `Decoder`
and `se0 se0c se75 se75c` encode through one `Encoder` (its uniform-image cache included) shared by every
input, in the sorted order of the input directory, so each one sees the state a different earlier input
left. The Rust oracle does the same with one `jpeg::Decoder` and one `jpeg::Encoder` (a thread-local
`Shared` in `examples/oracle.rs`), and the outputs are compared with Go's. Each reused decode
(`sd1`, `sy1`, `sr`) follows a fixed dirty decode on the same `Decoder`, on both sides: `synth_444_227x149.jpg`
(read from the input directory) to RGBA at scale 1, which leaves data in the planes and the RGBA image,
so a byte that a reused buffer keeps and a later smaller, sparser or truncated frame does not rewrite
shows. (`tests/decoder.rs` and the `decoder_reuse` and `encoder_reuse` fuzz targets compare the Rust
reuse with fresh calls.) The aarch64 subset always includes the dirty frame. `dst` and `dst0` are `Decoder.Decode` with `Dst` into a
window of a buffer of 0xAA bytes (with a row above, 8 bytes to the left and a padded stride, or tight)
against `decode_into_rgba`, the whole buffer and the error compared, on every input, truncated ones
included (`TestDecodeDstTruncated`, `TestDecodeDstWindows`). `rrx` is `Next` after the last row, after
`Close` and after an early `Close`. A damaged `strip_rst` or `dri-` stream (`~` in its name) also has a `.starts` sidecar (the undamaged offsets shifted, one huge, cut short or out of order) and a `.rect` sidecar (regions at scale 1 and 2), so `r<i>`, `ndpi-rg`, `ndpi-gr2`, `ndpi-reg` (`check_frame` on) and `ndpi-reg2`, `ndpi-reg2g` (a mid-frame region in Rgba and Gray with `check_frame` off, so the offset-landing jump and the region bands run) run on them: the counterpart of `TestRegionRenumberedRestart`, `TestRegionSpuriousRestartMarker`, `TestRegionRestartMarkerDeleted`, `TestRegionRestartPendingTruncated`, `TestRegionOffsetsTrusted`, `TestRegionBadRestartOffsets`, `TestRegionFewerEntropyBytes` and `TestRegionCorruptRestartFixtures`.

The oracle is bounded: every call has `Limits{max_pixels 2^22, max_alloc 128 MiB}` on both sides
(the process limiter is off), and `debug.SetMemoryLimit(1 GiB)`, so a hostile frame cannot take the
runner down. A panic in an operation is written as the class `panic:...`, which the Rust side
never writes for the same input, so it shows as a difference.

## Inputs

`oracle -gen` makes, from the fixtures in `tests/data/` and Go's `math/rand` (so the expected bytes
come from Go and the Rust side is not told):

| Input | What |
|---|---|
| `<fixture>.jpg` | Every `.jpg` of `jpg/`, `subsample/`, `truncated/` and `abbrev/image.jpg` (as `ab-image.jpg`, decoded against `tables.bin`). |
| `<fixture>~<tag>.jpg` | 6 to 64 mutations of each: cuts (`t<at>`: inside the headers and at 24 fractions), entropy bit flips and byte sets (`f`, `v`, `z`, `h`), header field edits (`s` SOF fields, `d` dimensions from 0 to 65535, `m` the SOF type, `o` SOS fields, `q` DQT, `c` DHT, `a` Adobe transform), marker insertions (`i`), segment length edits (`l`), appended bytes (`a`), a DRI (`r`), RSTn renumbered, dropped or doubled (`rn`, `rd`, `rr`), and the sampling relabels (`hv`). Spread over the kinds, so every kind is in. |
| `<fixture>.rect` | A region list: `TestRegionIsCrop`'s rectangles (the whole image, a pixel at each corner, the edge rows), 3 random ones from `randRect`, and rectangles past the edge, empty, outside and out of range, at scales 1, 2, 4, 8, with the three formats and `CheckFrame`. |
| `dri-<w>x<h>-ri<n>.jpg`, `.starts` | `synthDRI` of `jpeg_region_test.go`: a grayscale stream with a restart interval and one random DC block per MCU, its NDPI restart offsets, and 12 mutations. `strip_rst.starts` and `ndpi_wide_4096x32.starts` are the fixtures'. |
| `coef-<n>-...jpg` | 72 baseline streams with random quantized coefficients (DC only, sparse, dense, extreme; quantizers of 1 to 255, all 255, or 1 to 16; sampling 1x1, 2x1, 1x2, 2x2, 4x1, 4x2; RGB ids; restart intervals), written by the oracle's own Huffman writer with the standard tables: the IDCT and colour paths on blocks no encoder writes (`dct_test.go`'s `vectorBlock`). |
| `hdr-<n>-<name>.jpg` | The 45 header-only streams of `parityCases` (marker rules, SOF/DHT/DQT/DRI/DAC/SOS edge cases). |
| `tables-reset.jpg`, `.tables` | `TestTablesStateReset`: a tables stream with a DRI and an Adobe segment, put beside a stream that has neither (`<name>.tables` is read as the tables of any `.jpg`). |
| `between-<fixture>-<segment>.jpg`, `after-<eoi\|mid>-<segment>.jpg`, `after-eoi-sof9.rect` | `betweenScans` put before the second SOS of `synth_multiscan_16x16` and `prog_420_111x75`; `TestAfterScanTurboParity`'s second SOS, SOF9 and fatal segments after the last row or after a stray marker mid-scan, and its three crop regions. |
| `unscanned-<fixture>.jpg`, `notseq-<fixture>-<n>.jpg` | `TestUnscannedComponentTurboParity` (a gray stream whose SOF claims Cb and Cr) and `TestNotSequentialTurboParity` (Ss/Se/Ah/Al edits). |
| `mt-<name>.jpg` | The streams of `internal/codec/marker_test.go`: an unknown marker, a DC symbol over 15 and a second scan after the last, an RST, segments that overrun the data, a DHT with 300 codes. |
| `dac-<fixture>-<n>-<sos\|soi>.jpg` | Ten DAC (0xCC) segments (valid, odd, short, zero and 0xFFFF length, bad index) before the SOS or after the SOI of five fixtures. The generic mutations also insert and relabel 0xCC. |
| `dense-48x32.jpg`, `fast-marker-at-dc.jpg`, `fast-ac-overflow.jpg` | `denseStream` (no EOB, value sizes 1 to 10) with 40 mutations in place of `corruptFuzz`, and the two streams of `decode_fast_test.go` (a marker at the first DC symbol; a run of 15 that overflows the block). |
| `rst-<kind>-ri<n>.jpg`, `rst-<kind>-ri<n>~r<i>.jpg`, `~e<i>` | `rstCorpus`' synthetic gray, 4:4:4 and 4:2:0 streams with restart intervals 1 to 3 (and 7, 12, 16), and 20 + 20 `mutateRST` edits of each (byte deletes, arbitrary-byte and lone-0xFF inserts, RSTn inserts, drops and renumbers, cuts; anywhere, and in the 600 bytes before an RSTn or the EOI). |
| `tbl-<tag>.tbl` | `tables.bin` and 120 mutations of it, decoded against `ab-image.jpg` (`FuzzTables`). |
| `<name>.img` | Encoder inputs: 330 random images of the nine kinds (RGBA, Gray, YCbCr at six ratios, NRGBA), sub-images off the block grid with strides past the width and the modes of `diffImage` (`TestEncodeMatchesV040`), the uniform and near-uniform images of `TestUniformSameBytes`, the PPM fixtures, and the decoded fixtures as RGBA, Gray and YCbCr. |
| `fancy-<n>.fancy` | Kernel vectors: `TestFancyRowsMatchRef`'s rows plus the lengths Go panics on. |
| `seed-<name>.jpg`, `seed-rr-<name>.jpg` | The jpeg-go `FuzzDecode` and `FuzzRowReader` seeds (`tests/fuzz/seeds/decode`, `tests/fuzz/seeds/row_reader`, minus their four option bytes). |
| `rh~<seed>.jpg` | 400 baseline streams of 15x10 MCUs (4:4:4) with random valid DHT tables and random symbols, `randHuff` and `randScan` of `huff_comb_test.go`; longest code 3, 6, 8, 9, 10, 11, 12 or 16 bits, so a code plus its value bits is under, at and over the 10 bits of the combined lookup table; odd seeds have 6 bit flips. |

### Go tests

jpeg-go's tests (`jpeg.go`'s package, `internal/jpeg`, `internal/codec`), its fuzz targets and
benchmarks. Here:

| Go tests | Where |
|---|---|
| The fixture goldens (`golden_test`, `decoder_test`, `scaled_test`, `subsample_test`, `progressive_test`, `jpeg_recon_test`, `coef_test`, `rowreader_test`, the limits), the encoder md5s | `tests/`, and again against Go here on every fixture and every mutation. |
| `marker_parity_test` (`parityCases`, `TestTablesStateReset`, `betweenScans`, `TestAfterScanTurboParity`, `TestUnscannedComponentTurboParity`, `TestNotSequentialTurboParity`), `internal/codec/marker_test`, `decode_fast_test`, the input side of `TestSkipPathBitExact`, `TestFastMCUAsm`, `TestSlowMCU`, `TestRestartFastPathRandom` | Not in `tests/`: only the generated `hdr-`, `tables-reset`, `between-`, `after-`, `unscanned-`, `notseq-`, `mt-`, `dac-`, `fast-`, `dense-` and `rst-` inputs here, with Go's output as the reference (the tests' own turbo expectations are not ported; Go's output stands in for them). |
| `TestRegionIsCrop`, `TestNDPIRestart`, `TestRegion*` | `tests/region.rs`, and the `<fixture>.rect`, `.starts` and `dri-` inputs here. |
| `TestEncodeMatchesV040`, `TestUniformSameBytes`, `encoder_roundtrip_test`, `enc_test`: the encoder against its reference and against itself | The `.img` inputs: the bytes of the Go encoder (the uniform fast path is not ported: it changes speed, not output) at qualities 0, 1, 25, 50, 75, 90, 100, 101, 4:2:0 and 4:4:4. |
| `TestFancyRowsMatchRef` | The kernel inputs, and `tests/fuzz/fuzz_targets/kernels.rs` against the same references. |
| The kernel tests of `internal/codec` against their own scalar twins (`TestKernelIDCT*`, `fdct_simd_test`, `enccolor_*_test`, `huffprep_test`, `enc_block_asm_test`, the asm Huffman tests), `-tags purego`, `JPEG_NOSIMD` | `src/simd/tests.rs` (cross-tier) and the fuzz targets `simd_*`. The assembly Huffman path is not ported. |
| `TestCombinedRandomTables`, `TestCombinedLookupBitExact` | The `rh~` inputs: Go's combined table is the reference for the Rust one (Go compares it with its own plain table). |
| `TestDecodeDstMatchesOwned`, `TestDecodeDstTruncated`, `TestDecodeDstWindows` | The `dst` and `dst0` ops, and `tests/limits.rs`. |
| Zero-alloc, race, `Workers`, `Owned`, the limiter and cgroup tests, `tiletime`, benchmarks | Go runtime or API that does not exist here; `tests/parity/bench.sh` replaces the benchmarks. |
| `FuzzDecode`, `FuzzDecodeConfig`, `FuzzTables`, `FuzzDecodeCoefficients`, `FuzzRowReader`, `FuzzEncode`, `FuzzLedger` | `tests/fuzz/fuzz_targets/{decode,decode_config,tables,coefficients,row_reader,encode,ledger}.rs`, with the Go seeds; `FuzzLedger` is the `ledger` target (a counting allocator in the fuzz crate). The `row_reader` seed is also a parity input. |

The CI log prints `<n> inputs, <m> outputs` for the generator and the aarch64 subset. Every input has its own file: `oracle -gen` stops on a repeated name (a mutation tag that repeats gets a
counter), and the `go-run` stage fails unless the count it prints equals `ls /inputs | wc -l`.

## Operations and output files

Per `.jpg`: `cfg` (`DecodeConfigLimits`: `w h components ratio rgb restart_interval`), the decode
operations `yc1 rg1 gr1 yc2 rg2 gr2 yc4 rg4 gr4 yc8 rg8 gr8` (Format and Scale; a mutation runs `yc1 rg1
gr1 rg4`), `coef` (`DecodeCoefficientsLimits`), `rr` (`NewRowReader` read to the end), `rrn`
(`RowReaderNeed`), `rrx` (`Next` after the end and after `Close`), `sd1 sy1 sr` (a reused `Decoder`), `dst dst0` (`Dst` windows), `lb2` (`MaxBytes` 3 B/px) and, for a fixture, `lpx` (`MaxPixels` one under the frame), `lb0 lb1 lb2`
(`MaxBytes` 16 KiB, 128 KiB, 6 B/px + 64 KiB), `rrl`, `coefl`, `cx`, `coefx`, `rrcx` (the cancel flag set before the
call: decode, coefficients, row reader; the class is `cancelled` on both sides), `r<i>` for each line of the `.rect` sidecar, `ndpi-rg`, `ndpi-gr2`, `ndpi-reg`, `ndpi-reg2`, `ndpi-reg2g` for a `.starts`
sidecar, and `tb-yc`, `tb-rg` with `tables.bin` for `ab-` names. Per `.tbl`: `tb-cfg`, `tb-yc`, `tb-rg`,
`tb-coef`. Once per run, on three header-only gray frames made in the oracle (`cap-<w>x<h>.synth`):
`capd`, a decode under Go's zero `Limits` and Rust's `Limits::default()`, at Go's 1 GiB cap (the first
frame is the largest that fits and decodes to an empty plane; the others are refused as `cap`). Per `.img`: `e<q>` and `e<q>c` (Chroma444), `jpeg.Encode`, `ecx` (the same with the cancel flag set), and `se0 se0c se75 se75c` through a shared `Encoder`. Per `.fancy`: `fancy`.

Each writes `<input>.<op>.out`, or `<input>.<op>.err` with the error class and a newline. Where Go
returns an image with an error (`ErrTruncated`, `ErrAfterScan`, the coefficients, a `RowReader`
that fails on a later band) it also writes `<input>.<op>.partial.out` with it; the Rust crate returns
the same (`Decoded`, `DecodedCoefficients`), and the two must be equal.

### Wire formats

An image is a header line, then its visible samples row by row, without stride or block padding:
`rgba|gray|cmyk <min_x> <min_y> <max_x> <max_y>` or `ycbcr <ratio> <min_x> ...` (ratio 444, 422, 420,
440, 411, 410), then the rows (Y, then Cb, then Cr: each chroma row once, at the chroma samples of the
rect). Over 8 KiB of samples the body is `fnv <fnv1a64 of all> <len>` and one `r <fnv1a64>` per row, so
a difference shows which row. The same for streams (`jpeg\n`, rows of 1 KiB), the coefficients
(`coef <cfg>` and `c <h> <v> <bw> <bh>` lines; per component the 64 quantizer values then each block row,
int16 LE), the row reader (`rr <w> <h> <truncated>`, the rows of every band) and the kernels.

An `.img` is `[kind u8][w u16 LE][h u16 LE][ox u8][oy u8][pad u8]`, then the buffers of the parent image
(w+ox) x (h+oy), of which the encoder gets the sub-image (ox, oy, ox+w, oy+h): `kind` 0 RGBA, 1 Gray, 2 to
7 YCbCr 444, 422, 420, 440, 411, 410, 8 NRGBA; `pad` is added to every stride. A `.fancy` is records of `[op][n u16][other u16][dst u16]`, the row,
and for op 2 the other row; the destination starts as 0xAA. `E invalid-argument` is a record Go panics
on and Rust refuses.

## Error classes

| class | Go | Rust |
|---|---|---|
| `unsupported` | `ErrUnsupported` | `Error::Unsupported` |
| `truncated` | `ErrTruncated` | `Error::Truncated` |
| `afterscan:<class>[+truncated]` | `ErrAfterScan` and its cause | `Error::AfterScan { cause, truncated }` |
| `limit-pixels`, `limit-bytes` | `*LimitError` (`MaxPixels`, `MaxBytes`) | `Error::PixelLimit`, `Error::AllocLimit` |
| `cancelled` | `context.Canceled` | `Error::Cancelled` |
| `region` | `jpeg: region ... outside +-2^29` | `Error::Region` |
| `region-empty` | `jpeg: region ... does not intersect image` | `Error::RegionEmpty` |
| `cap` | `jpeg: decode needs N bytes, over the 1 GiB cap` (default `Limits`, the `capd` op only) | `Error::AllocLimit` under `Limits::default()` |
| `closed` | `RowReader is closed` | `Error::RowReaderClosed` |
| `invalid-argument` | a panic of a kernel on a bad length | `Error::InvalidArgument` |
| `unexpected-eof` | `io.ErrUnexpectedEOF` | `Error::UnexpectedEof` |
| `oom` | none: an allocation failure is a Go runtime fatal error | `Error::OutOfMemory` (a failed `try_reserve`) |
| `bad-image` | none: the encoder given a zero-size image, an inverted rect or a plane shorter than its rect panics (index out of range) | `Error::BadImage` |
| `io` | the `io.Writer`'s own error, returned as it is | `Error::Io(kind)` (`encode_to`; the `io::ErrorKind` only) |
| the message, e.g. `jpeg: missing SOI` | the marker and decode errors | the same variants (`Display` repeats Go's message) |
| `jpeg: image dimension over 65500` | the decoder's `errTooBig`; the encoder has no such check: `put16` writes the low 16 bits of a side over 65535 into the SOF | `Error::TooBig`, from the decoder, and from the encoder for a side over 65535 |

`oom`, `bad-image`, `io` and the encoder's `TooBig` are Rust-only: no parity input reaches them (the
oracle's inputs are bounded, and Go cannot be fed a hostile rect without panicking). `bad-image` and
the encoder's `TooBig` are tested in `tests/encode.rs` and `tests/limits.rs` and fuzzed by the `encode`
target; `io` is tested in `tests/encode.rs` with a failing writer; `oom` needs a failing allocator
and has no test.

## Benchmark

`tests/parity/bench.sh` (the `bench` workflow; not a required check) runs `oracle -bench` and
`examples/bench.rs` in one container on the `jpg/` fixtures. Go and Rust time the same calls, the best of
five batches of three runs: decode to RGBA, YCbCr and Gray at scale 1 and RGBA at scales 4 and 8 (Go's
package-level `jpeg.Decode`, which returns a copy the caller owns, against `decode`, which returns an
owned image), RGBA at scale 1 into a buffer the caller keeps (`Decoder.Decode` with `Dst` against
`decode_into_rgba`), the row reader, the coefficients, and encode at quality 75 of RGBA at 4:2:0 and
4:4:4, of Gray and of YCbCr (a reused `Encoder` against `encode`). The Go decode of the RGBA classes uses `Owned`, so it returns its image uncopied as Rust's does. The class table has three decimals, and a second table gives the ratio per output size (under 1.5 MB of output, which holds the 512x512 tiles, and above). It prints ms per MB of output per
class and the ratio, and per file the worst ratios and the big inputs (stderr). `tests/parity/profile.sh`
runs one operation under callgrind (the `profile` workflow reads `tests/parity/profile.txt`). Both
benches also run a generated input after the files, `gen_smooth_gray_1536x64` (a gray ramp, quality 95,
encoded by each side), and print its length and md5 on stderr; the two lines must show the same stream.
