# Benchmarks

`tests/parity/bench.sh` (workflow `bench.yml`, on `main` and on dispatch; not required) times Go and Rust on the
`jpg/` fixtures in one container and prints the ratio per class (decode to RGBA, YCbCr, Gray and at
scales 2, 4 and 8, the row reader, coefficients, encode). The result is in the job summary.

Build profile, `target-cpu=x86-64-v4`, `Decoder` reuse and PGO/BOLT: [performance.md](performance.md).

Each table of `bench.sh` also has three v4 columns: Go built with `GOAMD64=v4` (`oracle-v4`) and Rust
built with `RUSTFLAGS="-C target-cpu=x86-64-v4 -C target-feature=+false-deps-tzcnt,+false-deps-bls"` (`bench-v4`, its own cargo cache mount), each in ms
per MB of output and the ratio of the two, next to the default build's columns. The v4 and default
builds must print the same output bytes for every file, or the script fails; the v4 Rust output is
also checked byte for byte against libjpeg-turbo by `tests/golden.rs` and `tests/encode.rs` in the
`ci` stage (the v4 tests run only on a runner with AVX-512). The v4 numbers are in the tables below. `pgo.yml` (dispatch only) times the
default, PGO and PGO plus BOLT builds of the bench example; its table is in
[performance.md](performance.md).

## Performance

`tests/parity/bench.sh` (workflow `bench`, one container for both sides) times the Go and Rust decode and
encode of the `jpg/` fixtures and prints ms per MB of output per class, best of three runs; the weights are the fixtures' output
sizes, so the big images (a 2048x2048 DICOM tile, a 1024x1120 restart strip, the 512x512 tiles) set the
totals. Each class compares the same call on both sides. The `dec-*` classes run on one reused
decoder per class and file (Go's `jpeg.Decoder`, Rust's `jpeg::Decoder`): both keep their planes and
output between calls, neither copies and neither pools (Go's package-level `Decode` pools its
decoders but copies a YCbCr or Gray result, which Rust does not, so it is not used). `dec-into` is
Go's `Dst` against `decode_into_rgba`, on the same reused decoders. `rowreader`, `coef` (entropy
decoding only, on a fresh decoder per call as on both sides) and `enc-*` (quality 75, a reused
`Encoder` on both sides) likewise. The tables are in two passes: single-threaded (Go `Workers` 0,
Rust `max_threads` 1) and default threads (Go `Workers` = `GOMAXPROCS`, Rust `max_threads` =
`available_parallelism`, 16 on that machine). Rust ms/MB over Go's, best of three runs, single-threaded
then default threads, each for the default build and the v4 build.

### Hosted baseline

`bench.yml` run 45712772708 on a GitHub-hosted `ubuntu-24.04` runner (2 vCPU, AVX2, BMI2 and
AVX-512 F/BW/VL/VNNI), the baseline for later hosted runs. Rust ms/MB over Go's per class. Shared
VMs are noisy, and with 2 vCPU the default-threads pass has little parallel work. A hosted x86_64
runner may have no AVX-512, in which case the v4 columns are skipped.

| class | 1 thread | 1 thread v4 | default threads | default threads v4 |
|---|---|---|---|---|
| coef | 0.901 | 0.856 | 0.888 | 0.868 |
| dec-gray | 0.915 | 0.883 | 0.900 | 0.888 |
| dec-into | 0.919 | 0.892 | 0.896 | 0.879 |
| dec-region | 0.959 | 0.924 | 0.638 | 0.614 |
| dec-rgba | 0.911 | 0.881 | 0.897 | 0.882 |
| dec-rgba2 | 0.969 | 0.938 | 0.954 | 0.929 |
| dec-rgba4 | 0.873 | 0.847 | 0.862 | 0.846 |
| dec-rgba8 | 0.907 | 0.875 | 0.900 | 0.876 |
| dec-ycc | 0.930 | 0.898 | 0.910 | 0.897 |
| enc-gray | 0.953 | 1.015 | 0.950 | 1.020 |
| enc-rgba420 | 1.162 | 1.181 | 1.158 | 1.198 |
| enc-rgba444 | 1.065 | 1.080 | 1.061 | 1.106 |
| enc-ycc | 1.102 | 1.150 | 1.099 | 1.150 |
| rowreader | 0.898 | 0.814 | 0.832 | 0.823 |
| all | 0.940 | 0.911 | 0.918 | 0.912 |

On this runner the encoder classes are slower than Go (up to 1.20); on the dedicated machine
below they were not. This is a new baseline, not compared with the old numbers.

### Earlier result (dedicated machine)

The numbers below were measured on the old dedicated machine (an AMD Ryzen 7 9800X3D with AVX2,
BMI2 and AVX-512), not on a hosted runner.

| class | 1 thread | 1 thread v4 | default threads | default threads v4 |
|---|---|---|---|---|
| coef | 0.92 | 0.89 | 0.91 | 0.88 |
| dec-gray | 0.93 | 0.90 | 0.90 | 0.89 |
| dec-into | 0.90 | 0.88 | 0.90 | 0.87 |
| dec-region | 0.95 | 0.92 | 0.12 | 0.12 |
| dec-rgba | 0.91 | 0.88 | 0.88 | 0.87 |
| dec-rgba2 | 0.93 | 0.91 | 0.93 | 0.91 |
| dec-rgba4 | 0.87 | 0.85 | 0.86 | 0.84 |
| dec-rgba8 | 0.92 | 0.90 | 0.92 | 0.89 |
| dec-ycc | 0.94 | 0.91 | 0.92 | 0.91 |
| enc-gray | 0.86 | 0.88 | 0.86 | 0.87 |
| enc-rgba420 | 0.96 | 0.95 | 0.96 | 0.96 |
| enc-rgba444 | 0.93 | 0.92 | 0.92 | 0.91 |
| enc-ycc | 0.95 | 0.96 | 0.94 | 0.96 |
| rowreader | 0.84 | 0.76 | 0.77 | 0.75 |
| all | 0.91 | 0.88 | 0.87 | 0.86 |

The totals are weighted by output bytes, so a few big files carry them and the outliers below hide in
them. Per output size (under and over 1.5 MB) the rows run from 0.75 (`rowreader`) to 1.03
(`enc-ycc` under 1.5 MB, both builds); `dec-region` is 0.95 single-threaded and 0.12 with threads,
where the restart bands run it in parallel. Scaled decodes (scale 2, 4, 8, pyramid) of restart streams are not banded (`bandable()` needs 8x8 blocks), so threads do not speed them up (`strip_rst` at scale 2 takes the single-threaded time, a full decode about a tenth). The files that are slower than Go (the
`bench.sh` log, on stderr, lists the 60 worst and then every file over 1.10x Go in each build):

- Tiny frames (8x8 to 17x9 and the 5x1 and 15x1 strips: `synth_ac_overflow_8x8`, the multiscan,
  restart and progressive small fixtures) take 1.66 to 2.92 times Go's time in every `dec-*` class,
  a fixed cost of about 1 to 5 microseconds per call that a frame of a few hundred bytes of output
  cannot hide. They weigh nothing in the totals.
- `ndpi_wide_4096x32` is 1.00 to 1.05; `gen_smooth_gray` is 1.04 in `dec-gray`.
- `cmp_evil_q255` (below, not in the totals) is 0.78 to 0.97 in `dec-*` and `rowreader` now, but
  `enc-ycc` is 1.05 (v4 1.04) and `enc-rgba444` 1.03 (v4 1.01) there.
- `dicom_2048x2048` is below 1 in `dec-rgba` and `dec-into` (0.88 in this run); `svs_sparse_420`
  is 0.73 to 0.84 at default threads.

The first run of this benchmark (the scalar port with AVX2 kernels) had decode at 1.1 to 1.25 and
encode at 2.4 to 2.9 times Go. What closed the gap, each in its own pull request and each checked
against Go by the parity job: Go's bit writer (a 64-bit accumulator spilled a word at a time, with a
byte-wise 0xFF check) and block coder with the prep in SIMD; a reciprocal quantizer; the FDCT as
libjpeg-turbo's AVX2 kernel (16-bit lanes, two rows to a register); the 16-pixel colour kernels; the
AVX-512 kernels Go has for the encoder (two blocks per FDCT, 16 pixels per colour step, the zigzag
permute); and on the decode side the combined Huffman table (a short code and its value bits, and EOB
and ZRL, in one lookup) and the BMI build of the MCU loop. Per-block costs were found with `callgrind`
(`tests/parity/profile.sh`, the `profile` workflow, a development tool and not a check).

What closed the decode gap in the latest round (each step checked against Go by the parity job):
the `Decoder` (the planes and the RGBA image were allocated and zeroed on every call, Go's are
reused); the entropy refill's common case (six bytes with no 0xFF) inline in the MCU loop, with the
byte-by-byte case out of line; the search for the next marker in a restart stream eight bytes at a
time (it was a byte loop, about 9% of `strip_rst`); a DC-only test with no early exit, which
vectorises; the MCU's row and column kept in step instead of divided out; and the tail of a scan with
no restart intervals decoded by the fast path with a roll-back, as between two restarts, when a real
marker ends it.

The tiff-rs strip gap: tiff-rs's `jpeg-grey-strip` bench (16 gray 1536x64 strips, quality 95, smooth
content `byte(2x+y)`) ran at 1.22 times Go raw at one thread. A profile put the gap in instruction count, not in
memset and not in a missing Go gray fast path (Go has none): about 6 µs of it in entropy decoding
(long Huffman codes through an out-of-line, bit-at-a-time `slow_huff`, and an out-of-line
`Fast::value`) and about 5 µs in the IDCT and its stores (a bounds check on each row store and a
pass 2 with extra rounding and +128 work). Every block of this stream has AC, so the DC-only exit
does not help. Two changes, both with
output byte-identical to before: the DC-only test looks at coefficients 1 and 8 before it folds the
other 61 (Go's loop stops at the first AC; the fold has no early exit and a ramp block has AC in
those two), and the AVX2 `idct8` transposes once, not twice (the pass-2 output is packed to bytes
and the bytes are transposed, as Go's `idct8AVX2Asm`). To measure this, both benches generate the
input `gen_smooth_gray_1536x64` (a 1536x64 ramp `byte(2x+y)`, encoded at quality 95 by each side's own
encoder) after the files, and print its length and md5 on stderr, which must agree. It is in the
`dec-gray` and `coef` lines of `tests/parity/profile.txt`. Later changes, all byte-identical: `Fast::slow_huff` shifts the code out of the left-aligned
buffer (as Go's `SLOWHUFF`, no refill and no state copy), `symbol` and `value` are inlined, the
AVX2 `idct8` stores its eight rows after one bounds check and runs pass 2 with the constants
scaled by 16 so the range limit is one `sra` and a signed pack with a sign flip, as Go's asm.
Result (one thread): `dec-gray` of this input went from 1.20 to 1.04 times
Go, `coef` from 1.05 to 0.91.

`dec-region` is one fixture, a 512x512 region of `strip_rst` with its restart offsets
(`tests/data/strip_rst.starts`), the call a slide viewer makes; Go's `Region` and `Restart` options
run it serially, Rust bands it. Rust ms/MB over Go's: 0.95 single-threaded and 0.12 at default threads
(v4: 0.92 and 0.12).

Threads: the threaded passes (restart bands, then colour stripes) run on a pool that outlives the
call (the caller's `ThreadPool`, or the crate's lazy shared one; see [limits.md](limits.md)), so a call
pays no thread spawn. A call takes one worker for each 8 KiB of entropy data in a restart stream and
one for each 131072 pixels of colour conversion, at most `max_threads` (the caller included). Small
frames under those amounts run on the calling thread, and frames above them pay no thread start. Exception: scaled decodes of restart streams never use restart bands, pyramid mode included (`bandable()`
needs 8x8 blocks), so extra threads do not speed them up: a scale 2 decode of `strip_rst` takes about
as long with 16 threads as with one, where a full decode is about 10 times faster.

The IDCT for blocks outside the 16-bit range (`cmp_evil_q255`, a high-quantisation fixture) used to
fall back to the scalar code, 1.17 to 1.63 times Go in `dec-*`. It now runs in 32-bit lanes on AVX2
and NEON for every block (see [safety.md](safety.md)), and the scale 2 IDCT is SIMD too, with a new
`dec-rgba2` class (0.93 over Go). `cmp_evil_q255` stays out of the tables, since Go's 16-bit kernels give other
bytes there. Its ratios with these changes are 0.78 to 0.97 in decode and 0.95 in `dec-rgba2`. The scale 4 and 8 IDCTs (Go has none either) and aarch64's encoder block coder are not SIMD. A
dependent crate's own release profile applies to this code (`codegen-units = 1` here only affects
this package's examples), so these are the numbers of this repository's build. A dependent that wants
them sets `[profile.release.package.jpeg] codegen-units = 1` in its own workspace `Cargo.toml`; with
the default 16 units the ratios are somewhat higher. The per-file ratios are in the bench log (stderr),
since the class totals are weighted by output bytes.

AVX-512 in decode: the colour conversion (32 pixels a step) and the h2v1 and h2v2 fancy upsample
(64 and 32 columns a step) have AVX-512 kernels, where Go's decode kernels are AVX2 only. Gain on the
512x512 `dec-rgba` files, single-threaded: 5.6% on `svs_420_512x512`, 12.3% on
`svs_sparse_420_512x512` and 8.6% on `svs_sparse_422_512x512`.

## Pyramid mode

`dec-rgba2`, `dec-rgba2-pyr`, `dec-rgba4` and `dec-rgba4-pyr` time the scale 2 and 4 decodes without and
with `DecodeOptions::pyramid`. Go has no counterpart, so `bench.sh` prints them as a Rust-only table
(ms per MB of output, best of three), and the join with Go's rows leaves them out of the totals above. The
`-pyr` rows leave out the fixtures the mode refuses (under 2 or 4 samples in a direction). The
cost of the mode over the plain scaled decode (single-threaded, then default
threads): scale 2 is 3.008 against 2.791 ms/MB (+7.8%) and 2.966 against 2.772 (+7.0%); scale 4 is
11.846 against 11.592 (+2.2%) and 11.778 against 11.532 (+2.1%). The mode is not compared with a full
decode followed by an average of 2x2 blocks: this crate has no such function, so no class times it.
