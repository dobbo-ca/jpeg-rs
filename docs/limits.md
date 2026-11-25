# Resource limits

Every entry point that takes input of any size takes `&Limits` first, and a `&AtomicBool` cancel flag
where it can run long. A constrained call, with no pool touched and at most 64 MiB for the whole call:

```rust
let limits = Limits { max_threads: 1, max_alloc: 64 << 20, ..Limits::default() };
let d = jpeg::decode(bytes, &DecodeOptions::default(), &limits, &cancel)?;
```

| Field | Type, unit | Default | Over the limit |
|---|---|---|---|
| `max_pixels` | `u64`, pixels (width times height) | 2^32 (a frame is at most 65500 on a side) | `Error::PixelLimit { got, max }`, from the SOF before any allocation |
| `max_alloc` | `u64`, bytes | 1 GiB | `Error::AllocLimit { got, max }`; the allocator's own refusal is `Error::OutOfMemory` |
| `max_threads` | `usize`, threads, the calling thread included | `std::thread::available_parallelism()` | none: the call uses fewer threads and gives the same output |

- `max_alloc` is the enforced peak of the whole call, every thread together: memory is charged before
  it is allocated, so the peak never passes it, with one exception: under the default (1 GiB) the
  ledger's budget is the larger of `max_alloc` and the checked need of the frame, so a frame that
  Go's cap lets through may peak up to 1 GiB plus the page rounding (see below). It is a `u64`, not
  a `usize`, because tiff-rs passes a `u64` budget down to it.
- `max_threads = 1` runs on the calling thread and never touches a pool; 0 counts as 1 and more than 256
  as 256. The default is the machine's available parallelism (Go's default is serial).
- Threads are used by `decode`, `decode_into_rgba` and `Decoder` only: the entropy decode of a baseline
  scan with a DRI in bands of restart intervals, and the YCbCr to RGBA conversion in row stripes. Each
  pass takes one worker per 8 KiB of restart-interval data or per 131072 pixels of colour conversion,
  at most `max_threads` (a region decode with restart offsets counts only the bytes of the intervals the region needs), so a frame under about 16 KiB of restart data and 262144 pixels runs on the
  calling thread. Each threaded pass opens its own scope on the persistent pool: the pool's threads
  outlive the call, so a second pass costs a wake-up, not a thread start. `RowReader`,
  `decode_coefficients`, `decode_config` and the encoder are serial.
- A reused `Decoder` keeps its buffers between calls (planes, RGBA image, the coefficient blocks of
  progressive and multi-scan frames, the colour rows); the call is charged for all it uses, reused or
  not. The kept memory is not counted against any later call and is freed when the `Decoder` drops.
- The workers are a `jpeg::ThreadPool`. Pass your own (`Decoder::with_pool`, or `DecodeOptions::pool`
  for any call): it always wins. Without one, a call with `max_threads` above 1 uses the crate's shared
  pool, a `OnceLock` built on the first call that has a pass worth threading, with
  `available_parallelism` threads at most and no other state. `max_threads` 1 never starts or touches
  it. Each call takes at most `max_threads` workers from either pool, the calling thread included, and
  never more than the pool's threads plus the caller.
- Cancellation: set the `AtomicBool` from any thread and the call returns `Error::Cancelled` at its
  next poll, which is once per MCU row, and every 16 rows in the whole-image colour conversion, the
  region padding and the encoder's YCbCr-to-RGBA copy (details below). A caller's own `to_rgba` hook
  runs to its end between two polls.
- Every public function's rustdoc has an `# Errors` section naming the limits it enforces and whether
  it may spawn threads.

## Defaults, charging and cancellation

Every entry point takes `&Limits { max_pixels, max_alloc, max_threads }`. Go's `Limits` had the first
two, unbounded at zero; the defaults here are finite: `max_pixels` 2^32 (a frame is at most 65500 on a
side), `max_alloc` 1 GiB and `max_threads` `std::thread::available_parallelism()`. `max_alloc` is a
`u64`, not a `usize`, because tiff-rs passes a `u64` budget down to it.
The default `max_alloc` stands for Go's unset `MaxBytes`, whose only bound is a fixed 1 GiB cap on one
decode, checked on the need less the page rounding: a `decode` under the default refuses
(`Error::AllocLimit`) exactly the frames Go does, and the ledger still enforces: its budget is the
larger of `max_alloc` and that checked need, so the peak stays under 1 GiB plus the rounding. Any
other `max_alloc` is Go's `MaxBytes`: the whole need and every charge count against it, and it may be
above 1 GiB, which Go cannot do. Memory is charged to `max_alloc`
before it is allocated, in whole 8 KiB pages as Go's ledger does (`Error::AllocLimit`), from the SOF
and never from an input-controlled size unchecked; allocation failures are `Error::OutOfMemory`.
Cancel: `decode`, `decode_coefficients`, `RowReader` and `encode` take a
`&AtomicBool` and check it once per MCU row, and the whole-image colour conversion, the region
padding and the encoder's YCbCr-to-RGBA copy every 16 rows, and the
RGB-to-gray pass and the 4:2:0 overread patch likewise (`Error::Cancelled`). A `to_rgba` hook of
the caller's own runs to its end between two polls.

Threads: see [threading.md](threading.md).

`DecodeOptions::pyramid` is charged as the scaled decode it is, plus one scratch row of the chroma
width for the blur (charged to the ledger when the colour pass starts; `decode_need` does not add
it, the 64 KiB fixed-state allowance of the up-front need covers it, as the row is at most 65535
bytes (a JPEG's width), and the ledger still refuses it when `max_alloc` cannot hold it). The blur works in
place on the planes and polls the cancel flag every 16 rows or columns, and the floor-size crop
allocates nothing. With a scale of 2, 4 or 8 it returns `Error::Unsupported`, before any allocation,
for a `region`, `Format::YCbCr`, a `Recon` or a frame smaller than the scale in either direction.

## Other allocations and exceptions

Allocations the ledger does not see are fixed state under 64 KiB. The encoder charges its output
vector to the same ledger as its strips (the output may grow to `max_alloc` less what they hold): a
row is refused before its worst case (`row_bound`, up to 12.5 MiB for a 65535-wide 4:4:4 row) is
reserved, and the end of the stream is checked again before EOI.

Every public function that takes input of any size takes `&Limits`, `row_reader_need`,
`max_row_reader_need` and the fancy-upsample rows among them. One exception, recorded here: the
`#[doc(hidden)]` `kernel_check` functions are the fuzz targets' differential checks, not API.

`RowReader::new` also refuses a frame whose `row_reader_need` is over `ROW_READER_MAX_NEED`
(1100 MiB, Go's cap); a caller that wants frames above 1 GiB sets `max_alloc` to it.
