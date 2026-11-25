# Threading

`max_threads` is the most threads one call uses, the calling thread included: 1 runs on the
calling thread and never touches a pool, 0 counts as 1 and more than 256 as 256. The output and the
error are the same for every value and every pool, and equal to Go's. Two decode steps use threads (Go's
two parallel sites): the entropy decode of a baseline scan with a DRI in bands of whole MCU rows, as
Go's `decodeBands` (a band that loses data or ends off its restart marker sends the whole scan back to
the serial decode), and the YCbCr to RGBA upsampling in row stripes. The workers are the threads of a
rayon pool that outlive the call, so a pass costs a wake-up, not a thread start: a band decode gets one
worker per 8 KiB of entropy data and the colour conversion one per 131072 pixels (so a 512x512 tile
gets two). A region decode with `DecodeOptions::restart` offsets (how NDPI slides are read) is
banded too, where Go decodes it serially: the rows of the region are cut into bands at rows where the
serial decode would jump to the offset of a restart interval anyway, so each band lands on the same
byte, and the colour conversion of such a region runs in stripes. A region with no offsets, one as
wide as the frame, one with `check_frame`, and any region whose offsets are not truthful at the cut
(RST(k-1) not just before offset k) stay serial, and a band that loses data or ends short sends the
whole scan back to the serial decode, as for a whole frame. Scaled decodes (scale 2, 4 or 8, `pyramid` included) of restart streams are never banded, because `bandable()` requires 8x8 blocks: with more threads they take about the single-threaded time, while a full decode runs about 10x faster (`strip_rst`). The output is the serial output for every
thread count (`tests/threads.rs`, the `decode` fuzz target). `RowReader`, the encoder and `decode_coefficients` are serial, as in Go.
The scratch of the extra workers (the band offsets, their colour rows) is held in the ledger only while
they run, and a call that cannot hold it runs on fewer workers instead, so `max_alloc` bounds the whole
call and never turns into an error a single-threaded call would not give. The thread stacks (512 KiB)
and a pool's bookkeeping belong to the pool, not to a call, and are not charged to `max_alloc`; a
caller who budgets memory should count them with the pool (the shared pool is started once and lives
for the process). `max_alloc` and the cancel flag cover every worker; `decode_probe` (hidden) reports
a call's charged peak and the workers it used (1 for `max_threads` 1), which `tests/resources.rs`
checks against both limits, and `shared_pool_started` (hidden) says whether the shared pool has
started, which `tests/pool.rs` checks.

A threaded call still allocates a few small job lists and one boxed rayon task per extra worker, also
on a reused `Decoder`; at `max_threads` 1 a reused `Decoder` allocates nothing after its first call of a
frame size (`tests/reuse_alloc.rs` asserts both; [performance.md](performance.md)). Under
`panic = "abort"` a panic in a job stops the process, where otherwise the scope re-raises it on the caller.

The pool and `max_threads` rules are also in [limits.md](limits.md); the bench numbers per thread count are in [benchmarks.md](benchmarks.md).

`DecodeOptions::pyramid` adds no thread use: its chroma blur is one serial pass over the chroma planes
before the (striped) colour conversion, and the output is the same for every `max_threads`, which
`tests/pyramid.rs` and the `pyramid` fuzz target check.
