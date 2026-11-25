//! JPEG (ITU-T T.81) decoder and encoder for whole slide image tiles.
//!
//! The default decode matches libjpeg-turbo 3.2.0 bit for bit (islow IDCT, fancy upsampling).
//! A Rust port of the Go library jpeg-go; see `docs/go-differences.md` for how the API differs
//! from Go's.
//!
//! # Resource limits
//!
//! Every entry point that reads input of any size takes [`Limits`] and, where it can run long, a
//! cancel flag (`&AtomicBool`). Set them first:
//!
//! ```
//! use jpeg::{DecodeOptions, EncodeImage, EncodeOptions, Format, Limits, Rect, Rgba};
//! use std::sync::atomic::AtomicBool;
//!
//! # fn main() -> Result<(), jpeg::Error> {
//! // At most one thread (no pool is touched) and 64 MiB for the whole call.
//! let limits = Limits { max_threads: 1, max_alloc: 64 << 20, ..Limits::default() };
//! let cancel = AtomicBool::new(false);
//!
//! // A small JPEG to decode: a 16x16 RGBA image, encoded under the same limits.
//! let img = Rgba { pix: vec![128; 16 * 16 * 4], stride: 16 * 4, rect: Rect::new(0, 0, 16, 16) };
//! let bytes = jpeg::encode(EncodeImage::Rgba(&img), &EncodeOptions::default(), &limits, &cancel)?;
//!
//! let o = DecodeOptions { format: Format::Rgba, ..Default::default() };
//! let d = jpeg::decode(&bytes, &o, &limits, &cancel)?;
//! assert_eq!(d.image.bounds(), Rect::new(0, 0, 16, 16));
//!
//! // A frame over `max_pixels` is refused from its headers, before anything is allocated.
//! let tiny = Limits { max_pixels: 100, ..limits };
//! assert!(matches!(
//!     jpeg::decode(&bytes, &o, &tiny, &cancel),
//!     Err(jpeg::Error::PixelLimit { .. })
//! ));
//! # Ok(())
//! # }
//! ```
//!
//! * `max_pixels` (default 2^32): the most pixels of a frame, checked from the SOF.
//!   Over it: [`Error::PixelLimit`].
//! * `max_alloc` (default 1 GiB, a `u64` of bytes, not a `usize`, because tiff-rs passes a `u64`
//!   budget): the enforced peak of the whole call, all threads together. Memory is charged
//!   before it is allocated, in whole 8 KiB pages. Over it: [`Error::AllocLimit`]. A refusal
//!   by the allocator itself is [`Error::OutOfMemory`]. Under the default, the ledger's budget
//!   is the larger of `max_alloc` and the checked need of the frame (the frames Go's 1 GiB cap
//!   lets through), so the peak may reach 1 GiB plus the page rounding; for any other value
//!   the peak never passes it.
//! * `max_threads` (default [`std::thread::available_parallelism`]): the most threads of one
//!   call, the calling thread included. 1 runs on the calling thread and never touches a pool.
//!   Only [`decode`], [`decode_into_rgba`] and [`Decoder`] use threads (restart bands of a
//!   baseline scan with a DRI, and colour stripes, as many workers as the input is big enough to
//!   feed, and only when the call's scratch fits in `max_alloc`). [`RowReader`],
//!   [`decode_coefficients`] and the encoders are serial. The output and the error are the same
//!   for every value, so there is no error for it.
//! * The workers are a [`ThreadPool`]: the caller's (`DecodeOptions::pool`, [`Decoder::with_pool`])
//!   or the crate's lazy shared one, started by the first call that has a pass worth threading,
//!   with at most `available_parallelism` threads and no other state. A pool's thread stacks are
//!   the pool's, not charged to `max_alloc`.
//! * Cancellation: set the `AtomicBool` from any thread; the call returns [`Error::Cancelled`] at
//!   its next poll (once per MCU row, every 16 rows in the whole-image colour and copy passes).
//!   A call that has finished is not affected.
//!
//! Every public function documents the limits it enforces under `# Errors`. The hidden
//! `kernel_check` functions take no `Limits`: they work on fixed-size arrays, allocate nothing
//! from input and use no thread.
//!
//! AVX2 (x86_64) and NEON (aarch64) kernels run the hot loops, selected once at run time and
//! bit-exact with the scalar code; see [`simd_tier`].
#![deny(
    unsafe_code,
    unsafe_op_in_unsafe_fn,
    clippy::undocumented_unsafe_blocks
)]
// The ported code keeps Go's index loops and long argument lists.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]
#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::todo,
        clippy::unimplemented
    )
)]

mod coef;
mod dct;
mod decoder;
mod encoder;
mod error;
mod huff;
mod image;
#[doc(hidden)]
pub mod kernel_check;
mod limits;
mod marker;
mod par;
mod progressive;
mod rowreader;
mod sample;
mod scan;
mod simd;

pub use coef::{
    Coefficients, ComponentCoefficients, DecodedCoefficients, decode_coefficients,
    decode_coefficients_any_components,
};
pub use decoder::{
    ColorModel, Config, DecodeOptions, Decoded, Decoder, Format, Probe, Recon, RgbaWindow,
    ToRgbaHook, decode, decode_config, decode_into_rgba, decode_probe,
};
pub use encoder::{EncodeImage, EncodeOptions, Encoder, encode, encode_to};
pub use error::Error;
pub use image::{Cmyk, Gray, Image, Rect, Rgba, Subsample, YCbCr};
pub use limits::Limits;
pub use par::{ThreadPool, shared_pool_started};
pub use rowreader::{ROW_READER_MAX_NEED, RowReader, max_row_reader_need, row_reader_need};
pub use sample::{h2v1_fancy_row, h2v2_fancy_row};
pub use scan::IdctHook;
pub use simd::simd_tier;
