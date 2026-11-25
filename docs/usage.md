# Usage

The default decode matches libjpeg-turbo 3.2.0 (islow IDCT, fancy upsampling) bit for bit, and the
default encode matches `cjpeg -quality Q` and `tj3Compress8` byte for byte. This is the scalar
port: every Go package and public API is here, with AVX2 and NEON kernels beside the scalar code
(see [safety.md](safety.md)). `tests/parity/` runs jpeg-go itself on the same inputs and diffs the outputs, `tests/fuzz/` has the
fuzz targets and `tests/parity/bench.sh` the benchmark against Go (see [parity.md](parity.md), [fuzzing.md](fuzzing.md) and [benchmarks.md](benchmarks.md)).

```rust
use jpeg::{DecodeOptions, EncodeImage, EncodeOptions, Format, Limits, Rect};
use std::sync::atomic::AtomicBool;

// At most 2 threads and 256 MiB for the whole call, threads included.
let limits = Limits { max_threads: 2, max_alloc: 256 << 20, ..Limits::default() };
let cancel = AtomicBool::new(false);

// Decode. `error` is Some(Truncated) or Some(AfterScan { .. }) when Go returns an image with
// an error: the image is still usable.
let d = jpeg::decode(bytes, &DecodeOptions { format: Format::Rgba, ..Default::default() }, &limits, &cancel)?;
let image = d.image;

// A run of decodes on one Decoder reuses its buffers (no zeroing; with max_threads 1 the second
// decode of a frame of the same shape allocates nothing, see performance.md):
let mut dec = jpeg::Decoder::new();
let d = dec.decode(bytes, &DecodeOptions { format: Format::Rgba, ..Default::default() }, &limits, &cancel)?;
let pixels = &d.image; // valid until the next call on `dec`

// Scale 1/2, 1/4, 1/8 inside the IDCT, a region of the scaled output, restart offsets
// (NDPI tag 65426), an abbreviated tables stream (TIFF tag 347):
let o = DecodeOptions {
    scale: 4,
    region: Some(Rect::new(10, 10, 200, 100)),
    tables: Some(jpeg_tables),
    restart: Some(&starts),
    ..Default::default()
};

// A region with `restart` offsets on a restart stream decodes in bands on up to `max_threads`
// threads, with the same pixels and error as one thread.

let cfg = jpeg::decode_config(bytes, None, &limits)?;                      // shape, no allocation
let c = jpeg::decode_coefficients(bytes, None, &limits, &cancel)?;         // quantized blocks
let mut rows = jpeg::RowReader::new(bytes, None, &limits, &cancel)?;       // RGBA, one MCU row per next()

let jpeg_bytes = jpeg::encode(EncodeImage::Rgba(&rgba), &EncodeOptions { quality: 75, chroma444: false }, &limits, &cancel)?;
```

A constrained call, with no pool touched and at most 64 MiB for the whole call:

```rust
let limits = Limits { max_threads: 1, max_alloc: 64 << 20, ..Limits::default() };
let d = jpeg::decode(bytes, &DecodeOptions::default(), &limits, &cancel)?;
```

## Colour model

A 3-component stream says how its samples map to RGB only through markers: JFIF means YCbCr, an
Adobe APP14 segment with transform 0 means RGB, component ids R, G, B mean RGB, and anything else is
read as YCbCr (libjpeg-turbo's `default_decompress_parms`, and Go). `DecodeOptions::color_model`
picks the reading:

- `ColorModel::Auto` (default): the markers, as above. Unchanged from earlier releases.
- `ColorModel::Rgb`: RGB as stored, no colour transform, whatever the markers say. Use it when the
  container says the samples are RGB: a TIFF strip or tile with Compression 7 and
  PhotometricInterpretation 2 (Aperio SVS stores its JPEG tiles this way, with no Adobe segment).
  libtiff (`JPEGPreDecode` sets `JCS_UNKNOWN`, no conversion) and OpenSlide (`jpeg_color_space`
  from the Photometric tag) decode such tiles the same way. The components must share one sampling,
  else `Error::Unsupported` (libtiff refuses such a tile too). `Format::YCbCr` is
  `Error::Unsupported`; `Format::Gray` is turbo's `rgb_gray`. One or four components: no effect.
- `RowReader::with_color_model` is `RowReader::new` with the model; `row_reader_need` bounds both.
- `decode_config` reads the markers only: its `Config::rgb` does not know the option.

```rust
let o = DecodeOptions { format: Format::Rgba, color_model: ColorModel::Rgb, ..Default::default() };
```

## Pyramid levels

`DecodeOptions::pyramid` is for building an image pyramid from JPEG tiles: with `scale` 2, 4 or 8 and
`format` Rgba or Gray, the decode returns level 1, 2 or 3 of the pyramid in one pass, `floor(W / scale)`
by `floor(H / scale)`, without decoding the tile at full size.

```rust
let o = DecodeOptions { format: Format::Rgba, scale: 2, pyramid: true, ..Default::default() };
let level1 = jpeg::decode(bytes, &o, &limits, &cancel)?; // floor(W/2) x floor(H/2)
```

- The IDCT already computes the 2x2, 4x4 or 8x8 mean of each block (libjpeg-turbo's reduced IDCTs), so
  the luma is the mean of the full decode to within 1 to 4 levels (about 54 dB). The chroma is the gap:
  a 4:2:0 frame at scale 2 or 4 has its chroma at the output's own size, with no fancy upsampling,
  where the reference path (decode at full size, then average) upsamples it first. The pyramid mode
  blurs that chroma before the colour conversion: `[1 6 1] / 8` on each subsampled axis at scale 2 (the
  exact mean of the four fancy-upsampled samples around a chroma sample), `[1 14 1] / 16` at scale 4, none
  at scale 8. The output approximates the full decode averaged in blocks; it is not bit-exact with
  libjpeg-turbo or Go, and `pyramid: false` (the default) is unchanged.
- It does not need a second level API: build levels 2 and up from the level 1 image by
  averaging 2x2 blocks (its cost is not measured here), or from
  the level 0 pixels if the caller decodes them anyway.
- No effect at scale 1. With a scale, `Error::Unsupported` for a `region`, for `Format::YCbCr`, for a
  non-empty `Recon`, and for a frame under `scale` samples in either direction.
- The blur runs only where the IDCT leaves a subsampled chroma axis at output size (4:2:0 at scale 2 and
  4); gray, 4:4:4, 4:2:2 and every frame at scale 8 equal the plain scaled decode cropped to the floor
  size. It is serial (a pass over the chroma planes, no allocation), and the output is the same for every
  `max_threads` and for a reused `Decoder`.
- `tests/pyramid.rs` and the `pyramid` fuzz target test it. `examples/ppm --scale N` writes its output
  as PPM for comparison with other tools.
- Caution: the mode averages before the colour conversion, so it holds only where the conversion does
  not clamp. A stream whose samples are RGB but carry no Adobe marker or R,G,B component ids (Aperio
  SVS tiles: the RGB is in the TIFF Photometric tag only) is converted as YCbCr by every decoder that
  sees only the stream, the conversion clamps on most pixels, and the mode differs from the reference
  by up to 100 levels there (on real Aperio slides about a quarter of the
  tiles fall under 40 dB at scale 2). Use the mode on such slides only with the colour model fixed first: `ColorModel::Rgb`
  (see Colour model), or an Adobe APP14 segment with transform 0 in the stream.
