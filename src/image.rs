//! Image types: Go's `image.Gray`, `image.RGBA` and `image.YCbCr`, owning their pixels.

/// A rectangle of pixels, `min` inclusive and `max` exclusive (Go's `image.Rectangle`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Rect {
    pub min_x: i32,
    pub min_y: i32,
    pub max_x: i32,
    pub max_y: i32,
}

impl Rect {
    pub fn new(min_x: i32, min_y: i32, max_x: i32, max_y: i32) -> Rect {
        Rect {
            min_x,
            min_y,
            max_x,
            max_y,
        }
    }

    pub fn width(&self) -> i32 {
        self.max_x.saturating_sub(self.min_x)
    }

    pub fn height(&self) -> i32 {
        self.max_y.saturating_sub(self.min_y)
    }

    pub fn is_empty(&self) -> bool {
        self.min_x >= self.max_x || self.min_y >= self.max_y
    }

    /// The largest rectangle contained in both; empty (the zero rectangle) when they do not meet.
    pub fn intersect(&self, o: Rect) -> Rect {
        let r = Rect {
            min_x: self.min_x.max(o.min_x),
            min_y: self.min_y.max(o.min_y),
            max_x: self.max_x.min(o.max_x),
            max_y: self.max_y.min(o.max_y),
        };
        if r.is_empty() { Rect::default() } else { r }
    }
}

/// Chroma subsampling of a 3-component stream (Go's `image.YCbCrSubsampleRatio`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Subsample {
    S444,
    S422,
    S420,
    S440,
    S411,
    S410,
}

impl Subsample {
    /// The ratio for Y's sampling factors (Cb and Cr are 1x1).
    pub(crate) fn from_sampling(h: u8, v: u8) -> Subsample {
        match (h, v) {
            (1, 1) => Subsample::S444,
            (2, 1) => Subsample::S422,
            (1, 2) => Subsample::S440,
            (4, 1) => Subsample::S411,
            (4, 2) => Subsample::S410,
            _ => Subsample::S420, // (2, 2), the only other case the parser allows
        }
    }
}

/// An 8-bit grayscale image. Pixel (x, y) is `pix[(y - rect.min_y) * stride + (x - rect.min_x)]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gray {
    pub pix: Vec<u8>,
    pub stride: usize,
    pub rect: Rect,
}

/// An 8-bit CMYK image (Go's `image.CMYK`), 4 bytes per pixel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cmyk {
    pub pix: Vec<u8>,
    pub stride: usize,
    pub rect: Rect,
}

/// An 8-bit RGBA image, alpha 255 from the decoder. Pixel (x, y) starts at
/// `pix[(y - rect.min_y) * stride + (x - rect.min_x) * 4]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rgba {
    pub pix: Vec<u8>,
    pub stride: usize,
    pub rect: Rect,
}

/// Planar YCbCr at the stream's sampling. The planes keep the decoder's block padding: strides
/// are padded to whole blocks and the samples past `rect` are real IDCT output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct YCbCr {
    pub y: Vec<u8>,
    pub cb: Vec<u8>,
    pub cr: Vec<u8>,
    pub y_stride: usize,
    pub c_stride: usize,
    pub ratio: Subsample,
    pub rect: Rect,
}

/// `row * stride + col * bpp`, panicking (in every build profile) if the coordinates sit left of
/// or above `min` or the sum overflows.
#[allow(clippy::expect_used)] // the documented panic of the offset helpers
fn offset(min: (i32, i32), (x, y): (i32, i32), stride: usize, bpp: usize) -> usize {
    let rel = |v: i32, m: i32| usize::try_from(i64::from(v) - i64::from(m)).expect("outside rect");
    rel(y, min.1)
        .checked_mul(stride)
        .and_then(|r| r.checked_add(rel(x, min.0).checked_mul(bpp)?))
        .expect("pixel offset overflows")
}

impl Gray {
    /// The index of pixel (x, y) in `pix`.
    ///
    /// # Panics
    /// If (x, y) is above or left of `rect`, or the offset overflows `usize`.
    pub fn pix_offset(&self, x: i32, y: i32) -> usize {
        offset((self.rect.min_x, self.rect.min_y), (x, y), self.stride, 1)
    }
}

impl Rgba {
    /// The index of pixel (x, y) in `pix`.
    ///
    /// # Panics
    /// If (x, y) is above or left of `rect`, or the offset overflows `usize`.
    pub fn pix_offset(&self, x: i32, y: i32) -> usize {
        offset((self.rect.min_x, self.rect.min_y), (x, y), self.stride, 4)
    }
}

impl YCbCr {
    /// The index of pixel (x, y) in `y`.
    ///
    /// # Panics
    /// If (x, y) is above or left of `rect`, or the offset overflows `usize`.
    pub fn y_offset(&self, x: i32, y: i32) -> usize {
        offset((self.rect.min_x, self.rect.min_y), (x, y), self.y_stride, 1)
    }

    /// The chroma sample's column and row for pixel (x, y).
    pub(crate) fn c_xy(&self, x: i32, y: i32) -> (i64, i64) {
        let (x, y) = (x as i64, y as i64);
        let (min_x, min_y) = (self.rect.min_x as i64, self.rect.min_y as i64);
        match self.ratio {
            Subsample::S422 => (x / 2 - min_x / 2, y - min_y),
            Subsample::S420 => (x / 2 - min_x / 2, y / 2 - min_y / 2),
            Subsample::S440 => (x - min_x, y / 2 - min_y / 2),
            Subsample::S411 => (x / 4 - min_x / 4, y - min_y),
            Subsample::S410 => (x / 4 - min_x / 4, y / 2 - min_y / 2),
            Subsample::S444 => (x - min_x, y - min_y),
        }
    }

    /// The index of the chroma sample of pixel (x, y) in `cb` and `cr`.
    ///
    /// # Panics
    /// If the chroma sample is above or left of the chroma origin, or the offset overflows `usize`.
    #[allow(clippy::expect_used)] // the documented panic
    pub fn c_offset(&self, x: i32, y: i32) -> usize {
        let (cx, cy) = self.c_xy(x, y);
        usize::try_from(cy)
            .ok()
            .and_then(|r| r.checked_mul(self.c_stride))
            .and_then(|r| r.checked_add(usize::try_from(cx).ok()?))
            .expect("chroma offset outside rect")
    }
}

/// A decoded image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Image {
    Gray(Gray),
    Rgba(Rgba),
    YCbCr(YCbCr),
}

impl Image {
    pub fn bounds(&self) -> Rect {
        match self {
            Image::Gray(g) => g.rect,
            Image::Rgba(r) => r.rect,
            Image::YCbCr(y) => y.rect,
        }
    }
}
