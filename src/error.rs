use std::fmt;

/// One variant per Go error of jpeg-go. `Truncated` and `AfterScan` come with an image: see
/// [`Decoded`](crate::Decoded).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// `ErrUnsupported`: a stream or option out of scope (arithmetic, lossless, 12-bit, CMYK, ...).
    Unsupported,
    /// `ErrTruncated`: entropy data ran out, held a Huffman code no table matches, or restart
    /// markers were out of sequence (libjpeg-turbo's warnings). The image still comes back.
    Truncated,
    /// `ErrAfterScan`: a fatal marker after a single-scan frame's last row. The image still comes
    /// back. `cause` is the marker's own error, and `truncated` is set when the scan also lost data.
    AfterScan {
        cause: Box<Error>,
        truncated: bool,
    },
    /// `ErrLimit` (`MaxPixels`): the frame has more pixels than `Limits::max_pixels`.
    PixelLimit {
        got: u64,
        max: u64,
    },
    /// `ErrLimit` (`MaxBytes`): the call would hold more than `Limits::max_alloc` bytes. Go's
    /// 1 GiB cap on one decode is the default of `max_alloc`.
    AllocLimit {
        got: u64,
        max: u64,
    },
    /// The allocator refused a size that is within the limits.
    OutOfMemory,
    /// The cancel flag was set (Go's `ctx.Err()`).
    Cancelled,
    /// A call argument is outside what the function takes (a short buffer, a bad size). Go panics
    /// or errors with a message.
    InvalidArgument(&'static str),
    NoSoi,
    DupSoi,
    BadLength,
    UnknownMarker,
    NoSof,
    NoSos,
    SosBeforeSof,
    DupSof,
    EmptyImage,
    TooBig,
    ComponentId,
    DqtIndex,
    DhtIndex,
    HuffTable,
    TablesStream,
    DacIndex,
    ScanSelector,
    NoQuantTable,
    NoHuffTable,
    /// Another scan after one covering every component.
    EoiExpected,
    BadProgression,
    BadDctCoef,
    /// A `Region` outside +-2^29.
    Region,
    /// A `Region` that does not meet the image (Go's other region error).
    RegionEmpty,
    /// `RowReader` used after `close`.
    RowReaderClosed,
    /// The encoder was given an image it cannot take (zero size, a short plane).
    BadImage,
    /// An error from the caller's `Write`, as its kind.
    Io(std::io::ErrorKind),
    /// Go `image/jpeg`'s `FormatError` (with its message). No decoder here returns it; it is kept
    /// for dependents that report Go-style errors through `jpeg::Error`.
    Format(&'static str),
    /// Go `image/jpeg`'s `UnsupportedError` (with its message); kept as `Format` is.
    UnsupportedFeature(&'static str),
    /// `io.ErrUnexpectedEOF`: the stream ended inside a segment.
    UnexpectedEof,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Unsupported => "jpeg: unsupported stream or option",
            Self::Truncated => "jpeg: truncated entropy data",
            Self::AfterScan { cause, .. } => {
                return write!(f, "jpeg: fatal marker after the last scan: {cause}");
            }
            Self::PixelLimit { got, max } => {
                return write!(
                    f,
                    "jpeg: per-call limit exceeded: max_pixels {got} over {max}"
                );
            }
            Self::AllocLimit { got, max } => {
                return write!(
                    f,
                    "jpeg: per-call limit exceeded: max_alloc {got} over {max}"
                );
            }
            Self::OutOfMemory => "jpeg: out of memory",
            Self::Cancelled => "jpeg: cancelled",
            Self::InvalidArgument(s) => return write!(f, "jpeg: invalid argument: {s}"),
            Self::NoSoi => "jpeg: missing SOI",
            Self::DupSoi => "jpeg: two SOI markers",
            Self::BadLength => "jpeg: bad marker length",
            Self::UnknownMarker => "jpeg: unknown marker",
            Self::NoSof => "jpeg: no SOF marker",
            Self::NoSos => "jpeg: missing SOS marker",
            Self::SosBeforeSof => "jpeg: SOS before SOF",
            Self::DupSof => "jpeg: multiple SOF markers",
            Self::EmptyImage => "jpeg: empty image",
            Self::TooBig => "jpeg: image dimension over 65500",
            Self::ComponentId => "jpeg: bad component id in SOS",
            Self::DqtIndex => "jpeg: bad DQT index",
            Self::DhtIndex => "jpeg: bad DHT index",
            Self::HuffTable => "jpeg: bad DHT table",
            Self::TablesStream => "jpeg: tables stream has SOF or SOS",
            Self::DacIndex => "jpeg: bad DAC value",
            Self::ScanSelector => "jpeg: bad Huffman table selector in SOS",
            Self::NoQuantTable => "jpeg: quantization table was not defined",
            Self::NoHuffTable => "jpeg: Huffman table was not defined",
            Self::EoiExpected => "jpeg: another scan after one covering every component",
            Self::BadProgression => "jpeg: invalid progressive parameters",
            Self::BadDctCoef => "jpeg: DC coefficient out of range",
            Self::Region => "jpeg: region outside the allowed range",
            Self::RegionEmpty => "jpeg: region does not intersect the image",
            Self::RowReaderClosed => "jpeg: RowReader is closed",
            Self::BadImage => "jpeg: image the encoder cannot take",
            Self::Io(k) => return write!(f, "jpeg: write error: {k}"),
            Self::Format(s) => return write!(f, "invalid JPEG format: {s}"),
            Self::UnsupportedFeature(s) => return write!(f, "unsupported JPEG feature: {s}"),
            Self::UnexpectedEof => "unexpected EOF",
        };
        f.write_str(s)
    }
}

impl std::error::Error for Error {}
