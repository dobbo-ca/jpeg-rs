//! The Rust side of the parity check: `oracle <inputs> <outputs>` runs every operation of
//! `tests/parity/go/oracle` on every input and writes `<name>.<op>.out` (the output), or
//! `<name>.<op>.err` (the error class) and `<name>.<op>.partial.out` (the image Go returns beside
//! an error). The wire formats and the class table are in `docs/parity.md`.

use jpeg::{
    DecodeOptions, Decoded, Decoder, EncodeImage, EncodeOptions, Encoder, Error, Format, Gray,
    Image, Limits, Rect, Rgba, RgbaWindow, RowReader, Subsample, YCbCr,
};
use std::cell::RefCell;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::{env, fs};

const MAX_PIXELS: u64 = 1 << 22;
const MAX_BYTES: u64 = 128 << 20;

/// `--threads N`: the `max_threads` of every call; 0 (the default) is `Limits::default()`'s.
static THREADS: AtomicUsize = AtomicUsize::new(0);

fn threads() -> usize {
    match THREADS.load(Ordering::Relaxed) {
        0 => Limits::default().max_threads,
        n => n,
    }
}

fn lim() -> Limits {
    limits(MAX_BYTES)
}

fn limits(max_alloc: u64) -> Limits {
    Limits {
        max_pixels: MAX_PIXELS,
        max_alloc,
        max_threads: threads(),
    }
}

/// The reused `Decoder` and `Encoder` of the ops `sd1 sy1 sr` and `se*`: one pair for the whole run,
/// so each input sees the state the earlier ones left, as Go's `sharedDec` and `sharedEnc`. `dirty`
/// is the fixed frame every reused decode follows (`dirtyFrame` of tests/parity/go/oracle/main.go).
#[derive(Default)]
struct Shared {
    dec: Decoder,
    enc: Encoder,
    dirty: Option<Vec<u8>>,
}

thread_local! {
    static SHARED: RefCell<Shared> = RefCell::new(Shared::default());
}

static NO: AtomicBool = AtomicBool::new(false);
/// The cancel flag set before the call: Go's already-cancelled context.
static YES: AtomicBool = AtomicBool::new(true);

// ---- error classes ----

fn class(e: &Error) -> String {
    match e {
        Error::PixelLimit { .. } => "limit-pixels".into(),
        Error::AllocLimit { .. } => "limit-bytes".into(),
        Error::AfterScan { cause, truncated } => format!(
            "afterscan:{}{}",
            class(cause),
            if *truncated { "+truncated" } else { "" }
        ),
        Error::Truncated => "truncated".into(),
        Error::Unsupported => "unsupported".into(),
        Error::Cancelled => "cancelled".into(),
        Error::UnexpectedEof => "unexpected-eof".into(),
        Error::Region => "region".into(),
        Error::RegionEmpty => "region-empty".into(),
        Error::RowReaderClosed => "closed".into(),
        Error::InvalidArgument(_) => "invalid-argument".into(),
        Error::OutOfMemory => "oom".into(),
        Error::BadImage => "bad-image".into(),
        Error::Io(_) => "io".into(),
        e => e.to_string(),
    }
}

// ---- output ----

const RAW_MAX: usize = 8 << 10;

fn fnv64(mut h: u64, b: &[u8]) -> u64 {
    for &c in b {
        h = (h ^ u64::from(c)).wrapping_mul(0x0100_0000_01b3);
    }
    h
}

const FNV_BASIS: u64 = 0xcbf2_9ce4_8422_2325;

/// `hdr`, then the rows, or over `RAW_MAX` bytes `hdr`, "fnv <hash> <len>" and "r <hash>" per row.
fn encode_rows(hdr: &str, rows: &[&[u8]]) -> Vec<u8> {
    let n: usize = rows.iter().map(|r| r.len()).sum();
    let mut out = hdr.as_bytes().to_vec();
    if n <= RAW_MAX {
        for r in rows {
            out.extend_from_slice(r);
        }
        return out;
    }
    let h = rows.iter().fold(FNV_BASIS, |h, r| fnv64(h, r));
    out.extend_from_slice(format!("fnv {h:016x} {n}\n").as_bytes());
    for r in rows {
        out.extend_from_slice(format!("r {:016x}\n", fnv64(FNV_BASIS, r)).as_bytes());
    }
    out
}

fn owned_rows(rows: &[Vec<u8>]) -> Vec<&[u8]> {
    rows.iter().map(|r| r.as_slice()).collect()
}

/// The bytes of a stream as rows of 1 KiB.
fn chunk_rows(b: &[u8]) -> Vec<&[u8]> {
    b.chunks(1024).collect()
}

fn ratio_name(r: Subsample) -> &'static str {
    match r {
        Subsample::S444 => "444",
        Subsample::S422 => "422",
        Subsample::S420 => "420",
        Subsample::S440 => "440",
        Subsample::S411 => "411",
        Subsample::S410 => "410",
    }
}

fn pix_rows(pix: &[u8], stride: usize, r: Rect, bpp: usize) -> Vec<&[u8]> {
    if r.is_empty() {
        return Vec::new();
    }
    let w = r.width() as usize * bpp;
    (0..r.height() as usize)
        .map(|y| &pix[y * stride..y * stride + w])
        .collect()
}

fn rect_hdr(name: &str, r: Rect) -> String {
    format!("{name} {} {} {} {}\n", r.min_x, r.min_y, r.max_x, r.max_y)
}

fn ser_rgba(m: &Rgba) -> Vec<u8> {
    encode_rows(
        &rect_hdr("rgba", m.rect),
        &pix_rows(&m.pix, m.stride, m.rect, 4),
    )
}

fn ser_gray(m: &Gray) -> Vec<u8> {
    encode_rows(
        &rect_hdr("gray", m.rect),
        &pix_rows(&m.pix, m.stride, m.rect, 1),
    )
}

fn ser_ycbcr(m: &YCbCr) -> Vec<u8> {
    let r = m.rect;
    let hdr = format!(
        "ycbcr {} {} {} {} {}\n",
        ratio_name(m.ratio),
        r.min_x,
        r.min_y,
        r.max_x,
        r.max_y
    );
    let mut rows: Vec<&[u8]> = Vec::new();
    if !r.is_empty() {
        let w = r.width() as usize;
        for y in r.min_y..r.max_y {
            let o = m.y_offset(r.min_x, y);
            rows.push(&m.y[o..o + w]);
        }
        let cw = m.c_offset(r.max_x - 1, r.min_y) - m.c_offset(r.min_x, r.min_y) + 1;
        let mut offs = Vec::new();
        let mut last = usize::MAX;
        for y in r.min_y..r.max_y {
            let o = m.c_offset(r.min_x, y);
            if o != last {
                offs.push(o);
                last = o;
            }
        }
        for p in [&m.cb, &m.cr] {
            for &o in &offs {
                rows.push(&p[o..o + cw]);
            }
        }
    }
    encode_rows(&hdr, &rows)
}

fn serialize(img: &Image) -> Vec<u8> {
    match img {
        Image::Rgba(m) => ser_rgba(m),
        Image::Gray(m) => ser_gray(m),
        Image::YCbCr(m) => ser_ycbcr(m),
    }
}

struct Out<'a> {
    dir: &'a Path,
    name: &'a str,
}

impl Out<'_> {
    fn file(&self, op: &str, ext: &str, b: &[u8]) {
        let p = self.dir.join(format!("{}.{op}{ext}", self.name));
        fs::write(&p, b).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    }

    fn ok(&self, op: &str, b: &[u8]) {
        self.file(op, ".out", b);
    }

    fn fail(&self, op: &str, e: &Error, partial: Option<Vec<u8>>) {
        self.file(op, ".err", format!("{}\n", class(e)).as_bytes());
        if let Some(p) = partial {
            self.file(op, ".partial.out", &p);
        }
    }

    /// A decode's result: the image, or the error with the image returned beside it.
    fn decoded(&self, op: &str, r: Result<Decoded, Error>) {
        self.decoded_ref(op, r.as_ref());
    }

    fn decoded_ref(&self, op: &str, r: Result<&Decoded, &Error>) {
        match r {
            Ok(Decoded { image, error: None }) => self.ok(op, &serialize(image)),
            Ok(Decoded {
                image,
                error: Some(e),
            }) => self.fail(op, e, Some(serialize(image))),
            Err(e) => self.fail(op, e, None),
        }
    }

    /// Runs `f`; a panic is the result "panic", which Go never writes for the same input.
    fn safe(&self, op: &str, f: impl FnOnce()) {
        if let Err(p) = catch_unwind(AssertUnwindSafe(f)) {
            let msg = p
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default();
            self.file(op, ".err", format!("panic:{msg}\n").as_bytes());
        }
    }
}

// ---- operations ----

struct Dop {
    name: String,
    f: Format,
    scale: u8,
}

fn dops(light: bool) -> Vec<Dop> {
    let mut l = Vec::new();
    for s in [1u8, 2, 4, 8] {
        for (n, f) in [
            ("yc", Format::YCbCr),
            ("rg", Format::Rgba),
            ("gr", Format::Gray),
        ] {
            if light && !(s == 1 || (s == 4 && n == "rg")) {
                continue;
            }
            l.push(Dop {
                name: format!("{n}{s}"),
                f,
                scale: s,
            });
        }
    }
    l
}

fn dec(src: &[u8], o: &DecodeOptions, l: &Limits, cancel: &AtomicBool) -> Result<Decoded, Error> {
    if l.max_threads > 1 {
        // The size threshold off, so the small parity inputs run on bands and stripes too.
        jpeg::decode_probe(src, o, l, cancel, true).0
    } else {
        jpeg::decode(src, o, l, cancel)
    }
}

fn decode_opts(src: &[u8], o: DecodeOptions) -> Result<Decoded, Error> {
    dec(src, &o, &lim(), &NO)
}

fn cfg_line(c: &jpeg::Config) -> String {
    format!(
        "{} {} {} {} {} {}\n",
        c.width,
        c.height,
        c.components,
        ratio_name(c.ratio),
        c.rgb,
        c.restart_interval
    )
}

fn config_ops(o: &Out, src: &[u8], tables: Option<&[u8]>, pfx: &str) {
    let op = format!("{pfx}cfg");
    o.safe(&op, || match jpeg::decode_config(src, tables, &lim()) {
        Ok(c) => o.ok(&op, cfg_line(&c).as_bytes()),
        Err(e) => o.fail(&op, &e, None),
    });
}

/// The wire form of coefficients: `hdr_tag`, the frame's config line, a line per component, then
/// per component its quantization table and a row of blocks per block row (little endian i16).
fn coef_bytes(hdr_tag: &str, c: &jpeg::Coefficients) -> Vec<u8> {
    let mut hdr = format!("{hdr_tag} {}", cfg_line(&c.config));
    let mut rows: Vec<Vec<u8>> = Vec::new();
    for cc in &c.comp {
        hdr += &format!(
            "c {} {} {} {}\n",
            cc.h, cc.v, cc.blocks_wide, cc.blocks_high
        );
        rows.push(cc.quant.iter().flat_map(|v| v.to_le_bytes()).collect());
        for by in 0..cc.blocks_high {
            let row = &cc.blocks[by * cc.blocks_wide..(by + 1) * cc.blocks_wide];
            rows.push(
                row.iter()
                    .flat_map(|b| b.iter().flat_map(|v| v.to_le_bytes()))
                    .collect(),
            );
        }
    }
    encode_rows(&hdr, &owned_rows(&rows))
}

fn coef_op(o: &Out, cancel: &AtomicBool, op: &str, src: &[u8], tables: Option<&[u8]>, l: &Limits) {
    o.safe(op, || {
        match jpeg::decode_coefficients(src, tables, l, cancel) {
            Err(e) => o.fail(op, &e, None),
            Ok(d) => {
                let b = coef_bytes("coef", &d.coefficients);
                match d.error {
                    Some(e) => o.fail(op, &e, Some(b)),
                    None => o.ok(op, &b),
                }
            }
        }
    });
}

fn rr_op(o: &Out, cancel: &AtomicBool, op: &str, src: &[u8], tables: Option<&[u8]>, l: &Limits) {
    o.safe(op, || {
        let mut r = match RowReader::new(src, tables, l, cancel) {
            Ok(r) => r,
            Err(e) => return o.fail(op, &e, None),
        };
        let c = r.config();
        let hdr = format!("rr {} {} {}\n", c.width, c.height, r.truncated());
        let mut rows: Vec<Vec<u8>> = Vec::new();
        loop {
            match r.next(cancel) {
                Ok(None) => break,
                Ok(Some(img)) => {
                    for row in pix_rows(&img.pix, img.stride, img.rect, 4) {
                        rows.push(row.to_vec());
                    }
                }
                Err(e) => return o.fail(op, &e, Some(encode_rows(&hdr, &owned_rows(&rows)))),
            }
        }
        o.ok(op, &encode_rows(&hdr, &owned_rows(&rows)));
    });
}

fn fmt_of(f: i32) -> Format {
    match f {
        1 => Format::Rgba,
        2 => Format::Gray,
        _ => Format::YCbCr,
    }
}

/// `dstOp` of tests/parity/go/oracle/main.go: `decode_into_rgba` into a window of a buffer of 0xAA.
fn dst_op(o: &Out, op: &str, src: &[u8], tables: Option<&[u8]>, c: &jpeg::Config, pad: usize) {
    o.safe(op, || {
        let (w, h) = (c.width, c.height);
        let stride = 4 * w + pad;
        let mut buf = vec![0xaau8; (h + 2) * stride];
        let off = if pad == 0 { 0 } else { stride + 8 };
        if pad == 0 {
            buf.truncate(h * stride);
        }
        let opts = DecodeOptions {
            format: Format::Rgba,
            scale: 1,
            tables,
            ..Default::default()
        };
        let r = jpeg::decode_into_rgba(
            src,
            &opts,
            &lim(),
            &NO,
            RgbaWindow {
                pix: &mut buf[off..],
                stride,
            },
        );
        let b = encode_rows(&format!("dst {w} {h} {pad}\n"), &chunk_rows(&buf));
        match r {
            Ok(None) => o.ok(op, &b),
            Ok(Some(e)) | Err(e) => o.fail(op, &e, Some(b)),
        }
    });
}

/// `rrxOp` of tests/parity/go/oracle/main.go: what `next` returns after the last row and after `close`.
fn rrx_op(o: &Out, src: &[u8], tables: Option<&[u8]>) {
    o.safe("rrx", || {
        let next = |r: &mut RowReader| match r.next(&NO) {
            Ok(None) => "eof".to_string(),
            Ok(Some(_)) => "rows".to_string(),
            Err(e) => class(&e),
        };
        let mut r = match RowReader::new(src, tables, &lim(), &NO) {
            Ok(r) => r,
            Err(e) => return o.fail("rrx", &e, None),
        };
        let mut sb = String::new();
        let mut n = 0;
        loop {
            let s = next(&mut r);
            if s != "rows" {
                sb += &format!("end {n} {s}\n");
                break;
            }
            n += 1;
        }
        sb += &format!("after end {}\n", next(&mut r));
        r.close();
        r.close();
        sb += &format!("after close {}\n", next(&mut r));
        match RowReader::new(src, tables, &lim(), &NO) {
            Err(e) => sb += &format!("second new {}\n", class(&e)),
            Ok(mut r2) => {
                sb += &format!("first {}\n", next(&mut r2));
                r2.close();
                sb += &format!("after early close {}\n", next(&mut r2));
            }
        }
        o.ok("rrx", sb.as_bytes());
    });
}

/// `sharedDec.Decode` of Go: the shared `Decoder` decodes the dirty frame, then `src`.
fn reused_decode(o: &Out, op: &str, src: &[u8], opts: DecodeOptions) {
    o.safe(op, || {
        SHARED.with_borrow_mut(|s| {
            if let Some(d) = &s.dirty {
                let dirty = DecodeOptions {
                    format: Format::Rgba,
                    scale: 1,
                    ..Default::default()
                };
                let _ = s.dec.decode(d, &dirty, &lim(), &NO);
            }
            let r = s.dec.decode(src, &opts, &lim(), &NO);
            o.decoded_ref(op, r.as_ref().map(|d| *d));
        });
    });
}

fn run_jpeg(dir: &Path, o: &Out, name: &str, src: &[u8]) {
    let light = name.contains('~');
    let base = name.strip_suffix(".jpg").unwrap_or(name);
    let tables: Option<Vec<u8>> = if name.starts_with("ab-") {
        fs::read(dir.join("tables.bin")).ok()
    } else {
        fs::read(dir.join(format!("{base}.tables"))).ok() // beside the stream (TestTablesStateReset)
    };
    let tables = tables.as_deref();
    let cfg = jpeg::decode_config(src, tables, &lim());

    config_ops(o, src, tables, "");
    for d in dops(light) {
        o.safe(&d.name, || {
            let r = decode_opts(
                src,
                DecodeOptions {
                    format: d.f,
                    scale: d.scale,
                    tables,
                    ..Default::default()
                },
            );
            o.decoded(&d.name, r);
        });
    }
    coef_op(o, &NO, "coef", src, tables, &lim());
    rr_op(o, &NO, "rr", src, tables, &lim());
    rrx_op(o, src, tables);
    // One reused Decoder, each decode after the fixed dirty one (Go's sharedDec).
    for (n, f) in [("sd1", Format::Rgba), ("sy1", Format::YCbCr)] {
        reused_decode(
            o,
            n,
            src,
            DecodeOptions {
                format: f,
                scale: 1,
                tables,
                ..Default::default()
            },
        );
    }
    if let Ok(c) = &cfg {
        reused_decode(
            o,
            "sr",
            src,
            DecodeOptions {
                format: Format::Rgba,
                scale: 1,
                tables,
                region: Some(region(
                    -5,
                    -3,
                    c.width as i32 / 2 + 1,
                    c.height as i32 / 2 + 1,
                )),
                ..Default::default()
            },
        );
        dst_op(o, "dst", src, tables, c, 16);
        dst_op(o, "dst0", src, tables, c, 0);
    }
    o.safe("rrn", || {
        match jpeg::row_reader_need(
            src,
            tables,
            &Limits {
                max_pixels: u64::MAX,
                max_alloc: u64::MAX,
                max_threads: threads(),
            },
        ) {
            Ok(n) => o.ok("rrn", format!("{n}\n").as_bytes()),
            Err(e) => o.fail("rrn", &e, None),
        }
    });

    if let Ok(c) = cfg {
        let (w, h) = (c.width as u64, c.height as u64);
        let rgba = |l: &Limits| {
            decode_opts_l(
                src,
                DecodeOptions {
                    format: Format::Rgba,
                    tables,
                    ..Default::default()
                },
                l,
            )
        };
        o.safe("lb2", || o.decoded("lb2", rgba(&limits(w * h * 3))));
        if !light {
            if w * h > 1 {
                o.safe("lpx", || {
                    let l = Limits {
                        max_pixels: w * h - 1,
                        max_alloc: MAX_BYTES,
                        max_threads: threads(),
                    };
                    o.decoded("lpx", rgba(&l));
                });
            }
            for (i, mb) in [1u64 << 14, 1 << 17, w * h * 6 + (1 << 16)]
                .into_iter()
                .enumerate()
            {
                let op = format!("lb{i}");
                o.safe(&op, || o.decoded(&op, rgba(&limits(mb))));
            }
            rr_op(o, &NO, "rrl", src, tables, &limits(w * h * 3));
            coef_op(o, &NO, "coefl", src, tables, &limits(w * h));
            o.safe("cx", || {
                let cancel = AtomicBool::new(true);
                let r = dec(
                    src,
                    &DecodeOptions {
                        format: Format::Rgba,
                        tables,
                        ..Default::default()
                    },
                    &lim(),
                    &cancel,
                );
                o.decoded("cx", r);
            });
            coef_op(o, &YES, "coefx", src, tables, &lim());
            rr_op(o, &YES, "rrcx", src, tables, &lim());
        }
    }
    if tables.is_some() {
        for (n, f) in [("tb-yc", Format::YCbCr), ("tb-rg", Format::Rgba)] {
            o.safe(n, || {
                let r = decode_opts(
                    src,
                    DecodeOptions {
                        format: f,
                        scale: 1,
                        tables,
                        ..Default::default()
                    },
                );
                o.decoded(n, r);
            });
        }
    }

    // A sidecar <base>.rect (damaged streams have one only for strip_rst and the dri- streams): one region decode per line "scale x0 y0 x1 y1 format check".
    if let Ok(b) = fs::read_to_string(dir.join(format!("{base}.rect"))) {
        for (i, line) in b.trim().split('\n').enumerate() {
            let v: Vec<i32> = line
                .split_whitespace()
                .filter_map(|x| x.parse().ok())
                .collect();
            let [s, x0, y0, x1, y1, f, c] = v[..] else {
                continue;
            };
            let op = format!("r{i}");
            o.safe(&op, || {
                let r = decode_opts(
                    src,
                    DecodeOptions {
                        format: fmt_of(f),
                        scale: s as u8,
                        tables,
                        region: Some(region(x0, y0, x1, y1)),
                        check_frame: c != 0,
                        ..Default::default()
                    },
                );
                o.decoded(&op, r);
            });
        }
    }
    // A sidecar <base>.starts: restart offsets (NDPI tag 65426), one per line.
    if let Ok(b) = fs::read_to_string(dir.join(format!("{base}.starts"))) {
        let starts: Vec<usize> = b
            .split_whitespace()
            .filter_map(|x| x.parse().ok())
            .collect();
        for (n, f, s) in [
            ("ndpi-rg", Format::Rgba, 1u8),
            ("ndpi-gr2", Format::Gray, 2),
        ] {
            o.safe(n, || {
                let r = decode_opts(
                    src,
                    DecodeOptions {
                        format: f,
                        scale: s,
                        restart: Some(&starts),
                        ..Default::default()
                    },
                );
                o.decoded(n, r);
            });
        }
        if let Ok(c) = jpeg::decode_config(src, tables, &lim()) {
            o.safe("ndpi-reg", || {
                let (w, h) = (c.width as i32, c.height as i32);
                let r = decode_opts(
                    src,
                    DecodeOptions {
                        format: Format::Rgba,
                        scale: 1,
                        restart: Some(&starts),
                        region: Some(region(w / 3, h / 3, w * 2 / 3 + 1, h * 2 / 3 + 1)),
                        check_frame: true,
                        ..Default::default()
                    },
                );
                o.decoded("ndpi-reg", r);
            });
            // The same offsets with check_frame off: the serial offset-landing jump and the region bands.
            for (n, f) in [("ndpi-reg2", Format::Rgba), ("ndpi-reg2g", Format::Gray)] {
                o.safe(n, || {
                    let (w, h) = (c.width as i32, c.height as i32);
                    let r = decode_opts(
                        src,
                        DecodeOptions {
                            format: f,
                            scale: 1,
                            restart: Some(&starts),
                            region: Some(region(w / 4, h / 4, w * 3 / 4 + 1, h * 3 / 4 + 1)),
                            ..Default::default()
                        },
                    );
                    o.decoded(n, r);
                });
            }
        }
    }
}

/// Go's `image.Rect`: the corners in order.
fn region(x0: i32, y0: i32, x1: i32, y1: i32) -> Rect {
    Rect::new(x0.min(x1), y0.min(y1), x0.max(x1), y0.max(y1))
}

fn decode_opts_l(src: &[u8], o: DecodeOptions, l: &Limits) -> Result<Decoded, Error> {
    dec(src, &o, l, &NO)
}

/// Decodes the fixed abbreviated tile ab-image.jpg against a tables stream.
fn run_tables(dir: &Path, o: &Out, tables: &[u8]) {
    let Ok(img) = fs::read(dir.join("ab-image.jpg")) else {
        return;
    };
    config_ops(o, &img, Some(tables), "tb-");
    for (n, f) in [("tb-yc", Format::YCbCr), ("tb-rg", Format::Rgba)] {
        o.safe(n, || {
            let r = decode_opts(
                &img,
                DecodeOptions {
                    format: f,
                    scale: 1,
                    tables: Some(tables),
                    ..Default::default()
                },
            );
            o.decoded(n, r);
        });
    }
    coef_op(o, &NO, "tb-coef", &img, Some(tables), &lim());
}

// ---- encoder inputs (.img): see run_img in tests/parity/go/oracle/ops.go ----

const RATIO_DIV: [(usize, usize); 6] = [(1, 1), (2, 1), (2, 2), (1, 2), (4, 1), (4, 2)];
const RATIOS: [Subsample; 6] = [
    Subsample::S444,
    Subsample::S422,
    Subsample::S420,
    Subsample::S440,
    Subsample::S411,
    Subsample::S410,
];

/// `n` bytes from the front of `p`, zero-filled if it is short.
fn take(p: &mut &[u8], n: usize) -> Vec<u8> {
    let k = n.min(p.len());
    let mut out = p[..k].to_vec();
    out.resize(n, 0);
    *p = &p[k..];
    out
}

enum Input {
    Rgba(Rgba),
    Gray(Gray),
    YCbCr(YCbCr),
}

impl Input {
    fn image(&self) -> EncodeImage<'_> {
        match self {
            Input::Rgba(m) => EncodeImage::Rgba(m),
            Input::Gray(m) => EncodeImage::Gray(m),
            Input::YCbCr(m) => EncodeImage::YCbCr(m),
        }
    }
}

/// The encoder's input: the sub-image (ox, oy, ox+w, oy+h) of the parent in the file.
fn parse_img(b: &[u8]) -> Option<(u8, Input)> {
    if b.len() < 8 || b[0] > 8 {
        return None;
    }
    let kind = b[0];
    let (w, h) = (
        usize::from(u16::from_le_bytes([b[1], b[2]])),
        usize::from(u16::from_le_bytes([b[3], b[4]])),
    );
    let (ox, oy, pad) = (usize::from(b[5]), usize::from(b[6]), usize::from(b[7]));
    if w == 0 || h == 0 {
        return None;
    }
    let mut p = &b[8..];
    let (pw, ph) = (w + ox, h + oy);
    let rect = Rect::new(ox as i32, oy as i32, (ox + w) as i32, (oy + h) as i32);
    Some((
        kind,
        match kind {
            0 | 8 => {
                let stride = 4 * pw + pad;
                let pix = take(&mut p, stride * ph);
                Input::Rgba(Rgba {
                    pix: pix[oy * stride + 4 * ox..].to_vec(),
                    stride,
                    rect,
                })
            }
            1 => {
                let stride = pw + pad;
                let pix = take(&mut p, stride * ph);
                Input::Gray(Gray {
                    pix: pix[oy * stride + ox..].to_vec(),
                    stride,
                    rect,
                })
            }
            _ => {
                let ratio = (kind - 2) as usize;
                let (dx, dy) = RATIO_DIV[ratio];
                let (cw, ch) = (pw.div_ceil(dx), ph.div_ceil(dy));
                let (ys, cs) = (pw + pad, cw + pad);
                let y = take(&mut p, ys * ph);
                let cb = take(&mut p, cs * ch);
                let cr = take(&mut p, cs * ch);
                let (yi, ci) = (oy * ys + ox, (oy / dy) * cs + ox / dx);
                Input::YCbCr(YCbCr {
                    y: y[yi..].to_vec(),
                    cb: cb[ci..].to_vec(),
                    cr: cr[ci..].to_vec(),
                    y_stride: ys,
                    c_stride: cs,
                    ratio: RATIOS[ratio],
                    rect,
                })
            }
        },
    ))
}

const ENC_QUALITIES: [i32; 8] = [0, 1, 25, 50, 75, 90, 100, 101];

fn run_img(o: &Out, b: &[u8]) {
    let Some((kind, input)) = parse_img(b) else {
        return;
    };
    let both = matches!(kind, 0 | 8 | 3 | 5 | 6 | 7);
    for q in ENC_QUALITIES {
        for c in [false, true] {
            if c && !both {
                continue;
            }
            let op = format!("e{q}{}", if c { "c" } else { "" });
            o.safe(&op, || {
                let opts = EncodeOptions {
                    quality: q,
                    chroma444: c,
                };
                match jpeg::encode(input.image(), &opts, &lim(), &NO) {
                    Ok(j) => o.ok(&op, &encode_rows("jpeg\n", &chunk_rows(&j))),
                    Err(e) => o.fail(&op, &e, None),
                }
            });
        }
    }
    o.safe("ecx", || {
        let opts = EncodeOptions {
            quality: 75,
            chroma444: false,
        };
        match jpeg::encode(input.image(), &opts, &lim(), &YES) {
            Ok(j) => o.ok("ecx", &encode_rows("jpeg\n", &chunk_rows(&j))),
            Err(e) => o.fail("ecx", &e, None),
        }
    });
    // One Encoder shared by every input (Go's sharedEnc).
    for q in [0, 75] {
        for c in [false, true] {
            if c && !both {
                continue;
            }
            let op = format!("se{q}{}", if c { "c" } else { "" });
            o.safe(&op, || {
                let opts = EncodeOptions {
                    quality: q,
                    chroma444: c,
                };
                match SHARED.with_borrow_mut(|s| s.enc.encode(input.image(), &opts, &lim(), &NO)) {
                    Ok(j) => o.ok(&op, &encode_rows("jpeg\n", &chunk_rows(&j))),
                    Err(e) => o.fail(&op, &e, None),
                }
            });
        }
    }
}

// ---- kernels ----

fn run_fancy(o: &Out, mut b: &[u8]) {
    let mut rows: Vec<Vec<u8>> = Vec::new();
    while b.len() >= 7 {
        let op = b[0];
        let (n, no, dl) = (
            usize::from(u16::from_le_bytes([b[1], b[2]])),
            usize::from(u16::from_le_bytes([b[3], b[4]])),
            usize::from(u16::from_le_bytes([b[5], b[6]])),
        );
        b = &b[7..];
        let row = take(&mut b, n);
        let other = if op == 2 {
            take(&mut b, no)
        } else {
            Vec::new()
        };
        let mut dst = vec![0xaau8; dl];
        let r = if op == 1 {
            jpeg::h2v1_fancy_row(&mut dst, &row, &lim())
        } else {
            jpeg::h2v2_fancy_row(&mut dst, &row, &other, &lim())
        };
        match r {
            Ok(()) => {
                rows.push(format!("K {}\n", dst.len()).into_bytes());
                rows.push(dst);
            }
            Err(_) => rows.push(b"E invalid-argument\n".to_vec()),
        }
    }
    o.ok("fancy", &encode_rows("fancy\n", &owned_rows(&rows)));
}

fn main() -> std::io::Result<()> {
    let mut args: Vec<String> = env::args().collect();
    if let Some(i) = args.iter().position(|a| a == "--threads") {
        let n = args.get(i + 1).and_then(|n| n.parse::<usize>().ok());
        let Some(n) = n else {
            eprintln!("usage: oracle [--threads N] <inputs> <outputs>");
            std::process::exit(2);
        };
        THREADS.store(n, Ordering::Relaxed);
        args.drain(i..i + 2);
    }
    let [_, input, output] = args.as_slice() else {
        eprintln!("usage: oracle [--threads N] <inputs> <outputs>");
        std::process::exit(2);
    };
    let (input, output) = (Path::new(input), Path::new(output));
    fs::create_dir_all(output)?;
    let mut names: Vec<String> = Vec::new();
    for entry in fs::read_dir(input)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            names.push(entry.file_name().to_string_lossy().into_owned());
        }
    }
    names.sort();
    let dirty = fs::read(input.join("synth_444_227x149.jpg")).ok();
    SHARED.with_borrow_mut(|s| s.dirty = dirty);
    for name in &names {
        let data = fs::read(input.join(name))?;
        let o = Out { dir: output, name };
        match Path::new(name).extension().and_then(|e| e.to_str()) {
            Some("jpg") => run_jpeg(input, &o, name, &data),
            Some("tbl") => run_tables(input, &o, &data),
            Some("img") => run_img(&o, &data),
            Some("fancy") => run_fancy(&o, &data),
            _ => {}
        }
    }
    cap_ops(output);
    Ok(())
}

/// `capStream` of tests/parity/go/oracle/main.go.
fn cap_stream(w: usize, h: usize) -> Vec<u8> {
    let mut b = vec![0xff, 0xd8, 0xff, 0xdb, 0, 67, 0];
    b.extend_from_slice(&[1; 64]);
    b.extend_from_slice(&[0xff, 0xc0, 0, 11, 8, (h >> 8) as u8, h as u8]);
    b.extend_from_slice(&[(w >> 8) as u8, w as u8, 1, 1, 0x11, 0]);
    b.extend_from_slice(&[0xff, 0xda, 0, 8, 1, 1, 0, 0, 63, 0]);
    b
}

/// The op `capd` (`capOps` of tests/parity/go/oracle/main.go): a default-`Limits` decode of frames at
/// Go's 1 GiB cap. Go's "over the 1 GiB cap" is the class `cap`, here `AllocLimit`.
fn cap_ops(output: &Path) {
    for (w, h) in [(16776, 64000), (16777, 64000), (65500, 65500)] {
        let name = format!("cap-{w}x{h}.synth");
        let o = Out {
            dir: output,
            name: &name,
        };
        o.safe("capd", || {
            let opts = DecodeOptions {
                format: Format::Gray,
                ..Default::default()
            };
            let r = jpeg::decode(&cap_stream(w, h), &opts, &Limits::default(), &NO);
            let (err, rect) = match r {
                Ok(d) => (d.error, Some(d.image.bounds())),
                Err(Error::AllocLimit { .. }) => return o.file("capd", ".err", b"cap\n"),
                Err(e) => (Some(e), None),
            };
            let b = rect.map(|r| {
                format!("gray {} {} {} {}\n", r.min_x, r.min_y, r.max_x, r.max_y).into_bytes()
            });
            match err {
                Some(e) => o.fail("capd", &e, b),
                None => o.ok("capd", &b.unwrap_or_default()),
            }
        });
    }
}
