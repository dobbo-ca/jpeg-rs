//! `bench <testdata> <ms> [filter [threads]]`: prints `<class>/<fixture>`, output bytes and ns per
//! call for the decode and encode of the `jpg/` fixtures, as `oracle -bench` does for Go. With a
//! filter, only the `<class>/<fixture>` names that contain it run (`tests/parity/profile.sh` profiles
//! one). `threads` is `Limits::max_threads`: 1 (the default) or 0 for `Limits::default()`'s. The
//! generated input `gen_smooth_gray_1536x64` runs after the files. See `tests/parity/bench.sh`.

use jpeg::{
    DecodeOptions, EncodeImage, EncodeOptions, Format, Gray, Image, Limits, Rect, RgbaWindow,
    RowReader,
};
use std::sync::atomic::AtomicBool;
use std::time::Instant;
use std::{env, fs, io};

/// The best time of one call over five batches, each about `ms / 5` long.
fn time_op(ms: u64, mut f: impl FnMut()) -> u128 {
    f();
    let t = Instant::now();
    f();
    let d = t.elapsed().as_nanos().max(1);
    let k = (u128::from(ms) * 1_000_000 / 5 / d).max(1);
    let mut best = u128::MAX;
    for _ in 0..5 {
        let t = Instant::now();
        for _ in 0..k {
            f();
        }
        best = best.min(t.elapsed().as_nanos() / k);
    }
    best
}

/// MD5 (RFC 1321), to show on stderr that both benches decode the same generated stream.
fn md5(data: &[u8]) -> String {
    const S: [u32; 16] = [7, 12, 17, 22, 5, 9, 14, 20, 4, 11, 16, 23, 6, 10, 15, 21];
    let k: Vec<u32> = (0..64)
        .map(|i| ((i as f64 + 1.0).sin().abs() * 4294967296.0) as u32)
        .collect();
    let mut h = [0x67452301u32, 0xefcdab89, 0x98badcfe, 0x10325476];
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&((data.len() as u64) * 8).to_le_bytes());
    for chunk in msg.as_chunks::<64>().0 {
        let m: Vec<u32> = chunk
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes(*w))
            .collect();
        let [mut a, mut b, mut c, mut d] = h;
        for i in 0..64 {
            let (f, g) = match i / 16 {
                0 => ((b & c) | (!b & d), i),
                1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                2 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let f = f.wrapping_add(a).wrapping_add(k[i]).wrapping_add(m[g]);
            a = d;
            d = c;
            c = b;
            b = b.wrapping_add(f.rotate_left(S[i / 16 * 4 + i % 4]));
        }
        for (x, v) in h.iter_mut().zip([a, b, c, d]) {
            *x = x.wrapping_add(v);
        }
    }
    h.iter()
        .flat_map(|v| v.to_le_bytes())
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The generated input: a 1536x64 gray ramp `byte(2x+y)` (a tiff-rs strip), encoded at quality 95
/// with this crate's encoder. `oracle -bench` makes the same with Go's; both print the length and
/// the md5 of the stream, which must agree.
fn gen_smooth_gray_1536x64(limits: &Limits, no: &AtomicBool) -> io::Result<Vec<u8>> {
    let (w, h) = (1536usize, 64usize);
    let g = Gray {
        pix: (0..h)
            .flat_map(|y| (0..w).map(move |x| (2 * x + y) as u8))
            .collect(),
        stride: w,
        rect: Rect::new(0, 0, w as i32, h as i32),
    };
    let o = EncodeOptions {
        quality: 95,
        chroma444: false,
    };
    let j = jpeg::encode(EncodeImage::Gray(&g), &o, limits, no).map_err(io::Error::other)?;
    eprintln!("rust: {GEN}: {} bytes, md5 {}", j.len(), md5(&j));
    Ok(j)
}

const GEN: &str = "gen_smooth_gray_1536x64";

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let (dir, ms, filter, threads) = match args.as_slice() {
        [_, dir, ms] => (dir, ms, "", "1"),
        [_, dir, ms, filter] => (dir, ms, filter.as_str(), "1"),
        [_, dir, ms, filter, threads] => (dir, ms, filter.as_str(), threads.as_str()),
        _ => {
            eprintln!("usage: bench <testdata> <ms> [filter [threads]]");
            std::process::exit(2);
        }
    };
    let ms: u64 = ms.parse().unwrap_or(1).max(1);
    let mut limits = Limits::default();
    match threads.parse::<usize>() {
        Ok(0) => {}
        Ok(n) => limits.max_threads = n,
        Err(_) => {
            eprintln!("bench: threads must be a number");
            std::process::exit(2);
        }
    }
    eprintln!(
        "rust: available_parallelism {}, max_threads {}",
        std::thread::available_parallelism().map_or(1, |n| n.get()),
        limits.max_threads
    );
    let no = AtomicBool::new(false);
    let mut names: Vec<String> = fs::read_dir(format!("{dir}/jpg"))?
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter(|n| n.ends_with(".jpg"))
        .collect();
    names.sort();
    names.push(format!("{GEN}.jpg"));
    let emit =
        |class: &str, name: &str, out: usize, ns: u128| println!("{class}/{name}\t{out}\t{ns}");
    for file in names {
        let src = if file == format!("{GEN}.jpg") {
            gen_smooth_gray_1536x64(&limits, &no)?
        } else {
            fs::read(format!("{dir}/jpg/{file}"))?
        };
        let name = file.trim_end_matches(".jpg");
        let want = |class: &str| format!("{class}/{name}").contains(filter);
        let Ok(cfg) = jpeg::decode_config(&src, None, &limits) else {
            continue;
        };
        if cfg.width * cfg.height > 1 << 22 {
            continue;
        }
        let decode = |f: Format, scale: u8| {
            let o = DecodeOptions {
                format: f,
                scale,
                ..Default::default()
            };
            jpeg::decode(&src, &o, &limits, &no)
        };
        // One Decoder per class and file, as Go's bench holds one `jpeg.Decoder`: both sides
        // reuse their planes and output, and neither copies.
        for (class, f, scale, pyramid) in [
            ("dec-rgba", Format::Rgba, 1u8, false),
            ("dec-ycc", Format::YCbCr, 1, false),
            ("dec-gray", Format::Gray, 1, false),
            ("dec-rgba2", Format::Rgba, 2, false),
            ("dec-rgba4", Format::Rgba, 4, false),
            ("dec-rgba8", Format::Rgba, 8, false),
            // No Go counterpart (bench.sh prints them as Rust-only rows): scales 2 and 4 in the
            // pyramid mode (DecodeOptions::pyramid).
            ("dec-rgba2-pyr", Format::Rgba, 2, true),
            ("dec-rgba4-pyr", Format::Rgba, 4, true),
        ] {
            if !want(class) {
                continue;
            }
            let o = DecodeOptions {
                format: f,
                scale,
                pyramid,
                ..Default::default()
            };
            let mut dec = jpeg::Decoder::new();
            let first = dec.decode(&src, &o, &limits, &no).map(|d| {
                let r = d.image.bounds();
                (r.width() * r.height() * 4) as usize
            });
            if let Ok(out) = first {
                emit(
                    class,
                    name,
                    out,
                    time_op(ms, || {
                        drop(std::hint::black_box(dec.decode(&src, &o, &limits, &no)))
                    }),
                );
            }
        }
        // Into a buffer the caller keeps: Go's Dst, on the same reused Decoder.
        if want("dec-into") {
            let o = DecodeOptions {
                format: Format::Rgba,
                scale: 1,
                ..Default::default()
            };
            let mut pix = vec![0u8; 4 * cfg.width * cfg.height];
            let mut dec = jpeg::Decoder::new();
            let mut run = || {
                let w = RgbaWindow {
                    pix: &mut pix,
                    stride: 4 * cfg.width,
                };
                dec.decode_into_rgba(&src, &o, &limits, &no, w)
            };
            if run().is_ok() {
                emit(
                    "dec-into",
                    name,
                    cfg.width * cfg.height * 4,
                    time_op(ms, || drop(std::hint::black_box(run()))),
                );
            }
        }
        // A 512x512 region of the NDPI-style strip with its restart offsets (tests/data), the
        // call a slide viewer makes.
        if name == "strip_rst"
            && want("dec-region")
            && let Ok(list) = fs::read_to_string(format!("{dir}/strip_rst.starts"))
        {
            let starts: Vec<usize> = list
                .split_whitespace()
                .filter_map(|v| v.parse().ok())
                .collect();
            let o = DecodeOptions {
                format: Format::Rgba,
                region: Some(Rect::new(250, 300, 762, 812)),
                restart: Some(&starts),
                ..Default::default()
            };
            let mut dec = jpeg::Decoder::new();
            if dec.decode(&src, &o, &limits, &no).is_ok() {
                emit(
                    "dec-region",
                    name,
                    512 * 512 * 4,
                    time_op(ms, || {
                        drop(std::hint::black_box(dec.decode(&src, &o, &limits, &no)))
                    }),
                );
            }
        }
        if want("rowreader") {
            emit(
                "rowreader",
                name,
                cfg.width * cfg.height * 4,
                time_op(ms, || {
                    if let Ok(mut r) = RowReader::new(&src, None, &limits, &no) {
                        while let Ok(Some(img)) = r.next(&no) {
                            std::hint::black_box(img);
                        }
                    }
                }),
            );
        }
        if want("coef") {
            emit(
                "coef",
                name,
                cfg.width * cfg.height,
                time_op(ms, || {
                    drop(std::hint::black_box(jpeg::decode_coefficients(
                        &src, None, &limits, &no,
                    )))
                }),
            );
        }
        for (class, f, c444) in [
            ("enc-rgba420", Format::Rgba, false),
            ("enc-rgba444", Format::Rgba, true),
            ("enc-gray", Format::Gray, false),
            ("enc-ycc", Format::YCbCr, false),
        ] {
            if !want(class) {
                continue;
            }
            let Ok(d) = decode(f, 1) else { continue };
            // One Encoder per class and file, as Go's bench reuses its Encoder.
            let mut encoder = jpeg::Encoder::default();
            let mut enc = |img: &Image| {
                let o = EncodeOptions {
                    quality: 75,
                    chroma444: c444,
                };
                encoder.encode(EncodeImage::from(img), &o, &limits, &no)
            };
            let Ok(j) = enc(&d.image) else { continue };
            emit(
                class,
                name,
                j.len(),
                time_op(ms, || drop(std::hint::black_box(enc(&d.image)))),
            );
        }
    }
    Ok(())
}
