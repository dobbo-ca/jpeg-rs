//! `ppm [--scale N] <tiles> <outputs>`: decodes every `.jpg` in `<tiles>` (`Format::Rgba`, scale 1;
//! with `--scale N` of 2, 4 or 8, the pyramid mode at that scale, `DecodeOptions::pyramid`) and
//! writes `<stem>.ppm`, the RGB samples as binary PPM, to `<outputs>`, for comparing with
//! `tests/parity/go/oracle -ppm`. Real-file tiles must decode, so any error, also one returned
//! beside an image, stops the run. With `--scale`, a tile under N samples in either direction is
//! skipped.

use jpeg::{DecodeOptions, Format, Image, Limits};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::{env, fs};

fn usage() -> ! {
    eprintln!("usage: ppm [--scale 2|4|8] <tiles> <outputs>");
    std::process::exit(2);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args: Vec<String> = env::args().collect();
    let mut scale = 1u8;
    if let Some(i) = args.iter().position(|a| a == "--scale") {
        let n = args.get(i + 1).and_then(|n| n.parse::<u8>().ok());
        let Some(n @ (2 | 4 | 8)) = n else { usage() };
        scale = n;
        args.drain(i..i + 2);
    }
    let [_, input, output] = args.as_slice() else {
        usage()
    };
    fs::create_dir_all(output)?;
    let mut names: Vec<PathBuf> = Vec::new();
    for e in fs::read_dir(input)? {
        let p = e?.path();
        if p.extension().is_some_and(|x| x == "jpg") {
            names.push(p);
        }
    }
    names.sort();
    // The bounds of the Go side (ppmLim in tests/parity/go/oracle/ppm.go).
    let limits = Limits {
        max_pixels: 1 << 26,
        max_alloc: 1 << 30,
        ..Limits::default()
    };
    let opts = DecodeOptions {
        format: Format::Rgba,
        scale,
        pyramid: scale > 1,
        ..Default::default()
    };
    let cancel = AtomicBool::new(false);
    let mut sampling: BTreeMap<String, usize> = BTreeMap::new();
    let (mut skipped, mut rgb) = (0usize, 0usize);
    for p in &names {
        let fail = |e: &dyn std::fmt::Display| format!("{}: {e}", p.display());
        let src = fs::read(p)?;
        let cfg = jpeg::decode_config(&src, None, &limits).map_err(|e| fail(&e))?;
        if cfg.width.min(cfg.height) < usize::from(scale) {
            skipped += 1; // the pyramid mode refuses it
            continue;
        }
        *sampling.entry(format!("{:?}", cfg.ratio)).or_default() += 1;
        rgb += usize::from(cfg.rgb);
        let d = jpeg::decode(&src, &opts, &limits, &cancel).map_err(|e| fail(&e))?;
        if let Some(e) = d.error {
            return Err(fail(&e).into());
        }
        let Image::Rgba(m) = d.image else {
            return Err(fail(&"decoded a non-RGBA image").into());
        };
        let w = usize::try_from(m.rect.max_x - m.rect.min_x)?;
        let h = usize::try_from(m.rect.max_y - m.rect.min_y)?;
        let mut buf = format!("P6\n{w} {h}\n255\n").into_bytes();
        buf.reserve(3 * w * h);
        for row in m.pix.chunks(m.stride).take(h) {
            for px in row[..4 * w].as_chunks::<4>().0 {
                buf.extend_from_slice(&px[..3]);
            }
        }
        // The stem keeps its dots (`CMU-1_svs.ifd0.000123`), so no `with_extension`.
        let stem = p.file_stem().unwrap_or_default().to_string_lossy();
        fs::write(PathBuf::from(output).join(format!("{stem}.ppm")), buf)?;
    }
    println!(
        "{} tiles decoded, {skipped} smaller than the scale; sampling {sampling:?}; {rgb} read as RGB",
        names.len() - skipped
    );
    Ok(())
}
