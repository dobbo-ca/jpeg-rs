//! Shared test helpers: md5 (the goldens are md5s), a small JSON reader for `golden.json`, and
//! the plane shapes of `tj3DecompressToYUVPlanes8`.
#![allow(dead_code, clippy::needless_range_loop)]

use jpeg::{DecodeOptions, Decoded, Error, Format, Image, Limits, Subsample};
use std::collections::BTreeMap;
use std::sync::atomic::AtomicBool;

pub fn testdata(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(rel)
}

pub fn read(rel: &str) -> Vec<u8> {
    let p = testdata(rel);
    std::fs::read(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
}

pub fn fixtures() -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(testdata("jpg"))
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter(|n| n.ends_with(".jpg"))
        .collect();
    v.sort();
    v
}

pub fn dec(src: &[u8], o: &DecodeOptions) -> Result<Decoded, Error> {
    jpeg::decode(src, o, &Limits::default(), &AtomicBool::new(false))
}

pub fn opts(format: Format, scale: u8) -> DecodeOptions<'static> {
    DecodeOptions {
        format,
        scale,
        ..Default::default()
    }
}

/// MD5 (RFC 1321).
pub fn md5(data: &[u8]) -> String {
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5,
        9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10,
        15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    let k: Vec<u32> = (0..64)
        .map(|i| ((i as f64 + 1.0).sin().abs() * 4294967296.0) as u32)
        .collect();
    let (mut a0, mut b0, mut c0, mut d0) =
        (0x67452301u32, 0xefcdab89u32, 0x98badcfeu32, 0x10325476u32);
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&((data.len() as u64) * 8).to_le_bytes());
    for chunk in msg.chunks(64) {
        let m: Vec<u32> = chunk
            .chunks(4)
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect();
        let (mut a, mut b, mut c, mut d) = (a0, b0, c0, d0);
        for i in 0..64 {
            let (mut f, g) = match i / 16 {
                0 => ((b & c) | (!b & d), i),
                1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                2 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            f = f.wrapping_add(a).wrapping_add(k[i]).wrapping_add(m[g]);
            a = d;
            d = c;
            c = b;
            b = b.wrapping_add(f.rotate_left(S[i]));
        }
        a0 = a0.wrapping_add(a);
        b0 = b0.wrapping_add(b);
        c0 = c0.wrapping_add(c);
        d0 = d0.wrapping_add(d);
    }
    let mut out = String::new();
    for v in [a0, b0, c0, d0] {
        for b in v.to_le_bytes() {
            out.push_str(&format!("{b:02x}"));
        }
    }
    out
}

#[test]
fn md5_known() {
    assert_eq!(md5(b""), "d41d8cd98f00b204e9800998ecf8427e");
    assert_eq!(md5(b"abc"), "900150983cd24fb0d6963f7d28e17f72");
}

/// A JSON value; `golden.json` has objects and strings only.
#[derive(Debug, Clone)]
pub enum Json {
    Str(String),
    Obj(BTreeMap<String, Json>),
}

impl Json {
    pub fn get(&self, k: &str) -> &Json {
        match self {
            Json::Obj(m) => m.get(k).unwrap_or_else(|| panic!("no key {k}")),
            _ => panic!("not an object"),
        }
    }
    pub fn str(&self) -> &str {
        match self {
            Json::Str(s) => s,
            _ => panic!("not a string"),
        }
    }
    pub fn keys(&self) -> Vec<&String> {
        match self {
            Json::Obj(m) => m.keys().collect(),
            _ => panic!("not an object"),
        }
    }
    pub fn has(&self, k: &str) -> bool {
        matches!(self, Json::Obj(m) if m.contains_key(k))
    }
}

pub fn parse_json(s: &str) -> Json {
    let b = s.as_bytes();
    let mut i = 0;
    value(b, &mut i)
}

fn ws(b: &[u8], i: &mut usize) {
    while *i < b.len() && b[*i].is_ascii_whitespace() {
        *i += 1;
    }
}

fn string(b: &[u8], i: &mut usize) -> String {
    assert_eq!(b[*i], b'"');
    *i += 1;
    let st = *i;
    while b[*i] != b'"' {
        *i += 1;
    }
    let s = String::from_utf8(b[st..*i].to_vec()).unwrap();
    *i += 1;
    s
}

fn value(b: &[u8], i: &mut usize) -> Json {
    ws(b, i);
    if b[*i] == b'"' {
        return Json::Str(string(b, i));
    }
    if b[*i] != b'{' {
        // a number or a bool, kept as its text
        let st = *i;
        while !matches!(b[*i], b',' | b'}' | b']') && !b[*i].is_ascii_whitespace() {
            *i += 1;
        }
        return Json::Str(String::from_utf8(b[st..*i].to_vec()).unwrap());
    }
    assert_eq!(b[*i], b'{', "golden.json: unexpected byte at {}", *i);
    *i += 1;
    let mut m = BTreeMap::new();
    loop {
        ws(b, i);
        if b[*i] == b'}' {
            *i += 1;
            return Json::Obj(m);
        }
        if b[*i] == b',' {
            *i += 1;
            continue;
        }
        let k = string(b, i);
        ws(b, i);
        assert_eq!(b[*i], b':');
        *i += 1;
        let v = value(b, i);
        m.insert(k, v);
    }
}

pub fn golden() -> Json {
    parse_json(&String::from_utf8(read("golden.json")).unwrap())
}

fn pad(v: usize, p: usize) -> usize {
    v.div_ceil(p) * p
}

fn luma_size(w: usize, h: usize, r: Subsample) -> (usize, usize) {
    match r {
        Subsample::S422 => (pad(w, 2), h),
        Subsample::S420 => (pad(w, 2), pad(h, 2)),
        _ => (w, h),
    }
}

fn chroma_size(w: usize, h: usize, r: Subsample) -> (usize, usize) {
    let (lw, lh) = luma_size(w, h, r);
    match r {
        Subsample::S422 => (lw / 2, lh),
        Subsample::S420 => (lw / 2, lh / 2),
        _ => (lw, lh),
    }
}

/// Rows `0..h` by columns `0..w` of a stride-padded plane, packed.
pub fn tight(pix: &[u8], stride: usize, w: usize, h: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        out.extend_from_slice(&pix[y * stride..y * stride + w]);
    }
    out
}

/// The md5 of the `tj3DecompressToYUVPlanes8`-shaped buffer and of the `TJPF_GRAY` buffer.
pub fn yuv_and_gray_md5(img: &Image) -> (String, String) {
    match img {
        Image::Gray(g) => {
            let b = tight(
                &g.pix,
                g.stride,
                g.rect.width() as usize,
                g.rect.height() as usize,
            );
            (md5(&b), md5(&b))
        }
        Image::YCbCr(im) => {
            let (w, h) = (im.rect.width() as usize, im.rect.height() as usize);
            let (lw, lh) = luma_size(w, h, im.ratio);
            let (cw, ch) = chroma_size(w, h, im.ratio);
            let mut buf = tight(&im.y, im.y_stride, lw, lh);
            buf.extend(tight(&im.cb, im.c_stride, cw, ch));
            buf.extend(tight(&im.cr, im.c_stride, cw, ch));
            (md5(&buf), md5(&tight(&im.y, im.y_stride, w, h)))
        }
        _ => panic!("unexpected image type"),
    }
}

pub fn rgba_md5(img: &Image) -> String {
    match img {
        Image::Rgba(r) => md5(&r.pix),
        _ => panic!("not RGBA"),
    }
}
