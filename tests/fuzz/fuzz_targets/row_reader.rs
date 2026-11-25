#![no_main]
use jpeg::{DecodeOptions, Error, Format, Image, RowReader};
use libfuzzer_sys::fuzz_target;
use std::sync::atomic::{AtomicBool, Ordering};

#[path = "../common.rs"]
mod common;
use common::*;

// data = the option bytes of `decode`, then a stream. The reader never panics, and whenever
// decode gives an RGBA image the banded rows are that image (jpeg-go's FuzzRowReader). With bit 0
// of the first option byte the cancel flag is set before one of the first 8 reads: the reader
// then fails with Cancelled, keeps the rows it gave (a prefix of the image) and is closed.
fuzz_target!(|src: &[u8]| {
    let Some(inp) = parse(src) else { return };
    let l = limits();
    let o = DecodeOptions {
        format: Format::Rgba,
        tables: inp.tables,
        ..Default::default()
    };
    let whole = jpeg::decode(inp.data, &o, &l, &NO);
    let mut rr = match RowReader::new(inp.data, inp.tables, &l, &NO) {
        Ok(r) => r,
        Err(_) => {
            assert!(
                whole.is_err(),
                "RowReader::new failed but decode gave an image"
            );
            return;
        }
    };
    let stop_at = (inp.b[0] & 1 == 1).then(|| usize::from(inp.b[1] % 8));
    let cancel = AtomicBool::new(false);
    let (mut rows, mut cancelled) = (Vec::new(), false);
    for i in 0.. {
        if Some(i) == stop_at {
            cancel.store(true, Ordering::Relaxed);
        }
        match rr.next(&cancel) {
            Ok(None) => break,
            Ok(Some(img)) => rows.extend(tight_rgba(img)),
            Err(e) => {
                assert_eq!(e, Error::Cancelled);
                assert!(stop_at.is_some());
                cancelled = true;
                break;
            }
        }
    }
    // Closed after the end or an error: a stateful reader does not come back, on its own (the
    // self-close in `next`) and after `close`, which is idempotent.
    let _ = (rr.config(), rr.truncated());
    assert!(matches!(rr.next(&NO), Err(Error::RowReaderClosed)));
    rr.close();
    assert!(matches!(rr.next(&NO), Err(Error::RowReaderClosed)));
    if let Ok(d) = whole {
        let Image::Rgba(m) = &d.image else {
            panic!("decode to RGBA gave another type")
        };
        let want = tight_rgba(m);
        if cancelled {
            assert!(
                want.starts_with(&rows),
                "rows are not a prefix of the decode"
            );
        } else {
            assert!(rows == want, "rows differ from decode");
        }
    }
});
