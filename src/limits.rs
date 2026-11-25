use crate::Error;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Caps on one call, passed to every entry point that takes input of any size. Go's `Limits` had
/// `MaxPixels` and `MaxBytes`, both unbounded at zero; here every field is finite.
///
/// | Field | Unit | Default | Over the limit |
/// |---|---|---|---|
/// | `max_pixels` | pixels (width times height) | 2^32 | [`Error::PixelLimit`] |
/// | `max_alloc` | bytes, the peak of the whole call | 1 GiB | [`Error::AllocLimit`] |
/// | `max_threads` | threads, the caller's included | [`std::thread::available_parallelism`] | none: the call runs on fewer threads, same output |
///
/// ```
/// use jpeg::Limits;
/// // One thread (no pool is touched) and 64 MiB for the whole call.
/// let limits = Limits { max_threads: 1, max_alloc: 64 << 20, ..Limits::default() };
/// # assert_eq!(limits.max_pixels, 1 << 32);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// Most pixels (width times height) of a frame, checked from the SOF before any allocation;
    /// a larger frame gives [`Error::PixelLimit`]. The default is 2^32.
    pub max_pixels: u64,
    /// Most bytes of working memory one call may hold, all threads together: planes, coefficient
    /// blocks, restart starts, RGBA and copies, each charged before it is allocated in whole
    /// 8 KiB pages, plus the scratch rows and tables of any extra worker (a pool's thread stacks
    /// are the pool's, not charged). Fixed state (tables, under 64 KiB) is
    /// not charged. The ledger enforces it as the peak of the call; going over it gives
    /// [`Error::AllocLimit`].
    ///
    /// This is a `u64`, not a `usize`: tiff-rs passes a `u64` budget down
    /// and a 32-bit target must still be able to express a budget above 4 GiB.
    ///
    /// The default, 1 GiB, is also Go's unset `MaxBytes`: a `decode` then refuses a frame whose
    /// need, less the page rounding only the ledger counts, is over Go's fixed 1 GiB cap (Go's
    /// "over the 1 GiB cap" error, here [`Error::AllocLimit`]). The ledger still enforces: its
    /// budget is the larger of `max_alloc` and that already-checked need, so the peak stays under
    /// 1 GiB plus the rounding. Any other value is Go's `MaxBytes`: the whole need and every
    /// charge count against it, and Rust (unlike Go) lets it exceed 1 GiB.
    pub max_alloc: u64,
    /// Most threads one call uses, the calling thread included. 1 runs on the calling thread
    /// and never touches a pool; 0 counts as 1; more than 256 counts as 256. The default is
    /// [`std::thread::available_parallelism`]. A decode uses as many extra workers (from a
    /// [`crate::ThreadPool`]) as its input is big enough to feed, and only when the memory for
    /// them fits in `max_alloc`; otherwise it runs on fewer, so the output and any error are the
    /// same for every value.
    pub max_threads: usize,
}

/// The default `max_alloc`, and Go's fixed cap on one decode.
const DEFAULT_MAX_ALLOC: u64 = 1 << 30;

/// Go's `maxWorkers`.
const MAX_THREADS: usize = 256;

/// `available_parallelism`, read once: it reads cgroup files on every call.
fn default_threads() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_pixels: 1 << 32,
            max_alloc: DEFAULT_MAX_ALLOC,
            max_threads: default_threads(),
        }
    }
}

impl Limits {
    /// `max_threads` as the threads a call may use: 1 to 256.
    pub(crate) fn threads(&self) -> usize {
        self.max_threads.clamp(1, MAX_THREADS)
    }

    pub(crate) fn check_pixels(&self, w: usize, h: usize) -> Result<(), Error> {
        let n = w as u64 * h as u64;
        if n > self.max_pixels {
            return Err(Error::PixelLimit {
                got: n,
                max: self.max_pixels,
            });
        }
        Ok(())
    }

    pub(crate) fn check_bytes(&self, need: u64) -> Result<(), Error> {
        if need > self.max_alloc {
            return Err(Error::AllocLimit {
                got: need,
                max: self.max_alloc,
            });
        }
        Ok(())
    }

    /// `check_bytes` for a decode whose `need` includes `ext`, the page rounding only the ledger
    /// counts. Under the default `max_alloc` (Go's unset `MaxBytes`) only the 1 GiB cap on
    /// `need - ext` applies; the ledger then enforces `need`, which bounds the peak.
    pub(crate) fn check_decode_bytes(
        &self,
        need: u64,
        ext: u64,
        ledger: &Ledger,
    ) -> Result<(), Error> {
        if self.max_alloc != DEFAULT_MAX_ALLOC {
            return self.check_bytes(need);
        }
        let got = need.saturating_sub(ext);
        if got > DEFAULT_MAX_ALLOC {
            return Err(Error::AllocLimit {
                got,
                max: DEFAULT_MAX_ALLOC,
            });
        }
        ledger.raise_max(need);
        Ok(())
    }
}

/// The bytes a call has charged against `Limits::max_alloc`: Go's `internal/ledger`, shared by
/// every thread of a call. Charges stay booked for the call, except a [`Held`] one.
pub(crate) struct Ledger {
    max: AtomicU64,
    used: AtomicU64,
    peak: AtomicU64,
}

fn pages(n: u64) -> u64 {
    n.saturating_add(8191) & !8191
}

impl Ledger {
    pub(crate) fn new(l: &Limits) -> Self {
        Self {
            max: AtomicU64::new(l.max_alloc),
            used: AtomicU64::new(0),
            peak: AtomicU64::new(0),
        }
    }

    /// The bytes booked so far.
    pub(crate) fn used(&self) -> u64 {
        self.used.load(Ordering::Relaxed)
    }

    /// The most bytes booked at any one time: an upper bound of the call's heap peak.
    pub(crate) fn peak(&self) -> u64 {
        self.peak.load(Ordering::Relaxed)
    }

    /// Lifts the budget to at least `v`.
    pub(crate) fn raise_max(&self, v: u64) {
        self.max.fetch_max(v, Ordering::Relaxed);
    }

    /// Books `n` bytes (already page rounded); fails, booking nothing, when they would pass the
    /// budget.
    fn add(&self, n: u64) -> Result<(), Error> {
        let max = self.max.load(Ordering::Relaxed);
        let mut cur = self.used.load(Ordering::Relaxed);
        loop {
            let u = cur.saturating_add(n);
            if u > max {
                return Err(Error::AllocLimit { got: u, max });
            }
            match self
                .used
                .compare_exchange_weak(cur, u, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_) => {
                    self.peak.fetch_max(u, Ordering::Relaxed);
                    return Ok(());
                }
                Err(c) => cur = c,
            }
        }
    }

    /// Books `n` bytes about to be allocated, rounded up to a page; fails, booking nothing, when
    /// they would pass the budget.
    pub(crate) fn charge(&self, n: u64) -> Result<(), Error> {
        self.add(pages(n))
    }

    /// Books `n` bytes (a page rounded) until the guard drops, or `None` when they do not fit.
    /// For thread scratch: a caller that gets `None` runs serially, so the budget never turns
    /// into an error a single-threaded call would not give.
    pub(crate) fn hold(&self, n: u64) -> Option<Held<'_>> {
        let n = pages(n);
        self.add(n).ok()?;
        Some(Held { ledger: self, n })
    }

    /// Charges and allocates `n` zero bytes.
    pub(crate) fn bytes(&self, n: usize) -> Result<Vec<u8>, Error> {
        self.charge(n as u64)?;
        filled(n, 0u8)
    }
}

/// Bytes booked on a [`Ledger`] until this drops.
pub(crate) struct Held<'a> {
    ledger: &'a Ledger,
    n: u64,
}

impl Drop for Held<'_> {
    fn drop(&mut self) {
        self.ledger.used.fetch_sub(self.n, Ordering::Relaxed);
    }
}

/// `n` copies of `v`, with the allocation failure as an error.
pub(crate) fn filled<T: Clone>(n: usize, v: T) -> Result<Vec<T>, Error> {
    let mut out = Vec::new();
    out.try_reserve_exact(n).map_err(|_| Error::OutOfMemory)?;
    out.resize(n, v);
    Ok(out)
}

/// `v` as `n` bytes. With `zero` false the first `min(len, n)` bytes keep what they held (a
/// buffer a [`crate::Decoder`] reuses: the caller writes every byte before it reads one, as Go's
/// reused planes rely on); the rest, and all of them when `zero`, are zero. Only growing
/// writes memory.
pub(crate) fn refit(mut v: Vec<u8>, n: usize, zero: bool) -> Result<Vec<u8>, Error> {
    if zero {
        v.clear();
    }
    if v.len() >= n {
        v.truncate(n);
        return Ok(v);
    }
    v.try_reserve_exact(n - v.len())
        .map_err(|_| Error::OutOfMemory)?;
    v.resize(n, 0);
    Ok(v)
}

/// `n` copies of `z` in `v`'s buffer: it is cleared first, and only grows (one allocation) when
/// its capacity is below `n`. For a buffer a [`crate::Decoder`] reuses that must start zeroed.
pub(crate) fn refill<T: Clone>(mut v: Vec<T>, n: usize, z: T) -> Result<Vec<T>, Error> {
    v.clear();
    v.try_reserve_exact(n).map_err(|_| Error::OutOfMemory)?;
    v.resize(n, z);
    Ok(v)
}

/// Go's `ctx.Done()` poll.
pub(crate) fn check(cancel: &AtomicBool) -> Result<(), Error> {
    if cancel.load(Ordering::Relaxed) {
        return Err(Error::Cancelled);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Under the default `max_alloc` the ledger enforces the checked need, not nothing.
    #[test]
    fn default_max_alloc_keeps_the_ledger_enforcing() {
        let l = Limits::default();
        let ledger = Ledger::new(&l);
        let (need, ext) = ((1u64 << 30) + 3 * 8192, 4 * 8192);
        assert_eq!(l.check_decode_bytes(need, ext, &ledger), Ok(()));
        assert_eq!(ledger.charge(need), Ok(()));
        assert!(matches!(
            ledger.charge(1),
            Err(Error::AllocLimit { max, .. }) if max == need
        ));
        // Go's rule: need less ext over the cap is refused.
        let ledger = Ledger::new(&l);
        assert!(matches!(
            l.check_decode_bytes((1 << 30) + ext + 1, ext, &ledger),
            Err(Error::AllocLimit { .. })
        ));
    }

    /// A hold counts against the budget until it drops, and a failed hold books nothing.
    #[test]
    fn hold_releases_on_drop() {
        let l = Limits {
            max_alloc: 3 * 8192,
            ..Limits::default()
        };
        let ledger = Ledger::new(&l);
        {
            let _h = ledger.hold(2 * 8192).unwrap();
            assert_eq!(ledger.used(), 2 * 8192);
            assert!(ledger.hold(2 * 8192).is_none());
            assert_eq!(ledger.used(), 2 * 8192);
        }
        assert_eq!(ledger.used(), 0);
        assert_eq!(ledger.charge(3 * 8192), Ok(()));
        assert_eq!(ledger.peak(), 3 * 8192);
    }

    #[test]
    fn threads_are_clamped() {
        let t = |n| {
            Limits {
                max_threads: n,
                ..Limits::default()
            }
            .threads()
        };
        assert_eq!((t(0), t(1), t(7), t(256), t(1 << 20)), (1, 1, 7, 256, 256));
    }
}
