//! The only user of threads: runs a call's jobs on a rayon pool. A call's workers are at most
//! `Limits::max_threads`, the calling thread included. The pool is the caller's
//! ([`ThreadPool`], on `DecodeOptions::pool` or `Decoder::with_pool`) or the crate's lazy shared
//! one; its threads outlive the call, so a small frame pays a wake-up, not a thread start.
//! Every job's data is a disjoint `&mut` slice the caller split off, and rayon's scope lets a
//! pool's worker borrow it, so nothing here is `unsafe`. That is the one reason for the
//! dependency: `std::thread::scope` cannot reuse a thread.
//!
//! Memory: a pool's thread stacks belong to the pool, not to a call, so the ledger books a call's
//! own scratch only (the band starts, the extra colour rows).

use crate::Error;
use crate::scan::Ctx;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

/// The stack of a pool thread. The deepest frame is the scan loop (a few KiB of block scratch)
/// over the Huffman decode; the tables live in the caller's frame.
const THREAD_STACK: usize = 512 << 10;

/// Worker threads that decodes share. Create one, keep it, and pass it in `DecodeOptions::pool`
/// or to `Decoder::with_pool`: each call then runs its parallel passes on these threads. A call
/// without one uses the crate's lazy shared pool, which starts on the first call with
/// `max_threads` above 1 that has a pass worth threading, and never has more than
/// [`std::thread::available_parallelism`] threads; a caller's pool always wins.
/// `Limits::max_threads` still caps the workers of one call, the calling thread included, and a
/// call uses at most one more than the pool has threads. Clones share the threads, which stop
/// when the last clone drops. A pool is not charged to `Limits::max_alloc`; its threads' stacks
/// and bookkeeping are its owner's.
#[derive(Clone, Debug)]
pub struct ThreadPool(Arc<rayon::ThreadPool>);

impl ThreadPool {
    /// A pool of `threads` workers (at least 1), started now.
    ///
    /// # Errors
    /// [`Error::OutOfMemory`] if the operating system will not start the threads.
    pub fn new(threads: usize) -> Result<ThreadPool, Error> {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads.max(1))
            .stack_size(THREAD_STACK)
            .thread_name(|i| format!("jpeg-worker-{i}"))
            .build()
            .map(|p| ThreadPool(Arc::new(p)))
            .map_err(|_| Error::OutOfMemory)
    }

    /// How many worker threads the pool has.
    pub fn threads(&self) -> usize {
        self.0.current_num_threads()
    }
}

/// The crate's shared pool: built on first use, `available_parallelism` threads, no other
/// state. `None` if the threads could not be started (calls then run serially).
static SHARED: OnceLock<Option<ThreadPool>> = OnceLock::new();

fn shared_pool() -> Option<&'static ThreadPool> {
    SHARED
        .get_or_init(|| {
            ThreadPool::new(std::thread::available_parallelism().map_or(1, |n| n.get())).ok()
        })
        .as_ref()
}

/// Whether the shared pool has been started (for tests: `max_threads` 1, and a caller's pool,
/// never start it).
#[doc(hidden)]
pub fn shared_pool_started() -> bool {
    SHARED.get().is_some()
}

/// The pool a threaded pass of this call runs on and how many workers it may use, the calling
/// thread included; `None` means run it serially. The shared pool starts here, so a caller asks
/// only once it has a pass worth threading.
pub(crate) fn stage<'a>(cx: &Ctx<'a>) -> Option<(&'a ThreadPool, usize)> {
    if cx.threads < 2 {
        return None;
    }
    let pool = match cx.pool {
        Some(p) => p,
        None => shared_pool()?,
    };
    Some((pool, cx.threads.min(pool.threads() + 1)))
}

/// The bytes of entropy data that pay for one more worker of a band decode: a wake-up of a pool
/// thread costs about 10 us and decoding takes about 6 ns a byte.
// ponytail: a fixed byte count; tune it from the bench's per-size table.
const BAND_BYTES_PER_THREAD: usize = 8 << 10;

/// The pixels that pay for one more worker of the colour conversion (about 0.33 ns a pixel).
// ponytail: a fixed pixel count; tune it from the bench's per-size table.
const STRIPE_PIXELS_PER_THREAD: usize = 1 << 17;

/// The workers a band decode of `bytes` entropy bytes can use.
pub(crate) fn band_threads(bytes: usize) -> usize {
    bytes / BAND_BYTES_PER_THREAD
}

/// The workers the colour conversion of `pixels` output pixels can use.
pub(crate) fn stripe_threads(pixels: usize) -> usize {
    pixels / STRIPE_PIXELS_PER_THREAD
}

/// The threads of one call that are running now and the most at once, the calling thread
/// included. Per call, so a test can read it without global state.
pub(crate) struct Gauge {
    live: AtomicUsize,
    peak: AtomicUsize,
}

impl Gauge {
    pub(crate) fn new() -> Self {
        Self {
            live: AtomicUsize::new(1),
            peak: AtomicUsize::new(1),
        }
    }

    pub(crate) fn peak(&self) -> usize {
        self.peak.load(Ordering::Relaxed)
    }
}

/// A pool worker counted in a [`Gauge`] while it runs jobs.
struct Live<'a>(&'a Gauge);

impl<'a> Live<'a> {
    fn enter(g: &'a Gauge) -> Self {
        let n = g.live.fetch_add(1, Ordering::Relaxed) + 1;
        g.peak.fetch_max(n, Ordering::Relaxed);
        Live(g)
    }
}

impl Drop for Live<'_> {
    fn drop(&mut self) {
        self.0.live.fetch_sub(1, Ordering::Relaxed);
    }
}

/// A mutex's data even if a job panicked while holding it (the panic is re-raised by the scope).
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Runs `f` on every job and returns the results in job order, on at most `nw` workers: the
/// caller and up to `nw - 1` tasks on `pool`, which take jobs from one queue as they wake (a task
/// that wakes after the queue is empty does nothing). A pool whose threads are busy only leaves
/// the jobs to the caller, so this never fails. A panic in a job is re-raised here, after every
/// worker has returned.
pub(crate) fn run<J: Send, T: Send>(
    pool: &ThreadPool,
    gauge: Option<&Gauge>,
    nw: usize,
    jobs: Vec<J>,
    f: impl Fn(J) -> T + Sync,
) -> Vec<T> {
    let n = jobs.len();
    let mut queue: Vec<(usize, J)> = jobs.into_iter().enumerate().collect();
    queue.reverse(); // pop gives job 0 first
    let queue = Mutex::new(queue);
    let out = Mutex::new(Vec::with_capacity(n));
    // A pool task counts in the gauge from its first job, so a task that wakes to an empty queue
    // is not a worker the call used.
    let work = |gauge: Option<&Gauge>| {
        let mut live = None;
        loop {
            let job = lock(&queue).pop();
            let Some((i, j)) = job else { return };
            if live.is_none() {
                live = gauge.map(Live::enter);
            }
            let t = f(j);
            lock(&out).push((i, t));
        }
    };
    pool.0.in_place_scope(|s| {
        for _ in 1..nw.min(n) {
            let work = &work;
            s.spawn(move |_| work(gauge));
        }
        work(None);
    });
    let mut all = out.into_inner().unwrap_or_else(|e| e.into_inner());
    all.sort_by_key(|&(i, _)| i);
    all.into_iter().map(|(_, t)| t).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn results_come_back_in_job_order() {
        let pool = ThreadPool::new(3).unwrap();
        let g = Gauge::new();
        let out = run(&pool, Some(&g), 4, (0..40).collect(), |i: usize| i * 2);
        assert_eq!(out, (0..40).map(|i| i * 2).collect::<Vec<_>>());
        assert!(g.peak() >= 1 && g.peak() <= 4);
    }

    #[test]
    fn one_job_or_one_worker_spawns_nothing() {
        let pool = ThreadPool::new(3).unwrap();
        let g = Gauge::new();
        assert_eq!(run(&pool, Some(&g), 4, vec![7u8], |x| x + 1), vec![8]);
        assert_eq!(
            run(&pool, Some(&g), 1, vec![1u8, 2, 3], |x| x),
            vec![1, 2, 3]
        );
        assert_eq!(g.peak(), 1);
    }

    #[test]
    #[should_panic(expected = "boom")]
    fn a_panic_in_a_job_is_raised_to_the_caller() {
        let pool = ThreadPool::new(2).unwrap();
        run(&pool, None, 3, (0..8).collect(), |i: usize| {
            assert!(i != 5, "boom");
            i
        });
    }
}
