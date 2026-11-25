# CI image: every check runs in this Dockerfile under BuildKit (docker/setup-buildx-action). The
# aarch64 stages (arm64, parity-arm64, fuzz on arm64) build natively on ubuntu-24.04-arm.
# Dependencies (rayon) are fetched in the toolchain stage, before the source is copied, and every later
# stage builds offline from that layer (so the cargo registry is not a cache mount: it would hide it).
# Every tool is installed in a stage before `COPY . .`, so a new commit does not install it again.
FROM rust:1.99.1-bookworm AS toolchain
RUN rustup component add rustfmt clippy && rustup target add aarch64-unknown-linux-gnu \
 && apt-get update && apt-get install -y --no-install-recommends git-lfs ca-certificates \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
# The manifests and lockfiles of the crate and of the fuzz crate, with stub targets, so the fetch is
# a layer that only a change to them rebuilds.
COPY Cargo.toml Cargo.lock ./
COPY tests/fuzz/Cargo.toml tests/fuzz/Cargo.lock tests/fuzz/
RUN mkdir -p src tests/fuzz/fuzz_targets && touch src/lib.rs \
    && for t in $(sed -n 's#^path = "fuzz_targets/\(.*\)\.rs"$#\1#p' tests/fuzz/Cargo.toml); do echo 'fn main() {}' > tests/fuzz/fuzz_targets/$t.rs; done \
    && cargo fetch --locked && cargo fetch --locked --manifest-path tests/fuzz/Cargo.toml
ENV CARGO_NET_OFFLINE=true

FROM toolchain AS src
COPY . .
# The fixtures (git LFS) are pulled here, with the job's token as a build secret. Skipped when no
# secret is given (local builds). `--force` replaces the pre-push hook that the hosted checkout
# leaves in .git/hooks; the hook is not used in the image.
RUN --mount=type=secret,id=gh_token \
    if [ -s /run/secrets/gh_token ]; then \
      git lfs install --local --force && \
      git -c http.extraheader="AUTHORIZATION: basic $(printf 'x-access-token:%s' "$(cat /run/secrets/gh_token)" | base64 -w0)" lfs pull; \
    fi

# The checks run in the build, so a red check fails the build; a rerun of the same source is a
# BuildKit cache hit. The target dirs are cache mounts, so they persist across builds.
FROM src AS ci
# The target cache is shared by every branch and cargo compares mtimes: another branch's newer build
# would be reused as this one's, so the sources are touched first (as in the parity stages). A layer
# touch alone is not enough: two branches building at once write the mount after it, so every RUN
# that runs cargo on a shared mount also touches the .rs files itself, just before cargo starts.
RUN find src examples tests -type f -exec touch {} +
# Which SIMD tiers this runner has (the kernel tests skip the AVX-512 ones without it).
RUN grep -m1 '^model name' /proc/cpuinfo; grep -m1 '^flags' /proc/cpuinfo | tr ' ' '\n' | grep -E '^(avx2|bmi2|avx512(f|bw|vl))$' | tr '\n' ' '; echo
# Threads (std::thread, rayon) are used in src/par.rs only.
RUN if grep -rnE 'thread::(scope|spawn|Builder)|rayon' src | grep -v '^src/par.rs'; then echo 'threads outside src/par.rs'; exit 1; fi
RUN --mount=type=cache,id=jpeg-target,target=/src/target \
    find src examples tests -name '*.rs' -exec touch {} + && cargo fmt --check
RUN --mount=type=cache,id=jpeg-target,target=/src/target \
    find src examples tests -name '*.rs' -exec touch {} + && cargo clippy --locked --all-targets -- -D warnings
# The NEON module is not compiled for x86_64, so lint the aarch64 target too.
RUN --mount=type=cache,id=jpeg-target,target=/src/target \
    find src examples tests -name '*.rs' -exec touch {} + && cargo clippy --locked --all-targets --target aarch64-unknown-linux-gnu -- -D warnings
RUN --mount=type=cache,id=jpeg-target,target=/src/target \
    find src examples tests -name '*.rs' -exec touch {} + && cargo test --locked
RUN --mount=type=cache,id=jpeg-target,target=/src/target \
    find src examples tests -name '*.rs' -exec touch {} + && JPEG_SIMD=scalar cargo test --locked
# The x86-64-v4 build (docs/performance.md, the bench's v4 column) must give the same bytes as the
# default one: the golden decode (against libjpeg-turbo) and encode tests, and the thread, region
# and pyramid tests, released, with their own cache mount, since RUSTFLAGS changes every artifact.
# --target keeps RUSTFLAGS off the build scripts. Skipped on a runner without AVX-512.
RUN --mount=type=cache,id=jpeg-target-v4,target=/src/target \
    if grep -q avx512bw /proc/cpuinfo; then \
      find src examples tests -type f -exec touch {} + \
      && RUSTFLAGS="-C target-cpu=x86-64-v4 -C target-feature=+false-deps-tzcnt,+false-deps-bls" cargo test --locked --release --target x86_64-unknown-linux-gnu --test golden --test encode --test threads --test region --test pyramid; \
    else echo "no AVX-512: x86-64-v4 tests skipped"; fi
RUN --mount=type=cache,id=jpeg-target,target=/src/target \
    find src examples tests -name '*.rs' -exec touch {} + && cargo build --locked --release --target aarch64-unknown-linux-gnu
# The fuzz crate (libfuzzer-sys) has to compile; fuzz.yml runs its targets.
RUN --mount=type=cache,id=jpeg-fuzz-target,target=/src/tests/fuzz/target \
    find src examples tests -name '*.rs' -exec touch {} + && cd tests/fuzz && cargo check --locked --bins

# aarch64 tests, native on ubuntu-24.04-arm, both SIMD tiers (NEON, then JPEG_SIMD=scalar): only the
# tests that the target changes. The unit tests hold the NEON kernel tests, the cross-tier tests (tier
# against scalar) and the tier detection tests; tests/fancy.rs the fancy upsampling rows through the
# API; tests/golden.rs (decode against libjpeg-turbo) and tests/encode.rs (golden encode) run every
# NEON decode and encode kernel end to end; tests/coefficients.rs (truncated RGBA decode, row-by-row
# RowReader) and tests/region.rs (region decode against a crop) run the decode kernels on other row
# widths. limits is left out (no kernel paths of its own). The full suite runs on x86_64 in `ci`; Go
# parity on aarch64 is the parity-arm64 stage. Release builds: the golden tests decode every fixture
# at every scale, which is slow in a debug build.
FROM src AS arm64
RUN --mount=type=cache,id=jpeg-target,target=/src/target \
    find src examples tests -type f -exec touch {} + \
    && cargo test --locked --release --lib --test fancy --test golden --test encode --test coefficients --test region
RUN --mount=type=cache,id=jpeg-target,target=/src/target \
    find src examples tests -name '*.rs' -exec touch {} + && JPEG_SIMD=scalar cargo test --locked --release --lib --test fancy --test golden --test encode --test coefficients --test region

# Go parity: the Go oracle writes the inputs and runs its operations on them; the Rust example runs
# the same operations on the same inputs; the two output trees must be equal (docs/parity.md).
FROM golang:1.27-bookworm AS go-oracle
ENV GOTOOLCHAIN=local CGO_ENABLED=0
WORKDIR /src
COPY tests/parity/go tests/parity/go
RUN --mount=type=cache,target=/root/.cache/go-build cd tests/parity/go/oracle && go build -o /out/oracle .
# The same oracle built for x86-64-v4 (the bench's v4 column); the parity stages use /out/oracle.
RUN --mount=type=cache,target=/root/.cache/go-build cd tests/parity/go/oracle && GOAMD64=v4 go build -o /out/oracle-v4 .

# JPEG_NOSIMD: Go's SIMD kernels give other bytes than libjpeg-turbo on cmp_evil_q255.jpg and on
# streams like it (Go skips that fixture under SIMD), and the Rust kernels fall back to scalar
# there, so the reference is Go's scalar code. The fixtures (git LFS) come from the src stage.
FROM go-oracle AS go-run
ENV JPEG_NOSIMD=1
COPY --from=src /src/tests/data tests/data
COPY tests/fuzz/seeds tests/fuzz/seeds
# The generator prints "<n> inputs" and refuses a repeated name; the count must be the files.
RUN /out/oracle -gen tests/data /inputs tests/fuzz/seeds | tee /gen.txt && [ "$(cut -d' ' -f1 /gen.txt)" = "$(ls /inputs | wc -l)" ] && du -sh /inputs
RUN /out/oracle /inputs /go-out && ls /go-out | wc -l && du -sh /go-out
# The subset the aarch64 oracle runs (stage parity-arm64): every eighth input and the
# sidecar files and the dirty frame of the reused-decoder ops. Go runs it again, so the two trees have the same names.
RUN mkdir /sub && ls /inputs | awk '/^tables\.bin$|^ab-image\.jpg$|^synth_444_227x149\.jpg$|\.starts$|\.rect$|\.tables$/ || NR % 8 == 0' | (cd /inputs && xargs cp -t /sub) \
    && /out/oracle /sub /go-sub && ls /sub | wc -l && ls /go-sub | wc -l

# The target cache is shared by every branch and cargo compares mtimes: another branch's newer build
# would be reused as this one's, so the sources are touched before the oracle is built.
# Another branch's job can rebuild into the shared target between two runs below, so every run is
# preceded by its own touch (t).
# The Go trees are bind mounts from go-run, not COPY: a COPY of the output trees after `COPY . .`
# is slow on every commit. Each Rust output tree is deleted once diffed, so the layer (and the
# cache export) stays empty.
FROM src AS parity
RUN --mount=type=cache,id=jpeg-target,target=/src/target \
    --mount=type=bind,from=go-run,source=/inputs,target=/inputs \
    --mount=type=bind,from=go-run,source=/go-out,target=/go-out \
    t() { find src examples -type f -exec touch {} +; } \
    && t && cargo run --locked --release --example oracle -- --threads 1 /inputs /rust-out \
    && echo "$(ls /inputs | wc -l) inputs, $(ls /go-out | wc -l) outputs" \
    && [ "$(ls /go-out | wc -l)" -ge 40000 ] \
    && sh tests/parity/diff.sh /go-out /rust-out && rm -rf /rust-out \
    && t && cargo run --locked --release --example oracle -- /inputs /rust-out-mt \
    && sh tests/parity/diff.sh /go-out /rust-out-mt && rm -rf /rust-out-mt \
    && t && JPEG_SIMD=scalar cargo run --locked --release --example oracle -- --threads 4 /inputs /rust-out-scalar \
    && sh tests/parity/diff.sh /go-out /rust-out-scalar \
    && rm -rf /rust-out /rust-out-mt /rust-out-scalar

# aarch64 against Go, native on ubuntu-24.04-arm: the oracle example (the NEON kernels) on the
# subset, once per SIMD tier; the output must equal Go's.
FROM src AS parity-arm64
RUN --mount=type=cache,id=jpeg-target,target=/src/target \
    --mount=type=bind,from=go-run,source=/sub,target=/sub \
    --mount=type=bind,from=go-run,source=/go-sub,target=/go-sub \
    t() { find src examples -type f -exec touch {} +; } \
    && t && cargo run --locked --release --example oracle -- --threads 1 /sub /arm-out \
    && echo "$(ls /sub | wc -l) inputs, $(ls /go-sub | wc -l) outputs" \
    && [ "$(ls /go-sub | wc -l)" -ge 5000 ] \
    && sh tests/parity/diff.sh /go-sub /arm-out && rm -rf /arm-out \
    && t && cargo run --locked --release --example oracle -- /sub /arm-out-mt \
    && sh tests/parity/diff.sh /go-sub /arm-out-mt && rm -rf /arm-out-mt \
    && t && JPEG_SIMD=scalar cargo run --locked --release --example oracle -- --threads 4 /sub /arm-out-scalar \
    && sh tests/parity/diff.sh /go-sub /arm-out-scalar \
    && rm -rf /arm-out /arm-out-mt /arm-out-scalar

# Fuzzing (fuzz.yml): nightly and cargo-fuzz, with the targets built into the image. The workflow
# runs it with `docker run`, so a BuildKit cache hit cannot skip the fuzzing. The same stage builds
# natively, with ASan, on x86_64 and on aarch64. Seeds: the jpeg-go
# FuzzDecode and FuzzRowReader seeds (tests/fuzz/seeds) and, for the targets that read jpeg-go's four
# option bytes, each fixture under 64 KiB behind four zero bytes; decoder_reuse gets each fixture under
# 32 KiB twice behind its 13-byte header (see tests/fuzz/fuzz_targets/decoder_reuse.rs).
FROM toolchain AS fuzz-tools
RUN rustup toolchain install nightly --profile minimal \
    && CARGO_NET_OFFLINE=false cargo install cargo-fuzz --locked --version 0.13.1

FROM fuzz-tools AS fuzz
COPY . .
COPY --from=src /src/tests/data tests/data
RUN cargo +nightly fuzz build --fuzz-dir tests/fuzz
RUN mkdir -p /seeds/decode \
    && for f in $(find tests/data/jpg tests/data/truncated tests/data/subsample -name '*.jpg' -size -64k); do { printf '\0\0\0\0'; cat "$f"; } > "/seeds/decode/$(basename "$f")"; done \
    && for t in decode_config coefficients row_reader pyramid; do cp -r /seeds/decode /seeds/$t; done \
    && mkdir -p /seeds/decoder_reuse \
    && for f in $(find tests/data/jpg tests/data/truncated tests/data/subsample -name '*.jpg' -size -32k); do n=$(stat -c %s "$f"); { printf '\0\0\3\0\0'; printf "$(printf '\\%03o\\%03o' $((n / 256)) $((n % 256)))"; printf '\0\0\0\0\0\0'; cat "$f" "$f"; } > "/seeds/decoder_reuse/$(basename "$f")"; done \
    && mkdir -p /seeds/ledger \
    && for f in /seeds/decode/*; do { printf '\377\377\010\003'; cat "$f"; } > "/seeds/ledger/$(basename "$f")"; done \
    && ls /seeds/decode | wc -l

# Benchmark (bench.yml): the Go oracle and the Rust bench example in one image, each built twice:
# the default build, and x86-64-v4 (GOAMD64=v4; RUSTFLAGS as docs/performance.md recommends) as oracle-v4 and
# bench-v4; tests/parity/bench.sh prints both.
FROM src AS bench
COPY --from=go-oracle /out/oracle /usr/local/bin/oracle
COPY --from=go-oracle /out/oracle-v4 /usr/local/bin/oracle-v4
RUN --mount=type=cache,id=jpeg-target,target=/src/target \
    find src examples tests -name '*.rs' -exec touch {} + && cargo build --locked --release --example bench && cp target/release/examples/bench /usr/local/bin/bench
# Its own cache mount (RUSTFLAGS changes every artifact); the sources are touched first, as in `ci`,
# since the mount is shared by every branch.
RUN --mount=type=cache,id=jpeg-target-v4,target=/src/target \
    find src examples tests -type f -exec touch {} + \
    && RUSTFLAGS="-C target-cpu=x86-64-v4 -C target-feature=+false-deps-tzcnt,+false-deps-bls" cargo build --locked --release --target x86_64-unknown-linux-gnu --example bench \
    && cp target/x86_64-unknown-linux-gnu/release/examples/bench /usr/local/bin/bench-v4

# Profile (profile.yml): the bench example with line tables, under valgrind's callgrind.
FROM toolchain AS profile-tools
RUN apt-get update && apt-get install -y --no-install-recommends valgrind && rm -rf /var/lib/apt/lists/*

FROM profile-tools AS profile
COPY . .
COPY --from=src /src/tests/data tests/data
RUN --mount=type=cache,id=jpeg-target,target=/src/target \
    find src examples tests -name '*.rs' -exec touch {} + && CARGO_PROFILE_RELEASE_DEBUG=line-tables-only cargo build --locked --release --example bench && cp target/release/examples/bench /usr/local/bin/bench

# PGO and BOLT (pgo.yml, dispatch only; docs/performance.md): cargo-pgo builds the bench example
# instrumented, trains it on the fixtures, rebuilds it with the profile, and, when BOLT is
# installed, once more through BOLT. The tools are installed before `COPY . .`, so a new commit
# does not install them again. BOLT comes from apt.llvm.org (bookworm has none of its own); if the
# repository or the package is missing the image still builds and the job reports PGO only.
FROM toolchain AS pgo-tools
RUN (rustup component add llvm-tools || rustup component add llvm-tools-preview) \
    && CARGO_NET_OFFLINE=false cargo install cargo-pgo --locked
RUN apt-get update && apt-get install -y --no-install-recommends wget gnupg \
    && { wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o /usr/share/keyrings/llvm.gpg \
         && echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] http://apt.llvm.org/bookworm/ llvm-toolchain-bookworm-19 main" > /etc/apt/sources.list.d/llvm.list \
         && apt-get update && apt-get install -y --no-install-recommends bolt-19 libbolt-19-dev \
         && ln -s /usr/lib/llvm-19/bin/llvm-bolt /usr/local/bin/llvm-bolt \
         && ln -s /usr/lib/llvm-19/bin/merge-fdata /usr/local/bin/merge-fdata \
         && ln -s "$(find /usr/lib/llvm-19 -name libbolt_rt_instr.a | head -n 1)" /usr/local/lib/libbolt_rt_instr.a \
         && test -s /usr/local/lib/libbolt_rt_instr.a; } \
    || echo "BOLT is not available: the PGO job reports PGO only"; \
    rm -rf /var/lib/apt/lists/*

# bench (default build), bench-pgo and, when BOLT worked, bench-bolt (PGO then BOLT), all in
# /usr/local/bin with the Go oracle; tests/parity/pgo.sh times them. The profiles of an earlier run
# stay in the cache mount, so they are deleted first.
FROM pgo-tools AS pgo
COPY . .
COPY --from=src /src/tests/data tests/data
COPY --from=go-oracle /out/oracle /usr/local/bin/oracle
RUN --mount=type=cache,id=jpeg-target-pgo,target=/src/target \
    find src examples tests -type f -exec touch {} + \
    && rm -rf target/pgo-profiles target/bolt-profiles && find target -name 'bench-bolt-*' -delete \
    && host=$(rustc -vV | sed -n 's/^host: //p') && bin=target/$host/release/examples/bench \
    && cargo build --locked --release --example bench && cp target/release/examples/bench /usr/local/bin/bench \
    && cargo pgo build -- --locked --example bench \
    && cargo pgo run -- --locked --example bench -- tests/data 40 "" 1 \
    && cargo pgo run -- --locked --example bench -- tests/data 40 "" 0 \
    && cargo pgo optimize build -- --locked --example bench && cp $bin /usr/local/bin/bench-pgo \
    && if command -v llvm-bolt >/dev/null && command -v merge-fdata >/dev/null; then \
         { cargo pgo bolt build --with-pgo -- --locked --example bench \
           && "$(dirname $bin)/bench-bolt-instrumented" tests/data 40 "" 1 >/dev/null \
           && cargo pgo bolt optimize --with-pgo -- --locked --example bench \
           && cp "$(find target -type f -name 'bench-bolt-optimized' | head -n 1)" /usr/local/bin/bench-bolt; } \
         || echo "BOLT failed: the job reports PGO only"; \
       else echo "no BOLT: the job reports PGO only"; fi
