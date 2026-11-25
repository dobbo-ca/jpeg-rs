package main

import (
	"bytes"
	"crypto/md5"
	"fmt"
	"image"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"

	jpeg "github.com/dobbo-ca/jpeg-go"
)

// benchDir times the decode and encode of the jpg/ fixtures of a tests/data directory, one line per
// operation: "<class>/<fixture>\t<output bytes>\t<ns>", the best of five batches of calls that
// last about ms in all. examples/bench.rs does the same with the Rust crate; tests/parity/bench.sh joins
// the two. After the files it runs the generated input genGray (see genSmoothGray). workers is
// DecodeOptions.Workers of every dec-* class (0 is Go's default, serial; -1 is GOMAXPROCS); the row
// reader, coefficients and encoder have no workers in Go.
func benchDir(dir string, ms, workers int) error {
	if workers < 0 {
		workers = runtime.GOMAXPROCS(0)
	}
	fmt.Fprintf(os.Stderr, "go: GOMAXPROCS %d, NumCPU %d, workers %d\n", runtime.GOMAXPROCS(0), runtime.NumCPU(), workers)
	m, err := filepath.Glob(filepath.Join(dir, "jpg", "*.jpg"))
	if err != nil || len(m) == 0 {
		return fmt.Errorf("no fixtures in %s/jpg", dir)
	}
	sort.Strings(m)
	m = append(m, filepath.Join(dir, "jpg", genGray+".jpg"))
	for _, p := range m {
		var b []byte
		var err error
		if strings.HasSuffix(p, genGray+".jpg") {
			b, err = genSmoothGray()
		} else {
			b, err = os.ReadFile(p)
		}
		if err != nil {
			return err
		}
		name := strings.TrimSuffix(filepath.Base(p), ".jpg")
		cfg, err := jpeg.DecodeConfig(b, nil)
		if err != nil || cfg.Width*cfg.Height > 1<<22 {
			continue
		}
		for _, d := range []struct {
			class string
			f     jpeg.Format
			scale int
		}{{"dec-rgba", jpeg.FormatRGBA, 1}, {"dec-ycc", jpeg.FormatYCbCr, 1}, {"dec-gray", jpeg.FormatGray, 1},
			{"dec-rgba2", jpeg.FormatRGBA, 2}, {"dec-rgba4", jpeg.FormatRGBA, 4}, {"dec-rgba8", jpeg.FormatRGBA, 8}} {
			// One Decoder per class and file, reused (Rust's jpeg::Decoder): its planes and output
			// are kept between calls, and nothing is copied. The package-level jpeg.Decode would
			// pool the planes but copy a YCbCr or Gray result, which Rust does not.
			o := &jpeg.DecodeOptions{Format: d.f, Scale: d.scale, Workers: workers}
			var dec jpeg.Decoder
			if img, err := dec.Decode(bg, b, o); err == nil {
				r := img.Bounds()
				emit(d.class, name, r.Dx()*r.Dy()*4, timeOp(ms, func() { _, _ = dec.Decode(bg, b, o) }))
			}
		}
		// Into a buffer the caller keeps (Go's Dst; the Rust decode_into_rgba).
		{
			dst := image.NewRGBA(image.Rect(0, 0, cfg.Width, cfg.Height))
			var dec jpeg.Decoder
			o := &jpeg.DecodeOptions{Format: jpeg.FormatRGBA, Scale: 1, Dst: dst, Workers: workers}
			if _, err := dec.Decode(bg, b, o); err == nil {
				emit("dec-into", name, cfg.Width*cfg.Height*4, timeOp(ms, func() { _, _ = dec.Decode(bg, b, o) }))
			}
		}
		// A 512x512 region of the NDPI-style strip with its restart offsets (the Rust bench's
		// dec-region).
		if name == "strip_rst" {
			if sb, err := os.ReadFile(filepath.Join(dir, "strip_rst.starts")); err == nil {
				o := &jpeg.DecodeOptions{Format: jpeg.FormatRGBA, Scale: 1, Region: image.Rect(250, 300, 762, 812),
					Restart: parseInts(string(sb)), Workers: workers}
				var dec jpeg.Decoder
				if _, err := dec.Decode(bg, b, o); err == nil {
					emit("dec-region", name, 512*512*4, timeOp(ms, func() { _, _ = dec.Decode(bg, b, o) }))
				}
			}
		}
		emit("rowreader", name, cfg.Width*cfg.Height*4, timeOp(ms, func() {
			r, err := jpeg.NewRowReader(bg, b, nil, jpeg.Limits{})
			if err != nil {
				return
			}
			for {
				if _, err := r.Next(); err != nil {
					break
				}
			}
		}))
		emit("coef", name, cfg.Width*cfg.Height, timeOp(ms, func() {
			var d jpeg.Decoder
			_, _ = d.DecodeCoefficients(bg, b, nil)
		}))
		for _, e := range []struct {
			class string
			f     jpeg.Format
			c444  bool
		}{{"enc-rgba420", jpeg.FormatRGBA, false}, {"enc-rgba444", jpeg.FormatRGBA, true}, {"enc-gray", jpeg.FormatGray, false}, {"enc-ycc", jpeg.FormatYCbCr, false}} {
			var dec jpeg.Decoder
			img, err := dec.Decode(bg, b, &jpeg.DecodeOptions{Format: e.f})
			if err != nil {
				continue
			}
			o := &jpeg.EncodeOptions{Quality: 75, Chroma444: e.c444}
			var enc jpeg.Encoder
			var n countW
			if enc.Encode(bg, &n, img, o) != nil {
				continue
			}
			emit(e.class, name, int(n), timeOp(ms, func() {
				var c countW
				_ = enc.Encode(bg, &c, img, o)
			}))
		}
	}
	return nil
}

const genGray = "gen_smooth_gray_1536x64"

// genSmoothGray is the generated input: a 1536x64 gray ramp byte(2x+y) (a tiff-rs strip), encoded at
// quality 95 with Go's encoder. examples/bench.rs makes the same with the Rust one; both print the
// length and the md5 of the stream, which must agree.
func genSmoothGray() ([]byte, error) {
	const w, h = 1536, 64
	g := image.NewGray(image.Rect(0, 0, w, h))
	for y := range h {
		for x := range w {
			g.Pix[y*g.Stride+x] = byte(2*x + y)
		}
	}
	var buf bytes.Buffer
	if err := jpeg.Encode(bg, &buf, g, &jpeg.EncodeOptions{Quality: 95}); err != nil {
		return nil, err
	}
	fmt.Fprintf(os.Stderr, "go: %s: %d bytes, md5 %x\n", genGray, buf.Len(), md5.Sum(buf.Bytes()))
	return buf.Bytes(), nil
}

type countW int

func (c *countW) Write(p []byte) (int, error) { *c += countW(len(p)); return len(p), nil }

var _ io.Writer = (*countW)(nil)

func emit(class, name string, out int, ns int64) {
	fmt.Printf("%s/%s\t%d\t%d\n", class, name, out, ns)
}

// timeOp is the best time of one call over five batches, each about ms/5 long.
func timeOp(ms int, f func()) int64 {
	f()
	t0 := time.Now()
	f()
	d := max(1, time.Since(t0).Nanoseconds())
	k := max(1, int64(ms)*1_000_000/5/d)
	best := int64(1 << 62)
	for range 5 {
		t := time.Now()
		for range k {
			f()
		}
		if ns := time.Since(t).Nanoseconds() / k; ns < best {
			best = ns
		}
	}
	return best
}
