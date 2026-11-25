// Command oracle is the Go side of the parity check (see ../../../../docs/parity.md).
//
//	oracle -gen <testdata> <inputs> [<seeds>]   write the test inputs
//	oracle <inputs> <outputs>                   run every operation on every input
//	oracle -bench N [-workers W] <inputs>       time the decode and encode of the fixtures
//	oracle -ppm <tiles> <outputs>               decode every .jpg to <stem>.ppm
//
// Run mode writes, per input and operation, <name>.<op>.out (the output), or <name>.<op>.err (a
// short error class) and <name>.<op>.partial.out (what Go returned beside the error, if
// anything). examples/oracle.rs does the same with the Rust crate. The wire formats are in
// docs/parity.md.
package main

import (
	"bytes"
	"context"
	"errors"
	"flag"
	"fmt"
	"image"
	"io"
	"os"
	"path/filepath"
	"runtime/debug"
	"sort"
	"strconv"
	"strings"

	jpeg "github.com/dobbo-ca/jpeg-go"
)

// The oracle runs inside BuildKit: every call is bounded, so a hostile SOF cannot take the node
// down. Rust runs with the same two limits.
const (
	maxPixels = 1 << 22
	maxBytes  = 128 << 20
)

var lim = jpeg.Limits{MaxPixels: maxPixels, MaxBytes: maxBytes}

func main() {
	gen := flag.Bool("gen", false, "write the inputs: -gen <testdata> <inputs> [<seeds>]")
	bench := flag.Int("bench", 0, "time every fixture, at least N ms per operation: -bench N <inputs>")
	ppm := flag.Bool("ppm", false, "decode every .jpg to RGB .ppm: -ppm <tiles> <outputs>")
	workers := flag.Int("workers", 0, "with -bench: DecodeOptions.Workers of the dec-* classes (0 serial, -1 GOMAXPROCS)")
	flag.Parse()
	a := flag.Args()
	debug.SetMemoryLimit(1 << 30)
	jpeg.SetMemoryLimit(0) // the per-call Limits bound each call
	switch {
	case *ppm && len(a) == 2:
		check(ppmDir(a[0], a[1]))
	case *bench > 0 && len(a) == 1:
		check(benchDir(a[0], *bench, *workers))
	case *gen && (len(a) == 2 || len(a) == 3):
		seeds := ""
		if len(a) == 3 {
			seeds = a[2]
		}
		check(generate(a[0], a[1], seeds))
	case !*gen && !*ppm && *bench == 0 && len(a) == 2:
		check(runDir(a[0], a[1]))
	default:
		flag.Usage()
		os.Exit(2)
	}
}

func check(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// ---- error classes ----

// class is the stable token of an error; docs/parity.md has the table. The marker errors keep
// their message, which the Rust Display impl repeats.
func class(err error) string {
	var le *jpeg.LimitError
	msg := err.Error()
	switch {
	case errors.As(err, &le):
		return "limit-" + strings.ToLower(strings.TrimPrefix(le.Name, "Max"))
	case errors.Is(err, jpeg.ErrAfterScan):
		c := "afterscan:"
		if cause := afterCause(err); cause != nil {
			c += class(cause)
		}
		if errors.Is(err, jpeg.ErrTruncated) {
			c += "+truncated"
		}
		return c
	case errors.Is(err, jpeg.ErrTruncated):
		return "truncated"
	case errors.Is(err, jpeg.ErrUnsupported):
		return "unsupported"
	case errors.Is(err, context.Canceled):
		return "cancelled"
	case errors.Is(err, io.ErrUnexpectedEOF):
		return "unexpected-eof"
	case strings.HasPrefix(msg, "jpeg: region") && strings.Contains(msg, "does not intersect"):
		return "region-empty"
	case strings.HasPrefix(msg, "jpeg: region"):
		return "region"
	case strings.Contains(msg, "over the 1 GiB cap"):
		return "cap"
	case strings.HasSuffix(msg, "RowReader is closed"):
		return "closed"
	case strings.HasPrefix(msg, "jpeg: "):
		return msg
	}
	return "other:" + msg
}

// afterCause is the marker error inside an ErrAfterScan error.
func afterCause(err error) error {
	if m, ok := err.(interface{ Unwrap() []error }); ok {
		es := m.Unwrap()
		if len(es) == 2 && es[0] == jpeg.ErrAfterScan {
			return es[1]
		}
		for _, e := range es {
			if c := afterCause(e); c != nil {
				return c
			}
		}
		return nil
	}
	if u := errors.Unwrap(err); u != nil {
		return afterCause(u)
	}
	return nil
}

// ---- output ----

// rawMax is the largest output written as bytes; a bigger one is a digest of the whole and of
// every row, so a difference shows which row.
const rawMax = 8 << 10

func fnv64(h uint64, b []byte) uint64 {
	for _, c := range b {
		h = (h ^ uint64(c)) * 0x100000001b3
	}
	return h
}

const fnvBasis = 0xcbf29ce484222325

// encodeRows is hdr then the rows, or, over rawMax bytes, hdr, "fnv <hash> <len>" and "r <hash>"
// per row.
func encodeRows(hdr string, rows [][]byte) []byte {
	n := 0
	for _, r := range rows {
		n += len(r)
	}
	var out bytes.Buffer
	out.WriteString(hdr)
	if n <= rawMax {
		for _, r := range rows {
			out.Write(r)
		}
		return out.Bytes()
	}
	h := uint64(fnvBasis)
	for _, r := range rows {
		h = fnv64(h, r)
	}
	fmt.Fprintf(&out, "fnv %016x %d\n", h, n)
	for _, r := range rows {
		fmt.Fprintf(&out, "r %016x\n", fnv64(fnvBasis, r))
	}
	return out.Bytes()
}

// chunks cuts b into rows of 1 KiB, for the bytes of a stream.
func chunks(b []byte) [][]byte {
	var rows [][]byte
	for len(b) > 1024 {
		rows, b = append(rows, b[:1024]), b[1024:]
	}
	return append(rows, b)
}

var ratioNames = [...]string{"444", "422", "420", "440", "411", "410"}

func ratioName(r image.YCbCrSubsampleRatio) string {
	if int(r) < len(ratioNames) {
		return ratioNames[r]
	}
	return "?"
}

// serialize is the wire form of an image: a header line with the type, the rect and for YCbCr the
// ratio, then the visible samples row by row (Y, Cb, Cr), without stride or block padding.
func serialize(img image.Image) []byte {
	var hdr string
	var rows [][]byte
	switch m := img.(type) {
	case *image.RGBA:
		r := m.Rect
		hdr = fmt.Sprintf("rgba %d %d %d %d\n", r.Min.X, r.Min.Y, r.Max.X, r.Max.Y)
		if !r.Empty() {
			for y := r.Min.Y; y < r.Max.Y; y++ {
				o := m.PixOffset(r.Min.X, y)
				rows = append(rows, m.Pix[o:o+4*r.Dx()])
			}
		}
	case *image.CMYK:
		r := m.Rect
		hdr = fmt.Sprintf("cmyk %d %d %d %d\n", r.Min.X, r.Min.Y, r.Max.X, r.Max.Y)
		if !r.Empty() {
			for y := r.Min.Y; y < r.Max.Y; y++ {
				o := m.PixOffset(r.Min.X, y)
				rows = append(rows, m.Pix[o:o+4*r.Dx()])
			}
		}
	case *image.Gray:
		r := m.Rect
		hdr = fmt.Sprintf("gray %d %d %d %d\n", r.Min.X, r.Min.Y, r.Max.X, r.Max.Y)
		if !r.Empty() {
			for y := r.Min.Y; y < r.Max.Y; y++ {
				o := m.PixOffset(r.Min.X, y)
				rows = append(rows, m.Pix[o:o+r.Dx()])
			}
		}
	case *image.YCbCr:
		r := m.Rect
		hdr = fmt.Sprintf("ycbcr %s %d %d %d %d\n", ratioName(m.SubsampleRatio), r.Min.X, r.Min.Y, r.Max.X, r.Max.Y)
		if !r.Empty() {
			for y := r.Min.Y; y < r.Max.Y; y++ {
				o := m.YOffset(r.Min.X, y)
				rows = append(rows, m.Y[o:o+r.Dx()])
			}
			cw := m.COffset(r.Max.X-1, r.Min.Y) - m.COffset(r.Min.X, r.Min.Y) + 1
			var offs []int
			last := -1
			for y := r.Min.Y; y < r.Max.Y; y++ {
				if o := m.COffset(r.Min.X, y); o != last {
					offs = append(offs, o)
					last = o
				}
			}
			for _, p := range [][]byte{m.Cb, m.Cr} {
				for _, o := range offs {
					rows = append(rows, p[o:o+cw])
				}
			}
		}
	default:
		hdr = fmt.Sprintf("other %T\n", img)
	}
	return encodeRows(hdr, rows)
}

// out writes the files of one operation.
type out struct{ dir, name string }

func (o out) file(op, ext string, b []byte) {
	check(os.WriteFile(filepath.Join(o.dir, o.name+"."+op+ext), b, 0o644))
}

// ok writes <op>.out.
func (o out) ok(op string, b []byte) { o.file(op, ".out", b) }

// fail writes <op>.err and, when there is one, <op>.partial.out.
func (o out) fail(op string, err error, partial []byte) {
	o.file(op, ".err", []byte(class(err)+"\n"))
	if partial != nil {
		o.file(op, ".partial.out", partial)
	}
}

// image writes the result of a decode: the image, or the error with the image Go returned beside it.
func (o out) image(op string, img image.Image, err error) {
	switch {
	case err == nil:
		o.ok(op, serialize(img))
	case img != nil:
		o.fail(op, err, serialize(img))
	default:
		o.fail(op, err, nil)
	}
}

// safe runs f; a panic is the result "panic", which Rust never produces, so it shows as a difference.
func (o out) safe(op string, f func()) {
	defer func() {
		if r := recover(); r != nil {
			o.file(op, ".err", []byte(fmt.Sprintf("panic:%v\n", r)))
		}
	}()
	f()
}

// ---- operations ----

var bg = context.Background()

// cancelled is a context that is already done: Rust's flag set before the call.
var cancelled = func() context.Context {
	ctx, cancel := context.WithCancel(bg)
	cancel()
	return ctx
}()

func runDir(in, outDir string) error {
	ents, err := os.ReadDir(in)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return err
	}
	var names []string
	for _, e := range ents {
		names = append(names, e.Name())
	}
	sort.Strings(names)
	dirtyFrame, _ = os.ReadFile(filepath.Join(in, "synth_444_227x149.jpg"))
	for _, n := range names {
		b, err := os.ReadFile(filepath.Join(in, n))
		if err != nil {
			return err
		}
		o := out{outDir, n}
		switch filepath.Ext(n) {
		case ".jpg":
			runJPEG(in, o, n, b)
		case ".tbl":
			runTables(in, o, b)
		case ".img":
			runImg(o, b)
		case ".fancy":
			runFancy(o, b)
		}
	}
	capOps(outDir)
	return nil
}

// capStream is a w x h grayscale baseline stream with the standard tables not sent and no entropy
// data: a decode reads its headers, sizes itself from them and then finds the data truncated.
func capStream(w, h int) []byte {
	b := []byte{0xff, 0xd8, 0xff, 0xdb, 0, 67, 0}
	for range 64 {
		b = append(b, 1)
	}
	b = append(b, 0xff, 0xc0, 0, 11, 8, byte(h>>8), byte(h), byte(w>>8), byte(w), 1, 1, 0x11, 0)
	return append(b, 0xff, 0xda, 0, 8, 1, 1, 0, 0, 63, 0)
}

// capOps is the op capd on three synthetic frames at Go's 1 GiB cap, under zero Limits (Rust:
// Limits::default()): the first needs 1 GiB less 16 KiB of page rounding, so it decodes (an
// all-zero plane, then ErrTruncated); the others are over the cap and are refused from the
// headers. The image is written as its rect only, not its 1 GiB of samples.
func capOps(outDir string) {
	for _, c := range [][2]int{{16776, 64000}, {16777, 64000}, {65500, 65500}} {
		o := out{outDir, fmt.Sprintf("cap-%dx%d.synth", c[0], c[1])}
		o.safe("capd", func() {
			var d jpeg.Decoder
			img, err := d.Decode(bg, capStream(c[0], c[1]), &jpeg.DecodeOptions{Format: jpeg.FormatGray, Scale: 1})
			var b []byte
			if img != nil {
				r := img.Bounds()
				b = []byte(fmt.Sprintf("gray %d %d %d %d\n", r.Min.X, r.Min.Y, r.Max.X, r.Max.Y))
			}
			if err != nil {
				o.fail("capd", err, b)
				return
			}
			o.ok("capd", b)
		})
		debug.FreeOSMemory()
	}
}

type dop struct {
	name  string
	f     jpeg.Format
	scale int
}

func dops(light bool) []dop {
	var l []dop
	for _, s := range []int{1, 2, 4, 8} {
		for _, f := range []struct {
			n string
			f jpeg.Format
		}{{"yc", jpeg.FormatYCbCr}, {"rg", jpeg.FormatRGBA}, {"gr", jpeg.FormatGray}} {
			if light && !(s == 1 || (s == 4 && f.n == "rg")) {
				continue
			}
			l = append(l, dop{f.n + strconv.Itoa(s), f.f, s})
		}
	}
	return l
}

// decodeWith is Decoder.Decode on a new Decoder: the package-level jpeg.Decode copies the image
// and its Limits count the copy, which Rust, returning the image itself, has not.
func decodeWith(ctx context.Context, src []byte, o jpeg.DecodeOptions) (image.Image, error) {
	var d jpeg.Decoder
	return d.Decode(ctx, src, &o)
}

func decodeOpts(src []byte, o jpeg.DecodeOptions) (image.Image, error) {
	o.Limits = lim
	return decodeWith(bg, src, o)
}

func cfgLine(c jpeg.Config) []byte {
	return []byte(fmt.Sprintf("%d %d %d %s %t %d\n", c.Width, c.Height, c.Components, ratioName(c.Ratio), c.RGB, c.RestartInterval))
}

func configOps(o out, src, tables []byte, pfx string) {
	o.safe(pfx+"cfg", func() {
		c, err := jpeg.DecodeConfigLimits(src, tables, lim)
		if err != nil {
			o.fail(pfx+"cfg", err, nil)
			return
		}
		o.ok(pfx+"cfg", cfgLine(c))
	})
}

func coefOp(o out, ctx context.Context, op string, src, tables []byte, l jpeg.Limits) {
	o.safe(op, func() {
		var d jpeg.Decoder
		c, err := d.DecodeCoefficientsLimits(ctx, src, tables, l, false)
		if c == nil {
			o.fail(op, err, nil)
			return
		}
		hdr := "coef " + string(cfgLine(c.Config))
		var rows [][]byte
		for _, cc := range c.Comp {
			hdr += fmt.Sprintf("c %d %d %d %d\n", cc.H, cc.V, cc.BlocksWide, cc.BlocksHigh)
			q := make([]byte, 0, 128)
			for _, v := range cc.Quant {
				q = append(q, byte(v), byte(v>>8))
			}
			rows = append(rows, q)
			for by := 0; by < cc.BlocksHigh; by++ {
				row := make([]byte, 0, cc.BlocksWide*128)
				for bx := 0; bx < cc.BlocksWide; bx++ {
					for _, v := range cc.Blocks[by*cc.BlocksWide+bx] {
						row = append(row, byte(v), byte(v>>8))
					}
				}
				rows = append(rows, row)
			}
		}
		b := encodeRows(hdr, rows)
		if err != nil {
			o.fail(op, err, b)
			return
		}
		o.ok(op, b)
	})
}

func rrOp(o out, ctx context.Context, op string, src, tables []byte, l jpeg.Limits) {
	o.safe(op, func() {
		r, err := jpeg.NewRowReader(ctx, src, tables, l)
		if err != nil {
			o.fail(op, err, nil)
			return
		}
		c := r.Config()
		hdr := fmt.Sprintf("rr %d %d %t\n", c.Width, c.Height, r.Truncated())
		var rows [][]byte
		for {
			img, err := r.Next()
			if err == io.EOF {
				break
			}
			if err != nil {
				o.fail(op, err, encodeRows(hdr, rows))
				return
			}
			for y := img.Rect.Min.Y; y < img.Rect.Max.Y; y++ {
				p := img.PixOffset(img.Rect.Min.X, y)
				rows = append(rows, bytes.Clone(img.Pix[p:p+4*img.Rect.Dx()]))
			}
		}
		o.ok(op, encodeRows(hdr, rows))
	})
}

// sharedDec and sharedEnc are one Decoder and one Encoder used for every input, in the sorted order
// of the input directory (sizes, formats, regions, truncated streams, uniform images in turn): the
// ops sd1, sy1, sr and se<q> run through them, and Rust runs the same ops through its own reused
// Decoder and Encoder, so state a reused object keeps from an earlier input shows as a difference.
// Each reused decode follows dirtyFrame, a fixed decode that leaves data in the Decoder's buffers
// (the Rust oracle does the same).
var (
	sharedDec  jpeg.Decoder
	sharedEnc  jpeg.Encoder
	dirtyFrame []byte
)

// sharedDecode is sharedDec.Decode of src after the dirty frame.
func sharedDecode(src []byte, opt *jpeg.DecodeOptions) (image.Image, error) {
	if dirtyFrame != nil {
		_, _ = sharedDec.Decode(bg, dirtyFrame, &jpeg.DecodeOptions{Format: jpeg.FormatRGBA, Scale: 1, Limits: lim})
	}
	return sharedDec.Decode(bg, src, opt)
}

// dstOp is Decoder.Decode with Dst: a w x h RGBA window of a buffer filled with 0xAA. pad 0 is the
// tight buffer; otherwise the stride is 4w+pad and the window has a row above it and 8 bytes to its
// left. It writes the whole buffer, so a write outside the window shows, and the error beside it.
func dstOp(o out, op string, src, tables []byte, cfg jpeg.Config, pad int) {
	o.safe(op, func() {
		w, h := cfg.Width, cfg.Height
		stride := 4*w + pad
		buf := bytes.Repeat([]byte{0xaa}, (h+2)*stride)
		pix := buf[stride+8:]
		if pad == 0 {
			buf = buf[:h*stride]
			pix = buf
		}
		dst := &image.RGBA{Pix: pix, Stride: stride, Rect: image.Rect(0, 0, w, h)}
		var d jpeg.Decoder
		_, err := d.Decode(bg, src, &jpeg.DecodeOptions{Format: jpeg.FormatRGBA, Scale: 1, Tables: tables, Limits: lim, Dst: dst})
		b := encodeRows(fmt.Sprintf("dst %d %d %d\n", w, h, pad), chunks(buf))
		if err != nil {
			o.fail(op, err, b)
			return
		}
		o.ok(op, b)
	})
}

// rrxOp is the RowReader's life after its last row and after Close: what Next returns then.
func rrxOp(o out, src, tables []byte) {
	o.safe("rrx", func() {
		var sb strings.Builder
		next := func(r *jpeg.RowReader) string {
			_, err := r.Next()
			switch {
			case err == io.EOF:
				return "eof"
			case err != nil:
				return class(err)
			}
			return "rows"
		}
		r, err := jpeg.NewRowReader(bg, src, tables, lim)
		if err != nil {
			o.fail("rrx", err, nil)
			return
		}
		n := 0
		for {
			s := next(r)
			if s != "rows" {
				fmt.Fprintf(&sb, "end %d %s\n", n, s)
				break
			}
			n++
		}
		fmt.Fprintf(&sb, "after end %s\n", next(r))
		_ = r.Close()
		_ = r.Close()
		fmt.Fprintf(&sb, "after close %s\n", next(r))
		r, err = jpeg.NewRowReader(bg, src, tables, lim)
		if err != nil {
			fmt.Fprintf(&sb, "second new %s\n", class(err))
		} else {
			fmt.Fprintf(&sb, "first %s\n", next(r))
			_ = r.Close()
			fmt.Fprintf(&sb, "after early close %s\n", next(r))
		}
		o.ok("rrx", []byte(sb.String()))
	})
}

func runJPEG(dir string, o out, name string, src []byte) {
	light := strings.Contains(name, "~")
	base := strings.TrimSuffix(name, ".jpg")
	var tables []byte
	if strings.HasPrefix(name, "ab-") {
		tables, _ = os.ReadFile(filepath.Join(dir, "tables.bin"))
	} else if b, err := os.ReadFile(filepath.Join(dir, base+".tables")); err == nil {
		tables = b // a tables stream put beside the stream (TestTablesStateReset)
	}
	cfg, cfgErr := jpeg.DecodeConfigLimits(src, tables, lim)

	configOps(o, src, tables, "")
	for _, d := range dops(light) {
		o.safe(d.name, func() {
			img, err := decodeOpts(src, jpeg.DecodeOptions{Format: d.f, Scale: d.scale, Tables: tables})
			o.image(d.name, img, err)
		})
	}
	coefOp(o, bg, "coef", src, tables, lim)
	rrOp(o, bg, "rr", src, tables, lim)
	rrxOp(o, src, tables)
	for _, d := range []dop{{"sd1", jpeg.FormatRGBA, 1}, {"sy1", jpeg.FormatYCbCr, 1}} {
		o.safe(d.name, func() {
			opt := jpeg.DecodeOptions{Format: d.f, Scale: d.scale, Tables: tables, Limits: lim}
			img, err := sharedDecode(src, &opt)
			o.image(d.name, img, err)
		})
	}
	if cfgErr == nil {
		// A region that leaves the image: the reused Decoder's padded copy.
		o.safe("sr", func() {
			opt := jpeg.DecodeOptions{Format: jpeg.FormatRGBA, Scale: 1, Tables: tables, Limits: lim,
				Region: image.Rect(-5, -3, cfg.Width/2+1, cfg.Height/2+1)}
			img, err := sharedDecode(src, &opt)
			o.image("sr", img, err)
		})
		dstOp(o, "dst", src, tables, cfg, 16)
		dstOp(o, "dst0", src, tables, cfg, 0)
	}
	o.safe("rrn", func() {
		n, err := jpeg.RowReaderNeed(src, tables)
		if err != nil {
			o.fail("rrn", err, nil)
			return
		}
		o.ok("rrn", []byte(strconv.FormatInt(n, 10)+"\n"))
	})

	if cfgErr == nil {
		w, h := int64(cfg.Width), int64(cfg.Height)
		o.safe("lb2", func() {
			l := jpeg.Limits{MaxPixels: maxPixels, MaxBytes: w * h * 3}
			img, err := decodeWith(bg, src, jpeg.DecodeOptions{Format: jpeg.FormatRGBA, Tables: tables, Limits: l})
			o.image("lb2", img, err)
		})
		if !light {
			// MaxPixels 0 is "no limit" in Go: a 1-pixel frame has no Rust twin of this case.
			if w*h > 1 {
				o.safe("lpx", func() {
					l := jpeg.Limits{MaxPixels: w*h - 1, MaxBytes: maxBytes}
					img, err := decodeWith(bg, src, jpeg.DecodeOptions{Format: jpeg.FormatRGBA, Tables: tables, Limits: l})
					o.image("lpx", img, err)
				})
			}
			for i, mb := range []int64{1 << 14, 1 << 17, w*h*6 + 1<<16} {
				op := "lb" + strconv.Itoa(i)
				o.safe(op, func() {
					l := jpeg.Limits{MaxPixels: maxPixels, MaxBytes: mb}
					img, err := decodeWith(bg, src, jpeg.DecodeOptions{Format: jpeg.FormatRGBA, Tables: tables, Limits: l})
					o.image(op, img, err)
				})
			}
			rrOp(o, bg, "rrl", src, tables, jpeg.Limits{MaxPixels: maxPixels, MaxBytes: w * h * 3})
			coefOp(o, bg, "coefl", src, tables, jpeg.Limits{MaxPixels: maxPixels, MaxBytes: w * h})
			o.safe("cx", func() {
				img, err := decodeWith(cancelled, src, jpeg.DecodeOptions{Format: jpeg.FormatRGBA, Tables: tables, Limits: lim})
				o.image("cx", img, err)
			})
			// The cancel flag set before the call, on the coefficients and the row reader.
			coefOp(o, cancelled, "coefx", src, tables, lim)
			rrOp(o, cancelled, "rrcx", src, tables, lim)
		}
	}
	if tables != nil {
		for _, d := range []dop{{"tb-yc", jpeg.FormatYCbCr, 1}, {"tb-rg", jpeg.FormatRGBA, 1}} {
			o.safe(d.name, func() {
				img, err := decodeOpts(src, jpeg.DecodeOptions{Format: d.f, Scale: d.scale, Tables: tables})
				o.image(d.name, img, err)
			})
		}
	}

	// A sidecar <base>.rect (damaged streams have one only for strip_rst and the dri- streams): one region decode per line "scale x0 y0 x1 y1 format check".
	if b, err := os.ReadFile(filepath.Join(dir, base+".rect")); err == nil {
		for i, line := range strings.Split(strings.TrimSpace(string(b)), "\n") {
			var s, x0, y0, x1, y1, f, c int
			if _, err := fmt.Sscan(line, &s, &x0, &y0, &x1, &y1, &f, &c); err != nil {
				continue
			}
			op := "r" + strconv.Itoa(i)
			o.safe(op, func() {
				img, err := decodeOpts(src, jpeg.DecodeOptions{Format: jpeg.Format(f), Scale: s, Tables: tables,
					Region: image.Rect(x0, y0, x1, y1), CheckFrame: c != 0})
				o.image(op, img, err)
			})
		}
	}
	// A sidecar <base>.starts: restart offsets (NDPI tag 65426), one per line.
	if b, err := os.ReadFile(filepath.Join(dir, base+".starts")); err == nil {
		var starts []int
		for _, f := range strings.Fields(string(b)) {
			if v, err := strconv.Atoi(f); err == nil {
				starts = append(starts, v)
			}
		}
		for _, d := range []dop{{"ndpi-rg", jpeg.FormatRGBA, 1}, {"ndpi-gr2", jpeg.FormatGray, 2}} {
			o.safe(d.name, func() {
				img, err := decodeOpts(src, jpeg.DecodeOptions{Format: d.f, Scale: d.scale, Restart: starts})
				o.image(d.name, img, err)
			})
		}
		if cfgErr == nil {
			o.safe("ndpi-reg", func() {
				r := image.Rect(cfg.Width/3, cfg.Height/3, cfg.Width*2/3+1, cfg.Height*2/3+1)
				img, err := decodeOpts(src, jpeg.DecodeOptions{Format: jpeg.FormatRGBA, Scale: 1, Restart: starts, Region: r, CheckFrame: true})
				o.image("ndpi-reg", img, err)
			})
			// The same offsets with CheckFrame off: the serial offset-landing jump and the region bands.
			for _, d := range []dop{{"ndpi-reg2", jpeg.FormatRGBA, 1}, {"ndpi-reg2g", jpeg.FormatGray, 1}} {
				o.safe(d.name, func() {
					r := image.Rect(cfg.Width/4, cfg.Height/4, cfg.Width*3/4+1, cfg.Height*3/4+1)
					img, err := decodeOpts(src, jpeg.DecodeOptions{Format: d.f, Scale: d.scale, Restart: starts, Region: r})
					o.image(d.name, img, err)
				})
			}
		}
	}
}

// runTables decodes the fixed abbreviated tile ab-image.jpg against a tables stream.
func runTables(dir string, o out, tables []byte) {
	img, err := os.ReadFile(filepath.Join(dir, "ab-image.jpg"))
	if err != nil {
		return
	}
	configOps(o, img, tables, "tb-")
	for _, d := range []dop{{"tb-yc", jpeg.FormatYCbCr, 1}, {"tb-rg", jpeg.FormatRGBA, 1}} {
		o.safe(d.name, func() {
			m, err := decodeOpts(img, jpeg.DecodeOptions{Format: d.f, Scale: d.scale, Tables: tables})
			o.image(d.name, m, err)
		})
	}
	coefOp(o, bg, "tb-coef", img, tables, lim)
}
