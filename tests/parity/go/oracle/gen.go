package main

import (
	"bytes"
	"fmt"
	"image"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strconv"
	"strings"

	jpeg "github.com/dobbo-ca/jpeg-go"
)

// generate writes the parity inputs: the fixtures, deterministic mutations of each, the inputs the
// Go tests build with math/rand (see docs/parity.md for the table), and the decoded fixtures as
// encoder inputs. Go supplies the expected output for all of them.
func generate(td, dir, seeds string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	g := &gen{dir: dir, seen: map[string]bool{}}
	fxs, err := loadFixtures(td)
	if err != nil {
		return err
	}
	for _, f := range fxs {
		g.put(f.name, f.b)
	}
	for _, n := range []string{"tables.bin"} {
		if b, err := os.ReadFile(filepath.Join(td, "abbrev", n)); err == nil {
			g.put(n, b)
		}
	}
	for _, n := range []string{"strip_rst.starts", "ndpi_wide_4096x32.starts"} {
		if b, err := os.ReadFile(filepath.Join(td, n)); err == nil {
			g.put(n, b)
		}
	}
	g.mutated(fxs, td)
	g.rects(fxs)
	g.tablesMutations(td)
	g.encoderInputs(td, fxs)
	g.synthDRIs()
	g.markerStreams(fxs)
	g.coefStreams(fxs)
	g.randHuffStreams()
	g.kernels()
	if seeds != "" {
		if err := g.seedFiles(seeds); err != nil {
			return err
		}
	}
	fmt.Println(g.n, "inputs")
	return nil
}

type gen struct {
	dir  string
	n    int
	seen map[string]bool
}

func (g *gen) put(name string, b []byte) {
	if g.seen[name] {
		log.Fatalf("generated input %q twice: two inputs would share one file", name)
	}
	g.seen[name] = true
	check(os.WriteFile(filepath.Join(g.dir, name), b, 0o644))
	g.n++
}

type fx struct {
	name string
	b    []byte
}

func loadFixtures(td string) ([]fx, error) {
	var l []fx
	for _, d := range []string{"jpg", "subsample", "truncated"} {
		m, err := filepath.Glob(filepath.Join(td, d, "*.jpg"))
		if err != nil {
			return nil, err
		}
		sort.Strings(m)
		for _, p := range m {
			b, err := os.ReadFile(p)
			if err != nil {
				return nil, err
			}
			l = append(l, fx{filepath.Base(p), b})
		}
	}
	if len(l) < 100 {
		return nil, fmt.Errorf("%d fixtures: the testdata (git LFS) is missing", len(l))
	}
	b, err := os.ReadFile(filepath.Join(td, "abbrev", "image.jpg"))
	if err != nil {
		return nil, err
	}
	return append(l, fx{"ab-image.jpg", b}), nil
}

// ---- mutations ----

type seg struct {
	m      byte
	off, n int
}

// segments are the marker segments up to the end of the last SOS header, and the offset of the
// first entropy byte.
func segments(b []byte) (segs []seg, ent0 int) {
	i := 2
	for i+4 <= len(b) && b[i] == 0xff {
		m := b[i+1]
		if m == 0xd8 || m == 0x01 || (m >= 0xd0 && m <= 0xd7) {
			i += 2
			continue
		}
		n := int(b[i+2])<<8 | int(b[i+3])
		segs = append(segs, seg{m, i, n})
		i += 2 + n
		if m == 0xda {
			if ent0 == 0 {
				ent0 = i
			}
			for i+1 < len(b) && !(b[i] == 0xff && b[i+1] != 0 && b[i+1] != 0xff && !(b[i+1] >= 0xd0 && b[i+1] <= 0xd7)) {
				i++
			}
		}
	}
	if ent0 == 0 {
		ent0 = min(len(b), 20)
	}
	return segs, ent0
}

type mut struct {
	tag string
	b   []byte
}

func setBytes(b []byte, at int, v ...byte) []byte {
	if at < 0 || at+len(v) > len(b) {
		return nil
	}
	c := bytes.Clone(b)
	copy(c[at:], v)
	return c
}

func insertAt(b []byte, at int, v ...byte) []byte {
	if at < 0 || at > len(b) {
		return nil
	}
	c := make([]byte, 0, len(b)+len(v))
	c = append(c, b[:at]...)
	c = append(c, v...)
	return append(c, b[at:]...)
}

// mutations returns n deterministic damaged copies of b, spread over the kinds: cuts, entropy
// flips, header field edits, marker insertions, segment length edits, appended bytes, restart
// interval and RSTn edits, and the sampling relabels.
func mutations(b []byte, rng *rand.Rand, n int) []mut {
	segs, ent0 := segments(b)
	var cuts, flips, fields, inserts, lengths, appends, dris, rsts, hvs []mut
	seen := map[string][]byte{}
	add := func(dst *[]mut, tag string, c []byte) {
		if c == nil {
			return
		}
		// A repeated tag is the same cut or edit again (skipped) or a random edit at the same
		// offset (kept, with a counter), so every mutation has its own file name.
		if old, ok := seen[tag]; ok {
			if bytes.Equal(old, c) {
				return
			}
			for i := 2; ; i++ {
				t := fmt.Sprintf("%s_%d", tag, i)
				if _, dup := seen[t]; !dup {
					tag = t
					break
				}
			}
		}
		seen[tag] = c
		*dst = append(*dst, mut{tag, c})
	}
	L := len(b)
	for _, at := range []int{2, 3, 4, 6, ent0 - 1, ent0, ent0 + 1, ent0 + 2, L - 2, L - 1} {
		if at > 0 && at < L {
			add(&cuts, fmt.Sprintf("t%d", at), bytes.Clone(b[:at]))
		}
	}
	for i := 1; i < 24; i++ {
		at := L * i / 24
		add(&cuts, fmt.Sprintf("t%d", at), bytes.Clone(b[:at]))
	}
	pickEnt := func() int {
		if L-ent0 < 2 {
			return rng.Intn(L)
		}
		return ent0 + rng.Intn(L-ent0)
	}
	for range 24 {
		at := pickEnt()
		add(&flips, fmt.Sprintf("f%d", at), setBytes(b, at, b[at]^byte(1<<rng.Intn(8))))
		v := byte(rng.Intn(256))
		add(&flips, fmt.Sprintf("v%d", at), setBytes(b, at, v))
	}
	for range 6 {
		at := pickEnt()
		c := bytes.Clone(b)
		clear(c[at:min(L, at+1+rng.Intn(32))])
		add(&flips, fmt.Sprintf("z%d", at), c)
	}
	for range 12 {
		at := rng.Intn(L)
		add(&flips, fmt.Sprintf("h%d", at), setBytes(b, at, b[at]^byte(1<<rng.Intn(8))))
	}
	markers := []byte{0x00, 0x01, 0x80, 0x81, 0xc0, 0xc2, 0xc4, 0xcc, 0xd0, 0xd3, 0xd7, 0xd8, 0xd9, 0xda, 0xdb, 0xdd, 0xe0, 0xfe, 0xff}
	for range 20 {
		at, m := pickEnt(), markers[rng.Intn(len(markers))]
		add(&inserts, fmt.Sprintf("i%d-%02x", at, m), insertAt(b, at, 0xff, m))
	}
	sofVals := map[int][]byte{ // offset from the marker: values to try
		4:  {0, 4, 12, 16},
		9:  {0, 1, 2, 4, 5, 255},
		10: {0, 1, 2, 3, 4, 82, 71, 66, 255},
		11: {0x00, 0x10, 0x01, 0x11, 0x12, 0x21, 0x22, 0x14, 0x41, 0x42, 0x24, 0x44, 0x33, 0xff},
		12: {0, 1, 2, 3, 4, 5, 255},
		13: {0, 1, 2, 3, 4, 82, 71, 66, 255},
		14: {0x00, 0x11, 0x12, 0x21, 0x22, 0x41, 0xff},
		16: {0, 1, 2, 3, 4, 82, 71, 66, 255},
		17: {0x00, 0x11, 0x12, 0x21, 0x22, 0x41, 0xff},
	}
	dims := [][2]int{{0, 0}, {1, 1}, {1, 0}, {0, 1}, {8, 8}, {9, 9}, {255, 255}, {256, 256}, {4096, 4096}, {65500, 1}, {1, 65500}, {65501, 1}, {65535, 65535}}
	for _, s := range segs {
		switch {
		case s.m >= 0xc0 && s.m <= 0xcf && s.m != 0xc4 && s.m != 0xc8 && s.m != 0xcc:
			offs := make([]int, 0, len(sofVals))
			for o := range sofVals {
				offs = append(offs, o)
			}
			sort.Ints(offs)
			for _, o := range offs {
				for _, v := range sofVals[o] {
					add(&fields, fmt.Sprintf("s%d=%02x", s.off+o, v), setBytes(b, s.off+o, v))
				}
			}
			for _, d := range dims {
				add(&fields, fmt.Sprintf("d%dx%d", d[0], d[1]), setBytes(b, s.off+5, byte(d[1]>>8), byte(d[1]), byte(d[0]>>8), byte(d[0])))
			}
			for _, m := range []byte{0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcc, 0xcd, 0xce, 0xcf} {
				add(&fields, fmt.Sprintf("m%02x", m), setBytes(b, s.off+1, m))
			}
		case s.m == 0xda:
			for _, f := range []struct {
				o    int
				vals []byte
			}{{4, []byte{0, 1, 2, 3, 4, 255}}, {5, []byte{0, 1, 2, 3, 4, 82, 255}}, {6, []byte{0x00, 0x11, 0x22, 0x33, 0x01, 0x10, 0xff}},
				{7, []byte{0, 1, 2, 3, 82, 255}}, {8, []byte{0x00, 0x11, 0x22, 0xff}}, {9, []byte{0, 1, 2, 5, 63, 64, 255}},
				{10, []byte{0, 1, 5, 63, 64, 70, 255}}, {11, []byte{0x00, 0x01, 0x10, 0x11, 0x21, 0x22, 0x77, 0xff}},
				{12, []byte{0, 1, 63, 255}}, {13, []byte{0, 1, 63, 255}}} {
				for _, v := range f.vals {
					add(&fields, fmt.Sprintf("o%d=%02x", s.off+f.o, v), setBytes(b, s.off+f.o, v))
				}
			}
		case s.m == 0xdb:
			for _, v := range []byte{0x00, 0x10, 0x01, 0x03, 0x04, 0x14, 0xff} {
				add(&fields, fmt.Sprintf("q%d=%02x", s.off+4, v), setBytes(b, s.off+4, v))
			}
			for _, v := range []byte{0, 1, 255} {
				add(&fields, fmt.Sprintf("q%d=%02x", s.off+5, v), setBytes(b, s.off+5, v))
				add(&fields, fmt.Sprintf("q%d=%02x", s.off+20, v), setBytes(b, s.off+20, v))
			}
		case s.m == 0xc4:
			for _, v := range []byte{0x00, 0x01, 0x03, 0x04, 0x10, 0x11, 0x14, 0x20, 0xff} {
				add(&fields, fmt.Sprintf("c%d=%02x", s.off+4, v), setBytes(b, s.off+4, v))
			}
			for k := range 16 {
				for _, v := range []byte{0, 255} {
					add(&fields, fmt.Sprintf("c%d=%02x", s.off+5+k, v), setBytes(b, s.off+5+k, v))
				}
			}
		case s.m == 0xee:
			for _, v := range []byte{0, 1, 2, 3, 255} {
				add(&fields, fmt.Sprintf("a%d=%02x", s.off+15, v), setBytes(b, s.off+15, v))
			}
		}
		if s.m != 0xda {
			for _, d := range []int{-1, 1, 2, -2, -s.n, 0xffff - s.n} {
				if v := s.n + d; v >= 0 && v <= 0xffff {
					add(&lengths, fmt.Sprintf("l%d%+d", s.off, d), setBytes(b, s.off+2, byte(v>>8), byte(v)))
				}
			}
		}
	}
	for i, a := range [][]byte{{0xff, 0xd9}, {0xff, 0xd9, 0xff, 0xd9}, {0, 0, 0, 0, 0, 0, 0, 0}, {0xff, 0xd8}, {0xff, 0xda, 0, 8, 1, 1, 0, 0, 0x3f, 0},
		{0xff, 0xe0, 0, 2}, {0xff, 0xff, 0xff}, {0xff, 0xc0, 0, 17, 8, 0, 8, 0, 8, 3, 1, 0x11, 0, 2, 0x11, 1, 3, 0x11, 1}, {0xff, 0xc4, 0, 3}, {0xff, 0xdb, 0, 2}, {0xff, 0xdd, 0, 4, 0, 1}} {
		add(&appends, fmt.Sprintf("a%d", i), append(bytes.Clone(b), a...))
	}
	junk := make([]byte, 40)
	rng.Read(junk)
	add(&appends, "ar", append(bytes.Clone(b), junk...))
	add(&appends, "eoi0", append(bytes.Clone(b[:max(0, L-2)]), 0xff, 0x00))
	if len(segs) > 0 {
		sos := -1
		for _, s := range segs {
			if s.m == 0xda {
				sos = s.off
				break
			}
		}
		for _, ri := range []int{0, 1, 2, 3, 5, 7, 16, 100, 1000, 65535} {
			if sos > 0 {
				add(&dris, fmt.Sprintf("r%d", ri), insertAt(b, sos, 0xff, 0xdd, 0, 4, byte(ri>>8), byte(ri)))
			}
		}
	}
	var rstAt []int
	for i := ent0; i+1 < L; i++ {
		if b[i] == 0xff && b[i+1] >= 0xd0 && b[i+1] <= 0xd7 {
			rstAt = append(rstAt, i)
		}
	}
	for range 10 {
		if len(rstAt) == 0 {
			break
		}
		at := rstAt[rng.Intn(len(rstAt))]
		add(&rsts, fmt.Sprintf("rn%d", at), setBytes(b, at+1, 0xd0+(b[at+1]-0xd0+1+byte(rng.Intn(7)))&7))
		c := bytes.Clone(b)
		add(&rsts, fmt.Sprintf("rd%d", at), append(c[:at], c[at+2:]...))
		add(&rsts, fmt.Sprintf("rr%d", at), insertAt(b, at, 0xff, b[at+1]))
	}
	for _, hv := range []byte{0x11, 0x12, 0x21, 0x22, 0x41, 0x42} {
		for _, s := range segs {
			if (s.m == 0xc0 || s.m == 0xc1 || s.m == 0xc2) && s.n >= 17 && b[s.off+9] == 3 {
				add(&hvs, fmt.Sprintf("hv%02x", hv), setBytes(b, s.off+11, hv, b[s.off+12], 0x11, b[s.off+15], 0x11))
			}
		}
	}

	kinds := [][]mut{cuts, flips, fields, inserts, lengths, appends, dris, rsts, hvs}
	for _, k := range kinds {
		rng.Shuffle(len(k), func(i, j int) { k[i], k[j] = k[j], k[i] })
	}
	var sel []mut
	for len(sel) < n {
		more := false
		for i, k := range kinds {
			if len(k) > 0 && len(sel) < n {
				sel = append(sel, k[0])
				kinds[i] = k[1:]
				more = true
			}
		}
		if !more {
			break
		}
	}
	return sel
}

func (g *gen) mutated(fxs []fx, td string) {
	var stripStarts []int
	if b, err := os.ReadFile(filepath.Join(td, "strip_rst.starts")); err == nil {
		stripStarts = parseInts(string(b))
	}
	for i, f := range fxs {
		n := 64
		switch {
		case len(f.b) > 400<<10:
			n = 6
		case len(f.b) > 40<<10:
			n = 24
		case len(f.b) < 1200:
			n = 40
		}
		base := strings.TrimSuffix(f.name, ".jpg")
		for k, m := range mutations(f.b, rand.New(rand.NewSource(int64(1000+i))), n) {
			g.put(base+"~"+m.tag+".jpg", m.b)
			if f.name == "strip_rst.jpg" {
				w0, h0 := dims(f.b)
				g.regionSidecars(base+"~"+m.tag, w0, h0, stripStarts, k)
			}
		}
	}
}

func parseInts(s string) []int {
	var v []int
	for _, f := range strings.Fields(s) {
		if n, err := strconv.Atoi(f); err == nil {
			v = append(v, n)
		}
	}
	return v
}

// regionSidecars writes <name>.starts (the offsets of the undamaged stream, damaged as
// TestRegionBadRestartOffsets, TestRegionOffsetsTrusted and TestRegionFewerEntropyBytes do:
// shifted, huge, short, out of order) and <name>.rect (scale 1 and 2 regions) for a damaged
// stream, so the region and offset paths run on it in both oracles.
func (g *gen) regionSidecars(name string, w0, h0 int, starts []int, k int) {
	if w0 == 0 || h0 == 0 || len(starts) < 2 {
		return
	}
	st := slices.Clone(starts)
	switch k % 5 {
	case 1:
		for i := 1; i < len(st); i += 2 {
			st[i] += k%7 - 3
		}
	case 2:
		st[1+k%(len(st)-1)] = 1 << 40
	case 3:
		st = st[:1+len(st)/2]
	case 4:
		st[1], st[len(st)-1] = st[len(st)-1], st[1]
	}
	var sb strings.Builder
	for _, v := range st {
		fmt.Fprintln(&sb, v)
	}
	g.put(name+".starts", []byte(sb.String()))
	g.put(name+".rect", []byte(rectLines(rand.New(rand.NewSource(int64(k))), w0, h0, []int{1, 2})))
}

// tablesMutations: the abbreviated tables stream of TestFuzzTables, damaged, against ab-image.jpg.
func (g *gen) tablesMutations(td string) {
	b, err := os.ReadFile(filepath.Join(td, "abbrev", "tables.bin"))
	if err != nil {
		return
	}
	g.put("tbl-base.tbl", b)
	for _, m := range mutations(b, rand.New(rand.NewSource(77)), 120) {
		g.put("tbl-"+m.tag+".tbl", m.b)
	}
}

// ---- regions ----

func dims(b []byte) (w, h int) {
	segs, _ := segments(b)
	for _, s := range segs {
		if s.m >= 0xc0 && s.m <= 0xcf && s.m != 0xc4 && s.m != 0xc8 && s.m != 0xcc && s.off+9 <= len(b) {
			return int(b[s.off+7])<<8 | int(b[s.off+8]), int(b[s.off+5])<<8 | int(b[s.off+6])
		}
	}
	return 0, 0
}

// rects writes a <fixture>.rect sidecar of region decodes: TestRegionIsCrop's rectangles (the
// whole image, a pixel at each corner, the edge rows) and random ones from randRect, at every
// scale and format, plus rectangles past the image edge.
func (g *gen) rects(fxs []fx) {
	rng := rand.New(rand.NewSource(1))
	for _, f := range fxs {
		w0, h0 := dims(f.b)
		if w0 == 0 || h0 == 0 || w0*h0 > 1<<22 {
			continue
		}
		g.put(strings.TrimSuffix(f.name, ".jpg")+".rect", []byte(rectLines(rng, w0, h0, []int{1, 2, 4, 8})))
	}
}

// rectLines is the body of a .rect sidecar for a w0 x h0 image at the given scales.
func rectLines(rng *rand.Rand, w0, h0 int, scales []int) string {
	var sb strings.Builder
	line := func(scale, x0, y0, x1, y1, format, check int) {
		fmt.Fprintf(&sb, "%d %d %d %d %d %d %d\n", scale, x0, y0, x1, y1, format, check)
	}
	for _, s := range scales {
		w, h := (w0+s-1)/s, (h0+s-1)/s
		rs := [][4]int{{0, 0, w, h}, {0, 0, 1, 1}, {w - 1, h - 1, w, h}, {0, 0, w, 1}, {0, h - 1, w, h}}
		for range 3 {
			x0, y0 := rng.Intn(w), rng.Intn(h)
			rs = append(rs, [4]int{x0, y0, x0 + 1 + rng.Intn(w-x0), y0 + 1 + rng.Intn(h-y0)})
		}
		for i, r := range rs {
			line(s, r[0], r[1], r[2], r[3], (i+s)%3, i%2)
		}
		line(s, -3, -5, w/2+1, h/2+1, 1, 0)               // zero-filled on the top left
		line(s, w/2, h/2, w+7, h+9, (s+1)%3, 1)           // past the bottom right
		line(s, w+3, h+3, w+9, h+9, 1, 0)                 // outside: an error
		line(s, 0, 0, 0, 0, 1, 0)                         // empty: the whole image
		line(s, -(1<<29)-1, 0, 5, 5, 1, 0)                // past the range
		line(s, 1<<29-2, 1<<29-2, 1<<29+1, 1<<29+1, 0, 0) // at the range's edge
	}
	return sb.String()
}

// ---- encoder inputs ----

func (g *gen) putSpec(name string, s imgSpec, bufs ...[]byte) {
	b := s.header()
	for _, p := range bufs {
		b = append(b, p...)
	}
	g.put(name+".img", b)
}

// fillSpec builds the buffers of s from sample(x, y, c), where (x, y) are the sub-image's
// coordinates (negative in the parent's margin), as encdiff_test's diffImage does.
func fillSpec(s imgSpec, sample func(x, y, c int) byte) [][]byte {
	pw, ph := s.w+s.ox, s.h+s.oy
	switch s.kind {
	case 0, 8:
		p := make([]byte, s.pixBytes)
		for y := range ph {
			for x := range pw {
				i := y*s.pixStride + 4*x
				p[i], p[i+1], p[i+2] = sample(x-s.ox, y-s.oy, 0), sample(x-s.ox, y-s.oy, 1), sample(x-s.ox, y-s.oy, 2)
				p[i+3] = 255
				if s.kind == 8 {
					p[i+3] = sample(x, y, 1)
				}
			}
		}
		return [][]byte{p}
	case 1:
		p := make([]byte, s.pixBytes)
		for y := range ph {
			for x := range pw {
				p[y*s.pixStride+x] = sample(x-s.ox, y-s.oy, 0)
			}
		}
		return [][]byte{p}
	}
	d := ratioDiv[s.ratio]
	yb, cb, cr := make([]byte, s.ylen), make([]byte, s.clen), make([]byte, s.clen)
	for y := range ph {
		for x := range pw {
			yb[y*s.ystride+x] = sample(x-s.ox, y-s.oy, 0)
			c := (y/d[1])*s.cstride + x/d[0]
			cb[c], cr[c] = sample(x-s.ox, y-s.oy, 1), sample(x-s.ox, y-s.oy, 2)
		}
	}
	return [][]byte{yb, cb, cr}
}

// diffSample is the pixel function of diffImage: base colour with a mode (one odd pixel, flat
// blocks among noise, noise, stripes, bytes as pixels, flat MCUs).
func diffSample(rng *rand.Rand, w, h int, mode byte, base [4]byte, amp int, p []byte) func(x, y, c int) byte {
	return func(x, y, c int) byte {
		v := int(base[c])
		switch mode {
		case 1:
			if x == w-1 && y == h-1 {
				v ^= 0x5a
			}
		case 2:
			if (x/8+y/8)%3 != 0 {
				v += rng.Intn(2*amp+1) - amp
			}
		case 3:
			v += rng.Intn(2*amp+1) - amp
		case 4:
			v = rng.Intn(256)
		case 5:
			v += (x % 3) * 40
		case 6:
			if len(p) > 0 {
				v = int(p[uint((y*w+x)*3+c)%uint(len(p))])
			}
		case 7:
			v = int(base[c] ^ byte((x/16)*31+(y/16)*17+c*7))
		}
		return byte(min(255, max(0, v)))
	}
}

func (g *gen) encoderInputs(td string, fxs []fx) {
	rng := rand.New(rand.NewSource(75))
	// TestEncodeMatchesV040 / diffImage: random images of every kind, sub-images off the block grid.
	for i := range 330 {
		p := make([]byte, 64)
		rng.Read(p)
		w, h := 1+rng.Intn(100), 1+rng.Intn(100)
		switch i % 11 {
		case 0:
			w, h = 1+rng.Intn(17), 1+rng.Intn(17)
		case 1:
			w, h = 16*(1+rng.Intn(6)), 8*(1+rng.Intn(6))
		case 2:
			w, h = 1+rng.Intn(300), 1+rng.Intn(24)
		}
		if i%60 == 0 { // Rx tiles: 770 px
			w, h = 770, 769-i/60%3*200
		}
		kind := byte(i % 9)
		mode := byte(rng.Intn(8))
		ox, oy, pad := rng.Intn(5), rng.Intn(5), []int{0, 0, 4, 8, 3}[rng.Intn(5)]
		s := newSpec(kind, w, h, ox, oy, pad)
		base := [4]byte{p[1], p[3], p[10], 255}
		if mode == 0 || mode == 5 || mode == 7 {
			base[0], base[1], base[2] = p[0], p[1], p[2]
		}
		g.putSpec(fmt.Sprintf("rnd-%03d-k%d-m%d", i, kind, mode), s, fillSpec(s, diffSample(rng, w, h, mode, base, 1+int(p[11]%6), p))...)
	}
	// TestUniformSameBytes: one colour, and one pixel off.
	for _, sz := range [][2]int{{1, 1}, {17, 9}, {64, 64}, {130, 70}} {
		for kind := byte(0); kind <= 8; kind++ {
			c := [3]byte{byte(rng.Intn(256)), byte(rng.Intn(256)), byte(rng.Intn(256))}
			s := newSpec(kind, sz[0], sz[1], 0, 0, 0)
			g.putSpec(fmt.Sprintf("uni-%dx%d-k%d", sz[0], sz[1], kind), s, fillSpec(s, func(x, y, ch int) byte { return c[ch] })...)
			if sz[0]*sz[1] > 1 {
				g.putSpec(fmt.Sprintf("near-%dx%d-k%d", sz[0], sz[1], kind), s, fillSpec(s, func(x, y, ch int) byte {
					if x == sz[0]-1 && y == sz[1]-1 && ch == 2 {
						return c[ch] + 1
					}
					return c[ch]
				})...)
			}
		}
	}
	// The PPM fixtures as RGBA and NRGBA.
	ppms, _ := filepath.Glob(filepath.Join(td, "ppm", "*.ppm"))
	sort.Strings(ppms)
	for _, pp := range ppms {
		b, err := os.ReadFile(pp)
		if err != nil {
			continue
		}
		var magic string
		var w, h, mx int
		if _, err := fmt.Fscan(bytes.NewReader(b), &magic, &w, &h, &mx); err != nil || magic != "P6" || len(b) < w*h*3 {
			continue
		}
		pix := b[len(b)-w*h*3:]
		for _, kind := range []byte{0, 8, 1, 4, 2} {
			s := newSpec(kind, w, h, 0, 0, 0)
			g.putSpec("ppm-"+strings.TrimSuffix(filepath.Base(pp), ".ppm")+fmt.Sprintf("-k%d", kind), s,
				fillSpec(s, func(x, y, c int) byte { return pix[(y*w+x)*3+c] })...)
		}
	}
	// The decoded fixtures, at the formats the tests encode: RGBA, Gray and YCbCr at the stream's sampling.
	for _, f := range fxs {
		if strings.HasPrefix(f.name, "ab-") || strings.HasPrefix(f.name, "prog_") && len(f.b) > 8000 {
			continue
		}
		fw, fh := dims(f.b)
		if fw == 0 || fh == 0 || fw*fh > 1<<22 || strings.Contains(f.name, "cut") || strings.Contains(f.name, "badhuff") || strings.Contains(f.name, "rst_") {
			continue
		}
		sc := 1
		for (fw/sc)*(fh/sc) > 200000 {
			sc *= 2
		}
		base := strings.TrimSuffix(f.name, ".jpg")
		for _, fm := range []struct {
			f jpeg.Format
			n string
		}{{jpeg.FormatRGBA, "rgba"}, {jpeg.FormatGray, "gray"}, {jpeg.FormatYCbCr, "ycc"}} {
			img, err := jpeg.Decode(bg, f.b, &jpeg.DecodeOptions{Format: fm.f, Scale: sc, Limits: lim})
			if err != nil || img == nil {
				continue
			}
			if s, bufs, ok := specOf(img); ok {
				g.putSpec("dec-"+base+"-"+fm.n, s, bufs...)
			}
		}
	}
}

// specOf is the visible samples of a decoded image as encoder input, with no padding and origin 0.
func specOf(img image.Image) (imgSpec, [][]byte, bool) {
	switch m := img.(type) {
	case *image.RGBA:
		w, h := m.Rect.Dx(), m.Rect.Dy()
		s := newSpec(0, w, h, 0, 0, 0)
		p := make([]byte, 0, 4*w*h)
		for y := 0; y < h; y++ {
			o := m.PixOffset(m.Rect.Min.X, m.Rect.Min.Y+y)
			p = append(p, m.Pix[o:o+4*w]...)
		}
		return s, [][]byte{p}, w > 0 && h > 0
	case *image.Gray:
		w, h := m.Rect.Dx(), m.Rect.Dy()
		s := newSpec(1, w, h, 0, 0, 0)
		p := make([]byte, 0, w*h)
		for y := 0; y < h; y++ {
			o := m.PixOffset(m.Rect.Min.X, m.Rect.Min.Y+y)
			p = append(p, m.Pix[o:o+w]...)
		}
		return s, [][]byte{p}, w > 0 && h > 0
	case *image.YCbCr:
		w, h := m.Rect.Dx(), m.Rect.Dy()
		s := newSpec(byte(2+m.SubsampleRatio), w, h, 0, 0, 0)
		d := ratioDiv[m.SubsampleRatio]
		cw, ch := (w+d[0]-1)/d[0], (h+d[1]-1)/d[1]
		var y, cb, cr []byte
		for r := 0; r < h; r++ {
			o := m.YOffset(m.Rect.Min.X, m.Rect.Min.Y+r)
			y = append(y, m.Y[o:o+w]...)
		}
		for r := 0; r < ch; r++ {
			o := m.COffset(m.Rect.Min.X, m.Rect.Min.Y+r*d[1])
			cb = append(cb, m.Cb[o:o+cw]...)
			cr = append(cr, m.Cr[o:o+cw]...)
		}
		return s, [][]byte{y, cb, cr}, w > 0 && h > 0
	}
	return imgSpec{}, nil, false
}

// ---- synthetic streams ----

// synthDRI is jpeg_region_test's synthDRI: a w x h grayscale baseline stream with restart interval
// ri, one random DC-only block per MCU and the standard tables not sent. starts are the offsets of
// each interval's first entropy byte.
func synthDRI(w, h, ri int) ([]byte, []int) {
	b := []byte{0xff, 0xd8, 0xff, 0xdb, 0, 67, 0}
	for range 64 {
		b = append(b, 1)
	}
	b = append(b, 0xff, 0xc0, 0, 11, 8, byte(h>>8), byte(h), byte(w>>8), byte(w), 1, 1, 0x11, 0)
	b = append(b, 0xff, 0xdd, 0, 4, byte(ri>>8), byte(ri))
	b = append(b, 0xff, 0xda, 0, 8, 1, 1, 0, 0, 63, 0)
	dcCode := [10]struct{ c, n uint32 }{{0, 2}, {2, 3}, {3, 3}, {4, 3}, {5, 3}, {6, 3}, {14, 4}, {30, 5}, {62, 6}, {126, 7}}
	var acc uint32
	var nacc uint
	put := func(v uint32, n uint) {
		for i := int(n) - 1; i >= 0; i-- {
			acc = acc<<1 | v>>uint(i)&1
			if nacc++; nacc == 8 {
				b = append(b, byte(acc))
				if byte(acc) == 0xff {
					b = append(b, 0)
				}
				acc, nacc = 0, 0
			}
		}
	}
	rng := rand.New(rand.NewSource(4096))
	mcus := ((w + 7) / 8) * ((h + 7) / 8)
	starts := []int{len(b)}
	var pred int32
	for m := 0; m < mcus; m++ {
		if m > 0 && m%ri == 0 {
			for nacc != 0 {
				put(1, 1)
			}
			b = append(b, 0xff, 0xd0+byte((m/ri-1)&7))
			starts = append(starts, len(b))
			pred = 0
		}
		dc := rng.Int31n(511) - 255
		diff := dc - pred
		pred = dc
		mag, cat := diff, uint(0)
		if mag < 0 {
			mag = -mag
		}
		for mag>>cat != 0 {
			cat++
		}
		put(dcCode[cat].c, uint(dcCode[cat].n))
		if cat > 0 {
			v := diff
			if diff < 0 {
				v = diff + 1<<cat - 1
			}
			put(uint32(v), cat)
		}
		put(0xa, 4)
	}
	for nacc != 0 {
		put(1, 1)
	}
	return append(b, 0xff, 0xd9), starts
}

func (g *gen) synthDRIs() {
	for _, c := range [][3]int{{64, 64, 1}, {100, 37, 3}, {256, 40, 5}, {512, 64, 7}, {1024, 16, 16}, {200, 200, 9}, {8, 8, 1}, {1000, 8, 100}} {
		b, starts := synthDRI(c[0], c[1], c[2])
		name := fmt.Sprintf("dri-%dx%d-ri%d", c[0], c[1], c[2])
		g.put(name+".jpg", b)
		var sb strings.Builder
		for _, s := range starts {
			fmt.Fprintln(&sb, s)
		}
		g.put(name+".starts", []byte(sb.String()))
		for k, m := range mutations(b, rand.New(rand.NewSource(int64(c[0]+c[2]))), 12) {
			g.put(name+"~"+m.tag+".jpg", m.b)
			g.regionSidecars(name+"~"+m.tag, c[0], c[1], starts, k)
		}
	}
}

// ---- coefficient-level streams ----

type huffTab struct {
	tc, th byte
	bits   [16]byte
	vals   []byte
	code   [256]uint32
	size   [256]uint
}

// assign gives every value its canonical code (T.81 Annex C) from bits and vals.
func (h *huffTab) assign() {
	code, k := uint32(0), 0
	for l := 1; l <= 16; l++ {
		for range h.bits[l-1] {
			h.code[h.vals[k]], h.size[h.vals[k]] = code, uint(l)
			code++
			k++
		}
		code <<= 1
	}
}

// stdTables reads the four standard DHT tables out of a fixture.
func stdTables(b []byte) (map[byte]*huffTab, [][]byte) {
	segs, _ := segments(b)
	t := map[byte]*huffTab{}
	var raw [][]byte
	for _, s := range segs {
		if s.m != 0xc4 {
			continue
		}
		raw = append(raw, b[s.off:s.off+2+s.n])
		p := b[s.off+4 : s.off+2+s.n]
		for len(p) >= 17 {
			h := &huffTab{tc: p[0] >> 4, th: p[0] & 15}
			copy(h.bits[:], p[1:17])
			n := 0
			for _, c := range h.bits {
				n += int(c)
			}
			if len(p) < 17+n {
				break
			}
			h.vals = p[17 : 17+n]
			h.assign()
			t[h.tc<<4|h.th] = h
			p = p[17+n:]
		}
	}
	return t, raw
}

type bitw struct {
	b    []byte
	acc  uint32
	nacc uint
}

func (w *bitw) put(v uint32, n uint) {
	for i := int(n) - 1; i >= 0; i-- {
		w.acc = w.acc<<1 | v>>uint(i)&1
		if w.nacc++; w.nacc == 8 {
			w.b = append(w.b, byte(w.acc))
			if byte(w.acc) == 0xff {
				w.b = append(w.b, 0)
			}
			w.acc, w.nacc = 0, 0
		}
	}
}

func (w *bitw) flush() {
	for w.nacc != 0 {
		w.put(1, 1)
	}
}

func bitlen(v int) uint {
	if v < 0 {
		v = -v
	}
	n := uint(0)
	for v>>n != 0 {
		n++
	}
	return n
}

func amp(v int, n uint) uint32 {
	if v < 0 {
		v += 1<<n - 1
	}
	return uint32(v) & (1<<n - 1)
}

// coefStream builds a baseline stream whose quantized coefficients are random, so the decode runs
// the IDCT and colour paths on blocks no encoder would write (dct_test's vectorBlock, with
// extreme quantizers). tabs are the standard tables; tq is the two quantizer tables in zigzag order.
func coefStream(rng *rand.Rand, tabs map[byte]*huffTab, rawDHT [][]byte, w, h, ncomp, hv, ri int, rgb bool, mode int) []byte {
	hs, vs := hv>>4, hv&15
	var quant [2][64]byte
	for t := range quant {
		for k := range 64 {
			switch mode % 4 {
			case 0:
				quant[t][k] = byte(1 + rng.Intn(16))
			case 1:
				quant[t][k] = byte(1 + rng.Intn(255))
			case 2:
				quant[t][k] = 255
			default:
				quant[t][k] = byte(1 + (k*7+t)%40)
			}
		}
	}
	b := []byte{0xff, 0xd8}
	if rgb {
		b = append(b, 0xff, 0xee, 0, 14, 'A', 'd', 'o', 'b', 'e', 0, 100, 0, 0, 0, 0, 0)
	}
	for t := range quant {
		if t == 1 && ncomp == 1 {
			break
		}
		b = append(b, 0xff, 0xdb, 0, 67, byte(t))
		b = append(b, quant[t][:]...)
	}
	ids := []byte{1, 2, 3}
	if rgb {
		ids = []byte{'R', 'G', 'B'}
	}
	b = append(b, 0xff, 0xc0, 0, byte(8+3*ncomp), 8, byte(h>>8), byte(h), byte(w>>8), byte(w), byte(ncomp))
	for c := range ncomp {
		sf, tq := byte(0x11), byte(1)
		if c == 0 {
			sf, tq = byte(hs<<4|vs), 0
		}
		b = append(b, ids[c], sf, tq)
	}
	for _, d := range rawDHT {
		b = append(b, d...)
	}
	if ri > 0 {
		b = append(b, 0xff, 0xdd, 0, 4, byte(ri>>8), byte(ri))
	}
	b = append(b, 0xff, 0xda, 0, byte(6+2*ncomp), byte(ncomp))
	for c := range ncomp {
		td := byte(0)
		if c > 0 {
			td = 0x11
		}
		b = append(b, ids[c], td)
	}
	b = append(b, 0, 63, 0)

	w8, h8 := 8*hs, 8*vs
	if ncomp == 1 {
		w8, h8 = 8, 8
	}
	mcus := ((w + w8 - 1) / w8) * ((h + h8 - 1) / h8)
	bw := &bitw{b: b}
	var pred [3]int
	block := func(c int) {
		dcT, acT := tabs[0x00], tabs[0x10]
		if c > 0 {
			dcT, acT = tabs[0x01], tabs[0x11]
		}
		var z [64]int
		switch rng.Intn(5) {
		case 0: // DC only
			z[0] = rng.Intn(2001) - 1000
		case 1: // sparse
			z[0] = rng.Intn(201) - 100
			for range 1 + rng.Intn(6) {
				z[1+rng.Intn(63)] = rng.Intn(41) - 20
			}
		case 2: // dense, small
			for k := range z {
				z[k] = rng.Intn(21) - 10
			}
			z[0] = rng.Intn(401) - 200
		case 3: // extreme
			for k := range z {
				if rng.Intn(3) == 0 {
					z[k] = []int{-1023, 1023, -1, 1, 512, -512}[rng.Intn(6)]
				}
			}
			z[0] = []int{-1000, 1000, 0, 777}[rng.Intn(4)]
		default: // all zero
		}
		diff := z[0] - pred[c]
		pred[c] = z[0]
		n := bitlen(diff)
		bw.put(dcT.code[n], dcT.size[n])
		bw.put(amp(diff, n), n)
		run := 0
		for k := 1; k < 64; k++ {
			if z[k] == 0 {
				run++
				continue
			}
			for run > 15 {
				bw.put(acT.code[0xf0], acT.size[0xf0])
				run -= 16
			}
			n := bitlen(z[k])
			sym := byte(run<<4) | byte(n)
			bw.put(acT.code[sym], acT.size[sym])
			bw.put(amp(z[k], n), n)
			run = 0
		}
		if run > 0 {
			bw.put(acT.code[0], acT.size[0])
		}
	}
	for m := 0; m < mcus; m++ {
		if ri > 0 && m > 0 && m%ri == 0 {
			bw.flush()
			bw.b = append(bw.b, 0xff, 0xd0+byte((m/ri-1)&7))
			pred = [3]int{}
		}
		if ncomp == 1 {
			block(0)
			continue
		}
		for range hs * vs {
			block(0)
		}
		block(1)
		block(2)
	}
	bw.flush()
	return append(bw.b, 0xff, 0xd9)
}

func (g *gen) coefStreams(fxs []fx) {
	var src []byte
	for _, f := range fxs {
		if f.name == "synth_420_203x157.jpg" {
			src = f.b
		}
	}
	tabs, raw := stdTables(src)
	if len(tabs) != 4 {
		check(fmt.Errorf("synth_420_203x157.jpg: %d tables, want 4", len(tabs)))
	}
	rng := rand.New(rand.NewSource(0x39))
	sampl := []int{0x11, 0x21, 0x12, 0x22, 0x41, 0x42}
	for i := range 72 {
		w, h := 1+rng.Intn(70), 1+rng.Intn(70)
		if i%6 == 0 {
			w, h = 8*(1+rng.Intn(8)), 8*(1+rng.Intn(8))
		}
		nc := 3
		if i%5 == 4 {
			nc = 1
		}
		hv := sampl[rng.Intn(len(sampl))]
		rgb := i%9 == 8
		if rgb {
			hv = 0x11
		}
		ri := []int{0, 0, 1, 3, 7}[rng.Intn(5)]
		g.put(fmt.Sprintf("coef-%03d-%dx%d-c%d-hv%02x-ri%d-m%d.jpg", i, w, h, nc, hv, ri, i%4),
			coefStream(rng, tabs, raw, w, h, nc, hv, ri, rgb, i))
	}
}

// ---- random Huffman tables ----

// randHuff is huff_comb_test.go's randHuff: a random valid table (no all-ones code) whose longest
// code is capped at 3, 6, 8, 9, 10, 11, 12 or 16 bits, so codes plus their value bits reach, and
// pass, the 10 bits of the decoder's combined lookup table.
func randHuff(rng *rand.Rand, isDC bool, tc, th byte) *huffTab {
	var syms []byte
	if isDC {
		for _, s := range rng.Perm(16)[:1+rng.Intn(12)] {
			syms = append(syms, byte(s))
		}
	} else {
		for _, s := range rng.Perm(256)[:2+rng.Intn(70)] {
			syms = append(syms, byte(s))
		}
		if rng.Intn(5) != 0 {
			syms = append(syms, 0x00, 0xf0)
		}
	}
	maxL := []int{3, 6, 8, 9, 10, 11, 12, 16}[rng.Intn(8)]
	room := 65535
	var byLen [17][]byte
	seen := [256]bool{}
	for _, s := range syms {
		if seen[s] {
			continue
		}
		seen[s] = true
		l := 1 + rng.Intn(maxL)
		for l <= 16 && 65536>>l > room {
			l++
		}
		if l > 16 {
			continue
		}
		room -= 65536 >> l
		byLen[l] = append(byLen[l], s)
	}
	h := &huffTab{tc: tc, th: th}
	for l := 1; l <= 16; l++ {
		h.bits[l-1] = byte(len(byLen[l]))
		h.vals = append(h.vals, byLen[l]...)
	}
	if len(h.vals) == 0 {
		h.bits[1], h.vals = 1, []byte{0}
	}
	h.assign()
	return h
}

// randHuffStream is randScan of huff_comb_test.go as a whole stream: 15x10 MCUs of 4:4:4 (a block
// of each component) whose symbols are random picks from random tables, and, on odd seeds, 6 bit
// flips. TestCombinedRandomTables compares the combined table with the plain one inside Go; here
// Go's output is the reference for the Rust decoder's.
func randHuffStream(seed int64) []byte {
	rng := rand.New(rand.NewSource(seed))
	var dc, ac [2]*huffTab
	for k := range dc {
		dc[k] = randHuff(rng, true, 0, byte(k))
		ac[k] = randHuff(rng, false, 1, byte(k))
	}
	bw := &bitw{}
	for range 150 {
		for _, k := range [3]int{0, 1, 1} {
			s := dc[k].vals[rng.Intn(len(dc[k].vals))]
			bw.put(dc[k].code[s], dc[k].size[s])
			bw.put(rng.Uint32(), uint(s))
			for kk := 1; kk < 64; kk++ {
				rs := ac[k].vals[rng.Intn(len(ac[k].vals))]
				bw.put(ac[k].code[rs], ac[k].size[rs])
				run, size := int(rs>>4), uint(rs&15)
				if size == 0 {
					if run != 15 {
						break
					}
					kk += 15
					continue
				}
				kk += run
				bw.put(rng.Uint32(), size)
			}
		}
	}
	bw.flush()
	data := bw.b
	for i := 0; seed%2 == 1 && i < 6; i++ {
		data[rng.Intn(len(data))] ^= byte(1 << uint(rng.Intn(8)))
	}
	b := []byte{0xff, 0xd8}
	for t := byte(0); t < 2; t++ {
		b = append(b, 0xff, 0xdb, 0, 67, t)
		for k := range 64 {
			b = append(b, byte(1+(k+int(t))%8))
		}
	}
	b = append(b, 0xff, 0xc0, 0, 17, 8, 0, 80, 0, 120, 3, 1, 0x11, 0, 2, 0x11, 1, 3, 0x11, 1)
	for _, h := range []*huffTab{dc[0], dc[1], ac[0], ac[1]} {
		b = append(b, 0xff, 0xc4, 0, byte(19+len(h.vals)), h.tc<<4|h.th)
		b = append(b, h.bits[:]...)
		b = append(b, h.vals...)
	}
	b = append(b, 0xff, 0xda, 0, 12, 3, 1, 0x00, 2, 0x11, 3, 0x11, 0, 63, 0)
	b = append(b, data...)
	return append(b, 0xff, 0xd9)
}

func (g *gen) randHuffStreams() {
	for seed := range int64(400) {
		g.put(fmt.Sprintf("rh~%03d.jpg", seed), randHuffStream(seed))
	}
}

// ---- kernels ----

// kernels: the fancy rows (fancy_test, plus lengths Go panics on).
func (g *gen) kernels() {
	rng := rand.New(rand.NewSource(7))
	row := func(n, kind int) []byte {
		b := make([]byte, n)
		for i := range b {
			switch kind {
			case 0:
				b[i] = byte(rng.Intn(256))
			case 1:
				b[i] = byte(255 * (i & 1))
			default:
				b[i] = 255
			}
		}
		return b
	}
	for f := range 8 {
		var b []byte
		for n := f * 40; n < f*40+40; n++ {
			for kc := range 3 {
				for _, op := range []byte{1, 2} {
					no := n + rng.Intn(3)
					if f == 7 { // lengths Go panics on
						no = max(0, n-rng.Intn(3))
					}
					dl := []int{2 * n, 2*n - 1, 2*n + 3}[rng.Intn(3)]
					if f == 7 {
						dl = []int{2*n - 2, 2*n - 1, n, 0, 2 * n}[rng.Intn(5)]
					}
					dl = max(0, dl)
					b = append(b, op, byte(n), byte(n>>8), byte(no), byte(no>>8), byte(dl), byte(dl>>8))
					b = append(b, row(n, kc)...)
					if op == 2 {
						b = append(b, row(no, rng.Intn(3))...)
					}
				}
			}
		}
		g.put(fmt.Sprintf("fancy-%d.fancy", f), b)
	}
}

// ---- fuzz seeds ----

// seedFiles adds the fuzz seeds (tests/fuzz/seeds/decode, with the 4-byte options prefix FuzzDecode
// reads) as seed-<name>.jpg without the prefix.
func (g *gen) seedFiles(dir string) error {
	// The FuzzRowReader seeds have the same prefix; their names start with seed-rr-.
	for _, t := range []struct{ sub, pfx string }{{"decode", "seed-"}, {"row_reader", "seed-rr-"}} {
		ents, err := os.ReadDir(filepath.Join(dir, t.sub))
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			return err
		}
		for _, e := range ents {
			b, err := os.ReadFile(filepath.Join(dir, t.sub, e.Name()))
			if err != nil {
				return err
			}
			if len(b) > 4 {
				g.put(t.pfx+e.Name()+".jpg", b[4:])
			}
		}
	}
	return nil
}
