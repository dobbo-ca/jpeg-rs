package main

import (
	"bytes"
	"fmt"
	"math/rand"
	"slices"
	"strings"
)

// The streams Go's hand-built marker and restart tests make (internal/jpeg/marker_parity_test.go,
// internal/codec/marker_test.go, decode_fast_test.go, decode_rstfast_test.go,
// decode_bitexact_test.go), as inputs: Go supplies the expected output for each, so the tests'
// own expectations (turbo's behaviour) are not needed here.

var (
	mkSOI     = []byte{0xff, 0xd8}
	mkDQT     = append([]byte{0xff, 0xdb, 0, 0x43, 0}, bytes.Repeat([]byte{1}, 64)...)
	mkSOS3    = []byte{0xff, 0xda, 0, 12, 3, 1, 0, 2, 0, 3, 0, 0, 0x3f, 0}
	mkSOSGray = []byte{0xff, 0xda, 0, 8, 1, 1, 0, 0, 0x3f, 0}
	mkAdobe   = []byte{0xff, 0xee, 0, 14, 'A', 'd', 'o', 'b', 'e', 0, 100, 0, 0, 0, 0, 0}
)

func cat(parts ...[]byte) []byte { return bytes.Join(parts, nil) }

// mkSOF3 is a 3-component SOF0; a length past 17 pads zeros.
func mkSOF3(w, h int, ySamp byte, length int) []byte {
	b := []byte{0xff, 0xc0, 0, byte(length), 8, byte(h >> 8), byte(h), byte(w >> 8), byte(w), 3,
		1, ySamp, 0, 2, 0x11, 0, 3, 0x11, 0}
	return append(b, make([]byte, length-17)...)
}

func mkSOFGray(samp byte) []byte {
	return []byte{0xff, 0xc0, 0, 11, 8, 0, 8, 0, 8, 1, 1, samp, 0}
}

// mkDHT holds one code of length 1.
func mkDHT(index byte) []byte {
	return append([]byte{0xff, 0xc4, 0, 20, index, 1}, make([]byte, 16)...)
}

func mkDQT16(pqtq byte) []byte {
	return append([]byte{0xff, 0xdb, 0, 131, pqtq}, bytes.Repeat([]byte{0, 1}, 64)...)
}

// hdrCases are parityCases: header-only streams, so the config, truncated and error paths run on
// every marker rule.
func hdrCases() []mut {
	s8 := mkSOF3(8, 8, 0x11, 17)
	d := func(b ...byte) []byte { return b }
	return []mut{
		{"base", cat(mkSOI, mkDQT, s8, mkSOS3)},
		{"gray 2x2", cat(mkSOI, mkDQT, mkSOFGray(0x22), mkSOSGray)},
		{"gray 4x4", cat(mkSOI, mkDQT, mkSOFGray(0x44), mkSOSGray)},
		{"gray 5x1", cat(mkSOI, mkDQT, mkSOFGray(0x51), mkSOSGray)},
		{"height 0", cat(mkSOI, mkDQT, mkSOF3(8, 0, 0x11, 17), mkSOS3)},
		{"width 0", cat(mkSOI, mkDQT, mkSOF3(0, 8, 0x11, 17), mkSOS3)},
		{"width 65500", cat(mkSOI, mkDQT, mkSOF3(65500, 8, 0x11, 17), mkSOS3)},
		{"width 65501", cat(mkSOI, mkDQT, mkSOF3(65501, 8, 0x11, 17), mkSOS3)},
		{"sof length long", cat(mkSOI, mkDQT, mkSOF3(8, 8, 0x11, 18), mkSOS3)},
		{"dht class 1 index 3", cat(mkSOI, mkDQT, mkDHT(0x13), s8, mkSOS3)},
		{"dht class 2", cat(mkSOI, mkDQT, mkDHT(0x20), s8, mkSOS3)},
		{"dht index 4", cat(mkSOI, mkDQT, mkDHT(0x04), s8, mkSOS3)},
		{"dqt precision 2", cat(mkSOI, mkDQT16(0x20), s8, mkSOS3)},
		{"dqt index 4", cat(mkSOI, mkDQT16(0x14), s8, mkSOS3)},
		{"dri", cat(mkSOI, mkDQT, d(0xff, 0xdd, 0, 4, 0, 7), s8, mkSOS3)},
		{"dri length 5", cat(mkSOI, mkDQT, d(0xff, 0xdd, 0, 5, 0, 7, 0), s8, mkSOS3)},
		{"junk before sof", cat(mkSOI, mkDQT, d(0x00, 0x11), s8, mkSOS3)},
		{"ff00 before sof", cat(mkSOI, mkDQT, d(0xff, 0x00), s8, mkSOS3)},
		{"fill before sof", cat(mkSOI, mkDQT, d(0xff, 0xff), s8, mkSOS3)},
		{"com and app1", cat(mkSOI, d(0xff, 0xfe, 0, 3, 'x', 0xff, 0xe1, 0, 2), mkDQT, s8, mkSOS3)},
		{"reserved marker", cat(mkSOI, d(0xff, 0xf0, 0, 2), mkDQT, s8, mkSOS3)},
		{"duplicate soi", cat(mkSOI, mkSOI, mkDQT, s8, mkSOS3)},
		{"eoi before sos", cat(mkSOI, mkDQT, s8, d(0xff, 0xd9))},
		{"no sos", cat(mkSOI, mkDQT, s8)},
		{"sos before sof", cat(mkSOI, mkDQT, mkSOS3, s8)},
		{"sos bad id", cat(mkSOI, mkDQT, s8, d(0xff, 0xda, 0, 8, 1, 9, 0, 0, 0x3f, 0))},
		{"sos dup id", cat(mkSOI, mkDQT, s8, d(0xff, 0xda, 0, 10, 2, 1, 0, 1, 0, 0, 0x3f, 0))},
		{"sos one comp", cat(mkSOI, mkDQT, s8, d(0xff, 0xda, 0, 8, 1, 2, 0, 0, 0x3f, 0))},
		{"sos ids 3,1", cat(mkSOI, mkDQT, s8, d(0xff, 0xda, 0, 10, 2, 3, 0, 1, 0, 0, 0x3f, 0))},
		{"sos ids 2,3", cat(mkSOI, mkDQT, s8, d(0xff, 0xda, 0, 10, 2, 2, 0, 3, 0, 0, 0x3f, 0))},
		{"com length 0", cat(mkSOI, d(0xff, 0xfe, 0, 0), mkDQT, s8, mkSOS3)},
		{"com length 1", cat(mkSOI, d(0xff, 0xfe, 0, 1), mkDQT, s8, mkSOS3)},
		{"app0 length 0", cat(mkSOI, d(0xff, 0xe0, 0, 0), mkDQT, s8, mkSOS3)},
		{"app1 length 1", cat(mkSOI, d(0xff, 0xe1, 0, 1), mkDQT, s8, mkSOS3)},
		{"app14 length 0", cat(mkSOI, d(0xff, 0xee, 0, 0), mkDQT, s8, mkSOS3)},
		{"dnl length 0", cat(mkSOI, d(0xff, 0xdc, 0, 0), mkDQT, s8, mkSOS3)},
		{"dqt length 0", cat(mkSOI, d(0xff, 0xdb, 0, 0), mkDQT, s8, mkSOS3)},
		{"dac length 0", cat(mkSOI, d(0xff, 0xcc, 0, 0), mkDQT, s8, mkSOS3)},
		{"dac dc ok", cat(mkSOI, d(0xff, 0xcc, 0, 4, 0x00, 0x10), mkDQT, s8, mkSOS3)},
		{"dac dc L>U", cat(mkSOI, d(0xff, 0xcc, 0, 4, 0x00, 0x01), mkDQT, s8, mkSOS3)},
		{"dac ac 0", cat(mkSOI, d(0xff, 0xcc, 0, 4, 0x10, 0x00), mkDQT, s8, mkSOS3)},
		{"dac index 32", cat(mkSOI, d(0xff, 0xcc, 0, 4, 0x20, 0x05), mkDQT, s8, mkSOS3)},
		{"dac odd", cat(mkSOI, d(0xff, 0xcc, 0, 3, 0x00), mkDQT, s8, mkSOS3)},
		{"sos length long", cat(mkSOI, mkDQT, s8, d(0xff, 0xda, 0, 9, 1, 1, 0, 0, 0x3f, 0, 0))},
	}
}

// betweenScans are the segments TestBetweenScansTurboParity puts before a second SOS; fatal ones
// also feed TestAfterScanTurboParity.
var betweenScans = []struct {
	name  string
	seg   []byte
	fatal bool
}{
	{"com", []byte{0xff, 0xfe, 0, 4, 'h', 'i'}, false},
	{"app1", []byte{0xff, 0xe1, 0, 3, 'x'}, false},
	{"dac", []byte{0xff, 0xcc, 0, 4, 0x00, 0x10}, false},
	{"dnl", []byte{0xff, 0xdc, 0, 4, 0, 16}, false},
	{"dri", []byte{0xff, 0xdd, 0, 4, 0, 1}, false},
	{"rst0", []byte{0xff, 0xd0}, false},
	{"junk", []byte{0x00, 0x11}, false},
	{"dht class 2", append([]byte{0xff, 0xc4, 0, 19, 0x20}, make([]byte, 16)...), true},
	{"dht index 4", append([]byte{0xff, 0xc4, 0, 19, 0x04}, make([]byte, 16)...), true},
	{"dht short", []byte{0xff, 0xc4, 0, 10, 0, 1}, true},
	{"dqt index 4", append([]byte{0xff, 0xdb, 0, 67, 0x04}, make([]byte, 64)...), true},
	{"dqt length 0", []byte{0xff, 0xdb, 0, 0}, true},
	{"dri length 5", []byte{0xff, 0xdd, 0, 5, 0, 1, 0}, true},
	{"reserved marker", []byte{0xff, 0xf0, 0, 2}, true},
	{"second sof", []byte{0xff, 0xc0, 0, 11, 8, 0, 16, 0, 16, 1, 1, 0x11, 0}, true},
	{"second soi", []byte{0xff, 0xd8}, true},
}

var slugger = strings.NewReplacer(" ", "-", ",", "", ">", "gt")

func slug(s string) string { return slugger.Replace(s) }

func findFx(fxs []fx, name string) []byte {
	for _, f := range fxs {
		if f.name == name {
			return bytes.Clone(f.b)
		}
	}
	check(fmt.Errorf("fixture %s is missing", name))
	return nil
}

// markerStreams writes the marker and restart streams of Go's tests as inputs.
func (g *gen) markerStreams(fxs []fx) {
	for i, c := range hdrCases() {
		g.put(fmt.Sprintf("hdr-%02d-%s.jpg", i, slug(c.tag)), c.b)
	}
	// TestTablesStateReset: DRI and Adobe in the tables stream must not leak into the image.
	g.put("tables-reset.jpg", cat(mkSOI, mkSOF3(8, 8, 0x11, 17), mkSOS3))
	g.put("tables-reset.tables", cat(mkSOI, []byte{0xff, 0xdd, 0, 4, 0, 7}, mkAdobe, mkDQT, []byte{0xff, 0xd9}))

	// TestBetweenScansTurboParity.
	for _, name := range []string{"synth_multiscan_16x16.jpg", "prog_420_111x75.jpg"} {
		b := findFx(fxs, name)
		first := bytes.Index(b, []byte{0xff, 0xda})
		at := first + 2 + bytes.Index(b[first+2:], []byte{0xff, 0xda})
		for _, tc := range betweenScans {
			g.put("between-"+strings.TrimSuffix(name, ".jpg")+"-"+slug(tc.name)+".jpg", cat(b[:at], tc.seg, b[at:]))
		}
	}

	// TestAfterScanTurboParity: a second SOS, a SOF9 or a fatal segment after the last row, or
	// after a stray marker halfway through the entropy data.
	b := findFx(fxs, "synth_420_203x157.jpg")
	sos := bytes.Index(b, []byte{0xff, 0xda})
	eoi := len(b) - 2
	sof9 := []byte{0xff, 0xc9, 0, 11, 8, 0, 16, 0, 16, 1, 1, 0x11, 0}
	segs := []struct {
		name string
		seg  []byte
	}{{"sos", b[sos : sos+2+int(b[sos+2])<<8|int(b[sos+3])]}, {"sof9", sof9}}
	for _, tc := range betweenScans {
		if tc.fatal {
			segs = append(segs, struct {
				name string
				seg  []byte
			}{tc.name, tc.seg})
		}
	}
	mid := (sos + eoi) / 2
	for b[mid-1] == 0xff {
		mid-- // keep a stuffed 0xff00 whole
	}
	for _, cut := range []struct {
		name string
		at   int
	}{{"eoi", eoi}, {"mid", mid}} {
		g.put("after-"+cut.name+"-none.jpg", cat(b[:cut.at], b[eoi:]))
		for _, s := range segs {
			g.put("after-"+cut.name+"-"+slug(s.name)+".jpg", cat(b[:cut.at], s.seg, b[eoi:]))
		}
	}
	// turbo's crop reads the marker only if it reaches the bottom row.
	g.put("after-eoi-sof9.rect", []byte("1 0 0 100 157 1 0\n1 0 0 203 150 1 0\n1 100 140 203 157 1 0\n"))

	// TestUnscannedComponentTurboParity: a gray stream whose SOF claims Cb and Cr no scan covers.
	for _, name := range []string{"cmp_gray.jpg", "prog_gray_111x75.jpg"} {
		b := findFx(fxs, name)
		i := bytes.Index(b, []byte{0xff, 0xc0, 0, 11})
		if i < 0 {
			i = bytes.Index(b, []byte{0xff, 0xc2, 0, 11})
		}
		if i < 0 {
			check(fmt.Errorf("%s: no gray SOF", name))
		}
		sof := cat(b[i:i+2], []byte{0, 17}, b[i+4:i+9], []byte{3, 1, 0x11, 0, 2, 0x11, 0, 3, 0x11, 0})
		g.put("unscanned-"+strings.TrimSuffix(name, ".jpg")+".jpg", cat(b[:i], sof, b[i+13:]))
	}

	// TestNotSequentialTurboParity: a sequential SOS with Ss/Se/Ah/Al other than 0/63/0/0.
	for _, name := range []string{"synth_420_16x16.jpg", "ms_ycb_cr_420.jpg"} {
		b := findFx(fxs, name)
		sos := bytes.LastIndex(b, []byte{0xff, 0xda})
		ss := sos + 5 + 2*int(b[sos+4])
		for i, p := range [][3]byte{{1, 63, 0}, {0, 62, 0}, {0, 63, 0x10}, {0, 63, 0x01}} {
			src := bytes.Clone(b)
			copy(src[ss:], p[:])
			g.put(fmt.Sprintf("notseq-%s-%d.jpg", strings.TrimSuffix(name, ".jpg"), i), src)
		}
	}

	g.codecMarkerStreams(fxs)
	g.dacStreams(fxs)
	g.fastMCUStreams(fxs)
	g.rstStreams(fxs)
}

// codecMarkerStreams are the streams of internal/codec/marker_test.go.
func (g *gen) codecMarkerStreams(fxs []fx) {
	s420 := findFx(fxs, "synth_420_16x16.jpg")
	n := len(s420) - 2
	eoi := bytes.Index(s420, []byte{0xff, 0xd9})
	dht := bytes.Index(s420, []byte{0xff, 0xc4})
	sos := bytes.Index(s420, []byte{0xff, 0xda})
	g.put("mt-unknown-marker-after-scan.jpg", cat(s420[:eoi], []byte{0xff, 0x30, 0x00, 0x02}, s420[eoi:]))
	dc16 := bytes.Clone(s420)
	dc16[dht+4+1+16+11] = 0x10 // the last symbol of DC table 0
	g.put("mt-dc-symbol-16.jpg", dc16)
	s444 := findFx(fxs, "synth_444_16x16.jpg")
	sos444 := bytes.Index(s444, []byte{0xff, 0xda})
	n444 := len(s444) - 2
	g.put("mt-second-scan.jpg", cat(s444[:n444], s444[sos444:n444], s444[n444:]))
	g.put("mt-rst-after-scan.jpg", cat(s420[:n], []byte{0xff, 0xd0}, s420[n:]))
	g.put("mt-overrun-app3.jpg", cat(s420[:n], []byte{0xff, 0xe3, 0x01, 0x00, 'a'}))
	g.put("mt-overrun-dht.jpg", cat(s420[:n], s420[dht:dht+30]))
	g.put("mt-overrun-app3-no-eoi.jpg", cat(s420[:n], []byte{0xff, 0xe3, 0x00, 0x02, 'a'}))
	g.put("mt-overrun-sos-ss.jpg", bytes.Clone(s420[:sos+11]))
	// TestReadDHTTooManyCodes: BITS that sum to 300.
	bits := make([]byte, 16)
	bits[14], bits[15] = 45, 255
	many := cat([]byte{0xff, 0xc4, 0x01, 0x3f, 0x00}, bits, make([]byte, 300))
	g.put("mt-dht-300-codes.jpg", cat(s420[:sos], many, s420[sos:]))
}

// dacStreams put 0xcc (DAC, arithmetic conditioning: read and ignored by a Huffman decoder)
// segments before the SOS and right after the SOI of a few fixtures.
func (g *gen) dacStreams(fxs []fx) {
	dacs := [][]byte{
		{0xff, 0xcc, 0, 4, 0x00, 0x10},
		{0xff, 0xcc, 0, 4, 0x10, 0x00},
		{0xff, 0xcc, 0, 4, 0x00, 0x01},
		{0xff, 0xcc, 0, 4, 0x20, 0x05},
		{0xff, 0xcc, 0, 3, 0x00},
		{0xff, 0xcc, 0, 2},
		{0xff, 0xcc, 0, 0},
		{0xff, 0xcc, 0, 6, 0x00, 0x10, 0x10, 0x05},
		{0xff, 0xcc, 0, 8, 0x00, 0x10, 0x10, 0x05, 0x01, 0x11},
		{0xff, 0xcc, 0xff, 0xff},
	}
	for _, name := range []string{"synth_420_16x16.jpg", "synth_444_16x16.jpg", "prog_420_111x75.jpg", "cmp_gray.jpg", "synth_multiscan_16x16.jpg"} {
		b := findFx(fxs, name)
		sos := bytes.Index(b, []byte{0xff, 0xda})
		base := strings.TrimSuffix(name, ".jpg")
		for i, d := range dacs {
			g.put(fmt.Sprintf("dac-%s-%d-sos.jpg", base, i), cat(b[:sos], d, b[sos:]))
			g.put(fmt.Sprintf("dac-%s-%d-soi.jpg", base, i), cat(b[:2], d, b[2:]))
		}
	}
}

// fastMCUStreams are denseStream (TestSkipPathBitExact, TestFastMCUAsm, TestSlowMCU) and the two
// streams of decode_fast_test.go (a marker at the first DC symbol, a run of 15 that overflows
// the block), with the generic mutations in place of corruptFuzz.
func (g *gen) fastMCUStreams(fxs []fx) {
	tabs, raw := stdFrom(fxs)
	dense := rstStream(tabs, raw, false, false, 48, 32, 0)
	g.put("dense-48x32.jpg", dense)
	for _, m := range mutations(dense, rand.New(rand.NewSource(48)), 40) {
		g.put("dense-48x32~"+m.tag+".jpg", m.b)
	}
	gray := func(dcBits [16]byte, dcVals []byte, acBits [16]byte, acVals []byte, data ...byte) []byte {
		b := cat(mkSOI, mkDQT, mkSOFGray(0x11))
		b = append(b, 0xff, 0xc4, 0, byte(19+len(dcVals)), 0x00)
		b = append(append(b, dcBits[:]...), dcVals...)
		b = append(b, 0xff, 0xc4, 0, byte(19+len(acVals)), 0x10)
		b = append(append(b, acBits[:]...), acVals...)
		b = append(b, mkSOSGray...)
		return append(append(b, data...), 0xff, 0xd9)
	}
	dcBits := [16]byte{1, 1}
	g.put("fast-marker-at-dc.jpg", gray(dcBits, []byte{0, 1}, [16]byte{1}, []byte{0x00}, 0xff, 0xd9, 0, 0, 0, 0, 0, 0))
	// DC "0", ZRL "10" x3, run 15 size 1 "110" and its bit "1": 01010101101.
	g.put("fast-ac-overflow.jpg", gray(dcBits, []byte{0, 1}, [16]byte{1, 1, 1}, []byte{0x00, 0xf0, 0xf1}, 0x55, 0xa0, 0, 0, 0, 0, 0, 0))
}

func stdFrom(fxs []fx) (map[byte]*huffTab, [][]byte) {
	tabs, raw := stdTables(findFx(fxs, "synth_420_203x157.jpg"))
	if len(tabs) != 4 {
		check(fmt.Errorf("synth_420_203x157.jpg: %d tables, want 4", len(tabs)))
	}
	return tabs, raw
}

// denseBlock is the block of rstStream and denseStream: value sizes 1 to 10, zero runs of one,
// and no EOB (the last coefficient is never zero).
func denseBlock(n int) [64]int {
	var blk [64]int
	for k := range blk {
		if k != 63 && (k*3+n)%7 == 0 {
			continue
		}
		s := (k+n)%10 + 1
		v := 1<<(s-1) + (k*7+n)%(1<<(s-1))
		if (k+n)%3 == 0 {
			v = -v
		}
		blk[k] = v
	}
	return blk
}

// writeBlk writes one block with the standard tables: component c's DC difference, then the AC
// run-length codes.
func writeBlk(bw *bitw, z *[64]int, pred *int, c int, tabs map[byte]*huffTab) {
	dcT, acT := tabs[0x00], tabs[0x10]
	if c > 0 {
		dcT, acT = tabs[0x01], tabs[0x11]
	}
	diff := z[0] - *pred
	*pred = z[0]
	n := bitlen(diff)
	bw.put(dcT.code[n], dcT.size[n])
	bw.put(amp(diff, n), n)
	run := 0
	for k := 1; k < 64; k++ {
		if z[k] == 0 {
			run++
			continue
		}
		for ; run > 15; run -= 16 {
			bw.put(acT.code[0xf0], acT.size[0xf0])
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

// rstStream is rstStream of decode_rstfast_test.go: denseBlock's blocks in a w x h frame (gray,
// 4:4:4 or 4:2:0, quantizers all 1) with a DRI of ri MCUs; ri 0 sends none (denseStream).
func rstStream(tabs map[byte]*huffTab, raw [][]byte, gray, c444 bool, w, h, ri int) []byte {
	nc, per, luma, mw := 3, 6, 4, 16
	switch {
	case gray:
		nc, per, luma, mw = 1, 1, 1, 8
	case c444:
		per, luma, mw = 3, 1, 8
	}
	mcus := ((w + mw - 1) / mw) * ((h + mw - 1) / mw)
	ones := bytes.Repeat([]byte{1}, 64)
	b := cat(mkSOI, []byte{0xff, 0xdb, 0, 67, 0}, ones)
	if nc == 3 {
		b = cat(b, []byte{0xff, 0xdb, 0, 67, 1}, ones)
	}
	ysamp := byte(0x11)
	if nc == 3 && !c444 {
		ysamp = 0x22
	}
	b = append(b, 0xff, 0xc0, 0, byte(8+3*nc), 8, byte(h>>8), byte(h), byte(w>>8), byte(w), byte(nc), 1, ysamp, 0)
	if nc == 3 {
		b = append(b, 2, 0x11, 1, 3, 0x11, 1)
	}
	for _, d := range raw {
		b = append(b, d...)
	}
	if ri > 0 {
		b = append(b, 0xff, 0xdd, 0, 4, byte(ri>>8), byte(ri))
	}
	b = append(b, 0xff, 0xda, 0, byte(6+2*nc), byte(nc), 1, 0x00)
	if nc == 3 {
		b = append(b, 2, 0x11, 3, 0x11)
	}
	b = append(b, 0, 63, 0)
	bw := &bitw{b: b}
	var pred [3]int
	n := 0
	for m := 0; m < mcus; m++ {
		if ri > 0 && m > 0 && m%ri == 0 {
			bw.flush()
			bw.b = append(bw.b, 0xff, 0xd0+byte((m/ri-1)&7))
			pred = [3]int{}
		}
		for k := range per {
			c := 0
			if k >= luma {
				c = k - luma + 1
			}
			blk := denseBlock(n)
			writeBlk(bw, &blk, &pred[c], c, tabs)
			n++
		}
	}
	bw.flush()
	return append(bw.b, 0xff, 0xd9)
}

// mutateRST is mutateRST of decode_rstfast_test.go: 1 to 3 edits (bit flips, byte drops and
// inserts, inserted, dropped and renumbered RSTn, truncation). With end, each edit lands in the
// 600 bytes before a random RSTn or the EOI.
func mutateRST(rng *rand.Rand, src []byte, from int, end bool) []byte {
	b := slices.Clone(src)
	for range 1 + rng.Intn(3) {
		if from >= len(b) {
			break
		}
		off := from + rng.Intn(len(b)-from)
		var rst []int
		for i := from; i+1 < len(b); i++ {
			if b[i] == 0xff && b[i+1]&0xf8 == 0xd0 {
				rst = append(rst, i)
			}
		}
		ends := rst
		if e := bytes.LastIndex(b, []byte{0xff, 0xd9}); e >= from {
			ends = append(slices.Clip(rst), e) // the last interval ends at the EOI
		}
		if end && len(ends) > 0 {
			off = max(from, ends[rng.Intn(len(ends))]-rng.Intn(600))
		}
		switch rng.Intn(8) {
		case 0:
			b[off] ^= 1 << rng.Intn(8)
		case 1:
			b = slices.Delete(b, off, off+1)
		case 2:
			b = slices.Insert(b, off, byte(rng.Intn(256)))
		case 3:
			b = slices.Insert(b, off, 0xff, 0xd0+byte(rng.Intn(8)))
		case 4:
			b = slices.Insert(b, off, 0xff)
		case 5:
			if len(rst) > 0 {
				i := rst[rng.Intn(len(rst))]
				b = slices.Delete(b, i, i+2)
			}
		case 6:
			if len(rst) > 0 {
				b[rst[rng.Intn(len(rst))]+1] = 0xd0 + byte(rng.Intn(8))
			}
		case 7:
			b = b[:off:off]
		}
	}
	return b
}

// rstStreams are rstCorpus' synthetic streams (gray, 4:4:4 and 4:2:0 with restart intervals of
// 1 to 3, and longer ones) with mutateRST's edits, anywhere ("r") and at interval ends ("e"), as
// TestRestartFastPathRandom makes them.
func (g *gen) rstStreams(fxs []fx) {
	tabs, raw := stdFrom(fxs)
	type spec struct {
		name       string
		gray, c444 bool
		w, h, ri   int
	}
	specs := []spec{{"gray-ri16", true, false, 64, 64, 16}, {"420-ri12", false, false, 96, 64, 12}, {"420-ri7", false, false, 96, 64, 7}}
	for _, ri := range []int{1, 2, 3} {
		specs = append(specs,
			spec{fmt.Sprintf("gray-ri%d", ri), true, false, 64, 64, ri},
			spec{fmt.Sprintf("444-ri%d", ri), false, true, 64, 64, ri},
			spec{fmt.Sprintf("420-ri%d", ri), false, false, 128, 128, ri})
	}
	rng := rand.New(rand.NewSource(20261018))
	for _, s := range specs {
		b := rstStream(tabs, raw, s.gray, s.c444, s.w, s.h, s.ri)
		g.put("rst-"+s.name+".jpg", b)
		_, from := segments(b)
		for i := range 20 {
			g.put(fmt.Sprintf("rst-%s~r%02d.jpg", s.name, i), mutateRST(rng, b, from, false))
			g.put(fmt.Sprintf("rst-%s~e%02d.jpg", s.name, i), mutateRST(rng, b, from, true))
		}
	}
}
