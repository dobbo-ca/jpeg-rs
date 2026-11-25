package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"image"
	"strconv"

	jpeg "github.com/dobbo-ca/jpeg-go"
)

// ---- encoder inputs (.img) ----

// An .img file is the encoder input: [kind u8][w u16 LE][h u16 LE][ox u8][oy u8][pad u8], then the
// buffers of a parent image of (w+ox) x (h+oy) pixels, of which the encoder gets the sub-image
// (ox, oy, ox+w, oy+h), as Go's tests do (strides past the width, origins off the block grid). kind
// 0 is RGBA, 1 Gray, 2..7 YCbCr 444, 422, 420, 440, 411, 410, 8 NRGBA. pad is added to every
// stride (4*width for the RGBA kinds). The buffers are Pix; or Y, Cb, Cr, with the plane sizes of
// image.NewYCbCr. A short file is read as zeros.
type imgSpec struct {
	kind                byte
	w, h, ox, oy, pad   int
	ratio               image.YCbCrSubsampleRatio
	ystride, cstride    int
	ylen, clen          int
	pixStride, pixBytes int
}

var ratioDiv = [...][2]int{{1, 1}, {2, 1}, {2, 2}, {1, 2}, {4, 1}, {4, 2}}

func newSpec(kind byte, w, h, ox, oy, pad int) imgSpec {
	s := imgSpec{kind: kind, w: w, h: h, ox: ox, oy: oy, pad: pad}
	pw, ph := w+ox, h+oy
	switch {
	case kind == 0 || kind == 8:
		s.pixStride = 4*pw + pad
		s.pixBytes = s.pixStride * ph
	case kind == 1:
		s.pixStride = pw + pad
		s.pixBytes = s.pixStride * ph
	default:
		s.ratio = image.YCbCrSubsampleRatio(kind - 2)
		d := ratioDiv[s.ratio]
		cw, ch := (pw+d[0]-1)/d[0], (ph+d[1]-1)/d[1]
		s.ystride, s.cstride = pw+pad, cw+pad
		s.ylen, s.clen = s.ystride*ph, s.cstride*ch
	}
	return s
}

func (s imgSpec) header() []byte {
	b := []byte{s.kind, 0, 0, 0, 0, byte(s.ox), byte(s.oy), byte(s.pad)}
	binary.LittleEndian.PutUint16(b[1:], uint16(s.w))
	binary.LittleEndian.PutUint16(b[3:], uint16(s.h))
	return b
}

func parseSpec(b []byte) (imgSpec, []byte, bool) {
	if len(b) < 8 || b[0] > 8 {
		return imgSpec{}, nil, false
	}
	w, h := int(binary.LittleEndian.Uint16(b[1:])), int(binary.LittleEndian.Uint16(b[3:]))
	if w == 0 || h == 0 {
		return imgSpec{}, nil, false
	}
	return newSpec(b[0], w, h, int(b[5]), int(b[6]), int(b[7])), b[8:], true
}

// take copies n bytes from *p, zero-filled if it is short.
func take(p *[]byte, n int) []byte {
	out := make([]byte, n)
	k := copy(out, *p)
	*p = (*p)[k:]
	return out
}

// image builds the encoder's input from the buffers.
func (s imgSpec) image(p []byte) image.Image {
	parent := image.Rect(0, 0, s.w+s.ox, s.h+s.oy)
	sub := image.Rect(s.ox, s.oy, s.ox+s.w, s.oy+s.h)
	switch s.kind {
	case 0:
		return (&image.RGBA{Pix: take(&p, s.pixBytes), Stride: s.pixStride, Rect: parent}).SubImage(sub)
	case 8:
		return (&image.NRGBA{Pix: take(&p, s.pixBytes), Stride: s.pixStride, Rect: parent}).SubImage(sub)
	case 1:
		return (&image.Gray{Pix: take(&p, s.pixBytes), Stride: s.pixStride, Rect: parent}).SubImage(sub)
	}
	m := &image.YCbCr{Y: take(&p, s.ylen), Cb: take(&p, s.clen), Cr: take(&p, s.clen),
		YStride: s.ystride, CStride: s.cstride, SubsampleRatio: s.ratio, Rect: parent}
	return m.SubImage(sub)
}

var encQualities = []int{0, 1, 25, 50, 75, 90, 100, 101}

func runImg(o out, b []byte) {
	s, p, ok := parseSpec(b)
	if !ok {
		return
	}
	img := s.image(p)
	// Chroma444 matters for the RGBA kinds and for YCbCr that goes through RGBA.
	both := s.kind == 0 || s.kind == 8 || s.kind == 3 || s.kind == 5 || s.kind == 6 || s.kind == 7
	for _, q := range encQualities {
		for _, c := range []bool{false, true} {
			if c && !both {
				continue
			}
			op := "e" + strconv.Itoa(q)
			if c {
				op += "c"
			}
			o.safe(op, func() {
				var buf bytes.Buffer
				err := jpeg.Encode(bg, &buf, img, &jpeg.EncodeOptions{Quality: q, Chroma444: c})
				if err != nil {
					o.fail(op, err, nil)
					return
				}
				o.ok(op, encodeRows("jpeg\n", chunks(buf.Bytes())))
			})
		}
	}
	o.safe("ecx", func() {
		var buf bytes.Buffer
		err := jpeg.Encode(cancelled, &buf, img, &jpeg.EncodeOptions{Quality: 75})
		if err != nil {
			o.fail("ecx", err, nil)
			return
		}
		o.ok("ecx", encodeRows("jpeg\n", chunks(buf.Bytes())))
	})
	// The same image through the Encoder every input shares (buffers, the uniform-image cache).
	for _, q := range []int{0, 75} {
		for _, c := range []bool{false, true} {
			if c && !both {
				continue
			}
			op := "se" + strconv.Itoa(q)
			if c {
				op += "c"
			}
			o.safe(op, func() {
				var buf bytes.Buffer
				err := sharedEnc.Encode(bg, &buf, img, &jpeg.EncodeOptions{Quality: q, Chroma444: c})
				if err != nil {
					o.fail(op, err, nil)
					return
				}
				o.ok(op, encodeRows("jpeg\n", chunks(buf.Bytes())))
			})
		}
	}
}

// ---- kernels ----

// A .fancy file is records of [op u8: 1 h2v1, 2 h2v2][n u16][other u16][dst u16] then n bytes of
// row, and for op 2 `other` bytes. dst starts as 0xAA bytes. A bad length is Go's panic, which is
// Rust's InvalidArgument.
func runFancy(o out, b []byte) {
	var rows [][]byte
	for len(b) >= 7 {
		op, n, no, dl := b[0], int(binary.LittleEndian.Uint16(b[1:])), int(binary.LittleEndian.Uint16(b[3:])), int(binary.LittleEndian.Uint16(b[5:]))
		b = b[7:]
		row := take(&b, n)
		var other []byte
		if op == 2 {
			other = take(&b, no)
		}
		dst := bytes.Repeat([]byte{0xaa}, dl)
		func() {
			defer func() {
				if recover() != nil {
					rows = append(rows, []byte("E invalid-argument\n"))
				}
			}()
			if op == 1 {
				jpeg.H2V1FancyRow(dst, row)
			} else {
				jpeg.H2V2FancyRow(dst, row, other)
			}
			rows = append(rows, []byte(fmt.Sprintf("K %d\n", len(dst))), dst)
		}()
	}
	o.ok("fancy", encodeRows("fancy\n", rows))
}
