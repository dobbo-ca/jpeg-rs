package main

import (
	"context"
	"fmt"
	"image"
	"os"
	"path/filepath"
	"sort"
	"strings"

	jpeg "github.com/dobbo-ca/jpeg-go"
)

// ppmLim bounds one tile or strip of a real slide.
var ppmLim = jpeg.Limits{MaxPixels: 1 << 26, MaxBytes: 1 << 30}

// ppmDir decodes every .jpg in dir (FormatRGBA,
// Scale 1) and writes <stem>.ppm, the RGB samples as binary PPM, to out. examples/ppm.rs is the
// Rust side. Real-file tiles must decode, so any error, also one Go returns beside an image,
// stops the run.
func ppmDir(dir, out string) error {
	if err := os.MkdirAll(out, 0o755); err != nil {
		return err
	}
	names, err := filepath.Glob(filepath.Join(dir, "*.jpg"))
	if err != nil {
		return err
	}
	sort.Strings(names)
	for _, p := range names {
		src, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		img, err := jpeg.Decode(context.Background(), src, &jpeg.DecodeOptions{Format: jpeg.FormatRGBA, Limits: ppmLim})
		if err != nil {
			return fmt.Errorf("%s: %w", p, err)
		}
		m, ok := img.(*image.RGBA)
		if !ok {
			return fmt.Errorf("%s: decoded %T, want *image.RGBA", p, img)
		}
		w, h := m.Rect.Dx(), m.Rect.Dy()
		buf := fmt.Appendf(make([]byte, 0, 32+3*w*h), "P6\n%d %d\n255\n", w, h)
		for y := range h {
			row := m.Pix[y*m.Stride:]
			for x := range w {
				buf = append(buf, row[4*x:4*x+3]...)
			}
		}
		name := strings.TrimSuffix(filepath.Base(p), ".jpg") + ".ppm"
		if err := os.WriteFile(filepath.Join(out, name), buf, 0o644); err != nil {
			return err
		}
	}
	fmt.Printf("%d tiles decoded\n", len(names))
	return nil
}
