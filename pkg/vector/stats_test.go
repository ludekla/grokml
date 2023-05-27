package vector

import (
	"math"
	"testing"
)

func TestStats(t *testing.T) {
	vecs := []Vector{{1.0, 3.5, -1.0}, {0.1, 1.0, 2.4}}
	labels := []float64{2.1, 3.2}
	expDS := DataStats{
		XMean: Vector{0.55, 2.25, 0.7},
		XStd:  Vector{0.45, 1.25, 1.7},
		YMean: 2.65,
		YStd:  0.55,
	}
	stats := GetDataStats(vecs, labels)
	if !equal(expDS.XMean, stats.XMean) {
		t.Errorf("Expected %v, got %v", expDS.XMean, stats.XMean)
	}
	if !equal(expDS.XStd, stats.XStd) {
		t.Errorf("Expected %v, got %v", expDS.XMean, stats.XMean)
	}
	if math.Abs(expDS.YMean-stats.YMean) > 1e-6 {
		t.Errorf("Expected %v, got %v", expDS.YMean, stats.YMean)
	}
	if math.Abs(expDS.YStd-stats.YStd) > 1e-6 {
		t.Errorf("Expected %v, got %v", expDS.YStd, stats.YStd)
	}
}

func TestNormalise(t *testing.T) {
	vecs := []Vector{Vector{1.0, 3.5, -1.0}, Vector{0.1, 1.0, 2.4}}
	labels := []float64{2.1, 3.2}
	// Compute statistics
	stats := GetDataStats(vecs, labels)
	// Normalise data
	outvecs, outlabels := stats.Normalise(vecs, labels)
	// expected values
	expVecs := []Vector{Vector{1.0, 1.0, -1.0}, Vector{-1.0, -1.0, 1.0}}
	expLabels := []float64{-1.0, 1.0}
	for i, vec := range outvecs {
		if !equal(expVecs[i], vec) {
			t.Errorf("Expected %v, got %v", expVecs[i], vec)
		}
		if math.Abs(expLabels[i]-outlabels[i]) > 1e-6 {
			t.Errorf("Expected %v, got %v", expLabels[i], outlabels[i])
		}
	}
}
