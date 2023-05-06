package utils

import (
	"math"
	"testing"
)

func TestStats(t *testing.T) {
	ds := DataSet{
		x: []Vector{
			Vector{1.0, 3.5, -1.0},
			Vector{0.1, 1.0, 2.4},
		},
		y:    []float64{2.1, 3.2},
		size: 2,
	}
	expDS := DataStats{
		XMean: Vector{0.55, 2.25, 0.7},
		XStd:  Vector{0.45, 1.25, 1.7},
		YMean: 2.65,
		YStd:  0.55,
	}

	stats := ds.Stats()
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
	ds := DataSet{
		x: []Vector{
			Vector{1.0, 3.5, -1.0},
			Vector{0.1, 1.0, 2.4},
		},
		y:    []float64{2.1, 3.2},
		size: 2,
	}
	expDS := DataSet{
		x: []Vector{
			Vector{1.0, 1.0, -1.0},
			Vector{-1.0, -1.0, 1.0},
		},
		y:    []float64{-1.0, 1.0},
		size: 2,
	}
	stats := ds.Stats()
	nDS := ds.Normalise(stats)
	for i, vec := range nDS.x {
		if !equal(expDS.x[i], vec) {
			t.Errorf("Expected %v, got %v", expDS.x[i], vec)
		}
		if math.Abs(expDS.y[i]-nDS.y[i]) > 1e-6 {
			t.Errorf("Expected %v, got %v", expDS.y[i], nDS.y[i])
		}
	}
}
