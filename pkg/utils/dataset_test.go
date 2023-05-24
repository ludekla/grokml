package utils

import (
	"math"
	"testing"
)

func TestStats(t *testing.T) {
	ds := DataSet[Vector]{
		dpoints: []Vector{Vector{1.0, 3.5, -1.0}, Vector{0.1, 1.0, 2.4}},
		labels:  []float64{2.1, 3.2},
		size:    2,
	}
	expDS := DataStats{
		XMean: Vector{0.55, 2.25, 0.7},
		XStd:  Vector{0.45, 1.25, 1.7},
		YMean: 2.65,
		YStd:  0.55,
	}

	stats := NewDataStats(ds)
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
	ds := DataSet[Vector]{
		dpoints: []Vector{Vector{1.0, 3.5, -1.0}, Vector{0.1, 1.0, 2.4}},
		labels:  []float64{2.1, 3.2},
		size:    2,
	}
	expDS := DataSet[Vector]{
		dpoints: []Vector{Vector{1.0, 1.0, -1.0}, Vector{-1.0, -1.0, 1.0}},
		labels:  []float64{-1.0, 1.0},
		size:    2,
	}
	stats := NewDataStats(ds)
	nDS := stats.Normalise(ds)
	for i, vec := range nDS.dpoints {
		if !equal(expDS.dpoints[i], vec) {
			t.Errorf("Expected %v, got %v", expDS.dpoints[i], vec)
		}
		if math.Abs(expDS.labels[i]-nDS.labels[i]) > 1e-6 {
			t.Errorf("Expected %v, got %v", expDS.labels[i], nDS.labels[i])
		}
	}
}

func TestDataSetSplit(t *testing.T) {
	path := "../../data/reviews.csv"
	csv := NewCSVReader(path, []string{"sentiment", "review"})
	ds := NewDataSet[string](csv, ToStr)
	exp := 22
	if ds.Size() != exp {
		t.Errorf("expected dataset size %d, got %d instead", exp, ds.Size())
	}
	train, test := ds.Split(0.1)
	exp = 20
	if train.Size() != exp {
		t.Errorf("Expected %d, got %d", exp, train.Size())
	}
	exp = 2
	if test.Size() != exp {
		t.Errorf("Expected %d, got %d", exp, test.Size())
	}
}
