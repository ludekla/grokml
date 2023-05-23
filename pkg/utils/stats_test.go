package utils

import (
	"testing"
)

func TestVectorStats(t *testing.T) {
	vecs := []Vector{
		Vector{1.0, 3.5, -1.0},
		Vector{0.1, 1.0, 2.4},
	}
	expMean := Vector{0.55, 2.25, 0.7}
	expStd := Vector{0.45, 1.25, 1.7}
	mean, std := VectorStats(vecs)
	if !equal(expMean, mean) {
		t.Errorf("Expected %v, got %v", expMean, mean)
	}
	if !equal(expStd, std) {
		t.Errorf("Expected %v, got %v", expMean, mean)
	}
}

func TestMean(t *testing.T) {
	vecs := []Vector{
		Vector{1.0, 3.5, -1.0},
		Vector{0.1, 1.0, 2.4},
	}
	got, exp := Mean(vecs), Vector{0.55, 2.25, 0.7}
	if !equal(got, exp) {
		t.Errorf("Expected %v, got %v", exp, got)
	}
}
