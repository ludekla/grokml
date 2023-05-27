package vector

import (
	"math"
	"testing"
)

// Helper function to decide whether two vectors are the same.
func equal(lvec, rvec Vector) bool {
	for i, val := range lvec {
		if math.Abs(val-rvec[i]) > 1e-6 {
			return false
		}
	}
	return true
}

func TestDot(t *testing.T) {
	v := Vector{1.0, 3.5, 1.0}
	w := Vector{0.1, 1.0, 2.4}
	got, exp := v.Dot(w), 6.0
	if got != exp {
		t.Errorf("Expected %v, got %v", exp, got)
	}
}

func TestScalMul(t *testing.T) {
	v := Vector{1.0, 3.5, 1.0}
	fs := []float64{2.0, -20.0, 0.0}
	exps := []Vector{{2.0, 7.0, 2.0}, {-20.0, -70.0, -20.0}, NewVector(3)}
	for i, f := range fs {
		got, exp := v.ScaMul(f), exps[i]
		if !equal(got, exp) {
			t.Errorf("Expected %v, got %v", exp, got)
		}
	}
}

func TestIScalMul(t *testing.T) {
	v := Vector{1.0, 3.5, 1.0}
	fs := []float64{2.0, -3.0, 0.0}
	exps := []Vector{{2.0, 7.0, 2.0}, {-6.0, -21.0, -6.0}, NewVector(3)}
	for i, f := range fs {
		v.IScaMul(f)
		exp := exps[i]
		if !equal(v, exp) {
			t.Errorf("Expected %v, got %v", exp, v)
		}
	}
}

func TestAdd(t *testing.T) {
	v := Vector{1.0, 3.5, -1.0}
	w := Vector{0.1, 1.0, 2.4}
	got, exp := v.Add(w), Vector{1.1, 4.5, 1.4}
	if !equal(got, exp) {
		t.Errorf("Expected %v, got %v", exp, got)
	}
}

func TestIAdd(t *testing.T) {
	v := Vector{1.0, 3.5, -1.0}
	w := Vector{0.1, 1.0, 2.4}
	exp := Vector{1.1, 4.5, 1.4}
	v.IAdd(w)
	if !equal(v, exp) {
		t.Errorf("Expected %v, got %v", exp, v)
	}
	exp = Vector{0.1, 1.0, 2.4}
	if !equal(w, exp) {
		t.Errorf("Expected %v, got %v", exp, w)
	}
}

func TestMul(t *testing.T) {
	v := Vector{1.0, 3.5, -1.0}
	w := Vector{0.1, 1.0, 2.4}
	got, exp := v.Mul(w), Vector{0.1, 3.5, -2.4}
	if !equal(got, exp) {
		t.Errorf("Expected %v, got %v", exp, got)
	}
}

func TestDiv(t *testing.T) {
	v := Vector{1.2, -0.6, -8.4}
	w := Vector{2.0, 3.0, -4.0}
	exp := Vector{0.6, -0.2, 2.1}
	res := v.Div(w)
	if !equal(res, exp) {
		t.Errorf("Expected %v, got %v", exp, res)
	}
}

func TestIDiv(t *testing.T) {
	v := Vector{1.2, -0.6, -8.4}
	w := Vector{2.0, 3.0, -4.0}
	exp := Vector{0.6, -0.2, 2.1}
	v.IDiv(w)
	if !equal(v, exp) {
		t.Errorf("Expected %v, got %v", exp, v)
	}
}

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