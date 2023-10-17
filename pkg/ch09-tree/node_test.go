package ch09

import (
	"testing"
)

func TestNode(t *testing.T) {
	examples := []Example{
		{[]float64{18, 12000, 1}, 0.91},
		{[]float64{32, 30000, 0}, 0.23},
		{[]float64{21, 32000, 1}, 0.12},
	}
	nd := NewNode(0, 0.1)
	nd.Fit(examples, Gini)
	exp := 0.910
	if nd.Left.Label != exp {
		t.Errorf("Expected label %.3f, got %.3f", exp, nd.Left.Label)
	}
	exp = 0.175
	if nd.Right.Label != exp {
		t.Errorf("Expected label %.3f, got %.3f", exp, nd.Right.Label)
	}
}
