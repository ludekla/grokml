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
	nd := NewNode(examples, 0)
	nd.Fit(NewImpurity(0.5, Gini), 0.1)
	exp := 0.910
	if nd.Left.Label != exp {
		t.Errorf("Expected label %.3f, got %.3f", exp, nd.Left.Label)
	}
	if len(nd.Left.examples) != 1 {
		t.Errorf("Expected 1 example, got %d", len(nd.Left.examples))
	}
	exp = 0.175
	if nd.Right.Label != exp {
		t.Errorf("Expected label %.3f, got %.3f", exp, nd.Right.Label)
	}
	if len(nd.Right.examples) != 2 {
		t.Errorf("Expected 2 examples, got %d", len(nd.Right.examples))
	}
}

func TestEquals(t *testing.T) {
	n1 := Node{Label: 0.32, Split: SplitInfo{6, 3, 0.66}}
	n2 := Node{Label: 0.32, Split: SplitInfo{1, 3, 0.66}}
	if !n1.Equals(&n2) {
		t.Errorf("Expected %v, got %v", n1, n2)
	}
}
