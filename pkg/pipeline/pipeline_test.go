package pipeline

import (
	"testing"

	tk "grokml/pkg/tokens"
	vc "grokml/pkg/vector"
)

// Helper
func sum(dpoint []float64) float64 {
	var sum float64
	for _, val := range dpoint {
		sum += val
	}
	return sum
}

// First kind of transformer
func TestTransformerI(t *testing.T) {
	texts := [][]string{{"name"}, {"age"}, {"profession"}, {"sex"}}
	var str2tm Transformer[string, tk.TokenMap] = tk.NewTokeniser(true)
	tmaps := str2tm.Transform(texts)
	if len(texts) != len(tmaps) {
		t.Errorf("Expected %d token maps, got %d", len(texts), len(tmaps))
	}
}

// Second kind of transformer
func TestTransformerII(t *testing.T) {
	// Vectoriser wraps data points
	var f2vec Transformer[float64, vc.Vector] = vc.NewVectoriser(true)
	dpoints := [][]float64{{1.2, 0.3}, {-6.2, 9.1}}

	vecs := f2vec.Transform(dpoints)
	vecs[0].IScaMul(0.0)
	exp := 0.0
	if len(vecs) != len(dpoints) {
		t.Errorf("Expected %d vectors, got %d", len(dpoints), len(vecs))
	} else if sum(dpoints[0]) != exp {
		t.Errorf("Expected zero sum, got %f", sum(dpoints[0]))
	}
	dpoints[0] = []float64{1.2, 0.3}
	// copy data points
	f2vec = vc.NewVectoriser(false)
	vecs = f2vec.Transform(dpoints)
	vecs[0].IScaMul(0.0)
	exp = 1.5
	if len(vecs) != len(dpoints) {
		t.Errorf("Expected %d vectors, got %d", len(dpoints), len(vecs))
	} else if sum(dpoints[0]) == 0.0 {
		t.Errorf("Expected sum to be %f, got %f", exp, sum(dpoints[0]))
	}
}
