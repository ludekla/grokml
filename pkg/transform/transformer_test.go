package transform

import (
	"testing"

	"grokml/pkg/vector"
)

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
	var str2tm Transformer[string, TokenMap] = NewTokeniser(true)
	tmaps := str2tm.Transform(texts)
	if len(texts) != len(tmaps) {
		t.Errorf("Expected %d token maps, got %d", len(texts), len(tmaps))
	}
}

// Second kind of transformer 
func TestTransformerII(t *testing.T) {
	// Vectoriser wraps data points
	var f2vec Transformer[float64, vector.Vector] = NewVectoriser(true)
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
	f2vec = NewVectoriser(false)
	vecs = f2vec.Transform(dpoints)
	vecs[0].IScaMul(0.0)
	exp = 1.5
	if len(vecs) != len(dpoints) {
		t.Errorf("Expected %d vectors, got %d", len(dpoints), len(vecs))
	} else if sum(dpoints[0]) == 0.0 {
		t.Errorf("Expected sum to be %f, got %f", exp, sum(dpoints[0]))
	}
}

func TestTokeniserTransform(t *testing.T) {
	texts := [][]string{
		{"I liked the film very much, it made me happy"},
		{"The film was awful, not awesome"},
		{"The film was awful, and not awesome"},
	}
	// Now character lowering
	tokeniser := NewTokeniser(false)
	tmaps := tokeniser.Transform(texts)
	expSize := 3
	if len(tmaps) != expSize {
		t.Errorf("Expected %d token maps, got %d", expSize, len(tmaps))
	}
	sizes := []int{10, 6, 7}
	for i, tmap := range tmaps {
		if len(tmap) != sizes[i] {
			t.Errorf("Expected %d tokens, got %d", sizes[i], len(tmap))
		}
	}
	tokeniser.ToLower = true
	texts = [][]string{{"The cat and", "the mouse."}, {"the cat and the mice"}}
	tmaps = tokeniser.Transform(texts)
	if len(tmaps) != len(texts) {
		t.Errorf("Expected %d token maps, got %d", len(texts), len(tmaps))
	} else if len(tmaps[0]) != 4 {
		t.Errorf("Expected 4 tokens, got %d", len(tmaps[0]))
	} else if len(tmaps[1]) != 4 {
		t.Errorf("Expected 4 tokens, got %d", len(tmaps[1]))
	}
}

func TestTokeniserTokenFreqs(t *testing.T) {
	txt := "Lutz hat Geburtstag heute heute"
	exp := TokenMap{"Lutz": 0.2, "hat": 0.2, "Geburtstag": 0.2, "heute": 0.4}
	tokeniser := NewTokeniser(false)
	got := tokeniser.TokenFreqs(txt)
	if len(exp) != len(got) {
		t.Errorf("word counts do not have the same number of keys")
	}
	for k, v := range exp {
		if val, ok := got[k]; !ok {
			t.Errorf("expected token %q not found in word frequencies", k)
		} else if val != v {
			t.Errorf("expected count %v got %v", v, val)
		}
	}
}
