package tokens

import (
	"testing"
)

func TestTokenMap(t *testing.T) {

}

func TestTokenDot(t *testing.T) {

}

func TestTokenIAdd(t *testing.T) {

}

func TestTokenScaMul(t *testing.T) {

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
	tokeniser.toLower = true
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
	got := TokenFreqs(txt, tokeniser.toLower)
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