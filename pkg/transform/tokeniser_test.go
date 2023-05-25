package utils

import (
	"testing"
)

func TestTokenise(t *testing.T) {
	texts := []string{
		"I liked the film very much, it made me happy",
		"The film was awful, not awesome",
		"The film was awful, and not awesome",
	}
	tmaps := Tokenise(texts, false)
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
}

func TestGetTokenFreqs(t *testing.T) {
	txt := "Lutz hat Geburtstag heute heute"
	exp := TokenMap{"Lutz": 0.2, "hat": 0.2, "Geburtstag": 0.2, "heute": 0.4}
	got := GetTokenFreqs(txt, false)
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
