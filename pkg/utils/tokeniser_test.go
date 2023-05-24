package utils

import (
	"testing"
)

func TestTokeniser(t *testing.T) {
	path := "../../data/reviews.csv"
	csv := NewCSVReader(path, []string{"sentiment", "review"})
	dsraw := NewDataSet[string](csv, ToStr)
	ds := Transform(dsraw)
	expSize := 22
	if ds.Size() != expSize {
		t.Errorf("expected dataset size %d, got %d instead", expSize, ds.Size())
	}
	var count int
	size := 10
	for i, dp := range ds.dpoints {
		if ds.labels[i] == 1.0 {
			count++
			if len(dp) != size {
				t.Errorf("expected %d tokens, got %d instead", size, len(dp))
			}
		}
	}
	expCount := 12
	if count != expCount {
		t.Errorf("expected dataset size %d, got %d instead", expCount, count)
	}
}

func TestGetTokenFreqs(t *testing.T) {
	txt := "Lutz hat Geburtstag heute heute"
	exp := TokenMap{"Lutz": 0.2, "hat": 0.2, "Geburtstag": 0.2, "heute": 0.4}
	got := GetTokenFreqs(txt)
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
