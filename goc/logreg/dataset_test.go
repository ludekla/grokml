package logreg

import (
	// "fmt"
	"testing"
)

func TestNewDataSet(t *testing.T) {
	path := "test.csv"
	ds := NewDataSet(path)
	expSize := 22
	if ds.Size != expSize {
		t.Errorf("expected %d to be the dataset size, got %d instead", expSize, ds.Size)
	}
	expPath := "test.csv"
	if ds.Path != expPath {
		t.Errorf("expected %s as path, got %s instead", expPath, ds.Path)
	}
	var count int
	size := 10
	for i, tokens := range ds.X {
		if ds.Y[i] == "positive" {
			count++
			if len(tokens) != size {
				t.Errorf("expected %d tokens, got %d intead", size, len(tokens))
			}
		}
	}
	expCount := 12
	if count != expCount {
		t.Errorf("expected %d to be the dataset size, got %d instead", expCount, count)
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

func TestDataSetSplit(t *testing.T) {
	path := "test.csv"
	ds := NewDataSet(path)
	exp := 22
	if ds.Size != exp {
		t.Errorf("expected %d to be the dataset size, got %d instead", exp, ds.Size)
	}
	train, test := ds.Split(0.1)
	exp = 20
	if train.Size != exp {
		t.Errorf("Expected %d, got %d", exp, train.Size)
	}
	exp = 2
	if test.Size != exp {
		t.Errorf("Expected %d, got %d", exp, test.Size)
	}
}