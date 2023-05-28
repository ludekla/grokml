package dataset

import (
	"testing"
)

// Data must be split correctly into training and test set.
func TestDataSetSplit(t *testing.T) {
	path := "../../data/reviews.csv"
	csv := NewCSVReader(path, "sentiment", "review")
	ds := NewDataSet[string](csv, AtoA)
	exp := 22
	if ds.Size() != exp {
		t.Errorf("expected dataset size %d, got %d instead", exp, ds.Size())
	}
	train, test := ds.Split(0.1)
	exp = 20
	if train.Size() != exp {
		t.Errorf("Expected %d, got %d", exp, train.Size())
	}
	exp = 2
	if test.Size() != exp {
		t.Errorf("Expected %d, got %d", exp, test.Size())
	}
}
