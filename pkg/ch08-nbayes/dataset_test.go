package ch08

import (
	"testing"
)

func TestNewDataSet(t *testing.T) {
	ds := NewDataSet("data/synreviews.csv")
	exp := 22
	if exp != ds.Size {
		t.Errorf("Expected size %d, got %d instead", exp, ds.Size)
	}
}

func TestDataSplit(t *testing.T) {
	ds := NewDataSet("data/synreviews.csv")
	train, test := ds.Split(0.1)
	exp := 20
	if train.Size != exp {
		t.Errorf("Expected %d, got %d", exp, train.Size)
	}
	exp = 2
	if test.Size != exp {
		t.Errorf("Expected %d, got %d", exp, test.Size)
	}
}
