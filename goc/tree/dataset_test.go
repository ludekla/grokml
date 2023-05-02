package tree

import (
	"testing"
)

func TestNewDataSet(t *testing.T) {
	ds := NewDataSet("test.csv")
	exp1 := 24
	if ds.Size != exp1 {
		t.Errorf("Expected %d, got %d", exp1, ds.Size)
	}
	exp2 := []string{"Age", "Salary", "Gender", "Admit"}
	for i, colname := range ds.Header {
		if colname != exp2[i] {
			t.Errorf("expected header column name %s, got %s", exp2[i], colname)
		}
	}
}

func TestSplit(t *testing.T) {
	ds := NewDataSet("test.csv")
	train, test := ds.Split(0.2)
	exp1 := 20
	if train.Size != exp1 {
		t.Errorf("Expected %d, got %d", exp1, train.Size)
	}
	exp2 := []string{"Age", "Salary", "Gender", "Admit"}
	for i, colname := range train.Header {
		if colname != exp2[i] {
			t.Errorf("expected header column name %s, got %s", exp2[i], colname)
		}
	}
	exp1 = 4
	if test.Size != exp1 {
		t.Errorf("Expected %d, got %d", exp1, test.Size)
	}
	for i, colname := range test.Header {
		if colname != exp2[i] {
			t.Errorf("expected header column name %s, got %s", exp2[i], colname)
		}
	}
}
