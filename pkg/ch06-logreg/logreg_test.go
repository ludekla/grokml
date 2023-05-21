package ch06

import (
	"math"
	"testing"
)

func TestLogReg(t *testing.T) {
	lr := NewLogReg(20, 0.7)
	path := "../../data/reviews.csv"
	ds := NewDataSet(path)
	trainSet, testSet := ds.Split(0.2)
	lr.Fit(trainSet)
	got := lr.Accuracy(trainSet) + lr.Accuracy(testSet)
	exp1, exp2 := 1.944, 1.75
	if math.Abs(got-exp1) > 1e-3 && math.Abs(got-exp2) > 1e-6 {
		t.Errorf("expected %v or %v, got %v", exp1, exp2, got)
	}
}
