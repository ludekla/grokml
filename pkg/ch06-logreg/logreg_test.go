package ch06

import (
	"math"
	"testing"

	ds "grokml/pkg/dataset"
	tk "grokml/pkg/tokens"
)

func TestLogReg(t *testing.T) {
	lr := NewLogReg[tk.TokenMap](new(TokenMapUpdater), 20, 0.7)
	csv := ds.NewCSVReader("../../data/reviews.csv", "sentiment", "review")
	dset := ds.NewDataSet[string](csv, ds.AtoA)
	trainSet, testSet := dset.Split(0.2)
	// Transform strings into token maps.
	tokeniser := tk.NewTokeniser(false)
	dpoints := tokeniser.Transform(trainSet.DPoints())
	// Learn.
	lr.Fit(dpoints, trainSet.Labels())
	// Score on training set
	trainScore := lr.Score(dpoints, trainSet.Labels())
	// Compute score on test set
	dpoints = tokeniser.Transform(testSet.DPoints())
	got := trainScore + lr.Score(dpoints, testSet.Labels())
	exp1, exp2 := 1.944, 1.75
	if math.Abs(got-exp1) > 1e-3 && math.Abs(got-exp2) > 1e-6 {
		t.Errorf("expected %v or %v, got %v", exp1, exp2, got)
	}
}
