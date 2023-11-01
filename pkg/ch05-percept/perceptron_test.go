package ch05

import (
	"math"
	"math/rand"
	"testing"

	ds "grokml/pkg/dataset"
	tk "grokml/pkg/tokens"
)

func TestPerceptron(t *testing.T) {
	rand.Seed(0)
	pc := NewTextPerceptron(20, 0.7)
	csv := ds.NewCSVReader("../../data/reviews.csv", "sentiment", "review")
	dset := ds.NewDataSet[string](csv, ds.AtoA)
	trainSet, testSet := dset.Split(0.2)
	// Transform strings into token maps.
	tokeniser := tk.NewTokeniser(false)
	dpoints := tokeniser.Transform(trainSet.DPoints())
	// Learn.
	pc.Fit(dpoints, trainSet.Labels())
	// Score on training set
	trainScore := pc.Score(dpoints, trainSet.Labels())
	// Compute score on test set
	dpoints = tokeniser.Transform(testSet.DPoints())
	got := trainScore + pc.Score(dpoints, testSet.Labels())
	exp1, exp2 := 1.944, 1.75
	if math.Abs(got-exp1) > 1e-3 && math.Abs(got-exp2) > 1e-6 {
		t.Errorf("expected %v or %v, got %v", exp1, exp2, got)
	}
}
