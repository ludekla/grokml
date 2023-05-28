package ch08

import (
	"testing"

	ds "grokml/pkg/dataset"
	tk "grokml/pkg/tokens"
)

func TestNaiveBayesPredict(t *testing.T) {
	csv := ds.NewCSVReader("../../data/synreviews.csv", "spam", "text")
	ds := ds.NewDataSet[string](csv, ds.AtoA)
	nb := NewNaiveBayes(0.5)
	trainSet, testSet := ds.Split(0.2)
	// Tokenise.
	tokeniser := tk.NewTokeniser(true)
	tmaps := tokeniser.Transform(trainSet.DPoints())
	// Learn.
	labels := trainSet.Labels()
	nb.Fit(tmaps, labels)
	preds := nb.Predict(tmaps)
	for i, spam := range preds {
		if spam != (labels[i] == 1.0) {
			t.Errorf("expected %v (spam), got %v", labels[i] == 1.0, spam)
		}
	}
	// Test set.
	tmaps = tokeniser.Transform(testSet.DPoints())
	score := nb.Score(tmaps, testSet.Labels())
	if score < 1.0 {
		t.Errorf("expected accuracy of 1.0, got %.3f", score)
	}
	if score != nb.report.Accuracy {
		t.Errorf("reported accuracy %.2f != %.2f score", nb.report.Accuracy, score)
	}
}
