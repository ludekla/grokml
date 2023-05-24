package ch08

import (
	"testing"

	"grokml/pkg/utils"
)

func TestNaiveBayesPredict(t *testing.T) {
	csv := utils.NewCSVReader("../../data/synreviews.csv", []string{"sentiment", "review"})
	ds := utils.NewDataSet[string](csv, utils.ToStr)
	nb := NewNaiveBayes(0.5)
	trainSet, testSet := ds.Split(0.2)
	nb.Fit(trainSet)
	labels := ds.Y()
	for i, dpoint := range testSet.X() {
		isSpam := nb.Predict(dpoint)
		if isSpam != (labels[i] == 1.0) {
			t.Errorf("expected %v (spam), got %v", labels[i] == 1.0, isSpam)
		}
	}
	rep := nb.Score(testSet)
	if rep.Accuracy < 1.0 {
		t.Errorf("expected accuracy of 1.0, got %.3f", rep.Accuracy)
	}
}
