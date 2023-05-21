package nbayes

import (
	"testing"
)

func TestNewNaiveBayes(t *testing.T) {
	ds := NewDataSet("test.csv")
	nb := NewNaiveBayes(0.5)
	nb.Fit(ds)

}

func TestNaiveBayesPredict(t *testing.T) {
	nb := NewNaiveBayes(0.5)
	ds := NewDataSet("test.csv")
	trainSet, testSet := ds.Split(0.2)
	nb.Fit(trainSet)
	for _, x := range testSet.Data {
		got := nb.Predict(x.Text)
		if x.Spam != got {
			t.Errorf("expected %v (spam), got %v", x.Spam, got)
		}
	}
	rep := nb.Score(testSet)
	if rep.Accuracy < 1.0 {
		t.Errorf("expected accuracy of 1.0, got %.3f", rep.Accuracy)
	}
}
