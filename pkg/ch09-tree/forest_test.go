package ch09

import (
	"math"
	"math/rand"
	"testing"

	"grokml/pkg/persist"
)

func init() {
	rand.Seed(1)
}

func TestForestClassifier(t *testing.T) {
	dpoints := [][]float64{
		{7, 1}, {3, 2}, {2, 3}, {1, 5}, {2, 6}, {4, 7},
		{1, 9}, {8, 10}, {6, 5}, {7, 8}, {8, 4}, {9, 6},
	}
	labels := []float64{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1}

	fc := NewForestClassifier(3, Entropy, 0.1)
	fc.Fit(dpoints, labels)
	fc.Score(dpoints, labels)
	rep := fc.Report
	exp := 1.0
	if math.Abs(rep.FScore(1.0)-exp) > 1e-5 {
		t.Errorf("expected F-score %.7f, got %.7f", exp, rep.FScore(1.0))
	}
	persist.Dump(fc, "../../models/ch09-tree/forest.json")

	fc2 := &ForestClassifier{}
	persist.Load(fc2, "../../models/ch09-tree/forest.json")
	fc2.Score(dpoints, labels)
	rep = fc2.Report
	exp = 1.0
	if math.Abs(rep.FScore(1.0)-exp) > 1e-5 {
		t.Errorf("expected F-score %.7f, got %.7f", exp, rep.FScore(1.0))
	}
}
