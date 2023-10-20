package ch09

import (
	"math"
	"testing"

	"grokml/pkg/persist"
)

func TestTreeEntropy(t *testing.T) {
	dpoints := [][]float64{
		{7, 1}, {3, 2}, {2, 3}, {1, 5}, {2, 6}, {4, 7},
		{1, 9}, {8, 10}, {6, 5}, {7, 8}, {8, 4}, {9, 6},
	}
	labels := []float64{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1}

	dt := NewTreeClassifier(Entropy, 0.1)
	dt.Fit(dpoints, labels)
	dt.Score(dpoints, labels)
	rep := dt.Report
	exp := 0.9230769
	if math.Abs(rep.FScore(1.0)-exp) > 1e-5 {
		t.Errorf("expected F-score %.7f, got %.7f", exp, rep.FScore(1.0))
	}
	persist.Dump(&dt, "../../models/ch09-tree/entree.json")

	dt2 := &TreeClassifier{}
	persist.Load(dt2, "../../models/ch09-tree/entree.json")
	dt2.Score(dpoints, labels)
	rep = dt2.Report
	if math.Abs(rep.FScore(1.0)-exp) > 1e-5 {
		t.Errorf("expected F-score %.7f, got %.7f", exp, rep.FScore(1.0))
	}
}

func TestTreeGini(t *testing.T) {
	dpoints := [][]float64{
		{7, 1}, {3, 2}, {2, 3}, {1, 5}, {2, 6}, {4, 7},
		{1, 9}, {8, 10}, {6, 5}, {7, 8}, {8, 4}, {9, 6},
	}
	labels := []float64{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1}

	dt := NewTreeClassifier(Gini, 0.1)
	dt.Fit(dpoints, labels)
	dt.Score(dpoints, labels)
	rep := dt.Report
	exp := 0.9230769
	if math.Abs(rep.FScore(1.0)-exp) > 1e-5 {
		t.Errorf("expected F-score %.7f, got %.7f", exp, rep.FScore(1.0))
	}
	persist.Dump(&dt, "../../models/ch09-tree/ginitree.json")

	dt2 := TreeClassifier{}
	persist.Load(&dt2, "../../models/ch09-tree/ginitree.json")
	dt2.Score(dpoints, labels)
	rep = dt2.Report
	if math.Abs(rep.FScore(1.0)-exp) > 1e-5 {
		t.Errorf("expected F-score %.7f, got %.7f", exp, rep.FScore(1.0))
	}
}

func TestTreeMSE(t *testing.T) {
	dpoints := [][]float64{{10}, {20}, {30}, {40}, {50}, {60}, {70}, {80}, {86}}
	labels := []float64{7, 5, 7, 1, 2, 1, 5, 4, 3.6}

	dt := NewTreeRegressor(0.1)
	dt.Fit(dpoints, labels)
	got := dt.Score(dpoints, labels)
	exp := 0.9822822
	if math.Abs(got-exp) > 1e-5 {
		t.Errorf("expected R2 score %.7f, got %.7f", exp, got)
	}
	persist.Dump(&dt, "../../models/ch09-tree/msetree.json")

	dt2 := TreeRegressor{}
	persist.Load(&dt2, "../../models/ch09-tree/msetree.json")
	got = dt2.Score(dpoints, labels)
	if math.Abs(got-exp) > 1e-5 {
		t.Errorf("expected R2 score %.7f, got %.7f", exp, got)
	}
}
