package ch12

import (
	"math"
	"testing"

	"grokml/pkg/persist"
	"grokml/pkg/ch09-tree"
)

func TestAdaBoostClassifier(t *testing.T) {
	dpoints := [][]float64{
		{7, 1}, {3, 2}, {2, 3}, {1, 5}, {2, 6}, {4, 7},
		{1, 9}, {8, 10}, {6, 5}, {7, 8}, {8, 4}, {9, 6},
	}
	labels := []float64{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1}

	ac := NewAdaBoostClassifier(3, ch09.Entropy, 0.1)
	ac.Fit(dpoints, labels)
	ac.Score(dpoints, labels)
	rep := ac.Report
	exp := 0.9230769
	if math.Abs(rep.FScore(1.0)-exp) > 1e-5 {
		t.Errorf("expected F-score %.7f, got %.7f", exp, rep.FScore(1.0))
		t.Errorf("Report %v", rep)
	}
	persist.Dump(ac, "../../models/ch09-tree/adaBoost.json")

	ac2 := &AdaBoostClassifier{}
	persist.Load(ac2, "../../models/ch09-tree/adaBoost.json")
	ac2.Score(dpoints, labels)
	rep = ac2.Report
	exp = 0.9230769
	if math.Abs(rep.FScore(1.0)-exp) > 1e-5 {
		t.Errorf("expected F-score %.7f, got %.7f", exp, rep.FScore(1.0))
	}
}

func TestGradBoostRegressor(t *testing.T) {
	dpoints := [][]float64{{10}, {20}, {30}, {40}, {50}, {60}, {70}, {80}, {86}}
	labels := []float64{7, 5, 7, 1, 2, 1, 5, 4, 3.6}

	gb := NewGradBoostRegressor(2, 0.1, 0.8)
	gb.Fit(dpoints, labels)
	got := gb.Score(dpoints, labels)
	exp := 0.9822822
	if math.Abs(got-exp) > 1e-5 {
		t.Errorf("expected R2 score %.7f, got %.7f", exp, got)
	}
	persist.Dump(gb, "../../models/ch09-tree/gradboost.json")

	gb2 := &GradBoostRegressor{}
	persist.Load(gb2, "../../models/ch09-tree/gradboost.json")
	got = gb2.Score(dpoints, labels)
	if math.Abs(got-exp) > 1e-5 {
		t.Errorf("expected R2 score %.7f, got %.7f", exp, got)
	}
}
