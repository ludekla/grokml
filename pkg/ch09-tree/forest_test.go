package ch09

import (
	"math"
	"math/rand"
	"testing"
)

func init() {
	rand.Seed(1)
}

func TestForestClassifier(t *testing.T) {
	dpoints := [][]float64{
		{7, 1}, {3, 2}, {2, 3}, {1, 5}, {2, 6}, {4, 7}, 
		{1, 9}, {8, 10},{6, 5}, {7, 8}, {8, 4}, {9, 6},
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
	fc.Save("../../models/ch09-tree/forest.json")

	fc2 := &ForestClassifier{}
	fc2.Load("../../models/ch09-tree/forest.json")
	fc2.Score(dpoints, labels)
	rep = fc2.Report
	exp = 1.0
	if math.Abs(rep.FScore(1.0)-exp) > 1e-5 {
		t.Errorf("expected F-score %.7f, got %.7f", exp, rep.FScore(1.0))
	}
}

func TestAdaBoostClassifier(t *testing.T) {
	dpoints := [][]float64{
		{7, 1}, {3, 2}, {2, 3}, {1, 5}, {2, 6}, {4, 7}, 
		{1, 9}, {8, 10},{6, 5}, {7, 8}, {8, 4}, {9, 6},
	}
	labels := []float64{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1} 

	ac := NewAdaBoostClassifier(3, Entropy, 0.1)
	ac.Fit(dpoints, labels)
	ac.Score(dpoints, labels)
	rep := ac.Report
	exp := 0.9230769
	if math.Abs(rep.FScore(1.0)-exp) > 1e-5 {
		t.Errorf("expected F-score %.7f, got %.7f", exp, rep.FScore(1.0))
	}
	ac.Save("../../models/ch09-tree/adaBoost.json")

	ac2 := &AdaBoostClassifier{}
	ac2.Load("../../models/ch09-tree/adaBoost.json")
	ac2.Score(dpoints, labels)
	rep = ac2.Report
	exp = 0.9230769
	if math.Abs(rep.FScore(1.0)-exp) > 1e-5 {
		t.Errorf("expected F-score %.7f, got %.7f", exp, rep.FScore(1.0))
	}
}

func TestGradBoostRegressor(t *testing.T) {
	dpoints := [][]float64{{10}, {20}, {30}, {40}, {50}, {60}, {70}, {80}, {86}}
	labels := []float64{7, 5,7, 1, 2, 1, 5, 4, 3.6}

	gb := NewGradBoostRegressor(2, 0.1, 0.8)
	gb.Fit(dpoints, labels)
	got := gb.Score(dpoints, labels)
	exp := 0.9822822
	if math.Abs(got-exp) > 1e-5 {
		t.Errorf("expected R2 score %.7f, got %.7f", exp, got)
	}
	gb.Save("../../models/ch09-tree/gradboost.json")

	gb2 := GradBoostRegressor{}
	gb2.Load("../../models/ch09-tree/gradboost.json")
	got = gb2.Score(dpoints, labels)
	if math.Abs(got-exp) > 1e-5 {
		t.Errorf("expected R2 score %.7f, got %.7f", exp, got)
	}
}