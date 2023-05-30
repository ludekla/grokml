package ch09

import (
	// "fmt"
	"math"
	"testing"
)

func TestForestClassifier(t *testing.T) {
	trainSet := DataSet{
		Size:   12,
		Header: []string{"x0", "x1", "y"},
		Examples: []Example{
			{[]float64{7, 1}, 0}, {[]float64{3, 2}, 0}, {[]float64{2, 3}, 0},
			{[]float64{1, 5}, 0}, {[]float64{2, 6}, 0}, {[]float64{4, 7}, 0},
			{[]float64{1, 9}, 1}, {[]float64{8, 10}, 1}, {[]float64{6, 5}, 1},
			{[]float64{7, 8}, 1}, {[]float64{8, 4}, 1}, {[]float64{9, 6}, 1},
		},
	}
	fc := NewForestClassifier(3, NewEntropy(0.5), 0.1)
	fc.Fit(trainSet)
	rep := fc.Score(trainSet)
	exp := 1.0
	if math.Abs(rep.FScore(1.0)-exp) > 1e-5 {
		t.Errorf("expected F-score %.7f, got %.7f", exp, rep.FScore(1.0))
	}
	fc.Save("../../models/forest.json")

	fc2 := &ForestClassifier{}
	fc2.Load("../../models/forest.json")
	rep = fc2.Score(trainSet)
	exp = 1.0
	if math.Abs(rep.FScore(1.0)-exp) > 1e-5 {
		t.Errorf("expected F-score %.7f, got %.7f", exp, rep.FScore(1.0))
	}
}

func TestAdaBoostClassifier(t *testing.T) {
	trainSet := DataSet{
		Size:   12,
		Header: []string{"x0", "x1", "y"},
		Examples: []Example{
			{[]float64{7, 1}, 0}, {[]float64{3, 2}, 0}, {[]float64{2, 3}, 0},
			{[]float64{1, 5}, 0}, {[]float64{2, 6}, 0}, {[]float64{4, 7}, 0},
			{[]float64{1, 9}, 1}, {[]float64{8, 10}, 1}, {[]float64{6, 5}, 1},
			{[]float64{7, 8}, 1}, {[]float64{8, 4}, 1}, {[]float64{9, 6}, 1},
		},
	}
	ac := NewAdaBoostClassifier(3, NewEntropy(0.5), 0.1)
	ac.Fit(trainSet)
	rep := ac.Score(trainSet)
	exp := 0.9230769
	if math.Abs(rep.FScore(1.0)-exp) > 1e-5 {
		t.Errorf("expected F-score %.7f, got %.7f", exp, rep.FScore(1.0))
	}
	ac.Save("../../models/adaBoost.json")

	ac2 := &AdaBoostClassifier{}
	ac2.Load("../../models/adaBoost.json")
	rep = ac2.Score(trainSet)
	exp = 0.9230769
	if math.Abs(rep.FScore(1.0)-exp) > 1e-5 {
		t.Errorf("expected F-score %.7f, got %.7f", exp, rep.FScore(1.0))
	}
}

func TestGradBoostRegressor(t *testing.T) {
	trainSet := DataSet{
		Size:   9,
		Header: []string{"Age", "Engagement"},
		Examples: []Example{
			{[]float64{10}, 7}, {[]float64{20}, 5}, {[]float64{30}, 7},
			{[]float64{40}, 1}, {[]float64{50}, 2}, {[]float64{60}, 1},
			{[]float64{70}, 5}, {[]float64{80}, 4}, {[]float64{86}, 3.6},
		},
	}
	gb := NewGradBoostRegressor(2, 0.1, 0.8)
	gb.Fit(trainSet)
	got := gb.Score(trainSet)
	exp := 0.9822822
	if math.Abs(got-exp) > 1e-5 {
		t.Errorf("expected R2 score %.7f, got %.7f", exp, got)
	}
	gb.Save("../../models/gradboost.json")

	gb2 := GradBoostRegressor{}
	gb2.Load("../../models/gradboost.json")
	got = gb2.Score(trainSet)
	if math.Abs(got-exp) > 1e-5 {
		t.Errorf("expected R2 score %.7f, got %.7f", exp, got)
	}
}