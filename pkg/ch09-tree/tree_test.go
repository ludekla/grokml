package ch09

import (
	// "fmt"
	"math"
	"testing"
)

func TestTreeEntropy(t *testing.T) {
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
	dt := NewTreeClassifier(NewImpurity(0.5, Entropy), 0.1)
	dt.Fit(trainSet)
	rep := dt.Score(trainSet)
	exp := 0.83333
	if math.Abs(rep.FScore(1.0)-exp) > 1e-5 {
		t.Errorf("expected F-score %.7f, got %.7f", exp, rep.FScore(1.0))
	}
	dt.Save("../../models/entree.json")

	dt2 := TreeClassifier{}
	dt2.Load("../../models/entree.json")
	rep = dt2.Score(trainSet)
	if math.Abs(rep.FScore(1.0)-exp) > 1e-5 {
		t.Errorf("expected F-score %.7f, got %.7f", exp, rep.FScore(1.0))
	}
}

func TestTreeGini(t *testing.T) {
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
	dt := NewTreeClassifier(NewImpurity(0.5, Gini), 0.1)
	dt.Fit(trainSet)
	rep := dt.Score(trainSet)
	exp := 0.83333
	if math.Abs(rep.FScore(1.0)-exp) > 1e-5 {
		t.Errorf("expected F-score %.7f, got %.7f", exp, rep.FScore(1.0))
	}
	dt.Save("../../models/ginitree.json")

	dt2 := TreeClassifier{}
	dt2.Load("../../models/ginitree.json")
	rep = dt2.Score(trainSet)
	if math.Abs(rep.FScore(1.0)-exp) > 1e-5 {
		t.Errorf("expected F-score %.7f, got %.7f", exp, rep.FScore(1.0))
	}
}

func TestTreeMSE(t *testing.T) {
	trainSet := DataSet{
		Size:   9,
		Header: []string{"Age", "Engagement"},
		Examples: []Example{
			{[]float64{10}, 7}, {[]float64{20}, 5}, {[]float64{30}, 7},
			{[]float64{40}, 1}, {[]float64{50}, 2}, {[]float64{60}, 1},
			{[]float64{70}, 5}, {[]float64{80}, 4}, {[]float64{86}, 3.6},
		},
	}
	dt := NewTreeRegressor(0.1)
	dt.Fit(trainSet)
	got := dt.Score(trainSet)
	exp := 0.9822822
	if math.Abs(got-exp) > 1e-5 {
		t.Errorf("expected R2 score %.7f, got %.7f", exp, got)
	}
	dt.Save("../../models/msetree.json")

	dt2 := TreeRegressor{}
	dt2.Load("../../models/msetree.json")
	got = dt2.Score(trainSet)
	if math.Abs(got-exp) > 1e-5 {
		t.Errorf("expected R2 score %.7f, got %.7f", exp, got)
	}
	if !dt.Equals(dt2) {
		t.Error("trees not the same as expected")
	}
}
