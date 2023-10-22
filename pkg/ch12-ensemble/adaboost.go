package ch12

import (
	"encoding/json"
	"math"

	"grokml/pkg/ch09-tree"
	pl "grokml/pkg/pipeline"
)

// AdaBoostClassifier implements a forest classifier with AdaBoost.
type AdaBoostClassifier struct {
	ch09.Forest
	Coeffs []float64 `json:"coeffs"`
}

// NewAdaBoostClassifier is the constructor function for AdaBoostClassifier.
// Parameters are the same as for NewForestClassifier.
func NewAdaBoostClassifier(nTrees int, imp ch09.Impurity, ming float64) *AdaBoostClassifier {
	trees := make([]*ch09.TreeClassifier, nTrees)
	for i, _ := range trees {
		clf := ch09.NewTreeClassifier(imp, ming)
		trees[i] = &clf
	}
	return &AdaBoostClassifier{ch09.Forest{Size: nTrees, Estimators: trees}, nil}
}

// Fit implements the training of the tree classifiers where the coefficients
// are assigned a value known as log-odds which is computed based on the
// accuracy of the corresponding tree on the training set.
func (ad *AdaBoostClassifier) Fit(dpoints [][]float64, labels []float64) {
	ad.Coeffs = make([]float64, ad.Size)
	for i, tree := range ad.Estimators {
		tree.Fit(dpoints, labels)
		acc := tree.Score(dpoints, labels)
		// make acc reasonable to protect the log
		if acc > 0.9999 {
			acc = 0.9999
		} else if acc < 0.0001 {
			acc = 0.0001
		}
		ad.Coeffs[i] = math.Log(acc / (1.0 - acc))
	}
}

// Predict maps the predictions of each tree classifier into the interval
// [-1, 1] and computes their linear combination weighted with the
// associated log-odds.
func (ad *AdaBoostClassifier) Predict(dpoints [][]float64) []float64 {
	preds := make([]float64, len(dpoints))
	for i, tree := range ad.Estimators {
		for j, pred := range tree.Predict(dpoints) {
			preds[j] += ad.Coeffs[i] * (2*pred - 1.0)
		}
	}
	for j, pred := range preds {
		if pred < 0.0 {
			preds[j] = 0.0
		} else {
			preds[j] = 1.0
		}
	}
	return preds
}

// Score implements the Estimator interface and additionally computes the
// quantities of a Report struct.
func (ad *AdaBoostClassifier) Score(dpoints [][]float64, labels []float64) float64 {
	predictions := ad.Predict(dpoints)
	ad.Report = pl.GetReport(predictions, labels)
	return ad.Report.Accuracy
}

// Marshal and Unmarshal implement the JSONable interface (pkg: persist).
func (ad AdaBoostClassifier) Marshal() ([]byte, error) {
	return json.MarshalIndent(ad, "", "    ")
}

func (ad *AdaBoostClassifier) Unmarshal(bs []byte) error {
	return json.Unmarshal(bs, ad)
}
