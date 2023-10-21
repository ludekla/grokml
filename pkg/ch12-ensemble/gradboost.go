package ch12

import (
	"encoding/json"

	"grokml/pkg/ch09-tree"
	pl "grokml/pkg/pipeline"
)

// GradBoostClassifier implements a forest classifier with Gradient Boost.
type GradBoostRegressor struct {
	Size       int              `json:"size"`
	Regressors []*ch09.TreeRegressor `json:"trees"`
	lRate      float64          `json:"-"`
}

// NewGradBoostClassifier is the constructor function for GradBoostClassifier.
// Parameters are the same as for NewForestClassifier.
func NewGradBoostRegressor(nTrees int, ming float64, lrate float64) *GradBoostRegressor {
	trees := make([]*ch09.TreeRegressor, nTrees)
	for i, _ := range trees {
		reg := ch09.NewTreeRegressor(ming)
		trees[i] = &reg
	}
	return &GradBoostRegressor{Size: nTrees, Regressors: trees, lRate: lrate}
}

// Fit implements the training of the gradient boosting algorithm for a forest of
// tree classifiers. Except for the first tree each tree is trained on the error of
// its predecessor so that the sum of their predcitions is a better prediction.
func (gb *GradBoostRegressor) Fit(dpoints [][]float64, labels []float64) {
	clabels := make([]float64, len(labels))
	copy(clabels, labels)
	for _, tree := range gb.Regressors {
		tree.Fit(dpoints, clabels)
		for j, pred := range tree.Predict(dpoints) {
			clabels[j] -= pred
		}
	}
}

// Predict performs the inference for the given data points by computing a linear
// combination of their predictions.
func (gb *GradBoostRegressor) Predict(dpoints [][]float64) []float64 {
	preds := gb.Regressors[0].Predict(dpoints)
	for _, tree := range gb.Regressors[1:] {
		for i, pred := range tree.Predict(dpoints) {
			preds[i] += gb.lRate * pred
		}
	}
	return preds
}

// Score computes the coefficient of determination.
func (gb *GradBoostRegressor) Score(dpoints [][]float64, labels []float64) float64 {
	preds := gb.Predict(dpoints)
	return pl.GetCoD(preds, labels)
}

// Marshal and Unmarshal implement the JSONable interface (pkg: persist).
func (gb GradBoostRegressor) Marshal() ([]byte, error) {
	return json.MarshalIndent(gb, "", "    ")
}

func (gb *GradBoostRegressor) Unmarshal(bs []byte) error {
	return json.Unmarshal(bs, gb)
}