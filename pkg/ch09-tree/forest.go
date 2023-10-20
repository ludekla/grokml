package ch09

import (
	"encoding/json"
	"math"
	"math/rand"

	pl "grokml/pkg/pipeline"
)

// Forest implements a collection of tree classifiers. Its methods are
// promoted by the structs that embed it.
type Forest struct {
	Size       int               `json:"size"`
	Estimators []*TreeClassifier `json:"trees"`
	Report     pl.Report         `json:"-"`
}

// ForestClassifier is the basic Forest classifier.
type ForestClassifier struct {
	Forest
}

// AdaBoostClassifier implements a forest classifier with AdaBoost.
type AdaBoostClassifier struct {
	Forest
	Coeffs []float64 `json:"coeffs"`
}

// GradBoostClassifier implements a forest classifier with Gradient Boost.
type GradBoostRegressor struct {
	Size       int              `json:"size"`
	Estimators []*TreeRegressor `json:"trees"`
	lRate      float64          `json:"-"`
}

// NewForestClassifier is the constructor function for ForestClassifier.
// Its parameters are the number of trees, the impurity and the minimum gain.
func NewForestClassifier(nTrees int, imp Impurity, ming float64) *ForestClassifier {
	trees := make([]*TreeClassifier, nTrees)
	for i, _ := range trees {
		clf := NewTreeClassifier(imp, ming)
		trees[i] = &clf
	}
	return &ForestClassifier{Forest{Size: nTrees, Estimators: trees}}
}

// Fit implements the training for the trees of the forest.
func (f *Forest) Fit(dpoints [][]float64, labels []float64) {
	chunkSize := int(0.9 * float64(len(dpoints)))
	for _, tree := range f.Estimators {
		tree.Fit(dpoints[:chunkSize], labels[:chunkSize])
		rand.Shuffle(len(labels), func(i, j int) {
			dpoints[i], dpoints[j] = dpoints[j], dpoints[i]
			labels[i], labels[j] = labels[j], labels[i]
		})
	}
}

// Predict polls the trees for their predictions for each data point
// and computes the average of their predicted labels as final prediction.
func (f *Forest) Predict(dpoints [][]float64) []float64 {
	avg := make([]float64, len(dpoints))
	for _, tree := range f.Estimators {
		preds := tree.Predict(dpoints)
		for i, pred := range preds {
			avg[i] += pred
		}
	}
	size := float64(f.Size)
	for i, val := range avg {
		avg[i] = val / size
	}
	return avg
}

// Score implements the Estimator interface and additionally computes the
// quantities of a Report struct.
func (f *Forest) Score(dpoints [][]float64, labels []float64) float64 {
	preds := f.Predict(dpoints)
	f.Report = getReport(preds, labels)
	return f.Report.Accuracy
}

// Marshal and Unmarshal implement the JSONable interface (pkg: persist).
func (f Forest) Marshal() ([]byte, error) {
	return json.MarshalIndent(f, "", "    ")
}

func (f *Forest) Unmarshal(bs []byte) error {
	return json.Unmarshal(bs, f)
}

// NewAdaBoostClassifier is the constructor function for AdaBoostClassifier.
// Parameters are the same as for NewForestClassifier.
func NewAdaBoostClassifier(nTrees int, imp Impurity, ming float64) *AdaBoostClassifier {
	trees := make([]*TreeClassifier, nTrees)
	for i, _ := range trees {
		clf := NewTreeClassifier(imp, ming)
		trees[i] = &clf
	}
	return &AdaBoostClassifier{Forest{Size: nTrees, Estimators: trees}, nil}
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
	return preds
}

// Score implements the Estimator interface and additionally computes the
// quantities of a Report struct.
func (ad *AdaBoostClassifier) Score(dpoints [][]float64, labels []float64) float64 {
	predictions := ad.Predict(dpoints)
	ad.Report = getReport(predictions, labels)
	return ad.Report.Accuracy
}

// Marshal and Unmarshal implement the JSONable interface (pkg: persist).
func (ad AdaBoostClassifier) Marshal() ([]byte, error) {
	return json.MarshalIndent(ad, "", "    ")
}

func (ad *AdaBoostClassifier) Unmarshal(bs []byte) error {
	return json.Unmarshal(bs, ad)
}

// NewGradBoostClassifier is the constructor function for GradBoostClassifier.
// Parameters are the same as for NewForestClassifier.
func NewGradBoostRegressor(nTrees int, ming float64, lrate float64) *GradBoostRegressor {
	trees := make([]*TreeRegressor, nTrees)
	for i, _ := range trees {
		reg := NewTreeRegressor(ming)
		trees[i] = &reg
	}
	return &GradBoostRegressor{Size: nTrees, Estimators: trees, lRate: lrate}
}

// Fit implements the training of the gradient boosting algorithm for a forest of
// tree classifiers. Except for the first tree each tree is trained on the error of
// its predecessor so that the sum of their predcitions is a better prediction.
func (gb *GradBoostRegressor) Fit(dpoints [][]float64, labels []float64) {
	clabels := make([]float64, len(labels))
	copy(clabels, labels)
	for _, tree := range gb.Estimators {
		tree.Fit(dpoints, clabels)
		for j, pred := range tree.Predict(dpoints) {
			clabels[j] -= pred
		}
	}
}

// Predict performs the inference for the given data points by computing a linear
// combination of their predictions.
func (gb *GradBoostRegressor) Predict(dpoints [][]float64) []float64 {
	preds := gb.Estimators[0].Predict(dpoints)
	for _, tree := range gb.Estimators[1:] {
		for i, pred := range tree.Predict(dpoints) {
			preds[i] += gb.lRate * pred
		}
	}
	return preds
}

// Score computes the coefficient of determination.
func (gb *GradBoostRegressor) Score(dpoints [][]float64, labels []float64) float64 {
	preds := gb.Predict(dpoints)
	return getCoD(preds, labels)
}

// Marshal and Unmarshal implement the JSONable interface (pkg: persist).
func (gb GradBoostRegressor) Marshal() ([]byte, error) {
	return json.MarshalIndent(gb, "", "    ")
}

func (gb *GradBoostRegressor) Unmarshal(bs []byte) error {
	return json.Unmarshal(bs, gb)
}
