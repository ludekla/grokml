package ch09

import (
	"encoding/json"
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
	f.Report = pl.GetReport(preds, labels)
	return f.Report.Accuracy
}

// Marshal and Unmarshal implement the JSONable interface (pkg: persist).
func (f Forest) Marshal() ([]byte, error) {
	return json.MarshalIndent(f, "", "    ")
}

func (f *Forest) Unmarshal(bs []byte) error {
	return json.Unmarshal(bs, f)
}
