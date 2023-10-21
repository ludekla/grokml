package ch09

import (
	"encoding/json"
	"fmt"

	pl "grokml/pkg/pipeline"
)

// Tree implements a binary tree structure.
type Tree struct {
	Root    *Node    `json:"root"`
	Imp     Impurity `json:"-"`
	MinGain float64
}

// TreeClassifier implements a decision tree for classification.
type TreeClassifier struct {
	Tree
	Report pl.Report `json:"-"`
}

// TreeRegressor implements a regression tree.
type TreeRegressor struct {
	Tree
}

// NewTreeClassifier is a factory function for TreeClassifier.
func NewTreeClassifier(imp Impurity, ming float64) TreeClassifier {
	return TreeClassifier{Tree{Imp: imp, MinGain: ming}, pl.Report{}}
}

// NewTreeRegressor is a factory function for TreeRegressor.
func NewTreeRegressor(ming float64) TreeRegressor {
	return TreeRegressor{Tree{Imp: MSE, MinGain: ming}}
}

// String makes a Tree a Stringer (for debugging purposes).
func (dt Tree) String() string {
	s := fmt.Sprintf("Tree {\n  root: %v\n}\n", dt.Root)
	return s
}

// Fit performs the training of a tree. This method is called
// on the embedded Tree struct inside a classification tree or
// regression tree.
func (dt *Tree) Fit(dpoints [][]float64, labels []float64) {
	examples := MakeExamples(dpoints, labels)
	dt.Root = NewNode(0, dt.MinGain)
	dt.Root.Fit(examples, dt.Imp)
}

// Predict implements the inference of the labels for the given data points.
func (dt Tree) Predict(dpoints [][]float64) []float64 {
	labels := make([]float64, len(dpoints))
	for i, dpoint := range dpoints {
		nd := dt.Root
		// Loop until you hit a leaf.
		for nd.Left != nil {
			if dpoint[nd.Split.Dimension] < nd.Split.Threshold {
				nd = nd.Left
			} else {
				nd = nd.Right
			}
		}
		labels[i] = nd.Label
	}
	return labels
}

// Marshal and Unmarshal implement the JSONable interface of the persist pkg.
func (dt Tree) Marshal() ([]byte, error) {
	return json.MarshalIndent(dt, "", "    ")
}

func (dt *Tree) Unmarshal(bs []byte) error {
	return json.Unmarshal(bs, dt)
}

// Score computes the quantities necessary to populate a Report struct and
// returns the accuracy.
func (dt *TreeClassifier) Score(dpoints [][]float64, labels []float64) float64 {
	predictions := dt.Predict(dpoints)
	dt.Report = pl.GetReport(predictions, labels)
	return dt.Report.Accuracy
}

// Score computes the coefficient of determination as a measure of performance
// for a regression tree.
func (dt TreeRegressor) Score(dpoints [][]float64, labels []float64) float64 {
	predictions := dt.Predict(dpoints)
	return pl.GetCoD(predictions, labels)
}
