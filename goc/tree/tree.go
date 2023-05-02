package tree

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
)

// Helper functions
func save(obj interface{}, filename string) {
	bs, err := json.MarshalIndent(obj, "", "  ")
	if err != nil {
		log.Fatal(err)
	}
	if err := os.WriteFile(filename, bs, 0666); err != nil {
		log.Fatal(err)
	}
}

func load(obj interface{}, filename string) {
	bs, err := os.ReadFile(filename)
	if err != nil {
		log.Fatal(err)
	}
	if err != json.Unmarshal(bs, obj) {
		log.Fatal(err)
	}
}

type Report struct {
	Accuracy    float64
	Precision   float64
	Recall      float64
	Specificity float64
}

type Tree struct {
	Root    *Node    `json:"root"`
	Imp     Impurity `json:"impurity"`
	MinGain float64
}

type TreeClassifier struct {
	Tree
}

type TreeRegressor struct {
	Tree
}

func (rp Report) FScore(beta float64) float64 {
	denominator := beta*rp.Recall + rp.Precision
	if denominator == 0.0 {
		return 0.0
	}
	return (1 + beta*beta) * rp.Recall * rp.Precision / denominator
}

func NewTreeClassifier(imp Impurity, ming float64) TreeClassifier {
	return TreeClassifier{Tree{Imp: imp, MinGain: ming}}
}

func NewTreeRegressor(ming float64) TreeRegressor {
	return TreeRegressor{Tree{Imp: Impurity{eval: mse}, MinGain: ming}}
}

func (dt Tree) String() string {
	s := fmt.Sprintf("Tree {\n  root: %v\n}\n", dt.Root)
	return s
}

func (dt *Tree) Fit(ds DataSet) {
	dt.Root = NewNode(ds.Examples, 0)
	dt.Root.Fit(dt.Imp, dt.MinGain)
}

func (dt Tree) Predict(features []float64) float64 {
	nd := dt.Root
	for nd.Left != nil {
		if features[nd.Split.Dimension] < nd.Split.Threshold {
			nd = nd.Left
		} else {
			nd = nd.Right
		}
	}
	return nd.Label
}

func (dt Tree) Save(jsonfile string) {
	var obj interface{} = dt
	save(obj, jsonfile)
}

func (dt *Tree) Load(jsonfile string) {
	var obj interface{} = dt
	load(obj, jsonfile)
}

func (dt TreeClassifier) Equals(other TreeClassifier) bool {
	return dt.Root.Equals(other.Root)
}

func (dt TreeClassifier) Score(ds DataSet) Report {
	return getReport(dt.Predict, ds.Examples, dt.Imp.Val)
}

func (dt TreeRegressor) Equals(other TreeRegressor) bool {
	return dt.Root.Equals(other.Root)
}

// coefficient of determination
func (dt TreeRegressor) Score(ds DataSet) float64 {
	return getCoD(dt.Predict, ds.Examples)
}

type predFunc func(features []float64) float64

func getReport(predict predFunc, examples []Example, threshold float64) Report {
	var tp, tn, fp, fn float64
	for _, example := range examples {
		p := predict(example.features)
		if p > threshold && example.target > threshold {
			tp += 1.0
		} else if p > threshold && example.target <= threshold {
			fp += 1.0
		} else if p <= threshold && example.target > threshold {
			fn += 1.0
		} else {
			tn += 1.0
		}
	}
	var precision, recall, specificity float64
	if tp == 0.0 {
		precision = 0.0
		recall = 0.0
	} else {
		precision = tp / (tp + fp)
		recall = tp / (tp + fn)
	}
	if tn == 0.0 {
		specificity = 0.0
	} else {
		specificity = tn / (tn + fp)
	}
	return Report{
		Accuracy:    (tp + tn) / (tp + tn + fp + fn),
		Precision:   precision,
		Recall:      recall,
		Specificity: specificity,
	}
}

// Coefficient of Determination
func getCoD(predict predFunc, examples []Example) float64 {
	msq := mse(examples, 0.0)
	var sum float64
	for _, example := range examples {
		p := predict(example.features)
		sum += (p - example.target) * (p - example.target)
	}
	sum /= float64(len(examples))
	return 1.0 - sum/msq
}
