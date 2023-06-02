package ch09

import (
	"encoding/json"
	"fmt"
	"log"
	"os"

	pl "grokml/pkg/pipeline"
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

func Mean(vals []float64) float64 {
	var mean float64
	for _, val := range vals {
		mean += val
	}
	mean /= float64(len(vals))
	return mean
}

type Tree struct {
	Root    *Node    `json:"root"`
	Imp     Impurity `json:"-"`
	MinGain float64
}

type TreeClassifier struct {
	Tree
	Report pl.Report `json:"-"`
}

type TreeRegressor struct {
	Tree
}

func NewTreeClassifier(imp Impurity, ming float64) TreeClassifier {
	return TreeClassifier{Tree{Imp: imp, MinGain: ming}, pl.Report{}}
}

func NewTreeRegressor(ming float64) TreeRegressor {
	return TreeRegressor{Tree{Imp: MSE, MinGain: ming}}
}

func (dt Tree) String() string {
	s := fmt.Sprintf("Tree {\n  root: %v\n}\n", dt.Root)
	return s
}

func (dt *Tree) Fit(dpoints [][]float64, labels []float64) {
	examples := MakeExamples(dpoints, labels)
	dt.Root = NewNode(0, dt.MinGain)
	dt.Root.Fit(examples, dt.Imp)
}

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

func (dt Tree) Save(jsonfile string) {
	var obj interface{} = dt
	save(obj, jsonfile)
}

func (dt *Tree) Load(jsonfile string) {
	var obj interface{} = dt
	load(obj, jsonfile)
}

func (dt *TreeClassifier) Score(dpoints [][]float64, labels []float64) float64 {
	predictions := dt.Predict(dpoints)
	dt.Report = getReport(predictions, labels)
	return dt.Report.Accuracy
}

// coefficient of determination
func (dt TreeRegressor) Score(dpoints [][]float64, labels []float64) float64 {
	predictions := dt.Predict(dpoints)
	return getCoD(predictions, labels)
}

func getReport(predictions []float64, labels []float64) pl.Report {
	var tp, tn, fp, fn float64
	for i, p := range predictions {
		if p > threshold && labels[i] > threshold {
			tp++
		} else if p > threshold && labels[i] <= threshold {
			fp++
		} else if p <= threshold && labels[i] > threshold {
			fn++
		} else {
			tn++
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
	return pl.Report{
		Accuracy:    (tp + tn) / (tp + tn + fp + fn),
		Precision:   precision,
		Recall:      recall,
		Specificity: specificity,
	}
}

// Coefficient of Determination
func getCoD(predictions []float64, labels []float64) float64 {
	mean := Mean(labels)
	var rss float64 // residual square sum
	var tss float64 // total square sum
	for i, pred := range predictions {
		rss += (pred - labels[i]) * (pred - labels[i])
		tss += (mean - labels[i]) * (mean - labels[i])
	}
	return 1.0 - rss/tss
}
