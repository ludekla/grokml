package ch09

import (
	"math"
	"math/rand"

	pl "grokml/pkg/pipeline"
)

// Types
type Forest struct {
	Size       int               `json:"size"`
	Estimators []*TreeClassifier `json:"trees"`
	Report pl.Report `json:"-"`
}

type ForestClassifier struct {
	Forest
}

type AdaBoostClassifier struct {
	Forest
	Coeffs []float64 `json:"coeffs"`
}

type GradBoostRegressor struct {
	Size       int              `json:"size"`
	Estimators []*TreeRegressor `json:"trees"`
	lRate      float64          `json:"-"`
}

// Forest Classifier: Constructor Function
func NewForestClassifier(nTrees int, imp Impurity, ming float64) *ForestClassifier {
	trees := make([]*TreeClassifier, nTrees)
	for i, _ := range trees {
		clf := NewTreeClassifier(imp, ming)
		trees[i] = &clf
	}
	return &ForestClassifier{Forest{Size: nTrees, Estimators: trees}}
}

// Forest Classifier: Methods
func (f *Forest) Fit(dpoints [][]float64, labels []float64) {
	chunkSize := int(0.9*float64(len(dpoints)))
	for _, tree := range f.Estimators {
		tree.Fit(dpoints[:chunkSize], labels[:chunkSize])
		rand.Shuffle(len(labels), func(i, j int) {
			dpoints[i], dpoints[j] = dpoints[j], dpoints[i]
			labels[i], labels[j] = labels[j], labels[i]
		})
	}
}

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

func (f *Forest) Score(dpoints [][]float64, labels []float64) float64 {
	preds := f.Predict(dpoints)
	f.Report = getReport(preds, labels)
	return f.Report.Accuracy
}

func (f *Forest) Save(filename string) {
	var obj interface{} = f
	save(obj, filename)
}

func (f *Forest) Load(filename string) {
	var obj interface{} = f
	load(obj, filename)
}

// AdaBoost Classifier: Constructor and Methods
func NewAdaBoostClassifier(n int, imp Impurity, ming float64) *AdaBoostClassifier {
	trees := make([]*TreeClassifier, n)
	for i, _ := range trees {
		clf := NewTreeClassifier(imp, ming)
		trees[i] = &clf
	}
	return &AdaBoostClassifier{Forest{Size: n, Estimators: trees}, nil}
}
// AdaBoost Classifier: Methods
func (ad *AdaBoostClassifier) Fit(dpoints [][]float64, labels []float64) {
	ad.Coeffs = make([]float64, ad.Size)
	for i, tree := range ad.Estimators {
		tree.Fit(dpoints, labels)
		acc := tree.Score(dpoints, labels)
		if acc > 0.9999 {
			acc = 0.9999
		} else if acc < 0.0001 {
			acc = 0.0001
		}
		ad.Coeffs[i] = math.Log(acc / (1.0 - acc))
	}
}

func (ad *AdaBoostClassifier) Predict(dpoints [][]float64) []float64 {
	preds := make([]float64, len(dpoints))
	for i, tree := range ad.Estimators {
		for j, pred := range tree.Predict(dpoints) {
			preds[j] += ad.Coeffs[i] * (2*pred - 1.0)
		}
	}
	return preds
}

func (ad *AdaBoostClassifier) Score(dpoints [][]float64, labels []float64) float64 {
	predictions := ad.Predict(dpoints)
	ad.Report = getReport(predictions, labels)
	return ad.Report.Accuracy
}

func (ad *AdaBoostClassifier) Save(filename string) {
	var obj interface{} = ad
	save(obj, filename)
}

func (ad *AdaBoostClassifier) Load(filename string) {
	var obj interface{} = ad
	load(obj, filename)
}

// Gradient Boosting Regressor: Constructor Function
func NewGradBoostRegressor(n int, ming float64, lrate float64) *GradBoostRegressor {
	trees := make([]*TreeRegressor, n)
	for i, _ := range trees {
		reg := NewTreeRegressor(ming)
		trees[i] = &reg
	}
	return &GradBoostRegressor{Size: n, Estimators: trees, lRate: lrate}
}
// Gradient Boosting Regressor: Methods
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

func (gb *GradBoostRegressor) Predict(dpoints [][]float64) []float64 {
	preds := gb.Estimators[0].Predict(dpoints)
	for _, tree := range gb.Estimators[1:] {
		for i, pred := range tree.Predict(dpoints) {
			preds[i] += gb.lRate * pred
		} 
	}
	return preds
}

// coefficient of determination
func (gb *GradBoostRegressor) Score(dpoints [][]float64, labels []float64) float64 {
	preds := gb.Predict(dpoints)
	return getCoD(preds, labels)
}

func (gb *GradBoostRegressor) Save(filename string) {
	var obj interface{} = gb
	save(obj, filename)
}

func (gb *GradBoostRegressor) Load(filename string) {
	var obj interface{} = gb
	load(obj, filename)
}

