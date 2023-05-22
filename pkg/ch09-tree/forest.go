package ch09

import (
	"math"
)

// Types
type Forest struct {
	Size       int               `json:"size"`
	Estimators []*TreeClassifier `json:"trees"`
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
	lRate      float64          `json:"learning_rate"`
}

// Forest Classifier: Constructor Function
func NewForestClassifier(n int, imp Impurity, ming float64) *ForestClassifier {
	trees := make([]*TreeClassifier, n)
	for i, _ := range trees {
		clf := NewTreeClassifier(imp, ming)
		trees[i] = &clf
	}
	return &ForestClassifier{Forest{Size: n, Estimators: trees}}
}
// Forest Classifier: Methods
func (f *Forest) Fit(ds DataSet) {
	for _, tree := range f.Estimators {
		tree.Fit(ds)
	}
}

func (f *Forest) Predict(features []float64) float64 {
	var avg float64
	for _, tree := range f.Estimators {
		avg += tree.Predict(features)
	}
	return avg / float64(f.Size)
}

func (f *Forest) Score(ds DataSet) Report {
	return getReport(f.Predict, ds.Examples, f.Estimators[0].Imp.Val)
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
func (ad *AdaBoostClassifier) Fit(ds DataSet) {
	ad.Coeffs = make([]float64, ad.Size)
	for i, tree := range ad.Estimators {
		trainSet, _ := ds.Split(0.2)
		tree.Fit(trainSet)
		rep := tree.Score(trainSet)
		if rep.Accuracy > 0.99 {
			rep.Accuracy = 0.99
		}
		ad.Coeffs[i] = math.Log(rep.Accuracy / (1.0 - rep.Accuracy))
	}
}

func (ad *AdaBoostClassifier) Predict(features []float64) float64 {
	var sum float64
	for i, tree := range ad.Estimators {
		p := tree.Predict(features)
		sum += ad.Coeffs[i] * (2*p - 1.0)
	}
	return sum
}

func (ad *AdaBoostClassifier) Score(ds DataSet) Report {
	return getReport(ad.Predict, ds.Examples, 0.0)
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
func (gb *GradBoostRegressor) Fit(ds DataSet) {
	dset := ds.Copy()
	for _, tree := range gb.Estimators {
		tree.Fit(dset)
		for j, example := range dset.Examples {
			example.target -= tree.Predict(example.features)
			dset.Examples[j] = example
		}
	}
}

func (gb *GradBoostRegressor) Predict(features []float64) float64 {
	pred := gb.Estimators[0].Predict(features)
	for _, tree := range gb.Estimators[1:] {
		pred += gb.lRate * tree.Predict(features)
	}
	return pred
}

// coefficient of determination
func (gb *GradBoostRegressor) Score(ds DataSet) float64 {
	return getCoD(gb.Predict, ds.Examples)
}

func (gb *GradBoostRegressor) Save(filename string) {
	var obj interface{} = gb
	save(obj, filename)
}

func (gb *GradBoostRegressor) Load(filename string) {
	var obj interface{} = gb
	load(obj, filename)
}

