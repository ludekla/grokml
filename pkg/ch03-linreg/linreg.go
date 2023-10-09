package ch03

import (
	"encoding/json"
	"math"
	"math/rand"

	vc "grokml/pkg/vector"
)

// LinReg implements a linear regression engine.
type LinReg struct {
	Weights vc.Vector `json:"weights"`
	Bias    float64   `json:"bias"`
	LRate   float64   `json:"lrate"`
	NEpochs int       `json:"nepochs"`
}

// NewLinReg is the constructor function for LinReg.
func NewLinReg(lrate float64, epochs int) *LinReg {
	return &LinReg{LRate: lrate, NEpochs: epochs}
}

// Fit performs the training.
func (lr *LinReg) Fit(dpoints []vc.Vector, labels []float64) []float64 {
	stats := vc.GetDataStats(dpoints, labels)
	dpoints, labels = stats.Normalise(dpoints, labels)
	weights := vc.RandVector(len(stats.XMean))
	bias := rand.Float64()
	errs := make([]float64, 0, lr.NEpochs)
	size := float64(len(dpoints))
	for ep := 0; ep < lr.NEpochs; ep++ {
		var err float64
		for i, vec := range dpoints {
			delta := weights.Dot(vec) + bias - labels[i]
			weights.IAdd(vec.ScaMul(-lr.LRate * delta))
			bias -= lr.LRate * delta
			err += delta * delta
		}
		errs = append(errs, math.Sqrt(err/size))
	}
	weights.IDiv(stats.XStd)
	weights.IScaMul(stats.YStd)
	lr.Bias = stats.YMean + stats.YStd*bias - weights.Dot(stats.XMean)
	lr.Weights = weights
	return errs
}

// Predict returns the estimated output values.
func (lr LinReg) Predict(dpoints []vc.Vector) []float64 {
	preds := make([]float64, len(dpoints))
	for i, vec := range dpoints {
		preds[i] = lr.Weights.Dot(vec) + lr.Bias
	}
	return preds
}

// Score computes the coefficient of determination.
func (lr LinReg) Score(dpoints []vc.Vector, labels []float64) float64 {
	preds := lr.Predict(dpoints)
	ym := mean(preds)
	// Residual Sum of Squares, Total Sum of Squares
	var rss, tss float64
	for i, label := range labels {
		// Residual Sum of Squares
		rss += (label - preds[i]) * (label - preds[i])
		// Total Sum of Squares
		tss += (label - ym) * (label - ym)
	}
	return 1.0 - rss/tss
}

// mean is a helper function for computing the mean of a slice of floats.
func mean(numbers []float64) float64 {
	var m float64
	for _, val := range numbers {
		m += val
	}
	return m / float64(len(numbers))
}

// Marshal and Unmarshal implement the JSONable interface from the persist package.
func (lr LinReg) Marshal() ([]byte, error) {
	return json.MarshalIndent(lr, "", "    ")
}

func (lr *LinReg) Unmarshal(bs []byte) error {
	return json.Unmarshal(bs, lr)
}
