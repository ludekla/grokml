package ch03

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"

	"grokml/pkg/utils"
)

type Regression interface {
	Fit(dpoints [][]float64, labels []float64) []float64
	Predict(points [][]float64) []float64
	Score(dpoints [][]float64, labels []float64) float64
	Save(filepath string) error
}

type LinReg struct {
	Weights utils.Vector `json:"weights"`
	Bias    float64      `json:"bias"`
	LRate   float64      `json:"lrate"`
	NEpochs int          `json:"nepochs"`
}

func NewLinReg(lr float64, epochs int) *LinReg {
	return &LinReg{LRate: lr, NEpochs: epochs}
}

func (l *LinReg) Fit(dpoints [][]float64, labels []float64) []float64 {
	vecs := utils.ToVectors(dpoints)
	stats := utils.GetDataStats(vecs, labels)
	vecs, labels = stats.Normalise(vecs, labels)
	weights := utils.RandVector(len(stats.XMean))
	bias := rand.Float64()
	errs := make([]float64, 0, l.NEpochs)
	size := float64(len(vecs))
	for ep := 0; ep < l.NEpochs; ep++ {
		var err float64
		for i, vec := range vecs {
			delta := weights.Dot(vec) + bias - labels[i]
			weights.IAdd(vec.ScaMul(-l.LRate * delta))
			bias -= l.LRate * delta
			err += delta * delta
		}
		errs = append(errs, math.Sqrt(err/size))
	}
	weights.IDiv(stats.XStd)
	weights.IScaMul(stats.YStd)
	l.Bias = stats.YMean + stats.YStd*bias - weights.Dot(stats.XMean)
	l.Weights = weights
	return errs
}

func (l LinReg) Predict(points [][]float64) []float64 {
	vecs := utils.ToVectors(points)
	preds := make([]float64, len(vecs))
	for i, vec := range vecs {
		preds[i] = l.Weights.Dot(vec) + l.Bias
	}
	return preds
}

// Computes the coefficient of determination
func (l LinReg) Score(dpoints [][]float64, labels []float64) float64 {
	preds := l.Predict(dpoints)
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

func mean(numbers []float64) float64 {
	var m float64
	for _, val := range numbers {
		m += val
	}
	return m / float64(len(numbers))
}

func (l LinReg) Save(filepath string) error {
	asBytes, err := json.MarshalIndent(l, "", "    ")
	if err != nil {
		return fmt.Errorf("cannot marshal %v into JSON bytes", l)
	}
	err = os.WriteFile(filepath, asBytes, 0666)
	if err != nil {
		return fmt.Errorf("cannot write JSON bytes into file %s", filepath)
	}
	return nil
}

func LinRegFromJSON(filepath string) *LinReg {
	lr := LinReg{}
	asBytes, err := os.ReadFile(filepath)
	if err != nil {
		log.Fatalf("cannot read from file %s: %v", filepath, err)
	}
	err = json.Unmarshal(asBytes, &lr)
	if err != nil {
		log.Fatalf("cannot unmarshal JSON bytes: %v", err)
	}
	return &lr
}
