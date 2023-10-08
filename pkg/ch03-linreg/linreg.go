package ch03

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"

	vc "grokml/pkg/vector"
)

type LinReg struct {
	Weights vc.Vector `json:"weights"`
	Bias    float64   `json:"bias"`
	LRate   float64   `json:"lrate"`
	NEpochs int       `json:"nepochs"`
}

func NewLinReg(lrate float64, epochs int) *LinReg {
	return &LinReg{LRate: lrate, NEpochs: epochs}
}

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

func (lr LinReg) Predict(dpoints []vc.Vector) []float64 {
	preds := make([]float64, len(dpoints))
	for i, vec := range dpoints {
		preds[i] = lr.Weights.Dot(vec) + lr.Bias
	}
	return preds
}

// Computes the coefficient of determination
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

// helper
func mean(numbers []float64) float64 {
	var m float64
	for _, val := range numbers {
		m += val
	}
	return m / float64(len(numbers))
}

type JSONable interface {
	Marshal() ([]byte, error)
	Unmarshal(bs []byte) error
}

// Marshal and Unmarshal implement the JSONable interface.
func (lr LinReg) Marshal() ([]byte, error) {
	return json.MarshalIndent(lr, "", "    ")
}

func (lr *LinReg) Unmarshal(bs []byte) error {
	return json.Unmarshal(bs, lr)
}

func Dump(jn JSONable, filepath string) error {
	asBytes, err := jn.Marshal()
	if err != nil {
		return fmt.Errorf("cannot marshal %v into JSON bytes", jn)
	}
	err = os.WriteFile(filepath, asBytes, 0666)
	if err != nil {
		return fmt.Errorf("cannot write JSON bytes into file %s", filepath)
	}
	return nil
}

func Load(jn JSONable, filepath string) error {
	asBytes, err := os.ReadFile(filepath)
	if err != nil {
		return fmt.Errorf("cannot read from file %s: %v", filepath, err)
	}
	err = jn.Unmarshal(asBytes)
	if err != nil {
		return fmt.Errorf("cannot unmarshal JSON bytes: %v", err)
	}
	return nil
}
