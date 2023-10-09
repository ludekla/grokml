package ch05

import (
	"encoding/json"
	"math/rand"

	"grokml/pkg/ch06-logreg"
	tk "grokml/pkg/tokens"
	vc "grokml/pkg/vector"
)

// Perceptron implements a perceptron classifier. The weights are maintained by
// an object that satisfies the Updater interface.
// The weights are not held directly because their type must not be fixed
// but kept hidden behind the interface.
type Perceptron[D ch06.DataPoint] struct {
	Updater ch06.Updater[D] `json:"updater"`
	Bias    float64         `json:"bias"`
	NEpochs int             `json:"nepochs"`
	LRate   float64         `json:"lrate"`
}

// NewTextPerceptron provides a variant of LogReg that works with text data.
// Every word, ie token, found in the text corpus will be assigned a weight.
func NewTextPerceptron(nEpo int, lrate float64) *Perceptron[tk.TokenMap] {
	return &Perceptron[tk.TokenMap]{
		Bias:    rand.Float64(),
		NEpochs: nEpo,
		LRate:   lrate,
		Updater: new(ch06.TokenMapUpdater),
	}
}

// NewNumPerceptron provides a variant of LogReg that works with vectorial data points.
func NewNumPerceptron(nEpo int, lrate float64) *Perceptron[vc.Vector] {
	return &Perceptron[vc.Vector]{
		Bias:    rand.Float64(),
		NEpochs: nEpo,
		LRate:   lrate,
		Updater: new(ch06.VectorUpdater),
	}
}

// Marshal and Unmarhsal implement the JSONable interface from the persist package.
func (pc Perceptron[D]) Marshal() ([]byte, error) {
	return json.MarshalIndent(pc, "", "   ")
}

func (pc *Perceptron[D]) Unmarshal(bs []byte) error {
	return json.Unmarshal(bs, pc)
}

// Fit performs the training.
func (pc *Perceptron[D]) Fit(dpoints []D, labels []float64) []float64 {
	size := len(dpoints)
	if size == 0 {
		return nil
	}
	// Initialise weights and bias.
	pc.Updater.Init(len(dpoints[0]))
	var bias float64
	errs := make([]float64, pc.NEpochs)
	nDPs := float64(size)
	for i := 0; i < pc.NEpochs; i++ {
		var sum float64
		for j, dpoint := range dpoints {
			pred := heaviside(pc.Updater.Dot(dpoint) + bias)
			diff := pred - labels[j]
			if diff != 0.0 {
				pc.Updater.Update(dpoint, -pc.LRate*diff)
				bias -= pc.LRate * diff
			} else {
				sum++
			}
		}
		errs[i] = sum / nDPs
	}
	pc.Bias = bias
	return errs
}

// Predict computes the output pushed through a Heaviside function.
func (pc Perceptron[D]) Predict(dpoints []D) []float64 {
	res := make([]float64, len(dpoints))
	for i, dpoint := range dpoints {
		res[i] = heaviside(pc.Updater.Dot(dpoint) + pc.Bias)
	}
	return res
}

// Score computes the accuracy.
func (pc Perceptron[D]) Score(dpoints []D, labels []float64) float64 {
	var acc float64
	preds := pc.Predict(dpoints)
	for i, pred := range preds {
		if pred > 0.5 && labels[i] == 1.0 {
			acc++
		} else if pred < 0.5 && labels[i] == 0.0 {
			acc++
		}
	}
	return acc / float64(len(dpoints))
}

// heaviside is a helper function representing the Heaviside function.
func heaviside(val float64) float64 {
	if val < 0.0 {
		return 0.0
	} else {
		return 1.0
	}
}
