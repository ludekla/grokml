package ch06

import (
	"encoding/json"
	"math"
	"math/rand"

	tk "grokml/pkg/tokens"
	vc "grokml/pkg/vector"
)

// LogReg implements a logistic regression engine. The weights are maintained by
// an object that satisfies the Updater interface.
// The weights are not held directly because their type must not be fixed
// but kept hidden behind the interface.
type LogReg[D DataPoint] struct {
	Updater Updater[D] `json:"updater"`
	Bias    float64    `json:"bias"`
	NEpochs int        `json:"nepochs"`
	LRate   float64    `json:"lrate"`
}

// NewTextLogReg provides a variant of LogReg that works with text data.
// Every word, ie token, found in the text corpus will be assigned a weight.
func NewTextLogReg(nEpo int, lrate float64) *LogReg[tk.TokenMap] {
	return &LogReg[tk.TokenMap]{
		Updater: new(TokenMapUpdater),
		Bias:    rand.Float64(),
		NEpochs: nEpo,
		LRate:   lrate,
	}
}

// NewNumLogReg provides a variant of LogReg that works with vectorial data points.
func NewNumLogReg(nEpo int, lrate float64) *LogReg[vc.Vector] {
	return &LogReg[vc.Vector]{
		Updater: new(VectorUpdater),
		Bias:    rand.Float64(),
		NEpochs: nEpo,
		LRate:   lrate,
	}
}

// Marshal and Unmarhsal implement the JSONable interface from the persist package.
func (lr LogReg[D]) Marshal() ([]byte, error) {
	return json.MarshalIndent(lr, "", "   ")
}

func (lr *LogReg[D]) Unmarshal(bs []byte) error {
	return json.Unmarshal(bs, lr)
}

// Fit performs the training.
func (lr *LogReg[D]) Fit(dpoints []D, labels []float64) []float64 {
	size := len(dpoints)
	if size == 0 {
		return nil
	}
	// Initialise weights and bias.
	lr.Updater.Init(len(dpoints[0]))
	var bias float64
	errs := make([]float64, lr.NEpochs)
	nDPs := float64(size)
	for i := 0; i < lr.NEpochs; i++ {
		var sum float64
		for j, dpoint := range dpoints {
			pred := sigmoid(lr.Updater.Dot(dpoint) + bias)
			diff := pred - labels[j]
			lr.Updater.Update(dpoint, -lr.LRate*diff)
			bias -= lr.LRate * diff
			sum += xentropy(pred, labels[j])
		}
		errs[i] = sum / nDPs
	}
	lr.Bias = bias
	return errs
}

// Predict returns the output of the sigmoid squasher.
func (lr LogReg[D]) Predict(dpoints []D) []float64 {
	res := make([]float64, len(dpoints))
	for i, dpoint := range dpoints {
		res[i] = sigmoid(lr.Updater.Dot(dpoint) + lr.Bias)
	}
	return res
}

// Score computes the accuracy.
func (lr LogReg[D]) Score(dpoints []D, labels []float64) float64 {
	var acc float64
	preds := lr.Predict(dpoints)
	for i, pred := range preds {
		if pred > 0.5 && labels[i] == 1.0 {
			acc++
		} else if pred < 0.5 && labels[i] == 0.0 {
			acc++
		}
	}
	return acc / float64(len(dpoints))
}

// sigmoid is a helper function representing the squasher (activation function).
func sigmoid(val float64) float64 {
	return 1.0 / (1.0 + math.Exp(-val))
}

// xentropy is a helper function for computing the cost (cross-entropy).
func xentropy(pred float64, label float64) float64 {
	if label == 1.0 {
		return -math.Log(pred)
	} else {
		return -math.Log(1.0 - pred)
	}
}
