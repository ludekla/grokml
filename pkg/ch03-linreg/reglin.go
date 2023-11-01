package ch03

import (
	"encoding/json"
	"math"
	"math/rand"

	vc "grokml/pkg/vector"
)

// RegLin implements a regularised linear regression engine. Two types of
// regularisation can be switched on: Lasso and Ridge.
type RegLin struct {
	*LinReg
	LassoPen float64 `json:"lasso_penalty"` // L1
	RidgePen float64 `json:"ridge_penalty"` // L2
}

// NewRegLin is a constructor function for RegLin.
func NewRegLin(lrate float64, nEpochs int, lpen, rpen float64) *RegLin {
	return &RegLin{
		LinReg:   NewLinReg(lrate, nEpochs),
		LassoPen: lpen,
		RidgePen: rpen,
	}
}

// Fit performs the training.
func (rl *RegLin) Fit(dpoints []vc.Vector, labels []float64) []float64 {
	weights := vc.RandVector(len(dpoints[0]))
	bias := rand.Float64()
	errs := make([]float64, 0, rl.NEpochs)
	size := float64(len(dpoints))
	for ep := 0; ep < rl.NEpochs; ep++ {
		var err float64
		for i, vec := range dpoints {
			delta := weights.Dot(vec) + bias - labels[i]
			weights = weights.
				Add(vec.ScaMul(-rl.LRate * delta)).
				Add(l1grad(weights).ScaMul(-rl.LassoPen)).
				Add(weights.ScaMul(-rl.RidgePen))
			bias -= rl.LRate * delta
			err += delta * delta
		}
		errs = append(errs, math.Sqrt(err/float64(size)))
	}
	rl.Bias = bias
	rl.Weights = weights
	return errs
}

// l1grad is a helper function that computes the L1 gradient.
func l1grad(vec vc.Vector) vc.Vector {
	res := make(vc.Vector, len(vec))
	for i, val := range vec {
		switch {
		case val > 0:
			res[i] = 1.0
		case val < 0:
			res[i] = -1.0
		default:
			res[i] = 0.0
		}
	}
	return vec
}

// Marshal and Unmarshal implement the JSONable interface from the persist package.
func (rl RegLin) Marshal() ([]byte, error) {
	return json.MarshalIndent(rl, "", "    ")
}

func (rl *RegLin) Unmarshal(bs []byte) error {
	return json.Unmarshal(bs, rl)
}
