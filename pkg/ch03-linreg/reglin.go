package ch03

import (
	"encoding/json"
	"math"
	"math/rand"

	vc "grokml/pkg/vector"
)

type RegLin struct {
	*LinReg
	LassoPen float64 `json:"lasso_penalty"` // L1
	RidgePen float64 `json:"ridge_penalty"` // L2
}

func NewRegLin(lrate float64, nEpochs int, lpen, rpen float64) *RegLin {
	return &RegLin{
		LinReg:   NewLinReg(lrate, nEpochs),
		LassoPen: lpen,
		RidgePen: rpen,
	}
}

func (rl *RegLin) Fit(dpoints []vc.Vector, labels []float64) []float64 {
	stats := vc.GetDataStats(dpoints, labels)
	dpoints, labels = stats.Normalise(dpoints, labels)
	weights := vc.RandVector(len(stats.XMean))
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
	weights.IDiv(stats.XStd)
	weights.IScaMul(stats.YStd)
	rl.Bias = stats.YMean + stats.YStd*bias - weights.Dot(stats.XMean)
	rl.Weights = weights
	return errs
}

// Helper
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

// Marshal and Unmarshal implement the JSONable interface as defined in this package.
func (rl RegLin) Marshal() ([]byte, error) {
	return json.MarshalIndent(rl, "", "    ")
}

func (rl *RegLin) Unmarshal(bs []byte) error {
	return json.Unmarshal(bs, rl)
}
