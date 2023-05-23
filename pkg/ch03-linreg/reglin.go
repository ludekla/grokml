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

type RegLin struct {
	*LinReg
	LassoPen float64 `json:"lasso_penalty"`
	RidgePen float64 `json:"ridge_penalty"`
}

func NewRegLin(lrate float64, nEpochs int, lpen, rpen float64) *RegLin {
	return &RegLin{
		LinReg:   NewLinReg(lrate, nEpochs),
		LassoPen: lpen,
		RidgePen: rpen,
	}
}

func (l *RegLin) Fit(ds utils.DataSet[utils.Vector, float64]) []float64 {
	stats := utils.NewDataStats(ds)
	nds := stats.Normalise(ds)
	l.Weights = utils.RandVector(len(stats.XMean))
	l.Bias = rand.Float64()
	errs := make([]float64, 0, l.NEpochs)
	size := nds.Size()
	for ep := 0; ep < l.NEpochs; ep++ {
		var err float64
		for i := 0; i < size; i++ {
			x, y := nds.Random()
			delta := l.Weights.Dot(x) + l.Bias - y
			l.Weights = l.Weights.
				Add(x.ScaMul(-l.LRate * delta)).
				Add(l1grad(l.Weights).ScaMul(-l.LassoPen)).
				Add(l.Weights.ScaMul(-l.RidgePen))
			l.Bias -= l.LRate * delta
			err += delta * delta
		}
		errs = append(errs, math.Sqrt(err/float64(size)))
	}
	l.Weights = l.Weights.Div(stats.XStd).ScaMul(stats.YStd)
	l.Bias = stats.YMean + stats.YStd*l.Bias - l.Weights.Dot(stats.XMean)
	return errs
}

func l1grad(vec utils.Vector) utils.Vector {
	res := make(utils.Vector, len(vec))
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

func (l RegLin) Save(filepath string) error {
	asBytes, err := json.MarshalIndent(l, "", "    ")
	if err != nil {
		return fmt.Errorf("unable to marshal LassoReg into JSON: %v", err)
	}
	err = os.WriteFile(filepath, asBytes, 0666)
	if err != nil {
		return fmt.Errorf("cannot write JSON bytes to file")
	}
	return nil
}

func RegLinFromJSON(filepath string) *RegLin {
	lr := RegLin{}
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
