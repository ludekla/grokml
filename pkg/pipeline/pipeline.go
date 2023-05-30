package pipeline

import (
	tk "grokml/pkg/tokens"
	vc "grokml/pkg/vector"
)

type intype interface {
	float64 | string
}

type outtype interface {
	vc.Vector | tk.TokenMap
}

type Transformer[I intype, O outtype] interface {
	Transform([][]I) []O
}

type Estimator[O outtype] interface {
	Fit(dpoints []O, labels []float64) []float64
	Predict(dpoints []O) []float64
	Score(dpoints []O, labels []float64) float64
	Load(jsonfile string) error
	Save(jsonfile string) error
}

type Pipeline[I intype, O outtype] struct {
	transformer Transformer[I, O]
	estimator   Estimator[O]
}

type Report struct {
	Accuracy    float64
	Precision   float64
	Recall      float64
	Specificity float64
}

func (rp Report) FScore(beta float64) float64 {
	denominator := beta*rp.Recall + rp.Precision
	if denominator == 0.0 {
		return 0.0
	}
	return (1 + beta*beta) * rp.Recall * rp.Precision / denominator
}
