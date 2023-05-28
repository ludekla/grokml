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
