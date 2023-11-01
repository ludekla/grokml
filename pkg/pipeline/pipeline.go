package pipeline

import (
	"encoding/json"

	tk "grokml/pkg/tokens"
	vc "grokml/pkg/vector"
)

// InType is the input type for the pipeline input.
type InType interface {
	float64 | string
}

// OutType is the output type for the transformer and at the same
// time the input type for the estimator.
type OutType interface {
	vc.Vector | tk.TokenMap
}

// Transformer is the interface for a data transformation engine.
type Transformer[I InType, O OutType] interface {
	Transform([][]I) []O
}

// Scaler implements the a scaling engine.
type Scaler[O OutType] interface {
	Fit([]O)
	Transform([]O) []O
}

// Estimator is the generic type of all classifier and regression
// engines.
type Estimator[O OutType] interface {
	Fit(dpoints []O, labels []float64) []float64
	Predict(dpoints []O) []float64
	Score(dpoints []O, labels []float64) float64
}

// Pipeline implements the ML pipeline concept. It consists of a
// transformer that transforms the data into a form digestable
// for the estimator.
type Pipeline[I InType, O OutType] struct {
	Transformer Transformer[I, O] `json:"transformer"`
	Scaler      Scaler[O]         `json:"scaler"`
	Estimator   Estimator[O]      `json:"estimator"`
}

// NewPipeline is the factory function for Pipeline.
func NewPipeline[I InType, O OutType](trf Transformer[I, O], sc Scaler[O], est Estimator[O]) *Pipeline[I, O] {
	return &Pipeline[I, O]{Transformer: trf, Scaler: sc, Estimator: est}
}

// Fit implements the training of the pipeline. It returns the epoch errors.
func (pl *Pipeline[I, O]) Fit(dpoints [][]I, labels []float64) []float64 {
	tdpoints := pl.Transformer.Transform(dpoints)
	if pl.Scaler != nil {
		pl.Scaler.Fit(tdpoints)
	}
	return pl.Estimator.Fit(tdpoints, labels)
}

// Predict implements the prediction method. It returns the predicted labels.
func (pl *Pipeline[I, O]) Predict(dpoints [][]I) []float64 {
	tdpoints := pl.Transformer.Transform(dpoints)
	if pl.Scaler != nil {
		tdpoints = pl.Scaler.Transform(tdpoints)
	}
	return pl.Estimator.Predict(tdpoints)
}

// Score computes the accuracy of the estimator on the given dataset.
func (pl *Pipeline[I, O]) Score(dpoints [][]I, labels []float64) float64 {
	tdpoints := pl.Transformer.Transform(dpoints)
	if pl.Scaler != nil {
		tdpoints = pl.Scaler.Transform(tdpoints)
	}
	return pl.Estimator.Score(tdpoints, labels)
}

// Marshal and Unmarshal implement the JSONable interface (pkg/persist).
func (pl Pipeline[I, O]) Marshal() ([]byte, error) {
	return json.MarshalIndent(pl, "", "   ")
}

func (pl *Pipeline[I, O]) Unmarshal(bs []byte) error {
	return json.Unmarshal(bs, pl)
}
