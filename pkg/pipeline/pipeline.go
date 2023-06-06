package pipeline

import (
	"encoding/json"
	"fmt"
	"os"

	tk "grokml/pkg/tokens"
	vc "grokml/pkg/vector"
)

type InType interface {
	float64 | string
}

type OutType interface {
	vc.Vector | tk.TokenMap
}

type Transformer[I InType, O OutType] interface {
	Transform([][]I) []O
}

type Estimator[O OutType] interface {
	Fit(dpoints []O, labels []float64) []float64
	Predict(dpoints []O) []float64
	Score(dpoints []O, labels []float64) float64
}

type Pipeline[I InType, O OutType] struct {
	Transformer Transformer[I, O] `json:"transformer"`
	Estimator   Estimator[O]      `json:"estimator"`
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

func NewPipeline[I InType, O OutType](trf Transformer[I, O], est Estimator[O]) *Pipeline[I, O] {
	return &Pipeline[I, O]{Transformer: trf, Estimator: est}
}

func (pl *Pipeline[I, O]) Fit(dpoints [][]I, labels []float64) []float64 {
	tdpoints := pl.Transformer.Transform(dpoints)
	return pl.Estimator.Fit(tdpoints, labels)
}

func (pl *Pipeline[I, O]) Predict(dpoints [][]I) []float64 {
	tdpoints := pl.Transformer.Transform(dpoints)
	return pl.Estimator.Predict(tdpoints)
}

func (pl *Pipeline[I, O]) Score(dpoints [][]I, labels []float64) float64 {
	tdpoints := pl.Transformer.Transform(dpoints)
	return pl.Estimator.Score(tdpoints, labels)
}

func (pl *Pipeline[I, O]) Save(jsonfile string) error {
	plBytes, err := json.MarshalIndent(*pl, "", "   ")
	if err != nil {
		return fmt.Errorf("cannot marshal model into bytes, %v", err)
	}
	err = os.WriteFile(jsonfile, plBytes, 0666)
	if err != nil {
		return fmt.Errorf("cannot write model file %v", err)
	}
	return nil
}

func (pl *Pipeline[I, O]) Load(jsonfile string) error {
	fileBytes, err := os.ReadFile(jsonfile)
	if err != nil {
		return fmt.Errorf("cannot read model file %v", err)
	}
	err = json.Unmarshal(fileBytes, pl)
	if err != nil {
		return fmt.Errorf("cannot load model %v", err)
	}
	return nil
}
