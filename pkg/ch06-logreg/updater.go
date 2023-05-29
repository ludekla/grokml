package ch06

import (
	tk "grokml/pkg/tokens"
	vc "grokml/pkg/vector"
)

// Types for weight parameters.
type DataPoint interface {
	tk.TokenMap | vc.Vector
}

// Interface for a weight updating engine.
type Updater[D DataPoint] interface {
	Init(size int)
	Update(dpoint D, delta float64)
	Get() D
	Dot(other D) float64
}

// Implementation of Updater for vectors.
type VectorUpdater struct {
	Weights vc.Vector `json:"weights"`
}

func (vu *VectorUpdater) Init(size int) {
	vu.Weights = vc.New(size)
}

func (vu VectorUpdater) Update(vec vc.Vector, delta float64) {
	vu.Weights.IAdd(vec.ScaMul(delta))
}

func (vu VectorUpdater) Get() vc.Vector {
	return vu.Weights
}

func (vu VectorUpdater) Dot(other vc.Vector) float64 {
	return vu.Weights.Dot(other)
}

// Implementation of Updater for token maps.
type TokenMapUpdater struct {
	Weights tk.TokenMap `json:"weights"`
}

func (tu *TokenMapUpdater) Init(size int) {
	tu.Weights = tk.New(size)
}

func (tu TokenMapUpdater) Update(tmap tk.TokenMap, delta float64) {
	tu.Weights.IAdd(tmap.ScaMul(delta))
}

func (tu TokenMapUpdater) Get() tk.TokenMap {
	return tu.Weights
}

func (tu TokenMapUpdater) Dot(other tk.TokenMap) float64 {
	return tu.Weights.Dot(other)
}
