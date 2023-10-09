package ch06

import (
	tk "grokml/pkg/tokens"
	vc "grokml/pkg/vector"
)

// DataPoint are the types for weight parameters.
type DataPoint interface {
	tk.TokenMap | vc.Vector
}

// Updater is the interface for a weight updating engine.
type Updater[D DataPoint] interface {
	Init(size int)
	Update(dpoint D, delta float64)
	Get() D
	Dot(other D) float64
}

// VectorUpdater implements the Updater interface for vectors.
type VectorUpdater struct {
	Weights vc.Vector `json:"weights"`
}

// Init initialises the weights by setting them all to zero.
func (vu *VectorUpdater) Init(size int) {
	vu.Weights = vc.New(size)
}

// Update performs an in-place update of the weight vector.
func (vu VectorUpdater) Update(vec vc.Vector, delta float64) {
	vu.Weights.IAdd(vec.ScaMul(delta))
}

// Get is a simple getter for the weights.
func (vu VectorUpdater) Get() vc.Vector {
	return vu.Weights
}

// Dot performs the Dot product of the weights against another vector.
func (vu VectorUpdater) Dot(other vc.Vector) float64 {
	return vu.Weights.Dot(other)
}

// TokenMapUpdater implements the Updater interface for token maps.
type TokenMapUpdater struct {
	Weights tk.TokenMap `json:"weights"`
}

// Init initialises the weights by setting them to a token map.
func (tu *TokenMapUpdater) Init(size int) {
	tu.Weights = tk.New(size)
}

// Update performs an in-place update of the weights.
func (tu TokenMapUpdater) Update(tmap tk.TokenMap, delta float64) {
	tu.Weights.IAdd(tmap.ScaMul(delta))
}

// Get is a simples getter for the weights.
func (tu TokenMapUpdater) Get() tk.TokenMap {
	return tu.Weights
}

// Dot computes the dot product against another token map.
func (tu TokenMapUpdater) Dot(other tk.TokenMap) float64 {
	return tu.Weights.Dot(other)
}
