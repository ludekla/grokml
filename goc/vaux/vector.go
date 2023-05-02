package vaux

import (
	"math"
	"math/rand"
	"time"
)

func init() {
	seed := time.Now().UnixNano()
	rand.Seed(seed)
}

type Vector []float64

func NewVector(size int) Vector {
	vec := make(Vector, size)
	return vec
}

func RandVector(size int) Vector {
	vec := make(Vector, size)
	for i, _ := range vec {
		vec[i] = rand.Float64()
	}
	return vec
}

func FromSlice(s []float64) Vector {
	vec := NewVector(len(s))
	for i, val := range s {
		vec[i] = val
	}
	return vec
}

func (v Vector) Dot(other Vector) float64 {
	if len(v) != len(other) {
		panic("vectors do not have the same size")
	}
	var sum float64
	for i, val := range other {
		sum += val * v[i]
	}
	return sum
}

func (v Vector) ScaMul(factor float64) Vector {
	res := make(Vector, len(v))
	for i, val := range v {
		res[i] = factor * val
	}
	return res
}

func (v Vector) IScaMul(factor float64) {
	for i, _ := range v {
		v[i] *= factor
	}
}

func (v Vector) Add(other Vector) Vector {
	res := make(Vector, len(v))
	for i, val := range other {
		res[i] = v[i] + val
	}
	return res
}

func (v Vector) IAdd(other Vector) {
	for i, val := range other {
		v[i] += val
	}
}

func (v Vector) Mul(other Vector) Vector {
	res := make(Vector, len(v))
	for i, val := range other {
		res[i] = v[i] * val
	}
	return res
}

func (v Vector) Div(other Vector) Vector {
	res := make(Vector, len(v))
	for i, val := range other {
		res[i] = v[i] / val
	}
	return res
}

func (v Vector) IDiv(other Vector) {
	for i, val := range other {
		v[i] /= val
	}
}

func (v Vector) L1Norm() float64 {
	var sum float64
	for _, val := range v {
		sum += math.Abs(val)
	}
	return sum
}

func Mean(vecs []Vector) Vector {
	mean := NewVector(len(vecs[0]))
	for _, vec := range vecs {
		mean.IAdd(vec)
	}
	return mean.ScaMul(1.0 / float64(len(vecs)))
}

func VecStats(vecs []Vector) (Vector, Vector) {
	mean := Mean(vecs)
	std := NewVector(len(mean))
	for _, vec := range vecs {
		delta := vec.Add(mean.ScaMul(-1.0))
		std.IAdd(delta.Mul(delta))
	}
	std.IScaMul(1.0 / float64(len(vecs)))
	for i, val := range std {
		std[i] = math.Sqrt(val)
	}
	return mean, std
}

func (v Vector) Normalise(mean, std Vector) Vector {
	inv := NewVector(len(std))
	for i, val := range std {
		inv[i] = 1.0 / val
	}
	return v.Add(mean.ScaMul(-1.0)).Mul(inv)
}
