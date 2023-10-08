// Implements the basic data structure employed all over this ML repo.
// It furnishes data points with a number of linear-algebraic methods
// and helper functions.
package vector

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

// New constructs a vector with zeros.
func New(size int) Vector {
	return make(Vector, size)
}

// RandVector makes vector with random values drawn
// from a uniform distribution over [0, 1].
func RandVector(size int) Vector {
	vec := make(Vector, size)
	for i, _ := range vec {
		vec[i] = rand.Float64()
	}
	return vec
}

// Dot computes the dot product against the other vector.
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

// ScalMul performs a multiplication with a scalar.
func (v Vector) ScaMul(factor float64) Vector {
	res := make(Vector, len(v))
	for i, val := range v {
		res[i] = factor * val
	}
	return res
}

// IScalMul performs an in-place scalar multiplication.
func (v Vector) IScaMul(factor float64) {
	for i, _ := range v {
		v[i] *= factor
	}
}

// Add performs vector addition (component-wise) with the other vector.
func (v Vector) Add(other Vector) Vector {
	res := make(Vector, len(v))
	for i, val := range other {
		res[i] = v[i] + val
	}
	return res
}

// IAdd performs an in-place vector addition with the other vector.
func (v Vector) IAdd(other Vector) {
	for i, val := range other {
		v[i] += val
	}
}

// Mul computes component-wise vector multiplication (aka Hadamard product)
// with the other vector.
func (v Vector) Mul(other Vector) Vector {
	res := make(Vector, len(v))
	for i, val := range other {
		res[i] = v[i] * val
	}
	return res
}

// Div computes component-wise vector division. It panics if the other vector
// has a zero component.
func (v Vector) Div(other Vector) Vector {
	res := make(Vector, len(v))
	for i, val := range other {
		res[i] = v[i] / val
	}
	return res
}

// IDiv is the in-place version of component-wise vector division.
func (v Vector) IDiv(other Vector) {
	for i, val := range other {
		v[i] /= val
	}
}

// L1Norm computes the L1 norm of the vector.
func (v Vector) L1Norm() float64 {
	var sum float64
	for _, val := range v {
		sum += math.Abs(val)
	}
	return sum
}

// Mean computes the vectorial mean of a slice of vectors.
func Mean(vecs []Vector) Vector {
	mean := New(len(vecs[0]))
	for _, vec := range vecs {
		mean.IAdd(vec)
	}
	return mean.ScaMul(1.0 / float64(len(vecs)))
}

// VectorStats computes the vectorial mean and standard deviation.
func VectorStats(vecs []Vector) (Vector, Vector) {
	mean := Mean(vecs)
	std := New(len(mean))
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

// Normalise normalises the vector by the vectorial mean and standard deviation.
func (v Vector) Normalise(mean, std Vector) Vector {
	inv := New(len(std))
	for i, val := range std {
		inv[i] = 1.0 / val
	}
	return v.Add(mean.ScaMul(-1.0)).Mul(inv)
}

// Vectoriser implements the Transformer interface needed for pipelines.
// It makes slices floats into vectors, either by simply wrapping them
// without using copies (when wrap is set to true) or using copies.
type Vectoriser struct {
	Wrap bool `json:"wrap"`
}

// NewVectoriser constructs Vectoriser.
func NewVectoriser(wrap bool) *Vectoriser {
	return &Vectoriser{wrap}
}

// Transform makes slices of floats into vectors.
func (v Vectoriser) Transform(slices [][]float64) []Vector {
	vecs := make([]Vector, len(slices))
	var vec Vector
	for i, dpoint := range slices {
		if v.Wrap {
			vec = Vector(dpoint)
		} else {
			vec = New(len(dpoint))
			copy(vec, dpoint)
		}
		vecs[i] = vec
	}
	return vecs
}
