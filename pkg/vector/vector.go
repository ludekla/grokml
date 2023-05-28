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

// Makes vector with zeros.
func New(size int) Vector {
	return make(Vector, size)
}

// Makes vector with random values from a uniform distribution over [0, 1].
func RandVector(size int) Vector {
	vec := make(Vector, size)
	for i, _ := range vec {
		vec[i] = rand.Float64()
	}
	return vec
}

// Computes dot product against other vector.
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

// Performs a multiplication with a scalar.
func (v Vector) ScaMul(factor float64) Vector {
	res := make(Vector, len(v))
	for i, val := range v {
		res[i] = factor * val
	}
	return res
}

// Performs an in-place scalar multiplication.
func (v Vector) IScaMul(factor float64) {
	for i, _ := range v {
		v[i] *= factor
	}
}

// Performs vector addition (component-wise) with other vector.
func (v Vector) Add(other Vector) Vector {
	res := make(Vector, len(v))
	for i, val := range other {
		res[i] = v[i] + val
	}
	return res
}

// Performs in-place vector addition with other vector.
func (v Vector) IAdd(other Vector) {
	for i, val := range other {
		v[i] += val
	}
}

// Computes component-wise vector multiplication (Hadamard product).
func (v Vector) Mul(other Vector) Vector {
	res := make(Vector, len(v))
	for i, val := range other {
		res[i] = v[i] * val
	}
	return res
}

// Computes component-wise vector division. Will panics if other vector
// has a zero component.
func (v Vector) Div(other Vector) Vector {
	res := make(Vector, len(v))
	for i, val := range other {
		res[i] = v[i] / val
	}
	return res
}

// In-place version of component-wise vector division.
func (v Vector) IDiv(other Vector) {
	for i, val := range other {
		v[i] /= val
	}
}

// Computes the L1 norm of the vector.
func (v Vector) L1Norm() float64 {
	var sum float64
	for _, val := range v {
		sum += math.Abs(val)
	}
	return sum
}

// Computes the vectorial mean of a slice of vectors.
func Mean(vecs []Vector) Vector {
	mean := New(len(vecs[0]))
	for _, vec := range vecs {
		mean.IAdd(vec)
	}
	return mean.ScaMul(1.0 / float64(len(vecs)))
}

// Computes the vectorial mean and the standard deviation vector.
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

// Normalises the vector by its mean and standard deviation.
func (v Vector) Normalise(mean, std Vector) Vector {
	inv := New(len(std))
	for i, val := range std {
		inv[i] = 1.0 / val
	}
	return v.Add(mean.ScaMul(-1.0)).Mul(inv)
}

type Vectoriser struct {
	wrap bool
}

func NewVectoriser(wrap bool) Vectoriser {
	return Vectoriser{wrap}
}

func (v Vectoriser) Transform(slices [][]float64) []Vector {
	vecs := make([]Vector, len(slices))
	var vec Vector
	for i, dpoint := range slices {
		if v.wrap {
			vec = Vector(dpoint)
		} else {
			vec = New(len(dpoint))
			copy(vec, dpoint)
		}
		vecs[i] = vec
	}
	return vecs
}
