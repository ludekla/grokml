package tokens

import (
	"strings"
)

// TokenMap implements a mapping from tokens to floats. It acts as a vector with
// a float parameter for every word, ie token, as obtained from text data, and all
// the required canonical vector operations. 
type TokenMap map[string]float64

// New is a factory function for TokenMap objects.
func New(size int) TokenMap {
	return make(TokenMap, size)
}

// IAdd implements in-place addition with another token map.
func (tm TokenMap) IAdd(other TokenMap) {
	for token, val := range other {
		tm[token] += val
	}
}

// ScalMul implements scalar multiplication.
func (tm TokenMap) ScaMul(factor float64) TokenMap {
	otm := make(TokenMap)
	for token, val := range tm {
		otm[token] = factor * val
	}
	return otm
}

// Dot implements the dot product against another token map.
func (tm TokenMap) Dot(other TokenMap) float64 {
	var sum float64
	for token, val := range other {
		sum += tm[token] * val
	}
	return sum
}

// Tokeniser implements the Transformer interface.
type Tokeniser struct {
	toLower bool
}

// NewTokeniser is a factory function for Tokenisers.
func NewTokeniser(toLower bool) *Tokeniser {
	return &Tokeniser{toLower}
}

// Transform is the defining method for Transformers: it takes a slice of 
// string slices - construed as documents - and returns a token map for every
// document. 
func (t Tokeniser) Transform(docs [][]string) []TokenMap {
	tmaps := make([]TokenMap, len(docs))
	for i, doc := range docs {
		tm := TokenMap{}
		for _, txt := range doc {
			tm.IAdd(TokenFreqs(txt, t.toLower))
		}
		tmaps[i] = tm
	}
	return tmaps
}

// TokenFreqs is a helper function that tokenises a string, computes
// their relative frequencies with the string and stores them inside
// a token map which is then returned.
func TokenFreqs(txt string, toLower bool) TokenMap {
	tmap := make(TokenMap)
	tokens := strings.Split(txt, " ")
	var total float64
	for _, token := range tokens {
		if toLower {
			token = strings.ToLower(token)
		}
		tmap[token] += 1.0
		total++
	}
	for token, count := range tmap {
		tmap[token] = count / total
	}
	return tmap
}