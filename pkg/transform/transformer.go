package transform

import (
	"strings"

	"grokml/pkg/vector"
)

type TokenMap map[string]float64

func (t TokenMap) Update(other TokenMap) {
	for token, val := range other {
		t[token] += val
	}
}

type intype interface {
	float64 | string
}

type outtype interface {
	vector.Vector | TokenMap
}

type Transformer[I intype, O outtype] interface {
	Transform([][]I) []O
}

type Tokeniser struct {
	ToLower bool
}

func NewTokeniser(toLower bool) Tokeniser {
	return Tokeniser{toLower}
}

func (t Tokeniser) Transform(docs [][]string) []TokenMap {
	tmaps := make([]TokenMap, len(docs))
	for i, doc := range docs {
		tm := TokenMap{}
		for _, txt := range doc {
			tm.Update(t.TokenFreqs(txt))
		}
		tmaps[i] = tm
	}
	return tmaps
}

func (t Tokeniser) TokenFreqs(txt string) TokenMap {
	tmap := make(TokenMap)
	tokens := strings.Split(txt, " ")
	var total float64
	for _, token := range tokens {
		if t.ToLower {
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

type Vectoriser struct {
	wrap bool
}

func NewVectoriser(wrap bool) Vectoriser {
	return Vectoriser{wrap}
}

func (v Vectoriser) Transform(slices [][]float64) []vector.Vector {
	vecs := make([]vector.Vector, len(slices))
	var vec vector.Vector
	for i, dpoint := range slices {
		if v.wrap {
			vec = vector.Vector(dpoint)
		} else {
			vec = vector.NewVector(len(dpoint))
			copy(vec, dpoint)	
		}
		vecs[i] = vec
	}
	return vecs
}
