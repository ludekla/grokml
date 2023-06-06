package tokens

import (
	"strings"
)

type TokenMap map[string]float64

func New(size int) TokenMap {
	return make(TokenMap, size)
}

func (tm TokenMap) IAdd(other TokenMap) {
	for token, val := range other {
		tm[token] += val
	}
}

func (tm TokenMap) ScaMul(factor float64) TokenMap {
	otm := make(TokenMap)
	for token, val := range tm {
		otm[token] = factor * val
	}
	return otm
}

func (tm TokenMap) Dot(other TokenMap) float64 {
	var sum float64
	for token, val := range other {
		sum += tm[token] * val
	}
	return sum
}

type Tokeniser struct {
	ToLower bool
}

func NewTokeniser(toLower bool) *Tokeniser {
	return &Tokeniser{toLower}
}

func (t Tokeniser) Transform(docs [][]string) []TokenMap {
	tmaps := make([]TokenMap, len(docs))
	for i, doc := range docs {
		tm := TokenMap{}
		for _, txt := range doc {
			tm.IAdd(t.TokenFreqs(txt))
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
/*
func (t Tokeniser) MarshalJSON() ([]byte, error) {
	bs, err := json.Marshal(t)
	if err != nil {
		return bs, fmt.Errorf("cannot marshal vectoriser")
	}
	return bs, nil
}

func (t *Tokeniser) UnmarshalJSON(bs []byte) error {
	return json.Unmarshal(bs, t)
}
*/