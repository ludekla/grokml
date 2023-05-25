package utils

import (
	"strings"
)

type TokenMap map[string]float64

func Tokenise(docs []string, toLower bool) []TokenMap {
	tmaps := make([]TokenMap, len(docs))
	for i, doc := range docs {
		tmaps[i] = GetTokenFreqs(doc, toLower)
	}
	return tmaps
}

func GetTokenFreqs(txt string, toLower bool) TokenMap {
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
