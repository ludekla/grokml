package logreg

import (
	"encoding/csv"
	"io"
	"log"
	"math/rand"
	"os"
	"strings"
)

type TokenMap map[string]float64

type DataSet struct {
	Path string
	Size int
	Header []string
	X []TokenMap
	Y []string
}

func NewDataSet(csvfile string) DataSet {
	file, err := os.Open(csvfile)
	if err != nil {
		log.Fatalf("Cannot open CSV file: %v", err)
	}
	defer file.Close()
	revs := make([]TokenMap, 0, 100)
	lbs := make([]string, 0, 100)
	reader := csv.NewReader(file)
	header, err := reader.Read()
	if err != nil {
		log.Fatal("cannot read csv line")
	}
	for {
		rec, err := reader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			log.Fatal("Cannot read csv line")
		}
		freqs := GetTokenFreqs(rec[0])
		revs = append(revs, freqs)
		lbs = append(lbs, rec[1])
	}
	return DataSet{
		Path: csvfile, 
		Size: len(revs), 
		Header: header, X: revs, Y: lbs,
	}
}

func (ds DataSet) Random() (TokenMap, string) {
	i := rand.Intn(ds.Size)
	return ds.X[i], ds.Y[i]
}

func (ds DataSet) Split(ratio float64) (DataSet, DataSet) {
	nTest := int(float64(ds.Size) * ratio)
	rand.Shuffle(ds.Size, func(i, j int) {
		ds.X[i], ds.X[j] = ds.X[j], ds.X[i]
		ds.Y[i], ds.Y[j] = ds.Y[j], ds.Y[i]
	})
	testSet := DataSet{
		X: ds.X[:nTest], Y: ds.Y[:nTest], 
		Header: ds.Header, Size: nTest,
	}
	trainSet := DataSet{
		X: ds.X[nTest:], Y: ds.Y[nTest:], 
		Header: ds.Header, Size: ds.Size - nTest,
	}
	return trainSet, testSet
}

func GetTokenFreqs(txt string) TokenMap {
	tmap := make(TokenMap)
	tokens := strings.Split(txt, " ")
	var total float64
 	for _, token := range tokens {
		tmap[token] += 1.0
		total++
	}
	for token, count := range tmap {
		tmap[token] = count / total
	} 
	return tmap
}

func (tm TokenMap) Update(tokens TokenMap) {
	for token, _ := range tokens {
		tm[token] = 1.0
	}
}
