package utils

import (
	"runtime"
	"strings"
	"sync"
)

type TokenMap map[string]float64

type Example[D DataPoint] struct {
	dpoint D
	label  float64
}

func Transform(ds DataSet[string]) DataSet[TokenMap] {
	// goroutine sets up examples and puts them into channel
	// input example channel
	inexCh := make(chan Example[string])
	go func() {
		defer close(inexCh)
		for i, dp := range ds.dpoints {
			inexCh <- Example[string]{dp, ds.labels[i]}
		}
	}()
	// channel for output examples
	outexCh := make(chan Example[TokenMap])

	wg := sync.WaitGroup{}
	for i := 0; i < runtime.NumCPU(); i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for ex := range inexCh {
				tm := GetTokenFreqs(ex.dpoint)
				outexCh <- Example[TokenMap]{dpoint: tm, label: ex.label}
			}
		}()
	}
	go func() {
		wg.Wait()
		close(outexCh)
	}()

	dpoints := make([]TokenMap, 0, len(ds.dpoints))
	labels := make([]float64, 0, len(ds.dpoints))

	for ex := range outexCh {
		dpoints = append(dpoints, ex.dpoint)
		labels = append(labels, ex.label)
	}
	return DataSet[TokenMap]{len(dpoints), ds.header, dpoints, labels}
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
