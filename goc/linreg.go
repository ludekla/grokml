package main

import (
	"fmt"
	"log"
	"os"

	"goc/linreg"
	"goc/vaux"
	"gopkg.in/yaml.v3"
)

type Parms struct {
	Path      string    `yaml:"path"`
	Target    string    `yaml:"target"`
	Features  []string  `yaml:"features"`
	Predict   []float64 `yaml:"predict"`
	NEpochs   int       `yaml:"nepochs"`
	LRate     float64   `yaml:"lrate"`
	ModelFile string    `yaml:"modelfile"`
	Lasso     float64   `yaml:"lasso"`
	Ridge     float64   `yaml:"ridge"`
}

// fetch parameters from YAML file
func GetParms(ymlfile string) Parms {
	data, err := os.ReadFile(ymlfile)
	if err != nil {
		log.Fatal(err)
	}
	var parms Parms
	err = yaml.Unmarshal(data, &parms)
	if err != nil {
		log.Fatal(err)
	}
	return parms
}

func linregMain() {
	parms := GetParms("../lrparms.yml")

	var lr linreg.Regression

	if *train {
		fmt.Println("Training")
		dset := vaux.NewDataSet(parms.Path, parms.Target, parms.Features)
		trainSet, testSet := dset.Split(0.1)
		if parms.Lasso+parms.Ridge == 0.0 {
			lr = linreg.NewLinReg(parms.LRate, parms.NEpochs)
		} else {
			lr = linreg.NewRegLinReg(parms.LRate, parms.NEpochs, parms.Lasso, parms.Ridge)
		}
		lr.Fit(trainSet)
		fmt.Printf("test score: %.3f\n", lr.Score(testSet))
		lr.Save(parms.ModelFile)
	} else {
		fmt.Println("Use already trained model")
		if parms.Lasso+parms.Ridge == 0.0 {
			lr = linreg.LinRegFromJSON(parms.ModelFile)
		} else {
			lr = linreg.RegLinRegFromJSON(parms.ModelFile)
		}
	}

	v := vaux.FromSlice(parms.Predict)
	preds := lr.Predict([]vaux.Vector{v})
	fmt.Printf("Predicted: %v -> %.3f\n", parms.Predict, preds)
}
