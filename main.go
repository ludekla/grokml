package main

import (
	"fmt"

	"grokml/pkg/utils"
)

func main() {

	fmt.Println("Hello World!")

	x := []float64{2.1, -9.2, 8.4}
	v := utils.Vector(x)

	fmt.Println(x, v)



}
