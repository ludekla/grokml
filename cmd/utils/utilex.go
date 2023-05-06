package main

import (
	"fmt"
	"os"
	"strconv"

	"grokml/pkg/utils"
)

func main() {

	fmt.Println("Hello utils!")

	if len(os.Args) < 2 {
		fmt.Println("Usage: utilex [number of examples]")
		os.Exit(0)
	}
	n, err := strconv.Atoi(os.Args[1])
	if err != nil {
		fmt.Println(err)
		fmt.Println("Usage: utilex [number of examples]")
		os.Exit(1)
	}

	ds := utils.NewDataSet(
		".manning/Chapter_3_Linear_Regression/Hyderabad.csv",
		"Price",
		[]string{"Area", "No. of Bedrooms"},
	)

	fmt.Println(ds)

	fmt.Println("No  | Area    | # Bedrooms | Price")
	fmt.Println("-----------------------------------")
	for i := 0; i < n; i++ {
		x, y := ds.Random()
		fmt.Printf("%4d| %-8.2f| %-11.0f| %v\n", i+1, x[0], x[1], y)
	}

}
