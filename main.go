package main

import (
	"flag"
	"fmt"

	"grokml/pkg/ch06-logreg"
	tk "grokml/pkg/tokens"
	vc "grokml/pkg/vector"
)

var train = flag.Bool("t", false, "train model before prediction")

type Runner[D ch06.DataPoint] struct {
	ud      ch06.Updater[D]
	weights D
}

func (ru *Runner[D]) Run(size int, dpoint D) {
	ud := ru.ud
	ud.Init(size)
	fmt.Println(ud, dpoint)
	ud.Update(dpoint, 0.2)
	dp := ud.Get()
	ru.weights = dp
}

func main() {

	flag.Parse()

	fmt.Println("Hello Naive Bayes! Train? ", *train)

	r1 := Runner[vc.Vector]{ud: new(ch06.VectorUpdater)}
	r1.Run(2, vc.RandVector(2))
	fmt.Println(r1.weights)

	r2 := Runner[tk.TokenMap]{ud: new(ch06.TokenMapUpdater)}
	r2.Run(2, tk.TokenMap{"A": 1.0, "B": -2.0})
	fmt.Println(r2.weights)

}
