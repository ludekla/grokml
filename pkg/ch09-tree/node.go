package ch09

import (
	"fmt"
	"math"
	"sort"
)

type SplitInfo struct {
	split     int
	Dimension int     `json:"dimension"`
	Threshold float64 `json:"threshold"`
}

type EvalFunc func(examples []Example, val float64) float64

type Impurity struct {
	Val  float64 `json:"val"`
	eval EvalFunc
}

type Node struct {
	Label    float64   `json:"label"`
	Split    SplitInfo `json:"split_info"`
	Depth    int       `json:"depth"`
	Gain     float64   `json:"gain"`
	Left     *Node     `json:"left"`
	Right    *Node     `json:"right"`
	examples []Example
}

// Helper Functions
func prob(examples []Example, val float64) float64 {
	var sum float64
	for _, example := range examples {
		if example.target > val {
			sum += 1
		}
	}
	return sum / float64(len(examples))
}

func Entropy(examples []Example, val float64) float64 {
	p := prob(examples, val)
	if math.Abs(p-1.0) < 1e-4 || p < 1e-4 {
		return 0.0
	}
	return -p*math.Log(p) - (1.0-p)*math.Log(1.0-p)
}

func Gini(examples []Example, val float64) float64 {
	p := prob(examples, val)
	return 2.0 * p * (1.0 - p)
}

func mse(examples []Example, val float64) float64 {
	var mean float64
	size := float64(len(examples))
	for _, example := range examples {
		mean += example.target
	}
	mean /= size
	val = 0.0
	for _, example := range examples {
		val += (example.target - mean) * (example.target - mean)
	}
	val /= size
	return val
}

// Methods and Associated Functions
func NewImpurity(value float64, evalfun EvalFunc) Impurity {
	return Impurity{Val: value, eval: evalfun}
}

func (im Impurity) Eval(examples []Example, split int) float64 {
	oldVal := im.eval(examples, im.Val)
	newVal := 0.5 * (im.eval(examples[:split], im.Val) + im.eval(examples[split:], im.Val))
	return oldVal - newVal
}

func NewNode(examps []Example, d int) *Node {
	return &Node{Depth: d, examples: examps}
}

func (n *Node) String() string {
	var pre string = ""
	for i := 0; i < n.Depth; i++ {
		pre += "  "
	}
	s := fmt.Sprintf(
		"Node {\n"+pre+"  depth: %d  label: %.3f\n"+pre+"  split: %+v gain: %.3f\n",
		n.Depth, n.Label, n.Split, n.Gain,
	)
	s += fmt.Sprintf(pre+"  Examples: %v\n", n.examples)
	s += fmt.Sprintf(pre+"  Left: %v\n"+pre+"  Right: %v\n", n.Left, n.Right)
	s += pre + "}"
	return s
}

func (n *Node) Fit(imp Impurity, minGain float64) {
	var gain float64
	var splitInfo SplitInfo
	nFeats := len(n.examples[0].features)
	size := len(n.examples)
	for i := 0; i < nFeats; i++ {
		sort.Slice(n.examples, func(k, j int) bool {
			return n.examples[k].features[i] < n.examples[j].features[i]
		})
		for j := 1; j < size; j++ {
			newGain := imp.Eval(n.examples, j)
			if newGain > gain {
				gain = newGain
				th := 0.5 * (n.examples[j-1].features[i] + n.examples[j].features[i])
				splitInfo = SplitInfo{split: j, Dimension: i, Threshold: th}
			}
		}
	}
	n.Gain = gain
	if gain > minGain {
		n.Split = splitInfo
		d := splitInfo.Dimension
		sort.Slice(n.examples, func(k, j int) bool {
			return n.examples[k].features[d] < n.examples[j].features[d]
		})
		n.grow(n.examples, splitInfo.split)
		n.Left.Fit(imp, minGain)
		n.Right.Fit(imp, minGain)
	} else {
		var avg float64
		for _, example := range n.examples {
			avg += example.target
		}
		avg /= float64(len(n.examples))
		n.Label = avg
	}
}

func (n *Node) grow(examples []Example, split int) {
	n.Left = NewNode(n.examples[:split], n.Depth+1)
	n.Right = NewNode(n.examples[split:], n.Depth+1)
}

func (n *Node) Equals(other *Node) bool {
	switch {
	case n == nil && other == nil:
		return true
	case n == nil && other != nil:
		return false
	case n != nil && other == nil:
		return false
	}
	labels := n.Label == other.Label
	dim := n.Split.Dimension == other.Split.Dimension
	threshold := n.Split.Threshold == other.Split.Threshold
	if n.Left == nil {
		return labels && dim && threshold
	}
	return n.Left.Equals(other.Left) && n.Right.Equals(other.Right)
}
