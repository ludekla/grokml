package ch09

import (
	"fmt"
	"sort"
)

// Examples implements a container class for labelled training examples.
// This is suitable for both decision and regression trees.
type Example struct {
	dpoint []float64 // data point
	label  float64
}

// MakeExamples is a factory function to package up data points and their
// labels into Example objects.
func MakeExamples(dpoints [][]float64, labels []float64) []Example {
	examples := make([]Example, len(dpoints))
	for i, dpoint := range dpoints {
		examples[i] = Example{dpoint, labels[i]}
	}
	return examples
}

// SplitInfo holds necessary information about a split.
type SplitInfo struct {
	Dimension int     `json:"dimension"`
	Threshold float64 `json:"threshold"`
}

// Node implements the nodes of a decision or regression tree.
// It knows about its depth as counted from the root and the
// minimum gain in purity needed to be obtained by a potential
// split. If it has children, it also holds split information.
type Node struct {
	Label   float64   `json:"label"`
	Split   SplitInfo `json:"split_info"`
	Depth   int       `json:"depth"`
	MinGain float64   `json:"min_gain"`
	Left    *Node     `json:"left"`
	Right   *Node     `json:"right"`
}

// NewNode is a factory function for Node structs. The depth
// parameter must be passed by the parent node which is in charge
// of calling this function. minGain is a set requirement.
func NewNode(depth int, minGain float64) *Node {
	return &Node{Depth: depth, MinGain: minGain}
}

// String makes the Node struct a Stringer. It is needed for debugging purposes.
func (n *Node) String() string {
	var pre string = ""
	for i := 0; i < n.Depth; i++ {
		pre += "  " // this generates the appropriate indentation
	}
	s := fmt.Sprintf(
		"Node {\n"+pre+"  depth: %d  label: %.3f\n"+pre+"  split: %+v gain: %.3f\n",
		n.Depth, n.Label, n.Split, n.MinGain,
	)
	s += fmt.Sprintf(pre+"  Left: %v\n"+pre+"  Right: %v\n", n.Left, n.Right)
	s += pre + "}"
	return s
}

// Fit performs the decision-tree training. Every node minds its own splits.
// It will end up as a leaf if the gain is not big enough.
func (n *Node) Fit(examples []Example, imp Impurity) {
	var gain float64
	var splitInfo SplitInfo
	var splitAt int
	nDims := len(examples[0].dpoint)
	size := len(examples)
	for i := 0; i < nDims; i++ {
		// Sort examples by their i-th data point component.
		sort.Slice(examples, func(k, j int) bool {
			return examples[k].dpoint[i] < examples[j].dpoint[i]
		})
		// Find split with greatest gain.
		for j := 1; j < size; j++ {
			newGain := computeGain(imp, examples, j)
			if newGain > gain {
				gain = newGain
				thresh := (examples[j-1].dpoint[i] + examples[j].dpoint[i]) / 2.0
				splitAt = j
				splitInfo = SplitInfo{Dimension: i, Threshold: thresh}
			}
		}
	}
	if gain > n.MinGain { // it must be worth it
		n.Split = splitInfo
		d := splitInfo.Dimension
		sort.Slice(examples, func(k, j int) bool {
			return examples[k].dpoint[d] < examples[j].dpoint[d]
		})
		// grow two leaves
		n.Left = NewNode(n.Depth+1, n.MinGain)
		n.Right = NewNode(n.Depth+1, n.MinGain)
		// delegate potential further splits
		n.Left.Fit(examples[:splitAt], imp)
		n.Right.Fit(examples[splitAt:], imp)
	} else {
		var avg float64
		for _, example := range examples {
			avg += example.label
		}
		avg /= float64(size)
		n.Label = avg
	}
}
