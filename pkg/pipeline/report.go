package pipeline

const threshold = 0.5

// Report serves as a container for 4 standard ML performance measures.
// While accuracy is both for regressors and classifiers, precision
// recall and specificity are only used for the latter.
type Report struct {
	Accuracy    float64
	Precision   float64
	Recall      float64
	Specificity float64
}

// FScore computes the standard performance measure called F-score
// aka F1-score. Its extremes beta = 0 and beta = infinity are recall and
// precision respectively. For all in-between values, it is a
// mixture. In the case beta = 1 this mixture is a quotient of the geometric
// and the arithmetic average.
func (rp Report) FScore(beta float64) float64 {
	denominator := beta*beta*rp.Recall + rp.Precision
	if denominator == 0.0 {
		return 0.0
	}
	return (1 + beta*beta) * rp.Recall * rp.Precision / denominator
}

// GetReport is a helper function to compute the Report struct quantities
// precision, recall, specificity and accuracy.
func GetReport(predictions []float64, labels []float64) Report {
	var tp, tn, fp, fn float64
	for i, p := range predictions {
		if p > threshold && labels[i] > threshold {
			tp++
		} else if p > threshold && labels[i] <= threshold {
			fp++
		} else if p <= threshold && labels[i] > threshold {
			fn++
		} else {
			tn++
		}
	}
	var precision, recall, specificity float64
	if tp == 0.0 {
		precision = 0.0
		recall = 0.0
	} else {
		precision = tp / (tp + fp)
		recall = tp / (tp + fn)
	}
	if tn == 0.0 {
		specificity = 0.0
	} else {
		specificity = tn / (tn + fp)
	}
	return Report{
		Accuracy:    (tp + tn) / (tp + tn + fp + fn),
		Precision:   precision,
		Recall:      recall,
		Specificity: specificity,
	}
}

// getCoD is a helper function to compute the coefficient of determination.
// As a measure of performance for regression trees, it compares the mean-squared
// error with that of a regressor which predicts the label mean for every data point.
func GetCoD(predictions []float64, labels []float64) float64 {
	mean := Mean(labels)
	var rss float64 // residual square sum
	var tss float64 // total square sum
	for i, pred := range predictions {
		rss += (pred - labels[i]) * (pred - labels[i])
		tss += (mean - labels[i]) * (mean - labels[i])
	}
	return 1.0 - rss/tss
}

// Mean is a helper function to compute the average value of a slice.
func Mean(vals []float64) float64 {
	var mean float64
	for _, val := range vals {
		mean += val
	}
	mean /= float64(len(vals))
	return mean
}
