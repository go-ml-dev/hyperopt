package tests

/*
import (
	"fmt"
	"go-ml.dev/pkg/base/fu"
	"go-ml.dev/pkg/base/fu/verbose"
	"go-ml.dev/pkg/base/model"
	"go-ml.dev/pkg/dataset/mnist"
	"go-ml.dev/pkg/hyperopt"
	"go-ml.dev/pkg/iokit"
	"go-ml.dev/pkg/xgb"
	"testing"
)

func Test_Optimize_Mnist1(t *testing.T) {
	defer verbose.BeVerbose(verbose.Print).Revert()

	par := hyperopt.Space{
		Source:     mnist.Data.Rand(13, 0.35),
		Features:   mnist.Features,
		Kfold:      3,
		Iterations: 30,
		Metrics:    model.Classification{},
		Score:      model.AccuracyScore,
		ModelFunc:  xgb.Model{Algorithm: xgb.TreeBoost, Function: xgb.Softmax}.ModelFunc,
		Variance: hyperopt.Variance{
			"MaxDepth":     hyperopt.IntRange{1, 10},
			"LearningRate": hyperopt.Range{0.1, 0.9},
		},
	}.LuckyOptimize(30)

	fmt.Println(par)

	modelFile := iokit.File(fu.ModelPath("xgboost_mnist_v1.zip"))
	report := xgb.Model{
		Algorithm: xgb.TreeBoost,
		Function:  xgb.Softmax,
	}.Apply(par.Params).Feed(model.Dataset{
		Source:   mnist.T10k.RandomFlag(model.TestCol, 42, 0.2),
		Features: mnist.Features,
	}).LuckyTrain(model.Training{
		Iterations: 30,
		ModelFile:  modelFile,
		Metrics:    model.Classification{},
		Score:      model.AccuracyScore,
	})

	fmt.Println(report.History.Round(4))
}
*/
