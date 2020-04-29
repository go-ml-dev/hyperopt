package tests

import (
	"fmt"
	"go-ml.dev/pkg/base/fu"
	"go-ml.dev/pkg/base/model"
	"go-ml.dev/pkg/dataset/iris"
	"go-ml.dev/pkg/hyperopt"
	"go-ml.dev/pkg/iokit"
	"go-ml.dev/pkg/xgb"
	"testing"
)

func Test_Optimize_Iris1(t *testing.T) {
	//defer verbose.BeVerbose(verbose.Print).Revert()

	par := hyperopt.Space{
		Source:     iris.Data,
		Features:   iris.Features,
		Kfold:      3,
		Iterations: 19,
		Metrics:    &model.Classification{},
		Score:      model.AccuracyScore,
		ModelFunc:  xgb.Model{Algorithm: xgb.LinearBoost, Function: xgb.Softmax}.ModelFunc,
		Variance: hyperopt.Variance{
			"MaxDepth":     hyperopt.IntRange{1, 5},
			"LearningRate": hyperopt.Value(0.6),
		},
	}.LuckyOptimize(30)

	fmt.Println(par)

	modelFile := iokit.File(fu.ModelPath("xgboost_test_v1.zip"))
	report := xgb.Model{
		Algorithm: xgb.TreeBoost,
		Function:  xgb.Softmax,
	}.Apply(par.Params).Feed(model.Dataset{
		Source:   iris.Data.RandomFlag(model.TestCol, 42, 0.2),
		Features: iris.Features,
	}).LuckyTrain(model.Training{
		Iterations: 19,
		ModelFile:  modelFile,
		Metrics:    &model.Classification{},
		Score:      model.AccuracyScore,
	})

	fmt.Println(report.History.Round(4))
}
