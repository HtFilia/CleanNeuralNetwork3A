using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.GradientAdjustmentsParameters;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Optimizers
{
    class FixedLearningRateOptimizer: Optimizer
    {
        FixedLearningRateParameters _fLRParams;

        public FixedLearningRateOptimizer(FixedLearningRateParameters fixedLearningRateParameters)
        {
            _fLRParams = fixedLearningRateParameters;
        }

        public IGradientAdjustmentParameters GetGradient()
        {
            return _fLRParams;
        }

        public void UpdateParams(Matrix<double> errors, Matrix<double> inputs,ref Matrix<double> weights,ref Matrix<double> bias)
        {
            var gradientWeights = (inputs.Multiply(errors.Transpose()));
            var gradientBiais = errors.Multiply(Matrix<double>.Build.Dense(inputs.ColumnCount, 1, 1));
            weights -= gradientWeights.Multiply(_fLRParams.LearningRate / inputs.ColumnCount);
            bias -= gradientBiais.Multiply(_fLRParams.LearningRate / inputs.ColumnCount);
        }
    }
}
