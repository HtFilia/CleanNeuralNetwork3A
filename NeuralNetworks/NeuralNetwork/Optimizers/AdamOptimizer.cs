using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.GradientAdjustmentsParameters;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Optimizers
{
    class AdamOptimizer
    {
        AdamParameters _adamParameters;
        Matrix<double> _sWeights;
        Matrix<double> _rWeights;
        Matrix<double> _sBias;
        Matrix<double> _rBias;
        public AdamOptimizer(double stepSize, int layerSize, int inputSize)
        {
            _adamParameters = new AdamParameters(stepSize, 0.9, 0.999, Math.Pow(10, -8));
            _sWeights = Matrix<double>.Build.Dense(layerSize, inputSize);
            _rWeights = Matrix<double>.Build.Dense(layerSize, inputSize);
            _sBias = Matrix<double>.Build.Dense(layerSize, 1);
            _rBias = Matrix<double>.Build.Dense(layerSize, 1);
        }

         public void updateParams(Matrix<double> weightedErrors, Matrix<double> inputs, Matrix<double> weights, Matrix<double> bias)
        {
            var gradientWeights = (inputs.Multiply(weightedErrors.Transpose()));
            var gradientBias = weightedErrors.Multiply(Matrix<double>.Build.Dense(weightedErrors.ColumnCount, 1, 1));

            _sWeights = _sWeights.Multiply(_adamParameters.FirstMomentDecay) + gradientWeights.Multiply(1 - _adamParameters.FirstMomentDecay);
            _rWeights = _rWeights.Multiply(_adamParameters.SecondMomentDecay) + (gradientWeights.PointwiseMultiply(gradientWeights)).Multiply(1 - _adamParameters.SecondMomentDecay);

            _sBias = _sBias.Multiply(_adamParameters.FirstMomentDecay) + gradientBias.Multiply(1 - _adamParameters.FirstMomentDecay);
            _rBias = _rBias.Multiply(_adamParameters.SecondMomentDecay) + (gradientBias.PointwiseMultiply(gradientBias)).Multiply(1 - _adamParameters.SecondMomentDecay);

            var sPrimeWeights = _sWeights.Divide(1 - _adamParameters.FirstMomentDecay);
            var rPrimeWeights = _rWeights.Divide(1 - _adamParameters.SecondMomentDecay);

            var sPrimeBias = _sBias.Divide(1 - _adamParameters.FirstMomentDecay);
            var rPrimeBias = _rBias.Divide(1 - _adamParameters.SecondMomentDecay);

            var vWeights = sPrimeWeights.PointwiseDivide(rPrimeWeights.PointwiseSqrt().Add(_adamParameters.DenominatorFactor)).Multiply(- _adamParameters.StepSize);

            var vBias = sPrimeBias.PointwiseDivide(rPrimeBias.PointwiseSqrt().Add(_adamParameters.DenominatorFactor)).Multiply(-_adamParameters.StepSize);

            weights.Add(vWeights);
            bias.Add(vBias);
        }
    }
}
