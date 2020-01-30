using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.GradientAdjustmentsParameters;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Optimizers
{
    class AdamOptimizer: Optimizer
    {
        AdamParameters _adamParameters;
        Matrix<double> _sWeights;
        Matrix<double> _rWeights;
        Matrix<double> _sBias;
        Matrix<double> _rBias;
        double _t;
        public AdamOptimizer(double stepSize,
            int layerSize, 
            int inputSize, 
            double firstMomentDecay = 0.9, 
            double secondMomentDecay = 0.999,
            double denominatorFactor = 1E-8)
        {
            _adamParameters = new AdamParameters(stepSize, firstMomentDecay, secondMomentDecay, denominatorFactor);
            _sWeights = Matrix<double>.Build.Dense(inputSize, layerSize);
            _rWeights = Matrix<double>.Build.Dense(inputSize, layerSize);
            _sBias = Matrix<double>.Build.Dense(layerSize, 1);
            _rBias = Matrix<double>.Build.Dense(layerSize, 1);
            _t = 1;
        }

        public IGradientAdjustmentParameters GetGradient()
        {
            return _adamParameters;
        }

        public void UpdateParams(Matrix<double> errors, Matrix<double> inputs,ref Matrix<double> weights,ref Matrix<double> bias)
        {
            var gradientWeights = (inputs.Multiply(errors.Transpose()));
            var gradientBias = errors.Multiply(Matrix<double>.Build.Dense(errors.ColumnCount, 1, 1));

            _sWeights = _sWeights.Multiply(_adamParameters.FirstMomentDecay) + gradientWeights.Multiply(1 - _adamParameters.FirstMomentDecay);
            _rWeights = _rWeights.Multiply(_adamParameters.SecondMomentDecay) + (gradientWeights.PointwiseMultiply(gradientWeights)).Multiply(1 - _adamParameters.SecondMomentDecay);

            _sBias = _sBias.Multiply(_adamParameters.FirstMomentDecay) + gradientBias.Multiply(1 - _adamParameters.FirstMomentDecay);
            _rBias = _rBias.Multiply(_adamParameters.SecondMomentDecay) + (gradientBias.PointwiseMultiply(gradientBias)).Multiply(1 - _adamParameters.SecondMomentDecay);

            var sPrimeWeights = _sWeights.Divide(1 - Math.Pow(_adamParameters.FirstMomentDecay, _t));
            var rPrimeWeights = _rWeights.Divide(1 - Math.Pow(_adamParameters.SecondMomentDecay, _t));

            var sPrimeBias = _sBias.Divide(1 - Math.Pow(_adamParameters.FirstMomentDecay, _t));
            var rPrimeBias = _rBias.Divide(1 - Math.Pow(_adamParameters.SecondMomentDecay, _t));

            var vWeights = sPrimeWeights.PointwiseDivide(rPrimeWeights.PointwiseSqrt().Add(_adamParameters.DenominatorFactor)).Multiply(- _adamParameters.StepSize);

            var vBias = sPrimeBias.PointwiseDivide(rPrimeBias.PointwiseSqrt().Add(_adamParameters.DenominatorFactor)).Multiply(-_adamParameters.StepSize);

            weights += vWeights;
            bias += vBias;

            _t += 1;
        }
    }
}
