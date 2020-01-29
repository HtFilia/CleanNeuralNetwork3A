using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.GradientAdjustmentsParameters;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Optimizers
{
    interface Optimizer
    {
        void UpdateParams(Matrix<double> errors, Matrix<double> inputs,ref Matrix<double> weights,ref Matrix<double> bias);

        IGradientAdjustmentParameters GetGradient();
    }
}
