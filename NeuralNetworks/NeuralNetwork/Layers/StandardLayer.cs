using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common;
using NeuralNetwork.Common.Activators;
using NeuralNetwork.Common.Layers;
using NeuralNetwork.Common.GradientAdjustmentsParameters;
using System;
using NeuralNetwork.Optimizers;

namespace NeuralNetwork.Layers
{
    public sealed class StandardLayer : IEquatable<ILayer>, ILayer
    {
        private int _layerSize;
        private int _inputSize;
        private int _batchSize;
        private Matrix<double> _input;
        private Matrix<double> _weights;
        private Matrix<double> _bias;
        private Matrix<double> _zeta;
        private Matrix<double> _output;
        private Matrix<double> _errors;
        Matrix<double> _weightedError;
        private IActivator _activator;
        private Optimizer _optimizer;

        public StandardLayer(int layerSize, 
                             int inputSize, 
                             int batchSize, 
                             IGradientAdjustmentParameters gradientAdjustmentParameters,
                             IActivator activator)
        {
            _layerSize = layerSize;
            _inputSize = inputSize;
            _batchSize = batchSize;

            _weights = Matrix<double>.Build.Random(_inputSize, _layerSize);
            _bias = Matrix<double>.Build.Dense(_layerSize, 1);
            _output = Matrix<double>.Build.Dense(_layerSize, _batchSize);
            _zeta = Matrix<double>.Build.Dense(_layerSize, _batchSize);
            _weightedError = Matrix<double>.Build.Dense(_inputSize, _batchSize);
            _input = Matrix<double>.Build.Dense(_inputSize, _batchSize);
            _errors = Matrix<double>.Build.Dense(_layerSize, _batchSize);
            _activator = activator;

            switch(gradientAdjustmentParameters.Type)
            {
                case GradientAdjustmentType.FixedLearningRate:
                    {
                        FixedLearningRateParameters parameters = gradientAdjustmentParameters as FixedLearningRateParameters;
                        _optimizer = new FixedLearningRateOptimizer(parameters);
                        break;
                    }
                case GradientAdjustmentType.Adam:
                    {
                        AdamParameters parameters = gradientAdjustmentParameters as AdamParameters;
                        _optimizer = new AdamOptimizer(parameters.StepSize,
                            _layerSize,
                            _inputSize,
                            parameters.FirstMomentDecay,
                            parameters.SecondMomentDecay,
                            parameters.DenominatorFactor);
                        break;
                    }
                default: throw new ArgumentException("Gradient Adjusment Type should be valid.");
            }
        }

        internal StandardLayer(int layerSize, int inputSize, int batchSize, IActivator activator, Matrix<double> weights, Matrix<double> bias)
        {
            if (weights.RowCount != layerSize)
            {
                throw new ArgumentException(String.Format("Weights' Rows should be {0}.", layerSize));
            }
            if (weights.ColumnCount != inputSize)
            {
                throw new ArgumentException(String.Format("Weights' Columns should be {0}.", inputSize));
            }
            if (bias.RowCount != layerSize)
            {
                throw new ArgumentException(String.Format("Bias' Rows should be {0}.", layerSize));
            }
            if (bias.ColumnCount != 1)
            {
                throw new ArgumentException("Bias's Columns should be 1.");
            }
            _layerSize = layerSize;
            _inputSize = inputSize;
            _batchSize = batchSize;

            _weights = weights;
            _bias = bias;
            _output = Matrix<double>.Build.Dense(_layerSize, _batchSize);

            _activator = activator;
        }

        /// <summary>
        /// Gets the size of the layer.
        /// </summary>
        /// <value>
        /// The size of the layer.
        /// </value>
        public int LayerSize
        {
            get
            {
                return _layerSize;
            }
        }

        /// <summary>
        /// Gets the size of the input that is fed to the layer.
        /// </summary>
        /// <value>
        /// The size of the input.
        /// </value>
        public int InputSize
        {
            get
            {
                return _inputSize;
            }
        }

        /// <summary>
        /// Gets the gradient adjustment.
        /// </summary>
        /// <value>
        /// The gradient adjustment.
        /// </value>
        public IGradientAdjustmentParameters GradientAdjustmentParameters
        {
            get
            {
                return _optimizer.GetGradient();
            }
        }

        /// <summary>
        /// Gets or sets the batch size.
        /// </summary>
        /// <value>
        /// The batch size.
        /// </value>
        public int BatchSize
        {
            get
            {
                return _batchSize;
            }
            set
            {
                if (value > 0)
                {
                    _batchSize = value;
                }
            }
        }

        /// <summary>
        /// Propagates the specified input through the layer./>
        /// </summary>
        /// <param name="input">The input.</param>
        public void Propagate(Matrix<double> input)
        {
            _input = input;
            for (int i = 0; i < _batchSize; i++)
            {
                _zeta.SetColumn(i, (_weights.Transpose().Multiply(input.Column(i))).Add(_bias.Column(0)));
                for (int j = 0; j < _layerSize; j++)
                {
                    _output[j, i] = _activator.Apply(_zeta[j, i]);
                }
            }
        }

        /// <summary>
        /// Performs the backpropagation operation given the weighted errors of the upstream layer.
        /// </summary>
        /// <param name="upstreamWeightedErrors">The upstream weighted errors.</param>
        public void BackPropagate(Matrix<double> upstreamWeightedErrors)
        {
            Matrix<double> zetaDeriv = Matrix<double>.Build.Dense(_layerSize, _batchSize);
            for (int i = 0; i < _batchSize; i++)
            {
                for (int j = 0; j < _layerSize; j++)
                {
                    zetaDeriv[j, i] = _activator.ApplyDerivative(_zeta[j, i]);
                }
            }
            upstreamWeightedErrors.PointwiseMultiply(zetaDeriv, _errors);
            _weightedError = Weights.Multiply(_errors);
        }

        /// <summary>Creation
        /// Updates the parameters of the layer, typically its weights and bias.
        /// </summary>
        public void UpdateParameters()
        {
            _optimizer.UpdateParams(_errors, _input, ref _weights, ref _bias);
        }

        /// <summary>
        /// Gets the activation of the given layer.
        /// </summary>
        /// <value>
        /// The activation of the layer.
        /// </value>
        public Matrix<double> Activation
        {
            get
            {
                return _output;
            }
        }

        public Matrix<double> Input
        {
            get
            {
                return _input;
            }
            set
            {
                _input = value;
            }
        }

        public Matrix<double> Weights
        {
            get
            {
                return _weights;
            }
        }

        public Matrix<double> Bias
        {
            get
            {
                return _bias;
            }
        }

        /// <summary>
        /// Gets the weighted error, which is propagated to the preceeding layer.
        /// </summary>
        /// <value>
        /// The weighted error.
        /// </value>
        public Matrix<double> WeightedError {
            get 
            {
                return this._weightedError;
            }
        }

        public ActivatorType ActivatorType {
            get 
            {
                return _activator.Type;
            }
        }

        public bool Equals(ILayer other)
        {
            if (other.GetType() != this.GetType())
            {
                return false;
            }
            StandardLayer otherStandardLayer = other as StandardLayer;
            if (_layerSize != otherStandardLayer.LayerSize)
            {
                return false;
            }
            if (_inputSize != otherStandardLayer.InputSize)
            {
                return false;
            }
            if (_batchSize != otherStandardLayer.BatchSize)
            {
                return false;
            }
            if (!_weights.Equals(otherStandardLayer.Weights))
            {
                return false;
            }
            if (!_bias.Equals(otherStandardLayer.Bias))
            {
                return false;
            }
            if (!_activator.Type.Equals(otherStandardLayer.ActivatorType))
            {
                return false;
            }
            return true;
        }
    }
}