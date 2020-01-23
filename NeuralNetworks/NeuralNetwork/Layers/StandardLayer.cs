using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common;
using NeuralNetwork.Common.Activators;
using NeuralNetwork.Common.Layers;
using System;

namespace NeuralNetwork.Layers
{
    public sealed class StandardLayer : IEquatable<ILayer>, ILayer
    {
        private int _layerSize;
        private int _inputSize;
        private int _batchSize;
        private Matrix<double> _weights;
        private Matrix<double> _bias;
        private Matrix<double> _zeta;
        private Matrix<double> _output;
        Matrix<double> _weightedError;
        private IActivator _activator;

        public StandardLayer(int layerSize, int inputSize, int batchSize, IActivator activator)
        {
            _layerSize = layerSize;
            _inputSize = inputSize;
            _batchSize = batchSize;

            _weights = Matrix<double>.Build.Random(_layerSize, _inputSize);
            _bias = Matrix<double>.Build.Random(_layerSize, 1);
            _output = Matrix<double>.Build.Dense(_batchSize, _layerSize);
            _zeta = Matrix<double>.Build.Dense(_batchSize, _layerSize);
            _weightedError = Matrix<double>.Build.Dense(_batchSize, _layerSize);

            _activator = activator;
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
            _output = Matrix<double>.Build.Dense(_batchSize, _layerSize);

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
            _inputSize = input.ColumnCount;
            for (int i = 0; i < _batchSize; i++)
            {
                _zeta.SetRow(i, (_weights.Transpose().Multiply(input.Row(i))).Add(_bias.Column(0)));
                for (int j = 0; j < _layerSize; j++)
                {
                    _output[i, j] = _activator.Apply(_zeta[i, j]);
                }
            }
        }

        /// <summary>
        /// Performs the backpropagation operation given the weighted errors of the upstream layer.
        /// </summary>
        /// <param name="upstreamWeightedErrors">The upstream weighted errors.</param>
        public void BackPropagate(Matrix<double> upstreamWeightedErrors)
        {
            Matrix<double> zetaDeriv = Matrix<double>.Build.Dense(_batchSize, _layerSize);
            for (int i = 0; i < _batchSize; i++)
            {
                for (int j = 0; j < _layerSize; j++)
                {
                    zetaDeriv[i, j] = _activator.ApplyDerivative(_zeta[i, j]);
                }
            }
            upstreamWeightedErrors.PointwiseMultiply(zetaDeriv, _weightedError);
        }

        /// <summary>Creation
        /// Updates the parameters of the layer, typically its weights and bias.
        /// </summary>
        public void UpdateParameters()
        {
            throw new NotImplementedException();
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
        public Matrix<double> WeightedError { get; }

        public ActivatorType ActivatorType {
            get {
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
            return true;
        }
    }
}