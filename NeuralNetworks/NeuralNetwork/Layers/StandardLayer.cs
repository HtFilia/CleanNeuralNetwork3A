﻿using MathNet.Numerics.LinearAlgebra;
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
        private Matrix<double> _output;
        private IActivator _activator;

        public StandardLayer(int layerSize, int inputSize, int batchSize, IActivator activator)
        {
            _layerSize = layerSize;
            _inputSize = inputSize;
            _batchSize = batchSize;

            _weights = Matrix<double>.Build.Random(_inputSize, _layerSize);
            _bias = Matrix<double>.Build.Random(_inputSize, _layerSize);
            _output = Matrix<double>.Build.Random(_batchSize, _inputSize);

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
            Matrix<double> zeta = Matrix<double>.Build.Random(_batchSize, _inputSize);
            for (int i = 0; i < _batchSize; i++)
            {
                zeta.Row(i) = ((weights.Row(i).Transpose()).Multiply(input.Row(i))).add(bias.Row(i));
                for (int j = 0; j < _inputSize; j++)
                {
                    _output[i, j] = _activator.Apply(zeta[i, j]);
                }
            }
        }

        /// <summary>
        /// Performs the backpropagation operation given the weighted errors of the upstream layer.
        /// </summary>
        /// <param name="upstreamWeightedErrors">The upstream weighted errors.</param>
        public void BackPropagate(Matrix<double> upstreamWeightedErrors)
        {
            throw new NotImplementedException();
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