using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common;
using NeuralNetwork.Common.Activators;
using NeuralNetwork.Common.Layers;
using System;
using NeuralNetwork.Layers;

namespace NeuralNetwork
{
    public sealed class Network : IEquatable<Network>, INetwork
    {
        // Fields
        private int _batchSize;
        private Matrix<double> _output;
        private ILayer[] _layers;
        private Mode _mode;


        // Getters / Setters
        public int BatchSize { get => _batchSize; set => _batchSize = value; }

        public Matrix<double> Output => _output;

        public ILayer[] Layers => _layers;

        public Mode Mode { get => _mode; set => _mode = value; }

        // Constructor
        public Network(int batchSize, ILayer[] layers)
        {
            // Exception handling
            if (layers == null || layers.Length == 0)
            {
                throw new ArgumentException("You need layers to create a neural network.");
            }
            if (batchSize == 0)
            {
                throw new ArgumentException("You need a positive integer for batchSize.");
            }
            // Parameters
            this._batchSize = batchSize;
            this._output = layers.Last().Activation;
            this._layers = layers;

            // We have to train a network first after creating it
            this._mode = Mode.Training;
        }

        public bool Equals(Network other)
        {
            // First compare the dimension of the two networks
            if (this._layers.Length != other._layers.Length)
            {
                return false;
            }
            // Then compare each layer to its counterpart
            for (int layer = 0; layer < this._layers.Length; layer++)
            {
                if (! this._layers[layer].Equals(other._layers[layer]))
                {
                    return false;
                }
            }
            // When dimensions and values are equals then it's over, both networks are equals
            return true;
        }

        public void Learn(Matrix<double> outputLayerError)
        {
            var weightedError = outputLayerError;
            for (int i = Layers.Length - 1; i > -1; i--)
            {
                Layers[i].BackPropagate(weightedError);
                Layers[i].UpdateParameters();
                weightedError = Layers[i].WeightedError;
            }
        }


        public void Propagate(Matrix<double> input)
        {
            // First propagation through argument input
            _layers[0].Propagate(input);
            // We propagate to the next layers. The previous Activation becomes the next input
            for (int layer = 1; layer < _layers.Length; layer++)
            {
                Matrix<double> newInput = _layers[layer - 1].Activation;
                _layers[layer].Propagate(newInput);
            }
            // Final layer's activation is network's output
            _output = _layers[Layers.Length - 1].Activation;
        }

        public override string ToString()
        {
            string res = "";
            int count = 0;
            foreach (ILayer layer in _layers)
            {
                StandardLayer standardLayer = layer as StandardLayer;
                res += String.Format("[Layer #{0}] Weights : ", count);
                res += standardLayer.Weights.ToMatrixString();
            }
            return res;
        }
    }
}