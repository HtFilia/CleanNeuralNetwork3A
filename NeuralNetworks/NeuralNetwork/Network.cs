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
        public int _batchSize;
        public Matrix<double> _output;
        public ILayer[] _layers;
        public Mode _mode;


        // Getters / Setters
        public int BatchSize { get => _batchSize; set => _batchSize = value; }

        public Matrix<double> Output => _output;

        public ILayer[] Layers => _layers;

        public Mode Mode { get => _mode; set => _mode = value; }

        // Constructor
        public Network(int batchSize, int inputSize, int nbHiddenLayers, int[] nbNeuronsPerLayer, IActivator activator)
        {
            // Argument Exception
            if (nbNeuronsPerLayer.Length != nbHiddenLayers)
            {
                throw new Exception("There should be a set number of neurons per layer " +
                    "for each layer.");
            }
            // Parameters
            this._batchSize = batchSize;
            this._output = Matrix<double>.Build.Dense(batchSize, nbNeuronsPerLayer[nbNeuronsPerLayer.Length - 1]);
            this._layers = new ILayer[nbHiddenLayers + 3];
            // First hidden layer is connected to user's input
            this._layers[0] = new StandardLayer(nbNeuronsPerLayer[0], inputSize, batchSize, activator);
            // Next hidden layers have an input size of previous layer's size
            for (int layer = 1; layer < nbHiddenLayers; layer++)
            {
                this._layers[layer] = new StandardLayer(nbNeuronsPerLayer[layer], nbNeuronsPerLayer[layer - 1], batchSize, activator);
            }
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
            // First back-propagation through argument outputLayerError
            // TODO
            // We back-propagate to the previous layers.
            // TODO
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
    }
}