using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common;
using NeuralNetwork.Common.Activators;
using NeuralNetwork.Common.Layers;
using System;

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
        public Network(int batchSize, int nbHiddenLayers, int[] nbNeuronsPerLayer, IActivator activator)
        {
            // Argument Exception
            if (nbNeuronsPerLayer.Length != nbHiddenLayers + 2)
            {
                throw new Exception("There should be a set number of neurons per layer " +
                    "for each layer.");
            }
            // Parameters
            this._batchSize = batchSize;
            this._output = Matrix<double>.Build.Random(batchSize, nbNeuronsPerLayer[nbNeuronsPerLayer.Length - 1]);
            this._layers = new ILayer[nbHiddenLayers + 3];
            // TODO: Add input and hidden layer with respective size and activator
            for (int layer = 0; layer <= nbHiddenLayers; layer++)
            {
                //this._layers[layer] = new StandardLayer(nbNeuronsPerLayer[layer], activator);
            }
            //this._layers[nbHiddenLayers + 2] = new StandardLayer(nbNeuronsPerLayer[nbHiddenLayers + 2], IdentityActivator);
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
            throw new NotImplementedException();
        }

        public void Propagate(Matrix<double> input)
        {
            throw new NotImplementedException();
        }
    }
}