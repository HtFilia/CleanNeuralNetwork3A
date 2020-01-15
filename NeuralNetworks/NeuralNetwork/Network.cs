using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common;
using NeuralNetwork.Common.Layers;
using System;

namespace NeuralNetwork
{
    public sealed class Network : IEquatable<Network>, INetwork
    {
        public int BatchSize { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }

        public Matrix<double> Output => throw new NotImplementedException();

        public ILayer[] Layers => throw new NotImplementedException();

        public Mode Mode { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }


        public bool Equals(Network other)
        {
            throw new NotImplementedException();
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