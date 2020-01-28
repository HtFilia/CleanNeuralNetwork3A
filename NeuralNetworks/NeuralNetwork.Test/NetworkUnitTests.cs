using System;
using NeuralNetwork.Activators;
using NeuralNetwork.Layers;
using MathNet.Numerics.LinearAlgebra;
using NUnit.Framework;

namespace NeuralNetwork.Tests
{
    public class NetworkUnitTests
    {
        [SetUp]
        public void Setup()
        {

        }

        [Test]
        public void PropagationTest()
        {
            StandardLayer outputLayer = new StandardLayer(1, 1, 1, new ActivatorIdentity());
            Network network = new Network(1, new Common.Layers.ILayer[] { outputLayer });
            network.Propagate(Matrix<double>.Build.Random(1, 1));
            Assert.NotZero(network.Output[0, 0]);
        }

        [Test]
        public void BackPropagationTest()
        {
            //TODO
        }
    }
}
