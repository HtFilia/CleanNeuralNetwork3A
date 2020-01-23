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
            Network network = new Network(1, 1, 0, new int[] { 1, 1 }, new ActivatorReLU());
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
