using System;
using NeuralNetwork.Activators;
using NeuralNetwork.Layers;
using MathNet.Numerics.LinearAlgebra;
using NUnit.Framework;
using Newtonsoft.Json;
using NeuralNetwork.Common.Layers;

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

        public void updateParamsTest()
        {
            Network network = new Network(1, 1, 0, new int[] { 1, 1 }, new ActivatorReLU());
            ILayer[] InitialLayers = network.Layers;
            network.Propagate(Matrix<double>.Build.Random(1, 1));
            network.Learn(network.Output);
            Assert.IsFalse(InitialLayers.Equals(network.Layers));
        }
    }
}
