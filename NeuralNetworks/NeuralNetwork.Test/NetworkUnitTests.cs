using System;
using NeuralNetwork.Activators;
using NeuralNetwork.Layers;
using MathNet.Numerics.LinearAlgebra;
using NUnit.Framework;
using Newtonsoft.Json;
using NeuralNetwork.Common.Layers;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using NeuralNetwork.Common.GradientAdjustmentsParameters;

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
            StandardLayer outputLayer = new StandardLayer(1, 1, 1, new FixedLearningRateParameters(0.1), new ActivatorIdentity());
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
            int batchSize = 1;
            StandardLayer inputLayer = new StandardLayer(2, 2, batchSize, new FixedLearningRateParameters(0.1), new ActivatorLeakyReLU()); ;
            StandardLayer outputLayer = new StandardLayer(1, 2, batchSize, new FixedLearningRateParameters(0.1), new ActivatorIdentity()); ;
            Network network = new Network(batchSize, new Common.Layers.ILayer[] { inputLayer, outputLayer });
            ILayer[] InitialLayers = network.Layers;
            network.Propagate(Matrix<double>.Build.Random(1, 1));
            network.Learn(network.Output);
            Assert.IsFalse(InitialLayers.Equals(network.Layers));
        }
    }
}
