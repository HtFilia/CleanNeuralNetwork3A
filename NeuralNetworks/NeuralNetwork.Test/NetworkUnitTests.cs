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
        Network networkToTest;

        [SetUp]
        public void Setup()
        {   
            // Create a network with random weights and biases, using fixed learning rate optimizer
            // 2 layers : input layer (2 neurons, 2 inputs, activator leaky reLU)
            //            output layer (1 neuron, 2 inputs, activator identity)
            int batchSize = 1;
            StandardLayer inputLayer = new StandardLayer(2, 2, batchSize, new FixedLearningRateParameters(0.1), new ActivatorLeakyReLU()); ;
            StandardLayer outputLayer = new StandardLayer(1, 2, batchSize, new FixedLearningRateParameters(0.1), new ActivatorIdentity()); ;
            networkToTest = new Network(batchSize, new ILayer[] { inputLayer, outputLayer });
        }

        [Test]
        public void PropagationTest()
        {
            // We create a random input that we feed to the network and check if network's output is not 0
            networkToTest.Propagate(Matrix<double>.Build.Random(2, 1));
            Assert.NotZero(networkToTest.Output[0, 0]);
        }

        [Test]
        public void BackPropagationTest()
        {
            //TODO: predict error values. Can it be done with random weights/biases/input?
        }

        public void updateParamsTest()
        {
            // We create a random input that we feed to the network and check if weights and biases have been updated
            ILayer[] InitialLayers = networkToTest.Layers;
            networkToTest.Propagate(Matrix<double>.Build.Random(2, 1));
            networkToTest.Learn(networkToTest.Output);
            Assert.IsFalse(InitialLayers.Equals(networkToTest.Layers));
        }
    }
}
