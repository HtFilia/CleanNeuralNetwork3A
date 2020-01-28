using System;
using NeuralNetwork.Activators;
using NeuralNetwork.Layers;
using MathNet.Numerics.LinearAlgebra;
using NUnit.Framework;
using NeuralNetwork.Common.GradientAdjustmentsParameters;
using NeuralNetwork.Common.GradientAdjustmentParameters;

namespace NeuralNetwork.Tests
{
    public class LayerUnitTests
    {
        [SetUp]
        public void Setup()
        {

        }

        [Test]
        public void PropagationTest()
        {
            // We create a basic layer containing 1 neuron and 1 random input
            StandardLayer testLayer = new StandardLayer(1, 1, 1, new FixedLearningRateParameters(0.1), new ActivatorIdentity());
            Matrix<double> testInput = Matrix<double>.Build.Random(1, 1);

            // We propagate the input to this new layer
            testLayer.Propagate(testInput);

            // If all is well the new activation is not zero anymore
            Assert.AreNotEqual(0d, testLayer.Activation[0, 0]);
        }
    }
}
