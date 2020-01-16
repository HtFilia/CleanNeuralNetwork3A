using System;
using NeuralNetwork.Layers;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetwork.Tests
{
    public class TestPropagation
    {
        public void testPropagation()
        {
            StandardLayer testLayer = new StandardLayer(3, 5, 2, new ActivatorIdentity());
            Matrix<double> testInput = Matrix<double>.Build.Random(2, 5);

            testLayer.propagate(testInput);

            Console.Write(testLayer.Activation);
        }
    }
}
