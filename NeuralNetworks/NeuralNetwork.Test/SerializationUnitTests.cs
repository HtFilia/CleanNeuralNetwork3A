using System;
using NeuralNetwork.Activators;
using NeuralNetwork.Layers;
using MathNet.Numerics.LinearAlgebra;
using NUnit.Framework;
using NeuralNetwork.Serialization;
using NeuralNetwork.Common.Serialization;

namespace NeuralNetwork.Tests
{
    public class SerializationUnitTests
    {
        [SetUp]
        public void Setup()
        {

        }

        [Test]
        public void SerializeAndDeserializeTest()
        {
            int batchSize = 1;
            StandardLayer inputLayer = new StandardLayer(2, 2, batchSize, new ActivatorLeakyReLU());
            StandardLayer outputLayer = new StandardLayer(1, 2, batchSize, new ActivatorIdentity());
            Network network = new Network(batchSize, new Common.Layers.ILayer[] { inputLayer, outputLayer });
            SerializedNetwork serializedNetwork = NetworkSerializer.Serialize(network);
            Network deserializedNetwork = NetworkDeserializer.Deserialize(serializedNetwork);
            Assert.IsTrue(network.Equals(deserializedNetwork));
        }
    }
}
