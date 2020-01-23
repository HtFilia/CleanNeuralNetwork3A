using System;
using NeuralNetwork.Activators;
using NeuralNetwork.Layers;
using MathNet.Numerics.LinearAlgebra;
using NUnit.Framework;
using NeuralNetwork.Serialization;

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
            Network network = new Network(1, 1, 0, new int[] { 1, 1 }, new ActivatorReLU());
            Network deserializedNetwork = NetworkDeserializer.Deserialize(NetworkSerializer.Serialize(network));
            Assert.IsTrue(network.Equals(deserializedNetwork));
        }
    }
}
