using System;
using NeuralNetwork.Activators;
using NeuralNetwork.Layers;
using MathNet.Numerics.LinearAlgebra;
using NUnit.Framework;
using NeuralNetwork.Serialization;

namespace NeuralNetwork.Tests
{
    public class SerializeTest
    {
        [SetUp]
        public void Setup()
        {

        }

        [Test]
        public void SerializeTesting()
        {
            Network network = new Network(1, 1, 1, new int[] { 1 }, new ActivatorReLU());
            Network networkAfterSerialization = NetworkDeserializer.Deserialize(NetworkSerializer.Serialize(network));
            Assert.IsTrue(network.Equals(networkAfterSerialization));
        }
    }
}
