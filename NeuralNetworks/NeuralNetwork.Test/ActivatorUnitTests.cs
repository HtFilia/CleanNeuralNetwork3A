using NUnit.Framework;
using NeuralNetwork.Activators;


namespace NeuralNetwork.Tests
{
    public class ActivatorUnitTests
    {
        [SetUp]
        public void Setup()
        {   
        }

        [Test]
        public void IdentityTest()
        {
            ActivatorIdentity activatorIdentity = new ActivatorIdentity();
            Assert.AreEqual(0.2, activatorIdentity.Apply(0.2), 0.01);
            Assert.AreEqual(1, activatorIdentity.ApplyDerivative(0.2), 0.001);
        }

        [Test]
        public void SigmoidTest()
        {
            ActivatorSigmoid activatorSigmoid = new ActivatorSigmoid();
            Assert.AreEqual(0.5498, activatorSigmoid.Apply(0.2), 0.001);
            Assert.AreEqual(0.5498 * (1 - 0.5498), activatorSigmoid.ApplyDerivative(0.2), 0.001);
        }

        [Test]
        public void ReLUTest()
        {
            ActivatorReLU activatorReLU = new ActivatorReLU();
            Assert.AreEqual(0.2, activatorReLU.Apply(0.2), 0.001);
            Assert.AreEqual(1, activatorReLU.ApplyDerivative(0.2), 0.001);
            Assert.AreEqual(0, activatorReLU.Apply(-0.2), 0.001);
            Assert.AreEqual(0, activatorReLU.ApplyDerivative(-0.2), 0.001);
        }
    }
}