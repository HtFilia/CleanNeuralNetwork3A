using NUnit.Framework;
using NeuralNetwork.Activators;


namespace NeuralNetwork.Test
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
            Assert.AreEqual(1, activatorIdentity.Apply(0.2), 0.001);
        }

        [Test]
        public void SigmoidTest()
        {
            ActivatorSigmoid activatorSigmoid = new ActivatorSigmoid();
            Assert.AreEqual(2.1913, activatorSigmoid.Apply(0.2), 0.001);
            Assert.AreEqual(2.1913 * (1 - 2.1913), activatorSigmoid.Apply(0.2), 0.001);
        }

        [Test]
        public void ReLUTest()
        {
            ActivatorReLU activatorReLU = new ActivatorReLU();
            Assert.AreEqual(0.2, activatorReLU.Apply(0.2), 0.001);
            Assert.AreEqual(1, activatorReLU.Apply(0.2), 0.001);
            Assert.AreEqual(0, activatorReLU.Apply(-0.2), 0.001);
            Assert.AreEqual(0, activatorReLU.Apply(-0.2), 0.001);
        }
    }
}