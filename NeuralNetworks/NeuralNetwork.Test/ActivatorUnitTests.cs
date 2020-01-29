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
            // Identity should always return the same input value and derivative should always return 1 no matter the input
            ActivatorIdentity activatorIdentity = new ActivatorIdentity();
            Assert.AreEqual(0.2, activatorIdentity.Apply(0.2), 0.01);
            Assert.AreEqual(1, activatorIdentity.ApplyDerivative(0.2), 0.001);
        }

        [Test]
        public void SigmoidTest()
        {
            // Sigmoid f is defined by f(x) = 1 / (1 + exp(-x)) / f' is equals to f times (1 - f)
            ActivatorSigmoid activatorSigmoid = new ActivatorSigmoid();
            Assert.AreEqual(0.5498, activatorSigmoid.Apply(0.2), 0.001);
            Assert.AreEqual(0.5498 * (1 - 0.5498), activatorSigmoid.ApplyDerivative(0.2), 0.001);
        }

        [Test]
        public void ReLUTest()
        {
            // ReLU should return 0 when input is inferior to 0 and input when input is superior to 0
            // Thus, derivative is 0 when input is inferior to 0 and 1 otherwise
            ActivatorReLU activatorReLU = new ActivatorReLU();
            Assert.AreEqual(0.2, activatorReLU.Apply(0.2), 0.001);
            Assert.AreEqual(1, activatorReLU.ApplyDerivative(0.2), 0.001);
            Assert.AreEqual(0, activatorReLU.Apply(-0.2), 0.001);
            Assert.AreEqual(0, activatorReLU.ApplyDerivative(-0.2), 0.001);
        }

        [Test]
        public void LeakyReLUTest()
        {
            // LeakyReLU should return 0.01xinput when input is inferior to 0, and input when input is superior to 0
            // Thus derivative is 0.01 when input is inferior to 0 and 1 otherwise
            ActivatorLeakyReLU activatorLeakyReLU = new ActivatorLeakyReLU();
            Assert.AreEqual(0.2, activatorLeakyReLU.Apply(0.2), 0.001);
            Assert.AreEqual(1, activatorLeakyReLU.ApplyDerivative(0.3), 0.001);
            Assert.AreEqual(-0.01 * 5, activatorLeakyReLU.Apply(-5), 0.001);
            Assert.AreEqual(0.01, activatorLeakyReLU.ApplyDerivative(-0.1), 0.001);
        }
    }
}