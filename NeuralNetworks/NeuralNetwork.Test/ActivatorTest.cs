using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork.Activators;
using System;

namespace NeuralNetwork.Test
{
    [TestClass]
    public class ActivatorTest
    {
        [TestMethod]
        public void IdentityTest()
        {
            ActivatorIdentity identityTestActivator = new ActivatorIdentity();
            Assert.AreEqual(0.2, identityTestActivator.Apply(0.2), 0.001, "Fail");
            Assert.AreEqual(1, identityTestActivator.Apply(0.2), 0.001, "Fail");
        }

        [TestMethod]
        public void SigmoidTest()
        {
            ActivatorSigmoid sigmoidTestActivator = new ActivatorSigmoid();
            Assert.AreEqual(2.1913, sigmoidTestActivator.Apply(0.2), 0.001, "Fail");
            Assert.AreEqual(2.1913 * (1 - 2.1913), sigmoidTestActivator.Apply(0.2), 0.001, "Fail");
        }

        [TestMethod]
        public void ReLUTest()
        {
            ActivatorReLU reLUTestActivator = new ActivatorReLU();
            Assert.AreEqual(0.2, reLUTestActivator.Apply(0.2), 0.001, "Fail");
            Assert.AreEqual(1, reLUTestActivator.Apply(0.2), 0.001, "Fail");
            Assert.AreEqual(0, reLUTestActivator.Apply(-0.2), 0.001, "Fail");
            Assert.AreEqual(0, reLUTestActivator.Apply(-0.2), 0.001, "Fail");
        }
    }
}
