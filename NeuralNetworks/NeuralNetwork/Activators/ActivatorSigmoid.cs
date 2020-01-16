using System;
using System.Collections.Generic;
using System.Text;
using NeuralNetwork.Common.Activators;

namespace NeuralNetwork.Activators
{
    class ActivatorSigmoid : IActivator
    {
        protected internal ActivatorType activatorType;

        public ActivatorSigmoid(ActivatorType type)
        {
            activatorType = type;
        }

        public ActivatorType Type
        {
            get
            {
                return activatorType;
            }

        }

        public Func<double, double> Apply
        {
            get
            {
                Func<double, double> sigmoid = x => 1 / (1 + Math.Exp(-x));
                return sigmoid;
            }
        }

        public Func<double, double> ApplyDerivative
        {
            get
            {
                Func<double, double> sigmoid = x => (1 / (1 + Math.Exp(-x))) * (1 - 1 / (1 + Math.Exp(-x)));
                return sigmoid;
            }
        }
    }
}
