using System;
using System.Collections.Generic;
using System.Text;
using NeuralNetwork.Common.Activators;

namespace NeuralNetwork.Activators
{
    class ActivatorIdentity : IActivator
    {
        protected internal ActivatorType activatorType;

        public ActivatorIdentity(ActivatorType type)
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
                Func<double, double> identity = x => x;
                return identity;
            }
        }

        public Func<double, double> ApplyDerivative
        {
            get
            {
                Func<double, double> identity = x => 1;
                return identity;
            }
        }
    }
}
