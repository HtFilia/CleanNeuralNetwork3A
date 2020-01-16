using System;
using System.Collections.Generic;
using System.Text;
using NeuralNetwork.Common.Activators;

namespace NeuralNetwork.Activators
{
    public sealed class ActivatorIdentity : IActivator
    {
        public ActivatorType activatorType;

        public ActivatorIdentity()
        {
            activatorType = Common.Activators.ActivatorType.Identity;
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
