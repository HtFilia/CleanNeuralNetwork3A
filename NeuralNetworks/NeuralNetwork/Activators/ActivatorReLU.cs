using System;
using System.Collections.Generic;
using System.Text;
using NeuralNetwork.Common.Activators;

namespace NeuralNetwork.Activators
{
    public sealed class ActivatorReLU : IActivator
    {
        public ActivatorType activatorType;

        public ActivatorReLU()
        {
            activatorType = Common.Activators.ActivatorType.ReLU;
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
                Func<double, double> reLU = x => (x > 0) ? x : 0;
                return reLU;
            }
        }

        public Func<double, double> ApplyDerivative
        {
            get
            {
                Func<double, double> reLU = x => (x > 0) ? 1 : 0;
                return reLU;
            }
        }
    }
}
