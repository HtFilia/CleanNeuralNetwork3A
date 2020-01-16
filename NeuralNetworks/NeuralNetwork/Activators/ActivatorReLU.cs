using System;
using System.Collections.Generic;
using System.Text;
using NeuralNetwork.Common.Activators;

namespace NeuralNetwork.Activators
{
    class ActivatorReLU : IActivator
    {
        protected internal ActivatorType activatorType;

        public ActivatorReLU(ActivatorType type)
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
