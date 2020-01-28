using System;
using System.Collections.Generic;
using System.Text;
using NeuralNetwork.Activators;
using NeuralNetwork.Layers;
using NeuralNetwork.Serialization;
using NeuralNetwork.Common.Activators;
using NeuralNetwork.Common.Layers;
using NeuralNetwork.Common.Serialization;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using NeuralNetwork.Common.GradientAdjustmentsParameters;

namespace NeuralNetwork.Utils
{
    public static class CorrectTypes
    {
        public static IActivator GetCorrectActivator(ActivatorType activatorType)
        {
            switch (activatorType)
            {
                case ActivatorType.Identity:
                    {
                        return new ActivatorIdentity();
                    }
                case ActivatorType.Sigmoid:
                    {
                        return new ActivatorSigmoid();
                    }
                case ActivatorType.ReLU:
                    {
                        return new ActivatorReLU();
                    }
                case ActivatorType.LeakyReLU:
                    {
                        return new ActivatorLeakyReLU();
                    }
                case ActivatorType.Tanh:
                    {
                        throw new NotImplementedException();
                    }
                default: throw new ArgumentException("Wrong Activator Type.");
            }
        }

        public static ILayer GetCorrectDeserializedLayer(ISerializedLayer serializedLayer, int batchSize)
        {
            switch (serializedLayer.Type)
            {
                case LayerType.Standard:
                    {
                        return GetCorrectStandardDeserializedLayer(serializedLayer, batchSize);
                    }
                case LayerType.InputStandardizing:
                    {
                        throw new NotImplementedException();
                    }
                case LayerType.Dropout:
                    {
                        throw new NotImplementedException();
                    }
                case LayerType.L2Penalty:
                    {
                        throw new NotImplementedException();
                    }
                case LayerType.WeightDecay:
                    {
                        throw new NotImplementedException();
                    }
                default: throw new ArgumentException("Wrong Layer Type.");
            }
        }

        private static StandardLayer GetCorrectStandardDeserializedLayer(ISerializedLayer serializedLayer, int batchSize)
        {
            // Cast to correct type
            SerializedStandardLayer serializedStandardLayer = serializedLayer as SerializedStandardLayer;
            // Recover correct gradient adjustments parameters
            IGradientAdjustmentParameters gradientAdjustmentParameters = GetCorrectGradientAdjustmentParameters(serializedStandardLayer.GradientAdjustmentParameters);
            // Recover correct dimensions
            int inputSize = serializedStandardLayer.Weights.GetLength(0);
            int layerSize = serializedStandardLayer.Weights.GetLength(1);
            IActivator activator = GetCorrectActivator(serializedStandardLayer.ActivatorType);
            StandardLayer deserializedStandardLayer = new StandardLayer(layerSize, inputSize, batchSize, gradientAdjustmentParameters, activator);
            // Fill correct values for weights, biases and parameters
            for (int j = 0; j < layerSize; j++)
            {
                deserializedStandardLayer.Bias[j, 0] = serializedStandardLayer.Bias[j];
                for (int i = 0; i < inputSize; i++)
                {
                    deserializedStandardLayer.Weights[i, j] = serializedStandardLayer.Weights[i, j];
                }
            }
            // All done
            return deserializedStandardLayer;
        }

        private static IGradientAdjustmentParameters GetCorrectGradientAdjustmentParameters(IGradientAdjustmentParameters gradientAdjustmentParameters)
        {
            switch (gradientAdjustmentParameters.Type)
            {
                case GradientAdjustmentType.FixedLearningRate:
                    {
                        return gradientAdjustmentParameters as FixedLearningRateParameters;

                    }
                case GradientAdjustmentType.Adam:
                    {
                        return gradientAdjustmentParameters as AdamParameters;
                    }
                case GradientAdjustmentType.Momentum:
                    {
                        return gradientAdjustmentParameters as MomentumParameters;
                    }
                default: throw new ArgumentException("Wrong Gradient Adjustment Parameters Type");
            }
        }
    }

}
