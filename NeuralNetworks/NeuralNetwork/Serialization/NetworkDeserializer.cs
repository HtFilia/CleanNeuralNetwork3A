using NeuralNetwork.Common.Activators;
using NeuralNetwork.Common.Serialization;
using NeuralNetwork.Layers;
using System;

namespace NeuralNetwork.Serialization
{
    public static class NetworkDeserializer
    {
        public static Network Deserialize(SerializedNetwork serializedNetwork)
        {
            // Get network's correct dimensions

            int[] nbNeuronsPerLayer = new int[serializedNetwork.SerializedLayers.Length];
            // Number of neurons in each layer
            for (int layer = 0; layer < nbNeuronsPerLayer.Length; layer++)
            {
                SerializedStandardLayer serializedStandardLayer = serializedNetwork.SerializedLayers[layer] as SerializedStandardLayer;
                nbNeuronsPerLayer[layer] = serializedStandardLayer.Weights.GetLength(0);
            }
            // Input layer is special among other layers
            SerializedStandardLayer inputLayer = serializedNetwork.SerializedLayers[0] as SerializedStandardLayer;
            // Input size is previous layer's size
            int inputSize = inputLayer.Weights.GetLength(1);
            // Hidden layers are every layer except input layer
            int nbHiddenLayers = serializedNetwork.SerializedLayers.Length - 2;
            // This activator will be applied first because of constructor specs. Will be changed if necessary when creating layers
            IActivator activator = GetCorrectActivator(inputLayer.ActivatorType);
            // Create Network with correct dimensions
            Network deserializedNetwork = new Network(serializedNetwork.BatchSize, inputSize, nbHiddenLayers, nbNeuronsPerLayer, activator);
            // Fill layers values with corresponding serialized layer's values
            for (int layer = 0; layer < deserializedNetwork.Layers.Length; layer++)
            {
                deserializedNetwork.Layers[layer] = DeserializeStandardLayer(serializedNetwork.SerializedLayers[layer] as SerializedStandardLayer, serializedNetwork.BatchSize);
            }
            // Network is properly deserialized
            return deserializedNetwork;
        }

        private static StandardLayer DeserializeStandardLayer(SerializedStandardLayer serializedStandardLayer, int batchSize)
        {
            // Recover correct dimensions
            int layerSize = serializedStandardLayer.Weights.GetLength(0);
            int inputSize = serializedStandardLayer.Weights.GetLength(1);
            IActivator activator = GetCorrectActivator(serializedStandardLayer.ActivatorType);
            StandardLayer deserializedStandardLayer = new StandardLayer(layerSize, inputSize, batchSize, activator);
            // Fill correct values for weights and biases 
            for (int i = 0; i < layerSize; i++)
            {
                deserializedStandardLayer.Bias[i, 0] = serializedStandardLayer.Bias[i];
                for (int j = 0; j < inputSize; j++)
                {
                    deserializedStandardLayer.Weights[i, j] = serializedStandardLayer.Weights[i, j];
                }
            }
            // All done
            return deserializedStandardLayer;
        }

        private static IActivator GetCorrectActivator(ActivatorType activatorType)
        {
            switch (activatorType)
            {
                case ActivatorType.Identity:
                    {
                        return new Activators.ActivatorIdentity();
                    }
                case ActivatorType.Sigmoid:
                    {
                        return new Activators.ActivatorSigmoid();
                    }
                case ActivatorType.ReLU:
                    {
                        return new Activators.ActivatorReLU();
                    }
                default: return null;
            }
        }
    }
}