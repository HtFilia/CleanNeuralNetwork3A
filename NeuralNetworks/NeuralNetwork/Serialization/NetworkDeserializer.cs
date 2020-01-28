using NeuralNetwork.Common.Activators;
using NeuralNetwork.Common.Serialization;
using NeuralNetwork.Common.Layers;
using NeuralNetwork.Activators;
using NeuralNetwork.Layers;
using NeuralNetwork.Utils;
using System;

namespace NeuralNetwork.Serialization
{
    public static class NetworkDeserializer
    {
        public static Network Deserialize(SerializedNetwork serializedNetwork)
        {
            // We recover the batch size from the serialized network
            int batchSize = serializedNetwork.BatchSize;
            // We recover each layer from the serialized network
            ILayer[] layers = new ILayer[serializedNetwork.SerializedLayers.Length];
            for (int layer = 0; layer < serializedNetwork.SerializedLayers.Length; layer++)
            {
                ISerializedLayer serializedLayer = serializedNetwork.SerializedLayers[layer];
                layers[layer] = CorrectTypes.GetCorrectDeserializedLayer(serializedLayer, batchSize);
            }
            // We return the correct deserialized network
            return new Network(serializedNetwork.BatchSize, layers);
        }
    }
}