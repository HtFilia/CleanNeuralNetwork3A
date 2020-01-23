using NeuralNetwork.Common;
using NeuralNetwork.Common.Layers;
using NeuralNetwork.Common.Serialization;
using NeuralNetwork.Layers;
using Newtonsoft.Json;
using System;

namespace NeuralNetwork.Serialization
{
    public static class NetworkSerializer
    {        

        public static SerializedNetwork Serialize(INetwork network)
        {
            SerializedNetwork serializedNetwork = new SerializedNetwork();
            serializedNetwork.BatchSize = network.BatchSize;
            ISerializedLayer[] serializedLayers = new ISerializedLayer[network.Layers.Length];
            for(int i = 0; i < network.Layers.Length; i++)
            {
                StandardLayer layer = network.Layers.GetValue(i) as StandardLayer;
                ISerializedLayer serializedLayer = new SerializedStandardLayer(layer.Bias.Column(0).ToArray(),
                    layer.Weights.ToArray(),
                    layer.ActivatorType,
                    layer.FixedLearningRate);
                serializedLayers[i] = serializedLayer;
            }
            return serializedNetwork;
        }
    }
}