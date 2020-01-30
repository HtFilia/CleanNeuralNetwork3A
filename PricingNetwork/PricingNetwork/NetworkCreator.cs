using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork;
using NeuralNetwork.Activators;
using NeuralNetwork.Common.GradientAdjustmentsParameters;
using NeuralNetwork.Common.Layers;
using NeuralNetwork.Common.Serialization;
using NeuralNetwork.Layers;
using NeuralNetwork.Serialization;
using Newtonsoft.Json;

namespace PricingNetwork
{
    class NetworkCreator
    {
        static void Main()
        {
            int batchSize = 20;
            int inputLayerSize = 5;
            //int otherLayerSize = 2;
            StandardLayer pricingHiddenLayer = new StandardLayer(inputLayerSize, 7, batchSize, 
                new AdamParameters(0.01, 0.9, 0.999, 1E-8), new ActivatorLeakyReLU());
            //StandardLayer pricingSecondHiddentLayer = new StandardLayer(otherLayerSize, inputLayerSize, batchSize,
            //    new AdamParameters(0.01, 0.9, 0.999, 1E-8), new ActivatorLeakyReLU());
            StandardLayer pricingOutputLayer = new StandardLayer(1, inputLayerSize, batchSize, 
                new AdamParameters(0.01, 0.9, 0.999, 1E-8), new ActivatorIdentity());
            Network princingNetwork = new Network(batchSize, new ILayer[] { pricingHiddenLayer, pricingOutputLayer });

            SerializedNetwork serializedPricingNetwork = NetworkSerializer.Serialize(princingNetwork);

            try
            {
                using FileStream pricingNetworkJson = File.Create("./pricingNetwork.json");
                byte[] jsonPricingText = new UTF8Encoding(true).GetBytes(JsonConvert.SerializeObject(serializedPricingNetwork));
                pricingNetworkJson.Write(jsonPricingText, 0, jsonPricingText.Length);
                pricingNetworkJson.Close();
            } catch (Exception e)
            {
                throw e;
            }
        }
    }
}
