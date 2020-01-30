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
            StandardLayer xorHiddenLayer = new StandardLayer(2, 2, 4,
                new AdamParameters(0.01, 0.9, 0.999, 1E-8), new ActivatorLeakyReLU());
            StandardLayer xorOutputLayer = new StandardLayer(1, 2, 4,
                new AdamParameters(0.01, 0.9, 0.999, 1E-8), new ActivatorIdentity());
            Network xorNetwork = new Network(4, new ILayer[] { xorHiddenLayer, xorOutputLayer });

            StandardLayer pricingHiddenLayer = new StandardLayer(5, 7, 1, 
                new AdamParameters(0.01, 0.9, 0.999, 1E-8), new ActivatorLeakyReLU());
            StandardLayer pricingOutputLayer = new StandardLayer(1, 5, 1, 
                new AdamParameters(0.01, 0.9, 0.999, 1E-8), new ActivatorIdentity());
            Network princingNetwork = new Network(1, new ILayer[] { pricingHiddenLayer, pricingOutputLayer });

            SerializedNetwork serializedXorNetwork = NetworkSerializer.Serialize(xorNetwork);
            SerializedNetwork serializedPricingNetwork = NetworkSerializer.Serialize(princingNetwork);

            try
            {
                using FileStream pricingNetworkJson = File.Create("./pricingNetwork.json");
                byte[] jsonPricingText = new UTF8Encoding(true).GetBytes(JsonConvert.SerializeObject(serializedPricingNetwork));
                pricingNetworkJson.Write(jsonPricingText, 0, jsonPricingText.Length);
                pricingNetworkJson.Close();
                using FileStream xorNetworkJson = File.Create("./xorNetwork.json");
                byte[] jsonXorText = new UTF8Encoding(true).GetBytes(JsonConvert.SerializeObject(serializedXorNetwork));
                xorNetworkJson.Write(jsonXorText, 0, jsonXorText.Length);
                xorNetworkJson.Close();
            } catch (Exception e)
            {
                throw e;
            }
        }
    }
}
