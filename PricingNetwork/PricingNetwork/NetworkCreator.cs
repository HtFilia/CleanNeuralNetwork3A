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
                StandardLayer hiddenLayer = new StandardLayer(5,
                    7, 20, new AdamParameters(0.007, 0.9, 0.999, 1E-8), new ActivatorLeakyReLU());
                StandardLayer outputLayer = new StandardLayer(1, 5, 20,
                    new AdamParameters(0.007, 0.9, 0.999, 1E-8), new ActivatorIdentity());

                Network princingNetwork = new Network(20, new ILayer[] { hiddenLayer, outputLayer });

            SerializedNetwork serializedNetwork = NetworkSerializer.Serialize(princingNetwork);

            try
            {
                using(FileStream networkJson = File.Create("./pricingNetwork.json"))
                {
                    byte[] jsonText = new UTF8Encoding(true).GetBytes(JsonConvert.SerializeObject(serializedNetwork));
                    networkJson.Write(jsonText, 0, jsonText.Length);
                    networkJson.Close();
                }
            } catch (Exception e)
            {
                throw e;
            }
        }
    }
}
