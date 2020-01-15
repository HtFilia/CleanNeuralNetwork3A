using DataProviders;
using NeuralNetwork.Common;
using NeuralNetwork.Common.Serialization;
using NeuralNetwork.Serialization;
using Newtonsoft.Json;
using System.IO;

namespace RegressionTester
{
    internal static class Program
    {
        private static void Main(string[] args)
        {
            var serializedNetworkFile = args[0];
            var outputFile = args[1];
            var network = LoadNetwork(serializedNetworkFile);
            var tester = new Evaluator(network, new RelativeDifference());
            var testData = LoadData();
            var result = tester.Test(testData);
            var summary = new StatisticsSummary(result);
            WriteToFile(summary, outputFile);
        }

        private static void WriteToFile(StatisticsSummary summary, string outputFile)
        {
            JsonSerializer serializer = new JsonSerializer();
            using (StreamWriter sw = new StreamWriter(outputFile))
            using (JsonWriter writer = new JsonTextWriter(sw))
            {
                serializer.Serialize(writer, summary);
            }
        }

        private static INetwork LoadNetwork(string serializedNetworkFile)
        {
            var serializedNetwork = JsonConvert.DeserializeObject<SerializedNetwork>(File.ReadAllText(serializedNetworkFile));
            return NetworkDeserializer.Deserialize(serializedNetwork);
        }

        private static MathData LoadData()
        {
            var bsProvider = new PricingDataProvider();
            var splitData = bsProvider.GetData();
            return splitData.TestData;
        }
    }
}