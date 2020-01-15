using NeuralNetwork.Common.GradientAdjustmentParameters;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;

namespace NeuralNetwork.Common.GradientAdjustmentsParameters
{
    /// <summary>
    /// Converter for generating objects of type <see cref="IGradientAdjustmentParameters"/> from a Json object.
    /// </summary>
    /// <seealso cref="Newtonsoft.Json.JsonConverter" />
    public class GradientAdjustmentParametersConverter : JsonConverter
    {
        public override bool CanWrite => false;
        public override bool CanRead => true;

        public override bool CanConvert(Type objectType)
        {
            return objectType == typeof(IGradientAdjustmentParameters);
        }

        public override void WriteJson(JsonWriter writer,
            object value, JsonSerializer serializer)
        {
            throw new InvalidOperationException("Use default serialization.");
        }

        //todo: switch to handle string enum
        public override object ReadJson(JsonReader reader,
            Type objectType, object existingValue,
            JsonSerializer serializer)
        {
            var jsonObject = JObject.Load(reader);
            IGradientAdjustmentParameters asset;
            switch (jsonObject.GetValue("type", StringComparison.OrdinalIgnoreCase).Value<int>())
            {
                case (int)GradientAdjustmentType.FixedLearningRate:
                    asset = new FixedLearningRateParameters();
                    break;

                case (int)GradientAdjustmentType.Adam:
                    asset = new AdamParameters();
                    break;

                case (int)GradientAdjustmentType.Momentum:
                    asset = new MomentumParameters();
                    break;

                default:
                    throw new InvalidOperationException("Unknown gradient accelerator parameter");
            }
            serializer.Populate(jsonObject.CreateReader(), asset);
            return asset;
        }
    }
}