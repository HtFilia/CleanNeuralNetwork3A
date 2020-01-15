using NeuralNetwork.Common.Layers;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;

namespace NeuralNetwork.Common.Serialization
{
    /// <summary>
    /// Converter for generating objects of type <see cref="ISerializedLayer"/> from a Json object.
    /// </summary>
    /// <seealso cref="Newtonsoft.Json.JsonConverter" />
    public class SerializedLayerConverter : JsonConverter
    {
        public override bool CanWrite => false;
        public override bool CanRead => true;

        public override bool CanConvert(Type objectType)
        {
            return objectType == typeof(ISerializedLayer);
        }

        public override void WriteJson(JsonWriter writer,
            object value, JsonSerializer serializer)
        {
            throw new InvalidOperationException("Use default serialization.");
        }

        //todo: change to handle string enum
        public override object ReadJson(JsonReader reader,
            Type objectType, object existingValue,
            JsonSerializer serializer)
        {
            var jsonObject = JObject.Load(reader);
            ISerializedLayer asset;
            switch (jsonObject.GetValue("type", StringComparison.OrdinalIgnoreCase).Value<int>())
            {
                case (int)LayerType.Standard:
                    asset = new SerializedStandardLayer();
                    break;

                case (int)LayerType.InputStandardizing:
                    asset = new SerializedInputStandardizingLayer();
                    break;

                case (int)LayerType.Dropout:
                    asset = new SerializedDropoutLayer();
                    break;

                default:
                    throw new InvalidOperationException("Unknown serialized layer");
            }
            serializer.Populate(jsonObject.CreateReader(), asset);
            return asset;
        }
    }
}