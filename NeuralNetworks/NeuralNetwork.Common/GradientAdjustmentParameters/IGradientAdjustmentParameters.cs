using Newtonsoft.Json;

namespace NeuralNetwork.Common.GradientAdjustmentsParameters
{
    /// <summary>
    /// Interface for gradient adjustment parameters.
    /// </summary>
    [JsonConverter(typeof(GradientAdjustmentParametersConverter))]
    public interface IGradientAdjustmentParameters
    {
        GradientAdjustmentType Type { get; }
    }
}