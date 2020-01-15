namespace NeuralNetwork.Common.GradientAdjustmentsParameters
{
    /// <summary>
    /// Parameters for adjusting the gradient update using a fixed learning rate.
    /// </summary>
    /// <seealso cref="NeuralNetwork.Common.GradientAdjustmentsParameters.IGradientAdjustmentParameters" />
    public class FixedLearningRateParameters : IGradientAdjustmentParameters
    {
        public double LearningRate { get; set; }

        public GradientAdjustmentType Type => GradientAdjustmentType.FixedLearningRate;

        public FixedLearningRateParameters(double learningRate)
        {
            LearningRate = learningRate;
        }

        public FixedLearningRateParameters()
        {
        }
    }
}