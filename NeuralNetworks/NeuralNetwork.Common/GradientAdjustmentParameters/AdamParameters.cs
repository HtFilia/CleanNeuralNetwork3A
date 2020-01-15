namespace NeuralNetwork.Common.GradientAdjustmentsParameters
{
    /// <summary>
    /// Parameters for adjusting the gradient update using the Adam technique.
    /// </summary>
    /// <seealso cref="NeuralNetwork.Common.GradientAdjustmentsParameters.IGradientAdjustmentParameters" />
    public class AdamParameters : IGradientAdjustmentParameters
    {
        public AdamParameters(double stepSize, double firstMomentDecay, double secondMomentDecay, double denominatorFactor)
        {
            StepSize = stepSize;
            FirstMomentDecay = firstMomentDecay;
            SecondMomentDecay = secondMomentDecay;
            DenominatorFactor = denominatorFactor;
        }

        public AdamParameters()
        {
        }

        public GradientAdjustmentType Type => GradientAdjustmentType.Adam;
        public double StepSize { get; set; }
        public double FirstMomentDecay { get; set; }
        public double SecondMomentDecay { get; set; }
        public double DenominatorFactor { get; set; }
    }
}