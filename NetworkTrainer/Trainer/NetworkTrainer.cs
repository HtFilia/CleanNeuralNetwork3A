using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common;
using System;
using Trainer.CostFunctions;
using Trainer.DataShufflers;

namespace Trainer
{
    public class NetworkTrainer
    {
        public NetworkTrainer(INetwork network, ICostFunction costFunction, IDataShuffler dataShuffler)
        {
            Network = network ?? throw new ArgumentNullException(nameof(network));
            CostFunction = costFunction ?? throw new ArgumentNullException(nameof(costFunction));
            DataShuffler = dataShuffler ?? throw new ArgumentNullException(nameof(dataShuffler));
            CostResult = Matrix<double>.Build.Dense(Network.Output.RowCount, Network.Output.ColumnCount);
            InitialBatchSize = Network.BatchSize;
            Network.Mode = Mode.Training;
        }

        public INetwork Network { get; }
        public ICostFunction CostFunction { get; }
        public IDataShuffler DataShuffler { get; }
        public Matrix<double> CostResult { get; private set; }
        public int InitialBatchSize { get; }

        public void SetBatchSize(int batchSize)
        {
            Network.BatchSize = batchSize;
            CostResult = Matrix<double>.Build.Dense(Network.Output.RowCount, Network.Output.ColumnCount);
        }

        public void Train(MathData data)
        {
            Network.Mode = Mode.Training;
            var batchSize = InitialBatchSize;
            var shuffledData = DataShuffler.MakeShuffledData(data);
            var inputs = shuffledData.Inputs;
            var outputs = shuffledData.Outputs;
            int totalInputs = inputs.ColumnCount;
            int batchNb = totalInputs / batchSize;
            var remainingBatchSize = totalInputs % batchSize;
            int entrySize = inputs.RowCount;
            int outputSize = outputs.RowCount;
            for (int i = 0; i < batchNb; i++)
            {
                var batchMatrix = inputs.SubMatrix(0, entrySize, i * batchSize, batchSize);
                Network.Propagate(batchMatrix);
                var expectedOutputMatrix = outputs.SubMatrix(0, outputSize, i * batchSize, batchSize);
                var actualOutputMatrix = Network.Output;
                GetCostGradient(expectedOutputMatrix, actualOutputMatrix, CostResult);
                Network.Learn(CostResult);
            }
            if (remainingBatchSize != 0)
            {
                SetBatchSize(remainingBatchSize);
                var remainingBatchMatrix = inputs.SubMatrix(0, entrySize, batchNb * batchSize, remainingBatchSize);
                Network.Propagate(remainingBatchMatrix);
                var remainingExpectedOutputMatrix = outputs.SubMatrix(0, outputSize, batchNb * batchSize, remainingBatchSize);
                var remainingActualOutputMatrix = Network.Output;
                GetCostGradient(remainingExpectedOutputMatrix, remainingActualOutputMatrix, CostResult);
                Network.Learn(CostResult);
                SetBatchSize(InitialBatchSize);
            }
        }

        public double Validate(MathData data)
        {
            Network.Mode = Mode.Evaluation;
            var batchSize = Network.BatchSize;
            var inputs = data.Inputs;
            var outputs = data.Outputs;
            int totalInputs = inputs.ColumnCount;
            int batchNb = totalInputs / batchSize;
            var remainingBatchSize = totalInputs % batchSize;
            int entrySize = inputs.RowCount;
            int outputSize = outputs.RowCount;
            double runningSum = 0;
            for (int i = 0; i < batchNb; i++)
            {
                var batchMatrix = inputs.SubMatrix(0, entrySize, i * batchSize, batchSize);
                Network.Propagate(batchMatrix);
                var expectedOutputMatrix = outputs.SubMatrix(0, outputSize, i * batchSize, batchSize);
                var actualOutputMatrix = Network.Output;
                GetCostOutput(expectedOutputMatrix, actualOutputMatrix, CostResult);
                runningSum += CostResult.RowSums()[0];
            }
            if (remainingBatchSize != 0)
            {
                SetBatchSize(remainingBatchSize);
                var remainingBatchMatrix = inputs.SubMatrix(0, entrySize, batchNb * batchSize, remainingBatchSize);
                Network.Propagate(remainingBatchMatrix);
                var remainingExpectedOutputMatrix = outputs.SubMatrix(0, outputSize, batchNb * batchSize, remainingBatchSize);
                var remainingActualOutputMatrix = Network.Output;
                GetCostOutput(remainingExpectedOutputMatrix, remainingActualOutputMatrix, CostResult);
                runningSum += CostResult.RowSums()[0];
                SetBatchSize(InitialBatchSize);
            }
            return runningSum / totalInputs;
        }

        private void GetCostGradient(Matrix<double> expectedOutput, Matrix<double> actualOutput, Matrix<double> result)
        {
            actualOutput.Map2(CostFunction.DerivApply, expectedOutput, result);
        }

        private void GetCostOutput(Matrix<double> expectedOutput, Matrix<double> actualOutput, Matrix<double> result)
        {
            actualOutput.Map2(CostFunction.Apply, expectedOutput, result);
        }
    }
}