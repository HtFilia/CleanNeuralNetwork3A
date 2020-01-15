NeuralNetwork solution
 - Should compile directly, without changing anything
 
NetworkTrainer solution
 - Add NeuralNetwork.Common.dll as a reference to the Trainer project
 - Add DataProviders.dll, NeuralNetwork.Common.dll and NeuralNetwork.dll to the Visualizer project

RegressionTester
 - Add DataProviders.dll, NeuralNetwork.Common.dll and NeuralNetwork.dll to the project
 - Run from Console with the command 'dotnet RegressionTester.dll <path-to-serialized-network> <path-to-output-file>
 
