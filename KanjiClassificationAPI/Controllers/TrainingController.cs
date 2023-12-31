using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using System;
using System.IO;
using System.Collections.Generic;
using Microsoft.ML;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Vision;
using Microsoft.Extensions.Logging;
using KanjiClassificationAPI.Models;


namespace KanjiClassificationAPI.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class TrainingController : ControllerBase
    {
        private readonly ILogger<TrainingController> _logger;
        private readonly MLContext _mlContext;
        private readonly string _projectDirectory;
        private readonly string _workspaceRelativePath;
        private readonly string _assetsRelativePath;

        public TrainingController(ILogger<TrainingController> logger)
        {
            _logger = logger;
            _mlContext = new MLContext();

            // Set paths at controller instantiation
            _projectDirectory = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../"));
            _workspaceRelativePath = Path.Combine(_projectDirectory, "workspace");
            _assetsRelativePath = Path.Combine(_projectDirectory, "assets");
        }



        [HttpGet("train")]
        public void Train() {

            try
            {
                    IEnumerable<ImageData> images = LoadImagesFromDirectory(folder: _assetsRelativePath, useFolderNameAsLabel: true);

                    IDataView imageData = _mlContext.Data.LoadFromEnumerable(images);

                    IDataView shuffledData = _mlContext.Data.ShuffleRows(imageData);

                    var preprocessingPipeline = _mlContext.Transforms.Conversion.MapValueToKey(
                            inputColumnName: "Label",
                            outputColumnName: "LabelAsKey")
                        .Append(_mlContext.Transforms.LoadRawImageBytes(
                            outputColumnName: "Image",
                            imageFolder: _assetsRelativePath,
                            inputColumnName: "ImagePath"));


                    IDataView preProcessedData = preprocessingPipeline
                                        .Fit(shuffledData)
                                        .Transform(shuffledData);


                    TrainTestData trainSplit = _mlContext.Data.TrainTestSplit(data: preProcessedData, testFraction: 0.3);
                    TrainTestData validationTestSplit = _mlContext.Data.TrainTestSplit(trainSplit.TestSet);

                    IDataView trainSet = trainSplit.TrainSet;
                    IDataView validationSet = validationTestSplit.TrainSet;
                    IDataView testSet = validationTestSplit.TestSet;

                    var classifierOptions = new ImageClassificationTrainer.Options()
                    {
                        FeatureColumnName = "Image",
                        LabelColumnName = "LabelAsKey",
                        ValidationSet = validationSet,
                        Arch = ImageClassificationTrainer.Architecture.ResnetV2101,
                        MetricsCallback = (metrics) => Console.WriteLine(metrics),
                        TestOnTrainSet = false,
                        ReuseTrainSetBottleneckCachedValues = true,
                        ReuseValidationSetBottleneckCachedValues = true
                    };

                    var trainingPipeline = _mlContext.MulticlassClassification.Trainers.ImageClassification(classifierOptions)
                        .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

                    ITransformer trainedModel = trainingPipeline.Fit(trainSet);

                _mlContext.Model.Save(trainedModel, imageData.Schema, "KanjiClassifier.zip");

                _logger.LogInformation("Training completed successfully.");
                
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "An error occurred during training.");
                throw; // Rethrow the exception to propagate it
            }

        }

        [HttpGet("classify")]
        public void Classify()
        {
            IEnumerable<ImageData> images = LoadImagesFromDirectory(folder: _assetsRelativePath, useFolderNameAsLabel: true);

            IDataView imageData = _mlContext.Data.LoadFromEnumerable(images);

            IDataView shuffledData = _mlContext.Data.ShuffleRows(imageData);

            var preprocessingPipeline = _mlContext.Transforms.Conversion.MapValueToKey(
                    inputColumnName: "Label",
                    outputColumnName: "LabelAsKey")
                .Append(_mlContext.Transforms.LoadRawImageBytes(
                    outputColumnName: "Image",
                    imageFolder: _assetsRelativePath,
                    inputColumnName: "ImagePath"));


            IDataView preProcessedData = preprocessingPipeline
                                .Fit(shuffledData)
                                .Transform(shuffledData);


            TrainTestData trainSplit = _mlContext.Data.TrainTestSplit(data: preProcessedData, testFraction: 0.3);
            TrainTestData validationTestSplit = _mlContext.Data.TrainTestSplit(trainSplit.TestSet);

            IDataView trainSet = trainSplit.TrainSet;
            IDataView validationSet = validationTestSplit.TrainSet;
            IDataView testSet = validationTestSplit.TestSet;

            DataViewSchema predictionPipelineSchema;
            ITransformer trainedModel = _mlContext.Model.Load("KanjiClassifier.zip", out predictionPipelineSchema);

            ClassifyImages(_mlContext, testSet, trainedModel);
        }
        void ClassifyImages(MLContext _mlContext, IDataView data, ITransformer trainedModel)
        {
            PredictionEngine<ModelInput, ModelOutput> predictionEngine = _mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(trainedModel);
            IDataView predictionData = trainedModel.Transform(data);
            IEnumerable<ModelOutput> predictions = _mlContext.Data.CreateEnumerable<ModelOutput>(predictionData, reuseRowObject: true).Take(2000);
            Console.WriteLine("Classifying multiple images");
            int correctPredictions = 0;
            int totalPredictions = 0;

            foreach (var prediction in predictions)
            {
                OutputPrediction(prediction);

                // Compare predicted label with actual label
                if (prediction.Label == prediction.PredictedLabel)
                {
                    correctPredictions++;
                }

                totalPredictions++;
            }

            // Calculate accuracy
            double accuracy = (double)correctPredictions / totalPredictions;
            Console.WriteLine($"Accuracy: {accuracy:P2} (Correct Predictions: {correctPredictions}, Total Predictions: {totalPredictions})");
        }

        void OutputPrediction(ModelOutput prediction)
        {
            string imageName = Path.GetFileName(prediction.ImagePath);
            Console.WriteLine($"Image: {imageName} | Actual Value: {prediction.Label} | Predicted Value: {prediction.PredictedLabel}");
        }

        IEnumerable<ImageData> LoadImagesFromDirectory(string folder, bool useFolderNameAsLabel = true)
        {
            var files = Directory.GetFiles(folder, "*",
            searchOption: SearchOption.AllDirectories);
            _logger.LogInformation($"Files loaded: {files.Length}");

            foreach (var file in files)
            {
                var label = Directory.GetParent(file).Name;

                yield return new ImageData()
                {
                    ImagePath = file,
                    Label = label,
                };
            }
        }

    }



}
