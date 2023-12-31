using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using System;
using SixLabors;
using Microsoft.ML;
using Microsoft.ML.Vision;
using KanjiClassificationAPI.Models;
using Microsoft.Extensions.Logging;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.Processing;

namespace KanjiClassificationAPI.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class ClassificationController : ControllerBase
    {
        private readonly MLContext _mlContext;
        private readonly ITransformer _trainedModel;
        private readonly PredictionEngine<ModelInput, ModelOutput> _predictionEngine;
        private readonly ILogger<ClassificationController> _logger;


        private readonly string _projectDirectory;
        private readonly string _workspaceRelativePath;
        private readonly string _assetsRelativePath;

        public ClassificationController(
            MLContext mlContext,
            ITransformer trainedModel,
            PredictionEngine<ModelInput, ModelOutput> predictionEngine,
            ILogger<ClassificationController> logger)
        {
            _mlContext = mlContext;
            _trainedModel = trainedModel;
            _predictionEngine = predictionEngine;
            _logger = logger;

            // Set paths at controller instantiation
            _projectDirectory = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../"));
            _workspaceRelativePath = Path.Combine(_projectDirectory, "workspace");
            _assetsRelativePath = Path.Combine(_projectDirectory, "assets");
        }

        [HttpPost("classify_image")]
        public IActionResult ClassifyImage([FromForm] IFormFile image)
        {
            try
            {
                // Resize the image
                var resizedImage = ResizeImage(image);

                // Convert the resized image to a byte array
                byte[] imageBytes;
                using (var memoryStream = new MemoryStream())
                {
                    resizedImage.Save(memoryStream, new JpegEncoder());
                    imageBytes = memoryStream.ToArray();
                }

                // Create a ModelInput object
                var modelInput = new ModelInput
                {
                    Image = imageBytes,
                    ImagePath = "", 
                    Label = "" 
                };

                // Use the PredictionEngine to predict the label for the input image
                var prediction = _predictionEngine.Predict(modelInput);

                // Return the predicted label
                return Ok(prediction.PredictedLabel);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "An error occurred during image classification.");
                return StatusCode(500, "Internal Server Error");
            }

        }

        private Image<Rgb24> ResizeImage(IFormFile image)
        {
            // Load the image from the form file
            using (var stream = image.OpenReadStream())
            {
                var imageSharp = Image.Load(stream);

                // Resize the image to your desired dimensions
                imageSharp.Mutate(x => x.Resize(new ResizeOptions
                {
                    Size = new Size(224, 224),
                    Mode = ResizeMode.Max
                }));

                return imageSharp.CloneAs<Rgb24>();
            }
        }
    }
}
