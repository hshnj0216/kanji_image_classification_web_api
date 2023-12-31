
using KanjiClassificationAPI.Models;
using Microsoft.ML;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddSingleton<MLContext>();
builder.Services.AddSingleton(sp =>
{
    // Load the trained model
    var mlContext = sp.GetRequiredService<MLContext>();
    var trainedModel = mlContext.Model.Load("KanjiClassifier.zip", out var predictionPipelineSchema);

    return trainedModel; 
});

// Create and register the PredictionEngine as a singleton
builder.Services.AddSingleton(sp =>
{
    var mlContext = sp.GetRequiredService<MLContext>();
    var trainedModel = sp.GetRequiredService<ITransformer>(); // Use the actual type
    return mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(trainedModel);
});

builder.Services.AddControllers();
// Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();
builder.Services.AddLogging();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

//app.UseHttpsRedirection();

app.UseAuthorization();

app.MapControllers();

app.Run();



    


