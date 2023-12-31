namespace KanjiClassificationAPI.Models
{
    public class ModelOutput
    {
        public string ImagePath { get; set; }
        public string Label { get; set; }
        public string PredictedLabel { get; set; }
    }
}
