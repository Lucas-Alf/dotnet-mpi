
using TorchSharp;
using static TorchSharp.torchvision.io;

class ImageClassification
{
    static void Main(string[] args)
    {
        // Get device
        var device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;

        // Read the categories
        var categories = File.ReadAllLines("imagenet_classes.txt").ToArray();

        // Get model
        var model = torchvision.models.inception_v3(num_classes: categories.Length, skipfc: false, weights_file: "inception_v3.dat");
        model.eval();
        model.to(device);

        // Image preprocessing
        var preprocess = torchvision.transforms.Compose(
            torchvision.transforms.ConvertImageDtype(torch.ScalarType.Float32),
            torchvision.transforms.Resize(299, 299)
        );

        var images = Directory.GetFiles("images", "*.jpg", SearchOption.AllDirectories);
        foreach (var file in images)
        {
            var img = read_image(file, ImageReadMode.RGB, new SkiaImager());
            var inputTensor = preprocess.call(img).to(device);
            var input_batch = inputTensor.unsqueeze(0);
            input_batch = input_batch.to(device);
            using (torch.no_grad())
            {
                var output = model.call(input_batch);
                var probabilities = torch.nn.functional.softmax(output[0], dim: 0);
                var (topProb, topCatId) = torch.topk(probabilities, 1);
                Console.WriteLine($@"
                    File: {file}
                    Label: {categories[topCatId[0].ToInt32()]}
                    Accuracy: {topProb[0].item<float>()}
                ");
            }
        }
    }
}