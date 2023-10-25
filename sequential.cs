using static TorchSharp.torchvision.io;
using System.Text.Json;
using TorchSharp;

namespace DotNetMPI
{
    public class Sequential
    {
        public static void ImageRecognition()
        {
            Console.WriteLine($"TorchVersion: {torch.__version__}");
            Console.WriteLine($"Mode: sequential");

            // Remove old results
            if (File.Exists("output.txt"))
                File.Delete("output.txt");

            // Get device
            var device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
            Console.WriteLine($"device: {device}");

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
            using (var fileWriter = File.AppendText("output.txt"))
            {
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
                        var label = categories[topCatId[0].ToInt32()];
                        var accuracy = topProb[0].item<float>();
                        var result = JsonSerializer.Serialize(new { file, label, accuracy });
                        fileWriter.WriteLine(result);
                    }
                }
            }
        }
    }
}