using static TorchSharp.torchvision.io;
using System.Text.Json;
using TorchSharp;

namespace DotNetMPI
{
    public class Sequential
    {
        public static void ImageRecognition()
        {
            Console.WriteLine($"Mode: sequential");

            // Remove old results
            if (File.Exists("output.txt"))
                File.Delete("output.txt");

            // Get device
            var device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
            // Console.WriteLine($"device: {device}");

            // Read the categories
            var categories = File.ReadAllLines("imagenet_classes.txt").ToArray();

            // Get model
            var model = torchvision.models.inception_v3(num_classes: categories.Length, skipfc: false, weights_file: "inception_v3.dat");
            model.eval();
            model.to(device);

            // Image preprocessing
            var preprocess = torchvision.transforms.Compose(
                torchvision.transforms.ConvertImageDtype(torch.ScalarType.Float32),
                torchvision.transforms.Resize(1000, 1000)
            );

            var images = Directory.GetFiles("images", "*.jpg", SearchOption.AllDirectories);
            using (var fileWriter = File.AppendText("output.txt"))
            {
                foreach (var file in images)
                {
                    var img = read_image(file, ImageReadMode.RGB, new SkiaImager()).to(device);
                    var inputTensor = preprocess.call(img);
                    var input_batch = inputTensor.unsqueeze(0);
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

        public static int[] GenerateRandomIntArray(int size)
        {
            var input = new int[size];
            var randNum = new Random();
            for (int i = 0; i < input.Length; i++)
                input[i] = randNum.Next(0, size);

            return input;
        }


        public static int[] BubbleSort(int[] arr)
        {
            var temp = 0;
            for (var write = 0; write < arr.Length; write++)
            {
                for (var sort = 0; sort < arr.Length - 1; sort++)
                {
                    if (arr[sort] > arr[sort + 1])
                    {
                        temp = arr[sort + 1];
                        arr[sort + 1] = arr[sort];
                        arr[sort] = temp;
                    }
                }
            }

            return arr;
        }
    }
}