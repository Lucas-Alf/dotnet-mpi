using static TorchSharp.torchvision.io;
using System.Text.Json;
using TorchSharp;

namespace DotNetMPI
{
    public class Distributed
    {
        public static void ImageRecognition()
        {
            MPI.Environment.Run(comm =>
            {
                // Source
                if (comm.Rank == 0)
                {
                    Console.WriteLine($"TorchVersion: {torch.__version__}");
                    Console.WriteLine($"Mode: distributed");

                    var minWorker = 1;
                    var maxWorker = comm.Size - 1;
                    var currentWorker = minWorker;
                    var images = Directory.GetFiles("images", "*.jpg", SearchOption.AllDirectories);
                    foreach (var file in images)
                    {
                        var imgBytes = File.ReadAllBytes(file);
                        comm.Send(new Record<byte[]>(file, imgBytes), currentWorker, 0);
                        // Console.WriteLine($"Source send to worker {currentWorker}");
                        currentWorker++;
                        if (currentWorker == maxWorker)
                            currentWorker = minWorker;
                    }

                    // Send EOS messages
                    for (int worker = minWorker; worker <= maxWorker; worker++)
                        comm.Send(new Record<byte[]>(file: null, data: null, eos: true), worker, 0);
                }
                // Workers
                else if (comm.Rank > 0 && comm.Rank < (comm.Size - 1))
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

                    while (true)
                    {
                        var record = comm.Receive<Record<byte[]>>(0, 0);
                        // Console.WriteLine($"Worker {comm.Rank} received message");
                        if (record.EOS)
                        {
                            // Send EOS message downstream
                            comm.Send(new Record<string>(file: null, data: null, eos: true), comm.Size - 1, 0);
                            break;
                        }

                        using (var stream = new MemoryStream(record.Data!))
                        {
                            var img = read_image(stream, ImageReadMode.RGB, new SkiaImager());
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
                                var result = JsonSerializer.Serialize(new { file = record.File, label, accuracy });
                                comm.Send(new Record<string>(record.File, result), comm.Size - 1, 0);
                            }
                        }
                    }
                }
                // Sink
                else
                {
                    // Remove old results
                    if (File.Exists("output.txt"))
                        File.Delete("output.txt");

                    var eosCounter = 0;
                    var minWorker = 1;
                    var maxWorker = comm.Size - 1;
                    var currentWorker = minWorker;

                    using (var fileWriter = File.AppendText("output.txt"))
                    {
                        while (true)
                        {
                            // Receive messages from workers
                            var result = comm.Receive<Record<string>>(currentWorker, 0);
                            currentWorker++;
                            if (currentWorker == maxWorker)
                                currentWorker = minWorker;

                            // On receive EOS from all workers finishes the process
                            if (result.EOS)
                            {
                                // Console.WriteLine($"Sink received message from worker {currentWorker}: {(result.EOS ? "EOS" : result.Data)}");
                                eosCounter++;
                                if (eosCounter == (maxWorker - 1))
                                    break;
                            }

                            // Append the results of output.txt
                            fileWriter.WriteLine(result.Data);
                        }
                    }
                }
            });
        }
    }
}