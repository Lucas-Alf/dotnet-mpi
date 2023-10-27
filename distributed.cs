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
                /////////////////////
                //     Source      //
                /////////////////////
                if (comm.Rank == 0)
                {
                    Console.WriteLine($"Mode: distributed");
                    var minWorker = 1;
                    var maxWorker = comm.Size - 1;
                    var currentWorker = minWorker;

                    // Get the list of images
                    var images = Directory.GetFiles("images", "*.jpg", SearchOption.AllDirectories);
                    foreach (var file in images)
                    {
                        // Read the image bytes and send to workers
                        var imgBytes = File.ReadAllBytes(file);
                        comm.Send(new Record<byte[]>(file, imgBytes), currentWorker, 0);
                        currentWorker++;
                        if (currentWorker == maxWorker)
                            currentWorker = minWorker;
                    }

                    // Send EOS messages
                    for (int worker = minWorker; worker <= maxWorker; worker++)
                        comm.Send(new Record<byte[]>(file: null, data: null, eos: true), worker, 0);
                }
                /////////////////////
                //     Workers     //
                /////////////////////
                else if (comm.Rank > 0 && comm.Rank < (comm.Size - 1))
                {
                    // Get GPU device if available
                    var device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;

                    // Read the categories list
                    var categories = File.ReadAllLines("imagenet_classes.txt").ToArray();

                    // Get the Object Recognition model
                    var model = torchvision.models.inception_v3(num_classes: categories.Length, skipfc: false, weights_file: "inception_v3.dat");
                    model.eval();
                    model.to(device);

                    // Image preprocessing
                    var preprocess = torchvision.transforms.Compose(
                        torchvision.transforms.ConvertImageDtype(torch.ScalarType.Float32),
                        torchvision.transforms.Resize(1000, 1000)
                    );

                    while (true)
                    {
                        // Receive the messages from source
                        var record = comm.Receive<Record<byte[]>>(0, 0);
                        if (record.EOS)
                        {
                            // If receive a EOS message send the message downstream and break the loop
                            comm.Send(new Record<string>(file: null, data: null, eos: true), comm.Size - 1, 0);
                            break;
                        }

                        // Convert the image bytes to a stream of bytes
                        using (var stream = new MemoryStream(record.Data!))
                        {
                            // Read the image
                            var img = read_image(stream, ImageReadMode.RGB, new SkiaImager()).to(device);
                            
                            // Run the image preprocessing
                            var inputTensor = preprocess.call(img);
                            var input_batch = inputTensor.unsqueeze(0);
                            using (torch.no_grad())
                            {
                                // Run the model over the image and decode the probabilities
                                var output = model.call(input_batch);
                                var probabilities = torch.nn.functional.softmax(output[0], dim: 0);
                                var (topProb, topCatId) = torch.topk(probabilities, 1);
                                var label = categories[topCatId[0].ToInt32()];
                                var accuracy = topProb[0].item<float>();
                                var result = JsonSerializer.Serialize(new { file = record.File, label, accuracy });
                                
                                // Send the result to sink
                                comm.Send(new Record<string>(record.File, result), comm.Size - 1, 0);
                            }
                        }
                    }
                }
                /////////////////////
                //      Sink       //
                /////////////////////
                else
                {
                    // Remove old results if exists
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

                            // On receive EOS messages from all workers finishes the process
                            if (result.EOS)
                            {
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