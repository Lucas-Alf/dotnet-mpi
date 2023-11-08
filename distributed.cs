// Devido ao código fonte possuir diversas classes
// foi optado por disponibilizar apenas o código da 
// versão distribuída como apêndice, o restante do 
// código pode ser encontrado no repositório do Github:
// https://github.com/Lucas-Alf/dotnet-mpi

using static TorchSharp.torchvision.io;
using static TorchSharp.torchvision;
using System.Text.Json;
using TorchSharp;
using TorchSharp.Modules;

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
                    var images = Directory.GetFiles(
                        path: "images",
                        searchPattern: "*.jpg",
                        searchOption: SearchOption.AllDirectories
                    );

                    foreach (var imgPath in images)
                    {
                        // Read the image bytes and send to workers
                        var imgBytes = File.ReadAllBytes(imgPath);
                        comm.Send(
                            value: new Record<byte[]>(
                                file: imgPath,
                                data: imgBytes
                            ),
                            dest: currentWorker,
                            tag: 0
                        );
                        currentWorker++;
                        if (currentWorker == maxWorker)
                            currentWorker = minWorker;
                    }

                    // Send EOS messages
                    for (int worker = minWorker; worker <= maxWorker; worker++)
                    {
                        comm.Send(
                            value: new Record<byte[]>(
                                file: null,
                                data: null,
                                eos: true
                            ),
                            dest: worker,
                            tag: 0
                        );
                    }
                }
                /////////////////////
                //     Workers     //
                /////////////////////
                else if (comm.Rank > 0 && comm.Rank < (comm.Size - 1))
                {
                    // Get GPU device if available
                    var device = torch.cuda.is_available()
                        ? torch.CUDA
                        : torch.CPU;

                    // Read the categories list
                    var categories = File
                        .ReadAllLines("imagenet_classes.txt")
                        .ToArray();

                    // Get the Object Recognition model
                    var model = models.inception_v3(
                        num_classes: categories.Length,
                        skipfc: false,
                        weights_file: "inception_v3.dat"
                    );

                    model.eval();
                    model.to(device);

                    // Image preprocessing
                    var preprocess = transforms.Compose(
                        transforms.ConvertImageDtype(torch.ScalarType.Float32),
                        transforms.Resize(1000, 1000)
                    );

                    while (true)
                    {
                        // Receive the messages from source
                        var record = comm.Receive<Record<byte[]>>(0, 0);
                        if (record.EOS)
                        {
                            // If receive a EOS message send the message 
                            // downstream and break the loop
                            comm.Send(
                                value: new Record<string>(
                                    file: null,
                                    data: null,
                                    eos: true
                                ),
                                dest: comm.Size - 1,
                                tag: 0
                            );
                            break;
                        }

                        // Convert the image bytes to a stream of bytes
                        using (var imgStream = new MemoryStream(record.Data!))
                        {
                            // Read the image
                            var img = read_image(
                                stream: imgStream,
                                mode: ImageReadMode.RGB,
                                imager: new SkiaImager()
                            );

                            img = img.to(device);

                            // Run the image preprocessing
                            var inputTensor = preprocess.call(img);
                            var input_batch = inputTensor.unsqueeze(0);
                            using (torch.no_grad())
                            {
                                // Run the model over the image 
                                // and decode the probabilities
                                var output = model.call(input_batch);
                                var probs = torch.nn.functional.softmax(output[0], 0);
                                var (topProb, topCatId) = torch.topk(probs, 1);
                                var label = categories[topCatId[0].ToInt32()];
                                var accuracy = topProb[0].item<float>();
                                var result = JsonSerializer.Serialize(new
                                {
                                    file = record.File,
                                    label = label,
                                    accuracy = accuracy
                                });

                                // Send the result to sink
                                comm.Send(
                                    value: new Record<string>(
                                        file: record.File,
                                        data: result
                                    ),
                                    dest: comm.Size - 1,
                                    tag: 0
                                );
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
                    var worker = minWorker;

                    using (var fileWriter = File.AppendText("output.txt"))
                    {
                        while (true)
                        {
                            // Receive messages from workers
                            var result = comm.Receive<Record<string>>(worker, 0);
                            worker++;
                            if (worker == maxWorker)
                                worker = minWorker;

                            // On receive EOS messages from all 
                            // workers finishes the process
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

        private static (int[], int[]) DivideArray(int[] array)
        {
            var sliceSize = array.Length / 2;
            var leftSide = array.Take(sliceSize).ToArray();
            var rightSide = array.Skip(sliceSize).Take(sliceSize).ToArray();
            return (leftSide, rightSide);
        }

        public static void BubbleSort()
        {
            MPI.Environment.Run(comm =>
            {
                if (comm.Rank == 0)
                {
                    var array = Sequential.GenerateRandomIntArray(40);
                    var delta = array.Length / comm.Size;
                    var (leftSide, rightSide) = DivideArray(array);
                    var depth = 0;

                    comm.Send((leftSide, delta, comm.Rank, depth), 1, 0);
                    comm.Send((rightSide, delta, comm.Rank, depth), 2, 0);

                    int[] resultLeft = new int[array.Length / 2];
                    int[] resultRight = new int[array.Length / 2];

                    comm.Receive(1, 0, ref resultLeft);
                    comm.Receive(2, 0, ref resultRight);

                    var concat = resultLeft.Concat(resultRight);

                    var output = Sequential.BubbleSort(array);

                    Console.WriteLine(String.Join(", ", output));
                }
                else
                {
                    var (array, delta, dad, depth) = comm.Receive<(int[], int, int, int)>(Communicator.anySource, 0);
                    if (array.Length <= delta || (comm.Rank + (4 + depth)) > comm.Size - 1)
                    {
                        var output = Sequential.BubbleSort(array);
                        comm.Send(output, dad, 0);
                    }
                    else
                    {
                        var leftChild = 0;
                        var rightChild = 0;
                        if (comm.Rank % 2 == 0) // Rank é par
                        {
                            leftChild = comm.Rank + (3 + depth);
                            rightChild = comm.Rank + (4 + depth);
                        }
                        else
                        {
                            leftChild = comm.Rank + (2 + depth);
                            rightChild = comm.Rank + (3 + depth);
                        }

                        //Console.WriteLine("Rank " + comm.Rank);
                        //Console.WriteLine("Left " + leftChild);
                        //Console.WriteLine("Right " + rightChild);

                        var (leftSide, rightSide) = DivideArray(array);

                        comm.Send((leftSide, delta, comm.Rank, depth + 1), leftChild, 0);
                        comm.Send((rightSide, delta, comm.Rank, depth + 1), rightChild, 0);

                        int[] resultLeft = new int[delta * 2]; // Needs to be better
                        int[] resultRight = new int[delta * 2]; // Needs to be better

                        comm.Receive(leftChild, 0, ref resultLeft);
                        comm.Receive(rightChild, 0, ref resultRight);
                        var concat = resultLeft.Concat(resultRight).ToArray();

                        var output = Sequential.BubbleSort(concat);

                        comm.Send(output, dad, 0);
                    }
                }
            });
        }
    }
}