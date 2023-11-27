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
using MPI;
using System.Text.Json.Serialization;

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
            var sliceSize = array.Length / 2d;
            var leftSize = Convert.ToInt32(Math.Floor(sliceSize));
            var rightSize = Convert.ToInt32(Math.Ceiling(sliceSize));

            var leftSide = array.Take(leftSize).ToArray();
            var rightSide = array.Skip(leftSize).Take(rightSize).ToArray();
            return (leftSide, rightSide);
        }

        private static int[] Interleaving(int[] arr1, int[] arr2)
        {
            var vector = arr1.Concat(arr2).ToArray();
            var size = vector.Length;
            var result = new int[size];
            var i1 = 0;
            var i2 = size / 2;

            for (var i_aux = 0; i_aux < size; i_aux++)
            {
                if ((i1 < size / 2) && ((i2 >= size) || (vector[i1] <= vector[i2])))
                    result[i_aux] = vector[i1++];
                else
                    result[i_aux] = vector[i2++];
            }

            return result;
        }

        public static void BubbleSort(int size)
        {
            MPI.Environment.Run(comm =>
            {
                if (comm.Rank == 0)
                {
                    var inputFilePath = $"input_file_{size}.json";
                    if (!File.Exists(inputFilePath))
                        File.WriteAllText(inputFilePath, JsonSerializer.Serialize(Sequential.GenerateRandomIntArray(size).OrderByDescending(x => x)));

                    var array = JsonSerializer.Deserialize<int[]>(File.ReadAllText(inputFilePath));
                    var delta = Convert.ToInt32(Math.Ceiling(array!.Length / (comm.Size - 1d)));
                    var (leftSide, rightSide) = DivideArray(array);

                    // Console.WriteLine($"Rank: {comm.Rank}, size: {array.Length}, delta: {delta}. Divide! (1, 2)");

                    comm.Send((leftSide, delta, comm.Rank), 1, 0);
                    comm.Send((rightSide, delta, comm.Rank), 2, 0);

                    var leftResult = new int[leftSide.Length];
                    var rightResult = new int[rightSide.Length];

                    comm.Receive(1, 0, ref leftResult);
                    comm.Receive(2, 0, ref rightResult);

                    var output = Interleaving(leftResult, rightResult);
                    File.WriteAllText($"output_distributed_{size}.json", JsonSerializer.Serialize(output));
                }
                else
                {
                    var (array, delta, dad) = comm.Receive<(int[], int, int)>(Communicator.anySource, 0);
                    var leftChild = comm.Rank * 2 + 1;
                    var rightChild = comm.Rank * 2 + 2;

                    if (array.Length <= delta || (leftChild >= comm.Size) || (rightChild >= comm.Size))
                    {
                        // Console.WriteLine($"Rank: {comm.Rank}, size: {array.Length}, delta: {delta}, dad: {dad}. Conquer!");
                        var output = Sequential.BubbleSort(array);
                        comm.Send(output, dad, 0);
                    }
                    else
                    {
                        // Console.WriteLine($"Rank: {comm.Rank}, size: {array.Length}, delta: {delta}, dad: {dad}. Divide! ({leftChild}, {rightChild})");

                        var (leftSide, rightSide) = DivideArray(array);
                        comm.Send((leftSide, delta, comm.Rank), leftChild, 0);
                        comm.Send((rightSide, delta, comm.Rank), rightChild, 0);

                        var leftResult = new int[leftSide.Length];
                        var rightResult = new int[rightSide.Length];

                        comm.Receive(leftChild, 0, ref leftResult);
                        comm.Receive(rightChild, 0, ref rightResult);

                        var output = Interleaving(leftResult, rightResult);
                        comm.Send(output, dad, 0);
                    }
                }
            });
        }

        public static void ParallelPhases(int size)
        {
            MPI.Environment.Run(comm =>
            {
                var inputFile = $"input_file_{comm.Rank}_{size}.json";
                if (!File.Exists(inputFile))
                {
                    var sliceSize = Convert.ToInt32(Math.Floor((double)(size / comm.Size)));
                    File.WriteAllText(inputFile, JsonSerializer.Serialize(Sequential.GenerateRandomIntArray(sliceSize).OrderByDescending(x => x)));
                }

                var array = JsonSerializer.Deserialize<int[]>(File.ReadAllText(inputFile));
                var finished = false;
                var test = 0;

                while (!finished)
                {
                    // ordeno vetor local
                    var output = Sequential.BubbleSort(array);
                    Console.WriteLine($"Rank {comm.Rank}, array: {String.Join(", ", output)}");

                    // verifico condição de parada
                    if (test == 10)
                        finished = true;

                    // se não for np-1, mando o meu maior elemento para a direita
                    if (comm.Rank != comm.Size - 1)
                        comm.Send(output.Last(), comm.Rank + 1, 0);

                    // se não for 0, recebo o maior elemento da esquerda
                    var neighbor = 0;
                    if (comm.Rank != 0)
                        neighbor = comm.Receive<int>(comm.Rank - 1, 0);

                    // comparo se o meu menor elemento é maior do que o maior elemento recebido (se sim, estou ordenado em relação ao meu vizinho)
                    var orderedToNeighbor = new bool[comm.Size];
                    if (output.First() >= neighbor)
                        orderedToNeighbor[comm.Rank] = true;

                    // compartilho o meu estado com todos os processos
                    for (int i = 0; i < comm.Size; i++)
                        comm.Broadcast(ref orderedToNeighbor[i], i);

                    // se todos estiverem ordenados com seus vizinhos, a ordenação do vetor global está pronta ( pronto = TRUE, break)
                    if (orderedToNeighbor.All(x => x == true))
                    {
                        Console.WriteLine($"BREAK Rank {comm.Rank}");
                        finished = true;
                        break;
                    }

                    // senão continuo
                    // troco valores para convergir
                    // se não for o 0, mando os menores valores do meu vetor para a esquerda
                    var slice = 5; // Variar esse valor
                    if (comm.Rank != 0)
                        comm.Send(output.Take(slice).ToArray(), comm.Rank - 1, 0);

                    // se não for np-1, recebo os menores valores da direita
                    var valuesNeighbor = new int[slice];
                    if (comm.Rank != comm.Size - 1)
                    {
                        comm.Receive(comm.Rank + 1, 0, ref valuesNeighbor);

                        // ordeno estes valores com a parte mais alta do meu vetor local
                        var greaterSlice = output.Skip(output.Length - slice).Take(slice).ToArray();
                        var combined = Interleaving(greaterSlice, valuesNeighbor);
                        combined = Sequential.BubbleSort(combined);

                        // Console.WriteLine($"Rank{comm.Rank}, combined: {String.Join(", ", combined)} \n");

                        // Coloca os valores de volta no correto
                        for (int i = 0; i < slice; i++)
                            array[array.Length - slice + i] = combined[i];

                        // devolvo os valores que recebi para a direita
                        comm.Send(combined.Skip(slice).Take(slice).ToArray(), comm.Rank + 1, 0);
                    }

                    // se não for o 0, recebo de volta os maiores valores da esquerda
                    var valuesBack = new int[slice];
                    if (comm.Rank != 0)
                    {
                        comm.Receive(comm.Rank - 1, 0, ref valuesBack);
                        for (int i = 0; i < slice; i++)
                            array[i] = valuesBack[i];
                    }

                    test++;
                }
            });
        }
    }
}