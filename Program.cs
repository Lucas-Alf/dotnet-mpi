using System.Text.Json;
using Xunit;

namespace DotNetMPI
{
    class DotNetMPI
    {
        static void Main(string[] args)
        {
            if (args.Length < 1)
            {
                Console.WriteLine("command: dotnet-mpi -sequential|-distributed -application");
                Console.WriteLine("WARNING: on distributed mode, the command must be called with mpiexec.");
                Console.WriteLine("Ex: mpiexec -n <number-of-processes> dotnet-mpi -distributed <path> <batch-size>");
                Environment.Exit(0);
            }

            var mode = args[0];
            var application = args[1];
            switch (mode, application)
            {
                case ("-sequential", "-imageRecognition"):
                    Sequential.ImageRecognition();
                    break;
                case ("-distributed", "-imageRecognition"):
                    Distributed.ImageRecognition();
                    break;
                case ("-sequential", "-bubbleSort"):
                    {
                        var size = Convert.ToInt32(args[2]);
                        var inputFilePath = $"input_file_{size}.json";
                        if (!File.Exists(inputFilePath))
                            File.WriteAllText(inputFilePath, JsonSerializer.Serialize(Sequential.GenerateRandomIntArray(size).OrderByDescending(x => x)));

                        var array = JsonSerializer.Deserialize<int[]>(File.ReadAllText(inputFilePath));
                        var sorted = Sequential.BubbleSort(array!);
                        File.WriteAllText($"output_sequential_{size}.json", JsonSerializer.Serialize(sorted));
                    }
                    break;
                case ("-distributed", "-bubbleSort"):
                    {
                        var size = Convert.ToInt32(args[2]);
                        Distributed.BubbleSort(size);
                    }
                    break;
                case ("-sequential", "-parallelPhases"):
                    Console.WriteLine("Sorry, this is not implemented");
                    break;
                case ("-distributed", "-parallelPhases"):
                    {
                        var size = Convert.ToInt32(args[2]);
                        var slicePercent = Convert.ToDecimal(args[3]);
                        Directory.CreateDirectory("input");
                        Directory.CreateDirectory("output");
                        Distributed.ParallelPhases(size, slicePercent);
                    }
                    break;
                default:
                    throw new Exception("Invalid mode");
            }
        }
    }
}