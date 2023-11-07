
using Microsoft.VisualBasic;
using TorchSharp;
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
                        var array = Sequential.GenerateRandomIntArray(1000);
                        var sorted = Sequential.BubbleSort(array);
                        Assert.Equal(sorted, array.Order());
                        Console.WriteLine(String.Join(", ", sorted));
                    }
                    break;
                case ("-distributed", "-bubbleSort"):
                        Distributed.BubbleSort();
                    break;
                default:
                    throw new Exception("Invalid mode");
            }
        }
    }
}