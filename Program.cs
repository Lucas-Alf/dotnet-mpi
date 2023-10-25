
using TorchSharp;

namespace DotNetMPI
{
    class DotNetMPI
    {
        static void Main(string[] args)
        {
            if (args.Length < 1)
            {
                Console.WriteLine("command: dotnet-mpi -sequential|-distributed");
                Console.WriteLine("WARNING: on distributed mode, the command must be called with mpiexec.");
                Console.WriteLine("Ex: mpiexec -n <number-of-processes> dotnet-mpi -distributed <path> <batch-size>");
                Environment.Exit(0);
            }

            var mode = args[0];
            switch (mode)
            {
                case "-sequential":
                    Sequential.ImageRecognition();
                    break;
                case "-distributed":
                    Distributed.ImageRecognition();
                    break;
                default:
                    throw new Exception("Invalid mode");
            }
        }
    }
}