namespace DotNetMPI
{
    [Serializable]
    public class Record<T>
    {
        public string? File { get; set; }
        public T? Data { get; set; }
        public bool EOS { get; set; }

        public Record(string? file, T? data, bool eos = false)
        {
            File = file;
            Data = data;
            EOS = eos;
        }
    }
}
