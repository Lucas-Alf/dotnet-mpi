# Install on linux

### Install packages

`sudo apt install gcc openmpi-bin openmpi-common libopenmpi-dev`

###  Dependencies
Extract `MPI_Deps.zip` in `bin/Debug/net7.0`

### Create ready-to-run self-contained build to run on LAD
```
dotnet publish -c Release -r linux-x64 --self-contained -p:PublishReadyToRun=true
```

### Run on LAD
```
ladrun -np <number-of-processes> ./DLinq <file> <batch-size>
```