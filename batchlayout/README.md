## System Requirements

Users need to have following softwares/tools installed in their PC. The source code was compiled and run successfully in both linux and macOS.
```
GCC version >= 4.9
OpenMP version >= 4.5
```
Some helpful links for installation can be found at [GCC](https://gcc.gnu.org/install/), [OpenMP](https://clang-omp.github.io) and [Environment Setup](http://heather.cs.ucdavis.edu/~matloff/158/ToolsInstructions.html#compile_openmp).

## Compile BatchLayout
To compile batchlayout, type the following command on terminal:
```
$ make clean
$ make
```
This will generate an executible file in bin folder.

## Run batchlayout from command line

Input file must be in matrix market format ([check here for details about .mtx file](https://math.nist.gov/MatrixMarket/formats.html)). A lot of datasets can be found at [suitesparse website](https://sparse.tamu.edu). We provide few example input files in datasets/input directory. To run BatchLayout, use the following command:
```
$ ./bin/BatchLayoutEmbed -input ../datasets/cora.mtx -output ../datasets/ -iter 600 -batch 256 -threads 32
```
Here, `-input` is the full path of input file, `-output` is the directory where output/embedding file will be saved, `-iter` is the number of iterations, `-batch` is the size of minibatch which is 256 here, and `-threads` is the maximum number of threads which is 32. All options are described below:
```
-input <string>, full path of input file (required).
-output <string>, directory where output file will be stored.
-batch <int>, size of minibatch.
-iter <int>, number of iteration.
-threads <int>, number of threads, default value is maximum available threads in the machine.
```

### Contact 
If you have questions, please don't hesitate to ask me (Md. Khaledur Rahman) by sending email to `morahma@iu.edu`.
