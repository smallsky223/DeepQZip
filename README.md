# DeepQZip

## Description
A losslessly compressor using LSTM network combined with quality score for the third generation sequencing of FASTQ files.


## Requirements
0. GPU
1. CUDA 11.1
2. CUDNN 8
3. python 3
4. numpy
5. pytorch-gpu 1.8.0
6. tqdm
7. libtorch 1.8.0


## Code
To run a compression experiment: 


### Data Preparation
Place all the data(FASTQ files) to be compressed in data/files_to_be_compressed


### DeepQC Preparation
cd c++
wget https://download.pytorch.org/libtorch/cu111/libtorch-cxx11-abi-shared-with-deps-1.8.0%2Bcu111.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.8.0%2Bcu111.zip

compile in following steps:
*   Edit CMakeLists.txt to config Libtorch path
*   mkdir build && cd build
*   cmake ..
*   make
*   cp DeepQC ../

Make sure you have already installed CUDA, CUDNN, Libtorch successfully.


### Running
cd python
./run_experiments.sh Rate GPUID File_Name Alpha

Note:
Rate means the proportion of training data in original data; 
GPUID means the id of GPU used for training; 
File_Name means the file name of the file to be compressed; 
Alpha indicates how many parallel groups to divide all reads equally,
we suggest set alpha to 5000 when the quality score file size smaller than 100MB, 20000 when larger than 1GB.

For sample data SRR3211986_9000.fastq in data/files_to_be_compressed, The corresponding command would be then `./run_experiments.sh 0.01 0 SRR3211986_9000 5000`
