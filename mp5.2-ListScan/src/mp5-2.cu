// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512  //@@ You can change this

#define wbCheck(stmt)                                                      \
    do {                                                                   \
        cudaError_t err = stmt;                                            \
        if (err != cudaSuccess) {                                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                    \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err)); \
            return -1;                                                     \
        }                                                                  \
    } while (0)

__global__ void scan(float *input, float *output, int len, float *blockSumArray) {
    //@@ Modify the body of this function to complete the functionality of
    //@@ the scan on the device
    //@@ You may need multiple kernel calls; write your kernels before this
    //@@ function and call them from the host

    unsigned int tx = threadIdx.x;
    unsigned int bx = blockIdx.x;
    unsigned int bDimx = blockDim.x;

    unsigned int i = 2 * bx * bDimx + tx;

    __shared__ float XY[2 * BLOCK_SIZE];

    XY[tx] = i < len ? input[i] : 0;
    XY[tx + bDimx] = i + bDimx < len ? input[i + bDimx] : 0;

    // Reduction Step
    for (unsigned int stride = 1; stride <= bDimx; stride = stride << 1) {
        __syncthreads();
        int index = (tx + 1) * 2 * stride - 1;
        if (index < 2 * BLOCK_SIZE) {
            XY[index] += XY[index - stride];
        }
    }

    // Post Scan Step
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride = stride >> 1) {
        __syncthreads();
        int index = (tx + 1) * stride * 2 - 1;
        if (index + stride < 2 * BLOCK_SIZE) {
            XY[index + stride] += XY[index];
        }
    }

    __syncthreads();
    if (i < len) {
        output[i] = XY[tx];
    }
    if (i + bDimx < len) {
        output[i + bDimx] = XY[tx + bDimx];
    }

    __syncthreads();
    if (tx == BLOCK_SIZE - 1 && blockSumArray != nullptr) {
        blockSumArray[bx] = XY[2 * BLOCK_SIZE - 1];
    }
}

__global__ void addOffset(float *output, float *blockSumArray, int len) {
    unsigned int i = threadIdx.x + (blockIdx.x * blockDim.x * 2);

    __shared__ float offset;

    if (threadIdx.x == 0) {
        offset = blockIdx.x == 0 ? 0 : blockSumArray[blockIdx.x - 1];
    }

    __syncthreads();

    output[i] += offset;
    output[i + blockDim.x] += offset;
}


int main(int argc, char **argv) {
    wbArg_t args;
    float *hostInput;   // The input 1D list
    float *hostOutput;  // The output list
    float *deviceInput;
    float *deviceOutput;
    float *deviceBlockSumArray;
    int numElements;  // number of elements in the list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float *)malloc(numElements * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ",
          numElements);

    int blockSumSize = ceil(numElements / (2.0 * BLOCK_SIZE)) * sizeof(float);

    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
    wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
    wbCheck(cudaMalloc((void **)&deviceBlockSumArray, blockSumSize));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                       cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
    dim3 dimGrid(ceil(numElements / float(BLOCK_SIZE * 2)), 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);

    dim3 dimGridReduce(1, 1, 1);

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Modify this to complete the functionality of the scan
    //@@ on the deivce
    scan<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, numElements, deviceBlockSumArray);
    scan<<<dimGridReduce, dimBlock>>>(deviceBlockSumArray, deviceBlockSumArray, blockSumSize, nullptr);
    addOffset<<<dimGrid, dimBlock>>>(deviceOutput, deviceBlockSumArray, numElements);

    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                       cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);

    return 0;
}
