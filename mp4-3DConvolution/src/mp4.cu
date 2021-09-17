#include <wb.h>

#define wbCheck(stmt)                                              \
    do {                                                           \
        cudaError_t err = stmt;                                    \
        if (err != cudaSuccess) {                                  \
            wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err)); \
            wbLog(ERROR, "Failed to run stmt ", #stmt);            \
            return -1;                                             \
        }                                                          \
    } while (0)

//@@ Define any useful program-wide constants here
#define KERNEL_WIDTH 3
#define TILE_WIDTH 4
#define BLOCK_WIDTH (TILE_WIDTH + ((int)KERNEL_WIDTH / 2) * 2)

//@@ Define constant memory for device kernel here
__constant__ float deviceKernel[KERNEL_WIDTH][KERNEL_WIDTH][KERNEL_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
    //@@ Insert kernel code here

    __shared__ float N_ds[BLOCK_WIDTH][BLOCK_WIDTH][BLOCK_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int x_idx_out = blockIdx.x * TILE_WIDTH + tx;
    int y_idx_out = blockIdx.y * TILE_WIDTH + ty;
    int z_idx_out = blockIdx.z * TILE_WIDTH + tz;

    int x_idx_in = x_idx_out - ((int)KERNEL_WIDTH / 2);
    int y_idx_in = y_idx_out - ((int)KERNEL_WIDTH / 2);
    int z_idx_in = z_idx_out - ((int)KERNEL_WIDTH / 2);

    // Load all input in the tile into share memory
    if (x_idx_in >= 0 && y_idx_in >= 0 && z_idx_in >= 0 && x_idx_in < x_size &&
        y_idx_in < y_size && z_idx_in < z_size) {
        N_ds[tz][ty][tx] =
            input[z_idx_in * x_size * y_size + y_idx_in * x_size + x_idx_in];
    } else {
        N_ds[tz][ty][tx] = 0.0f;
    }
    __syncthreads();

    float sum = 0.0f;
    if ((tx < TILE_WIDTH) && (ty < TILE_WIDTH) && (tz < TILE_WIDTH)) {
        for (int i = 0; i < KERNEL_WIDTH; i++) {
            for (int j = 0; j < KERNEL_WIDTH; j++) {
                for (int k = 0; k < KERNEL_WIDTH; k++) {
                    sum += deviceKernel[i][j][k] * N_ds[i + tz][j + ty][k + tx];
                }
            }
        }
        if ((z_idx_out < z_size) && (y_idx_out < y_size) &&
            (x_idx_out < x_size)) {
            output[z_idx_out * x_size * y_size + y_idx_out * x_size +
                   x_idx_out] = sum;
        }
    }
}

int main(int argc, char *argv[]) {
    wbArg_t args;
    int z_size;
    int y_size;
    int x_size;
    int inputLength, kernelLength;
    float *hostInput;
    float *hostKernel;
    float *hostOutput;
    float *deviceInput;
    float *deviceOutput;

    args = wbArg_read(argc, argv);

    // Import data
    hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostKernel = (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
    hostOutput = (float *)malloc(inputLength * sizeof(float));

    // First three elements are the input dimensions
    z_size = hostInput[0];
    y_size = hostInput[1];
    x_size = hostInput[2];
    wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
    assert(z_size * y_size * x_size == inputLength - 3);
    assert(kernelLength == 27);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    //@@ Allocate GPU memory here
    // Recall that inputLength is 3 elements longer than the input data
    // because the first  three elements were the dimensions
    int inputCount = z_size * y_size * x_size;
    int inputSize = inputCount * sizeof(float);
    int kernelSize = kernelLength * sizeof(float);

    cudaMalloc((void **)&deviceInput, inputSize);
    cudaMalloc((void **)&deviceOutput, inputSize);

    wbTime_stop(GPU, "Doing GPU memory allocation");

    wbTime_start(Copy, "Copying data to the GPU");
    //@@ Copy input and kernel to GPU here
    // Recall that the first three elements of hostInput are dimensions and
    // do
    // not need to be copied to the gpu
    cudaMemcpy(deviceInput, hostInput + 3, inputSize, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(deviceKernel, hostKernel, kernelSize, 0,
                       cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");

    wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ Initialize grid and block dimensions here
    dim3 DimGrid(ceil(x_size / float(TILE_WIDTH)),
                 ceil(y_size / float(TILE_WIDTH)),
                 ceil(z_size / float(TILE_WIDTH)));
    dim3 DimBlock(BLOCK_WIDTH, BLOCK_WIDTH, BLOCK_WIDTH);

    conv3d<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, z_size, y_size,
                                  x_size);

    //@@ Launch the GPU kernel here
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Doing the computation on the GPU");

    wbTime_start(Copy, "Copying data from the GPU");
    //@@ Copy the device memory back to the host here
    // Recall that the first three elements of the output are the dimensions
    // and should not be set here (they are set below)
    cudaMemcpy(hostOutput + 3, deviceOutput, inputSize, cudaMemcpyDeviceToHost);

    // for (int i = 0; i < z_size; i++) {
    //     for (int j = 0; j < y_size; j++) {
    //         for (int k = 0; k < x_size; k++) {
    //             std::cout
    //                 << hostOutput[i * y_size * x_size + j * x_size + k + 3]
    //                 << " ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }

    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    // Set the output dimensions for correctness checking
    hostOutput[0] = z_size;
    hostOutput[1] = y_size;
    hostOutput[2] = x_size;
    wbSolution(args, hostOutput, inputLength);

    // Free device memory
    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    // Free host memory
    free(hostInput);
    free(hostOutput);
    return 0;
}
