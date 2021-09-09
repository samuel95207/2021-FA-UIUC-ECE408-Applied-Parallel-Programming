#include <wb.h>

#define wbCheck(stmt)                                                      \
    do {                                                                   \
        cudaError_t err = stmt;                                            \
        if (err != cudaSuccess) {                                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                    \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err)); \
            return -1;                                                     \
        }                                                                  \
    } while (0)

#define BLOCK_WIDTH 32

// Compute C = A * B
__global__ void matrixMultiply(float* A, float* B, float* C, int numARows,
                               int numAColumns, int numBRows, int numBColumns,
                               int numCRows, int numCColumns) {
    //@@ Insert code to implement matrix multiplication here
    __shared__ float subTileA[BLOCK_WIDTH][BLOCK_WIDTH];
    __shared__ float subTileB[BLOCK_WIDTH][BLOCK_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * BLOCK_WIDTH + ty;
    int col = bx * BLOCK_WIDTH + tx;

    float sum = 0.0;
    int tileColumnNum = (BLOCK_WIDTH + numAColumns - 1) / BLOCK_WIDTH;
    for (int i = 0; i < tileColumnNum; i++) {
        if (i * BLOCK_WIDTH + tx < numAColumns && row < numARows) {
            subTileA[ty][tx] = A[row * numAColumns + i * BLOCK_WIDTH + tx];
        } else {
            subTileA[ty][tx] = 0.0;
        }

        if (i * BLOCK_WIDTH + ty < numBRows && col < numBColumns) {
            subTileB[ty][tx] = B[(i * BLOCK_WIDTH + ty) * numBColumns + col];
        } else {
            subTileB[ty][tx] = 0.0;
        }

        __syncthreads();

        for (int j = 0; j < BLOCK_WIDTH; j++) {
            sum += subTileA[ty][j] * subTileB[j][tx];
        }
        __syncthreads();
    }

    if (row < numCRows && col < numCColumns) {
        C[row * numCColumns + col] = sum;
    }
}

int main(int argc, char** argv) {
    wbArg_t args;
    float* hostA;  // The A matrix
    float* hostB;  // The B matrix
    float* hostC;  // The output C matrix
    float* deviceA;
    float* deviceB;
    float* deviceC;
    int numARows;     // number of rows in the matrix A
    int numAColumns;  // number of columns in the matrix A
    int numBRows;     // number of rows in the matrix B
    int numBColumns;  // number of columns in the matrix B
    int numCRows;     // number of rows in the matrix C (you have to set this)
    int numCColumns;  // number of columns in the matrix C (you have to set
                      // this)

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostA =
        (float*)wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
    hostB =
        (float*)wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);

    //@@ Set numCRows and numCColumns
    numCRows = numARows;
    numCColumns = numBColumns;
    //@@ Allocate the hostC matrix
    hostC = (float*)malloc(numCRows * numCColumns * sizeof(float));

    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
    wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
    wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    int sizeA = numARows * numAColumns * sizeof(float);
    int sizeB = numBRows * numBColumns * sizeof(float);
    int sizeC = numCRows * numCColumns * sizeof(float);

    cudaMalloc((void**)&deviceA, sizeA);
    cudaMalloc((void**)&deviceB, sizeB);
    cudaMalloc((void**)&deviceC, sizeC);

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
    cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, sizeB, cudaMemcpyHostToDevice);

    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
    dim3 DimGrid(ceil(numCColumns / float(BLOCK_WIDTH)),
                 ceil(numCRows / float(BLOCK_WIDTH)), 1);
    dim3 DimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
    matrixMultiply<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC, numARows,
                                          numAColumns, numBRows, numBColumns,
                                          numCRows, numCColumns);

    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostC, deviceC, sizeC, cudaMemcpyDeviceToHost);

    // Print out matrix to debug
    // for (int i = 0; i < numCRows; i++)
    // {
    //     for (int j = 0; j < numCColumns; j++)
    //     {
    //         std::cout << hostC[i * numCColumns + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}