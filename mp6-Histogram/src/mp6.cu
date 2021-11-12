// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 512

//@@ insert code here

__global__ void imageFloatToUint8(float *inputImage, uint8_t *outputImage, int width, int height, int channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height * channels) {
        outputImage[idx] = (uint8_t)(255 * inputImage[idx]);
    }
}

__global__ void imageUint8ToFloat(uint8_t *inputImage, float *outputImage, int width, int height, int channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height * channels) {
        outputImage[idx] = (float)(inputImage[idx] / 255.0);
    }
}

__global__ void imageRgbToGrayScale(uint8_t *inputImage, uint8_t *outputImage, int width, int height, int channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height) {
        uint8_t r = inputImage[channels * idx];
        uint8_t g = inputImage[channels * idx + 1];
        uint8_t b = inputImage[channels * idx + 2];
        outputImage[idx] = (uint8_t)(0.21 * r + 0.71 * g + 0.07 * b);
    }
}

__global__ void buildHistogram(uint8_t *grayScaleImage, uint32_t *histogram, int width, int height) {
    __shared__ uint32_t blockHistogram[HISTOGRAM_LENGTH];

    int tx = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + tx;
    if (tx < HISTOGRAM_LENGTH) {
        blockHistogram[threadIdx.x] = 0;
    }
    __syncthreads();

    if (idx < width * height) {
        atomicAdd(&(blockHistogram[grayScaleImage[idx]]), 1);
    }
    __syncthreads();

    if (tx < HISTOGRAM_LENGTH) {
        atomicAdd(&(histogram[tx]), blockHistogram[tx]);
    }
}


__global__ void buildHistogramCDF(uint32_t *histogram, float *cdf, int width, int height) {
    __shared__ float cumulate[HISTOGRAM_LENGTH];

    int tx = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tx;
    float temp;

    if (idx < HISTOGRAM_LENGTH) {
        cumulate[idx] = (float)histogram[idx];
    } else {
        cumulate[idx] = 0.0;
    }

    for (int stride = 1; stride < blockDim.x; stride = stride << 1) {
        __syncthreads();
        if (tx >= stride) {
            temp = cumulate[tx] + cumulate[tx - stride];
        }
        __syncthreads();
        if (tx >= stride) {
            cumulate[tx] = temp;
        }
    }

    __syncthreads();
    if (idx < HISTOGRAM_LENGTH) {
        cdf[idx] = cumulate[idx] / ((float)(width * height));
    }
}


__global__ void imageEqualize(uint8_t *inputImage, uint8_t *outputImage, float *cdf, int width, int height, int channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height * channels) {
        uint8_t val = inputImage[idx];
        float cdfmin = cdf[0];
        float raw = 255.0 * (cdf[val] - cdfmin) / (1.0 - cdfmin);
        outputImage[idx] = (uint8_t)min(float(max(raw, 0.0)), 255.0);
    }
}



int main(int argc, char **argv) {
    wbArg_t args;
    int imageWidth;
    int imageHeight;
    int imageChannels;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float *hostInputImageData;
    float *hostOutputImageData;
    const char *inputImageFile;

    //@@ Insert more code here

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);

    wbTime_start(Generic, "Importing data and creating memory on host");
    inputImage = wbImport(inputImageFile);
    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);
    wbTime_stop(Generic, "Importing data and creating memory on host");

    //@@ insert code here  float* deviceInputImage;


    float *deviceInputFloatImage;
    uint8_t *deviceInputUint8Image;
    uint8_t *deviceGrayScaleUint8Image;
    uint8_t *deviceEqualizeUint8Image;
    float *deviceEqualizeFloatImage;

    uint32_t *deviceHistogram;
    float *deviceCDF;

    size_t floatImageSize = imageWidth * imageHeight * imageChannels * sizeof(float);
    size_t uint8ImageSize = imageWidth * imageHeight * imageChannels * sizeof(uint8_t);
    size_t uint8GrayImageSize = imageWidth * imageHeight * sizeof(uint8_t);
    size_t histogramSize = HISTOGRAM_LENGTH * sizeof(uint32_t);
    size_t cdfSize = HISTOGRAM_LENGTH * sizeof(float);


    wbTime_start(GPU, "Allocating GPU memory.");

    cudaMalloc((void **)&deviceInputFloatImage, floatImageSize);
    cudaMalloc((void **)&deviceInputUint8Image, uint8ImageSize);
    cudaMalloc((void **)&deviceGrayScaleUint8Image, uint8GrayImageSize);
    cudaMalloc((void **)&deviceEqualizeUint8Image, uint8ImageSize);
    cudaMalloc((void **)&deviceEqualizeFloatImage, floatImageSize);

    cudaMalloc((void **)&deviceHistogram, histogramSize);
    cudaMalloc((void **)&deviceCDF, cdfSize);

    wbTime_stop(GPU, "Allocating GPU memory.");



    wbTime_start(GPU, "Clearing histogram and cdf memory.");

    cudaMemset(deviceHistogram, 0, histogramSize);
    cudaMemset(deviceCDF, 0, cdfSize);

    wbTime_stop(GPU, "Clearing histogram and cdf memory.");



    wbTime_start(GPU, "Copying input memory to the GPU.");

    cudaMemcpy(deviceInputFloatImage, hostInputImageData, floatImageSize, cudaMemcpyHostToDevice);

    wbTime_stop(GPU, "Copying input memory to the GPU.");



    dim3 dimGrid;
    dim3 dimBlock;

    wbTime_start(Compute, "Performing CUDA imageFloatToUint8 computation");

    dimGrid = dim3(ceil((imageWidth * imageHeight * imageChannels) / float(BLOCK_SIZE)), 1, 1);
    dimBlock = dim3(BLOCK_SIZE, 1, 1);
    imageFloatToUint8<<<dimGrid, dimBlock>>>(deviceInputFloatImage, deviceInputUint8Image, imageWidth, imageHeight, imageChannels);
    cudaDeviceSynchronize();

    wbTime_stop(Compute, "Performing CUDA imageFloatToUint8 computation");



    wbTime_start(Compute, "Performing CUDA imageRgbToGrayScale computation");

    dimGrid = dim3(ceil((imageWidth * imageHeight) / float(BLOCK_SIZE)), 1, 1);
    dimBlock = dim3(BLOCK_SIZE, 1, 1);
    imageRgbToGrayScale<<<dimGrid, dimBlock>>>(deviceInputUint8Image, deviceGrayScaleUint8Image, imageWidth, imageHeight, imageChannels);
    cudaDeviceSynchronize();

    wbTime_stop(Compute, "Performing CUDA imageRgbToGrayScale computation");



    wbTime_start(Compute, "Performing CUDA buildHistogram computation");

    dimGrid = dim3(ceil((imageWidth * imageHeight) / float(BLOCK_SIZE)), 1, 1);
    dimBlock = dim3(BLOCK_SIZE, 1, 1);
    buildHistogram<<<dimGrid, dimBlock>>>(deviceGrayScaleUint8Image, deviceHistogram, imageWidth, imageHeight);
    cudaDeviceSynchronize();

    wbTime_stop(Compute, "Performing CUDA buildHistogram computation");



    wbTime_start(Compute, "Performing CUDA buildHistogramCDF computation");

    dimGrid = dim3(1, 1, 1);
    dimBlock = dim3(HISTOGRAM_LENGTH, 1, 1);
    buildHistogramCDF<<<dimGrid, dimBlock>>>(deviceHistogram, deviceCDF, imageWidth, imageHeight);
    cudaDeviceSynchronize();

    wbTime_stop(Compute, "Performing CUDA buildHistogramCDF computation");



    wbTime_start(Compute, "Performing CUDA imageEqualize computation");

    dimGrid = dim3(ceil((imageWidth * imageHeight * imageChannels) / float(BLOCK_SIZE)), 1, 1);
    dimBlock = dim3(BLOCK_SIZE, 1, 1);
    imageEqualize<<<dimGrid, dimBlock>>>(deviceInputUint8Image, deviceEqualizeUint8Image, deviceCDF, imageWidth, imageHeight, imageChannels);
    cudaDeviceSynchronize();

    wbTime_stop(Compute, "Performing CUDA imageEqualize computation");



    wbTime_start(Compute, "Performing CUDA imageUint8ToFloat computation");

    dimGrid = dim3(ceil((imageWidth * imageHeight * imageChannels) / float(BLOCK_SIZE)), 1, 1);
    dimBlock = dim3(BLOCK_SIZE, 1, 1);
    imageUint8ToFloat<<<dimGrid, dimBlock>>>(deviceEqualizeUint8Image, deviceEqualizeFloatImage, imageWidth, imageHeight, imageChannels);
    cudaDeviceSynchronize();

    wbTime_stop(Compute, "Performing CUDA imageUint8ToFloat computation");



    wbTime_start(Copy, "Copying output memory to the CPU");

    cudaMemcpy(hostOutputImageData, deviceEqualizeFloatImage, floatImageSize, cudaMemcpyDeviceToHost);

    wbTime_stop(Copy, "Copying output memory to the CPU");



    wbTime_start(GPU, "Freeing GPU Memory");

    cudaFree(deviceInputFloatImage);
    cudaFree(deviceInputUint8Image);
    cudaFree(deviceGrayScaleUint8Image);
    cudaFree(deviceEqualizeUint8Image);
    cudaFree(deviceEqualizeFloatImage);
    cudaFree(deviceHistogram);
    cudaFree(deviceCDF);

    wbTime_stop(GPU, "Freeing GPU Memory");




    wbSolution(args, outputImage);

    //@@ insert code here
    free(hostInputImageData);
    free(hostOutputImageData);

    return 0;
}
