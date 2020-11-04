#include "mandelbrot.h"

#include <cuda_runtime.h>

struct MandelbrotHandle
{
    int* gpu_buffer;
    MandelbrotParams params;
};

__global__ void mandelbrotKernel(MandelbrotParams params, CountsMatrix matrix);

MandelbrotHandle * initMandelbrotHandle(const MandelbrotParams * params, int bytes_count)
{
    auto h = new MandelbrotHandle;
    h->params = *params;
    cudaError_t code = cudaMalloc(&h->gpu_buffer, bytes_count);
    if (code != cudaSuccess) {
        delete h;
        return nullptr;
    }
    return h;
}

void fillMatrix(MandelbrotHandle * handle, CountsMatrix * matrix)
{
    // TODO: calculate block_size etc
    mandelbrotKernel<<<matrix->rows, matrix->columns>>>(handle->params, *matrix);
    auto bytes_count = matrix->rows * matrix->columns * sizeof(int);
    cudaMemcpy(matrix->buffer, handle->gpu_buffer, bytes_count, cudaMemcpyDeviceToHost);
}

void freeMandelbrotHandle(MandelbrotHandle * handle)
{
    cudaFree(handle->gpu_buffer);
    delete handle;
}

