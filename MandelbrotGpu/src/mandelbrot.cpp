#include "mandelbrot.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuComplex.h>

#define ABS_LIMIT (2.0)

struct MandelbrotHandle
{
    MandelbrotParams params;
    int* gpu_buffer;
    int rows, columns;
};

__global__ void mandelbrotKernel(MandelbrotParams params, int rows, int columns, int* buffer);
__device__ inline cuDoubleComplex mapPixel(double pixel_step, double min_real, double min_imag, int i, int j);

MandelbrotHandle * initMandelbrotHandle(const MandelbrotParams * params, int rows, int columns)
{
    auto h = new MandelbrotHandle;
    h->params = *params;
    auto bytes_count = rows * columns * sizeof(int);
    cudaError_t code = cudaMalloc(&h->gpu_buffer, bytes_count);
    if (code != cudaSuccess) {
        delete h;
        return nullptr;
    }
    return h;
}

void fillMatrix(const MandelbrotHandle * handle, int* out_buffer)
{
    // TODO: calculate block_size etc
    mandelbrotKernel<<<matrix->rows, matrix->columns>>>
        (handle->params, handle->rows, handle->columns, handle->gpu_buffer);
    auto bytes_count = handle->rows * handle->columns * sizeof(int);
    cudaMemcpy(out_buffer, handle->gpu_buffer, bytes_count, cudaMemcpyDeviceToHost);
}

void freeMandelbrotHandle(MandelbrotHandle * handle)
{
    cudaFree(handle->gpu_buffer);
    delete handle;
}

void mandelbrotKernel(MandelbrotParams params, int rows, int columns, int* buffer)
{
    auto i = blockIdx.y * blockDim.y + threadIdx.y;
    auto j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows || j >= columns) {
        return;
    }

    auto add = mapPixel(params.pixel_step, params.min_real, params.min_imag, i, j);
    cuDoubleComplex z = make_cuDoubleComplex(0.0, 0.0);
    int count = 0;
    while (count < params.max_iteration && cuCabs(z) < ABS_LIMIT) {
        z = cuCmul(cuCmul(z, z), add);
        ++count;
    }
    auto flat_index = i * columns + j;
    buffer[flat_index] = count;
}

cuDoubleComplex mapPixel(double pixel_step, double min_real, double min_imag, int i, int j)
{
    return make_cuDoubleComplex(
        min_real + i * pixel_step,
        min_imag + j * pixel_step
    );
}
