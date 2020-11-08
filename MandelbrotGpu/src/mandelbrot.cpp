#include "mandelbrot.h"

#include <algorithm>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuComplex.h>

static const int MAX_BLOCK_SIZE = 1024;
static const int MAX_SQUARE_BLOCK_SIDE = 32;

#define ABS_LIMIT (2.0)

struct MandelbrotHandle
{
    int* gpu_buffer;
    int rows, columns, items_count;
    dim3 grid_sizes, block_sizes;
};

__global__ void mandelbrotKernel(MandelbrotParams params, int rows, int columns, int* buffer);
__device__ inline cuDoubleComplex mapPixel(double pixel_step, double min_real, double min_imag, int i, int j);
static int ceilDivision(int value, int divider);

MandelbrotHandle * initMandelbrotHandle(int rows, int columns)
{
    auto h = new MandelbrotHandle;
    h->rows = rows;
    h->columns = columns;
    h->items_count = rows * columns;
    if (h->items_count > MAX_BLOCK_SIZE) {
        auto block_side = std::min({MAX_SQUARE_BLOCK_SIDE, h->rows, h->columns});
        h->block_sizes = dim3(block_side, block_side);
        h->grid_sizes.x = ceilDivision(h->columns, block_side);
        h->grid_sizes.y = ceilDivision(h->rows, block_side);
    } else {
        h->block_sizes = dim3(h->columns, h->rows);
        h->grid_sizes = dim3(1, 1);
    }
    auto bytes_count = h->items_count * sizeof(int);
    cudaError_t code = cudaMalloc(&h->gpu_buffer, bytes_count);
    if (code != cudaSuccess) {
        delete h;
        return nullptr;
    }
    return h;
}

void fillMatrix(const MandelbrotHandle * handle, const MandelbrotParams* params, int* out_buffer)
{
    mandelbrotKernel<<<handle->grid_sizes, handle->block_sizes>>>
        (*params, handle->rows, handle->columns, handle->gpu_buffer);
    auto bytes_count = handle->items_count * sizeof(int);
    cudaMemcpy(out_buffer, handle->gpu_buffer, bytes_count, cudaMemcpyDeviceToHost);
}

void freeMandelbrotHandle(MandelbrotHandle * handle)
{
    cudaFree(handle->gpu_buffer);
    delete handle;
}

__global__ void mandelbrotKernel(MandelbrotParams params, int rows, int columns, int* buffer)
{
    auto i = blockIdx.y * blockDim.y + threadIdx.y;
    auto j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows || j >= columns) {
        return;
    }

    auto add = mapPixel(params.pixel_step, params.min_real, params.min_imag, j, i);
    cuDoubleComplex z = make_cuDoubleComplex(0.0, 0.0);
    int count = 0;
    while (count < params.max_iteration && cuCabs(z) < ABS_LIMIT) {
        z = cuCadd(cuCmul(z, z), add);
        ++count;
    }
    auto flat_index = i * columns + j;
    buffer[flat_index] = count;
}

__device__ cuDoubleComplex mapPixel(double pixel_step, double min_real, double min_imag, int i, int j)
{
    return make_cuDoubleComplex(
        min_real + i * pixel_step,
        min_imag + j * pixel_step
    );
}

int ceilDivision(int value, int divider)
{
    return int(ceil(double(value) / divider));
}