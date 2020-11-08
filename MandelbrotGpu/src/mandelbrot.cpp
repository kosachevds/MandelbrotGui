#include "mandelbrot.h"

#include <algorithm>
#include <limits>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuComplex.h>

static const int MAX_BLOCK_SIZE = 1024;
static const int MAX_SQUARE_BLOCK_SIDE = 32;
constexpr auto MAX_COUNT = std::numeric_limits<CountType>::max();

#define ABS_LIMIT (2.0)

struct MandelbrotHandle
{
    CountType* gpu_buffer;
    int rows, columns, items_count;
    dim3 grid_sizes, block_sizes;

    MandelbrotHandle(int rows, int columns);
    ~MandelbrotHandle();
};

__global__ void mandelbrotKernel(MandelbrotParams params, int rows, int columns, CountType* buffer);
__device__ inline cuDoubleComplex mapPixel(double pixel_step, double min_real, double min_imag, int x, int y);
static int ceilDivision(int value, int divider);

MandelbrotHandle * initMandelbrotHandle(int rows, int columns)
{
    auto h = new MandelbrotHandle(rows, columns);
    auto bytes_count = h->items_count * sizeof(CountType);
    cudaError_t code = cudaMalloc(&h->gpu_buffer, bytes_count);
    if (code != cudaSuccess) {
        delete h;
        return nullptr;
    }
    return h;
}

void fillMatrix(const MandelbrotHandle * handle, const MandelbrotParams* params, CountType* out_buffer)
{
    mandelbrotKernel<<<handle->grid_sizes, handle->block_sizes>>>
        (*params, handle->rows, handle->columns, handle->gpu_buffer);
    auto bytes_count = handle->items_count * sizeof(CountType);
    cudaMemcpy(out_buffer, handle->gpu_buffer, bytes_count, cudaMemcpyDeviceToHost);
}

void freeMandelbrotHandle(MandelbrotHandle * handle)
{
    delete handle;
}

__global__ void mandelbrotKernel(MandelbrotParams params, int rows, int columns, CountType* buffer)
{
    auto y = blockIdx.y * blockDim.y + threadIdx.y;
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= rows || x >= columns) {
        return;
    }
    auto add = mapPixel(params.pixel_step, params.min_real, params.min_imag, x, y);
    cuDoubleComplex z = make_cuDoubleComplex(0.0, 0.0);
    int count = 0;
    while (count < params.max_iteration && count < MAX_COUNT && cuCabs(z) < ABS_LIMIT) {
        z = cuCadd(cuCmul(z, z), add);
        ++count;
    }
    auto flat_index = y * columns + x;
    buffer[flat_index] = count;
}

__device__ cuDoubleComplex mapPixel(double pixel_step, double min_real, double min_imag, int x, int y)
{
    return make_cuDoubleComplex(
        min_real + x * pixel_step,
        min_imag + y * pixel_step
    );
}

int ceilDivision(int value, int divider)
{
    return int(ceil(double(value) / divider));
}

MandelbrotHandle::MandelbrotHandle(int rows, int columns)
    : rows(rows), columns(columns), items_count(rows * columns)
{
    if (this->items_count > MAX_BLOCK_SIZE) {
        auto block_side = std::min({MAX_SQUARE_BLOCK_SIDE, this->rows, this->columns});
        this->block_sizes = dim3(block_side, block_side);
        this->grid_sizes.x = ceilDivision(this->columns, block_side);
        this->grid_sizes.y = ceilDivision(this->rows, block_side);
    } else {
        this->block_sizes = dim3(this->columns, this->rows);
        this->grid_sizes = dim3(1, 1);
    }
}

MandelbrotHandle::~MandelbrotHandle()
{
    cudaFree(this->gpu_buffer);
}
