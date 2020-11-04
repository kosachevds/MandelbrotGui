#include "mandelbrot.h"

#include <cuda_runtime.h>

struct MandelbrotHandle
{
    int* gpu_buffer;
    MandelbrotParams params;
};

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

void freeMandelbrotHandle(MandelbrotHandle * handle)
{
    cudaFree(handle->gpu_buffer);
    delete handle;
}
