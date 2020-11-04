#include "mandelbrot.h"

#include <cuda_runtime.h>

struct MandelbrotHandle
{
    int* gpu_buffer;
    MandelbrotParams params;
};
