#ifndef MANDELBROT_CUH
#define MANDELBROT_CUH

#define MANDELBROT_API extern "C" __declspec(dllexport)

struct CountsMatrix
{
    int* buffer;
    int rows;
    int columns;
};

MANDELBROT_API void fillMatrix(double pixel_step, double min_real, double min_imag, int max_iteration, CountsMatrix* matrix);

#endif // MANDELBROT_CUH
