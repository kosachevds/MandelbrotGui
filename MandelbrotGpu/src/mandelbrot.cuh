#ifndef MANDELBROT_CUH
#define MANDELBROT_CUH

#define MANDELBROT_API extern "C" __declspec(dllexport)

struct CountsMatrix
{
    int* buffer;
    int rows;
    int columns;
};

struct MandelbrotParams
{
    double pixel_step;
    double min_real;
    double min_imag;
    int max_iteration;
};

MANDELBROT_API void fillMatrix(const MandelbrotParams* params, CountsMatrix* matrix);

#endif // MANDELBROT_CUH
