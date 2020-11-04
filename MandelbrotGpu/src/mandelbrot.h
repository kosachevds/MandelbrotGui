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

struct MandelbrotHandle;

MANDELBROT_API MandelbrotHandle* initMandelbrotHandle(const MandelbrotParams* params);

MANDELBROT_API void fillMatrix(MandelbrotHandle* handle, CountsMatrix* matrix);

MANDELBROT_API void freeMandelbrotHandle(MandelbrotHandle* handle);

#endif // MANDELBROT_CUH
