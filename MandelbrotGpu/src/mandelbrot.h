#ifndef MANDELBROT_CUH
#define MANDELBROT_CUH

#define MANDELBROT_API extern "C" __declspec(dllexport)

struct MandelbrotParams
{
    double pixel_step;
    double min_real;
    double min_imag;
    int max_iteration;
};

struct MandelbrotHandle;

using CountType = unsigned char;

MANDELBROT_API MandelbrotHandle* initMandelbrotHandle(int rows, int columns);

MANDELBROT_API void fillMatrix(const MandelbrotHandle * handle, const MandelbrotParams* params, CountType* out_buffer);

MANDELBROT_API void freeMandelbrotHandle(MandelbrotHandle* handle);

#endif // MANDELBROT_CUH
