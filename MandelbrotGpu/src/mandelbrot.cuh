#ifndef MANDELBROT_CUH
#define MANDELBROT_CUH

#define MANDELBROT_API extern "C" __declspec(dllexport)

struct CountsMatrix
{
    int* buffer;
    int rows;
    int columns;
};

#endif // MANDELBROT_CUH
