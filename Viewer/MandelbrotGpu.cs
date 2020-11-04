using System;
using System.Runtime.InteropServices;

namespace Mandelbrot
{
    [StructLayout(LayoutKind.Sequential)]
    struct MandelbrotParams
    {
        public double PixelStep;
        public double MinReal;
        public double MinImag;
        public int MaxIteration;
    }

    class MandelbrotGpu
    {
        private const string LibraryName = "MandelbrotGpu.dll";

        [DllImport(LibraryName)]
        private static extern IntPtr initMandelbrotHandle(int rows, int columns);

        [DllImport(LibraryName)]
        private static extern void fillMatrix(ref MandelbrotParams mparams, IntPtr handle, IntPtr buffer);

        [DllImport(LibraryName)]
        private static extern void freeMandelbrotHandle(IntPtr handle);
    }
}
