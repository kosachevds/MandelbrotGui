using System;
using System.Runtime.InteropServices;

namespace Mandelbrot
{
    class MandelbrotGpu
    {
        private const string LibraryName = "MandelbrotGpu.dll";

        [DllImport(LibraryName)]
        private static extern void fillMatrix(IntPtr handle, IntPtr buffer);
    }
}
