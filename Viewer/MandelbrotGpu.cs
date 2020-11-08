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

        private IntPtr _handle;

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr initMandelbrotHandle(int rows, int columns);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern void fillMatrix(IntPtr handle, ref MandelbrotParams mparams, byte[] buffer);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern void freeMandelbrotHandle(IntPtr handle);

        public MandelbrotGpu(int rows, int columns)
        {
            this._handle = initMandelbrotHandle(rows, columns);
        }

        public void FillMatrix(ref MandelbrotParams mparams, byte[] matrix)
        {
            fillMatrix(this._handle, ref mparams, matrix);
        }

        ~MandelbrotGpu()
        {
            freeMandelbrotHandle(this._handle);
        }
    }
}
