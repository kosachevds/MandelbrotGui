using System;
using System.Drawing;
using System.Windows.Forms;

namespace Mandelbrot
{
    public partial class Viewer : Form
    {
        private const double ScaleFactor = 0.2;
        private const int MaxIterations = 50;
        private double scaleStep;
        private readonly Bitmap bitmap;
        private readonly int[] countsMatrix;  // TODO: with sbyte
        private readonly MandelbrotGpu mandelbrotGpu;
        private MandelbrotParams mbParams;

        private int AreaWidth => this.setView.Width;
        private int AreaHeight => this.setView.Height;
        private double PixelStep
        {
            get => this.mbParams.PixelStep;
            set => this.mbParams.PixelStep = value;
        }
        private double MinReal
        {
            get => this.mbParams.MinReal;
            set => this.mbParams.MinReal = value;
        }
        private double MinImag
        {
            get => this.mbParams.MinImag;
            set => this.mbParams.MinImag = value;
        }

        public Viewer()
        {
            // TODO: check MaxValue for matrix item
            InitializeComponent();

            this.mbParams = new MandelbrotParams
            {
                MinReal = -1.5,
                MinImag = -1.2,
                PixelStep = 1 / 175.0,
                MaxIteration = MaxIterations
            };
            this.scaleStep = ScaleFactor * this.AreaWidth * this.mbParams.PixelStep;
            this.countsMatrix = new int[this.AreaWidth * this.AreaHeight];
            this.bitmap = new Bitmap(this.AreaWidth, this.AreaHeight);
            this.mandelbrotGpu = new MandelbrotGpu(this.AreaHeight, this.AreaWidth);
            this.setView.Image = this.bitmap;
            DrawSet();
        }

        private void DrawSet()
        {
            mandelbrotGpu.FillMatrix(ref this.mbParams, this.countsMatrix);
            CountsToBitmap(this.countsMatrix, this.bitmap);
            this.setView.Refresh();
        }

        private void CountsToBitmap(int[] countsMatrix, Bitmap bitmap)
        {
            for (int x = 0; x < bitmap.Width; ++x)
            {
                for (int y = 0; y < bitmap.Height; ++y)
                {
                    var count = countsMatrix[y * bitmap.Width + x];
                    var color = GetColor(count);
                    bitmap.SetPixel(x, y, color);
                }
            }
        }

        private Color GetColor(int count)
        {
            if (count >= MaxIterations)
            {
                return Color.Black;
            }
            var r = (4 * count) % 255;
            var g = (6 * count) % 255;
            var b = (8 * count) % 255;
            return Color.FromArgb(r, g, b);
        }

        private void ZoomIn()
        {
            this.scaleStep = this.AreaWidth * this.PixelStep * ScaleFactor;
            Scale(-this.scaleStep);
        }

        private void ZoomOut()
        {
            Scale(this.scaleStep);
            this.scaleStep = this.AreaWidth * this.PixelStep * ScaleFactor /
                (1 - 2 * ScaleFactor);
        }

        private void Scale(double value)
        {
            this.MinReal -= value;
            this.MinImag -= value;
            this.PixelStep += 2 * value / this.AreaWidth;
        }

        private void MoveUp()
        {
            this.MinImag -= this.scaleStep;
        }

        private void MoveDown()
        {
            this.MinImag += this.scaleStep;
        }

        private void MoveLeft()
        {
            this.MinReal -= this.scaleStep;
        }

        private void MoveRight()
        {
            this.MinReal += this.scaleStep;
        }

        private void RedrawWith(Action action)
        {
            action();
            DrawSet();
        }

        private void btnPlus_Click(object sender, EventArgs e)
        {
            RedrawWith(ZoomIn);
        }

        private void btnUp_Click(object sender, EventArgs e)
        {
            RedrawWith(MoveUp);
        }

        private void btnMinus_Click(object sender, EventArgs e)
        {
            RedrawWith(ZoomOut);
        }

        private void btnLeft_Click(object sender, EventArgs e)
        {
            RedrawWith(MoveLeft);
        }

        private void btnRight_Click(object sender, EventArgs e)
        {
            RedrawWith(MoveRight);
        }

        private void btnDown_Click(object sender, EventArgs e)
        {
            RedrawWith(MoveDown);
        }
    }
}
