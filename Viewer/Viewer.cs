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
        private double minReal;
        private double minImag;
        private double pixelStep;
        private readonly Bitmap bitmap;
        private readonly int[,] countsMatrix;  // TODO: with sbyte

        private int AreaWidth => this.setView.Width;
        private int AreaHeight => this.setView.Height;

        public Viewer()
        {
            // TODO: check MaxValue for matrix item
            InitializeComponent();

            this.minReal = -1.5;
            this.minImag = -1.2;
            this.pixelStep = 1 / 175.0;
            this.scaleStep = ScaleFactor * this.AreaWidth * this.pixelStep;
            this.countsMatrix = new int[this.AreaWidth, this.AreaHeight];
            this.bitmap = new Bitmap(this.AreaWidth, this.AreaHeight);
            this.setView.Image = this.bitmap;
            DrawSet();
        }

        private void DrawSet()
        {
            MandelbrotSet.fillMatrix(this.pixelStep, this.minReal, this.minImag, MaxIterations, this.countsMatrix);
            CountsToBitmap(this.countsMatrix, this.bitmap);
            this.setView.Refresh();
        }

        private void CountsToBitmap(int[,] countsMatrix, Bitmap bitmap)
        {
            for (int x = 0; x < bitmap.Width; ++x)
            {
                for (int y = 0; y < bitmap.Height; ++y)
                {
                    var count = countsMatrix[x, y];
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
            this.scaleStep = this.AreaWidth * this.pixelStep * ScaleFactor;
            Scale(-this.scaleStep);
        }

        private void ZoomOut()
        {
            Scale(this.scaleStep);
            this.scaleStep = this.AreaWidth * this.pixelStep * ScaleFactor /
                (1 - 2 * ScaleFactor);
        }

        private void Scale(double value)
        {
            this.minReal -= value;
            this.minImag -= value;
            this.pixelStep += 2 * value / this.AreaWidth;
        }

        private void MoveUp()
        {
            this.minImag -= this.scaleStep;
        }

        private void MoveDown()
        {
            this.minImag += this.scaleStep;
        }

        private void MoveLeft()
        {
            this.minReal -= this.scaleStep;
        }

        private void MoveRight()
        {
            this.minReal += this.scaleStep;
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
