using System;
using System.Drawing;
using System.Windows.Forms;

namespace MandelbrotGui
{
    public partial class Viewer : Form
    {
        private const double ScaleFactor = 0.2;
        private double scaleStep;
        private int    iterations;
        private double minReal;
        private double minImag;
        private double pixelStep;
        private readonly Bitmap bitmap;

        private int AreaWidth => this.setView.Width;
        private int AreaHeight => this.setView.Height;

        public Viewer()
        {
            InitializeComponent();
            this.iterations = 50;
            this.minReal = -1.5;
            this.minImag = -1.2;
            this.pixelStep = 1 / 175.0;
            this.scaleStep = ScaleFactor * this.AreaWidth * this.pixelStep;
            this.bitmap = new Bitmap(this.AreaWidth, this.AreaHeight);
            DrawSet();
        }

        private void DrawSet()
        {
            this.setView.Image =
                Mandelbrot.MandelbrotSet.createImage(bitmap, pixelStep, minReal, minImag, iterations);
        }

        private void ZoomIn()
        {
            this.scaleStep = this.AreaWidth * this.pixelStep * ScaleFactor;
            Scale(-scaleStep);
        }

        private void ZoomOut()
        {
            Scale(scaleStep);
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
            this.minImag -= scaleStep;
        }

        private void MoveDown()
        {
            this.minImag += scaleStep;
        }

        private void MoveLeft()
        {
            this.minReal -= scaleStep;
        }

        private void MoveRight()
        {
            this.minReal += scaleStep;
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
