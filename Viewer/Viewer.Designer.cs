namespace Mandelbrot
{
    partial class Viewer
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.setView = new System.Windows.Forms.PictureBox();
            this.btnRight = new System.Windows.Forms.Button();
            this.btnUp = new System.Windows.Forms.Button();
            this.btnDown = new System.Windows.Forms.Button();
            this.btnLeft = new System.Windows.Forms.Button();
            this.btnPlus = new System.Windows.Forms.Button();
            this.btnMinus = new System.Windows.Forms.Button();
            this.tbMaxIterations = new System.Windows.Forms.TextBox();
            this.lblMaxIterations = new System.Windows.Forms.Label();
            this.btnMaxIterations = new System.Windows.Forms.Button();
            ((System.ComponentModel.ISupportInitialize)(this.setView)).BeginInit();
            this.SuspendLayout();
            // 
            // setView
            // 
            this.setView.Location = new System.Drawing.Point(0, 0);
            this.setView.Name = "setView";
            this.setView.Size = new System.Drawing.Size(400, 450);
            this.setView.TabIndex = 0;
            this.setView.TabStop = false;
            // 
            // btnRight
            // 
            this.btnRight.Location = new System.Drawing.Point(201, 489);
            this.btnRight.Name = "btnRight";
            this.btnRight.Size = new System.Drawing.Size(75, 23);
            this.btnRight.TabIndex = 4;
            this.btnRight.Text = "Right";
            this.btnRight.UseVisualStyleBackColor = true;
            this.btnRight.Click += new System.EventHandler(this.btnRight_Click);
            // 
            // btnUp
            // 
            this.btnUp.Location = new System.Drawing.Point(161, 460);
            this.btnUp.Name = "btnUp";
            this.btnUp.Size = new System.Drawing.Size(75, 23);
            this.btnUp.TabIndex = 1;
            this.btnUp.Text = "Up";
            this.btnUp.UseVisualStyleBackColor = true;
            this.btnUp.Click += new System.EventHandler(this.btnUp_Click);
            // 
            // btnDown
            // 
            this.btnDown.Location = new System.Drawing.Point(161, 518);
            this.btnDown.Name = "btnDown";
            this.btnDown.Size = new System.Drawing.Size(75, 23);
            this.btnDown.TabIndex = 2;
            this.btnDown.Text = "Down";
            this.btnDown.UseVisualStyleBackColor = true;
            this.btnDown.Click += new System.EventHandler(this.btnDown_Click);
            // 
            // btnLeft
            // 
            this.btnLeft.Location = new System.Drawing.Point(120, 489);
            this.btnLeft.Name = "btnLeft";
            this.btnLeft.Size = new System.Drawing.Size(75, 23);
            this.btnLeft.TabIndex = 3;
            this.btnLeft.Text = "Left";
            this.btnLeft.UseVisualStyleBackColor = true;
            this.btnLeft.Click += new System.EventHandler(this.btnLeft_Click);
            // 
            // btnPlus
            // 
            this.btnPlus.Location = new System.Drawing.Point(12, 460);
            this.btnPlus.Name = "btnPlus";
            this.btnPlus.Size = new System.Drawing.Size(75, 23);
            this.btnPlus.TabIndex = 5;
            this.btnPlus.Text = "+";
            this.btnPlus.UseVisualStyleBackColor = true;
            this.btnPlus.Click += new System.EventHandler(this.btnPlus_Click);
            // 
            // btnMinus
            // 
            this.btnMinus.Location = new System.Drawing.Point(12, 489);
            this.btnMinus.Name = "btnMinus";
            this.btnMinus.Size = new System.Drawing.Size(75, 23);
            this.btnMinus.TabIndex = 6;
            this.btnMinus.Text = "–";
            this.btnMinus.UseVisualStyleBackColor = true;
            this.btnMinus.Click += new System.EventHandler(this.btnMinus_Click);
            // 
            // tbMaxIterations
            // 
            this.tbMaxIterations.Location = new System.Drawing.Point(303, 489);
            this.tbMaxIterations.Name = "tbMaxIterations";
            this.tbMaxIterations.Size = new System.Drawing.Size(85, 20);
            this.tbMaxIterations.TabIndex = 7;
            this.tbMaxIterations.TextChanged += new System.EventHandler(this.tbMaxIterations_TextChanged);
            // 
            // lblMaxIterations
            // 
            this.lblMaxIterations.AutoSize = true;
            this.lblMaxIterations.Location = new System.Drawing.Point(300, 470);
            this.lblMaxIterations.Name = "lblMaxIterations";
            this.lblMaxIterations.Size = new System.Drawing.Size(73, 13);
            this.lblMaxIterations.TabIndex = 8;
            this.lblMaxIterations.Text = "Max Iterations";
            // 
            // btnMaxIterations
            // 
            this.btnMaxIterations.Location = new System.Drawing.Point(303, 518);
            this.btnMaxIterations.Name = "btnMaxIterations";
            this.btnMaxIterations.Size = new System.Drawing.Size(75, 23);
            this.btnMaxIterations.TabIndex = 9;
            this.btnMaxIterations.Text = "OK";
            this.btnMaxIterations.UseVisualStyleBackColor = true;
            this.btnMaxIterations.Click += new System.EventHandler(this.btnMaxIterations_Click);
            // 
            // Viewer
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(400, 554);
            this.Controls.Add(this.btnMaxIterations);
            this.Controls.Add(this.lblMaxIterations);
            this.Controls.Add(this.tbMaxIterations);
            this.Controls.Add(this.btnMinus);
            this.Controls.Add(this.btnPlus);
            this.Controls.Add(this.btnRight);
            this.Controls.Add(this.btnLeft);
            this.Controls.Add(this.btnDown);
            this.Controls.Add(this.btnUp);
            this.Controls.Add(this.setView);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "Viewer";
            this.ShowIcon = false;
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "Mandelbrot";
            ((System.ComponentModel.ISupportInitialize)(this.setView)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.PictureBox setView;
        private System.Windows.Forms.Button btnRight;
        private System.Windows.Forms.Button btnUp;
        private System.Windows.Forms.Button btnDown;
        private System.Windows.Forms.Button btnLeft;
        private System.Windows.Forms.Button btnPlus;
        private System.Windows.Forms.Button btnMinus;
        private System.Windows.Forms.TextBox tbMaxIterations;
        private System.Windows.Forms.Label lblMaxIterations;
        private System.Windows.Forms.Button btnMaxIterations;
    }
}

