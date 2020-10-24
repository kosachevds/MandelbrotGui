module Mandelbrot.MandelbrotSet

open System.Drawing
open System.Numerics

let private limit = 2.0

let private isMandelbrotPoint iter c =
    // TODO: remade with infinite Sequence
    let rec countIterations z c iter count =
        if Complex.Abs(z) < limit && count < iter then
            countIterations (z * z + c) c iter (count + 1)
        else
            count
    countIterations Complex.Zero c iter 0

let private mapPixel pixelStep minReal minImag x y =
    let real = (float x) * pixelStep + minReal
    let imag = (float y) * pixelStep + minImag
    Complex(real, imag)

let private colorize count =
    let r = (4 * count) % 255
    let g = (6 * count) % 255
    let b = (8 * count) % 255
    Color.FromArgb(r, g, b)

let createImage (image: Bitmap) pixelStep minReal minImag iter =
    let countIterations = isMandelbrotPoint iter
    let mapPixelWithScale = mapPixel pixelStep minReal minImag
    for x = 0 to image.Width - 1 do
        for y = 0 to image.Height - 1 do
            let gridValue = mapPixelWithScale x y
            let count = countIterations gridValue
            if count = iter then
                image.SetPixel(x, y, Color.Black)
            else
                image.SetPixel(x, y, colorize count)
    image