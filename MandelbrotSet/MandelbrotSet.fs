module Mandelbrot.MandelbrotSet

open System.Numerics

let private limit = 2.0

let private isMandelbrotPoint maxCount add =
    // TODO: remade with infinite Sequence
    let rec countIterations z count =
        if Complex.Abs(z) < limit && count < maxCount then
            countIterations (z * z + add) (count + 1)
        else
            count
    countIterations Complex.Zero 0

let private mapPixel pixelStep minReal minImag x y =
    let real = (float x) * pixelStep + minReal
    let imag = (float y) * pixelStep + minImag
    Complex(real, imag)

let fillMatrix pixelStep minReal minImag maxIterations (matrix: int[,]) =
    let countIterations = isMandelbrotPoint maxIterations
    let mapPixelWithScale = mapPixel pixelStep minReal minImag
    for x = 0 to matrix.GetLength(0) - 1 do
        let mapPixelWithScale = mapPixelWithScale x
        seq {0 .. matrix.GetLength(1) - 1}
        |> Seq.iter (fun y ->
            matrix.[x, y] <- (y |> mapPixelWithScale |> countIterations)
        )
    matrix
