package com.example.myapplication.detection

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint

/**
 * Records the letterboxing parameters applied to fit an arbitrary bitmap
 * inside the [targetSize]×[targetSize] canvas. These values are required
 * to map model output coordinates back to the original image space.
 *
 * @param scale    The uniform scale applied to the original bitmap.
 * @param padX     Left padding in the [targetSize] canvas (pixels).
 * @param padY     Top padding in the [targetSize] canvas (pixels).
 */
data class PadInfo(val scale: Float, val padX: Int, val padY: Int)

class LetterboxPreprocessor {

    /**
     * Converts a Bitmap to a [targetSize]×[targetSize] letterboxed float array
     * in CHW format (NCHW-compatible for [1, 3, targetSize, targetSize] tensors).
     *
     * Preprocessing steps:
     * 1. Scale image to fit inside targetSize × targetSize (preserves aspect ratio)
     * 2. Centre-pad with black pixels to exactly targetSize × targetSize
     * 3. Normalize each channel: (pixel/255 - mean) / std  (ImageNet statistics)
     * 4. Lay channels in R-plane first, then G, then B (CHW order)
     *
     * @param bitmap     Input bitmap (any size, any aspect ratio).
     * @param targetSize Side length of the square output canvas. Default 256.
     * @return Pair of the flat CHW float array and the [PadInfo] for denormalization.
     */
    fun preprocess(bitmap: Bitmap, targetSize: Int = 256): Pair<FloatArray, PadInfo> {
        // Step 1 — compute uniform scale so the longer side fits targetSize
        val scale = targetSize.toFloat() / maxOf(bitmap.width, bitmap.height)
        val scaledW = (bitmap.width  * scale).toInt().coerceAtLeast(1)
        val scaledH = (bitmap.height * scale).toInt().coerceAtLeast(1)

        // Padding required on each axis to centre the scaled image
        val padX = (targetSize - scaledW) / 2
        val padY = (targetSize - scaledH) / 2

        // Step 2 — draw scaled bitmap centred on a black targetSize×targetSize canvas
        val canvas256 = Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888)
        val c = Canvas(canvas256)
        c.drawColor(Color.BLACK)
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, scaledW, scaledH, /*filter=*/true)
        c.drawBitmap(scaledBitmap, padX.toFloat(), padY.toFloat(), null)
        scaledBitmap.recycle()

        // Step 3 & 4 — convert to CHW float array with ImageNet normalization
        // mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
        val pixels = IntArray(targetSize * targetSize)
        canvas256.getPixels(pixels, 0, targetSize, 0, 0, targetSize, targetSize)
        canvas256.recycle()

        val planeSize = targetSize * targetSize
        val floatArray = FloatArray(3 * planeSize)

        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std  = floatArrayOf(0.229f, 0.224f, 0.225f)

        for (i in pixels.indices) {
            val pixel = pixels[i]
            val r = ((pixel shr 16) and 0xFF) / 255f
            val g = ((pixel shr  8) and 0xFF) / 255f
            val b = ( pixel         and 0xFF) / 255f
            // CHW layout: R-plane [0..planeSize), G-plane [planeSize..2*planeSize), B-plane [2*planeSize..3*planeSize)
            floatArray[0 * planeSize + i] = (r - mean[0]) / std[0]
            floatArray[1 * planeSize + i] = (g - mean[1]) / std[1]
            floatArray[2 * planeSize + i] = (b - mean[2]) / std[2]
        }

        return Pair(floatArray, PadInfo(scale, padX, padY))
    }
}
