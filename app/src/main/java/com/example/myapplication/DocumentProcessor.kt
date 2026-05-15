package com.example.myapplication

import android.graphics.Bitmap

object DocumentProcessor {

    init {
        System.loadLibrary("document_detector")
    }

    /**
     * Scans the given bitmap natively and returns the bounding points as a JSON array string.
     */
    external fun nativeScanJSON(
        srcBitmap: Bitmap,
        shrunkImageHeight: Int,
        imageRotation: Int,
        scale: Double,
        options: String
    ): String?

    /**
     * Applies perspective transformation (cropping) based on the boundary points.
     * @param srcBitmap The original image.
     * @param points A JSON string containing the array of 4 Points representing vertices.
     * @param transforms Optional JSON string with transform settings (can be empty).
     * @param outBitmap An empty Bitmap of the desired cropped dimensions where the result is written.
     */
    external fun nativeCrop(
        srcBitmap: Bitmap,
        points: String,
        transforms: String,
        outBitmap: Bitmap
    )
}
