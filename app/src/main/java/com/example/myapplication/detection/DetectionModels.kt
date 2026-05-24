package com.example.myapplication.detection

/**
 * Normalized (0.0–1.0) point relative to the image dimensions.
 * x=0.0 is left edge, x=1.0 is right edge. Same for y top-to-bottom.
 */
data class NormalizedPoint(val x: Float, val y: Float)

/**
 * A quadrilateral defined by four normalized corner points, plus a confidence score
 * from whichever detector produced it.
 */
data class NormalizedQuad(
    val topLeft:     NormalizedPoint,
    val topRight:    NormalizedPoint,
    val bottomRight: NormalizedPoint,
    val bottomLeft:  NormalizedPoint,
    val confidence:  Float,
    val maskScore:   Float = 0f    // mean sigmoid of mask_logits within quad region
)

/**
 * Records which stage of the hybrid pipeline produced the final quad.
 * Used for logging, debugging, and downstream metadata.
 */
enum class DetectionMethod {
    OPENCV_CONTOUR,  // Classical C++ pipeline succeeded, ONNX not used or low-confidence
    ONNX_REFINED,    // ONNX model result used (confidence ≥ threshold)
    MANUAL_FALLBACK  // Both detectors returned nothing — full-frame quad shown to user
}
