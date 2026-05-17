package com.example.myapplication.detection

import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

private const val TAG = "HybridCoordinator"

/**
 * Decision logic that combines the OpenCV classical result (from the C++ pipeline,
 * called upstream via [DocumentProcessor.nativeScanJSON]) with the ONNX result
 * (from [DocQuadNetDetector]) and returns the best quad with its [DetectionMethod].
 *
 * Decision priority:
 *  1. ONNX result, if confidence ≥ [onnxConfidenceThreshold]  → ONNX_REFINED
 *  2. OpenCV result, if provided                              → OPENCV_CONTOUR
 *  3. Full-frame quad with 5% margin                         → MANUAL_FALLBACK
 *
 * Usage (capture path only — never on live viewfinder frames):
 *   val result = coordinator.detect(rotatedBitmap, opencvRelativePoints)
 *
 * @param onnxDetector The ONNX inference engine (session already initialized).
 * @param onnxConfidenceThreshold Minimum ONNX confidence to trust the result (default 0.65).
 */
class HybridDetectionCoordinator(
    private val onnxDetector: DocQuadNetDetector,
    private val onnxConfidenceThreshold: Float = 0.65f
) {
    /**
     * Runs the full hybrid detection flow on a captured (already-rotated) bitmap.
     *
     * @param bitmap         The rotated, full-resolution captured bitmap.
     * @param opencvPoints   Normalized (0.0–1.0) quad from the OpenCV path, or null if not detected.
     * @return Pair of the best [NormalizedQuad] and the [DetectionMethod] that produced it.
     */
    suspend fun detect(
        bitmap: Bitmap,
        opencvPoints: List<com.example.myapplication.PointF>?
    ): Pair<NormalizedQuad, DetectionMethod> = withContext(Dispatchers.Default) {

        // --- Step 1: ONNX inference ---
        val onnxQuad = try {
            onnxDetector.detect(bitmap)
        } catch (e: Exception) {
            Log.e(TAG, "ONNX inference failed: ${e.message}")
            null
        }

        // --- Step 2: Decision logic ---
        when {
            onnxQuad != null && onnxQuad.confidence >= onnxConfidenceThreshold -> {
                Log.d(TAG, "ONNX_REFINED (confidence=${onnxQuad.confidence})")
                Pair(onnxQuad, DetectionMethod.ONNX_REFINED)
            }

            opencvPoints != null && opencvPoints.size == 4 -> {
                Log.d(TAG, "OPENCV_CONTOUR (ONNX confidence=${onnxQuad?.confidence ?: "n/a"})")
                val quad = NormalizedQuad(
                    topLeft     = NormalizedPoint(opencvPoints[0].x, opencvPoints[0].y),
                    topRight    = NormalizedPoint(opencvPoints[1].x, opencvPoints[1].y),
                    bottomRight = NormalizedPoint(opencvPoints[2].x, opencvPoints[2].y),
                    bottomLeft  = NormalizedPoint(opencvPoints[3].x, opencvPoints[3].y),
                    confidence  = 0f
                )
                Pair(quad, DetectionMethod.OPENCV_CONTOUR)
            }

            else -> {
                Log.d(TAG, "MANUAL_FALLBACK (both detectors returned nothing)")
                Pair(fullFrameQuad(), DetectionMethod.MANUAL_FALLBACK)
            }
        }
    }

    /** Full-frame quad with a 5% inset margin on each side. */
    private fun fullFrameQuad(): NormalizedQuad {
        val m = 0.05f
        return NormalizedQuad(
            topLeft     = NormalizedPoint(m,      m    ),
            topRight    = NormalizedPoint(1f - m, m    ),
            bottomRight = NormalizedPoint(1f - m, 1f - m),
            bottomLeft  = NormalizedPoint(m,      1f - m),
            confidence  = 0f
        )
    }
}
