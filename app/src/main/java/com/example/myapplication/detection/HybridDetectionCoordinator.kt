package com.example.myapplication.detection

import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

import com.example.myapplication.telemetry.TelemetryCoordinator

private const val TAG = "HybridCoordinator"

/**
 * Decision logic that combines the OpenCV classical result (from the C++ pipeline,
 * called upstream via [DocumentProcessor.nativeScanJSON]) with the ONNX result
 * (from [DocQuadNetDetector]) and returns the best quad with its [DetectionMethod].
 *
 * Decision priority:
 *  1. ONNX result, if confidence ≥ [onnxConfidenceThreshold] AND mask ≥ [maskScoreThreshold]  → ONNX_REFINED
 *  2. M-LSD line detection, if ONNX was uncertain but lines form a quad                        → MLSD_LINES
 *  3. OpenCV result, if provided                                                               → OPENCV_CONTOUR
 *  4. Full-frame quad with 5% margin                                                           → MANUAL_FALLBACK
 *
 * Usage (capture path only — never on live viewfinder frames):
 *   val result = coordinator.detect(rotatedBitmap, opencvRelativePoints)
 *
 * @param onnxDetector The ONNX inference engine (session already initialized).
 * @param onnxConfidenceThreshold Minimum ONNX confidence to trust the result (default 0.65).
 * @param maskScoreThreshold Minimum mean-sigmoid mask score to trust the result (default 0.35).
 * @param mlsdDetector Optional M-LSD detector — null-safe if TFLite is not ready.
 */
class HybridDetectionCoordinator(
    private val onnxDetector: DocQuadNetDetector,
    private val onnxConfidenceThreshold: Float = 0.65f,
    private val maskScoreThreshold: Float = 0.35f,
    private val mlsdDetector: MLSDDetector? = null
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

        TelemetryCoordinator.recordRawOnnxResult(onnxQuad)

        // --- Step 2: M-LSD inference (only when ONNX is uncertain — saves ~30ms on happy path) ---
        val mlsdQuad = if (
            mlsdDetector != null &&
            (onnxQuad == null || onnxQuad.confidence < onnxConfidenceThreshold)
        ) {
            try {
                mlsdDetector.detect(bitmap)
            } catch (e: Exception) {
                Log.e(TAG, "M-LSD failed: ${e.message}")
                null
            }
        } else null

        // --- Step 3: Decision logic ---
        when {
            onnxQuad != null
            && onnxQuad.confidence >= onnxConfidenceThreshold
            && onnxQuad.maskScore  >= maskScoreThreshold -> {
                Log.d(TAG, "ONNX_REFINED " +
                    "conf=${onnxQuad.confidence}  " +
                    "mask=${onnxQuad.maskScore}  " +
                    "threshold_conf=${onnxConfidenceThreshold}  " +
                    "threshold_mask=${maskScoreThreshold}")
                Pair(onnxQuad, DetectionMethod.ONNX_REFINED)
            }

            mlsdQuad != null -> {
                Log.d(TAG, "MLSD_LINES conf=${mlsdQuad.confidence}")
                Pair(mlsdQuad, DetectionMethod.MLSD_LINES)
            }

            opencvPoints != null && opencvPoints.size == 4 -> {
                Log.d(TAG, "OPENCV_CONTOUR — " +
                    "onnx_conf=${onnxQuad?.confidence ?: "null"}  " +
                    "onnx_mask=${onnxQuad?.maskScore ?: "null"}  " +
                    "reason=${if (onnxQuad == null) "onnx_failed"
                              else if (onnxQuad.confidence < onnxConfidenceThreshold) "low_confidence"
                              else "low_mask_score"}")
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
                Log.d(TAG, "MANUAL_FALLBACK — onnx_conf=${onnxQuad?.confidence ?: "null"}  " +
                    "onnx_mask=${onnxQuad?.maskScore ?: "null"}")
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
