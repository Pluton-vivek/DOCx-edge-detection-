package com.example.myapplication.detection

import android.graphics.Bitmap
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtSession
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.FloatBuffer

private const val TAG = "DocQuadNetDetector"

/**
 * Runs DocQuadNet256 ONNX inference on a full-resolution bitmap.
 *
 * Pipeline:
 *  1. Letterbox the bitmap to 256×256 (preserves aspect ratio, pads with black)
 *  2. Apply ImageNet normalization (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
 *  3. Run ONNX inference → [corner_heatmaps] tensor: shape [1, 4, H, W]
 *  4. For each of the 4 corner heatmaps:
 *     a. Find the peak coordinate (argmax)
 *     b. Apply sub-pixel refinement via weighted centroid in a 3×3 neighbourhood
 *     c. Denormalize from 256px letterbox space → original image pixel coords
 *     d. Normalize to [0.0, 1.0] relative to original image dimensions
 *  5. Compute overall confidence from sigmoid of mean peak values
 *
 * @param onnxSessionManager Shared session manager (session initialized once at app startup).
 */
class DocQuadNetDetector(private val onnxSessionManager: OnnxSessionManager) {

    private val preprocessor = LetterboxPreprocessor()

    companion object {
        private const val TARGET_SIZE = 256

        // Input/output tensor names — verify against the model in Netron (netron.app)
        // if inference crashes with "unknown input/output name".
        private const val INPUT_NAME = "input"
        private const val HEATMAP_OUTPUT_NAME = "corner_heatmaps"
    }

    /**
     * Runs inference on [bitmap] and returns a [NormalizedQuad] with a confidence score.
     * Runs entirely on [Dispatchers.Default] — never call from the main thread.
     *
     * @throws IllegalStateException if the ONNX session is not initialized.
     */
    suspend fun detect(bitmap: Bitmap): NormalizedQuad = withContext(Dispatchers.Default) {
        val inferenceStart = System.currentTimeMillis()

        // 1. Preprocess: letterbox + ImageNet normalize → CHW float array
        val (floatArray, padInfo) = preprocessor.preprocess(bitmap, TARGET_SIZE)

        // 2. Get session (caller must have checked OnnxSessionManager.isReady())
        val session = onnxSessionManager.getSession()
        val env     = onnxSessionManager.env

        // 3. Create input tensor, run inference, close tensor
        val inputTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(floatArray),
            longArrayOf(1L, 3L, TARGET_SIZE.toLong(), TARGET_SIZE.toLong())
        )

        val outputMap: OrtSession.Result = session.run(mapOf(INPUT_NAME to inputTensor))
        inputTensor.close()

        Log.d(TAG, "ONNX inference: ${System.currentTimeMillis() - inferenceStart}ms")

        // 4. Decode heatmaps
        // ORT Java API: OrtSession.Result implements AutoCloseable and Iterable<Map.Entry<String,OnnxValue>>
        // The value() of a float tensor is a multi-dim Java array: Array<Array<Array<FloatArray>>>
        val heatmapValue = outputMap.get(HEATMAP_OUTPUT_NAME)
        outputMap.close()

        if (heatmapValue == null || !heatmapValue.isPresent) {
            Log.e(TAG, "Output '$HEATMAP_OUTPUT_NAME' not found. " +
                       "Check model output names in Netron. Returning full-frame fallback.")
            return@withContext fullFrameQuad()
        }

        // Cast: shape [1, 4, heatmapH, heatmapW] → [batch][corner][row][col]
        @Suppress("UNCHECKED_CAST")
        val heatmapRaw = heatmapValue.get().value as? Array<Array<Array<FloatArray>>>
        heatmapValue.get().close()

        if (heatmapRaw == null) {
            Log.e(TAG, "Heatmap tensor has unexpected type. Returning full-frame fallback.")
            return@withContext fullFrameQuad()
        }

        val heatmaps = heatmapRaw[0]          // shape [4][H][W]
        val heatmapH = heatmaps[0].size
        val heatmapW = heatmaps[0][0].size

        val corners = Array(4) { idx ->
            decodeHeatmap(heatmaps[idx], heatmapH, heatmapW, padInfo, bitmap.width, bitmap.height)
        }

        val confidence = computeConfidence(heatmaps, heatmapH, heatmapW)
        Log.d(TAG, "Confidence=$confidence  corners=${corners.map { "(${it.x.format()},${it.y.format()})" }}")

        NormalizedQuad(
            topLeft     = corners[0],
            topRight    = corners[1],
            bottomRight = corners[2],
            bottomLeft  = corners[3],
            confidence  = confidence
        )
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    /**
     * Finds the peak in a single heatmap and returns it as a [NormalizedPoint]
     * in the original image's coordinate space (0.0–1.0).
     *
     * Steps:
     *  - Argmax → (peakRow, peakCol)
     *  - Sub-pixel refinement: weighted centroid in 3×3 neighbourhood
     *  - Denorm: heatmap space → 256px canvas → original image pixels
     *  - Normalize: pixel / dimension → [0.0, 1.0]
     */
    private fun decodeHeatmap(
        heatmap: Array<FloatArray>,
        heatmapH: Int,
        heatmapW: Int,
        padInfo: PadInfo,
        origW: Int,
        origH: Int
    ): NormalizedPoint {
        // Argmax
        var maxVal = Float.NEGATIVE_INFINITY
        var peakRow = 0; var peakCol = 0
        for (r in 0 until heatmapH) for (c in 0 until heatmapW) {
            if (heatmap[r][c] > maxVal) { maxVal = heatmap[r][c]; peakRow = r; peakCol = c }
        }

        // Weighted centroid in 3×3 window (sub-pixel refinement)
        var sumW = 0f; var sumX = 0f; var sumY = 0f
        for (dr in -1..1) for (dc in -1..1) {
            val r = (peakRow + dr).coerceIn(0, heatmapH - 1)
            val c = (peakCol + dc).coerceIn(0, heatmapW  - 1)
            val w = heatmap[r][c].coerceAtLeast(0f)
            sumW += w; sumX += w * c; sumY += w * r
        }
        val subX = if (sumW > 0f) sumX / sumW else peakCol.toFloat()
        val subY = if (sumW > 0f) sumY / sumW else peakRow.toFloat()

        // Denormalize: heatmap coords → 256px canvas coords → original image coords
        // heatmap may be smaller than TARGET_SIZE (e.g. 64×64) — scale accordingly
        val hToCanvas = TARGET_SIZE.toFloat() / heatmapW
        val vToCanvas = TARGET_SIZE.toFloat() / heatmapH
        val canvasX = subX * hToCanvas
        val canvasY = subY * vToCanvas

        val imageX = ((canvasX - padInfo.padX) / padInfo.scale).coerceIn(0f, origW.toFloat())
        val imageY = ((canvasY - padInfo.padY) / padInfo.scale).coerceIn(0f, origH.toFloat())

        return NormalizedPoint(imageX / origW, imageY / origH)
    }

    /**
     * Overall confidence = sigmoid of the mean peak activation across all 4 heatmaps.
     * Returns a value in [0.0, 1.0].
     */
    private fun computeConfidence(
        heatmaps: Array<Array<FloatArray>>,
        heatmapH: Int,
        heatmapW: Int
    ): Float {
        var sumPeaks = 0f
        for (h in heatmaps) {
            var maxVal = Float.NEGATIVE_INFINITY
            for (r in 0 until heatmapH) for (c in 0 until heatmapW) {
                if (h[r][c] > maxVal) maxVal = h[r][c]
            }
            sumPeaks += maxVal
        }
        val meanPeak = sumPeaks / heatmaps.size
        return sigmoid(meanPeak)
    }

    private fun sigmoid(x: Float): Float = (1f / (1f + Math.exp(-x.toDouble()))).toFloat()

    private fun fullFrameQuad(): NormalizedQuad {
        val m = 0.05f
        return NormalizedQuad(
            NormalizedPoint(m, m), NormalizedPoint(1f - m, m),
            NormalizedPoint(1f - m, 1f - m), NormalizedPoint(m, 1f - m),
            confidence = 0f
        )
    }

    private fun Float.format() = "%.3f".format(this)
}
