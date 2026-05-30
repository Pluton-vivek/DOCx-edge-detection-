package com.example.myapplication.detection

import android.graphics.Bitmap
import android.util.Log
import ai.onnxruntime.OnnxTensor
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.FloatBuffer

private const val TAG = "DocQuadNetDetector"

/**
 * Runs DocQuadNet256 ONNX inference on a bitmap.
 *
 * Pipeline:
 *  1. Pre-scale the input bitmap to ≤ [PREPROCESS_MAX_DIM] px (saves ~400ms on 4K captures)
 *  2. Letterbox to [TARGET_SIZE]×[TARGET_SIZE] with ImageNet normalization
 *  3. Run ONNX inference → [corner_heatmaps] tensor: shape [1, 4, H, W]
 *  4. For each of the 4 corners:
 *     a. Argmax peak in that heatmap
 *     b. 3×3 weighted-centroid sub-pixel refinement
 *     c. Denormalize: heatmap → 256px canvas → pre-scaled image → original bitmap coords
 *     d. Normalize to [0.0, 1.0]
 *  5. Confidence = sigmoid(mean peak activation across all 4 heatmaps)
 *
 * RESOURCE CONTRACT: OrtSession.Result is closed BEFORE its contained OnnxValues are read.
 * The fix is to call `.value` (which copies native→JVM heap) BEFORE calling `outputMap.close()`.
 */
class DocQuadNetDetector(private val onnxSessionManager: OnnxSessionManager) {

    private val preprocessor = LetterboxPreprocessor()

    companion object {
        private const val TARGET_SIZE = 256

        // Pre-scale input to this size before letterboxing.
        // The model only needs 256px of resolution, so scaling a 3840px capture to 512px first
        // is lossless for the model and saves ~350ms of preprocessing time.
        private const val PREPROCESS_MAX_DIM = 512

        // Tensor names — must match the model. Verified in Netron by user.
        private const val INPUT_NAME  = "input"
        private const val HEATMAP_OUT = "corner_heatmaps"
        private const val MASK_OUT    = "mask_logits"
    }

    /**
     * Runs inference on [bitmap] and returns a [NormalizedQuad] with confidence.
     * Must NOT be called from the main thread (runs on [Dispatchers.Default]).
     */
    suspend fun detect(bitmap: Bitmap): NormalizedQuad = withContext(Dispatchers.Default) {
        val start = System.currentTimeMillis()

        // ── Step 1: Pre-scale to avoid processing 4K pixels through the preprocessor ──
        // Letterbox only needs TARGET_SIZE worth of detail. Scaling to PREPROCESS_MAX_DIM
        // first is lossless and saves ~350ms on 3840×2160 captures.
        val preBitmap = if (maxOf(bitmap.width, bitmap.height) > PREPROCESS_MAX_DIM) {
            val scale = PREPROCESS_MAX_DIM.toFloat() / maxOf(bitmap.width, bitmap.height)
            val w = (bitmap.width  * scale).toInt().coerceAtLeast(1)
            val h = (bitmap.height * scale).toInt().coerceAtLeast(1)
            Bitmap.createScaledBitmap(bitmap, w, h, /*filter=*/true)
        } else {
            bitmap
        }
        Log.d(TAG, "Pre-scaled: ${bitmap.width}×${bitmap.height} → ${preBitmap.width}×${preBitmap.height} " +
                   "(${System.currentTimeMillis() - start}ms)")

        // ── Step 2: Letterbox + ImageNet normalize → CHW float array ──
        val (floatArray, padInfo) = preprocessor.preprocess(preBitmap, TARGET_SIZE)
        if (preBitmap !== bitmap) preBitmap.recycle()

        // ── Step 3: Run inference ──
        val session = onnxSessionManager.getSession()
        val env     = onnxSessionManager.env

        val inputTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(floatArray),
            longArrayOf(1L, 3L, TARGET_SIZE.toLong(), TARGET_SIZE.toLong())
        )

        val onnxStart = System.currentTimeMillis()
        val outputResult = session.run(mapOf(INPUT_NAME to inputTensor))
        val onnxMs = System.currentTimeMillis() - onnxStart
        inputTensor.close()

        // ── Step 4: Extract the float data from the OnnxValue BEFORE closing the result ──
        //
        // CRITICAL: OrtSession.Result.close() destroys all contained OnnxValues and their
        // native backing buffers. If we call outputResult.close() first and then read
        // heatmapValue.get().value, ORT throws "Trying to use a closed OnnxValue".
        //
        // Fix: call .value on the OnnxTensor while the Result is still open.
        // OnnxTensor.getValue() allocates a JVM array and copies native→JVM.
        // After that copy, the JVM array is independent of native memory.
        // Then close the Result safely.
        val heatmapRaw: Array<Array<Array<FloatArray>>>?
        var maskRaw: Array<Array<Array<FloatArray>>>? = null
        try {
            val heatmapOpt = outputResult.get(HEATMAP_OUT)
            if (heatmapOpt == null || !heatmapOpt.isPresent) {
                Log.e(TAG, "Output '$HEATMAP_OUT' missing. Check model tensor names in Netron.")
                return@withContext fullFrameQuad()
            }

            // .value copies native float data → JVM array. Must happen BEFORE close().
            @Suppress("UNCHECKED_CAST")
            heatmapRaw = heatmapOpt.get().value as? Array<Array<Array<FloatArray>>>

            // Read mask_logits — must happen before outputResult.close()
            val maskOpt = outputResult.get(MASK_OUT)
            if (maskOpt != null && maskOpt.isPresent) {
                @Suppress("UNCHECKED_CAST")
                maskRaw = maskOpt.get().value as? Array<Array<Array<FloatArray>>>
            }
        } finally {
            // Safe to close now: all JVM arrays above are already copies of native data.
            outputResult.close()
        }

        val inferMs = System.currentTimeMillis() - start
        Log.d(TAG, "ONNX inference (incl. pre-scale + letterbox): ${inferMs}ms")

        if (heatmapRaw == null) {
            Log.e(TAG, "Heatmap cast failed — unexpected tensor shape or type.")
            return@withContext fullFrameQuad()
        }

        // ── Step 5: Decode ──
        // heatmapRaw: [batch=1][corner=0..3][row][col]
        val heatmaps = heatmapRaw[0]
        val heatmapH = heatmaps[0].size
        val heatmapW = heatmaps[0][0].size
        Log.d(TAG, "Heatmap shape: 4×${heatmapH}×${heatmapW}")

        // Log per-corner peak activations for diagnostic purposes
        // This tells us whether the model has signal for each corner or not
        val cornerNames = listOf("TL", "TR", "BR", "BL")
        val peakActivations = heatmaps.mapIndexed { idx, h ->
            var peak = Float.NEGATIVE_INFINITY
            for (r in 0 until heatmapH) for (c in 0 until heatmapW) {
                if (h[r][c] > peak) peak = h[r][c]
            }
            "${cornerNames[idx]}=${peak.format()}"
        }.joinToString(" ")
        Log.d(TAG, "Corner peak activations: $peakActivations")

        val corners = Array(4) { idx ->
            decodeHeatmap(
                heatmaps[idx], heatmapH, heatmapW,
                padInfo,
                origW = bitmap.width, origH = bitmap.height
            )
        }

        val confidence = computeConfidence(heatmaps, heatmapH, heatmapW)
        val maskScore  = computeMaskScore(maskRaw, corners, heatmapH, heatmapW)

        // CHPS: mean peak sharpness across all 4 corner heatmaps
        val sharpnessPerCorner = heatmaps.map { computePeakSharpness(it, heatmapH, heatmapW) }
        val meanSharpness = sharpnessPerCorner.average().toFloat()

        Log.d(TAG, "maskScore=${maskScore.format()}  conf=${confidence.format()}  " +
                   "chps=${meanSharpness.format(1)}  " +
                   "TL=${corners[0].fmt()}  TR=${corners[1].fmt()}  " +
                   "BR=${corners[2].fmt()}  BL=${corners[3].fmt()}")

        NormalizedQuad(
            topLeft     = corners[0],
            topRight    = corners[1],
            bottomRight = corners[2],
            bottomLeft  = corners[3],
            confidence  = confidence,
            maskScore   = maskScore,
            peakSharpness = meanSharpness,
            onnxInferenceMs = onnxMs
        )
    }

    // ─── Private helpers ──────────────────────────────────────────────────────

    /**
     * Decodes a single corner heatmap into a [NormalizedPoint] in the original bitmap space.
     *
     * Note: [origW]/[origH] are the ORIGINAL (pre-pre-scale) bitmap dimensions,
     * while [padInfo] was computed on the pre-scaled bitmap. The denormalization
     * correctly maps heatmap → letterbox canvas → pre-scaled image → original image.
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
        for (r in 0 until heatmapH) {
            for (c in 0 until heatmapW) {
                if (heatmap[r][c] > maxVal) {
                    maxVal = heatmap[r][c]; peakRow = r; peakCol = c
                }
            }
        }

        // Sub-pixel refinement: 3×3 weighted centroid around peak
        var sumW = 0f; var sumX = 0f; var sumY = 0f
        for (dr in -1..1) {
            for (dc in -1..1) {
                val r = (peakRow + dr).coerceIn(0, heatmapH - 1)
                val c = (peakCol + dc).coerceIn(0, heatmapW  - 1)
                val w = heatmap[r][c].coerceAtLeast(0f)
                sumW += w; sumX += w * c; sumY += w * r
            }
        }
        val subX = if (sumW > 0f) sumX / sumW else peakCol.toFloat()
        val subY = if (sumW > 0f) sumY / sumW else peakRow.toFloat()

        // Denormalize: heatmap coords → 256px canvas (accounting for heatmap downscale) → padded image
        // The heatmap may be smaller than TARGET_SIZE (e.g. 64×64 stride-4 output).
        val hScale = TARGET_SIZE.toFloat() / heatmapW
        val vScale = TARGET_SIZE.toFloat() / heatmapH
        val canvasX = subX * hScale
        val canvasY = subY * vScale

        // Canvas → pre-scaled image
        val preScaledX = (canvasX - padInfo.padX) / padInfo.scale
        val preScaledY = (canvasY - padInfo.padY) / padInfo.scale

        // Pre-scaled image → original image (undo the PREPROCESS_MAX_DIM downscale)
        // padInfo was computed on preBitmap. preBitmap dimensions are not directly available
        // here, but we can recover the pre-scale factor from origW/padInfo:
        // preBitmap.width = origW * (PREPROCESS_MAX_DIM / max(origW, origH)) if we pre-scaled,
        // which simplifies to: imageX = preScaledX / padInfo.scale * ... 
        // Simpler: the padInfo.scale already captures letterbox scale (preBitmap → 256).
        // To get original coords: imageX = preScaledX * (origW / preBitmapW)
        // But we don't have preBitmapW here. Compute via the PREPROCESS_MAX_DIM ratio:
        val preScaleRatio = if (maxOf(origW, origH) > PREPROCESS_MAX_DIM) {
            maxOf(origW, origH).toFloat() / PREPROCESS_MAX_DIM
        } else {
            1f
        }

        val imageX = (preScaledX * preScaleRatio).coerceIn(0f, origW.toFloat())
        val imageY = (preScaledY * preScaleRatio).coerceIn(0f, origH.toFloat())

        return NormalizedPoint(imageX / origW, imageY / origH)
    }

    /**
     * Computes the mean sigmoid of [mask_logits] values within the bounding box
     * of the detected quad, as a proxy for "does this region look like a document?"
     *
     * Uses normalized corner coordinates mapped to heatmap space (approximate but
     * sufficient for a validation gate).
     */
    private fun computeMaskScore(
        maskRaw: Array<Array<Array<FloatArray>>>?,
        corners: Array<NormalizedPoint>,
        heatmapH: Int,
        heatmapW: Int
    ): Float {
        if (maskRaw == null) return 0f
        val mask = maskRaw[0][0]  // shape [heatmapH][heatmapW]

        // Map corner normalized coords to heatmap space (approximate but sufficient)
        val colCoords = corners.map { it.x * heatmapW }
        val rowCoords = corners.map { it.y * heatmapH }
        val minCol = colCoords.min().toInt().coerceIn(0, heatmapW - 1)
        val maxCol = colCoords.max().toInt().coerceIn(0, heatmapW - 1)
        val minRow = rowCoords.min().toInt().coerceIn(0, heatmapH - 1)
        val maxRow = rowCoords.max().toInt().coerceIn(0, heatmapH - 1)

        // Mean sigmoid within bounding box of detected quad
        var sum = 0f
        var count = 0
        for (r in minRow..maxRow) {
            for (c in minCol..maxCol) {
                sum += sigmoid(mask[r][c])
                count++
            }
        }
        return if (count > 0) sum / count else 0f
    }

    /** Sigmoid of mean peak activation across all 4 corner heatmaps. */
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

    private fun computePeakSharpness(
        heatmap: Array<FloatArray>,
        heatmapH: Int,
        heatmapW: Int
    ): Float {
        var maxVal = Float.NEGATIVE_INFINITY
        var minVal = Float.MAX_VALUE
        var sumVal = 0f
        val count = heatmapH * heatmapW
        for (r in 0 until heatmapH) {
            for (c in 0 until heatmapW) {
                val v = heatmap[r][c]
                if (v > maxVal) maxVal = v
                if (v < minVal) minVal = v
                sumVal += v
            }
        }
        // Normalized peak position: how far above the minimum is the peak,
        // relative to the full activation range.
        // Returns 1.0 when peak is the only high point; 0.0 when all values equal.
        // Scale-invariant and bounded in [0.0, 1.0]. No division-by-zero risk.
        val range = maxVal - minVal
        val meanVal = if (count > 0) sumVal / count else minVal
        return if (range > 1e-6f) ((maxVal - meanVal) / range).coerceIn(0f, 1f) else 0f
    }

    private fun Float.format(decimals: Int = 3) = "%.${decimals}f".format(this)
    private fun NormalizedPoint.fmt() = "(${x.format()},${y.format()})"
}
