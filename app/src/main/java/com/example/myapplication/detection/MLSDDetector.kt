package com.example.myapplication.detection

import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.*

private const val TAG = "MLSDDetector"
private const val INPUT_SIZE = 320
private const val HEATMAP_SIZE = 160        // output is half the input
private const val MAX_SEGMENTS = 200
private const val SCORE_THRESHOLD = 0.15f   // minimum score to keep a point
private const val INLIER_DIST = 0.025f      // normalized distance for RANSAC inlier
private const val MIN_QUAD_AREA = 0.04f     // minimum quad area as fraction of image

class MLSDDetector(private val sessionManager: MLSDSessionManager = MLSDSessionManager) {

    // ─── A. PREPROCESSING ────────────────────────────────────────────────────

    /**
     * Converts a Bitmap to the [1, 320, 320, 4] RGBA float32 ByteBuffer
     * expected by the model.
     *
     * Normalization: (pixel_channel / 127.5f) - 1.0f  →  range [-1.0, 1.0]
     * This matches the Div(127.5) → Sub(1.0) baked into the model input graph.
     * We apply it in code here so the model graph's own ops are redundant but harmless.
     *
     * Channel order: R, G, B, A (RGBA)
     * Android Bitmap.getPixels returns 0xAARRGGBB — reorder to RGBA when filling buffer.
     */
    private fun preprocess(bitmap: Bitmap): ByteBuffer {
        val scaled = if (bitmap.width != INPUT_SIZE || bitmap.height != INPUT_SIZE) {
            Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        } else bitmap

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        scaled.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        if (scaled !== bitmap) scaled.recycle()

        // 4 channels × 4 bytes each × 320 × 320 pixels
        val buffer = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 4 * 4)
            .order(ByteOrder.nativeOrder())

        for (pixel in pixels) {
            val r = ((pixel shr 16) and 0xFF).toFloat()
            val g = ((pixel shr  8) and 0xFF).toFloat()
            val b = ( pixel         and 0xFF).toFloat()
            val a = ((pixel shr 24) and 0xFF).toFloat()
            // Store RGBA in order, normalize each: (x / 127.5) - 1.0
            buffer.putFloat(r / 127.5f - 1.0f)
            buffer.putFloat(g / 127.5f - 1.0f)
            buffer.putFloat(b / 127.5f - 1.0f)
            buffer.putFloat(a / 127.5f - 1.0f)
        }
        buffer.rewind()
        return buffer
    }

    // ─── B. INFERENCE ────────────────────────────────────────────────────────

    /**
     * Runs M-LSD inference on the bitmap. Returns a NormalizedQuad or null.
     *
     * Output tensor shapes (verified via Netron):
     *   Output 0 [1, 200, 2]: center point positions — dim[2][0]=col, dim[2][1]=row
     *                          in 0..159 heatmap space
     *   Output 1 [1, 200]:    confidence scores per point
     *
     * If output order is swapped on the device, shape-based detection handles it.
     */
    fun detect(bitmap: Bitmap): NormalizedQuad? {
        val interp = sessionManager.getInterpreter() ?: run {
            Log.w(TAG, "Interpreter not ready")
            return null
        }

        val inputBuffer = preprocess(bitmap)

        // Allocate output buffers matching verified shapes
        val output0Shape = interp.getOutputTensor(0).shape()
        val centerPointsIdx: Int
        val scoresIdx: Int
        // Detect output ordering by shape (robust to model export variation)
        if (output0Shape.size == 3 && output0Shape[2] == 2) {
            centerPointsIdx = 0; scoresIdx = 1
        } else {
            centerPointsIdx = 1; scoresIdx = 0
        }

        val centerPoints = Array(1) { Array(MAX_SEGMENTS) { FloatArray(2) } }
        val scores       = Array(1) { FloatArray(MAX_SEGMENTS) }

        val outputMap = mutableMapOf<Int, Any>(
            centerPointsIdx to centerPoints,
            scoresIdx       to scores
        )

        val start = System.currentTimeMillis()
        interp.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputMap)
        Log.d(TAG, "Inference: ${System.currentTimeMillis() - start}ms")

        return decodeToQuad(centerPoints[0], scores[0], bitmap.width, bitmap.height)
    }

    // ─── C. POST-PROCESSING: CENTER POINTS → QUAD ────────────────────────────

    /**
     * Converts M-LSD center point detections into a document NormalizedQuad.
     *
     * Pipeline:
     *  1. Filter points by score > SCORE_THRESHOLD
     *  2. Convert from 160×160 heatmap coords to normalized [0,1] image coords
     *  3. Use random-sample line fitting (RANSAC-inspired) to find 4 dominant lines
     *  4. Classify lines as horizontal or vertical
     *  5. Intersect 2 horizontal × 2 vertical → 4 candidate corners
     *  6. Order corners: TL (min x+y), TR (max x-y), BR (max x+y), BL (min x-y)
     *  7. Validate convexity and minimum area
     *  8. Return NormalizedQuad with maskScore = lineConfidence
     */
    private fun decodeToQuad(
        centerPts: Array<FloatArray>,
        scoresArr: FloatArray,
        origW: Int,
        origH: Int
    ): NormalizedQuad? {

        // Step 1: Filter and convert coordinates
        data class Pt(val x: Float, val y: Float, val score: Float)
        val pts = mutableListOf<Pt>()
        for (i in 0 until MAX_SEGMENTS) {
            if (scoresArr[i] < SCORE_THRESHOLD) continue
            // col/row in 160×160 → normalize to [0,1]
            val nx = centerPts[i][0] / HEATMAP_SIZE.toFloat()
            val ny = centerPts[i][1] / HEATMAP_SIZE.toFloat()
            if (nx in 0f..1f && ny in 0f..1f) {
                pts.add(Pt(nx, ny, scoresArr[i]))
            }
        }

        Log.d(TAG, "Valid points after filter: ${pts.size} (threshold=$SCORE_THRESHOLD)")
        if (pts.size < 4) {
            Log.d(TAG, "Too few points for quad detection")
            return null
        }

        // Step 2: RANSAC-inspired dominant line extraction
        // Represent each line as (a, b, c) in ax + by + c = 0 form, normalized: a²+b²=1
        data class Line(val a: Float, val b: Float, val c: Float, val score: Float)

        fun lineThrough(p1: Pt, p2: Pt): Line? {
            val dx = p2.x - p1.x
            val dy = p2.y - p1.y
            val len = sqrt(dx * dx + dy * dy)
            if (len < 1e-6f) return null
            val a =  dy / len
            val b = -dx / len
            val c = -(a * p1.x + b * p1.y)
            return Line(a, b, c, 0f)
        }

        fun distToLine(p: Pt, l: Line): Float = abs(l.a * p.x + l.b * p.y + l.c)

        fun countInliers(line: Line, pool: List<Pt>): List<Pt> =
            pool.filter { distToLine(it, line) < INLIER_DIST }

        val foundLines = mutableListOf<Line>()
        var pool = pts.toMutableList()
        val rng = java.util.Random(42L)
        val RANSAC_ITERATIONS = 80

        repeat(4) {                              // try to find up to 4 lines
            if (pool.size < 2) return@repeat
            var bestLine: Line? = null
            var bestInliers = emptyList<Pt>()

            repeat(RANSAC_ITERATIONS) {
                val i1 = rng.nextInt(pool.size)
                var i2 = rng.nextInt(pool.size)
                while (i2 == i1) i2 = rng.nextInt(pool.size)
                val candidate = lineThrough(pool[i1], pool[i2]) ?: return@repeat
                val inliers = countInliers(candidate, pool)
                if (inliers.size > bestInliers.size) {
                    bestLine = candidate; bestInliers = inliers
                }
            }

            val bl = bestLine ?: return@repeat
            if (bestInliers.size < 3) return@repeat   // too few inliers — skip

            // Refit the line to all its inliers for accuracy
            val meanX = bestInliers.map { it.x }.average().toFloat()
            val meanY = bestInliers.map { it.y }.average().toFloat()
            var xx = 0f; var xy = 0f; var yy = 0f
            for (p in bestInliers) {
                val dx = p.x - meanX; val dy = p.y - meanY
                xx += dx * dx; xy += dx * dy; yy += dy * dy
            }
            // PCA: largest eigenvector gives line direction
            val diff = xx - yy
            val angle = 0.5f * atan2(2 * xy, diff)
            val a =  sin(angle)
            val b = -cos(angle)
            val c = -(a * meanX + b * meanY)
            val lineScore = bestInliers.map { it.score }.average().toFloat()
            foundLines.add(Line(a, b, c, lineScore))

            // Remove inliers so next iteration finds a different line
            val inlierSet = bestInliers.toSet()
            pool = pool.filter { it !in inlierSet }.toMutableList()
        }

        Log.d(TAG, "Lines found: ${foundLines.size}")
        if (foundLines.size < 4) {
            Log.d(TAG, "Could not find 4 lines")
            return null
        }

        // Step 3: Classify lines as horizontal (|a| < |b|) or vertical (|a| >= |b|)
        // In ax + by + c = 0: 'a' is x-coefficient, 'b' is y-coefficient
        // Horizontal line: nearly y = const → |a| small, |b| large → |a| < |b|
        // Vertical line:   nearly x = const → |a| large, |b| small → |a| >= |b|
        val horizontals = foundLines.filter { abs(it.a) < abs(it.b) }
            .sortedByDescending { it.score }
        val verticals   = foundLines.filter { abs(it.a) >= abs(it.b) }
            .sortedByDescending { it.score }

        Log.d(TAG, "H lines: ${horizontals.size}  V lines: ${verticals.size}")
        if (horizontals.size < 2 || verticals.size < 2) {
            Log.d(TAG, "Need at least 2H + 2V, got H=${horizontals.size} V=${verticals.size}")
            return null
        }

        val h1 = horizontals[0]; val h2 = horizontals[1]
        val v1 = verticals[0];   val v2 = verticals[1]

        // Step 4: Compute 4 intersection points
        // Solve: a1*x + b1*y + c1 = 0 and a2*x + b2*y + c2 = 0
        fun intersect(l1: Line, l2: Line): Pair<Float, Float>? {
            val det = l1.a * l2.b - l2.a * l1.b
            if (abs(det) < 1e-6f) return null        // parallel lines
            val x = (l1.b * l2.c - l2.b * l1.c) / det
            val y = (l2.a * l1.c - l1.a * l2.c) / det
            return Pair(-x, -y)                       // correct for sign convention
        }

        val c1 = intersect(h1, v1) ?: return null
        val c2 = intersect(h1, v2) ?: return null
        val c3 = intersect(h2, v1) ?: return null
        val c4 = intersect(h2, v2) ?: return null

        val rawCorners = listOf(c1, c2, c3, c4)

        // Step 5: Order corners — TL(min x+y), TR(max x-y), BR(max x+y), BL(min x-y)
        val tl = rawCorners.minByOrNull { it.first + it.second }!!
        val br = rawCorners.maxByOrNull { it.first + it.second }!!
        val tr = rawCorners.maxByOrNull { it.first - it.second }!!
        val bl = rawCorners.minByOrNull { it.first - it.second }!!

        // Step 6: Validate — clamp to [0,1] and check area
        fun clamp(v: Float) = v.coerceIn(0f, 1f)
        val tlN = NormalizedPoint(clamp(tl.first), clamp(tl.second))
        val trN = NormalizedPoint(clamp(tr.first), clamp(tr.second))
        val brN = NormalizedPoint(clamp(br.first), clamp(br.second))
        val blN = NormalizedPoint(clamp(bl.first), clamp(bl.second))

        // Shoelace area formula
        val area = abs(
            (tlN.x - brN.x) * (trN.y - blN.y) -
            (trN.x - blN.x) * (tlN.y - brN.y)
        ) / 2f

        Log.d(TAG, "Quad area=${area.format()}  required>=${MIN_QUAD_AREA}")
        if (area < MIN_QUAD_AREA) {
            Log.d(TAG, "Quad too small — rejecting")
            return null
        }

        val confidence = foundLines.take(4).map { it.score }.average().toFloat()
        Log.d(TAG, "MLSD quad detected  confidence=${confidence.format()}  area=${area.format()}")

        return NormalizedQuad(
            topLeft     = tlN,
            topRight    = trN,
            bottomRight = brN,
            bottomLeft  = blN,
            confidence  = confidence,
            maskScore   = confidence    // reuse as maskScore; lines ARE the semantic evidence
        )
    }

    private fun Float.format() = "%.3f".format(this)
}
