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
     * The model graph contains Div(127.5) → Sub(1.0) internally, so we
     * feed RAW [0, 255] float values — no normalization in code.
     * (Previous code normalized here too, causing double-normalization
     * which collapsed all pixels to ~-1.0 and produced zero scores.)
     *
     * Channel order: R, G, B, A (RGBA)
     * Android Bitmap.getPixels returns 0xAARRGGBB — reorder to RGBA when filling buffer.
     */
    private fun preprocess(bitmap: Bitmap): ByteBuffer {
        // Letterbox: scale to fit INPUT_SIZE×INPUT_SIZE preserving aspect ratio.
        // Pad remaining space with black. This keeps line angles correct.
        val scale = minOf(INPUT_SIZE.toFloat() / bitmap.width,
                          INPUT_SIZE.toFloat() / bitmap.height)
        val scaledW = (bitmap.width  * scale).toInt().coerceAtLeast(1)
        val scaledH = (bitmap.height * scale).toInt().coerceAtLeast(1)
        val padLeft = (INPUT_SIZE - scaledW) / 2
        val padTop  = (INPUT_SIZE - scaledH) / 2

        val canvas = android.graphics.Bitmap.createBitmap(
            INPUT_SIZE, INPUT_SIZE, android.graphics.Bitmap.Config.ARGB_8888)
        val c = android.graphics.Canvas(canvas)
        c.drawColor(android.graphics.Color.BLACK)
        val scaled = android.graphics.Bitmap.createScaledBitmap(
            bitmap, scaledW, scaledH, true)
        c.drawBitmap(scaled, padLeft.toFloat(), padTop.toFloat(), null)
        if (scaled !== bitmap) scaled.recycle()

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        canvas.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        canvas.recycle()

        val buffer = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 4 * 4)
            .order(ByteOrder.nativeOrder())
        for (pixel in pixels) {
            val r = ((pixel shr 16) and 0xFF).toFloat()
            val g = ((pixel shr  8) and 0xFF).toFloat()
            val b = ( pixel         and 0xFF).toFloat()
            val a = ((pixel shr 24) and 0xFF).toFloat()
            buffer.putFloat(r)
            buffer.putFloat(g)
            buffer.putFloat(b)
            buffer.putFloat(a)
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
        val output0Type  = interp.getOutputTensor(0).dataType()
        val output1Type  = interp.getOutputTensor(1).dataType()
        Log.d(TAG, "Output0 shape=${output0Shape.contentToString()} type=$output0Type  " +
                   "Output1 type=$output1Type")

        val centerPointsIdx: Int
        val scoresIdx: Int
        // Detect output ordering by shape (robust to model export variation)
        if (output0Shape.size == 3 && output0Shape[2] == 2) {
            centerPointsIdx = 0; scoresIdx = 1
        } else {
            centerPointsIdx = 1; scoresIdx = 0
        }

        // Determine the data type of the center-points tensor.
        // M-LSD's tiny model outputs INT32 pixel coordinates, not FLOAT32.
        val cpType = interp.getOutputTensor(centerPointsIdx).dataType()

        val centerPointsFloat: Array<FloatArray>
        val scores = Array(1) { FloatArray(MAX_SEGMENTS) }
        var mlsdMs = 0L

        if (cpType.name == "INT32") {
            // INT32 path — allocate int buffers and convert after inference
            val centerPointsInt = Array(1) { Array(MAX_SEGMENTS) { IntArray(2) } }
            val outputMap = mutableMapOf<Int, Any>(
                centerPointsIdx to centerPointsInt,
                scoresIdx       to scores
            )
            val start = System.currentTimeMillis()
            interp.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputMap)
            mlsdMs = System.currentTimeMillis() - start
            Log.d(TAG, "Inference: ${mlsdMs}ms")

            // Convert int → float for downstream processing
            centerPointsFloat = Array(MAX_SEGMENTS) { i ->
                floatArrayOf(
                    centerPointsInt[0][i][0].toFloat(),
                    centerPointsInt[0][i][1].toFloat()
                )
            }
        } else {
            // FLOAT32 path (original assumption)
            val centerPointsBuf = Array(1) { Array(MAX_SEGMENTS) { FloatArray(2) } }
            val outputMap = mutableMapOf<Int, Any>(
                centerPointsIdx to centerPointsBuf,
                scoresIdx       to scores
            )
            val start = System.currentTimeMillis()
            interp.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputMap)
            mlsdMs = System.currentTimeMillis() - start
            Log.d(TAG, "Inference: ${mlsdMs}ms")

            centerPointsFloat = centerPointsBuf[0]
        }

        // Score distribution diagnostics (helps verify preprocessing is correct)
        val s = scores[0]
        val nonZero = s.count { it > 0f }
        Log.d(TAG, "Scores: min=%.4f max=%.4f mean=%.4f nonZero=$nonZero/${s.size}".format(
            s.min(), s.max(), s.average().toFloat()))

        return decodeToQuad(centerPointsFloat, scores[0], bitmap.width, bitmap.height, mlsdMs)
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
        origH: Int,
        mlsdMs: Long
    ): NormalizedQuad? {

        // Step 1: Filter and convert coordinates
        data class Pt(val x: Float, val y: Float, val score: Float)
        val pts = mutableListOf<Pt>()
        for (i in 0 until MAX_SEGMENTS) {
            if (scoresArr[i] < SCORE_THRESHOLD) continue
            // col/row in 160×160 → normalize to [0,1]
            // Recompute letterbox parameters matching preprocess() exactly
            val lbScale   = minOf(INPUT_SIZE.toFloat() / origW, INPUT_SIZE.toFloat() / origH)
            val lbPadLeft = (INPUT_SIZE - origW * lbScale) / 2f
            val lbPadTop  = (INPUT_SIZE - origH * lbScale) / 2f
            // Heatmap is half the input resolution (HEATMAP_SIZE = INPUT_SIZE / 2 = 160)
            // so multiply by 2 to convert to INPUT_SIZE space, then subtract padding
            val inputX = centerPts[i][0].toFloat() * 2f
            val inputY = centerPts[i][1].toFloat() * 2f
            val nx = (inputX - lbPadLeft) / (origW * lbScale)
            val ny = (inputY - lbPadTop)  / (origH * lbScale)
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
        val lineInfo = foundLines.mapIndexed { i, l ->
            val isH = abs(l.a) < abs(l.b)
            val ratio = "%.2f".format(abs(l.a) / (abs(l.b) + 1e-6f))
            "L$i:${if (isH) "H" else "V"}(ratio=$ratio,score=${"%.2f".format(l.score)})"
        }.joinToString("  ")
        Log.d(TAG, "Line orientations: $lineInfo")
        if (foundLines.size < 4) {
            Log.d(TAG, "Could not find 4 lines")
            return null
        }

        // Step 3: Classify lines using relative angles instead of a hard 45-degree threshold.
        // Sort all found lines by their absolute slope ratio: |a| / |b| = |tan(theta)|
        // The two lines with the smallest ratio are the most horizontal.
        // The two lines with the largest ratio are the most vertical.
        val sortedLines = foundLines.sortedBy { abs(it.a) / (abs(it.b) + 1e-6f) }
        
        val horizontals = sortedLines.take(2).sortedByDescending { it.score }
        val verticals   = sortedLines.takeLast(2).sortedByDescending { it.score }

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
            maskScore   = confidence,   // reuse as maskScore; lines ARE the semantic evidence
            mlsdInferenceMs = mlsdMs
        )
    }

    private fun Float.format() = "%.3f".format(this)
}
