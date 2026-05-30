package com.example.myapplication.telemetry

import android.util.Log
import com.example.myapplication.PointF
import com.example.myapplication.detection.DetectionMethod
import com.example.myapplication.detection.NormalizedQuad
import kotlin.math.*

/**
 * Singleton telemetry aggregator for DocFlow document detection evaluation.
 *
 * Measures 7 parameters:
 *  P1 FDC:  Frame Detection Consistency  — ratio of stable consecutive outcomes
 *  P2 tIoU: Temporal IoU Stability       — bounding box overlap between consecutive frames
 *  P3 CCG:  Confidence Calibration Gap   — onnxThreshold - mean(onnxConf)
 *  P4 PAD:  Path Activation Distribution — ONNX vs OpenCV vs M-LSD vs Fallback routing fractions
 *  P5 QGVS: Quad Geometric Validity      — convexity + AR + angle + coverage composite
 *  P6 MIL:  Mean Inference Latency       — per subsystem moving average
 *  P7 CHPS: Corner Heatmap Peak Sharpness— peak/mean heatmap activation ratio
 *
 * Thread-safety: all public methods are synchronized. Call from any thread.
 * Emission: call emitLiveSummary() every 30 live frames,
 *           call emitCaptureSummary() after each capture completes.
 * Filter Logcat by tag "DocFlowTelemetry" to see structured JSON output.
 */
object TelemetryCoordinator {

    private const val TAG = "DocFlowTelemetry"
    private const val ONNX_CONFIDENCE_THRESHOLD = 0.65f  // must match HybridDetectionCoordinator
    private const val LIVE_WINDOW = 60                   // ~10s at 6fps

    // ── Live frame state ──────────────────────────────────────────────────────
    @Volatile private var lastOutcome: Boolean? = null
    @Volatile private var lastQuadBBox: FloatArray? = null  // [minX, minY, maxX, maxY] normalized
    @Volatile private var lastFrameTimeMs: Long = 0L
    @Volatile private var noQuadSinceMs: Long = 0L         // for DLFL tracking
    @Volatile private var firstStableLockMs: Long = 0L
    @Volatile private var stableRunCount: Int = 0

    private val fdcWindow   = ArrayDeque<Boolean>()         // detection outcomes
    private val tIouWindow  = ArrayDeque<Float>()           // frame-to-frame IoU values
    private val cvLatencies = ArrayDeque<Long>()            // OpenCV ms per live frame

    // ── Capture-event state ───────────────────────────────────────────────────
    private var captureCount     = 0
    private var onnxCount        = 0
    private var opencvCount      = 0
    private var mlsdCount        = 0
    private var fallbackCount    = 0
    private val onnxConfs        = ArrayDeque<Float>()
    private val chpsValues       = ArrayDeque<Float>()
    private val qgvsValues       = ArrayDeque<Float>()
    private val onnxLatencies    = ArrayDeque<Long>()
    private val mlsdLatencies    = ArrayDeque<Long>()

    // ── P1 + P2: called per live frame ───────────────────────────────────────

    /**
     * Report the result of one live ImageAnalysis frame.
     * @param quad      Detected quad corners in PIXEL space (not normalized), or null if not found.
     * @param frameW    Frame width in pixels.
     * @param frameH    Frame height in pixels.
     * @param cvMs      Time taken by the OpenCV C++ pipeline for this frame.
     */
    @Synchronized
    fun reportLiveFrame(quad: List<PointF>?, frameW: Int, frameH: Int, cvMs: Long) {
        val found = quad != null && quad.size == 4
        val nowMs = System.currentTimeMillis()

        // FDC
        val prev = lastOutcome
        if (prev != null) {
            fdcWindow.addLast(found == prev)
            if (fdcWindow.size > LIVE_WINDOW) fdcWindow.removeFirst()
        }
        lastOutcome = found

        // DLFL (Detection Latency to First Lock)
        if (!found) {
            noQuadSinceMs = nowMs
            stableRunCount = 0
        } else {
            stableRunCount++
            if (stableRunCount == 3 && firstStableLockMs == 0L && noQuadSinceMs > 0L) {
                firstStableLockMs = nowMs  // first time 3 consecutive detections hit
            }
        }

        // tIoU — axis-aligned bounding box approximation of polygon IoU
        if (found) {
            val currBBox = quadToBBox(quad!!, frameW.toFloat(), frameH.toFloat())
            val prevBBox = lastQuadBBox
            if (prevBBox != null) {
                val iou = bboxIoU(prevBBox, currBBox)
                tIouWindow.addLast(iou)
                if (tIouWindow.size > LIVE_WINDOW) tIouWindow.removeFirst()
            }
            lastQuadBBox = currBBox
        } else {
            lastQuadBBox = null
        }

        // MIL (OpenCV)
        cvLatencies.addLast(cvMs)
        if (cvLatencies.size > LIVE_WINDOW) cvLatencies.removeFirst()

        lastFrameTimeMs = nowMs
    }

    // ── P3–P7: called per capture event ──────────────────────────────────────

    /**
     * Report the result of one capture (shutter tap).
     * @param quad      Final quad chosen by HybridCoordinator (normalized 0.0–1.0).
     * @param method    Which pipeline path produced it.
     * @param onnxConf  Raw ONNX confidence (0 if ONNX didn't run or failed).
     * @param chps      Corner Heatmap Peak Sharpness from DocQuadNetDetector.
     * @param onnxMs    ONNX inference duration in ms (0 if not run).
     * @param mlsdMs    M-LSD inference duration in ms (0 if not run).
     */
    @Synchronized
    fun reportCapture(
        quad: NormalizedQuad,
        method: DetectionMethod,
        onnxConf: Float,
        chps: Float,
        onnxMs: Long,
        mlsdMs: Long
    ) {
        captureCount++
        when (method) {
            DetectionMethod.ONNX_REFINED    -> onnxCount++
            DetectionMethod.OPENCV_CONTOUR  -> opencvCount++
            DetectionMethod.MLSD_LINES      -> mlsdCount++
            DetectionMethod.MANUAL_FALLBACK -> fallbackCount++
        }

        if (onnxConf > 0f) { onnxConfs.addLast(onnxConf); if (onnxConfs.size > 50) onnxConfs.removeFirst() }
        if (chps > 0f)     { chpsValues.addLast(chps);     if (chpsValues.size > 50) chpsValues.removeFirst() }
        if (onnxMs > 0)    { onnxLatencies.addLast(onnxMs); if (onnxLatencies.size > 20) onnxLatencies.removeFirst() }
        if (mlsdMs > 0)    { mlsdLatencies.addLast(mlsdMs); if (mlsdLatencies.size > 20) mlsdLatencies.removeFirst() }

        val qgvs = computeQGVS(quad)
        qgvsValues.addLast(qgvs)
        if (qgvsValues.size > 50) qgvsValues.removeFirst()
    }

    /**
     * Records raw ONNX inference metrics BEFORE the routing decision.
     * Must be called from HybridDetectionCoordinator after ONNX runs,
     * regardless of which path is ultimately selected.
     */
    @Synchronized
    fun recordRawOnnxResult(onnxQuad: NormalizedQuad?) {
        if (onnxQuad == null) return
        if (onnxQuad.confidence > 0f) {
            onnxConfs.addLast(onnxQuad.confidence)
            if (onnxConfs.size > 50) onnxConfs.removeFirst()
        }
        if (onnxQuad.peakSharpness > 0f) {
            chpsValues.addLast(onnxQuad.peakSharpness)
            if (chpsValues.size > 50) chpsValues.removeFirst()
        }
        if (onnxQuad.onnxInferenceMs > 0L) {
            onnxLatencies.addLast(onnxQuad.onnxInferenceMs)
            if (onnxLatencies.size > 20) onnxLatencies.removeFirst()
        }
        if (onnxQuad.mlsdInferenceMs > 0L) {
            mlsdLatencies.addLast(onnxQuad.mlsdInferenceMs)
            if (mlsdLatencies.size > 20) mlsdLatencies.removeFirst()
        }
    }

    /** Emit a JSON live-frame summary. Call every 30 live frames. */
    @Synchronized
    fun emitLiveSummary(framesAnalyzed: Int) {
        val fdc  = if (fdcWindow.isEmpty()) -1f else fdcWindow.average().toFloat()
        val tIou = if (tIouWindow.isEmpty()) -1f else tIouWindow.average().toFloat()
        val tIouMin = tIouWindow.minOrNull() ?: -1f
        val cvMil = cvLatencies.average().toLong()
        val dlfl = if (firstStableLockMs > 0L &&
                       noQuadSinceMs > 0L &&
                       firstStableLockMs > noQuadSinceMs)
                       (firstStableLockMs - noQuadSinceMs)
                   else -1L

        val json = buildString {
            append("{")
            append("\"type\":\"live\",")
            append("\"frames\":$framesAnalyzed,")
            append("\"fdc\":${fdc.fmt()},")
            append("\"t_iou_mean\":${tIou.fmt()},")
            append("\"t_iou_min\":${tIouMin.fmt()},")
            append("\"cv_mil_ms\":$cvMil,")
            append("\"dlfl_ms\":$dlfl")
            append("}")
        }
        Log.d(TAG, json)
    }

    /** Emit a JSON capture summary. Call after each shutter capture completes. */
    @Synchronized
    fun emitCaptureSummary() {
        val onnxMean = if (onnxConfs.isEmpty()) -1f
                       else (onnxConfs.sumOf { it.toDouble() } / onnxConfs.size).toFloat()
        val ccg = if (onnxMean < 0f) -1f else ONNX_CONFIDENCE_THRESHOLD - onnxMean
        val padOnnx  = if (captureCount > 0) onnxCount.toFloat()    / captureCount else 0f
        val padCv    = if (captureCount > 0) opencvCount.toFloat()  / captureCount else 0f
        val padMlsd  = if (captureCount > 0) mlsdCount.toFloat()    / captureCount else 0f
        val padFall  = if (captureCount > 0) fallbackCount.toFloat()/ captureCount else 0f
        val qgvsMean = qgvsValues.average().toFloat()
        val chpsMean = chpsValues.average().toFloat()
        val onnxMil  = onnxLatencies.average().toLong()
        val mlsdMil  = mlsdLatencies.average().toLong()

        val json = buildString {
            append("{")
            append("\"type\":\"capture\",")
            append("\"total_captures\":$captureCount,")
            append("\"ccg\":${ccg.fmt()},")
            append("\"onnx_conf_mean\":${onnxMean.fmt()},")
            append("\"chps_mean\":${chpsMean.fmt(1)},")
            append("\"pad\":{\"onnx\":${padOnnx.fmt()},\"opencv\":${padCv.fmt()},\"mlsd\":${padMlsd.fmt()},\"fallback\":${padFall.fmt()}},")
            append("\"qgvs_mean\":${qgvsMean.fmt()},")
            append("\"mil\":{\"onnx_ms\":$onnxMil,\"mlsd_ms\":$mlsdMil}")
            append("}")
        }
        Log.d(TAG, json)

        // Reset DLFL after each capture so it re-arms for the next document
        firstStableLockMs = 0L
    }

    // ── QGVS computation ─────────────────────────────────────────────────────

    /**
     * Quad Geometric Validity Score — composite of 4 independent geometry checks.
     * Input: NormalizedQuad with coordinates in [0.0, 1.0].
     * Returns: score in [0.0, 1.0]. Score < 0.45 indicates likely false positive.
     *
     * Sub-scores:
     *  (a) Convexity       — 1.0 if convex, 0.0 if not (cross product sign check)
     *  (b) Aspect Ratio    — 1.0 if AR in [1.1, 3.5], decays linearly outside [0.8, 5.0]
     *  (c) Min Angle       — min interior angle / 45.0, capped at 1.0
     *  (d) Frame Coverage  — 1.0 if quad area in [5%, 75%] of frame, else 0.0
     */
    private fun computeQGVS(q: NormalizedQuad): Float {
        val pts = listOf(
            floatArrayOf(q.topLeft.x,     q.topLeft.y),
            floatArrayOf(q.topRight.x,    q.topRight.y),
            floatArrayOf(q.bottomRight.x, q.bottomRight.y),
            floatArrayOf(q.bottomLeft.x,  q.bottomLeft.y)
        )

        // (a) Convexity via cross product sign consistency
        val n = pts.size
        var signPos = 0; var signNeg = 0
        for (i in 0 until n) {
            val a = pts[i]; val b = pts[(i + 1) % n]; val c = pts[(i + 2) % n]
            val cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
            if (cross > 0) signPos++ else if (cross < 0) signNeg++
        }
        val convexScore = if (signPos == 0 || signNeg == 0) 1.0f else 0.0f

        // (b) Aspect ratio: use width of top edge and height of left edge as proxies
        val topW  = dist(pts[0], pts[1])
        val botW  = dist(pts[3], pts[2])
        val leftH = dist(pts[0], pts[3])
        val rightH= dist(pts[1], pts[2])
        val w = (topW + botW) / 2f
        val h = (leftH + rightH) / 2f
        val ar = if (h > 0f) max(w, h) / min(w, h) else 0f
        val arScore = when {
            ar in 1.1f..3.5f -> 1.0f
            ar < 0.8f || ar > 5.0f -> 0.0f
            ar < 1.1f -> (ar - 0.8f) / 0.3f
            else      -> (5.0f - ar) / 1.5f
        }

        // (c) Minimum interior angle
        val angles = FloatArray(4) { i ->
            val prev = pts[(i + 3) % 4]; val curr = pts[i]; val next = pts[(i + 1) % 4]
            val v1 = floatArrayOf(prev[0] - curr[0], prev[1] - curr[1])
            val v2 = floatArrayOf(next[0] - curr[0], next[1] - curr[1])
            val dot = v1[0]*v2[0] + v1[1]*v2[1]
            val mag = sqrt((v1[0]*v1[0] + v1[1]*v1[1]) * (v2[0]*v2[0] + v2[1]*v2[1]))
            if (mag > 0f) Math.toDegrees(acos((dot / mag).coerceIn(-1f, 1f).toDouble())).toFloat()
            else 0f
        }
        val minAngle = angles.min()
        val angleScore = (minAngle / 45.0f).coerceIn(0f, 1f)

        // (d) Frame coverage via Shoelace formula
        var area = 0f
        for (i in 0 until n) {
            val j = (i + 1) % n
            area += pts[i][0] * pts[j][1]
            area -= pts[j][0] * pts[i][1]
        }
        area = abs(area) / 2f
        val coverageScore = if (area in 0.05f..0.75f) 1.0f else 0.0f

        return 0.25f * convexScore + 0.30f * arScore + 0.25f * angleScore + 0.20f * coverageScore
    }

    // ── Geometry helpers ─────────────────────────────────────────────────────

    private fun dist(a: FloatArray, b: FloatArray): Float {
        val dx = a[0] - b[0]; val dy = a[1] - b[1]
        return sqrt(dx * dx + dy * dy)
    }

    /** Convert pixel-space quad to normalized axis-aligned bounding box [minX, minY, maxX, maxY]. */
    private fun quadToBBox(pts: List<PointF>, w: Float, h: Float): FloatArray {
        val xs = pts.map { it.x / w }
        val ys = pts.map { it.y / h }
        return floatArrayOf(xs.min(), ys.min(), xs.max(), ys.max())
    }

    /** Axis-aligned bounding box IoU. Fast approximation of polygon IoU for convex quads. */
    private fun bboxIoU(a: FloatArray, b: FloatArray): Float {
        val interMinX = max(a[0], b[0]); val interMinY = max(a[1], b[1])
        val interMaxX = min(a[2], b[2]); val interMaxY = min(a[3], b[3])
        val interArea = max(0f, interMaxX - interMinX) * max(0f, interMaxY - interMinY)
        val aArea = (a[2] - a[0]) * (a[3] - a[1])
        val bArea = (b[2] - b[0]) * (b[3] - b[1])
        val union = aArea + bArea - interArea
        return if (union > 0f) interArea / union else 0f
    }

    private fun Float.fmt(decimals: Int = 3) = if (this < 0f) "-1" else "%.${decimals}f".format(this)
    @JvmName("averageFloat")
    private fun Collection<Float>.average() = if (isEmpty()) -1.0 else sumOf { it.toDouble() } / size
    @JvmName("averageLong")
    private fun Collection<Long>.average()  = if (isEmpty()) -1L  else (sumOf { it } / size)
    @JvmName("averageBoolean")
    private fun Collection<Boolean>.average() = if (isEmpty()) -1.0 else count { it }.toDouble() / size
}
