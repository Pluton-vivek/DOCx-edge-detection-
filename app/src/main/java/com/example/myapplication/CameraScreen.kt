package com.example.myapplication

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import com.example.myapplication.detection.DocQuadNetDetector
import com.example.myapplication.detection.HybridDetectionCoordinator
import com.example.myapplication.detection.MLSDDetector
import com.example.myapplication.detection.MLSDSessionManager
import com.example.myapplication.detection.OnnxSessionManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asExecutor
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import android.os.Handler
import android.os.Looper
import java.util.concurrent.Executors
import com.example.myapplication.telemetry.TelemetryCoordinator

// Data class to represent the points from the JSON
data class PointF(val x: Float, val y: Float)

@Composable
fun CameraScreen(
    onPhotoCaptured: (Bitmap, List<PointF>?) -> Unit
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val coroutineScope = androidx.compose.runtime.rememberCoroutineScope()

    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }
    var detectedPoints by remember { mutableStateOf<List<PointF>?>(null) }
    var imageBounds by remember { mutableStateOf(android.util.Size(1, 1)) }
    // Guard: prevents double-tap capture and shows loading state during background detection
    var isCapturing by remember { mutableStateOf(false) }
    var liveFrameCount by remember { mutableIntStateOf(0) }

    val imageCapture = remember {
        ImageCapture.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_16_9)
            .build()
    }
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

    // Photo Picker launcher — no storage permission required (Android 13+ Photo Picker API).
    // On older devices the system falls back to the legacy file picker.
    val galleryLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia()
    ) { uri ->
        if (uri == null || isCapturing) return@rememberLauncherForActivityResult
        isCapturing = true

        coroutineScope.launch(Dispatchers.IO) {
            try {
                // Step 1: Decode with size cap to avoid OOM on 12MP+ photos
                val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                context.contentResolver.openInputStream(uri)?.use {
                    BitmapFactory.decodeStream(it, null, opts)
                }
                var sampleSize = 1
                while (maxOf(opts.outWidth, opts.outHeight) / sampleSize > 2048) {
                    sampleSize *= 2
                }
                val decodeOpts = BitmapFactory.Options().apply {
                    inSampleSize = sampleSize
                    inPreferredConfig = Bitmap.Config.ARGB_8888
                }
                val rawBitmap = context.contentResolver.openInputStream(uri)?.use {
                    BitmapFactory.decodeStream(it, null, decodeOpts)
                } ?: throw IllegalStateException("Could not decode bitmap from URI")

                // Step 2: Apply EXIF rotation so portrait photos are never sideways
                val exifRotation = context.contentResolver.openInputStream(uri)?.use { stream ->
                    val exif = androidx.exifinterface.media.ExifInterface(stream)
                    when (exif.getAttributeInt(
                        androidx.exifinterface.media.ExifInterface.TAG_ORIENTATION,
                        androidx.exifinterface.media.ExifInterface.ORIENTATION_NORMAL
                    )) {
                        androidx.exifinterface.media.ExifInterface.ORIENTATION_ROTATE_90  -> 90f
                        androidx.exifinterface.media.ExifInterface.ORIENTATION_ROTATE_180 -> 180f
                        androidx.exifinterface.media.ExifInterface.ORIENTATION_ROTATE_270 -> 270f
                        else -> 0f
                    }
                } ?: 0f

                val bitmap = if (exifRotation != 0f) {
                    val matrix = android.graphics.Matrix().apply { postRotate(exifRotation) }
                    Bitmap.createBitmap(rawBitmap, 0, 0, rawBitmap.width, rawBitmap.height, matrix, true)
                        .also { if (it !== rawBitmap) rawBitmap.recycle() }
                } else rawBitmap

                // Step 3: Run OpenCV on the gallery bitmap.
                // imageRotation = 0 because EXIF rotation has already been applied above.
                val pointsJson = DocumentProcessor.nativeScanJSON(
                    srcBitmap = bitmap,
                    shrunkImageHeight = 500,
                    imageRotation = 0,
                    scale = 1.0,
                    options = "{}"
                )
                val opencvPoints = parseAndNormalizePoints(
                    pointsJson, bitmap.width, bitmap.height
                )

                // Step 4: Run hybrid detection — identical logic to the camera capture path.
                val finalPoints: List<PointF>? = try {
                    val sessionMgr = OnnxSessionManager.getInstance(context)
                    if (sessionMgr.isReady()) {
                        val detector = DocQuadNetDetector(sessionMgr)
                        val mlsdDetector = if (MLSDSessionManager.isReady()) MLSDDetector() else null
                        val coordinator = HybridDetectionCoordinator(
                            onnxDetector = detector,
                            mlsdDetector = mlsdDetector
                        )
                        val (quad, method) = coordinator.detect(bitmap, opencvPoints)

                        TelemetryCoordinator.reportCapture(
                            quad     = quad,
                            method   = method,
                            onnxConf = quad.confidence,
                            chps     = quad.peakSharpness,
                            onnxMs   = quad.onnxInferenceMs,
                            mlsdMs   = quad.mlsdInferenceMs
                        )
                        TelemetryCoordinator.emitCaptureSummary()

                        listOf(
                            PointF(quad.topLeft.x,     quad.topLeft.y),
                            PointF(quad.topRight.x,    quad.topRight.y),
                            PointF(quad.bottomRight.x, quad.bottomRight.y),
                            PointF(quad.bottomLeft.x,  quad.bottomLeft.y)
                        )
                    } else opencvPoints
                } catch (e: Exception) {
                    android.util.Log.e("GalleryImport", "Hybrid detection error: ${e.message}")
                    opencvPoints
                }

                // Step 5: Hand off to MainActivity via the same callback as camera capture.
                // isCapturing is reset when the AppState transitions to CROPPING.
                withContext(Dispatchers.Main) {
                    onPhotoCaptured(bitmap, finalPoints)
                }
            } catch (e: Exception) {
                android.util.Log.e("GalleryImport", "Gallery processing failed: ${e.message}")
                withContext(Dispatchers.Main) {
                    isCapturing = false
                }
            }
        }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { ctx ->
                // FIT_CENTER: the user sees the FULL camera frame (with letterboxing).
                // This ensures what's visible in the preview matches exactly what
                // ImageCapture will save — no hidden edges.
                val previewView = PreviewView(ctx).apply {
                    scaleType = PreviewView.ScaleType.FIT_CENTER
                }
                val cameraProvider = cameraProviderFuture.get()

                val preview = Preview.Builder()
                    .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                    .build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

                val imageAnalyzer = ImageAnalysis.Builder()
                    .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also {
                        it.setAnalyzer(cameraExecutor) { imageProxy ->
                            // Track rotated frame dimensions for overlay math
                            val rotation = imageProxy.imageInfo.rotationDegrees
                            val isPortrait = rotation == 90 || rotation == 270
                            val w = if (isPortrait) imageProxy.height else imageProxy.width
                            val h = if (isPortrait) imageProxy.width else imageProxy.height

                            processImageProxy(imageProxy) { points ->
                                // BOTH state updates must land on the main thread.
                                Handler(Looper.getMainLooper()).post {
                                    imageBounds = android.util.Size(w, h)
                                    detectedPoints = points
                                    liveFrameCount++
                                    if (liveFrameCount % 30 == 0) {
                                        TelemetryCoordinator.emitLiveSummary(liveFrameCount)
                                    }
                                }
                            }
                        }
                    }

                try {
                    cameraProvider.unbindAll()
                    cameraProvider.bindToLifecycle(
                        lifecycleOwner,
                        CameraSelector.DEFAULT_BACK_CAMERA,
                        preview,
                        imageCapture,
                        imageAnalyzer
                    )
                } catch (exc: Exception) {
                    Log.e("CameraScreen", "Use case binding failed", exc)
                }
                previewView
            }
        )

        // Overlay for drawing detected document boundaries
        detectedPoints?.let { points ->
            Canvas(modifier = Modifier.fillMaxSize()) {
                if (points.size == 4) {
                    val scaleX = size.width / imageBounds.width.toFloat()
                    val scaleY = size.height / imageBounds.height.toFloat()
                    // minOf = FIT math: matches PreviewView.ScaleType.FIT_CENTER.
                    // The overlay never draws outside the visible frame area.
                    val scale = minOf(scaleX, scaleY)
                    val offsetX = (size.width - imageBounds.width * scale) / 2f
                    val offsetY = (size.height - imageBounds.height * scale) / 2f

                    val path = Path().apply {
                        moveTo(offsetX + points[0].x * scale, offsetY + points[0].y * scale)
                        lineTo(offsetX + points[1].x * scale, offsetY + points[1].y * scale)
                        lineTo(offsetX + points[2].x * scale, offsetY + points[2].y * scale)
                        lineTo(offsetX + points[3].x * scale, offsetY + points[3].y * scale)
                        close()
                    }
                    drawPath(
                        path = path,
                        color = Color.Green,
                        style = Stroke(width = 8f)
                    )
                }
            }
        }

        // Bottom control row: Gallery button (left) + Shutter button (right)
        Row(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(bottom = 32.dp),
            horizontalArrangement = Arrangement.spacedBy(24.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Gallery button — launches the system Photo Picker (no storage permission needed)
            Button(
                enabled = !isCapturing,
                onClick = {
                    galleryLauncher.launch(
                        PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                    )
                },
                shape = CircleShape
            ) {
                Text("Gallery", modifier = Modifier.padding(12.dp))
            }

            // Shutter button — unchanged logic
            Button(
                enabled = !isCapturing,
                onClick = {
                    if (isCapturing) return@Button
                    isCapturing = true
                    imageCapture.takePicture(
                        ContextCompat.getMainExecutor(context),
                        object : ImageCapture.OnImageCapturedCallback() {
                            override fun onCaptureSuccess(image: ImageProxy) {
                                val raw = image.toBitmap()
                                val rotation = image.imageInfo.rotationDegrees
                                image.close()

                                // Rotate to upright orientation
                                val matrix = android.graphics.Matrix().apply { postRotate(rotation.toFloat()) }
                                val rotatedBitmap = Bitmap.createBitmap(
                                    raw, 0, 0, raw.width, raw.height, matrix, true
                                )
                                // raw is no longer needed — release to avoid holding two full-res bitmaps
                                if (rotatedBitmap !== raw) raw.recycle()

                                coroutineScope.launch(Dispatchers.Default) {
                                    // Step 1: OpenCV on captured bitmap
                                    val pointsJson = DocumentProcessor.nativeScanJSON(
                                        srcBitmap = rotatedBitmap,
                                        shrunkImageHeight = 500,
                                        imageRotation = 0,
                                        scale = 1.0,
                                        options = "{}"
                                    )
                                    val opencvPoints = parseAndNormalizePoints(
                                        pointsJson, rotatedBitmap.width, rotatedBitmap.height
                                    )

                                    // Step 2 + 3: Hybrid decision
                                    val finalPoints: List<PointF>? = try {
                                        val sessionMgr = OnnxSessionManager.getInstance(context)
                                        if (sessionMgr.isReady()) {
                                            val detector = DocQuadNetDetector(sessionMgr)
                                            val mlsdDetector = if (MLSDSessionManager.isReady()) MLSDDetector() else null
                                            val coordinator = HybridDetectionCoordinator(
                                                onnxDetector = detector,
                                                mlsdDetector = mlsdDetector
                                            )
                                            val (quad, method) = coordinator.detect(rotatedBitmap, opencvPoints)

                                            TelemetryCoordinator.reportCapture(
                                                quad     = quad,
                                                method   = method,
                                                onnxConf = quad.confidence,
                                                chps     = quad.peakSharpness,
                                                onnxMs   = quad.onnxInferenceMs,
                                                mlsdMs   = quad.mlsdInferenceMs
                                            )
                                            TelemetryCoordinator.emitCaptureSummary()

                                            listOf(
                                                PointF(quad.topLeft.x,     quad.topLeft.y),
                                                PointF(quad.topRight.x,    quad.topRight.y),
                                                PointF(quad.bottomRight.x, quad.bottomRight.y),
                                                PointF(quad.bottomLeft.x,  quad.bottomLeft.y)
                                            )
                                        } else {
                                            android.util.Log.d("CameraScreen", "ONNX not ready — using OPENCV_CONTOUR")
                                            opencvPoints
                                        }
                                    } catch (e: Exception) {
                                        android.util.Log.e("CameraScreen", "Hybrid detection error: ${e.message}")
                                        opencvPoints
                                    }

                                    withContext(Dispatchers.Main) {
                                        // isCapturing is reset by the caller (MainActivity/AppState)
                                        // when it transitions to CROPPING state. We do not reset it
                                        // here so the loading overlay stays until navigation completes.
                                        onPhotoCaptured(rotatedBitmap, finalPoints)
                                    }
                                }
                            }
                            override fun onError(exception: ImageCaptureException) {
                                Log.e("CameraScreen", "Photo capture failed", exception)
                                isCapturing = false
                            }
                        }
                    )
                },
                shape = CircleShape
            ) {
                Text(if (isCapturing) "Processing…" else "Capture",
                    modifier = Modifier.padding(16.dp))
            }
        }

        // Loading overlay: shown while hybrid detection runs in the background
        if (isCapturing) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(Color.Black.copy(alpha = 0.35f)),
                contentAlignment = Alignment.Center
            ) {
                CircularProgressIndicator(color = Color.White)
            }
        }
    }
}

// Processes a live ImageAnalysis frame: runs OpenCV, returns pixel-space points.
// Called on cameraExecutor (background thread). Never touch Compose state here.
private fun processImageProxy(imageProxy: ImageProxy, onPointsDetected: (List<PointF>?) -> Unit) {
    val bitmap = imageProxy.toBitmap()
    val rotation = imageProxy.imageInfo.rotationDegrees
    try {
        val cvStart = System.currentTimeMillis()
        val pointsJson = DocumentProcessor.nativeScanJSON(
            srcBitmap = bitmap,
            shrunkImageHeight = 500,
            imageRotation = rotation,
            scale = 1.0,
            options = "{}"
        )
        val cvMs = System.currentTimeMillis() - cvStart
        val points = parsePixelPoints(pointsJson)

        // Report to telemetry (called on cameraExecutor — TelemetryCoordinator is synchronized)
        val isPortrait = rotation == 90 || rotation == 270
        val fw = if (isPortrait) imageProxy.height else imageProxy.width
        val fh = if (isPortrait) imageProxy.width else imageProxy.height
        TelemetryCoordinator.reportLiveFrame(points, fw, fh, cvMs)

        // LiveDetect tag: filter Logcat by 'LiveDetect' to debug detection without noise.
        // Comment out in release builds.
        if (points != null) {
            Log.d("LiveDetect", "✓ found quad " +
                "TL=(${points[0].x.toInt()},${points[0].y.toInt()}) " +
                "TR=(${points[1].x.toInt()},${points[1].y.toInt()}) " +
                "BR=(${points[2].x.toInt()},${points[2].y.toInt()}) " +
                "BL=(${points[3].x.toInt()},${points[3].y.toInt()})")
        } else {
            Log.v("LiveDetect", "✗ no quad found")
        }
        onPointsDetected(points)
    } catch (e: Exception) {
        Log.e("CameraScreen", "Live frame detection error", e)
        onPointsDetected(null)
    } finally {
        bitmap.recycle()   // live frames are decoded fresh every ~100ms — always recycle
        imageProxy.close()
    }
}

// Parses nativeScanJSON output and normalizes pixel coordinates to [0.0, 1.0].
// Used by the capture path where the bitmap is already rotated (imageRotation = 0).
internal fun parseAndNormalizePoints(pointsJson: String?, bitmapWidth: Int, bitmapHeight: Int): List<PointF>? {
    val pixelPoints = parsePixelPoints(pointsJson) ?: return null
    return pixelPoints.map { pt ->
        PointF(pt.x / bitmapWidth.toFloat(), pt.y / bitmapHeight.toFloat())
    }
}

// Parses the raw JSON array of quadrilateral points into pixel-space PointF list.
private fun parsePixelPoints(pointsJson: String?): List<PointF>? {
    if (pointsJson.isNullOrBlank()) return null
    return try {
        val listType = object : TypeToken<List<List<List<Float>>>>() {}.type
        val parsed: List<List<List<Float>>> = Gson().fromJson(pointsJson, listType)
        if (parsed.isNotEmpty() && parsed.first().size == 4) {
            parsed.first().map { PointF(it[0], it[1]) }
        } else null
    } catch (e: Exception) {
        Log.e("CameraScreen", "Error parsing detection JSON", e)
        null
    }
}
