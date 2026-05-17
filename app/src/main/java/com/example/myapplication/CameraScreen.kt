package com.example.myapplication

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
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
import com.example.myapplication.detection.OnnxSessionManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asExecutor
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors
import kotlin.math.max

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

    val imageCapture = remember {
        ImageCapture.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_16_9)
            .build()
    }
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

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
                            // Update the tracked image resolution to map coordinates later
                            val rotation = imageProxy.imageInfo.rotationDegrees
                            val isPortrait = rotation == 90 || rotation == 270
                            val w = if (isPortrait) imageProxy.height else imageProxy.width
                            val h = if (isPortrait) imageProxy.width else imageProxy.height
                            imageBounds = android.util.Size(w, h)

                            processImageProxy(imageProxy) { points ->
                                detectedPoints = points
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

        // Shutter Button
        Button(
            onClick = {
                imageCapture.takePicture(
                    ContextCompat.getMainExecutor(context),
                    object : ImageCapture.OnImageCapturedCallback() {
                        override fun onCaptureSuccess(image: ImageProxy) {
                            val bitmap = image.toBitmap()
                            val rotation = image.imageInfo.rotationDegrees
                            image.close()

                            // Rotate bitmap so the user sees it upright
                            val matrix = android.graphics.Matrix().apply {
                                postRotate(rotation.toFloat())
                            }
                            val rotatedBitmap = Bitmap.createBitmap(
                                bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true
                            )

                            // Hybrid detection on the ACTUAL captured bitmap:
                            //   Step 1 — OpenCV (fast baseline, runs in C++)
                            //   Step 2 — ONNX via DocQuadNet256 (ML refinement)
                            //   Step 3 — HybridDetectionCoordinator picks the best result
                            coroutineScope.launch(Dispatchers.Default) {
                                // Step 1: OpenCV on captured bitmap
                                val pointsJson = DocumentProcessor.nativeScanJSON(
                                    srcBitmap = rotatedBitmap,
                                    shrunkImageHeight = 500,
                                    imageRotation = 0, // already rotated
                                    scale = 1.0,
                                    options = "{}"
                                )
                                val opencvPoints = parseAndNormalizePoints(
                                    pointsJson, rotatedBitmap.width, rotatedBitmap.height
                                )

                                // Step 2 + 3: Hybrid decision (ONNX if ready, else OpenCV fallback)
                                val finalPoints: List<PointF>? = try {
                                    val sessionMgr = OnnxSessionManager.getInstance(context)
                                    if (sessionMgr.isReady()) {
                                        val detector = DocQuadNetDetector(sessionMgr)
                                        val coordinator = HybridDetectionCoordinator(detector)
                                        val (quad, method) = coordinator.detect(rotatedBitmap, opencvPoints)
                                        // Convert NormalizedQuad back to List<PointF> for CroppingScreen
                                        listOf(
                                            PointF(quad.topLeft.x,     quad.topLeft.y),
                                            PointF(quad.topRight.x,    quad.topRight.y),
                                            PointF(quad.bottomRight.x, quad.bottomRight.y),
                                            PointF(quad.bottomLeft.x,  quad.bottomLeft.y)
                                        )
                                    } else {
                                        // ONNX not yet ready (model not downloaded / startup init pending)
                                        android.util.Log.d("CameraScreen", "ONNX not ready — using OPENCV_CONTOUR")
                                        opencvPoints
                                    }
                                } catch (e: Exception) {
                                    android.util.Log.e("CameraScreen", "Hybrid detection error: ${e.message}")
                                    opencvPoints  // graceful fallback
                                }

                                withContext(Dispatchers.Main) {
                                    onPhotoCaptured(rotatedBitmap, finalPoints)
                                }
                            }
                        }
                        override fun onError(exception: ImageCaptureException) {
                            Log.e("CameraScreen", "Photo capture failed", exception)
                        }
                    }
                )
            },
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(bottom = 32.dp),
            shape = CircleShape
        ) {
            Text("Capture", modifier = Modifier.padding(16.dp))
        }
    }
}

// Processes a live ImageAnalysis frame: runs OpenCV and returns pixel-space points.
// The returned points are in the *rotated* image's pixel space (imageBounds).
private fun processImageProxy(imageProxy: ImageProxy, onPointsDetected: (List<PointF>?) -> Unit) {
    val bitmap = imageProxy.toBitmap()
    val rotation = imageProxy.imageInfo.rotationDegrees
    try {
        val pointsJson = DocumentProcessor.nativeScanJSON(
            srcBitmap = bitmap,
            shrunkImageHeight = 500,
            imageRotation = rotation,
            scale = 1.0,
            options = "{}"
        )
        onPointsDetected(parsePixelPoints(pointsJson))
    } catch (e: Exception) {
        Log.e("CameraScreen", "Live frame detection error", e)
        onPointsDetected(null)
    } finally {
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
