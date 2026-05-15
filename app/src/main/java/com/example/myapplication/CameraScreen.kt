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
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asExecutor
import kotlinx.coroutines.launch
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

    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }
    var detectedPoints by remember { mutableStateOf<List<PointF>?>(null) }
    var imageBounds by remember { mutableStateOf(android.util.Size(1, 1)) }
    
    val imageCapture = remember { ImageCapture.Builder().build() }
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

    Box(modifier = Modifier.fillMaxSize()) {
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { ctx ->
                val previewView = PreviewView(ctx)
                val cameraProvider = cameraProviderFuture.get()

                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

                val imageAnalyzer = ImageAnalysis.Builder()
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
                    val scale = maxOf(scaleX, scaleY)
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
                            
                            // Rotate bitmap so the user sees it upright
                            val matrix = android.graphics.Matrix().apply {
                                postRotate(rotation.toFloat())
                            }
                            val rotatedBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)

                            // Translate detected point pixel coordinates to relative (0.0 - 1.0) coordinates
                            val relativePoints = detectedPoints?.map { pt ->
                                PointF(pt.x / imageBounds.width.toFloat(), pt.y / imageBounds.height.toFloat())
                            }
                            onPhotoCaptured(rotatedBitmap, relativePoints)
                            image.close()
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

// Function to process frames and extract the document boundary coordinates
private fun processImageProxy(imageProxy: ImageProxy, onPointsDetected: (List<PointF>?) -> Unit) {
    val bitmap = imageProxy.toBitmap()
    val rotation = imageProxy.imageInfo.rotationDegrees

    try {
        // Shrunk image height is usually to speed up detection (e.g., 500px).
        // Option string can be an empty JSON object for defaults 
        val pointsJson = DocumentProcessor.nativeScanJSON(
            srcBitmap = bitmap,
            shrunkImageHeight = 500,
            imageRotation = rotation,
            scale = 1.0,
            options = "{}" 
        )

        if (!pointsJson.isNullOrBlank()) {
            val listType = object : TypeToken<List<List<List<Float>>>>() {}.type
            // Typical output is an array of quadrilaterals: [ [ [x,y], [x,y], ... ] ]
            val parsed: List<List<List<Float>>> = Gson().fromJson(pointsJson, listType)
            
            if (parsed.isNotEmpty() && parsed.first().size == 4) {
                // Map the JSON result back to compose Canvas Points
                val points = parsed.first().map { PointF(it[0], it[1]) }
                onPointsDetected(points)
            } else {
                onPointsDetected(null)
            }
        } else {
            onPointsDetected(null)
        }
    } catch (e: Exception) {
        Log.e("CameraScreen", "Error parsing points", e)
        onPointsDetected(null)
    } finally {
        imageProxy.close()
    }
}
