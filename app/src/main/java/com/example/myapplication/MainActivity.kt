package com.example.myapplication

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.example.myapplication.detection.OnnxSessionManager
import com.example.myapplication.ui.theme.MyApplicationTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

enum class AppState {
    CAMERA, CROPPING, RESULT
}

class MainActivity : ComponentActivity() {

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (!isGranted) {
            Toast.makeText(this, "Camera permission required.", Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }

        // Initialize ONNX session in the background — never blocks the main thread.
        // Uses Dispatchers.IO because the init reads ~13MB from assets (disk I/O bound).
        // The capture path checks OnnxSessionManager.isReady() before using it,
        // so the app works correctly even if the model file fails to load.
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                OnnxSessionManager.getInstance(applicationContext)
                    .initialize("docquad/docquadnet256_trained_opset17.ort")
            } catch (e: Exception) {
                Log.w("MainActivity", "ONNX init failed — using OpenCV-only detection: ${e.message}")
            }
        }

        setContent {
            MyApplicationTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    var appState by remember { mutableStateOf(AppState.CAMERA) }
                    
                    var capturedPhoto by remember { mutableStateOf<Bitmap?>(null) }
                    var detectedPoints by remember { mutableStateOf<List<PointF>?>(null) }
                    var finalCroppedBitmap by remember { mutableStateOf<Bitmap?>(null) }

                    when (appState) {
                        AppState.CAMERA -> {
                            CameraScreen(onPhotoCaptured = { bitmap, points ->
                                capturedPhoto = bitmap
                                detectedPoints = points
                                appState = AppState.CROPPING
                            })
                        }
                        AppState.CROPPING -> {
                            capturedPhoto?.let { photo ->
                                CroppingScreen(
                                    imageBitmap = photo,
                                    initialPoints = detectedPoints,
                                    onCropConfirmed = { cropped ->
                                        finalCroppedBitmap = cropped
                                        appState = AppState.RESULT
                                    },
                                    onCancel = {
                                        appState = AppState.CAMERA
                                    }
                                )
                            }
                        }
                        AppState.RESULT -> {
                            finalCroppedBitmap?.let { resultImage ->
                                Image(
                                    bitmap = resultImage.asImageBitmap(),
                                    contentDescription = "Final Cropped Result",
                                    modifier = Modifier.fillMaxSize()
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}