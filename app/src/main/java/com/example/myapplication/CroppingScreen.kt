package com.example.myapplication

import android.graphics.Bitmap
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.unit.dp
import kotlin.math.pow
import kotlin.math.sqrt

@Composable
fun CroppingScreen(
    imageBitmap: Bitmap,
    initialPoints: List<PointF>?,
    onCropConfirmed: (Bitmap) -> Unit,
    onCancel: () -> Unit
) {
    // Basic setup for the 4 corners. If none detected, place them in a default rectangle
    var p1 by remember { mutableStateOf(Offset(100f, 100f)) }
    var p2 by remember { mutableStateOf(Offset(900f, 100f)) }
    var p3 by remember { mutableStateOf(Offset(900f, 1200f)) }
    var p4 by remember { mutableStateOf(Offset(100f, 1200f)) }

    var containerSize by remember { mutableStateOf(androidx.compose.ui.geometry.Size.Zero) }

    // Once we know the container size, layout the points using aspect ratio math
    LaunchedEffect(containerSize, initialPoints) {
        if (containerSize.width > 0 && containerSize.height > 0) {
            val imgRatio = imageBitmap.width.toFloat() / imageBitmap.height.toFloat()
            val boxRatio = containerSize.width / containerSize.height

            var renderWidth = containerSize.width
            var renderHeight = containerSize.height
            var left = 0f
            var top = 0f

            if (imgRatio > boxRatio) {
                renderHeight = containerSize.width / imgRatio
                top = (containerSize.height - renderHeight) / 2f
            } else {
                renderWidth = containerSize.height * imgRatio
                left = (containerSize.width - renderWidth) / 2f
            }

            if (initialPoints != null && initialPoints.size == 4) {
                // initialPoints are 0.0 to 1.0 relative coordinates
                p1 = Offset(left + initialPoints[0].x * renderWidth, top + initialPoints[0].y * renderHeight)
                p2 = Offset(left + initialPoints[1].x * renderWidth, top + initialPoints[1].y * renderHeight)
                p3 = Offset(left + initialPoints[2].x * renderWidth, top + initialPoints[2].y * renderHeight)
                p4 = Offset(left + initialPoints[3].x * renderWidth, top + initialPoints[3].y * renderHeight)
            } else {
                // Default rectangle in interior
                p1 = Offset(left + renderWidth * 0.1f, top + renderHeight * 0.1f)
                p2 = Offset(left + renderWidth * 0.9f, top + renderHeight * 0.1f)
                p3 = Offset(left + renderWidth * 0.9f, top + renderHeight * 0.9f)
                p4 = Offset(left + renderWidth * 0.1f, top + renderHeight * 0.9f)
            }
        }
    }

    Column(modifier = Modifier.fillMaxSize().background(Color.Black)) {
        Box(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                .onGloballyPositioned { coords ->
                    containerSize = androidx.compose.ui.geometry.Size(
                        coords.size.width.toFloat(),
                        coords.size.height.toFloat()
                    )
                }
        ) {
            // 1. Draw the underlying captured photo
            Image(
                bitmap = imageBitmap.asImageBitmap(),
                contentDescription = "Captured Document",
                modifier = Modifier.fillMaxSize()
            )

            // 2. Draw the interactive crop polygon overlay
            Canvas(
                modifier = Modifier
                    .fillMaxSize()
                    .pointerInput(Unit) {
                        detectDragGestures { change, dragAmount ->
                            change.consume()
                            val touchPoint = change.position
                            // Find whichever of the 4 corners is closest to the user's finger and move it
                            val dragRadius = 150f // Allow dragging somewhat loosely near the point
                            when {
                                distance(touchPoint, p1) < dragRadius -> p1 += dragAmount
                                distance(touchPoint, p2) < dragRadius -> p2 += dragAmount
                                distance(touchPoint, p3) < dragRadius -> p3 += dragAmount
                                distance(touchPoint, p4) < dragRadius -> p4 += dragAmount
                            }
                        }
                    }
            ) {
                // Connecting lines
                val path = Path().apply {
                    moveTo(p1.x, p1.y)
                    lineTo(p2.x, p2.y)
                    lineTo(p3.x, p3.y)
                    lineTo(p4.x, p4.y)
                    close()
                }
                drawPath(path = path, color = Color.White, style = Stroke(width = 6f))

                // Corner dots
                drawCircle(color = Color.Green, radius = 24f, center = p1)
                drawCircle(color = Color.Green, radius = 24f, center = p2)
                drawCircle(color = Color.Green, radius = 24f, center = p3)
                drawCircle(color = Color.Green, radius = 24f, center = p4)
            }
        }

        // Action Buttons Row
        Row(
            modifier = Modifier.fillMaxWidth().padding(16.dp),
            horizontalArrangement = Arrangement.SpaceEvenly
        ) {
            Button(onClick = onCancel, colors = ButtonDefaults.buttonColors(containerColor = Color.Red)) {
                Text("Retake")
            }
            Button(onClick = {
                val imgRatio = imageBitmap.width.toFloat() / imageBitmap.height.toFloat()
                val boxRatio = containerSize.width / containerSize.height
                
                var renderWidth = containerSize.width
                var renderHeight = containerSize.height
                var left = 0f
                var top = 0f

                if (imgRatio > boxRatio) {
                    renderHeight = containerSize.width / imgRatio
                    top = (containerSize.height - renderHeight) / 2f
                } else {
                    renderWidth = containerSize.height * imgRatio
                    left = (containerSize.width - renderWidth) / 2f
                }

                // Map visual points back to image pixel points
                fun mapBack(p: Offset): Pair<Float, Float> {
                    val relX = (p.x - left) / renderWidth
                    val relY = (p.y - top) / renderHeight
                    return Pair(relX * imageBitmap.width, relY * imageBitmap.height)
                }

                val (ix1, iy1) = mapBack(p1)
                val (ix2, iy2) = mapBack(p2)
                val (ix3, iy3) = mapBack(p3)
                val (ix4, iy4) = mapBack(p4)

                // The C++ nativeCrop expects a JSON array of array of doubles: [ [ [x,y], [x,y], [x,y], [x,y] ] ]
                val jsonPoints = """
                    [[
                        [${ix1}, ${iy1}],
                        [${ix2}, ${iy2}],
                        [${ix3}, ${iy3}],
                        [${ix4}, ${iy4}]
                    ]]
                """.trimIndent()
                
                // Create an empty bitmap matching exactly the dimensions we want. We'll use 800x1100 as a standard document ratio output
                val outBitmap = Bitmap.createBitmap(800, 1100, Bitmap.Config.ARGB_8888)
                
                DocumentProcessor.nativeCrop(
                    srcBitmap = imageBitmap,
                    points = jsonPoints,
                    transforms = "",
                    outBitmap = outBitmap
                )
                onCropConfirmed(outBitmap)
            }) {
                Text("Confirm & Crop")
            }
        }
    }
}

private fun distance(o1: Offset, o2: Offset): Float {
    return sqrt((o1.x - o2.x).pow(2) + (o1.y - o2.y).pow(2))
}