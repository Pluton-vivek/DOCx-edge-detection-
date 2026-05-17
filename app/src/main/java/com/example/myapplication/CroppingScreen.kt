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
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.clipPath
import androidx.compose.ui.graphics.drawscope.scale
import androidx.compose.ui.graphics.drawscope.translate
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
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
    val composeImageBitmap = remember(imageBitmap) { imageBitmap.asImageBitmap() }

    var p1 by remember { mutableStateOf(Offset(100f, 100f)) }
    var p2 by remember { mutableStateOf(Offset(900f, 100f)) }
    var p3 by remember { mutableStateOf(Offset(900f, 1200f)) }
    var p4 by remember { mutableStateOf(Offset(100f, 1200f)) }

    var containerSize by remember { mutableStateOf(androidx.compose.ui.geometry.Size.Zero) }
    var draggingPointIndex by remember { mutableIntStateOf(-1) }

    val imgRatio = imageBitmap.width.toFloat() / imageBitmap.height.toFloat()
    val boxRatio = if (containerSize.height > 0) containerSize.width / containerSize.height else 0f

    var renderWidth = containerSize.width
    var renderHeight = containerSize.height
    var left = 0f
    var top = 0f

    if (containerSize.width > 0 && containerSize.height > 0) {
        if (imgRatio > boxRatio) {
            renderHeight = containerSize.width / imgRatio
            top = (containerSize.height - renderHeight) / 2f
        } else {
            renderWidth = containerSize.height * imgRatio
            left = (containerSize.width - renderWidth) / 2f
        }
    }

    LaunchedEffect(containerSize, initialPoints) {
        if (containerSize.width > 0 && containerSize.height > 0) {
            if (initialPoints != null && initialPoints.size == 4) {
                p1 = Offset(left + initialPoints[0].x * renderWidth, top + initialPoints[0].y * renderHeight)
                p2 = Offset(left + initialPoints[1].x * renderWidth, top + initialPoints[1].y * renderHeight)
                p3 = Offset(left + initialPoints[2].x * renderWidth, top + initialPoints[2].y * renderHeight)
                p4 = Offset(left + initialPoints[3].x * renderWidth, top + initialPoints[3].y * renderHeight)
            } else {
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
            Image(
                bitmap = composeImageBitmap,
                contentDescription = "Captured Document",
                modifier = Modifier.fillMaxSize()
            )

            Canvas(
                modifier = Modifier
                    .fillMaxSize()
                    .pointerInput(Unit) {
                        detectDragGestures(
                            onDragStart = { touchPoint ->
                                val dragRadius = 150f
                                var minDest = Float.MAX_VALUE
                                var minIdx = -1
                                val points = listOf(p1, p2, p3, p4)
                                for (i in 0..3) {
                                    val d = distance(touchPoint, points[i])
                                    if (d < dragRadius && d < minDest) {
                                        minDest = d
                                        minIdx = i
                                    }
                                }
                                draggingPointIndex = minIdx
                            },
                            onDragEnd = { draggingPointIndex = -1 },
                            onDragCancel = { draggingPointIndex = -1 },
                            onDrag = { change, dragAmount ->
                                change.consume()
                                when (draggingPointIndex) {
                                    0 -> p1 += dragAmount
                                    1 -> p2 += dragAmount
                                    2 -> p3 += dragAmount
                                    3 -> p4 += dragAmount
                                }
                            }
                        )
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

                // Magnifier
                if (draggingPointIndex != -1) {
                    val dp = when (draggingPointIndex) {
                        0 -> p1
                        1 -> p2
                        2 -> p3
                        else -> p4
                    }
                    val magnifierRadius = 160f
                    // Position magnifier offset from the finger so it's visible
                    var magCenterY = dp.y - 300f
                    if (magCenterY < magnifierRadius) magCenterY = dp.y + 300f // go below if too high
                    
                    var magCenterX = dp.x
                    if (magCenterX < magnifierRadius) magCenterX = magnifierRadius
                    if (magCenterX > containerSize.width - magnifierRadius) magCenterX = containerSize.width - magnifierRadius

                    val magnifierCenter = Offset(magCenterX, magCenterY)
                    val zoom = 2.0f

                    // Magnifier outline
                    drawCircle(
                        color = Color.White,
                        radius = magnifierRadius + 8f,
                        center = magnifierCenter
                    )
                    drawCircle(
                        color = Color.Black,
                        radius = magnifierRadius,
                        center = magnifierCenter
                    )

                    clipPath(Path().apply { addOval(Rect(magnifierCenter, magnifierRadius)) }) {
                        translate(
                            left = magnifierCenter.x - dp.x * zoom,
                            top = magnifierCenter.y - dp.y * zoom
                        ) {
                            scale(zoom, zoom, pivot = Offset.Zero) {
                                drawImage(
                                    image = composeImageBitmap,
                                    dstOffset = IntOffset(left.toInt(), top.toInt()),
                                    dstSize = IntSize(renderWidth.toInt(), renderHeight.toInt())
                                )
                                drawPath(path = path, color = Color.White, style = Stroke(width = 6f / zoom))
                                
                                drawCircle(color = Color.Green, radius = 24f / zoom, center = p1)
                                drawCircle(color = Color.Green, radius = 24f / zoom, center = p2)
                                drawCircle(color = Color.Green, radius = 24f / zoom, center = p3)
                                drawCircle(color = Color.Green, radius = 24f / zoom, center = p4)
                            }
                        }
                    }

                    // Crosshair
                    drawLine(
                        color = Color.Red,
                        start = Offset(magnifierCenter.x - 20f, magnifierCenter.y),
                        end = Offset(magnifierCenter.x + 20f, magnifierCenter.y),
                        strokeWidth = 4f
                    )
                    drawLine(
                        color = Color.Red,
                        start = Offset(magnifierCenter.x, magnifierCenter.y - 20f),
                        end = Offset(magnifierCenter.x, magnifierCenter.y + 20f),
                        strokeWidth = 4f
                    )
                }
            }
        }

        Row(
            modifier = Modifier.fillMaxWidth().padding(16.dp),
            horizontalArrangement = Arrangement.SpaceEvenly
        ) {
            Button(onClick = onCancel, colors = ButtonDefaults.buttonColors(containerColor = Color.Red)) {
                Text("Retake")
            }
            Button(onClick = {
                fun mapBack(p: Offset): Pair<Float, Float> {
                    val relX = (p.x - left) / renderWidth
                    val relY = (p.y - top) / renderHeight
                    return Pair(relX * imageBitmap.width, relY * imageBitmap.height)
                }

                val (ix1, iy1) = mapBack(p1)
                val (ix2, iy2) = mapBack(p2)
                val (ix3, iy3) = mapBack(p3)
                val (ix4, iy4) = mapBack(p4)

                val jsonPoints = """
                    [[
                        [${ix1}, ${iy1}],
                        [${ix2}, ${iy2}],
                        [${ix3}, ${iy3}],
                        [${ix4}, ${iy4}]
                    ]]
                """.trimIndent()
                
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