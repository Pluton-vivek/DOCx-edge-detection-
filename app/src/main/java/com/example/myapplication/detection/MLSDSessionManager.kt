package com.example.myapplication.detection

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.channels.FileChannel

private const val TAG = "MLSDSession"

object MLSDSessionManager {

    private const val MODEL_ASSET = "docquad/M-LSD_320_tiny_fp32.tflite"

    @Volatile private var interpreter: Interpreter? = null

    fun initialize(context: Context) {
        if (interpreter != null) return
        try {
            // TFLite requires a MappedByteBuffer (memory-mapped) or a direct ByteBuffer.
            // ByteBuffer.wrap() creates a heap-backed buffer which TFLite rejects.
            // AssetFileDescriptor + FileChannel.map() is the standard Android pattern.
            val assetFd = context.assets.openFd(MODEL_ASSET)
            val inputStream = FileInputStream(assetFd.fileDescriptor)
            val buffer = inputStream.channel.map(
                FileChannel.MapMode.READ_ONLY,
                assetFd.startOffset,
                assetFd.declaredLength
            )
            assetFd.close()

            val options = Interpreter.Options().apply {
                numThreads = 2
            }
            interpreter = Interpreter(buffer, options)
            Log.d(TAG, "M-LSD session ready")
        } catch (e: Exception) {
            Log.e(TAG, "M-LSD init failed: ${e.message}", e)
        }
    }

    fun getInterpreter(): Interpreter? = interpreter

    fun isReady(): Boolean = interpreter != null

    fun close() {
        interpreter?.close()
        interpreter = null
    }
}
