package com.example.myapplication.detection

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

private const val TAG = "MLSDSession"

object MLSDSessionManager {

    private const val MODEL_ASSET = "mlsd/M-LSD_320_tiny_fp32.tflite"

    @Volatile private var interpreter: Interpreter? = null

    fun initialize(context: Context) {
        if (interpreter != null) return
        try {
            val bytes = context.assets.open(MODEL_ASSET).readBytes()
            val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.nativeOrder())
            val options = Interpreter.Options().apply {
                numThreads = 2
            }
            interpreter = Interpreter(buffer, options)
            Log.d(TAG, "M-LSD session ready")
        } catch (e: Exception) {
            Log.e(TAG, "M-LSD init failed: ${e.message}")
        }
    }

    fun getInterpreter(): Interpreter? = interpreter

    fun isReady(): Boolean = interpreter != null

    fun close() {
        interpreter?.close()
        interpreter = null
    }
}
