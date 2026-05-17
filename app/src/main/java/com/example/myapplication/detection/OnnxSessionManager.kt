package com.example.myapplication.detection

import android.content.Context
import android.util.Log
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Singleton that owns the ONNX Runtime environment and the DocQuadNet256 session.
 *
 * Design constraints (from TRD):
 * - Session is created ONCE at first use, never recreated per capture.
 * - Initialization runs on a background thread (never main thread).
 * - XNNPACK execution provider for float32 models on ARM.
 * - [close] is called once when the app is fully destroyed.
 *
 * Usage:
 *   val mgr = OnnxSessionManager.getInstance(context)
 *   mgr.initialize("docquad/DocQuadNet256.onnx")   // call once at app startup
 *   val session = mgr.getSession()                 // subsequent calls are instant
 */
class OnnxSessionManager private constructor(private val context: Context) {

    companion object {
        private const val TAG = "OnnxSessionManager"

        @Volatile
        private var INSTANCE: OnnxSessionManager? = null

        fun getInstance(context: Context): OnnxSessionManager {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: OnnxSessionManager(context.applicationContext).also { INSTANCE = it }
            }
        }
    }

    /** OrtEnvironment is thread-safe and long-lived. Created lazily on first access. */
    val env: OrtEnvironment by lazy { OrtEnvironment.getEnvironment() }

    @Volatile
    private var session: OrtSession? = null

    /**
     * Initialises the ONNX session from an asset file.
     * Safe to call multiple times — subsequent calls are no-ops.
     * Must be called from a background coroutine.
     *
     * @param modelAssetPath Path relative to the assets directory, e.g. "docquad/DocQuadNet256.onnx"
     */
    suspend fun initialize(modelAssetPath: String) = withContext(Dispatchers.Default) {
        if (session != null) return@withContext  // already initialized

        try {
            val modelBytes = context.assets.open(modelAssetPath).use { it.readBytes() }

            val sessionOptions = OrtSession.SessionOptions().apply {
                // XNNPACK: best EP for float32 models on Android (ARM SIMD optimized)
                addXnnpack(emptyMap())
                setIntraOpNumThreads(2)   // 2 threads is the sweet spot on mobile
                setInterOpNumThreads(1)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }

            session = env.createSession(modelBytes, sessionOptions)
            Log.d(TAG, "Session initialized successfully for: $modelAssetPath")
        } catch (e: Exception) {
            Log.e(TAG, "Session initialization failed for: $modelAssetPath — ${e.message}")
            throw e  // propagate so the caller can decide to disable ONNX
        }
    }

    /**
     * Returns the active session. Throws if [initialize] has not been called
     * or if initialization failed.
     */
    fun getSession(): OrtSession {
        return session ?: throw IllegalStateException(
            "OnnxSessionManager not initialized. Call initialize() at app startup."
        )
    }

    /**
     * Returns true if the session is ready for inference.
     * Use this as a guard before calling [getSession] in the detection path.
     */
    fun isReady(): Boolean = session != null

    /**
     * Releases the ONNX session and environment.
     * Call from Application.onTerminate() or equivalent lifecycle point.
     * Never call per-capture.
     */
    fun close() {
        session?.close()
        session = null
        // Note: do not close env — OrtEnvironment is a global singleton in ORT
        Log.d(TAG, "Session closed")
    }
}
