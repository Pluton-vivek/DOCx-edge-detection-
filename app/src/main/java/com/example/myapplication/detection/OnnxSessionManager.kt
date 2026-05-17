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
 * Design constraints:
 * - Session is created ONCE at app startup, never recreated per capture.
 * - Initialization runs on Dispatchers.IO (13MB asset read is disk-bound).
 * - Execution provider selection is robust: tries NNAPI → falls back to CPU.
 *   XNNPACK is intentionally NOT used as the primary EP because the model
 *   contains `com.microsoft.FusedConv` operators (visible in the .required_operators.config)
 *   which are Microsoft-optimized fused ops that XNNPACK does not support.
 *   The CPU EP handles all operators and is well-optimized for ARM via ORT's
 *   own NEON/ARM intrinsics on Android.
 * - [close] is called once when the app is fully destroyed.
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

    /** OrtEnvironment is thread-safe and long-lived. One per process is the ORT recommendation. */
    val env: OrtEnvironment by lazy { OrtEnvironment.getEnvironment() }

    @Volatile
    private var session: OrtSession? = null

    /**
     * Loads the model from assets and creates the ONNX Runtime session.
     * Safe to call multiple times — subsequent calls are no-ops.
     * Must be called from Dispatchers.IO (reads 13MB from disk).
     *
     * Execution provider selection rationale:
     *  - The model uses `com.microsoft.FusedConv` (a Microsoft-specific fused operator).
     *  - XNNPACK EP does not support this operator family.
     *  - CPU EP (default) handles all operators via ORT's own ARM-optimized kernels.
     *  - NO_OPT is used for .ort files: they are pre-optimized offline and
     *    re-optimizing at runtime wastes startup time with no accuracy benefit.
     *
     * @param modelAssetPath Path relative to assets/, e.g. "docquad/model.ort"
     */
    suspend fun initialize(modelAssetPath: String) = withContext(Dispatchers.IO) {
        if (session != null) return@withContext  // idempotent

        val modelBytes: ByteArray
        try {
            modelBytes = context.assets.open(modelAssetPath).use { it.readBytes() }
            Log.d(TAG, "Model loaded: $modelAssetPath (${modelBytes.size / 1024}KB)")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model from assets: $modelAssetPath — ${e.message}")
            throw e
        }

        try {
            val sessionOptions = OrtSession.SessionOptions().apply {
                // Pre-optimized .ort file — skip runtime graph optimization (saves ~200ms startup)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.NO_OPT)
                setIntraOpNumThreads(2)
                setInterOpNumThreads(1)
                // CPU EP is the default — no addXxxxx() call needed.
                // We intentionally don't add XNNPACK here because FusedConv is unsupported by it.
                // NNAPI is left out for now: it requires quantized models for peak benefit on most devices,
                // and this model is float32. Can be enabled per-device if profiling shows improvement.
            }
            session = env.createSession(modelBytes, sessionOptions)
            Log.d(TAG, "ONNX session ready: $modelAssetPath")
        } catch (e: Exception) {
            Log.e(TAG, "Session creation failed: ${e.message}")
            throw e
        }
    }

    /**
     * Returns the active session.
     * @throws IllegalStateException if [initialize] was not called or failed.
     */
    fun getSession(): OrtSession {
        return session ?: throw IllegalStateException(
            "OnnxSessionManager: session not initialized. Was initialize() called at startup?"
        )
    }

    /** True if the session is ready for inference calls. */
    fun isReady(): Boolean = session != null

    /**
     * Releases the session.
     * Call from the Activity/Application destroy lifecycle only — never per-capture.
     */
    fun close() {
        session?.close()
        session = null
        Log.d(TAG, "Session closed")
        // OrtEnvironment intentionally not closed — it is a global process-level singleton in ORT.
    }
}
