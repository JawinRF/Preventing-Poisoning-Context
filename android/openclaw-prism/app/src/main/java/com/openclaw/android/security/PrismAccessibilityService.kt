package com.openclaw.android.security

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.AccessibilityServiceInfo
import android.view.accessibility.AccessibilityEvent
import com.openclaw.android.AppLogger
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch

/**
 * Accessibility service that wires WindowContextBridge into the PRISM pipeline.
 * On every window/content change: capture nodes -> normalize -> Layer 2 -> audit.
 * Throttled to 750ms.
 */
class PrismAccessibilityService : AccessibilityService() {

    companion object {
        private const val TAG = "PrismAccessibility"

        /** Singleton reference for sidecar to query UI integrity checks. */
        @Volatile
        var instance: PrismAccessibilityService? = null
            private set
    }

    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private var classifier: OnnxClassifier? = null
    private lateinit var bridge: WindowContextBridge
    lateinit var uiIntegrity: UiIntegrityChecker
        private set

    override fun onServiceConnected() {
        super.onServiceConnected()
        try {
            classifier = OnnxClassifier(this)
        } catch (e: Exception) {
            AppLogger.w(TAG, "ONNX classifier unavailable: ${e.message}")
        }
        bridge = WindowContextBridge(this)
        uiIntegrity = UiIntegrityChecker(this)
        // Publish singleton AFTER all fields are initialized — prevents
        // sidecar from hitting uninitialized uiIntegrity on early requests.
        instance = this

        serviceInfo = serviceInfo.apply {
            eventTypes = AccessibilityEvent.TYPE_WINDOW_CONTENT_CHANGED or
                    AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED
            feedbackType = AccessibilityServiceInfo.FEEDBACK_GENERIC
            // 2000ms instead of 750ms — reduces ONNX inference frequency on the
            // emulator's 2-core CPU, preventing ANR in foreground apps.
            notificationTimeout = 2000L
            flags = AccessibilityServiceInfo.FLAG_RETRIEVE_INTERACTIVE_WINDOWS
        }
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        scope.launch {
            try {
                // captureScreenContext() acquires its own window root internally.
                val screenCtx = bridge.captureScreenContext() ?: return@launch

                // Skip our own UI — no point scanning OpenClaw's dashboard.
                val foregroundPkg = screenCtx.optString("foreground_package", "")
                if (foregroundPkg == packageName) return@launch

                // Skip captures with no visible text nodes — the ONNX model
                // scores empty JSON near 1.0 (out-of-distribution input).
                val visibleNodes = screenCtx.optJSONArray("visible_nodes")
                if (visibleNodes == null || visibleNodes.length() == 0) return@launch

                val payload = bridge.buildInspectPayload(screenCtx)
                val rawText = payload.optString("text", "")
                if (rawText.isBlank()) return@launch

                val norm = Normalizer.normalize(rawText)
                val l1 = PrismDetector.scan(norm.text)

                // L2 ONNX skipped here — the host Python shield (:8765) re-scans
                // every field that reaches the agent anyway. Running ONNX on every
                // accessibility event saturates the emulator CPU and causes ANR.
                // On-device L1 rule check is sufficient for the audit log.
                val verdict = if (l1.score >= 0.80f) "BLOCK" else "ALLOW"

                MemShieldDb.get(this@PrismAccessibilityService).auditDao().insert(
                    AuditEntry(
                        path = "ui_accessibility",
                        snippet = norm.text.take(120),
                        verdict = verdict,
                        layer1Score = l1.score,
                        layer2Prob = 0.0f,
                        matchedRules = l1.matchedRules.joinToString(",")
                    )
                )
            } catch (_: Exception) {
                // Service must not crash on bad node trees
            }
        }
    }

    override fun onInterrupt() = Unit

    override fun onDestroy() {
        super.onDestroy()
        instance = null
        scope.cancel()
        classifier?.close()
    }
}
