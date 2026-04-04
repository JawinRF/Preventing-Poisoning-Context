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
 * On every window/content change: capture nodes -> normalize -> Layer 1 -> Layer 2 -> audit.
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
            notificationTimeout = 750L
            flags = AccessibilityServiceInfo.FLAG_RETRIEVE_INTERACTIVE_WINDOWS
        }
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        val root = rootInActiveWindow ?: return
        scope.launch {
            try {
                val screenCtx = bridge.captureScreenContext() ?: return@launch
                val payload = bridge.buildInspectPayload(screenCtx)
                val rawText = payload.optString("text", "")
                if (rawText.isBlank()) return@launch

                val norm = Normalizer.normalize(rawText)
                val l1 = PrismDetector.scan(norm.text)

                val l2Prob = if (l1.score in 0.2f..0.7f && classifier != null) {
                    classifier!!.classify(norm.text).maliciousProb
                } else {
                    if (l1.verdict == PrismDetector.Verdict.BLOCK) 1.0f else 0.0f
                }

                val verdict = when {
                    l1.verdict == PrismDetector.Verdict.BLOCK -> "BLOCK"
                    l2Prob >= 0.70f -> "BLOCK"
                    else -> "ALLOW"
                }

                MemShieldDb.get(this@PrismAccessibilityService).auditDao().insert(
                    AuditEntry(
                        path = "ui_accessibility",
                        snippet = norm.text.take(120),
                        verdict = verdict,
                        layer1Score = l1.score,
                        layer2Prob = l2Prob,
                        matchedRules = l1.matchedRules.joinToString(",")
                    )
                )
            } catch (_: Exception) {
                // Service must not crash on bad node trees
            } finally {
                root.recycle()
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
