package com.openclaw.android.security

import android.accessibilityservice.AccessibilityService
import android.graphics.Rect
import android.os.SystemClock
import android.view.accessibility.AccessibilityNodeInfo
import android.view.accessibility.AccessibilityWindowInfo
import org.json.JSONArray
import org.json.JSONObject

class WindowContextBridge(private val service: AccessibilityService) {

    companion object {
        const val EMIT_THROTTLE_MS = 750L
    }

    private var lastCaptureElapsedMs = 0L

    fun shouldCapture(nowElapsedMs: Long = SystemClock.elapsedRealtime()): Boolean {
        if (nowElapsedMs - lastCaptureElapsedMs < EMIT_THROTTLE_MS) return false
        lastCaptureElapsedMs = nowElapsedMs
        return true
    }

    fun captureScreenContext(): JSONObject? {
        val windows = service.windows ?: return null
        val activeWindow = windows.firstOrNull { it.isActive } ?: windows.firstOrNull() ?: return null
        val rootNode = activeWindow.root ?: return null

        return try {
            val visibleNodes = JSONArray()
            collectVisibleNodes(rootNode, visibleNodes, depth = 0, maxDepth = 8)

            JSONObject()
                .put("foreground_package", rootNode.packageName?.toString() ?: "unknown")
                .put("window_id", activeWindow.id)
                .put("window_type", describeWindowType(activeWindow.type))
                .put("screen_width_px", service.resources.displayMetrics.widthPixels)
                .put("screen_height_px", service.resources.displayMetrics.heightPixels)
                .put("visible_nodes", visibleNodes)
        } finally {
            rootNode.recycle()
        }
    }

    fun buildInspectPayload(screenContext: JSONObject): JSONObject {
        val packageName = screenContext.optString("foreground_package", "unknown")
        val now = System.currentTimeMillis()
        return JSONObject()
            .put("entry_id", "android-ui-$now")
            .put("text", JSONObject().put("nodes", screenContext.optJSONArray("visible_nodes") ?: JSONArray()).toString())
            .put("ingestion_path", "ui_accessibility")
            .put("source_type", "accessibility")
            .put("source_name", packageName)
            .put("session_id", "android-session")
            .put("run_id", "android-run-$now")
            .put("metadata", JSONObject()
                .put("foreground_package", packageName)
                .put("screen_context", screenContext))
    }

    private fun collectVisibleNodes(node: AccessibilityNodeInfo, out: JSONArray, depth: Int, maxDepth: Int) {
        if (depth > maxDepth || !node.isVisibleToUser) return

        val bounds = Rect()
        node.getBoundsInScreen(bounds)
        val width = bounds.right - bounds.left
        val height = bounds.bottom - bounds.top
        if (bounds.left < 0 || bounds.top < 0 || width <= 0 || height <= 0 || width * height < 50) return

        val text = node.text?.toString()?.trim().orEmpty()
        val contentDescription = node.contentDescription?.toString()?.trim().orEmpty()
        if (text.isNotEmpty() || contentDescription.isNotEmpty()) {
            out.put(JSONObject()
                .put("resource_id", node.viewIdResourceName ?: "")
                .put("class", node.className?.toString() ?: "")
                .put("text", text)
                .put("content_desc", contentDescription)
                .put("bounds_px", JSONArray().put(bounds.left).put(bounds.top).put(bounds.right).put(bounds.bottom)))
        }

        for (index in 0 until node.childCount) {
            val child = node.getChild(index) ?: continue
            try { collectVisibleNodes(child, out, depth + 1, maxDepth) } finally { child.recycle() }
        }
    }

    private fun describeWindowType(type: Int): String = when (type) {
        AccessibilityWindowInfo.TYPE_APPLICATION -> "application"
        AccessibilityWindowInfo.TYPE_INPUT_METHOD -> "input_method"
        AccessibilityWindowInfo.TYPE_SYSTEM -> "system"
        AccessibilityWindowInfo.TYPE_ACCESSIBILITY_OVERLAY -> "accessibility_overlay"
        AccessibilityWindowInfo.TYPE_SPLIT_SCREEN_DIVIDER -> "split_screen_divider"
        else -> "unknown"
    }
}
