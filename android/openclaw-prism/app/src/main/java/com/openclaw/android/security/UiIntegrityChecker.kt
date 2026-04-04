package com.openclaw.android.security

import android.accessibilityservice.AccessibilityService
import android.graphics.Rect
import android.view.accessibility.AccessibilityNodeInfo
import android.view.accessibility.AccessibilityWindowInfo
import com.openclaw.android.AppLogger
import org.json.JSONArray
import org.json.JSONObject

/**
 * Deterministic UI integrity checks for the PRISM tap-safety layer.
 *
 * Replaces VLM-based visual grounding with OS-level signals:
 *   - Foreground package/activity verification
 *   - Overlay / obscuration window detection
 *   - Target node existence, bounds, enabled/clickable state
 *   - Dual-snapshot stability (same node in two rapid tree captures)
 *
 * Design rationale (research-backed):
 *   - ANDROIDWORLD (2024): text-only accessibility tree outperforms screenshot-VLM
 *   - TapTrap (USENIX Security 2025): animation/tapjacking bypasses overlay-focused
 *     defenses; OS-level flags are the correct primitive
 *   - Android security guidance: filterTouchesWhenObscured, FLAG_WINDOW_IS_PARTIALLY_OBSCURED,
 *     accessibilityDataSensitive are the platform-endorsed defenses
 *   - ScreenAI / SeeClick: even specialized UI-vision models achieve only ~53% grounding
 *     accuracy; a tiny general VLM (Moondream2) is not a defensible security boundary
 *
 * Called by the HTTP sidecar (OpenClawService) on behalf of defended_device.py.
 * All methods are synchronous and fast (<50ms).
 */
class UiIntegrityChecker(private val service: AccessibilityService) {

    companion object {
        private const val TAG = "UiIntegrity"
        private const val STABILITY_DELAY_MS = 80L
        private const val BOUNDS_TOLERANCE_PX = 8
        private const val MAX_TREE_DEPTH = 10
    }

    /**
     * Full UI integrity check for a tap target.
     *
     * @param targetText  text attribute of the element to verify (nullable)
     * @param targetDesc  content-description of the element to verify (nullable)
     * @param expectedPkg expected foreground package (nullable — skips check if null)
     * @return JSON object with verdict and details
     */
    fun check(
        targetText: String?,
        targetDesc: String?,
        expectedPkg: String?,
    ): JSONObject {
        val result = JSONObject()
        val checks = JSONArray()
        var blocked = false

        // ── 1. Foreground package verification ──────────────────────────────
        val fgState = getForegroundState()
        result.put("foreground_package", fgState.optString("package", "unknown"))
        result.put("window_type", fgState.optString("window_type", "unknown"))

        if (expectedPkg != null) {
            val actualPkg = fgState.optString("package", "")
            val pkgMatch = actualPkg == expectedPkg
            checks.put(JSONObject().apply {
                put("check", "foreground_package")
                put("pass", pkgMatch)
                put("expected", expectedPkg)
                put("actual", actualPkg)
            })
            if (!pkgMatch) blocked = true
        }

        // ── 2. Overlay / obscuration window detection ───────────────────────
        val overlayInfo = detectOverlays()
        val hasOverlay = overlayInfo.optBoolean("has_suspicious_overlay", false)
        checks.put(JSONObject().apply {
            put("check", "overlay_detection")
            put("pass", !hasOverlay)
            put("overlay_windows", overlayInfo.optJSONArray("overlay_windows") ?: JSONArray())
        })
        if (hasOverlay) blocked = true

        // ── 3. Target node lookup ───────────────────────────────────────────
        if (targetText != null || targetDesc != null) {
            val node1 = findTargetNode(targetText, targetDesc)
            if (node1 == null) {
                checks.put(JSONObject().apply {
                    put("check", "node_exists")
                    put("pass", false)
                    put("reason", "target node not found in accessibility tree")
                })
                blocked = true
            } else {
                val nodeInfo = describeNode(node1)
                val boundsValid = nodeInfo.optBoolean("bounds_valid", false)
                val enabled = nodeInfo.optBoolean("enabled", false)
                val clickable = nodeInfo.optBoolean("clickable", false)
                val visibleToUser = nodeInfo.optBoolean("visible_to_user", false)

                checks.put(JSONObject().apply {
                    put("check", "node_exists")
                    put("pass", true)
                    put("node", nodeInfo)
                })

                // Bounds sanity
                checks.put(JSONObject().apply {
                    put("check", "bounds_valid")
                    put("pass", boundsValid)
                    put("bounds", nodeInfo.optJSONArray("bounds"))
                })
                if (!boundsValid) blocked = true

                // Node must be enabled and visible
                checks.put(JSONObject().apply {
                    put("check", "node_interactable")
                    put("pass", enabled && visibleToUser)
                    put("enabled", enabled)
                    put("clickable", clickable)
                    put("visible_to_user", visibleToUser)
                })
                if (!enabled || !visibleToUser) blocked = true

                // ── 4. Dual-snapshot stability ──────────────────────────────
                node1.recycle()
                Thread.sleep(STABILITY_DELAY_MS)
                val node2 = findTargetNode(targetText, targetDesc)
                if (node2 == null) {
                    checks.put(JSONObject().apply {
                        put("check", "stability")
                        put("pass", false)
                        put("reason", "node disappeared between snapshots")
                    })
                    blocked = true
                } else {
                    val bounds1 = nodeInfo.optJSONArray("bounds")
                    val info2 = describeNode(node2)
                    val bounds2 = info2.optJSONArray("bounds")
                    node2.recycle()

                    val stable = areBoundsStable(bounds1, bounds2)
                    checks.put(JSONObject().apply {
                        put("check", "stability")
                        put("pass", stable)
                        put("bounds_snapshot_1", bounds1)
                        put("bounds_snapshot_2", bounds2)
                    })
                    if (!stable) blocked = true
                }
            }
        }

        result.put("verdict", if (blocked) "BLOCK" else "ALLOW")
        result.put("checks", checks)
        result.put("check_count", checks.length())
        result.put("timestamp", System.currentTimeMillis())
        return result
    }

    // ── Foreground state ────────────────────────────────────────────────────

    private fun getForegroundState(): JSONObject {
        val windows = try { service.windows } catch (_: Exception) { null }
        val activeWindow = windows?.firstOrNull { it.isActive }
            ?: windows?.firstOrNull()

        val root = activeWindow?.root
        val pkg = root?.packageName?.toString() ?: "unknown"
        val windowType = activeWindow?.type?.let { describeWindowType(it) } ?: "unknown"
        root?.recycle()

        return JSONObject().apply {
            put("package", pkg)
            put("window_type", windowType)
            put("window_id", activeWindow?.id ?: -1)
        }
    }

    // ── Overlay detection ───────────────────────────────────────────────────

    private fun detectOverlays(): JSONObject {
        val windows = try { service.windows } catch (_: Exception) { null }
            ?: return JSONObject().put("has_suspicious_overlay", false)

        val overlays = JSONArray()
        var suspicious = false

        for (w in windows) {
            when (w.type) {
                AccessibilityWindowInfo.TYPE_ACCESSIBILITY_OVERLAY -> {
                    overlays.put(describeWindow(w))
                    suspicious = true
                }
                AccessibilityWindowInfo.TYPE_SYSTEM -> {
                    // System windows (status bar, nav bar) are normal.
                    // But if a system window has an unusual root package, flag it.
                    val root = w.root
                    val pkg = root?.packageName?.toString()
                    root?.recycle()
                    if (pkg != null && pkg != "com.android.systemui" && pkg != "com.android.launcher3") {
                        overlays.put(describeWindow(w).put("suspicious_package", pkg))
                        suspicious = true
                    }
                }
                // TYPE_APPLICATION windows that are NOT the active window could be
                // transparent overlays from other apps
                AccessibilityWindowInfo.TYPE_APPLICATION -> {
                    if (!w.isActive && !w.isFocused) {
                        val root = w.root
                        val pkg = root?.packageName?.toString()
                        root?.recycle()
                        // Another app's window is visible alongside the active one
                        val activeRoot = windows.firstOrNull { it.isActive }?.root
                        val activePkg = activeRoot?.packageName?.toString()
                        activeRoot?.recycle()
                        if (pkg != null && pkg != activePkg && pkg != "com.android.systemui") {
                            overlays.put(describeWindow(w).put("background_app", pkg))
                            suspicious = true
                        }
                    }
                }
                else -> {}
            }
        }

        return JSONObject().apply {
            put("has_suspicious_overlay", suspicious)
            put("overlay_windows", overlays)
            put("total_windows", windows.size)
        }
    }

    // ── Node lookup ─────────────────────────────────────────────────────────

    private fun findTargetNode(text: String?, desc: String?): AccessibilityNodeInfo? {
        val root = try {
            service.rootInActiveWindow
        } catch (_: Exception) { null } ?: return null

        val found = findNodeRecursive(root, text, desc, depth = 0)
        if (found == null || found == root) {
            // Don't recycle root if it IS the found node
            if (found == null) root.recycle()
        } else {
            root.recycle()
        }
        return found
    }

    private fun findNodeRecursive(
        node: AccessibilityNodeInfo,
        text: String?,
        desc: String?,
        depth: Int,
    ): AccessibilityNodeInfo? {
        if (depth > MAX_TREE_DEPTH) return null

        val nodeText = node.text?.toString()?.trim() ?: ""
        val nodeDesc = node.contentDescription?.toString()?.trim() ?: ""

        // Exact match on text or content-description
        if (text != null && nodeText.equals(text, ignoreCase = true)) {
            return AccessibilityNodeInfo.obtain(node)
        }
        if (desc != null && nodeDesc.equals(desc, ignoreCase = true)) {
            return AccessibilityNodeInfo.obtain(node)
        }

        for (i in 0 until node.childCount) {
            val child = node.getChild(i) ?: continue
            val found = findNodeRecursive(child, text, desc, depth + 1)
            child.recycle()
            if (found != null) return found
        }
        return null
    }

    // ── Node description ────────────────────────────────────────────────────

    private fun describeNode(node: AccessibilityNodeInfo): JSONObject {
        val bounds = Rect()
        node.getBoundsInScreen(bounds)

        val screenW = service.resources.displayMetrics.widthPixels
        val screenH = service.resources.displayMetrics.heightPixels

        val width = bounds.right - bounds.left
        val height = bounds.bottom - bounds.top
        val boundsValid = bounds.left >= 0 && bounds.top >= 0 &&
                bounds.right <= screenW && bounds.bottom <= screenH &&
                width > 0 && height > 0

        return JSONObject().apply {
            put("text", node.text?.toString() ?: "")
            put("content_desc", node.contentDescription?.toString() ?: "")
            put("class_name", node.className?.toString() ?: "")
            put("resource_id", node.viewIdResourceName ?: "")
            put("bounds", JSONArray().put(bounds.left).put(bounds.top).put(bounds.right).put(bounds.bottom))
            put("bounds_valid", boundsValid)
            put("enabled", node.isEnabled)
            put("clickable", node.isClickable)
            put("visible_to_user", node.isVisibleToUser)
            put("focusable", node.isFocusable)
            put("screen_width", screenW)
            put("screen_height", screenH)
        }
    }

    // ── Bounds stability ────────────────────────────────────────────────────

    private fun areBoundsStable(bounds1: JSONArray?, bounds2: JSONArray?): Boolean {
        if (bounds1 == null || bounds2 == null) return false
        if (bounds1.length() != 4 || bounds2.length() != 4) return false
        for (i in 0 until 4) {
            if (Math.abs(bounds1.getInt(i) - bounds2.getInt(i)) > BOUNDS_TOLERANCE_PX) {
                return false
            }
        }
        return true
    }

    // ── Helpers ─────────────────────────────────────────────────────────────

    private fun describeWindow(w: AccessibilityWindowInfo): JSONObject {
        return JSONObject().apply {
            put("id", w.id)
            put("type", describeWindowType(w.type))
            put("is_active", w.isActive)
            put("is_focused", w.isFocused)
            val root = w.root
            put("package", root?.packageName?.toString() ?: "unknown")
            root?.recycle()
        }
    }

    private fun describeWindowType(type: Int): String = when (type) {
        AccessibilityWindowInfo.TYPE_APPLICATION -> "application"
        AccessibilityWindowInfo.TYPE_INPUT_METHOD -> "input_method"
        AccessibilityWindowInfo.TYPE_SYSTEM -> "system"
        AccessibilityWindowInfo.TYPE_ACCESSIBILITY_OVERLAY -> "accessibility_overlay"
        AccessibilityWindowInfo.TYPE_SPLIT_SCREEN_DIVIDER -> "split_screen_divider"
        else -> "unknown_$type"
    }
}
