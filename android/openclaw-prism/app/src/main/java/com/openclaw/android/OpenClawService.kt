package com.openclaw.android

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.BroadcastReceiver
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.Build
import android.os.IBinder
import com.openclaw.android.security.MemShieldDb
import com.openclaw.android.security.AuditEntry
import com.openclaw.android.security.MemShield
import com.openclaw.android.security.Normalizer
import com.openclaw.android.security.OnnxClassifier
import com.openclaw.android.security.PiiGuard
import com.openclaw.android.security.PrismAccessibilityService
import com.openclaw.android.security.PrismDetector
import com.openclaw.android.security.PrismNotificationListener
import com.openclaw.android.security.ContentProviderReader
import com.openclaw.android.security.HostPrismClient
import fi.iki.elonen.NanoHTTPD
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import org.json.JSONObject

/**
 * Unified foreground service: keeps terminal sessions alive AND runs PRISM Shield.
 *
 * Terminal: START_STICKY keeps sessions alive when app is backgrounded.
 * PRISM: HTTP sidecar on :8766, clipboard monitoring, notification scan receiver.
 *
 *   POST /v1/inspect        — Normalization + local Layer 2 + optional host deep scan
 *   POST /v1/guard          — PII Guard on outgoing agent actions
 *   POST /v1/ui-integrity   — OS-level tap integrity check (replaces VLM visual grounding)
 *   GET  /v1/context        — Unified PRISM-scanned device context (notifications, clipboard, SMS, contacts, calendar)
 *   GET  /v1/audit          — Recent on-device audit log (query param: limit, default 20)
 *   GET  /v1/status         — Health check + blocked count
 */
class OpenClawService : Service() {
    companion object {
        private const val TAG = "OpenClawService"
        private const val NOTIFICATION_ID = 1
        private const val CHANNEL_ID = "openclaw_service"
        const val SIDECAR_PORT = 8766
        const val ACTION_NOTIF_TEXT = "com.openclaw.android.NOTIFICATION_TEXT"
        const val EXTRA_TEXT = "text"
    }

    private val serviceScope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private var classifier: OnnxClassifier? = null
    private var memShield: MemShield? = null
    private var httpServer: NanoHTTPD? = null

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        startForeground(NOTIFICATION_ID, createNotification("OpenClaw PRISM active"))

        // Initialize PRISM security components
        try {
            classifier = OnnxClassifier(this)
            memShield = MemShield(this)
            AppLogger.i(TAG, "PRISM security initialized (ONNX classifier loaded)")
        } catch (e: Exception) {
            AppLogger.w(TAG, "PRISM ML classifier unavailable, running heuristics-only: ${e.message}")
        }

        startHttpSidecar()
        hookClipboard()
        registerNotifReceiver()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        startForeground(NOTIFICATION_ID, createNotification("OpenClaw PRISM active"))
        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        super.onDestroy()
        serviceScope.cancel()
        httpServer?.stop()
        classifier?.close()
    }

    // ── HTTP Sidecar (NanoHTTPD on :8766) ────────────────────────────────────

    private fun startHttpSidecar() {
        val svc = this
        httpServer = object : NanoHTTPD(SIDECAR_PORT) {
            override fun serve(session: IHTTPSession): Response {
                val uri = session.uri
                val body = try {
                    val map = mutableMapOf<String, String>()
                    session.parseBody(map)
                    map["postData"] ?: ""
                } catch (_: Exception) { "" }

                val responseJson = when (uri) {
                    "/v1/inspect" -> kotlinx.coroutines.runBlocking { svc.handleInspect(body) }
                    "/v1/guard" -> svc.handleGuard(body)
                    "/v1/ui-integrity" -> svc.handleUiIntegrity(body)
                    "/v1/context" -> kotlinx.coroutines.runBlocking {
                        // Hard ceiling: a slow device read can never hang the
                        // agent's per-step context fetch again. On timeout the
                        // agent proceeds in degraded mode rather than freezing.
                        kotlinx.coroutines.withTimeoutOrNull(6000L) {
                            svc.handleContext()
                        } ?: """{"error":"context_timeout","notifications":[],""" +
                             """"clipboard":"","sms":[],"contacts":[],"calendar":[],""" +
                             """"prism_context_blocked":0}"""
                    }
                    "/v1/audit" -> kotlinx.coroutines.runBlocking {
                        svc.handleAudit(session.parameters["limit"]?.firstOrNull())
                    }
                    "/v1/status" -> kotlinx.coroutines.runBlocking { svc.handleStatus() }
                    "/health" -> """{"status":"ok","sidecar":"android","port":$SIDECAR_PORT}"""
                    else -> """{"error":"unknown endpoint"}"""
                }
                return newFixedLengthResponse(Response.Status.OK, "application/json", responseJson)
            }
        }
        httpServer?.start(NanoHTTPD.SOCKET_READ_TIMEOUT, false)
        AppLogger.i(TAG, "PRISM HTTP sidecar listening on :$SIDECAR_PORT")
    }

    // POST /v1/inspect — normalization + local Layer 2 + optional host deep scan
    private suspend fun handleInspect(body: String): String {
        val json = JSONObject(body)
        val path = json.optString("path", json.optString("ingestion_path", "unknown"))
        val content = json.optString("content", json.optString("text", ""))
        val entryId = json.optString("entry_id", "android-${System.currentTimeMillis()}")
        val sourceType = json.optString("source_type", "android_sidecar")
        val sourceName = json.optString("source_name", "android")
        val sessionId = json.optString("session_id", "android-sidecar")
        val runId = json.optString("run_id", "android-sidecar")

        // Normalize
        val norm = Normalizer.normalize(content)

        // Layer 1 — telemetry only (rules logged, not used for enforcement)
        val l1 = PrismDetector.scan(norm.text)

        // Layer 2 — fast local ONNX screen.
        val l2Prob = classifier?.classify(norm.text)?.maliciousProb ?: 0.0f

        // If local Layer 2 does not already block, ask the host Python sidecar
        // for a deeper TinyBERT + DeBERTa scan when available.
        val hostDeepScan = if (l2Prob < 0.70f) {
            HostPrismClient.inspect(
                text = norm.text,
                ingestionPath = path,
                sourceType = sourceType,
                sourceName = sourceName,
                entryId = entryId,
                sessionId = sessionId,
                runId = runId,
            )
        } else null

        val finalVerdict: String
        val layerTriggered: String
        val confidence: Double
        val reason: String

        val l1Block = l1.verdict == PrismDetector.Verdict.BLOCK

        if (l2Prob >= 0.70f) {
            finalVerdict = "BLOCK"
            layerTriggered = "Layer2-ONNX"
            confidence = l2Prob.toDouble()
            reason = "Layer 2 ONNX identified prompt injection"
        } else if (l1Block && l2Prob >= 0.30f) {
            // L1 heuristic + moderate L2 agreement — combined gate
            finalVerdict = "BLOCK"
            layerTriggered = "Layer1+2-Combined"
            confidence = ((l1.score + l2Prob) / 2).toDouble()
            reason = "L1 heuristic + L2 ONNX combined: ${l1.matchedRules.joinToString()}"
        } else if (l1.score >= 0.80f) {
            // High-confidence heuristic alone (≥2 heavy categories matched)
            finalVerdict = "BLOCK"
            layerTriggered = "Layer1-HighConf"
            confidence = l1.score.toDouble()
            reason = "High-confidence heuristic: ${l1.matchedRules.joinToString()}"
        } else if (hostDeepScan != null && hostDeepScan.verdict == "BLOCK") {
            finalVerdict = "BLOCK"
            layerTriggered = hostDeepScan.layerTriggered
            confidence = hostDeepScan.confidence
            reason = hostDeepScan.reason
        } else {
            finalVerdict = "ALLOW"
            layerTriggered = hostDeepScan?.layerTriggered ?: "none"
            confidence = hostDeepScan?.confidence ?: (1.0 - l2Prob.toDouble())
            reason = if (hostDeepScan != null && hostDeepScan.reason.isNotBlank()) {
                hostDeepScan.reason
            } else {
                "clean"
            }
        }

        // Audit log
        MemShieldDb.get(this).auditDao().insert(
            AuditEntry(
                path = path,
                snippet = norm.text.take(120),
                verdict = finalVerdict,
                layer1Score = l1.score,
                layer2Prob = l2Prob,
                matchedRules = l1.matchedRules.joinToString(",")
            )
        )

        updateNotification(finalVerdict)

        val placeholderText = if (finalVerdict == "BLOCK") {
            "[PRISM_BLOCKED untrusted context removed before model assembly]"
        } else null

        // Response in Python sidecar-compatible schema
        return JSONObject().apply {
            put("verdict", finalVerdict)
            put("confidence", confidence)
            put("reason", reason)
            put("layer_triggered", layerTriggered)
            put("normalized_text", norm.text.take(200))
            put("ticket_id", JSONObject.NULL)
            put("placeholder", placeholderText ?: JSONObject.NULL)
            put("audit", JSONObject().apply {
                put("path", path)
                put("source_type", "android_sidecar")
                put("score", l1.score)
                put("l2_prob", l2Prob)
                put("rules", l1.matchedRules.joinToString(","))
                put("host_deep_scan_layer", hostDeepScan?.layerTriggered ?: JSONObject.NULL)
            })
            put("ingestion_path", path)
        }.toString()
    }

    // POST /v1/guard — PII Guard on agent actions
    private fun handleGuard(body: String): String {
        val json = JSONObject(body)
        val type = json.optString("action_type", "")
        val payload = json.optString("action_payload", "")
        val intent = json.optString("user_intent", "")

        val result = PiiGuard.check(type, payload, intent)

        return JSONObject().apply {
            put("verdict", result.verdict.name)
            put("reason", result.reason)
        }.toString()
    }

    // GET /v1/status
    private suspend fun handleStatus(): String {
        val blocked = MemShieldDb.get(this).auditDao().blockedCount()
        val total = MemShieldDb.get(this).auditDao().getRecent().size
        return JSONObject().apply {
            put("status", "running")
            put("port", SIDECAR_PORT)
            put("total_blocked", blocked)
            put("total_inspected", total)
            put("classifier_loaded", classifier != null)
        }.toString()
    }

    // POST /v1/ui-integrity — OS-level tap integrity check
    private fun handleUiIntegrity(body: String): String {
        val a11y = PrismAccessibilityService.instance
            ?: return JSONObject().apply {
                put("verdict", "ALLOW")
                put("reason", "accessibility_service_unavailable")
                put("checks", org.json.JSONArray())
            }.toString()

        val json = JSONObject(body)
        val targetText = json.optString("target_text").ifEmpty { null }
        val targetDesc = json.optString("target_desc").ifEmpty { null }
        val targetRid = json.optString("target_rid").ifEmpty { null }
        val targetClass = json.optString("target_class").ifEmpty { null }
        val targetX = if (json.has("target_x")) json.optInt("target_x") else null
        val targetY = if (json.has("target_y")) json.optInt("target_y") else null
        val expectedPkg = json.optString("expected_package").ifEmpty { null }

        return a11y.uiIntegrity.check(
            targetText = targetText,
            targetDesc = targetDesc,
            targetRid = targetRid,
            targetClass = targetClass,
            targetX = targetX,
            targetY = targetY,
            expectedPkg = expectedPkg,
        ).toString()
    }

    // GET /v1/context — Unified device context for the Python agent.
    // All user-readable text fields are PRISM-scanned before inclusion.
    // Poisoned fields are redacted with [PRISM_BLOCKED] rather than silently dropped.
    private fun handleContext(): String {
        val result = JSONObject()
        var contextBlockedCount = 0

        // Scan a text field and return it redacted if poisoned.
        // Audit entries are written asynchronously so this stays synchronous.
        fun scanField(text: String, path: String): String {
            if (text.isBlank()) return text
            val norm = Normalizer.normalize(text)
            val l1 = PrismDetector.scan(norm.text)
            // L2 ONNX scan intentionally NOT run here. It previously ran one
            // synchronous inference per field (notifications + clipboard + up
            // to 20 SMS + 20 contacts + 20 calendar = ~60 serial inferences)
            // on the single NanoHTTPD worker thread, which hung /v1/context on
            // a cold device. The Python agent re-scans every field through
            // MemShield / PRISM :8765 before the LLM sees it, so the on-device
            // L2 in the context path is redundant. Fast L1 regex stays.
            val l1Block = l1.verdict == PrismDetector.Verdict.BLOCK
            val blocked = l1Block || l1.score >= 0.80f
            if (blocked) {
                contextBlockedCount++
                serviceScope.launch {
                    MemShieldDb.get(this@OpenClawService).auditDao().insert(
                        AuditEntry(
                            path = path,
                            snippet = norm.text.take(120),
                            verdict = "BLOCK",
                            layer1Score = l1.score,
                            layer2Prob = 0.0f,
                            matchedRules = l1.matchedRules.joinToString(",")
                        )
                    )
                }
                return "[PRISM_BLOCKED]"
            }
            return text
        }

        // Notifications — from PrismNotificationListener singleton
        val listener = PrismNotificationListener.instance
        if (listener != null) {
            val notifArray = org.json.JSONArray()
            listener.getActiveNotificationsList().forEach { n ->
                notifArray.put(JSONObject().apply {
                    put("id", n.id)
                    put("package", n.packageName)
                    put("title", scanField(n.title, "notification_context"))
                    put("text", scanField(n.text, "notification_context"))
                    put("posted_time", n.postedTime)
                })
            }
            result.put("notifications", notifArray)
        } else {
            result.put("notifications", org.json.JSONArray())
            result.put("notifications_error", "NotificationListenerService not active")
        }

        // Clipboard
        val rawClip = try {
            val clipboard = getSystemService(CLIPBOARD_SERVICE) as ClipboardManager
            clipboard.primaryClip?.getItemAt(0)?.getText()?.toString() ?: ""
        } catch (_: Exception) { "" }
        result.put("clipboard", scanField(rawClip, "clipboard_context"))

        // SMS, Contacts, Calendar — from ContentProviderReader
        val reader = PrismNotificationListener.instance?.getContentReader()
            ?: ContentProviderReader(this)

        try {
            val smsArray = org.json.JSONArray()
            reader.getSmsMessages(limit = 8).forEach { m ->
                smsArray.put(JSONObject().apply {
                    put("id", m.id)
                    put("address", m.address)
                    put("body", scanField(m.body, "sms_context"))
                    put("date", m.date)
                })
            }
            result.put("sms", smsArray)
        } catch (e: Exception) {
            result.put("sms", org.json.JSONArray())
            result.put("sms_error", e.message ?: "unknown")
        }

        try {
            val contactsArray = org.json.JSONArray()
            reader.getContacts(limit = 8).forEach { c ->
                contactsArray.put(JSONObject().apply {
                    put("id", c.id)
                    put("name", c.name)
                    put("note", scanField(c.note, "contacts_context"))
                })
            }
            result.put("contacts", contactsArray)
        } catch (e: Exception) {
            result.put("contacts", org.json.JSONArray())
            result.put("contacts_error", e.message ?: "unknown")
        }

        try {
            val calendarArray = org.json.JSONArray()
            reader.getCalendarEvents(limit = 8).forEach { ev ->
                calendarArray.put(JSONObject().apply {
                    put("id", ev.id)
                    put("title", scanField(ev.title, "calendar_context"))
                    put("description", scanField(ev.description, "calendar_context"))
                    put("start_time", ev.startTime)
                    put("end_time", ev.endTime)
                })
            }
            result.put("calendar", calendarArray)
        } catch (e: Exception) {
            result.put("calendar", org.json.JSONArray())
            result.put("calendar_error", e.message ?: "unknown")
        }

        result.put("prism_context_blocked", contextBlockedCount)
        return result.toString()
    }

    // GET /v1/audit?limit=N — recent on-device audit log for Python daemon visibility
    private suspend fun handleAudit(limitStr: String?): String {
        val limit = limitStr?.toIntOrNull()?.coerceIn(1, 200) ?: 20
        val entries = MemShieldDb.get(this).auditDao().getRecent().take(limit)
        val arr = org.json.JSONArray()
        entries.forEach { e ->
            arr.put(JSONObject().apply {
                put("id", e.id)
                put("path", e.path)
                put("snippet", e.snippet)
                put("verdict", e.verdict)
                put("layer1_score", e.layer1Score)
                put("layer2_prob", e.layer2Prob)
                put("matched_rules", e.matchedRules)
                put("timestamp", e.timestamp)
            })
        }
        return JSONObject().apply {
            put("entries", arr)
            put("count", entries.size)
        }.toString()
    }

    // ── Clipboard Hook ────────────────────────────────────────────────────────

    private fun hookClipboard() {
        val clipboard = getSystemService(CLIPBOARD_SERVICE) as ClipboardManager
        clipboard.addPrimaryClipChangedListener {
            val text = clipboard.primaryClip
                ?.getItemAt(0)
                ?.getText()
                ?.toString() ?: return@addPrimaryClipChangedListener

            serviceScope.launch {
                val norm = Normalizer.normalize(text)
                val l1 = PrismDetector.scan(norm.text)
                val l2Prob = classifier?.classify(norm.text)?.maliciousProb ?: 0.0f
                if (l2Prob >= 0.70f) {
                    AppLogger.w(TAG, "Clipboard poison blocked: ${text.take(80)}")
                    MemShieldDb.get(this@OpenClawService).auditDao().insert(
                        AuditEntry(
                            path = "clipboard",
                            snippet = norm.text.take(120),
                            verdict = "BLOCK",
                            layer1Score = l1.score,
                            layer2Prob = l2Prob,
                            matchedRules = l1.matchedRules.joinToString(",")
                        )
                    )
                }
            }
        }
    }

    // ── Notification Receiver ─────────────────────────────────────────────────

    private fun registerNotifReceiver() {
        val filter = IntentFilter(ACTION_NOTIF_TEXT)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(notifReceiver, filter, RECEIVER_NOT_EXPORTED)
        } else {
            registerReceiver(notifReceiver, filter)
        }
    }

    private val notifReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            val text = intent?.getStringExtra(EXTRA_TEXT) ?: return
            serviceScope.launch {
                val norm = Normalizer.normalize(text)
                val l1 = PrismDetector.scan(norm.text)
                val l2Prob = classifier?.classify(norm.text)?.maliciousProb ?: 0.0f
                val verdict = if (l2Prob >= 0.70f) "BLOCK" else "ALLOW"

                MemShieldDb.get(this@OpenClawService).auditDao().insert(
                    AuditEntry(
                        path = "notification",
                        snippet = norm.text.take(120),
                        verdict = verdict,
                        layer1Score = l1.score,
                        layer2Prob = l2Prob,
                        matchedRules = l1.matchedRules.joinToString(",")
                    )
                )

                updateNotification(verdict)
            }
        }
    }

    // ── Notification helpers ──────────────────────────────────────────────────

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                getString(R.string.notification_channel_name),
                NotificationManager.IMPORTANCE_LOW,
            ).apply {
                description = "Keeps terminal sessions running and PRISM Shield active"
                setShowBadge(false)
            }
            getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
        }
    }

    private fun createNotification(text: String): Notification {
        val pendingIntent = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE,
        )

        val builder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            Notification.Builder(this, CHANNEL_ID)
        } else {
            @Suppress("DEPRECATION")
            Notification.Builder(this)
        }

        return builder
            .setContentTitle(getString(R.string.notification_title))
            .setContentText(text)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .build()
    }

    private fun updateNotification(verdict: String) {
        if (verdict == "BLOCK") {
            val nm = getSystemService(NotificationManager::class.java)
            nm.notify(NOTIFICATION_ID, createNotification("ALERT: Threat BLOCKED"))
        }
    }
}
