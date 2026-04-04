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
import com.openclaw.android.security.PrismDetector
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
 *   POST /v1/inspect  — Layer 1+2 defense (heuristics + ONNX ML)
 *   POST /v1/guard    — PII Guard on outgoing agent actions
 *   GET  /v1/status   — Health check + blocked count
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
                    "/v1/status" -> kotlinx.coroutines.runBlocking { svc.handleStatus() }
                    else -> """{"error":"unknown endpoint"}"""
                }
                return newFixedLengthResponse(Response.Status.OK, "application/json", responseJson)
            }
        }
        httpServer?.start(NanoHTTPD.SOCKET_READ_TIMEOUT, false)
        AppLogger.i(TAG, "PRISM HTTP sidecar listening on :$SIDECAR_PORT")
    }

    // POST /v1/inspect — PRISM Layer 1+2 scan (schema-compatible with Python sidecar)
    private suspend fun handleInspect(body: String): String {
        val json = JSONObject(body)
        val path = json.optString("path", json.optString("ingestion_path", "unknown"))
        val content = json.optString("content", json.optString("text", ""))
        val entryId = json.optString("entry_id", "android-${System.currentTimeMillis()}")

        // Normalize
        val norm = Normalizer.normalize(content)

        // Layer 1 — heuristics
        val l1 = PrismDetector.scan(norm.text)

        // Layer 2 — ONNX ML (only if L1 is uncertain and classifier available)
        val l2Prob = if (l1.score in 0.2f..0.7f && classifier != null) {
            classifier!!.classify(norm.text).maliciousProb
        } else {
            if (l1.verdict == PrismDetector.Verdict.BLOCK) 1.0f else 0.0f
        }

        val finalVerdict = when {
            l1.verdict == PrismDetector.Verdict.BLOCK -> "BLOCK"
            l2Prob >= 0.70f -> "BLOCK"
            else -> "ALLOW"
        }

        val layerTriggered = when {
            l1.verdict == PrismDetector.Verdict.BLOCK -> "Layer1-Heuristics"
            l2Prob >= 0.70f -> "Layer2-ONNX"
            else -> "none"
        }

        val confidence = if (finalVerdict == "BLOCK") {
            maxOf(l1.score, l2Prob).toDouble()
        } else {
            (1.0 - maxOf(l1.score, l2Prob).toDouble())
        }

        val reason = if (finalVerdict == "BLOCK") {
            "Matched: ${l1.matchedRules.joinToString(",").ifEmpty { "ML classifier" }}"
        } else {
            "clean"
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

        // Response in Python sidecar-compatible schema
        return JSONObject().apply {
            put("verdict", finalVerdict)
            put("confidence", confidence)
            put("reason", reason)
            put("layer_triggered", layerTriggered)
            put("normalized_text", norm.text.take(200))
            put("ingestion_path", path)
            put("score", l1.score)
            put("l2_prob", l2Prob)
            put("rules", l1.matchedRules.joinToString(","))
            put("ticket_id", JSONObject.NULL)
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
                if (l1.verdict == PrismDetector.Verdict.BLOCK) {
                    AppLogger.w(TAG, "Clipboard poison blocked: ${text.take(80)}")
                    MemShieldDb.get(this@OpenClawService).auditDao().insert(
                        AuditEntry(
                            path = "clipboard",
                            snippet = norm.text.take(120),
                            verdict = "BLOCK",
                            layer1Score = l1.score,
                            layer2Prob = 0f,
                            matchedRules = l1.matchedRules.joinToString(",")
                        )
                    )
                    (getSystemService(CLIPBOARD_SERVICE) as ClipboardManager).clearPrimaryClip()
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
