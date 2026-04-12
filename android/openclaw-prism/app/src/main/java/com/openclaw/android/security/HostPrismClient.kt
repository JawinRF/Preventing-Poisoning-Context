package com.openclaw.android.security

import org.json.JSONObject
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL

/**
 * Optional bridge from the Android sidecar (:8766) to the host Python sidecar
 * (:8765) for deeper text scanning.
 *
 * On the Android emulator, 10.0.2.2 resolves to the host machine. This lets the
 * on-device service keep fast local normalization + ONNX Layer 2, while
 * optionally consulting the host-side TinyBERT/DeBERTa stack when available.
 */
object HostPrismClient {
    private const val HOST_INSPECT_URL = "http://10.0.2.2:8765/v1/inspect"
    private const val TIMEOUT_MS = 2500

    data class InspectResult(
        val verdict: String,
        val confidence: Double,
        val reason: String,
        val layerTriggered: String,
        val normalizedText: String,
    )

    fun inspect(
        text: String,
        ingestionPath: String,
        sourceType: String,
        sourceName: String,
        entryId: String,
        sessionId: String,
        runId: String,
    ): InspectResult? {
        val conn = (URL(HOST_INSPECT_URL).openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = TIMEOUT_MS
            readTimeout = TIMEOUT_MS
            doOutput = true
            setRequestProperty("Content-Type", "application/json")
        }

        return try {
            val payload = JSONObject().apply {
                put("entry_id", entryId)
                put("text", text)
                put("ingestion_path", ingestionPath)
                put("source_type", sourceType)
                put("source_name", sourceName)
                put("session_id", sessionId)
                put("run_id", runId)
                put("metadata", JSONObject())
            }

            OutputStreamWriter(conn.outputStream, Charsets.UTF_8).use { writer ->
                writer.write(payload.toString())
            }

            val code = conn.responseCode
            if (code !in 200..299) {
                null
            } else {
                val body = conn.inputStream.bufferedReader(Charsets.UTF_8).use { it.readText() }
                val json = JSONObject(body)
                InspectResult(
                    verdict = json.optString("verdict", "ALLOW"),
                    confidence = json.optDouble("confidence", 0.0),
                    reason = json.optString("reason", "host_deep_scan_unavailable"),
                    layerTriggered = json.optString("layer_triggered", "none"),
                    normalizedText = json.optString("normalized_text", text),
                )
            }
        } catch (_: Exception) {
            null
        } finally {
            conn.disconnect()
        }
    }
}
