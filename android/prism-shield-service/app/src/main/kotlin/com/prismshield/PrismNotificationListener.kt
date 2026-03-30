package com.prismshield

import android.content.Intent
import android.service.notification.NotificationListenerService
import android.service.notification.StatusBarNotification
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.PrintWriter
import java.net.ServerSocket
import java.net.Socket
import java.util.concurrent.CopyOnWriteArrayList

/**
 * Hooks ALL incoming notifications.
 * Extracts text and forwards to PrismShieldService via broadcast.
 * Also serves active notifications via local socket for Python sidecar.
 *
 * Must be declared in AndroidManifest.xml with BIND_NOTIFICATION_LISTENER_SERVICE permission.
 * User must grant via: Settings > Notifications > Notification Access
 *
 * Socket API:
 *   - Connect to /data/local/tmp/prism_notif.sock
 *   - Send: {"action":"list"}
 *   - Response: {"notifications":[{"package":"...","title":"...","text":"...","id":...}]}
 *   - Send: {"action":"dismiss","id":...}
 *   - Send: {"action":"reply","id":"...","text":"..."}
 */
class PrismNotificationListener : NotificationListenerService() {

    companion object {
        const val SOCKET_PATH = "/data/local/tmp/prism_notif.sock"
        const val NOTIF_PORT = 8767  // Fixed port for Python to connect via ADB forward
    }

    private val activeNotifications = CopyOnWriteArrayList<NotificationEntry>()
    private var serverThread: Thread? = null

    data class NotificationEntry(
        val id: String,
        val packageName: String,
        val title: String,
        val text: String,
        val postedTime: Long
    )

    override fun onCreate() {
        super.onCreate()
        startSocketServer()
    }

    override fun onDestroy() {
        super.onDestroy()
        serverThread?.interrupt()
    }

    override fun onNotificationPosted(sbn: StatusBarNotification?) {
        if (sbn == null) return

        val extras = sbn.notification.extras
        val title = extras.getCharSequence("android.title")?.toString() ?: ""
        val text  = extras.getCharSequence("android.text")?.toString()  ?: ""
        val full  = "$title $text".trim()

        if (full.isBlank()) return

        // Store active notification
        val entry = NotificationEntry(
            id = sbn.key,
            packageName = sbn.packageName,
            title = title,
            text = text,
            postedTime = System.currentTimeMillis()
        )
        activeNotifications.add(entry)

        // Keep only last 50 notifications
        while (activeNotifications.size > 50) {
            activeNotifications.removeAt(0)
        }

        // Forward to PrismShieldService for Layer 1/2 scan
        sendBroadcast(Intent(PrismShieldService.ACTION_NOTIF_TEXT).apply {
            `package` = packageName
            putExtra(PrismShieldService.EXTRA_TEXT, full)
        })
    }

    override fun onNotificationRemoved(sbn: StatusBarNotification?) {
        sbn?.let {
            activeNotifications.removeIf { n -> n.id == it.key }
        }
    }

    private fun startSocketServer() {
        serverThread = Thread {
            try {
                // Use fixed TCP port for simple ADB forwarding
                val server = ServerSocket(NOTIF_PORT)
                android.util.Log.i("PrismNotif", "Notification socket listening on port $NOTIF_PORT")

                while (!Thread.interrupted()) {
                    val client = server.accept()
                    handleClient(client)
                }
            } catch (e: Exception) {
                android.util.Log.w("PrismNotif", "Socket server error: ${e.message}")
            }
        }.apply { start() }
    }

    private fun handleClient(client: Socket) {
        try {
            val reader = BufferedReader(InputStreamReader(client.getInputStream()))
            val writer = PrintWriter(client.getOutputStream())

            val request = reader.readLine()
            val response = processRequest(request)

            writer.println(response)
            writer.flush()
        } catch (e: Exception) {
            android.util.Log.w("PrismNotif", "Client handling error: ${e.message}")
        } finally {
            client.close()
        }
    }

    private fun processRequest(request: String?): String {
        if (request.isNullOrBlank()) {
            return jsonError("Empty request")
        }

        try {
            val json = JSONObject(request)
            val action = json.optString("action", "")

            return when (action) {
                "list" -> handleList()
                "dismiss" -> handleDismiss(json)
                "reply" -> handleReply(json)
                else -> jsonError("Unknown action: $action")
            }
        } catch (e: Exception) {
            return jsonError("Invalid JSON: ${e.message}")
        }
    }

    private fun handleList(): String {
        val notifs = JSONArray()
        for (n in activeNotifications) {
            notifs.put(JSONObject().apply {
                put("id", n.id)
                put("package", n.packageName)
                put("title", n.title)
                put("text", n.text)
                put("posted_time", n.postedTime)
            })
        }
        return JSONObject().put("notifications", notifs).toString()
    }

    private fun handleDismiss(json: JSONObject): String {
        val id = json.optString("id", "")
        if (id.isBlank()) {
            return jsonError("Missing 'id' for dismiss")
        }

        // Find and cancel the notification
        val entry = activeNotifications.find { it.id == id }
        if (entry != null) {
            // Note: Cannot directly cancel from NotificationListenerService
            // Would need to communicate back to app or use notificationManager
            return JSONObject().put("status", "not_implemented_in_listener").toString()
        }
        return jsonError("Notification not found")
    }

    private fun handleReply(json: JSONObject): String {
        val id = json.optString("id", "")
        val text = json.optString("text", "")

        if (id.isBlank() || text.isBlank()) {
            return jsonError("Missing 'id' or 'text' for reply")
        }

        // Reply requires posting as the same app - complex, return status
        return JSONObject().put("status", "reply_not_supported").put("reason", "Cannot reply as notification source from listener").toString()
    }

    private fun jsonError(message: String): String =
        JSONObject().put("error", message).toString()
}
