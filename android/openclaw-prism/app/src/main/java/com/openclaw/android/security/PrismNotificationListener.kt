package com.openclaw.android.security

import android.content.Intent
import android.service.notification.NotificationListenerService
import android.service.notification.StatusBarNotification
import com.openclaw.android.AppLogger
import com.openclaw.android.OpenClawService
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
 * Extracts text and forwards to OpenClawService via broadcast for Layer 1/2 scan.
 * Also serves notifications, SMS, Contacts, Calendar via local socket (port 8767)
 * for the Python sidecar to consume via ADB forward.
 */
class PrismNotificationListener : NotificationListenerService() {

    companion object {
        private const val TAG = "PrismNotifListener"
        const val NOTIF_PORT = 8767
    }

    private val activeNotifications = CopyOnWriteArrayList<NotificationEntry>()
    private var serverThread: Thread? = null
    private lateinit var contentReader: ContentProviderReader

    data class NotificationEntry(
        val id: String,
        val packageName: String,
        val title: String,
        val text: String,
        val postedTime: Long
    )

    override fun onCreate() {
        super.onCreate()
        contentReader = ContentProviderReader(this)
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
        val text = extras.getCharSequence("android.text")?.toString() ?: ""
        val full = "$title $text".trim()
        if (full.isBlank()) return

        activeNotifications.add(NotificationEntry(sbn.key, sbn.packageName, title, text, System.currentTimeMillis()))
        while (activeNotifications.size > 50) activeNotifications.removeAt(0)

        sendBroadcast(Intent(OpenClawService.ACTION_NOTIF_TEXT).apply {
            `package` = packageName
            putExtra(OpenClawService.EXTRA_TEXT, full)
        })
    }

    override fun onNotificationRemoved(sbn: StatusBarNotification?) {
        sbn?.let { activeNotifications.removeIf { n -> n.id == it.key } }
    }

    private fun startSocketServer() {
        serverThread = Thread {
            try {
                val server = ServerSocket(NOTIF_PORT)
                AppLogger.i(TAG, "Notification socket listening on port $NOTIF_PORT")
                while (!Thread.interrupted()) {
                    val client = server.accept()
                    handleClient(client)
                }
            } catch (e: Exception) {
                AppLogger.w(TAG, "Socket server error: ${e.message}")
            }
        }.apply { start() }
    }

    private fun handleClient(client: Socket) {
        try {
            val reader = BufferedReader(InputStreamReader(client.getInputStream()))
            val writer = PrintWriter(client.getOutputStream())
            val request = reader.readLine()
            writer.println(processRequest(request))
            writer.flush()
        } catch (e: Exception) {
            AppLogger.w(TAG, "Client handling error: ${e.message}")
        } finally {
            client.close()
        }
    }

    private fun processRequest(request: String?): String {
        if (request.isNullOrBlank()) return jsonError("Empty request")
        return try {
            val json = JSONObject(request)
            when (json.optString("action", "")) {
                "list_notifications" -> handleList()
                "get_sms" -> handleGetSms()
                "get_contacts" -> handleGetContacts()
                "get_calendar" -> handleGetCalendar()
                "dismiss" -> JSONObject().put("status", "not_implemented_in_listener").toString()
                "reply" -> JSONObject().put("status", "reply_not_supported").toString()
                else -> jsonError("Unknown action")
            }
        } catch (e: Exception) {
            jsonError("Invalid JSON: ${e.message}")
        }
    }

    private fun handleList(): String {
        val notifs = JSONArray()
        for (n in activeNotifications) {
            notifs.put(JSONObject().put("id", n.id).put("package", n.packageName).put("title", n.title).put("text", n.text).put("posted_time", n.postedTime))
        }
        return JSONObject().put("notifications", notifs).toString()
    }

    private fun handleGetSms(): String = try {
        JSONObject().put("sms", JSONArray(contentReader.smsToJson(contentReader.getSmsMessages()))).toString()
    } catch (e: Exception) { jsonError("Failed to read SMS: ${e.message}") }

    private fun handleGetContacts(): String = try {
        JSONObject().put("contacts", JSONArray(contentReader.contactsToJson(contentReader.getContacts()))).toString()
    } catch (e: Exception) { jsonError("Failed to read contacts: ${e.message}") }

    private fun handleGetCalendar(): String = try {
        JSONObject().put("calendar", JSONArray(contentReader.calendarToJson(contentReader.getCalendarEvents()))).toString()
    } catch (e: Exception) { jsonError("Failed to read calendar: ${e.message}") }

    private fun jsonError(message: String): String = JSONObject().put("error", message).toString()
}
