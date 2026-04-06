package com.openclaw.android.security

import android.content.Intent
import android.service.notification.NotificationListenerService
import android.service.notification.StatusBarNotification
import com.openclaw.android.AppLogger
import com.openclaw.android.OpenClawService
import java.util.concurrent.CopyOnWriteArrayList

/**
 * Hooks ALL incoming notifications.
 * Extracts text and forwards to OpenClawService via broadcast for Layer 1/2 scan.
 * Exposes notification list and ContentProviderReader to OpenClawService's
 * HTTP sidecar (:8766/v1/context) via the singleton [instance].
 */
class PrismNotificationListener : NotificationListenerService() {

    companion object {
        private const val TAG = "PrismNotifListener"

        /** Singleton for OpenClawService to read notifications via HTTP sidecar. */
        @Volatile
        var instance: PrismNotificationListener? = null
            private set
    }

    private val activeNotifications = CopyOnWriteArrayList<NotificationEntry>()
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
        instance = this
        contentReader = ContentProviderReader(this)
        AppLogger.i(TAG, "PrismNotificationListener started")
    }

    override fun onDestroy() {
        super.onDestroy()
        instance = null
    }

    /** Public accessor for OpenClawService sidecar to read current notifications. */
    fun getActiveNotificationsList(): List<NotificationEntry> = activeNotifications.toList()

    /** Public accessor for content provider reader (SMS, contacts, calendar). */
    fun getContentReader(): ContentProviderReader = contentReader

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
}
