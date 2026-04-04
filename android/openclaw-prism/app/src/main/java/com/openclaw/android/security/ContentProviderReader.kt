package com.openclaw.android.security

import android.content.ContentResolver
import android.content.Context
import android.net.Uri
import android.provider.CalendarContract
import android.provider.ContactsContract
import com.openclaw.android.AppLogger
import org.json.JSONArray
import org.json.JSONObject

class ContentProviderReader(private val context: Context) {

    private val contentResolver: ContentResolver = context.contentResolver

    // ── SMS ──────────────────────────────────────────────────────────────────

    data class SmsEntry(val id: String, val address: String, val body: String, val date: Long, val read: Boolean)

    fun getSmsMessages(limit: Int = 50): List<SmsEntry> {
        val messages = mutableListOf<SmsEntry>()
        try {
            val uri = Uri.parse("content://sms/inbox")
            contentResolver.query(uri, arrayOf("_id", "address", "body", "date", "read"), "read = 0", null, "date DESC LIMIT $limit")?.use { cursor ->
                while (cursor.moveToNext()) {
                    val body = cursor.getString(2) ?: ""
                    if (body.isNotBlank()) {
                        messages.add(SmsEntry(cursor.getLong(0).toString(), cursor.getString(1) ?: "unknown", body, cursor.getLong(3), cursor.getInt(4) == 1))
                    }
                }
            }
        } catch (e: SecurityException) {
            AppLogger.w("ContentProvider", "SMS permission not granted: ${e.message}")
        } catch (e: Exception) {
            AppLogger.w("ContentProvider", "Failed to read SMS: ${e.message}")
        }
        return messages
    }

    // ── Contacts ─────────────────────────────────────────────────────────────

    data class ContactEntry(val id: String, val name: String, val note: String)

    fun getContacts(limit: Int = 50): List<ContactEntry> {
        val contacts = mutableListOf<ContactEntry>()
        try {
            val nameMap = mutableMapOf<String, String>()
            contentResolver.query(ContactsContract.Data.CONTENT_URI, arrayOf(ContactsContract.Data.CONTACT_ID, ContactsContract.Data.DISPLAY_NAME), "${ContactsContract.Data.CONTACT_ID} IS NOT NULL", null, null)?.use { c ->
                while (c.moveToNext()) { nameMap[c.getLong(0).toString()] = c.getString(1) ?: "Unknown" }
            }
            val selection = "${ContactsContract.CommonDataKinds.Note.NOTE} IS NOT NULL AND ${ContactsContract.CommonDataKinds.Note.NOTE} != ''"
            contentResolver.query(ContactsContract.Data.CONTENT_URI, arrayOf(ContactsContract.Data.CONTACT_ID, ContactsContract.Data.DISPLAY_NAME, ContactsContract.CommonDataKinds.Note.NOTE), selection, null, "${ContactsContract.Data.CONTACT_ID} LIMIT $limit")?.use { cursor ->
                while (cursor.moveToNext()) {
                    val note = cursor.getString(2) ?: ""
                    if (note.isNotBlank()) {
                        val id = cursor.getLong(0)
                        contacts.add(ContactEntry(id.toString(), nameMap[id.toString()] ?: "Unknown", note))
                    }
                }
            }
        } catch (e: SecurityException) {
            AppLogger.w("ContentProvider", "Contacts permission not granted: ${e.message}")
        } catch (e: Exception) {
            AppLogger.w("ContentProvider", "Failed to read contacts: ${e.message}")
        }
        return contacts
    }

    // ── Calendar ─────────────────────────────────────────────────────────────

    data class CalendarEntry(val id: String, val title: String, val description: String, val startTime: Long, val endTime: Long)

    fun getCalendarEvents(limit: Int = 50): List<CalendarEntry> {
        val events = mutableListOf<CalendarEntry>()
        try {
            val now = System.currentTimeMillis()
            contentResolver.query(CalendarContract.Events.CONTENT_URI, arrayOf(CalendarContract.Events._ID, CalendarContract.Events.TITLE, CalendarContract.Events.DESCRIPTION, CalendarContract.Events.DTSTART, CalendarContract.Events.DTEND), "${CalendarContract.Events.DTSTART} >= ?", arrayOf(now.toString()), "${CalendarContract.Events.DTSTART} ASC LIMIT $limit")?.use { cursor ->
                while (cursor.moveToNext()) {
                    val title = cursor.getString(1) ?: ""
                    val desc = cursor.getString(2) ?: ""
                    if (title.isNotBlank() || desc.isNotBlank()) {
                        events.add(CalendarEntry(cursor.getLong(0).toString(), title, desc, cursor.getLong(3), cursor.getLong(4)))
                    }
                }
            }
        } catch (e: SecurityException) {
            AppLogger.w("ContentProvider", "Calendar permission not granted: ${e.message}")
        } catch (e: Exception) {
            AppLogger.w("ContentProvider", "Failed to read calendar: ${e.message}")
        }
        return events
    }

    // ── JSON Serialization ───────────────────────────────────────────────────

    fun smsToJson(messages: List<SmsEntry>): String {
        val arr = JSONArray()
        for (m in messages) { arr.put(JSONObject().put("id", m.id).put("address", m.address).put("body", m.body).put("date", m.date).put("read", m.read)) }
        return arr.toString()
    }

    fun contactsToJson(contacts: List<ContactEntry>): String {
        val arr = JSONArray()
        for (c in contacts) { arr.put(JSONObject().put("id", c.id).put("name", c.name).put("note", c.note)) }
        return arr.toString()
    }

    fun calendarToJson(events: List<CalendarEntry>): String {
        val arr = JSONArray()
        for (e in events) { arr.put(JSONObject().put("id", e.id).put("title", e.title).put("description", e.description).put("start_time", e.startTime).put("end_time", e.endTime)) }
        return arr.toString()
    }
}
