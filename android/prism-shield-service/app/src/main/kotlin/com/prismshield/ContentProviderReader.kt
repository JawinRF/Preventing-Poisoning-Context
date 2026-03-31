package com.prismshield

import android.content.ContentResolver
import android.content.Context
import android.database.Cursor
import android.net.Uri
import android.provider.CalendarContract
import android.provider.ContactsContract
import org.json.JSONArray
import org.json.JSONObject

/**
 * Reads SMS, Contacts, and Calendar data via ContentProviders.
 * Used by PrismShieldService to query these ingestion paths for defense.
 */
class ContentProviderReader(private val context: Context) {

    private val contentResolver: ContentResolver = context.contentResolver

    // ── SMS ───────────────────────────────────────────────────────────────────

    /**
     * Read recent SMS messages from inbox.
     * Requires READ_SMS permission.
     */
    fun getSmsMessages(limit: Int = 50): List<SmsEntry> {
        val messages = mutableListOf<SmsEntry>()
        
        try {
            val uri = Uri.parse("content://sms/inbox")
            val projection = arrayOf("_id", "address", "body", "date", "read")
            val selection = "read = 0"  // Unread messages are higher risk
            val sortOrder = "date DESC LIMIT $limit"

            contentResolver.query(
                uri,
                projection,
                selection,
                null,
                sortOrder
            )?.use { cursor ->
                while (cursor.moveToNext()) {
                    val id = cursor.getLong(0)
                    val address = cursor.getString(1) ?: "unknown"
                    val body = cursor.getString(2) ?: ""
                    val date = cursor.getLong(3)
                    val read = cursor.getInt(4)

                    if (body.isNotBlank()) {
                        messages.add(SmsEntry(
                            id = id.toString(),
                            address = address,
                            body = body,
                            date = date,
                            read = read == 1
                        ))
                    }
                }
            }
        } catch (e: SecurityException) {
            android.util.Log.w("ContentProvider", "SMS permission not granted: ${e.message}")
        } catch (e: Exception) {
            android.util.Log.w("ContentProvider", "Failed to read SMS: ${e.message}")
        }

        return messages
    }

    data class SmsEntry(
        val id: String,
        val address: String,
        val body: String,
        val date: Long,
        val read: Boolean
    )

    // ── Contacts ──────────────────────────────────────────────────────────────

    /**
     * Read contacts with notes (notes can contain injected prompts).
     * Requires READ_CONTACTS permission.
     */
    fun getContacts(limit: Int = 50): List<ContactEntry> {
        val contacts = mutableListOf<ContactEntry>()

        try {
            val uri = ContactsContract.Data.CONTENT_URI
            val projection = arrayOf(
                ContactsContract.Data.CONTACT_ID,
                ContactsContract.Data.DISPLAY_NAME,
                ContactsContract.CommonDataKinds.Note.NOTE
            )
            // Only get contacts with notes (potential injection vector)
            val selection = "${ContactsContract.CommonDataKinds.Note.NOTE} IS NOT NULL AND ${ContactsContract.CommonDataKinds.Note.NOTE} != ''"
            val sortOrder = "${ContactsContract.Data.CONTACT_ID} LIMIT $limit"

            contentResolver.query(
                uri,
                projection,
                selection,
                null,
                sortOrder
            )?.use { cursor ->
                val nameMap = mutableMapOf<String, String>()
                
                // First pass: collect names
                val nameProjection = arrayOf(
                    ContactsContract.Data.CONTACT_ID,
                    ContactsContract.Data.DISPLAY_NAME
                )
                contentResolver.query(
                    ContactsContract.Data.CONTENT_URI,
                    nameProjection,
                    "${ContactsContract.Data.CONTACT_ID} IS NOT NULL",
                    null,
                    null
                )?.use { nameCursor ->
                    while (nameCursor.moveToNext()) {
                        val id = nameCursor.getLong(0)
                        val name = nameCursor.getString(1) ?: "Unknown"
                        nameMap[id.toString()] = name
                    }
                }

                // Second pass: get contacts with notes
                while (cursor.moveToNext()) {
                    val id = cursor.getLong(0)
                    val note = cursor.getString(2) ?: ""

                    if (note.isNotBlank()) {
                        contacts.add(ContactEntry(
                            id = id.toString(),
                            name = nameMap[id.toString()] ?: "Unknown",
                            note = note
                        ))
                    }
                }
            }
        } catch (e: SecurityException) {
            android.util.Log.w("ContentProvider", "Contacts permission not granted: ${e.message}")
        } catch (e: Exception) {
            android.util.Log.w("ContentProvider", "Failed to read contacts: ${e.message}")
        }

        return contacts
    }

    data class ContactEntry(
        val id: String,
        val name: String,
        val note: String
    )

    // ── Calendar ─────────────────────────────────────────────────────────────

    /**
     * Read upcoming calendar events (descriptions can contain injected prompts).
     * Requires READ_CALENDAR permission.
     */
    fun getCalendarEvents(limit: Int = 50): List<CalendarEntry> {
        val events = mutableListOf<CalendarEntry>()

        try {
            val uri = CalendarContract.Events.CONTENT_URI
            val projection = arrayOf(
                CalendarContract.Events._ID,
                CalendarContract.Events.TITLE,
                CalendarContract.Events.DESCRIPTION,
                CalendarContract.Events.DTSTART,
                CalendarContract.Events.DTEND
            )
            // Get upcoming events only
            val now = System.currentTimeMillis()
            val selection = "${CalendarContract.Events.DTSTART} >= ?"
            val selectionArgs = arrayOf(now.toString())
            val sortOrder = "${CalendarContract.Events.DTSTART} ASC LIMIT $limit"

            contentResolver.query(
                uri,
                projection,
                selection,
                selectionArgs,
                sortOrder
            )?.use { cursor ->
                while (cursor.moveToNext()) {
                    val id = cursor.getLong(0)
                    val title = cursor.getString(1) ?: ""
                    val description = cursor.getString(2) ?: ""
                    val startTime = cursor.getLong(3)
                    val endTime = cursor.getLong(4)

                    if (title.isNotBlank() || description.isNotBlank()) {
                        events.add(CalendarEntry(
                            id = id.toString(),
                            title = title,
                            description = description,
                            startTime = startTime,
                            endTime = endTime
                        ))
                    }
                }
            }
        } catch (e: SecurityException) {
            android.util.Log.w("ContentProvider", "Calendar permission not granted: ${e.message}")
        } catch (e: Exception) {
            android.util.Log.w("ContentProvider", "Failed to read calendar: ${e.message}")
        }

        return events
    }

    data class CalendarEntry(
        val id: String,
        val title: String,
        val description: String,
        val startTime: Long,
        val endTime: Long
    )

    // ── JSON Serialization ─────────────────────────────────────────────────

    fun smsToJson(messages: List<SmsEntry>): String {
        val arr = JSONArray()
        for (m in messages) {
            arr.put(JSONObject().apply {
                put("id", m.id)
                put("address", m.address)
                put("body", m.body)
                put("date", m.date)
                put("read", m.read)
            })
        }
        return arr.toString()
    }

    fun contactsToJson(contacts: List<ContactEntry>): String {
        val arr = JSONArray()
        for (c in contacts) {
            arr.put(JSONObject().apply {
                put("id", c.id)
                put("name", c.name)
                put("note", c.note)
            })
        }
        return arr.toString()
    }

    fun calendarToJson(events: List<CalendarEntry>): String {
        val arr = JSONArray()
        for (e in events) {
            arr.put(JSONObject().apply {
                put("id", e.id)
                put("title", e.title)
                put("description", e.description)
                put("start_time", e.startTime)
                put("end_time", e.endTime)
            })
        }
        return arr.toString()
    }
}