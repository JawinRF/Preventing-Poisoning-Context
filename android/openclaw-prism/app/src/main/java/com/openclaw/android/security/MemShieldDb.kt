package com.openclaw.android.security

import android.content.Context
import androidx.room.*

// ── Entities ─────────────────────────────────────────────────────────────────

@Entity(tableName = "memory_chunks")
data class MemoryChunk(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val source: String,
    val content: String,
    val embedding: String,
    val insertedAt: Long = System.currentTimeMillis(),
    val scanVerdict: String = "PENDING"
)

@Entity(tableName = "audit_log")
data class AuditEntry(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val path: String,
    val snippet: String,
    val verdict: String,
    val layer1Score: Float,
    val layer2Prob: Float,
    val matchedRules: String,
    val timestamp: Long = System.currentTimeMillis()
)

// ── DAOs ─────────────────────────────────────────────────────────────────────

@Dao
interface MemoryChunkDao {
    @Insert
    suspend fun insert(chunk: MemoryChunk): Long

    @Query("SELECT * FROM memory_chunks WHERE scanVerdict = 'ALLOW' ORDER BY insertedAt DESC LIMIT 50")
    suspend fun getCleanChunks(): List<MemoryChunk>

    @Query("UPDATE memory_chunks SET scanVerdict = :verdict WHERE id = :id")
    suspend fun updateVerdict(id: Long, verdict: String)

    @Query("SELECT * FROM memory_chunks WHERE scanVerdict = 'PENDING'")
    suspend fun getPendingChunks(): List<MemoryChunk>

    @Query("DELETE FROM memory_chunks WHERE scanVerdict = 'BLOCK'")
    suspend fun deletePoisoned()
}

@Dao
interface AuditDao {
    @Insert
    suspend fun insert(entry: AuditEntry)

    @Query("SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT 200")
    suspend fun getRecent(): List<AuditEntry>

    @Query("SELECT COUNT(*) FROM audit_log WHERE verdict = 'BLOCK'")
    suspend fun blockedCount(): Int

    @Query("SELECT COUNT(*) FROM audit_log")
    suspend fun totalCount(): Int

    @Query("SELECT COUNT(*) FROM audit_log WHERE verdict = 'ALLOW'")
    suspend fun allowedCount(): Int
}

// ── Database ──────────────────────────────────────────────────────────────────

@Database(
    entities = [MemoryChunk::class, AuditEntry::class],
    version = 1,
    exportSchema = false
)
abstract class MemShieldDb : RoomDatabase() {
    abstract fun chunkDao(): MemoryChunkDao
    abstract fun auditDao(): AuditDao

    companion object {
        @Volatile private var INSTANCE: MemShieldDb? = null

        fun get(context: Context): MemShieldDb =
            INSTANCE ?: synchronized(this) {
                INSTANCE ?: Room.databaseBuilder(
                    context.applicationContext,
                    MemShieldDb::class.java,
                    "memshield.db"
                ).build().also { INSTANCE = it }
            }
    }
}

// ── MemShield Scanner ────────────────────────────────────────────────────────

class MemShield(private val context: Context) {

    private val db by lazy { MemShieldDb.get(context) }

    suspend fun storeChunk(source: String, content: String, embedding: FloatArray): Long {
        val l1 = PrismDetector.scan(content)
        val verdict = if (l1.verdict == PrismDetector.Verdict.BLOCK) "BLOCK" else "ALLOW"

        return db.chunkDao().insert(
            MemoryChunk(
                source = source,
                content = content,
                embedding = embedding.joinToString(","),
                scanVerdict = verdict
            )
        )
    }

    suspend fun scanChunks(chunks: List<MemoryChunk>): List<MemoryChunk> {
        return chunks.filter { chunk ->
            val l1 = PrismDetector.scan(chunk.content)
            if (l1.verdict == PrismDetector.Verdict.BLOCK) {
                db.chunkDao().updateVerdict(chunk.id, "BLOCK")
                db.auditDao().insert(
                    AuditEntry(
                        path = "rag_retrieval",
                        snippet = chunk.content.take(120),
                        verdict = "BLOCK",
                        layer1Score = l1.score,
                        layer2Prob = 0f,
                        matchedRules = l1.matchedRules.joinToString(",")
                    )
                )
                false
            } else true
        }
    }

    suspend fun purgePoison() = db.chunkDao().deletePoisoned()
}
