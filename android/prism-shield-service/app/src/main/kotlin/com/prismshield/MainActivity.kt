package com.prismshield

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.delay
import java.text.SimpleDateFormat
import java.util.*

class MainActivity : ComponentActivity() {

    companion object {
        private const val RC_PERMISSIONS = 1001
        private val REQUIRED_PERMISSIONS = arrayOf(
            Manifest.permission.READ_SMS,
            Manifest.permission.READ_CONTACTS,
            Manifest.permission.READ_CALENDAR,
        )
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requestMissingPermissions()
        startForegroundService(Intent(this, PrismShieldService::class.java))
        setContent {
            MaterialTheme(colorScheme = darkColorScheme()) {
                PrismDashboard(
                    onGrantNotifAccess = {
                        startActivity(Intent(Settings.ACTION_NOTIFICATION_LISTENER_SETTINGS))
                    }
                )
            }
        }
    }

    private fun requestMissingPermissions() {
        val missing = REQUIRED_PERMISSIONS.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        if (missing.isNotEmpty()) {
            ActivityCompat.requestPermissions(this, missing.toTypedArray(), RC_PERMISSIONS)
        }
    }
}

// ── Colour tokens ─────────────────────────────────────────────────────────────
private val BG       = Color(0xFF0D1117)
private val SURFACE  = Color(0xFF161B22)
private val BORDER   = Color(0xFF30363D)
private val BLUE     = Color(0xFF58A6FF)
private val GREEN    = Color(0xFF3FB950)
private val RED      = Color(0xFFF85149)
private val YELLOW   = Color(0xFFD29922)
private val TEXT     = Color(0xFFE6EDF3)
private val SUBTEXT  = Color(0xFF8B949E)
private val fmt = SimpleDateFormat("HH:mm:ss", Locale.getDefault())

@Composable
fun PrismDashboard(onGrantNotifAccess: () -> Unit) {
    val ctx = LocalContext.current
    var logs         by remember { mutableStateOf(listOf<AuditEntry>()) }
    var blockedTotal by remember { mutableIntStateOf(0) }
    var allowedTotal by remember { mutableIntStateOf(0) }

    LaunchedEffect(Unit) {
        while (true) {
            val db    = MemShieldDb.get(ctx)
            logs         = db.auditDao().getRecent()
            blockedTotal = db.auditDao().blockedCount()
            allowedTotal = logs.count { it.verdict == "ALLOW" }
            delay(2_000)
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(BG)
            .padding(horizontal = 16.dp, vertical = 12.dp)
    ) {
        // ── Header ────────────────────────────────────────────────────────────
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(
                "⚡ PRISM Shield",
                color      = BLUE,
                fontSize   = 20.sp,
                fontWeight = FontWeight.Bold,
                fontFamily = FontFamily.Monospace
            )
            Spacer(Modifier.weight(1f))
            StatusPill(active = true)
        }

        Text(
            "localhost:8766  •  On-device sidecar",
            color      = SUBTEXT,
            fontSize   = 11.sp,
            fontFamily = FontFamily.Monospace,
            modifier   = Modifier.padding(top = 2.dp, bottom = 12.dp)
        )

        // ── Stat row ──────────────────────────────────────────────────────────
        Row(
            modifier              = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            StatCard("BLOCKED", blockedTotal.toString(), RED,   Modifier.weight(1f))
            StatCard("ALLOWED", allowedTotal.toString(), GREEN, Modifier.weight(1f))
            StatCard("TOTAL",   logs.size.toString(),    BLUE,  Modifier.weight(1f))
        }

        Spacer(Modifier.height(12.dp))

        // ── Grant notification access button ──────────────────────────────────
        OutlinedButton(
            onClick = onGrantNotifAccess,
            modifier    = Modifier.fillMaxWidth(),
            colors      = ButtonDefaults.outlinedButtonColors(contentColor = YELLOW),
            shape       = RoundedCornerShape(6.dp)
        ) {
            Text(
                "⚙  Grant Notification Access",
                fontFamily = FontFamily.Monospace,
                fontSize   = 12.sp
            )
        }

        Spacer(Modifier.height(12.dp))

        // ── Section title ─────────────────────────────────────────────────────
        Text(
            "LIVE THREAT FEED",
            color      = SUBTEXT,
            fontSize   = 10.sp,
            fontWeight = FontWeight.SemiBold,
            fontFamily = FontFamily.Monospace,
            modifier   = Modifier.padding(bottom = 6.dp)
        )

        // ── Audit log list ────────────────────────────────────────────────────
        LazyColumn(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            if (logs.isEmpty()) {
                item {
                    Text(
                        "No events yet. Waiting for input…",
                        color    = SUBTEXT,
                        fontSize = 12.sp,
                        fontFamily = FontFamily.Monospace,
                        modifier = Modifier.padding(top = 20.dp).fillMaxWidth()
                            .wrapContentWidth(Alignment.CenterHorizontally)
                    )
                }
            }
            items(logs, key = { it.id }) { entry ->
                LogRow(entry)
            }
        }
    }
}

// ── Composables ──────────────────────────────────────────────────────────────

@Composable
private fun StatusPill(active: Boolean) {
    val color = if (active) GREEN else RED
    val label = if (active) "● ACTIVE" else "● OFF"
    Box(
        modifier = Modifier
            .background(color.copy(alpha = 0.15f), RoundedCornerShape(20.dp))
            .padding(horizontal = 10.dp, vertical = 3.dp)
    ) {
        Text(label, color = color, fontSize = 11.sp, fontFamily = FontFamily.Monospace)
    }
}

@Composable
private fun StatCard(label: String, value: String, color: Color, modifier: Modifier) {
    Column(
        modifier = modifier
            .background(SURFACE, RoundedCornerShape(8.dp))
            .padding(10.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(value, color = color, fontSize = 22.sp, fontWeight = FontWeight.Bold,
            fontFamily = FontFamily.Monospace)
        Text(label, color = SUBTEXT, fontSize = 9.sp, fontFamily = FontFamily.Monospace)
    }
}

@Composable
private fun LogRow(entry: AuditEntry) {
    val isBlock     = entry.verdict == "BLOCK"
    val accentColor = if (isBlock) RED else GREEN
    val icon        = if (isBlock) "✗" else "✓"

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(SURFACE, RoundedCornerShape(6.dp))
            .padding(horizontal = 10.dp, vertical = 8.dp),
        verticalAlignment = Alignment.Top
    ) {
        // Verdict icon
        Text(
            icon,
            color    = accentColor,
            fontSize = 14.sp,
            fontFamily = FontFamily.Monospace,
            modifier = Modifier.padding(end = 8.dp, top = 2.dp)
        )

        Column(modifier = Modifier.weight(1f)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                // Path badge
                Box(
                    modifier = Modifier
                        .background(accentColor.copy(alpha = 0.12f), RoundedCornerShape(4.dp))
                        .padding(horizontal = 5.dp, vertical = 1.dp)
                ) {
                    Text(entry.path, color = accentColor, fontSize = 9.sp,
                        fontFamily = FontFamily.Monospace)
                }
                Spacer(Modifier.width(6.dp))
                // Timestamp
                Text(
                    fmt.format(Date(entry.timestamp)),
                    color    = SUBTEXT,
                    fontSize = 9.sp,
                    fontFamily = FontFamily.Monospace
                )
            }

            Spacer(Modifier.height(4.dp))

            // Snippet
            Text(
                entry.snippet.take(90),
                color    = TEXT,
                fontSize = 11.sp,
                maxLines = 2,
                fontFamily = FontFamily.Monospace
            )

            // Scores row
            if (isBlock) {
                Spacer(Modifier.height(3.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    ScoreBadge("L1=${"%.2f".format(entry.layer1Score)}", YELLOW)
                    if (entry.layer2Prob > 0f)
                        ScoreBadge("L2=${"%.2f".format(entry.layer2Prob)}", RED)
                    if (entry.matchedRules.isNotEmpty())
                        ScoreBadge(entry.matchedRules.split(",").first(), SUBTEXT)
                }
            }
        }
    }
}

@Composable
private fun ScoreBadge(text: String, color: Color) {
    Text(
        text,
        color    = color,
        fontSize = 9.sp,
        fontFamily = FontFamily.Monospace,
        modifier = Modifier
            .background(color.copy(alpha = 0.10f), RoundedCornerShape(3.dp))
            .padding(horizontal = 4.dp, vertical = 1.dp)
    )
}
