package com.openclaw.android

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.PowerManager
import android.provider.Settings
import android.webkit.JavascriptInterface
import com.google.gson.Gson
import com.openclaw.android.security.MemShieldDb
import kotlinx.coroutines.CoroutineExceptionHandler
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking

/**
 * WebView <-> Kotlin bridge via @JavascriptInterface.
 * Callable from JavaScript as window.OpenClaw.<method>().
 * Includes all OpenClaw host methods + PRISM security methods.
 */
@Suppress("TooManyFunctions", "LargeClass")
class JsBridge(
    private val activity: MainActivity,
    private val sessionManager: TerminalSessionManager,
    private val bootstrapManager: BootstrapManager,
    private val eventBridge: EventBridge,
) {
    private val gson = Gson()

    companion object {
        private const val TAG = "JsBridge"
        private const val SHELL_INIT_DELAY_MS = 500L
        private const val COMMAND_TIMEOUT_MS = 5_000L
        private const val PLATFORM_LIST_TIMEOUT_MS = 10_000L
        private const val API_TIMEOUT_MS = 5000
        private const val PROGRESS_START = 0f
        private const val PROGRESS_HALF = 0.5f
        private const val PROGRESS_DONE = 1f
        private const val PROGRESS_DOWNLOAD = 0.2f
        private const val PROGRESS_EXTRACT = 0.6f
        private const val PROGRESS_APPLY = 0.9f
        private const val PROGRESS_BOOTSTRAP_START = 0.1f
    }

    private fun launchWithErrorHandling(
        errorEventType: String = "error",
        errorContext: Map<String, Any?> = emptyMap(),
        block: suspend CoroutineScope.() -> Unit,
    ) {
        val handler = CoroutineExceptionHandler { _, throwable ->
            AppLogger.e(TAG, "Coroutine error [$errorEventType]: ${throwable.message}", throwable)
            eventBridge.emit(errorEventType, errorContext + mapOf(
                "error" to (throwable.message ?: "Unknown error"),
                "progress" to PROGRESS_START,
                "message" to "Error: ${throwable.message}",
            ))
        }
        CoroutineScope(Dispatchers.IO + handler).launch(block = block)
    }

    // ═══════════════════════════════════════════
    // Terminal domain
    // ═══════════════════════════════════════════

    @JavascriptInterface
    fun showTerminal() {
        if (sessionManager.activeSession == null) {
            val session = sessionManager.createSession()
            if (bootstrapManager.needsPostSetup()) {
                val script = bootstrapManager.postSetupScript.absolutePath
                android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
                    session.write("bash $script\n")
                }, SHELL_INIT_DELAY_MS)
            }
        }
        activity.showTerminal()
    }

    @JavascriptInterface
    fun showWebView() = activity.showWebView()

    @JavascriptInterface
    fun createSession(): String {
        val session = sessionManager.createSession()
        return gson.toJson(mapOf("id" to session.mHandle, "name" to (session.title ?: "Terminal")))
    }

    @JavascriptInterface
    fun switchSession(id: String) = activity.runOnUiThread { sessionManager.switchSession(id) }

    @JavascriptInterface
    fun closeSession(id: String) = sessionManager.closeSession(id)

    @JavascriptInterface
    fun getTerminalSessions(): String = gson.toJson(sessionManager.getSessionsInfo())

    @JavascriptInterface
    fun writeToTerminal(id: String, data: String) {
        val session = if (id.isBlank()) sessionManager.activeSession
        else sessionManager.getSessionById(id) ?: sessionManager.activeSession
        session?.write(data)
    }

    @JavascriptInterface
    fun runInNewSession(command: String) {
        val session = sessionManager.createSession()
        activity.showTerminal()
        android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
            session.write(command)
        }, SHELL_INIT_DELAY_MS)
    }

    // ═══════════════════════════════════════════
    // Setup domain
    // ═══════════════════════════════════════════

    @JavascriptInterface
    fun getSetupStatus(): String = gson.toJson(bootstrapManager.getStatus())

    @JavascriptInterface
    fun getBootstrapStatus(): String = gson.toJson(mapOf(
        "installed" to bootstrapManager.isInstalled(),
        "prefixPath" to bootstrapManager.prefixDir.absolutePath,
    ))

    @JavascriptInterface
    fun startSetup() {
        launchWithErrorHandling(errorEventType = "setup_progress", errorContext = mapOf("progress" to PROGRESS_START)) {
            bootstrapManager.startSetup { progress, message ->
                eventBridge.emit("setup_progress", mapOf("progress" to progress, "message" to message))
            }
        }
    }

    @JavascriptInterface
    fun saveToolSelections(json: String) {
        val configFile = java.io.File(bootstrapManager.homeDir, ".openclaw-android/tool-selections.conf")
        configFile.parentFile?.mkdirs()
        val selections = gson.fromJson(json, Map::class.java) as? Map<*, *> ?: return
        val lines = selections.entries.joinToString("\n") { (key, value) ->
            "INSTALL_${(key as String).uppercase().replace("-", "_")}=$value"
        }
        configFile.writeText(lines + "\n")
    }

    // ═══════════════════════════════════════════
    // Platform domain
    // ═══════════════════════════════════════════

    @JavascriptInterface
    fun getAvailablePlatforms(): String = gson.toJson(listOf(
        mapOf("id" to "openclaw", "name" to "OpenClaw", "icon" to "/openclaw.svg", "desc" to "AI agent platform"),
    ))

    @JavascriptInterface
    fun getInstalledPlatforms(): String {
        val env = EnvironmentBuilder.build(activity)
        val result = CommandRunner.runSync("npm list -g --depth=0 --json 2>/dev/null", env, bootstrapManager.prefixDir, timeoutMs = PLATFORM_LIST_TIMEOUT_MS)
        return result.stdout.ifBlank { "[]" }
    }

    @JavascriptInterface
    fun installPlatform(id: String) {
        launchWithErrorHandling(errorEventType = "install_progress", errorContext = mapOf("target" to id)) {
            eventBridge.emit("install_progress", mapOf("target" to id, "progress" to PROGRESS_START, "message" to "Installing $id..."))
            val env = EnvironmentBuilder.build(activity)
            CommandRunner.runStreaming("npm install -g $id@latest --ignore-scripts", env, bootstrapManager.homeDir) { output ->
                eventBridge.emit("install_progress", mapOf("target" to id, "progress" to PROGRESS_HALF, "message" to output))
            }
            eventBridge.emit("install_progress", mapOf("target" to id, "progress" to PROGRESS_DONE, "message" to "$id installed"))
        }
    }

    @JavascriptInterface
    fun uninstallPlatform(id: String) {
        launchWithErrorHandling(errorEventType = "install_progress", errorContext = mapOf("target" to id)) {
            val env = EnvironmentBuilder.build(activity)
            CommandRunner.runSync("npm uninstall -g $id", env, bootstrapManager.homeDir)
        }
    }

    @JavascriptInterface
    fun switchPlatform(id: String) {
        val markerFile = java.io.File(bootstrapManager.homeDir, ".openclaw-android/.platform")
        markerFile.parentFile?.mkdirs()
        markerFile.writeText(id)
    }

    @JavascriptInterface
    fun getActivePlatform(): String {
        val markerFile = java.io.File(bootstrapManager.homeDir, ".openclaw-android/.platform")
        val id = if (markerFile.exists()) markerFile.readText().trim() else "openclaw"
        return gson.toJson(mapOf("id" to id, "name" to id.replaceFirstChar { it.uppercase() }))
    }

    // ═══════════════════════════════════════════
    // Tools domain
    // ═══════════════════════════════════════════

    @JavascriptInterface
    fun getInstalledTools(): String {
        val prefix = bootstrapManager.prefixDir.absolutePath
        val tools = mutableListOf<Map<String, String>>()
        val pkgChecks = mapOf("tmux" to "$prefix/bin/tmux", "ttyd" to "$prefix/bin/ttyd", "dufs" to "$prefix/bin/dufs", "openssh-server" to "$prefix/bin/sshd", "android-tools" to "$prefix/bin/adb", "code-server" to "$prefix/bin/code-server")
        for ((id, path) in pkgChecks) { if (java.io.File(path).exists()) tools.add(mapOf("id" to id, "name" to id, "version" to "installed")) }
        if (java.io.File("$prefix/bin/chromium-browser").exists() || java.io.File("$prefix/bin/chromium").exists()) tools.add(mapOf("id" to "chromium", "name" to "chromium", "version" to "installed"))
        val nodeBin = "${bootstrapManager.homeDir.absolutePath}/.openclaw-android/node/bin"
        val npmBinChecks = mapOf("claude-code" to "$nodeBin/claude", "gemini-cli" to "$nodeBin/gemini", "codex-cli" to "$nodeBin/codex", "opencode" to "$nodeBin/opencode")
        for ((id, path) in npmBinChecks) { if (java.io.File(path).exists()) tools.add(mapOf("id" to id, "name" to id, "version" to "installed")) }
        return gson.toJson(tools)
    }

    @JavascriptInterface
    fun installTool(id: String) {
        launchWithErrorHandling(errorEventType = "install_progress", errorContext = mapOf("target" to id)) {
            val env = EnvironmentBuilder.build(activity)
            val prefix = bootstrapManager.prefixDir.absolutePath
            val aptGet = "DEBIAN_FRONTEND=noninteractive $prefix/bin/apt-get -y -o Acquire::AllowInsecureRepositories=true -o APT::Get::AllowUnauthenticated=true"
            val cmd = when (id) {
                "tmux", "ttyd", "dufs", "openssh-server", "android-tools" -> "$aptGet install ${if (id == "openssh-server") "openssh" else id}"
                "chromium" -> "$aptGet install chromium"
                "code-server" -> "npm install -g code-server"
                "claude-code" -> "npm install -g @anthropic-ai/claude-code"
                "gemini-cli" -> "npm install -g @google/gemini-cli"
                "codex-cli" -> "npm install -g @openai/codex"
                "opencode" -> "curl -fsSL https://raw.githubusercontent.com/AidanPark/openclaw-android/main/scripts/install-opencode.sh | bash"
                else -> "echo 'Unknown tool: $id'"
            }
            eventBridge.emit("install_progress", mapOf("target" to id, "progress" to PROGRESS_START, "message" to "Installing $id..."))
            CommandRunner.runStreaming(cmd, env, bootstrapManager.homeDir) { output ->
                eventBridge.emit("install_progress", mapOf("target" to id, "progress" to PROGRESS_HALF, "message" to output))
            }
            eventBridge.emit("install_progress", mapOf("target" to id, "progress" to PROGRESS_DONE, "message" to "$id installed"))
        }
    }

    @JavascriptInterface
    fun uninstallTool(id: String) {
        launchWithErrorHandling(errorEventType = "install_progress", errorContext = mapOf("target" to id)) {
            val env = EnvironmentBuilder.build(activity)
            val cmd = when (id) {
                "tmux", "ttyd", "dufs", "openssh-server", "android-tools", "chromium" -> "${bootstrapManager.prefixDir.absolutePath}/bin/apt-get remove -y ${if (id == "openssh-server") "openssh" else id}"
                "code-server" -> "npm uninstall -g code-server"
                "claude-code" -> "npm uninstall -g @anthropic-ai/claude-code"
                "gemini-cli" -> "npm uninstall -g @google/gemini-cli"
                "codex-cli" -> "npm uninstall -g @openai/codex"
                "opencode" -> "rm -f \$PREFIX/bin/opencode \$HOME/.openclaw-android/bin/ld.so.opencode \$PREFIX/tmp/ld.so.opencode && rm -rf \$HOME/.config/opencode"
                else -> "echo 'Unknown tool: $id'"
            }
            CommandRunner.runSync(cmd, env, bootstrapManager.homeDir)
        }
    }

    @JavascriptInterface
    fun isToolInstalled(id: String): String {
        val prefix = bootstrapManager.prefixDir.absolutePath
        val env = EnvironmentBuilder.build(activity)
        val exists = when (id) {
            "openssh-server" -> java.io.File("$prefix/bin/sshd").exists()
            "tmux", "ttyd", "dufs", "android-tools" -> java.io.File("$prefix/bin/${if (id == "android-tools") "adb" else id}").exists()
            "chromium" -> java.io.File("$prefix/bin/chromium-browser").exists() || java.io.File("$prefix/bin/chromium").exists()
            "code-server" -> java.io.File("$prefix/bin/code-server").exists()
            else -> CommandRunner.runSync("command -v $id 2>/dev/null", env, bootstrapManager.prefixDir, timeoutMs = COMMAND_TIMEOUT_MS).stdout.trim().isNotEmpty()
        }
        return gson.toJson(mapOf("installed" to exists))
    }

    // ═══════════════════════════════════════════
    // Commands domain
    // ═══════════════════════════════════════════

    @JavascriptInterface
    fun runCommand(cmd: String): String {
        val env = EnvironmentBuilder.build(activity)
        return gson.toJson(CommandRunner.runSync(cmd, env, bootstrapManager.homeDir))
    }

    @JavascriptInterface
    fun runCommandAsync(callbackId: String, cmd: String) {
        launchWithErrorHandling(errorEventType = "command_output", errorContext = mapOf("callbackId" to callbackId, "done" to true)) {
            val env = EnvironmentBuilder.build(activity)
            CommandRunner.runStreaming(cmd, env, bootstrapManager.homeDir) { output ->
                eventBridge.emit("command_output", mapOf("callbackId" to callbackId, "data" to output, "done" to false))
            }
            eventBridge.emit("command_output", mapOf("callbackId" to callbackId, "data" to "", "done" to true))
        }
    }

    // ═══════════════════════════════════════════
    // Updates domain
    // ═══════════════════════════════════════════

    @JavascriptInterface
    fun checkForUpdates(): String {
        val updates = mutableListOf<Map<String, String>>()
        try {
            val configFile = java.io.File(activity.filesDir, "usr/share/openclaw-app/config.json")
            if (configFile.exists()) {
                val config = gson.fromJson(configFile.readText(), Map::class.java) as? Map<*, *>
                val localWwwVersion = activity.getSharedPreferences("openclaw", 0).getString("www_version", "0.0.0")
                val remoteWwwVersion = ((config?.get("www") as? Map<*, *>)?.get("version") as? String)
                if (remoteWwwVersion != null && remoteWwwVersion != localWwwVersion) {
                    updates.add(mapOf("component" to "www", "currentVersion" to (localWwwVersion ?: "0.0.0"), "newVersion" to remoteWwwVersion))
                }
            }
        } catch (_: Exception) {}
        return gson.toJson(updates)
    }

    @JavascriptInterface
    fun getApkUpdateInfo(): String = try {
        val url = java.net.URL("https://api.github.com/repos/AidanPark/openclaw-android/releases/latest")
        val conn = url.openConnection() as java.net.HttpURLConnection
        conn.connectTimeout = API_TIMEOUT_MS; conn.readTimeout = API_TIMEOUT_MS
        conn.setRequestProperty("Accept", "application/vnd.github+json")
        val body = conn.inputStream.bufferedReader().readText(); conn.disconnect()
        val release = gson.fromJson(body, Map::class.java) as? Map<*, *>
        val tagName = release?.get("tag_name") as? String ?: return gson.toJson(mapOf("error" to "no tag"))
        val latestVersion = tagName.trimStart('v')
        val currentVersion = activity.packageManager.getPackageInfo(activity.packageName, 0).versionName ?: "0.0.0"
        gson.toJson(mapOf("currentVersion" to currentVersion, "latestVersion" to latestVersion, "updateAvailable" to (compareVersions(latestVersion, currentVersion) > 0)))
    } catch (e: Exception) { gson.toJson(mapOf("error" to e.message)) }

    @JavascriptInterface
    fun applyUpdate(component: String) {
        launchWithErrorHandling(errorEventType = "install_progress", errorContext = mapOf("target" to component)) {
            emitProgress(component, PROGRESS_START, "Updating $component...")
            when (component) {
                "www" -> updateWww()
                "bootstrap" -> updateBootstrap()
            }
            emitProgress(component, PROGRESS_DONE, "$component updated")
        }
    }

    private fun emitProgress(target: String, progress: Float, message: String) {
        eventBridge.emit("install_progress", mapOf("target" to target, "progress" to progress, "message" to message))
    }

    private suspend fun updateWww() {
        try {
            val url = UrlResolver(activity).getWwwUrl()
            val stagingWww = java.io.File(activity.cacheDir, "www-staging"); stagingWww.deleteRecursively(); stagingWww.mkdirs()
            emitProgress("www", PROGRESS_DOWNLOAD, "Downloading...")
            val zipFile = java.io.File(activity.cacheDir, "www.zip")
            java.net.URL(url).openStream().use { input -> zipFile.outputStream().use { output -> input.copyTo(output) } }
            emitProgress("www", PROGRESS_EXTRACT, "Extracting...")
            extractZipToDir(zipFile, stagingWww); zipFile.delete()
            emitProgress("www", PROGRESS_APPLY, "Applying...")
            val wwwDir = bootstrapManager.wwwDir; wwwDir.deleteRecursively(); wwwDir.parentFile?.mkdirs(); stagingWww.renameTo(wwwDir)
            activity.runOnUiThread { activity.reloadWebView() }
        } catch (e: Exception) { emitProgress("www", PROGRESS_START, "Update failed: ${e.message}") }
    }

    private suspend fun updateBootstrap() {
        try {
            emitProgress("bootstrap", PROGRESS_BOOTSTRAP_START, "Downloading bootstrap...")
            bootstrapManager.startSetup { progress, message -> emitProgress("bootstrap", progress, message) }
        } catch (e: Exception) { emitProgress("bootstrap", PROGRESS_START, "Update failed: ${e.message}") }
    }

    private fun extractZipToDir(zipFile: java.io.File, targetDir: java.io.File) {
        java.util.zip.ZipInputStream(zipFile.inputStream()).use { zis ->
            var entry = zis.nextEntry
            while (entry != null) {
                val destFile = java.io.File(targetDir, entry.name)
                if (entry.isDirectory) destFile.mkdirs()
                else { destFile.parentFile?.mkdirs(); destFile.outputStream().use { out -> zis.copyTo(out) } }
                entry = zis.nextEntry
            }
        }
    }

    // ═══════════════════════════════════════════
    // System domain
    // ═══════════════════════════════════════════

    @JavascriptInterface
    fun getAppInfo(): String {
        val pInfo = activity.packageManager.getPackageInfo(activity.packageName, 0)
        return gson.toJson(mapOf("versionName" to (pInfo.versionName ?: "unknown"), "versionCode" to pInfo.versionCode, "packageName" to activity.packageName))
    }

    @JavascriptInterface
    fun getBatteryOptimizationStatus(): String {
        val pm = activity.getSystemService(Context.POWER_SERVICE) as PowerManager
        return gson.toJson(mapOf("isIgnoring" to pm.isIgnoringBatteryOptimizations(activity.packageName)))
    }

    @JavascriptInterface
    fun requestBatteryOptimizationExclusion() {
        activity.runOnUiThread {
            val intent = Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS)
            intent.data = Uri.parse("package:${activity.packageName}")
            activity.startActivity(intent)
        }
    }

    @JavascriptInterface
    fun openSystemSettings(page: String) {
        activity.runOnUiThread {
            val intent = when (page) {
                "battery" -> Intent(Settings.ACTION_BATTERY_SAVER_SETTINGS)
                "app_info" -> Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS).apply { data = Uri.parse("package:${activity.packageName}") }
                "notification_access" -> Intent(Settings.ACTION_NOTIFICATION_LISTENER_SETTINGS)
                "accessibility" -> Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)
                else -> Intent(Settings.ACTION_SETTINGS)
            }
            activity.startActivity(intent)
        }
    }

    @JavascriptInterface
    fun copyToClipboard(text: String) {
        activity.runOnUiThread {
            val clipboard = activity.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
            clipboard.setPrimaryClip(ClipData.newPlainText("OpenClaw", text))
        }
    }

    @JavascriptInterface
    fun getStorageInfo(): String {
        val filesDir = activity.filesDir
        val bootstrapSize = bootstrapManager.prefixDir.walkTopDown().sumOf { it.length() }
        val wwwSize = bootstrapManager.wwwDir.walkTopDown().sumOf { it.length() }
        return gson.toJson(mapOf("totalBytes" to filesDir.totalSpace, "freeBytes" to filesDir.freeSpace, "bootstrapBytes" to bootstrapSize, "wwwBytes" to wwwSize))
    }

    @JavascriptInterface
    fun clearCache() { activity.cacheDir.deleteRecursively(); activity.cacheDir.mkdirs() }

    @JavascriptInterface
    fun openUrl(url: String) {
        val intent = Intent(Intent.ACTION_VIEW, Uri.parse(url))
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        activity.startActivity(intent)
    }

    // ═══════════════════════════════════════════
    // PRISM Security domain (new)
    // ═══════════════════════════════════════════

    @JavascriptInterface
    fun getSecurityStatus(): String = runBlocking {
        val db = MemShieldDb.get(activity)
        val blocked = db.auditDao().blockedCount()
        val allowed = db.auditDao().allowedCount()
        val total = db.auditDao().totalCount()
        gson.toJson(mapOf(
            "status" to "active",
            "sidecarPort" to OpenClawService.SIDECAR_PORT,
            "blocked" to blocked,
            "allowed" to allowed,
            "total" to total,
        ))
    }

    @JavascriptInterface
    fun getAuditFeed(): String = runBlocking {
        val entries = MemShieldDb.get(activity).auditDao().getRecent()
        gson.toJson(entries.map { entry ->
            mapOf(
                "id" to entry.id,
                "path" to entry.path,
                "snippet" to entry.snippet,
                "verdict" to entry.verdict,
                "layer1Score" to entry.layer1Score,
                "layer2Prob" to entry.layer2Prob,
                "matchedRules" to entry.matchedRules,
                "timestamp" to entry.timestamp,
            )
        })
    }

    @JavascriptInterface
    fun getSidecarHealth(): String = try {
        val url = java.net.URL("http://127.0.0.1:8765/health")
        val conn = url.openConnection() as java.net.HttpURLConnection
        conn.connectTimeout = 2000; conn.readTimeout = 2000
        val body = conn.inputStream.bufferedReader().readText(); conn.disconnect()
        gson.toJson(mapOf("python_sidecar" to "online", "response" to body))
    } catch (_: Exception) {
        gson.toJson(mapOf("python_sidecar" to "offline"))
    }

    @JavascriptInterface
    fun getPermissionStatus(): String {
        val ctx = activity
        fun hasPermission(perm: String): Boolean =
            androidx.core.content.ContextCompat.checkSelfPermission(ctx, perm) == android.content.pm.PackageManager.PERMISSION_GRANTED
        return gson.toJson(mapOf(
            "read_sms" to hasPermission(android.Manifest.permission.READ_SMS),
            "read_contacts" to hasPermission(android.Manifest.permission.READ_CONTACTS),
            "read_calendar" to hasPermission(android.Manifest.permission.READ_CALENDAR),
        ))
    }

    private fun compareVersions(a: String, b: String): Int {
        val aParts = a.split(".").map { it.toIntOrNull() ?: 0 }
        val bParts = b.split(".").map { it.toIntOrNull() ?: 0 }
        for (i in 0 until maxOf(aParts.size, bParts.size)) {
            val diff = (aParts.getOrElse(i) { 0 }) - (bParts.getOrElse(i) { 0 })
            if (diff != 0) return diff
        }
        return 0
    }
}
