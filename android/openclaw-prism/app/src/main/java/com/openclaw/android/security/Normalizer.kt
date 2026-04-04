package com.openclaw.android.security

import android.util.Base64
import java.net.URLDecoder
import java.nio.charset.StandardCharsets

/**
 * Normalizer -- de-obfuscates text before Layer 1/2 scanners.
 * Strips: URL encoding, Base64, invisible Unicode, ANSI escape codes,
 *         HTML tags, homoglyphs, excessive whitespace.
 */
object Normalizer {

    private val INVISIBLE_UNICODE = Regex(
        "[\u00AD\u200B\u200C\u200D\u200E\u200F\uFEFF\u2060\u2061\u2062\u2063]"
    )
    private val ANSI_ESCAPE = Regex("\u001B\\[[0-9;]*[mGKHF]")
    private val HTML_TAGS = Regex("<[^>]{0,200}>")
    private val MULTI_SPACE = Regex("[ \\t]{2,}")

    private val HOMOGLYPH_MAP = mapOf(
        '\u0430' to 'a', '\u0435' to 'e', '\u043E' to 'o', '\u0440' to 'p', '\u0441' to 'c',
        '\u0445' to 'x', '\u0456' to 'i', '\u0455' to 's', '\u0501' to 'd', '\u0261' to 'g',
        '\u0410' to 'A', '\u0412' to 'B', '\u0415' to 'E', '\u041A' to 'K', '\u041C' to 'M',
        '\u041D' to 'H', '\u041E' to 'O', '\u0420' to 'P', '\u0421' to 'C', '\u0422' to 'T',
        '\u0425' to 'X', '\u028F' to 'Y'
    )

    data class NormResult(
        val text: String,
        val transformsApplied: List<String>
    )

    fun normalize(raw: String): NormResult {
        var text = raw
        val transforms = mutableListOf<String>()

        // 1. URL decode
        try {
            val decoded = URLDecoder.decode(text, StandardCharsets.UTF_8.name())
            if (decoded != text) { text = decoded; transforms += "url_decode" }
        } catch (_: Exception) {}

        // 2. Base64 decode (only pure b64 payloads)
        val b64Candidate = Regex("^[A-Za-z0-9+/]{20,}={0,2}$")
        if (b64Candidate.matches(text.trim())) {
            try {
                val decoded = Base64.decode(text.trim(), Base64.DEFAULT)
                    .toString(StandardCharsets.UTF_8)
                if (decoded.any { it.isLetterOrDigit() }) {
                    text = decoded; transforms += "base64_decode"
                }
            } catch (_: Exception) {}
        }

        // 3. Strip ANSI escape sequences
        val noAnsi = ANSI_ESCAPE.replace(text, "")
        if (noAnsi != text) { text = noAnsi; transforms += "strip_ansi" }

        // 4. Strip HTML/XML tags
        val noHtml = HTML_TAGS.replace(text, " ")
        if (noHtml != text) { text = noHtml; transforms += "strip_html" }

        // 5. Remove invisible Unicode
        val noInvis = INVISIBLE_UNICODE.replace(text, "")
        if (noInvis != text) { text = noInvis; transforms += "strip_invisible_unicode" }

        // 6. Map homoglyphs to Latin
        val mapped = text.map { HOMOGLYPH_MAP[it] ?: it }.joinToString("")
        if (mapped != text) { text = mapped; transforms += "homoglyph_map" }

        // 7. Collapse whitespace
        text = MULTI_SPACE.replace(text, " ").trim()

        return NormResult(text, transforms)
    }
}
