package com.openclaw.android.security

/**
 * Layer 1 -- Heuristic scanner.
 * Direct port of layer1_heuristics.py. Runs in ~0ms. No model needed.
 */
object PrismDetector {

    data class ScanResult(
        val verdict: Verdict,
        val score: Float,
        val matchedRules: List<String>
    )

    enum class Verdict { ALLOW, BLOCK }

    private val INJECTION_PATTERNS = listOf(
        Regex("""ignore\s+(all\s+)?(previous|prior|above)\s+instructions?""", RegexOption.IGNORE_CASE),
        Regex("""forget\s+(everything|all|your|prior)""", RegexOption.IGNORE_CASE),
        Regex("""you\s+are\s+now\s+(a|an)\s+\w+""", RegexOption.IGNORE_CASE),
        Regex("""act\s+as\s+(if\s+)?(you\s+are|an?)\s+\w+""", RegexOption.IGNORE_CASE),
        Regex("""new\s+(system\s+)?prompt\s*:""", RegexOption.IGNORE_CASE),
        Regex("""<\s*system\s*>.*?<\s*/\s*system\s*>""", setOf(RegexOption.IGNORE_CASE, RegexOption.DOT_MATCHES_ALL)),
        Regex("""\[SYSTEM\]|\[INST\]|\[\/INST\]|\[END\]"""),
        Regex("""###\s*(instruction|system|prompt)""", RegexOption.IGNORE_CASE),
    )

    private val EXFIL_PATTERNS = listOf(
        Regex("""(send|forward|email|post|upload|transmit)\s+.{0,40}(to|at)\s+\S+@\S+""", RegexOption.IGNORE_CASE),
        Regex("""(send|forward|post|upload)\s+.{0,40}(http|https|ftp)://""", RegexOption.IGNORE_CASE),
        Regex("""curl\s+-[xXdD]""", RegexOption.IGNORE_CASE),
        Regex("""wget\s+--post""", RegexOption.IGNORE_CASE),
        Regex("""(exfiltrate|steal|leak)\s+(data|info|contacts|passwords?)""", RegexOption.IGNORE_CASE),
    )

    private val ROLE_PATTERNS = listOf(
        Regex("""your\s+(real\s+)?name\s+is\s+\w+""", RegexOption.IGNORE_CASE),
        Regex("""you\s+(must|should|will)\s+(always|never)\s+\w+""", RegexOption.IGNORE_CASE),
        Regex("""(disable|bypass|override)\s+(safety|filter|guard|restriction)""", RegexOption.IGNORE_CASE),
        Regex("""DAN\s+mode|jailbreak|developer\s+mode""", RegexOption.IGNORE_CASE),
    )

    private val UNICODE_PATTERNS = listOf(
        Regex("""[\u200B-\u200D\uFEFF]"""),
        Regex("""[\u0400-\u04FF]"""),
    )

    private const val BLOCK_THRESHOLD = 0.45f

    private val RULE_WEIGHTS = mapOf(
        "injection" to 0.50f,
        "exfil" to 0.60f,
        "role" to 0.45f,
        "unicode" to 0.30f,
    )

    fun scan(text: String): ScanResult {
        val matched = mutableListOf<String>()
        var score = 0f

        fun check(label: String, patterns: List<Regex>) {
            if (patterns.any { it.containsMatchIn(text) }) {
                matched += label
                score += RULE_WEIGHTS[label] ?: 0.3f
            }
        }

        check("injection", INJECTION_PATTERNS)
        check("exfil", EXFIL_PATTERNS)
        check("role", ROLE_PATTERNS)
        check("unicode", UNICODE_PATTERNS)

        score = score.coerceAtMost(1.0f)
        val verdict = if (score >= BLOCK_THRESHOLD) Verdict.BLOCK else Verdict.ALLOW
        return ScanResult(verdict, score, matched)
    }
}
