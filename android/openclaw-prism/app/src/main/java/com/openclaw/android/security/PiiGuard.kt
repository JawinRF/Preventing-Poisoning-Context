package com.openclaw.android.security

/**
 * PII Guard -- runs AFTER OpenClaw proposes an action, BEFORE execution.
 * Checks that the agent's intended output/action was not induced by
 * a poisoned prompt (data exfiltration, unauthorized sends, etc.).
 */
object PiiGuard {

    data class GuardResult(
        val verdict: Verdict,
        val reason: String
    )

    enum class Verdict { ALLOW, BLOCK }

    private val EXFIL_ACTION_PATTERNS = listOf(
        Regex("""(send|forward|email|post|upload)\s+.{0,60}(http[s]?://|@\S+\.\S+)""", RegexOption.IGNORE_CASE),
        Regex("""curl\s+|wget\s+--post|nc\s+-[cez]""", RegexOption.IGNORE_CASE),
        Regex("""(contacts|messages|photos|passwords?|tokens?)\s+(to|at)\s+\S+""", RegexOption.IGNORE_CASE),
    )

    private val HIGH_RISK_ACTIONS = setOf(
        "send_email", "send_sms", "upload_file",
        "execute_shell", "post_request", "write_file"
    )

    fun check(actionType: String, actionPayload: String, userIntent: String): GuardResult {
        if (actionType in HIGH_RISK_ACTIONS) {
            val intentWords = userIntent.lowercase().split(Regex("\\s+")).toSet()
            val payloadWords = actionPayload.lowercase().split(Regex("\\s+")).toSet()
            val overlap = intentWords.intersect(payloadWords).size.toFloat()
            val similarity = overlap / (intentWords.size.coerceAtLeast(1))

            if (similarity < 0.15f) {
                return GuardResult(
                    Verdict.BLOCK,
                    "Action '$actionType' payload does not match user intent (similarity=${"%.2f".format(similarity)})"
                )
            }
        }

        for (pattern in EXFIL_ACTION_PATTERNS) {
            if (pattern.containsMatchIn(actionPayload)) {
                return GuardResult(
                    Verdict.BLOCK,
                    "Exfiltration pattern detected in action payload: ${pattern.pattern.take(40)}"
                )
            }
        }

        return GuardResult(Verdict.ALLOW, "clean")
    }
}
