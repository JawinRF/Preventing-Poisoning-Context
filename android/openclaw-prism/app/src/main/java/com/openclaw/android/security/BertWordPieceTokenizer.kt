package com.openclaw.android.security

import android.content.Context
import com.google.gson.JsonParser
import java.text.Normalizer
import java.util.Locale

/**
 * Lightweight BERT WordPiece tokenizer that reads the Hugging Face tokenizer
 * JSON exported with the TinyBERT v3 model. This keeps Android ONNX inputs
 * aligned with the host Python/HF path instead of using hash-based IDs.
 */
class BertWordPieceTokenizer(
    context: Context,
    assetFile: String = "tinybert_tokenizer.json",
) {

    companion object {
        private const val CLS_ID = 101L
        private const val SEP_ID = 102L
        private const val PAD_ID = 0L
        private const val DEFAULT_UNK_ID = 100L
        private const val MAX_INPUT_CHARS_PER_WORD = 100
        private const val CONTINUING_SUBWORD_PREFIX = "##"
    }

    private val vocab: Map<String, Long>
    private val unkToken: String
    private val unkId: Long
    private val maxInputCharsPerWord: Int
    private val continuingSubwordPrefix: String

    init {
        val root = context.assets.open(assetFile).bufferedReader().use { reader ->
            JsonParser.parseReader(reader).asJsonObject
        }
        val model = root.getAsJsonObject("model")
        val vocabJson = model.getAsJsonObject("vocab")

        val loadedVocab = LinkedHashMap<String, Long>(vocabJson.size())
        for ((token, value) in vocabJson.entrySet()) {
            loadedVocab[token] = value.asLong
        }
        vocab = loadedVocab

        unkToken = model.get("unk_token")?.asString ?: "[UNK]"
        unkId = vocab[unkToken] ?: DEFAULT_UNK_ID
        maxInputCharsPerWord = model.get("max_input_chars_per_word")?.asInt ?: MAX_INPUT_CHARS_PER_WORD
        continuingSubwordPrefix = model.get("continuing_subword_prefix")?.asString ?: CONTINUING_SUBWORD_PREFIX
    }

    fun encode(text: String, maxSeq: Int): LongArray {
        val wordPieceIds = tokenizeToIds(text).take((maxSeq - 2).coerceAtLeast(0))
        val ids = LongArray(maxSeq) { PAD_ID }
        if (maxSeq == 0) {
            return ids
        }

        ids[0] = CLS_ID
        for ((index, tokenId) in wordPieceIds.withIndex()) {
            ids[index + 1] = tokenId
        }
        val sepIndex = (wordPieceIds.size + 1).coerceAtMost(maxSeq - 1)
        ids[sepIndex] = SEP_ID
        return ids
    }

    private fun tokenizeToIds(text: String): List<Long> {
        val normalized = normalize(text)
        if (normalized.isBlank()) {
            return emptyList()
        }

        val tokens = mutableListOf<String>()
        for (whitespaceToken in normalized.split(Regex("\\s+"))) {
            if (whitespaceToken.isEmpty()) {
                continue
            }
            tokens += splitOnPunctuation(whitespaceToken)
        }

        val ids = ArrayList<Long>(tokens.size)
        for (token in tokens) {
            ids += wordPiece(token)
        }
        return ids
    }

    private fun normalize(text: String): String {
        val builder = StringBuilder(text.length + 8)
        var index = 0
        while (index < text.length) {
            val codePoint = Character.codePointAt(text, index)
            index += Character.charCount(codePoint)

            if (isControl(codePoint)) {
                continue
            }
            if (isWhitespace(codePoint)) {
                builder.append(' ')
                continue
            }
            if (isChineseChar(codePoint)) {
                builder.append(' ')
                builder.appendCodePoint(codePoint)
                builder.append(' ')
                continue
            }
            builder.appendCodePoint(codePoint)
        }

        val lower = builder.toString().lowercase(Locale.ROOT)
        val decomposed = Normalizer.normalize(lower, Normalizer.Form.NFD)
        val cleaned = StringBuilder(decomposed.length)
        var offset = 0
        while (offset < decomposed.length) {
            val codePoint = Character.codePointAt(decomposed, offset)
            offset += Character.charCount(codePoint)
            if (Character.getType(codePoint) == Character.NON_SPACING_MARK.toInt()) {
                continue
            }
            cleaned.appendCodePoint(codePoint)
        }
        return cleaned.toString()
    }

    private fun splitOnPunctuation(token: String): List<String> {
        val pieces = mutableListOf<String>()
        val current = StringBuilder()
        var index = 0
        while (index < token.length) {
            val codePoint = Character.codePointAt(token, index)
            index += Character.charCount(codePoint)
            if (isPunctuation(codePoint)) {
                if (current.isNotEmpty()) {
                    pieces += current.toString()
                    current.setLength(0)
                }
                pieces += String(Character.toChars(codePoint))
            } else {
                current.appendCodePoint(codePoint)
            }
        }
        if (current.isNotEmpty()) {
            pieces += current.toString()
        }
        return pieces
    }

    private fun wordPiece(token: String): List<Long> {
        if (token.length > maxInputCharsPerWord) {
            return listOf(unkId)
        }

        val pieces = mutableListOf<Long>()
        var start = 0
        while (start < token.length) {
            var end = token.length
            var currentPiece: Long? = null
            while (start < end) {
                val candidate = buildString {
                    if (start > 0) {
                        append(continuingSubwordPrefix)
                    }
                    append(token.substring(start, end))
                }
                currentPiece = vocab[candidate]
                if (currentPiece != null) {
                    break
                }
                end--
            }
            if (currentPiece == null) {
                return listOf(unkId)
            }
            pieces += currentPiece
            start = end
        }
        return pieces
    }

    private fun isControl(codePoint: Int): Boolean {
        if (codePoint == '\t'.code || codePoint == '\n'.code || codePoint == '\r'.code) {
            return false
        }
        return when (Character.getType(codePoint)) {
            Character.CONTROL.toInt(),
            Character.FORMAT.toInt(),
            Character.PRIVATE_USE.toInt(),
            Character.SURROGATE.toInt(),
            Character.UNASSIGNED.toInt() -> true
            else -> false
        }
    }

    private fun isWhitespace(codePoint: Int): Boolean {
        return Character.isWhitespace(codePoint) || Character.getType(codePoint) == Character.SPACE_SEPARATOR.toInt()
    }

    private fun isPunctuation(codePoint: Int): Boolean {
        return when {
            codePoint in 33..47 -> true
            codePoint in 58..64 -> true
            codePoint in 91..96 -> true
            codePoint in 123..126 -> true
            else -> when (Character.getType(codePoint)) {
                Character.CONNECTOR_PUNCTUATION.toInt(),
                Character.DASH_PUNCTUATION.toInt(),
                Character.START_PUNCTUATION.toInt(),
                Character.END_PUNCTUATION.toInt(),
                Character.INITIAL_QUOTE_PUNCTUATION.toInt(),
                Character.FINAL_QUOTE_PUNCTUATION.toInt(),
                Character.OTHER_PUNCTUATION.toInt() -> true
                else -> false
            }
        }
    }

    private fun isChineseChar(codePoint: Int): Boolean {
        return codePoint in 0x4E00..0x9FFF ||
            codePoint in 0x3400..0x4DBF ||
            codePoint in 0x20000..0x2A6DF ||
            codePoint in 0x2A700..0x2B73F ||
            codePoint in 0x2B740..0x2B81F ||
            codePoint in 0x2B820..0x2CEAF ||
            codePoint in 0xF900..0xFAFF ||
            codePoint in 0x2F800..0x2FA1F
    }
}
