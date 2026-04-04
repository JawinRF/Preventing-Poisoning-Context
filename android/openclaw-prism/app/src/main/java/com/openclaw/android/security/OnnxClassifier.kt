package com.openclaw.android.security

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.LongBuffer

/**
 * Layer 2 classifier using ONNX Runtime for Android.
 * Model: assets/tinybert_prism.onnx
 * Input:  int64[1, 128] x3  (input_ids, attention_mask, token_type_ids)
 * Output: float32[1, 2]     (benign_logit, malicious_logit)
 */
class OnnxClassifier(context: Context) {

    companion object {
        private const val MODEL_FILE = "tinybert_prism.onnx"
        private const val MAX_SEQ = 128
        private const val BLOCK_THRESHOLD = 0.70f
    }

    private val session: OrtSession
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()

    init {
        val modelBytes = context.assets.open(MODEL_FILE).readBytes()
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(2)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        session = env.createSession(modelBytes, opts)
    }

    data class ClassifierResult(
        val maliciousProb: Float,
        val isBlock: Boolean
    )

    private fun tokenize(text: String): LongArray {
        val tokens = text.lowercase()
            .replace(Regex("[^a-z0-9\\s]"), " ")
            .split(Regex("\\s+"))
            .filter { it.isNotEmpty() }
            .take(MAX_SEQ - 2)

        val ids = LongArray(MAX_SEQ) { 0L }
        ids[0] = 101L
        tokens.forEachIndexed { i, token ->
            ids[i + 1] = token.hashCode().and(Int.MAX_VALUE).toLong() % 30522L
        }
        ids[tokens.size + 1] = 102L
        return ids
    }

    fun classify(text: String): ClassifierResult {
        val ids = tokenize(text)
        val mask = LongArray(MAX_SEQ) { if (ids[it] != 0L) 1L else 0L }
        val types = LongArray(MAX_SEQ) { 0L }

        val shape = longArrayOf(1, MAX_SEQ.toLong())

        val inputIds = OnnxTensor.createTensor(env, LongBuffer.wrap(ids), shape)
        val inputMask = OnnxTensor.createTensor(env, LongBuffer.wrap(mask), shape)
        val inputType = OnnxTensor.createTensor(env, LongBuffer.wrap(types), shape)

        val inputs = mapOf(
            "input_ids" to inputIds,
            "attention_mask" to inputMask,
            "token_type_ids" to inputType,
        )

        val output = session.run(inputs)
        val logits = (output[0].value as Array<*>)[0] as FloatArray

        val maxL = maxOf(logits[0], logits[1])
        val expB = Math.exp((logits[0] - maxL).toDouble()).toFloat()
        val expM = Math.exp((logits[1] - maxL).toDouble()).toFloat()
        val maliciousProb = expM / (expB + expM)

        inputIds.close()
        inputMask.close()
        inputType.close()
        output.close()

        return ClassifierResult(
            maliciousProb = maliciousProb,
            isBlock = maliciousProb >= BLOCK_THRESHOLD
        )
    }

    fun close() {
        session.close()
        env.close()
    }
}
