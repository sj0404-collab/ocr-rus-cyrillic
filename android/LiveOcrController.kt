package ru.ocr.cyrillic

import android.graphics.Bitmap
import android.graphics.Color
import kotlin.math.abs
import kotlin.math.max

/**
 * Duplicate-frame suppression for CameraX/ImageAnalysis.
 * Feed this only a detected text-line crop for low-latency OCR.
 */
class LiveOcrController(
    private val recognizer: CyrillicRecognition,
    private val unchangedDelta: Float = 0.008f,
) {
    data class LiveResult(
        val text: String,
        val confidence: Float,
        val certain: Boolean,
        val frameId: Long,
        val changed: Boolean,
        val analyzed: Boolean,
        val reused: Boolean,
        val latencyMs: Long,
    )

    private var lastSignature: FloatArray? = null
    private var lastResult: LiveResult? = null
    private var frameId = 0L

    fun process(lineCrop: Bitmap, force: Boolean = false): LiveResult {
        val started = System.nanoTime()
        frameId++
        val signature = signature(lineCrop)
        val previous = lastSignature
        val same = !force && previous != null && meanAbsDifference(previous, signature) <= unchangedDelta
        if (same && lastResult != null) {
            val cached = lastResult!!
            return cached.copy(
                frameId = frameId,
                changed = false,
                analyzed = false,
                reused = true,
                latencyMs = (System.nanoTime() - started) / 1_000_000L,
            )
        }

        val result = recognizer.recognize(lineCrop)
        val live = LiveResult(
            text = result.text,
            confidence = result.confidence,
            certain = result.certain,
            frameId = frameId,
            changed = true,
            analyzed = true,
            reused = false,
            latencyMs = (System.nanoTime() - started) / 1_000_000L,
        )
        lastSignature = signature
        lastResult = live
        return live
    }

    fun reset() {
        lastSignature = null
        lastResult = null
        frameId = 0L
    }

    private fun signature(bitmap: Bitmap): FloatArray {
        val width = 48
        val height = 32
        val result = FloatArray(width * height)
        var index = 0
        var sum = 0f
        for (y in 0 until height) {
            val sourceY = y * bitmap.height / height
            for (x in 0 until width) {
                val sourceX = x * bitmap.width / width
                val pixel = bitmap.getPixel(sourceX, sourceY)
                val value = (0.299f * Color.red(pixel) + 0.587f * Color.green(pixel) + 0.114f * Color.blue(pixel)) / 255f
                result[index++] = value
                sum += value
            }
        }
        val mean = sum / result.size
        var scale = 0f
        for (i in result.indices) {
            result[i] -= mean
            scale += abs(result[i])
        }
        scale = max(scale / result.size, 0.000001f)
        for (i in result.indices) result[i] /= scale
        return result
    }

    private fun meanAbsDifference(a: FloatArray, b: FloatArray): Float {
        var total = 0f
        for (i in a.indices) total += abs(a[i] - b[i])
        return total / a.size
    }
}
