package ru.ocr.cyrillic

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.ceil
import kotlin.math.min

/**
 * Minimal TFLite text-line recognizer for the fixed-shape model in assets.
 * The detector and polygon crop/ordering are intentionally separate.
 */
class CyrillicRecognition(context: Context) : Closeable {
    private val interpreter: Interpreter
    private val chars: List<String>
    private val allowed = (
        " АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ" +
            "абвгдеёжзийклмнопрстуфхцчшщъыьэюя" +
            "0123456789.,!?;:-()[]{}\"'«»„“”%№+/=…—–"
        ).toSet()

    private val confusables = mapOf(
        'A' to 'А', 'B' to 'В', 'C' to 'С', 'E' to 'Е', 'H' to 'Н',
        'K' to 'К', 'M' to 'М', 'O' to 'О', 'P' to 'Р', 'T' to 'Т',
        'X' to 'Х', 'Y' to 'У', 'a' to 'а', 'c' to 'с', 'e' to 'е',
        'k' to 'к', 'm' to 'м', 'o' to 'о', 'p' to 'р', 't' to 'т',
        'x' to 'х', 'y' to 'у', 'i' to 'и', 'j' to 'й', 'u' to 'и',
        'b' to 'в'
    )

    init {
        val model = context.assets.open(
            "models/tflite/cyrillic_pp-ocrv3_mobile_rec_float32.tflite"
        ).use { it.readBytes() }
        val buffer = ByteBuffer.allocateDirect(model.size).order(ByteOrder.nativeOrder())
        buffer.put(model).rewind()
        interpreter = Interpreter(buffer, Interpreter.Options().apply {
            setNumThreads(2)
        })
        chars = context.assets.open("models/dicts/cyrillic_dict.txt")
            .bufferedReader(Charsets.UTF_8).use { it.readLines() }
    }

    data class Result(
        val text: String,
        val confidence: Float,
        val certain: Boolean,
    )

    /** Recognizes one already-cropped horizontal text line. */
    fun recognize(line: Bitmap, threshold: Float = 0.90f): Result {
        val input = ByteBuffer.allocateDirect(1 * 48 * 320 * 3 * 4)
            .order(ByteOrder.nativeOrder())
        repeat(48 * 320 * 3) { input.putFloat(0f) }

        val sourceWidth = line.width.coerceAtLeast(1)
        val sourceHeight = line.height.coerceAtLeast(1)
        val resizedWidth = min(320, ceil(48.0 * sourceWidth / sourceHeight).toInt().coerceAtLeast(1))
        val scaled = Bitmap.createScaledBitmap(line, resizedWidth, 48, true)

        // TFLite input is NHWC. The original Paddle model was trained with BGR
        // order, so write blue, green, red for each pixel.
        input.rewind()
        for (y in 0 until 48) {
            for (x in 0 until 320) {
                if (x < resizedWidth) {
                    val pixel = scaled.getPixel(x, y)
                    input.putFloat((Color.blue(pixel) / 255f - 0.5f) / 0.5f)
                    input.putFloat((Color.green(pixel) / 255f - 0.5f) / 0.5f)
                    input.putFloat((Color.red(pixel) / 255f - 0.5f) / 0.5f)
                } else {
                    // Matches the Python/ONNX preprocessing: normalized zero pad.
                    input.putFloat(0f).putFloat(0f).putFloat(0f)
                }
            }
        }
        scaled.recycle()
        input.rewind()

        val output = Array(1) { Array(40) { FloatArray(165) } }
        interpreter.run(input, output)
        val decoded = decode(output[0])
        return Result(decoded.first, decoded.second, decoded.second >= threshold)
    }

    private fun decode(timeSteps: Array<FloatArray>): Pair<String, Float> {
        val out = StringBuilder()
        var previous = -1
        var confidenceSum = 0f
        var confidenceCount = 0

        for (step in timeSteps) {
            var bestIndex = 0
            var bestProbability = step[0]
            for (index in 1 until min(step.size, chars.size + 1)) {
                val char = chars[index - 1]
                if (char !in allowed) continue
                if (step[index] > bestProbability) {
                    bestIndex = index
                    bestProbability = step[index]
                }
            }
            if (bestIndex != 0 && bestIndex != previous && bestIndex - 1 < chars.size) {
                out.append(chars[bestIndex - 1])
                confidenceSum += bestProbability
                confidenceCount++
            }
            previous = bestIndex
        }

        val normalized = normalize(out.toString())
        val confidence = if (confidenceCount == 0) 0f else confidenceSum / confidenceCount
        return normalized to confidence
    }

    private fun normalize(value: String): String {
        val mapped = buildString(value.length) {
            for (char in value) append(confusables[char] ?: char)
        }
        return mapped.filter { it in allowed || it.isWhitespace() }
            .replace(Regex("[ \\t]+"), " ")
            .trim()
    }

    override fun close() = interpreter.close()
}
