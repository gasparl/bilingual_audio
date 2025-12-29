## JP Text to JP-EN Audio Pipeline

Turns a Japanese source text (e.g., a novel) into listening-first study material: sentence-aligned Japanese/English pairs, plus optional short notes on one tricky grammar point or expression.

The pipeline: split the raw text into manageable chunks → translate (and optionally annotate) sentence-by-sentence → generate per-sentence audio files you can play like an “audiobook” with JP/EN pairs.

## Scripts

### `divide.py` — split raw text into API-sized batches
- Input: `jp.txt`
- Output: `divided_output.txt`
- Splits the text into paragraphs (blank-line separated), then groups paragraphs into batches capped by `MAX_CHARS_PER_BATCH` (and max paragraph count).
- If a paragraph is too long, it is split at Japanese sentence boundaries first, then by line breaks, then by character-count as a fallback.
- Batches in the output are separated by `---`.

### `deepseek_breakdown.py` — translate + optional micro-breakdown (DeepSeek API)
- Input: `divided_output.txt`
- Output: `translated_output.json` (+ incremental `translation_progress.json`)
- For each batch: splits into sentences on `。！？`, numbers them, and sends them together for contextual translation.
- Expects a JSON array with fields: `sentence_number`, `english`, `part_to_breakdown` (JP phrase), `breakdown` (short EN explanation, optional).
- Cleans Japanese characters out of English fields and fills placeholders if a sentence is missing.
- Reads API settings from `config.json` (or environment variables), with basic retry/backoff.

### `gen_audio.py` — synthesize WAVs for JP/EN pairs (Balabolka)
- Input: `translated_output.json`
- Output folder: `audio_sentences/`
- Uses Balabolka `bal4web.exe` with Microsoft voices; auto-detects working JA male/female voices and an EN voice (if available).
- Generates multiple WAVs per sentence (where possible): sentence number (JP), Japanese sentence (male), English translation, combined breakdown audio (JP term + EN note), and an alternating JP female→male version.
- Retries transient TTS failures and rejects empty/tiny WAVs.
