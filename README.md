## JP Text to JP-EN Audio Pipeline

Turns a Japanese source text (e.g., a novel) into listening-first study material: sentence-aligned Japanese/English pairs, plus optional short notes on up to three tricky grammar points or expressions.

The pipeline: clean + split the raw text into manageable chunks → translate (and optionally annotate) sentence-by-sentence with batch context → generate per-sentence audio files you can play like an “audiobook” with JP/EN pairs (plus optional breakdown audio).

## Scripts

### `divide.py` — clean + split raw text into API-sized batches
- Input: `jp.txt`
- Output: `divided_output.txt`
- Cleans the text first by removing ruby/furigana and transcription markup (e.g., `《...》`, `｜`, `［＃...］`).
- Reads with UTF-8 first, then falls back across common Japanese encodings if needed.
- Splits the text into paragraphs (blank-line separated), then groups paragraphs into batches capped by `MAX_CHARS_PER_BATCH` (and `MAX_PARAGRAPHS_PER_BATCH`).
- If a paragraph is too long, it is split at Japanese sentence boundaries first, then by line breaks, then by character-count as a fallback.
- Batches in the output are separated by `---` (specifically `\n\n---\n\n`).

### `deepseek_breakdown.py` — translate + optional micro-breakdowns (DeepSeek API)
- Input: `divided_output.txt`
- Output: `translated_output.json` (+ incremental `translation_progress.json`)
- For each batch: splits into sentences and numbers them, then sends them together for contextual translation. Sentence splitting avoids breaking on `。！？` inside Japanese quotes like `「...」` / `『...』`.
- Expects a JSON array with fields: `sentence_number`, `english`, and up to 3 optional breakdown pairs: `part_to_breakdown_1..3` and `breakdown_1..3`.
- Enforces that English/breakdown fields contain no Japanese characters; breakdown “parts” should be romanized.
- Saves the final output as a dict with `metadata` plus a `sentences` object keyed by global sentence index (each entry includes `japanese`, `english`, `batch_number`, and breakdown fields).
- Reads API settings from `config.json` (or environment variables), with basic retry/backoff.

### `gen_audio.py` — synthesize audio for JP/EN pairs (Balabolka)
- Input: `translated_output.json`
- Output folder: `audio_sentences/`
- Uses Balabolka `bal4web.exe` with Microsoft voices; auto-detects working JA male/female voices and tries to find an EN voice.
- Generates audio per sentence as MP3s (WAVs are generated as intermediates and then converted/cleaned up).
- Generates multiple segments per sentence (where possible): sentence number (JP), Japanese sentence (male), English translation, combined breakdown audio (up to 3 JP terms + EN notes), and an alternating JP female→male version.
- Uses configurable pauses/silence and per-segment speech-rate settings; retries transient TTS failures.

## Example files included

- `jp.txt` — example input Japanese text (No Longer Human / 人間失格 by Osamu Dazai)
- `divided_output.txt` — example output from `divide.py`
- `translated_output.json` — example output from `deepseek_breakdown.py`

## Notes

- This repo is not actively maintained.
- Upload the scripts to an AI assistant if you want a deeper, line-by-line explanation.

## License

GPL.
