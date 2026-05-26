#!/usr/bin/env python3
"""
DeepSeek English-to-Japanese Translator
Batch-based processing with progress saving
Creates Japanese-first study material from English source text.

Input:
  divided_output.txt

Output:
  translated_output.json
  translation_progress.json

Final sentence schema is kept compatible with the existing audio generator:
  japanese          = simple natural Japanese translation
  english           = original English sentence
  english_literal   = literal English translation of the Japanese structure
  part_to_breakdown_1..3 = Japanese phrase/grammar point
  breakdown_1..3         = brief English explanation
"""

import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

# ===================== CONFIGURATION =====================
CONFIG_FILE = "config.json"
DIVIDED_FILE = "divided_output.txt"
OUTPUT_JSON = "translated_output.json"
PROGRESS_FILE = "translation_progress.json"

# Default API settings
DEFAULT_API_BASE = "https://api.deepseek.com/v1/chat/completions"
MODEL_NAME = "deepseek-chat"

# Rate limiting / robustness
BATCH_DELAY = 1.0
MAX_RETRIES = 5
REQUEST_TIMEOUT = 90
# =========================================================


class Config:
    """Configuration manager."""

    def __init__(self) -> None:
        self.api_key: Optional[str] = None
        self.api_base: str = DEFAULT_API_BASE
        self.load_config()

    def load_config(self) -> bool:
        """Load configuration from config.json or environment variables."""
        config_paths = [
            Path(CONFIG_FILE),
            Path(__file__).parent / CONFIG_FILE,
            Path.cwd() / CONFIG_FILE,
        ]

        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        config = json.load(f)

                    self.api_key = config.get("DEEPSEEK_API_KEY") or config.get("OPENAI_API_KEY")

                    api_base = config.get("DEEPSEEK_API_BASE")
                    if api_base:
                        if api_base.endswith("/chat/completions"):
                            self.api_base = api_base
                        elif api_base.endswith("/v1"):
                            self.api_base = f"{api_base}/chat/completions"
                        else:
                            self.api_base = f"{api_base}/v1/chat/completions"

                    print(f"✓ Config loaded from {config_path}")
                    return bool(self.api_key)

                except Exception as e:
                    print(f"⚠️ Error loading config from {config_path}: {e}")

        self.api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            print("❌ No API key found")
            return False

        print("✓ Using API key from environment variable")
        return True


def read_batches() -> List[str]:
    """Read divided batches from file."""
    try:
        with open(DIVIDED_FILE, "r", encoding="utf-8") as f:
            content = f.read()

        batches = [b.strip() for b in content.split("\n\n---\n\n") if b.strip()]
        print(f"✓ Read {len(batches)} batches")
        return batches
    except FileNotFoundError:
        print(f"❌ Error: File '{DIVIDED_FILE}' not found")
        return []
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return []


def normalize_english_source(text: str) -> str:
    """Normalize English input while preserving readable text."""
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "\ufeff": "",
        "\u00a0": " ",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "—": "-",
        "–": "-",
        "…": "...",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


COMMON_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st",
    "e.g", "i.e", "etc", "vs", "fig", "no", "vol", "ch",
    "a.m", "p.m", "u.s", "u.k", "u.n",
}


def _looks_like_abbreviation(text_before_period: str) -> bool:
    """Return True if a period probably belongs to an abbreviation."""
    tail = text_before_period.strip().split()[-1:] or [""]
    token = tail[0].strip("\"'()[]{}")
    token_lower = token.lower().rstrip(".")

    if token_lower in COMMON_ABBREVIATIONS:
        return True

    # Initials or acronym patterns such as J. R. R. or U.S.
    if re.search(r"(?:\b[A-Za-z]\.){1,}$", text_before_period.strip()):
        return True
    if re.search(r"\b(?:[A-Z]\.){2,}$", text_before_period.strip()):
        return True

    return False


def split_english_sentences(text: str) -> List[Tuple[int, str]]:
    """
    Split English text into sentences.
    Keeps simple handling for quotes/parentheses and avoids common abbreviations.
    """
    text = normalize_english_source(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    sentences: List[str] = []

    for paragraph in paragraphs:
        current = ""
        quote_depth = 0
        paren_depth = 0
        bracket_depth = 0

        i = 0
        while i < len(paragraph):
            char = paragraph[i]
            current += char

            if char == '"':
                quote_depth = 1 - quote_depth
            elif char in "(":
                paren_depth += 1
            elif char in ")":
                paren_depth = max(0, paren_depth - 1)
            elif char in "[":
                bracket_depth += 1
            elif char in "]":
                bracket_depth = max(0, bracket_depth - 1)

            inside_nested = paren_depth > 0 or bracket_depth > 0

            if char in ".!?" and not inside_nested:
                if char == "." and _looks_like_abbreviation(current):
                    i += 1
                    continue

                # Attach closing quotes/brackets after punctuation.
                while i + 1 < len(paragraph) and paragraph[i + 1] in "'\")]}":
                    i += 1
                    current += paragraph[i]

                sentences.append(current.strip())
                current = ""

            # Emergency split for very long segments.
            elif len(current) > 350 and not inside_nested:
                split_pos = -1
                for lookback in range(min(180, len(current)), 0, -1):
                    pos = len(current) - lookback
                    if current[pos] in ";,:" and pos > 80:
                        split_pos = pos + 1
                        break

                if split_pos != -1:
                    before = current[:split_pos].strip()
                    after = current[split_pos:].strip()
                    if before:
                        sentences.append(before)
                    current = after

            i += 1

        if current.strip():
            sentences.append(current.strip())

    # Merge very short fragments into neighboring sentences.
    processed: List[str] = []
    i = 0
    while i < len(sentences):
        sent = sentences[i].strip()
        if not sent:
            i += 1
            continue

        is_fragment = len(sent) < 12 and not sent.endswith((".", "!", "?", '"', "'"))
        if is_fragment and processed and len(processed[-1]) + len(sent) < 220:
            processed[-1] = f"{processed[-1]} {sent}".strip()
        elif is_fragment and i + 1 < len(sentences):
            sentences[i + 1] = f"{sent} {sentences[i + 1].lstrip()}"
        else:
            processed.append(sent)
        i += 1

    return [(i + 1, sent) for i, sent in enumerate(processed) if sent.strip()]


def clean_english_text(text: str, is_breakdown: bool = False) -> str:
    """Clean English fields for safe TTS/JSON output."""
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", str(text))
    replacements = {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "—": "-",
        "–": "-",
        "…": "...",
        "\u00a0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    if is_breakdown:
        text = text.replace('"', "")

    # Keep English output ASCII-only, as in the original pipeline.
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    text = text.replace('\\"', "").replace("\\'", "'")
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    text = " ".join(text.split())
    text = re.sub(r'"{2,}', "", text)
    text = re.sub(r"'{2,}", "'", text)
    return text.strip()


def clean_japanese_text(text: str) -> str:
    """Light cleanup for Japanese output. Keeps Japanese characters intact."""
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", " ", text)
    return text.strip()


def extract_json_array(response: str) -> Optional[List[Dict[str, str]]]:
    """Extract a JSON array from a model response."""
    if not response:
        return None

    text = response.strip()

    # Remove Markdown code fences if present.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
        return data if isinstance(data, list) else None
    except json.JSONDecodeError:
        pass

    # Fallback: find the first plausible JSON array.
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(text[start:end + 1])
            return data if isinstance(data, list) else None
        except json.JSONDecodeError:
            return None

    return None


class BatchTranslator:
    """Translator that processes entire batches for context."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.session = requests.Session()

    def call_api(self, messages: List[Dict[str, str]], retry_count: int = 0) -> Optional[str]:
        """Call DeepSeek API with retry logic."""
        if retry_count >= MAX_RETRIES:
            return None

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 8000,
            "stream": False,
        }

        try:
            response = self.session.post(
                self.config.api_base,
                json=data,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code == 429:
                wait_time = 5 * (retry_count + 1)
                print(f"⚠️ Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
                return self.call_api(messages, retry_count + 1)

            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()

        except requests.exceptions.RequestException as e:
            print(f"⚠️ API error ({retry_count + 1}/{MAX_RETRIES}): {type(e).__name__}")
            if retry_count < MAX_RETRIES - 1:
                wait_time = 2 ** retry_count
                time.sleep(wait_time)
                return self.call_api(messages, retry_count + 1)
            return None
        except Exception as e:
            print(f"⚠️ Unexpected error: {e}")
            return None

    def translate_batch(self, batch_text: str, batch_number: int) -> Optional[List[Dict[str, str]]]:
        """
        Translate an English batch into simple Japanese and extract sentence records.
        Returns list of sentence dicts or None if failed.
        """
        sentences = split_english_sentences(batch_text)
        if not sentences:
            print(f"  Batch {batch_number}: No sentences found")
            return None

        print(f"  Batch {batch_number}: {len(sentences)} sentences")
        numbered_sentences = "\n".join([f"{num}. {text}" for num, text in sentences])

        system_prompt = """You are an expert English-to-Japanese translator creating Japanese learning audio material.

TASK:
Translate each English sentence into Japanese, maintaining context across sentences. For each sentence, provide:
1) A simple, natural Japanese translation.
   - The Japanese should be deliberately learner-friendly, roughly JLPT N4-N3 when possible.
   - Prefer short, clear sentence structure and common vocabulary.
   - Avoid rare kanji, literary wording, archaic expressions, and overly compressed phrasing.
   - If the English sentence is long or complex, you may split it into two short Japanese sentences.
   - Preserve the original meaning faithfully.
2) A closer, learning-oriented English literal translation of your Japanese sentence.
   - This should roughly follow the Japanese structure and word order as far as sensible.
   - It should still be understandable English.
   - It can be empty if the sentence is very short or if it would not add value.
3) Up to THREE (0-3) notable Japanese items worth explaining:
   - Identify the specific Japanese word/phrase/grammar point.
   - Give a concise English explanation, max 2-3 sentences each.

REQUIREMENTS:
* Keep the English source meaning accurate.
* For simple everyday sentences, leave all breakdown fields empty.
* Use only ASCII characters in English output fields: english_literal and breakdown_*.
* In breakdown explanations, use romanization only when referring to Japanese pronunciation.
* In part_to_breakdown fields, keep the original Japanese characters from your translation.

OUTPUT FORMAT:
Return a JSON array. Each object must be:
{
  "sentence_number": (number from input),
  "japanese": "Simple natural Japanese translation",
  "english_literal": "Literal English translation of the Japanese structure OR empty string",
  "part_to_breakdown_1": "Japanese word/phrase being explained OR empty string",
  "breakdown_1": "Brief English explanation OR empty string",
  "part_to_breakdown_2": "Japanese word/phrase being explained OR empty string",
  "breakdown_2": "Brief English explanation OR empty string",
  "part_to_breakdown_3": "Japanese word/phrase being explained OR empty string",
  "breakdown_3": "Brief English explanation OR empty string"
}

EXAMPLES:
Example 1 (simple - no breakdowns):
Input: "The weather is nice today, isn't it?"
Output: {
  "sentence_number": 1,
  "japanese": "今日はいい天気ですね。",
  "english_literal": "Today is nice weather, isn't it?",
  "part_to_breakdown_1": "",
  "breakdown_1": "",
  "part_to_breakdown_2": "",
  "breakdown_2": "",
  "part_to_breakdown_3": "",
  "breakdown_3": ""
}

Example 2 (one useful expression):
Input: "I have to submit the report by tomorrow."
Output: {
  "sentence_number": 2,
  "japanese": "明日までにレポートを出さなければなりません。",
  "english_literal": "By tomorrow, the report I must submit.",
  "part_to_breakdown_1": "出さなければなりません",
  "breakdown_1": "This is the nakereba naranai obligation pattern, meaning 'must' or 'have to.' Compared with naito ikenai, it sounds more formal and is more common in writing.",
  "part_to_breakdown_2": "",
  "breakdown_2": "",
  "part_to_breakdown_3": "",
  "breakdown_3": ""
}

Example 3 (long English made simpler in Japanese):
Input: "Although I wanted to answer right away, I could not find the right words."
Output: {
  "sentence_number": 3,
  "japanese": "すぐに答えたかったです。でも、いい言葉が見つかりませんでした。",
  "english_literal": "Right away I wanted to answer. But good words were not found.",
  "part_to_breakdown_1": "答えたかったです",
  "breakdown_1": "This is the past want-to form of kotaeru: wanted to answer. The ending desu makes the sentence polite.",
  "part_to_breakdown_2": "見つかりませんでした",
  "breakdown_2": "Were not found. Mitsukaru is an intransitive verb meaning to be found.",
  "part_to_breakdown_3": "",
  "breakdown_3": ""
}

Remember: Maintain consistency and context across all sentences in the batch."""

        user_prompt = f"""Translate these English sentences into simple natural Japanese.

ENGLISH SENTENCES:
{numbered_sentences}

Provide Japanese translations that work together as a coherent passage.
Return a JSON array with one object for each numbered sentence.
Follow the exact output format with sentence_number, japanese, english_literal, and up to 3 breakdown pairs."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        print("  Translating...", end="", flush=True)
        response = self.call_api(messages)

        if not response:
            print(" ❌ Failed")
            return None

        try:
            translations = extract_json_array(response)
            if not translations:
                print(" ❌ JSON parse error")
                print(f"  Response preview: {response[:200]}...")
                return None

            results: List[Dict[str, str]] = []
            for orig_num, orig_text in sentences:
                translation = None
                for t in translations:
                    if isinstance(t, dict) and t.get("sentence_number") == orig_num:
                        translation = t
                        break

                if not translation:
                    print(f"    ⚠️ Sentence {orig_num}: No translation dict found")
                    results.append(self._empty_result(orig_num, orig_text))
                    continue

                japanese = clean_japanese_text(translation.get("japanese", ""))
                english_literal = clean_english_text(translation.get("english_literal", ""), is_breakdown=False)

                if not japanese:
                    print(f"    ⚠️ Sentence {orig_num}: Empty Japanese translation")

                breakdown_data: Dict[str, str] = {}
                for i in range(1, 4):
                    part_key = f"part_to_breakdown_{i}"
                    breakdown_key = f"breakdown_{i}"

                    part = clean_japanese_text(translation.get(part_key, ""))
                    breakdown = clean_english_text(translation.get(breakdown_key, ""), is_breakdown=True)

                    if breakdown and not part:
                        print(f"    ⚠️ Sentence {orig_num}: Breakdown {i} has no part_to_breakdown")

                    breakdown_data[part_key] = part
                    breakdown_data[breakdown_key] = breakdown

                results.append({
                    "sentence_number": orig_num,
                    "japanese": japanese,
                    "english": clean_english_text(orig_text, is_breakdown=False),
                    "english_literal": english_literal,
                    **breakdown_data,
                })

            print(" ✓ Success")
            return results

        except Exception as e:
            print(f" ❌ Error: {e}")
            return None

    @staticmethod
    def _empty_result(sentence_number: int, english_source: str) -> Dict[str, str]:
        return {
            "sentence_number": sentence_number,
            "japanese": "",
            "english": clean_english_text(english_source, is_breakdown=False),
            "english_literal": "",
            "part_to_breakdown_1": "",
            "breakdown_1": "",
            "part_to_breakdown_2": "",
            "breakdown_2": "",
            "part_to_breakdown_3": "",
            "breakdown_3": "",
        }


def save_progress(sentences: Dict[str, Dict[str, str]], filename: str = PROGRESS_FILE) -> bool:
    """Save progress to file."""
    if not sentences:
        return False

    progress_data = {
        "last_saved": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_sentences": len(sentences),
        "sentences": sentences,
    }

    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def process_all_batches(translator: BatchTranslator) -> Dict[str, Dict[str, str]]:
    """Process all batches and return sentences dict."""
    batches = read_batches()
    if not batches:
        return {}

    all_sentences: Dict[str, Dict[str, str]] = {}
    global_sentence_counter = 1
    start_time = time.time()
    batch_times: List[float] = []

    print(f"\n🚀 Processing {len(batches)} batches")
    print("-" * 50)

    successful_batches = 0
    failed_batches = 0

    for batch_num, batch_text in enumerate(batches, 1):
        batch_start = time.time()
        print(f"\n[{batch_num}/{len(batches)}] ", end="")

        batch_results = translator.translate_batch(batch_text, batch_num)

        if batch_results:
            for result in batch_results:
                sentence_data = {
                    "japanese": result.get("japanese", ""),
                    "english": result.get("english", ""),
                    "english_literal": result.get("english_literal", ""),
                    "original_batch_sentence_num": result.get("sentence_number", 0),
                    "batch_number": batch_num,
                }

                for i in range(1, 4):
                    sentence_data[f"part_to_breakdown_{i}"] = result.get(f"part_to_breakdown_{i}", "")
                    sentence_data[f"breakdown_{i}"] = result.get(f"breakdown_{i}", "")

                all_sentences[str(global_sentence_counter)] = sentence_data
                global_sentence_counter += 1

            successful_batches += 1

            if batch_num % 5 == 0:
                if save_progress(all_sentences):
                    print(f"  💾 Progress saved ({len(all_sentences)} sentences)")
        else:
            failed_batches += 1

        batch_time = time.time() - batch_start
        batch_times.append(batch_time)

        if batch_times and batch_num < len(batches):
            avg_time = sum(batch_times) / len(batch_times)
            remaining = avg_time * (len(batches) - batch_num)

            if remaining > 3600:
                eta = f"{remaining / 3600:.1f} hours"
            elif remaining > 60:
                eta = f"{remaining / 60:.1f} minutes"
            else:
                eta = f"{remaining:.0f} seconds"

            print(f"  ⏱️  Batch: {batch_time:.1f}s | Remaining: ~{eta}")

        if batch_num < len(batches):
            time.sleep(BATCH_DELAY)

    elapsed = time.time() - start_time
    save_progress(all_sentences)

    print("\n" + "=" * 50)
    print(f"✅ Processing completed in {elapsed:.1f} seconds")
    print("📊 Batch statistics:")
    print(f"   Successful: {successful_batches}")
    print(f"   Failed: {failed_batches}")
    print(f"   Total sentences: {len(all_sentences)}")
    if batch_times:
        print(f"   Average batch time: {sum(batch_times) / len(batch_times):.1f}s")

    return all_sentences


def save_final_results(sentences: Dict[str, Dict[str, str]], filename: str = OUTPUT_JSON) -> bool:
    """Save final results to JSON file."""
    if not sentences:
        print("❌ No sentences to save")
        return False

    breakdown_counts = {1: 0, 2: 0, 3: 0}
    breakdown_types: Dict[str, int] = {}

    for s in sentences.values():
        breakdown_count = 0
        for i in range(1, 4):
            if s.get(f"breakdown_{i}") and s.get(f"breakdown_{i}").strip():
                breakdown_count += 1
                part = s.get(f"part_to_breakdown_{i}", "")
                if part:
                    breakdown_types[part] = breakdown_types.get(part, 0) + 1

        if breakdown_count > 0:
            breakdown_counts[breakdown_count] = breakdown_counts.get(breakdown_count, 0) + 1

    total_with_breakdown = sum(breakdown_counts.values())

    output = {
        "metadata": {
            "total_sentences": len(sentences),
            "sentences_with_breakdown": total_with_breakdown,
            "breakdown_distribution": breakdown_counts,
            "breakdown_types_count": len(breakdown_types),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "processing_method": "Batch-based English-to-Japanese translation for context",
            "translation_approach": "Simple natural Japanese, learner-friendly but not childish",
            "literal_field": "english_literal is a closer English translation of the Japanese structure, not an explanation",
            "breakdown_structure": "Up to 3 breakdown pairs per sentence (part_to_breakdown_1-3, breakdown_1-3)",
            "character_rule": "English fields are ASCII-only; Japanese fields preserve Japanese characters",
            "output_fields": "Audio-compatible schema: japanese target line, english original line, english_literal literal line",
        },
        "sentences": sentences,
    }

    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"💾 Final results saved to {filename}")
        return True
    except Exception as e:
        print(f"❌ Error saving final results: {e}")
        return False


def main() -> None:
    """Main function."""
    print("=" * 60)
    print("English to Japanese Translator")
    print("Batch Processing with Simple Japanese + Literal English + Breakdowns")
    print("Output is compatible with the Japanese-first audio generator")
    print("=" * 60)

    config = Config()
    if not config.load_config():
        sys.exit(1)

    translator = BatchTranslator(config)
    sentences = process_all_batches(translator)

    if not sentences:
        print("❌ No sentences were processed.")
        sys.exit(1)

    if not save_final_results(sentences):
        sys.exit(1)

    total = len(sentences)
    breakdown_counts = {1: 0, 2: 0, 3: 0}

    for s in sentences.values():
        breakdown_count = 0
        for i in range(1, 4):
            if s.get(f"breakdown_{i}") and s.get(f"breakdown_{i}").strip():
                breakdown_count += 1
        if breakdown_count > 0:
            breakdown_counts[breakdown_count] = breakdown_counts.get(breakdown_count, 0) + 1

    total_with_breakdown = sum(breakdown_counts.values())

    print("\n" + "=" * 60)
    print("📊 FINAL SUMMARY")
    print("=" * 60)
    print(f"Total sentences: {total}")
    print(f"Sentences with breakdown: {total_with_breakdown}")
    print(f"Sentences without breakdown: {total - total_with_breakdown}")
    print("\nBreakdown distribution:")
    print(f"  1 breakdown: {breakdown_counts.get(1, 0)} sentences")
    print(f"  2 breakdowns: {breakdown_counts.get(2, 0)} sentences")
    print(f"  3 breakdowns: {breakdown_counts.get(3, 0)} sentences")

    if total > 0:
        breakdown_rate = (total_with_breakdown / total) * 100
        print(f"Breakdown rate: {breakdown_rate:.1f}%")

    if sentences:
        print("\n📄 SAMPLE OUTPUT:")
        shown = 0
        for key, s in sentences.items():
            if shown >= 3:
                break

            breakdown_count = sum(
                1 for i in range(1, 4)
                if s.get(f"breakdown_{i}") and s.get(f"breakdown_{i}").strip()
            )

            print(f"\n{key}. ({breakdown_count} BREAKDOWN{'S' if breakdown_count != 1 else ''})")
            print(f"   English source: {s.get('english', '')[:70]}...")
            print(f"   Japanese: {s.get('japanese', '')[:70]}...")
            print(f"   Literal: {s.get('english_literal', '')[:70]}...")

            for i in range(1, breakdown_count + 1):
                part = s.get(f"part_to_breakdown_{i}", "")
                breakdown = s.get(f"breakdown_{i}", "")
                print(f"   Part {i}: '{part}'")
                print(f"   Breakdown {i}: {breakdown[:70]}...")

            shown += 1

    if Path(PROGRESS_FILE).exists():
        print(f"\n📁 Progress saved to: {PROGRESS_FILE}")


if __name__ == "__main__":
    main()