#!/usr/bin/env python3
"""
DeepSeek Japanese-to-English Translator
Batch-based processing with progress saving
Enhanced breakdown structure with up to 3 expressions + literal translation
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

# Rate limiting
BATCH_DELAY = 1.0
MAX_RETRIES = 5  # One initial attempt + one retry
# =========================================================


class Config:
    """Configuration manager."""
    
    def __init__(self) -> None:
        self.api_key: Optional[str] = None
        self.api_base: str = DEFAULT_API_BASE
        self.load_config()
    
    def load_config(self) -> bool:
        """Load configuration from config.json."""
        config_paths = [
            Path(CONFIG_FILE),
            Path(__file__).parent / CONFIG_FILE,
            Path.cwd() / CONFIG_FILE
        ]
        
        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
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
                    return True
                    
                except Exception as e:
                    print(f"⚠️ Error loading config from {config_path}: {e}")
        
        # Check environment variables
        self.api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            print("❌ No API key found")
            return False
        
        print("✓ Using API key from environment variable")
        return True


def read_batches() -> List[str]:
    """Read divided batches from file."""
    try:
        with open(DIVIDED_FILE, 'r', encoding='utf-8') as f:
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


def split_japanese_sentences(
    text: str,
    aggressive: bool = True,
    max_clause_length: int = 120
) -> List[Tuple[int, str]]:
    """
    Split Japanese text into sentences with intelligent hierarchical splitting.

    Args:
        text: Japanese text to split
        aggressive: If True, splits at more boundaries for better translation
        max_clause_length: Maximum characters before forcing split at clause boundary

    Returns:
        List of (sentence_number, sentence) tuples (1-based numbering)
    """
    if not text or not text.strip():
        return []

    # 1. Split into paragraphs (one or more newlines with optional spaces)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    all_sentences: List[str] = []
    for paragraph in paragraphs:
        all_sentences.extend(_split_paragraph(paragraph, aggressive, max_clause_length))

    # Number sentences starting from 1
    return [(i + 1, sent.strip()) for i, sent in enumerate(all_sentences) if sent.strip()]


def _split_paragraph(paragraph: str, aggressive: bool, max_clause_length: int) -> List[str]:
    """Split a single paragraph into sentences."""
    sentences: List[str] = []
    current = ""

    # Track nesting levels
    quote_depth = 0
    paren_depth = 0
    bracket_depth = 0

    i = 0
    while i < len(paragraph):
        char = paragraph[i]
        current += char

        # Update nesting levels
        if char == "「" or char == "『":
            quote_depth += 1
        elif char == "」" or char == "』":
            quote_depth = max(0, quote_depth - 1)
        elif char in "（(":
            paren_depth += 1
        elif char in "）)":
            paren_depth = max(0, paren_depth - 1)
        elif char in "【［〔｛[{":
            bracket_depth += 1
        elif char in "】］〕｝]}":
            bracket_depth = max(0, bracket_depth - 1)

        inside_nested = quote_depth > 0 or paren_depth > 0 or bracket_depth > 0

        # 1. Split at sentence enders (。, ！, ？)
        if char in "。！？" and not inside_nested:
            # Handle consecutive enders (！！, ！？, etc.)
            while i + 1 < len(paragraph) and paragraph[i + 1] in "。！？":
                i += 1
                current += paragraph[i]

            sentences.append(current.strip())
            current = ""
            i += 1
            continue

        # 2. Split at ellipsis (especially multiple ...)
        if char == "…" and not inside_nested:
            # Count consecutive ellipsis
            ellipsis_count = 1
            while i + 1 < len(paragraph) and paragraph[i + 1] == "…":
                i += 1
                current += paragraph[i]
                ellipsis_count += 1

            # Check if we should split
            should_split = False

            # Look for conjunctions after ellipsis
            next_text = paragraph[i + 1 : i + 10]
            conjunctions = ["また", "そして", "しかし", "だが", "それで", "だから"]
            if any(next_text.startswith(c) for c in conjunctions):
                should_split = True


            # Split at multiple ellipsis in aggressive mode
            if aggressive and ellipsis_count >= 2:
                should_split = True
            # Split at single ellipsis if we have enough content
            elif aggressive and ellipsis_count == 1 and len(current.strip()) > 20:
                should_split = True

            if should_split:
                sentences.append(current.strip())
                current = ""
                i += 1  # Move past the last ellipsis character we just processed
                continue  # Skip the i += 1 at end of loop

        # 3. Split at clause boundaries in long sentences (aggressive mode)
        if aggressive and char == "、" and not inside_nested and len(current) > max_clause_length:
            # Common clause markers
            markers = ["し、", "が、", "ので、", "から、", "けれど、", "ながら、", "たり、"]

            for marker in markers:
                if current.endswith(marker):
                    # Ensure we have meaningful content before marker
                    content_before = current[:-len(marker)]
                    if len(content_before.strip()) > 15:
                        sentences.append(current.strip())
                        current = ""
                    break

        # 4. Emergency split for very long segments
        if len(current) > 250 and not inside_nested:
            # Look back for a good split point
            split_pos = -1
            # Try ellipsis first, then comma
            for lookback in range(min(150, len(current)), 0, -1):
                pos = len(current) - lookback
                if current[pos] == "…" and pos > 50:
                    split_pos = pos + 1
                    break
            
            if split_pos == -1:  # No ellipsis found
                for lookback in range(min(150, len(current)), 0, -1):
                    pos = len(current) - lookback
                    if current[pos] == "、" and pos > 50:
                        split_pos = pos + 1
                        break

            if split_pos != -1:
                before = current[:split_pos].strip()
                after = current[split_pos:].strip()
                if before and len(before) > 30:
                    sentences.append(before)
                    current = after

        i += 1

    # Add remaining text
    if current.strip():
        sentences.append(current.strip())

    # Merge very short fragments
    if sentences:
        processed: List[str] = []
        i = 0
        while i < len(sentences):
            sent = sentences[i].strip()
            if not sent:
                i += 1
                continue
    
            is_fragment = (
                len(sent) < 15 and
                not sent.endswith("…") and
                not sent.endswith(("。", "！", "？"))
            )
    
            if is_fragment:
                prev_ends_sentence = bool(processed) and processed[-1].endswith(("。", "！", "？"))
                prev_ends_pause = bool(processed) and processed[-1].endswith("…")
    
                # Prefer merging backward only if it won't create "。だから" style output
                if processed and not prev_ends_sentence and not prev_ends_pause and (len(processed[-1]) + len(sent) < 150):
                    processed[-1] = processed[-1] + sent
                # Otherwise merge forward into the next segment if possible
                elif i + 1 < len(sentences):
                    sentences[i + 1] = (sent + sentences[i + 1].lstrip())
                else:
                    processed.append(sent)
    
                i += 1
                continue
    
            processed.append(sent)
            i += 1
    
        return processed

    return sentences


def clean_english_text(text: str, is_breakdown: bool = False) -> str:
    """
    Clean English text to ensure it contains only safe ASCII characters.
    
    Args:
        text: The text to clean
        is_breakdown: If True, remove all double quotes (for breakdown fields)
    """
    if not text:
        return ""
    
    # First, normalize Unicode characters
    text = unicodedata.normalize('NFKC', text)
    
    # Replace common punctuation with ASCII equivalents
    replacements = {
        '「': '"', '」': '"',   # Japanese quotes to ASCII quotes
        '『': '"', '』': '"',   # Japanese double quotes
        '、': ',', '。': '.',   # Japanese punctuation
        '・': '-', '〜': '~',   # Other Japanese chars
        'ー': '-', '…': '...',  # More replacements
        '―': '-', '‥': '..',
        '—': '-', '–': '-',     # Various dashes to hyphen
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # If this is a breakdown field, remove ALL double quotes
    # Breakdowns should explain, not quote Japanese phrases
    if is_breakdown:
        text = text.replace('"', '')
    
    # Remove any remaining non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    # Clean up quotes - remove escaped quotes completely
    text = text.replace('\\"', '').replace("\\'", "'")
    
    # Remove any control characters except newline and tab
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Clean up whitespace
    text = ' '.join(text.split())
    
    # Final cleanup: ensure no problematic quote sequences
    text = re.sub(r'"{2,}', '', text)  # Remove multiple consecutive quotes
    text = re.sub(r"'{2,}", "'", text)  # Remove multiple consecutive apostrophes
    
    return text.strip()


class BatchTranslator:
    """Translator that processes entire batches for context."""
    
    def __init__(self, config: Config) -> None:
        self.config = config
        self.session = requests.Session()
        self.session.timeout = 90
    
    def call_api(self, messages: List[Dict[str, str]], retry_count: int = 0) -> Optional[str]:
        """Call DeepSeek API with retry logic."""
        if retry_count >= MAX_RETRIES:
            return None
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 8000,
            "stream": False
        }
        
        try:
            response = self.session.post(
                self.config.api_base,
                json=data,
                headers=headers,
                timeout=90
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
        Translate an entire batch and extract individual sentences.
        Returns list of sentence dicts or None if failed.
        """
        # Split the batch into sentences
        sentences = split_japanese_sentences(batch_text)
        if not sentences:
            print(f"  Batch {batch_number}: No sentences found")
            return None
        
        print(f"  Batch {batch_number}: {len(sentences)} sentences")
        
        # Create numbered sentences for the prompt
        numbered_sentences = "\n".join([f"{num}. {text}" for num, text in sentences])
        
        # Optimized prompt with up to 3 breakdown expressions + literal translation
        system_prompt = """You are an expert Japanese-to-English translator.
        
        TASK:
        Translate each Japanese sentence to English, maintaining context across sentences. For each sentence, provide:
        1) A faithful English translation, as close to the original meaning as possible.
        2) An even closer, learning-oriented English translation that roughly follows Japanese structure and word order as far as it is sensible.
           - This should still be easily comprehensible in English. 
           - In case of long/complex sentences, this is less important, please prioritize readability and natural flow in English.
           - This can be skipped if the sentence is very short, such as a single word.
        3) Up to THREE (0-3) notable items worth explaining (grammar points, advanced expressions, or cultural references):
           - Identify the specific Japanese word/phrase for each.
           - Give a concise explanation (max 3-4 sentences each).
        
        REQUIREMENTS:
        * If context is missing (e.g., unclear subject), keep it neutral rather than guessing.
        * For very simple everyday sentences, leave all breakdown fields empty.
        * Use only ASCII characters in English output (no special Unicode quotes or punctuation).
        * In breakdown explanations, use romanization only (e.g., "hara-guroi" not "腹黒い").
        * In part_to_breakdown fields, keep the original Japanese characters.
        
        OUTPUT FORMAT:
        Return a JSON array. Each object must be:
        {
          "sentence_number": (number from input),
          "english": "English translation",
          "english_literal": "English translation closer to Japanese word order",
          "part_to_breakdown_1": "Japanese word/phrase being explained OR empty string",
          "breakdown_1": "Brief explanation OR empty string",
          "part_to_breakdown_2": "Japanese word/phrase being explained OR empty string",
          "breakdown_2": "Brief explanation OR empty string",
          "part_to_breakdown_3": "Japanese word/phrase being explained OR empty string",
          "breakdown_3": "Brief explanation OR empty string"
        }
        
        IMPORTANT:
        * Fill breakdowns sequentially (1, then 2, then 3). Do not skip numbers.
        * If fewer than 3 breakdowns apply, leave the remaining part_to_breakdown_* and breakdown_* fields empty.
        
        EXAMPLES:
        Example 1 (simple - no breakdowns):
        Input: "今日はいい天気ですね。"
        Output: {
          "sentence_number": 1,
          "english": "The weather is nice today, isn't it?",
          "english_literal": "",
          "part_to_breakdown_1": "",
          "breakdown_1": "",
          "part_to_breakdown_2": "",
          "breakdown_2": "",
          "part_to_breakdown_3": "",
          "breakdown_3": ""
        }
        
        Example 2 (one advanced expression):
        Input: "彼は腹黒い性格だ。"
        Output: {
          "sentence_number": 2,
          "english": "He has a scheming personality.",
          "english_literal": "He has a black-bellied personality.",
          "part_to_breakdown_1": "腹黒い",
          "breakdown_1": "The term (literally 'belly-black') describes someone who is deceitful or manipulative.",
          "part_to_breakdown_2": "",
          "breakdown_2": "",
          "part_to_breakdown_3": "",
          "breakdown_3": ""
        }
        
        Example 3 (two advanced expressions):
        Input: "明日までにレポートを出さなくてはならないが、気が重い。"
        Output: {
          "sentence_number": 3,
          "english": "By tomorrow, I have to submit the report, but I'm feeling reluctant.",
          "english_literal": "By tomorrow, the report I must submit, but my spirit is heavy.",
          "part_to_breakdown_1": "出さなくてはならない",
          "breakdown_1": "This is the '-nakute wa naranai' obligation pattern meaning 'must/have to.' It's generally more formal/written than the conversational '-nakute wa ikenai' variant.",
          "part_to_breakdown_2": "気が重い",
          "breakdown_2": "Literal 'spirit/heavy,' meaning to feel reluctant, burdened, or unmotivated about something.",
          "part_to_breakdown_3": "",
          "breakdown_3": ""
        }
        
        Example 4 (three advanced expressions):
        Input: "その提案は画竜点睛を欠くものだったが、焼け石に水でも試す価値はある。"
        Output: {
          "sentence_number": 4,
          "english": "The proposal was missing the finishing touch, but even if it's like pouring water on a hot stone, it's worth trying.",
          "english_literal": "That proposal was a thing lacking the dotting-the-dragon's-eyes, but even if it is water onto a hot stone, it's worth trying.",
          "part_to_breakdown_1": "画竜点睛",
          "breakdown_1": "Literally 'dotting the eyes of a painted dragon'. Ga means draw or paint. Ryuu means dragon. Ten means dot. Sei means the pupil of the eye. Meaning: the final crucial touch that makes the whole thing come alive.",
          "part_to_breakdown_2": "を欠く",
          "breakdown_2": "To lack/be missing. Often used with abstract nouns to indicate something is absent.",
          "part_to_breakdown_3": "焼け石に水",
          "breakdown_3": "Pouring water on a hot stone. Means a futile effort that has little to no effect."
        }
        
        Remember: Maintain consistency and context across all sentences in the batch."""

        
        user_prompt = f"""Translate these Japanese sentences to English.

        JAPANESE SENTENCES:
        {numbered_sentences}
        
        Provide faithful translations that work well together as a coherent passage.
        Return a JSON array with translations for each numbered sentence.
        Follow the exact output format with sentence_number, english, english_literal, and up to 3 breakdown pairs (part_to_breakdown_1-3, breakdown_1-3)."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        print(f"  Translating...", end="", flush=True)
        response = self.call_api(messages)
        
        if not response:
            print(f" ❌ Failed")
            # One retry
            print(f"  Retrying batch {batch_number}...")
            time.sleep(2)
            response = self.call_api(messages)
            if not response:
                print(f"  ❌ Batch {batch_number} failed after retry")
                return None
        
        # Parse the response
        try:
            # Try to extract JSON array
            json_match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)
            if json_match:
                translations = json.loads(json_match.group())
            else:
                translations = json.loads(response)
            
            if not isinstance(translations, list):
                print(f" ❌ Expected array, got {type(translations)}")
                return None
            
            # Process and validate translations
            results: List[Dict[str, str]] = []
            for orig_num, orig_text in sentences:
                # Find corresponding translation
                translation = None
                for t in translations:
                    if isinstance(t, dict) and t.get("sentence_number") == orig_num:
                        translation = t
                        break
                
                if translation and isinstance(translation, dict):
                    english = clean_english_text(translation.get("english", ""), is_breakdown=False)
                    english_literal = clean_english_text(translation.get("english_literal", ""), is_breakdown=False)
                    
                    # Get up to 3 breakdown pairs
                    breakdown_data = {}
                    for i in range(1, 4):
                        part_key = f"part_to_breakdown_{i}"
                        breakdown_key = f"breakdown_{i}"
                        
                        # part_to_breakdown should keep Japanese characters (it's the Japanese word/phrase)
                        part = translation.get(part_key, "")
                        
                        # breakdown should have no Japanese characters, only romanization
                        # Remove all double quotes from breakdowns
                        breakdown = clean_english_text(translation.get(breakdown_key, ""), is_breakdown=True)
                        
                        # If breakdown exists but part is empty, log warning and keep empty
                        if breakdown and not part:
                            print(f"    ⚠️ Sentence {orig_num}: Breakdown {i} has no part_to_breakdown")
                            # Keep part empty - TTS will skip empty strings
                        
                        breakdown_data[part_key] = part  # Keep Japanese characters here
                        breakdown_data[breakdown_key] = breakdown  # Cleaned of Japanese chars
                    
                    # English translation should always exist, but handle edge cases gracefully
                    if not english or not english.strip():
                        # If English is empty, use empty string - TTS will skip it
                        english = ""
                        print(f"    ⚠️ Sentence {orig_num}: Empty English translation")
                    
                    # english_literal can be empty - that's fine
                    
                    results.append({
                        "sentence_number": orig_num,
                        "japanese": orig_text,
                        "english": english,
                        "english_literal": english_literal,
                        **breakdown_data
                    })
                else:
                    # If no translation dict found, create minimal entry with empty fields
                    print(f"    ⚠️ Sentence {orig_num}: No translation dict found")
                    
                    results.append({
                        "sentence_number": orig_num,
                        "japanese": orig_text,
                        "english": "",  # Empty string, not placeholder
                        "english_literal": "",  # Empty string
                        "part_to_breakdown_1": "",
                        "breakdown_1": "",
                        "part_to_breakdown_2": "",
                        "breakdown_2": "",
                        "part_to_breakdown_3": "",
                        "breakdown_3": ""
                    })
            
            print(f" ✓ Success")
            return results
            
        except json.JSONDecodeError:
            print(f" ❌ JSON parse error")
            print(f"  Response preview: {response[:200]}...")
            return None
        except Exception as e:
            print(f" ❌ Error: {e}")
            return None


def save_progress(sentences: Dict[str, Dict[str, str]], filename: str = PROGRESS_FILE) -> bool:
    """Save progress to file."""
    if not sentences:
        return False
    
    progress_data = {
        "last_saved": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_sentences": len(sentences),
        "sentences": sentences
    }
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
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
                # Build the sentence dictionary with ALL fields including breakdowns
                sentence_data = {
                    "japanese": result["japanese"],
                    "english": result["english"],
                    "english_literal": result["english_literal"],
                    "original_batch_sentence_num": result.get("sentence_number", 0),
                    "batch_number": batch_num
                }
                
                # Add all breakdown fields (1-3)
                for i in range(1, 4):
                    sentence_data[f"part_to_breakdown_{i}"] = result.get(f"part_to_breakdown_{i}", "")
                    sentence_data[f"breakdown_{i}"] = result.get(f"breakdown_{i}", "")
                
                all_sentences[str(global_sentence_counter)] = sentence_data
                global_sentence_counter += 1
            successful_batches += 1
            
            # Save progress every 5 batches
            if batch_num % 5 == 0:
                if save_progress(all_sentences):
                    print(f"  💾 Progress saved ({len(all_sentences)} sentences)")
        else:
            failed_batches += 1
        
        # Record batch time for estimation
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        
        # Calculate and show estimated remaining time
        if batch_times and batch_num < len(batches):
            avg_time = sum(batch_times) / len(batch_times)
            remaining = avg_time * (len(batches) - batch_num)
            
            # Convert to readable format
            if remaining > 3600:
                eta = f"{remaining/3600:.1f} hours"
            elif remaining > 60:
                eta = f"{remaining/60:.1f} minutes"
            else:
                eta = f"{remaining:.0f} seconds"
            
            print(f"  ⏱️  Batch: {batch_time:.1f}s | Remaining: ~{eta}")
        
        # Delay between batches
        if batch_num < len(batches):
            time.sleep(BATCH_DELAY)
    
    elapsed = time.time() - start_time
    
    # Final progress save
    save_progress(all_sentences)
    
    print(f"\n" + "=" * 50)
    print(f"✅ Processing completed in {elapsed:.1f} seconds")
    print(f"📊 Batch statistics:")
    print(f"   Successful: {successful_batches}")
    print(f"   Failed: {failed_batches}")
    print(f"   Total sentences: {len(all_sentences)}")
    if batch_times:
        print(f"   Average batch time: {sum(batch_times)/len(batch_times):.1f}s")
    
    return all_sentences


def save_final_results(sentences: Dict[str, Dict[str, str]], filename: str = OUTPUT_JSON) -> bool:
    """Save final results to JSON file."""
    if not sentences:
        print("❌ No sentences to save")
        return False
    
    # Count breakdown statistics
    breakdown_counts = {1: 0, 2: 0, 3: 0}
    breakdown_types = {}
    
    for s in sentences.values():
        # Count how many breakdowns this sentence has
        breakdown_count = 0
        for i in range(1, 4):
            if s.get(f"breakdown_{i}") and s.get(f"breakdown_{i}").strip():
                breakdown_count += 1
                # Track unique parts
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
            "processing_method": "Batch-based translation for context",
            "translation_approach": "Faithful translation, neutral when context missing",
            "breakdown_structure": "Up to 3 breakdown pairs per sentence (part_to_breakdown_1-3, breakdown_1-3)",
            "character_rule": "No Japanese characters in English output, romanization only in breakdowns, ASCII-only output",
            "output_fields": "Includes both natural (english) and literal (english_literal) translations for learning"
        },
        "sentences": sentences
    }
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Final results saved to {filename}")
        return True
    except Exception as e:
        print(f"❌ Error saving final results: {e}")
        return False


def main() -> None:
    """Main function."""
    print("=" * 60)
    print("Japanese to English Translator")
    print("Batch Processing with Enhanced Breakdown Structure (up to 3 expressions)")
    print("Includes both natural and literal translations for learning")
    print("=" * 60)
    
    # Load configuration
    config = Config()
    if not config.load_config():
        sys.exit(1)
    
    # Initialize translator
    translator = BatchTranslator(config)
    
    # Process all batches
    sentences = process_all_batches(translator)
    
    if not sentences:
        print("❌ No sentences were processed.")
        sys.exit(1)
    
    # Save final results
    if not save_final_results(sentences):
        sys.exit(1)
    
    # Show summary
    total = len(sentences)
    
    # Calculate breakdown statistics
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
    
    # Show examples with new structure
    if sentences:
        print("\n📄 SAMPLE OUTPUT (with new breakdown structure):")
        
        # Find examples with different numbers of breakdowns
        examples = {0: [], 1: [], 2: [], 3: []}
        
        for key, s in sentences.items():
            breakdown_count = 0
            for i in range(1, 4):
                if s.get(f"breakdown_{i}") and s.get(f"breakdown_{i}").strip():
                    breakdown_count += 1
            
            if len(examples[breakdown_count]) < 2:
                examples[breakdown_count].append((key, s, breakdown_count))
            
            # Stop when we have at least one example of each type (except maybe 3)
            if all(len(examples[i]) >= 1 for i in [0, 1, 2]):
                if len(examples[3]) >= 1 or breakdown_count == 3:
                    break
        
        # Show examples
        for breakdown_count in [0, 1, 2, 3]:
            for key, s, count in examples[breakdown_count]:
                print(f"\n{key}. ({count} BREAKDOWN{'S' if count != 1 else ''})")
                print(f"   Japanese: {s['japanese'][:60]}...")
                print(f"   English: {s['english'][:60]}...")
                print(f"   Literal: {s['english_literal'][:60]}...")
                
                if count > 0:
                    for i in range(1, count + 1):
                        part = s.get(f"part_to_breakdown_{i}", "")
                        breakdown = s.get(f"breakdown_{i}", "")
                        print(f"   Part {i}: '{part}'")
                        if breakdown:
                            print(f"   Breakdown {i}: {breakdown[:60]}...")
                else:
                    print(f"   Part: '' (no breakdowns)")
                    print(f"   Breakdown: '' (no breakdowns)")
    
    # Note about progress file
    if Path(PROGRESS_FILE).exists():
        print(f"\n📁 Progress saved to: {PROGRESS_FILE}")


if __name__ == "__main__":
    main()