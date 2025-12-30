#!/usr/bin/env python3
"""
DeepSeek Japanese-to-English Translator
Batch-based processing with progress saving
Enhanced breakdown structure with up to 3 expressions
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
MAX_RETRIES = 2  # One initial attempt + one retry
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


def split_japanese_sentences(text: str) -> List[Tuple[int, str]]:
    """
    Split Japanese text into sentences with indices.
    Improved to handle Japanese quote marks properly.
    Returns list of (index, sentence) tuples.
    """
    if not text or not text.strip():
        return []
    
    sentences: List[str] = []
    current = ""
    quote_depth = 0  # Track nested quotes
    
    for char in text:
        current += char
        
        # Track quote nesting
        if char == '「' or char == '『':
            quote_depth += 1
        elif char == '」' or char == '』':
            quote_depth -= 1
        
        # Only split at sentence boundaries when not inside quotes
        if char in '。！？' and len(current) > 1 and quote_depth == 0:
            sentences.append(current.strip())
            current = ""
    
    if current.strip():
        sentences.append(current.strip())
    
    # Number each sentence
    return [(i + 1, s) for i, s in enumerate(sentences) if s]


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
        '・': '·', '〜': '~',   # Other Japanese chars
        'ー': '-', '…': '...',  # More replacements
        '―': '-', '‥': '..',
        '「': '"', '」': '"',   # Curly quotes to straight quotes
        '「': "'", '」': "'",   # Curly apostrophes
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
            "max_tokens": 4000,
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
        
        # Optimized prompt with up to 3 breakdown expressions
        system_prompt = """You are an expert Japanese-to-English translator.
        
        TASK:
        Translate each Japanese sentence to English, maintaining context across sentences.
        For each sentence, provide:
        1. A faithful English translation
        2. Up to THREE (0-3) grammar points, advanced expressions, or obscure cultural references worth explaining:
           - Identify the specific Japanese word/phrase for each
           - Provide a concise explanation for each (max 3-4 sentences per breakdown)
        
        REQUIREMENTS:
        * Provide translations as close and faithful as possible to the original.
        * If context is missing (e.g., who is the subject), keep it neutral rather than guessing.
        * Provide breakdowns for only the most important/significant expressions (max 3 per sentence).
        * For very simple everyday sentences, leave all breakdown fields empty.
        * English output must NOT contain any Japanese characters.
        * In breakdowns, use romanization only (e.g., "hara-guroi" not "腹黒い").
        * Use only ASCII characters in English output (no special Unicode quotes or punctuation).
        
        OUTPUT FORMAT:
        Return a JSON array where each object has:
        {
          "sentence_number": (number from input),
          "english": "English translation",
          "part_to_breakdown_1": "Japanese word/phrase being explained OR empty string",
          "breakdown_1": "Brief explanation OR empty string",
          "part_to_breakdown_2": "Japanese word/phrase being explained OR empty string",
          "breakdown_2": "Brief explanation OR empty string",
          "part_to_breakdown_3": "Japanese word/phrase being explained OR empty string",
          "breakdown_3": "Brief explanation OR empty string"
        }
        
        IMPORTANT:
        * Fill breakdowns sequentially (1, then 2, then 3). Don't skip numbers.
        * If a sentence has only 2 breakdowns, leave fields 3 empty.
        * If a sentence has only 1 breakdown, leave fields 2 and 3 empty.
        * In English translation field: Use only ASCII characters (A-Z, a-z, 0-9, and common punctuation).
        * In part_to_breakdown fields: Keep the actual Japanese characters (for reference).
        * In breakdown fields: Use only ASCII characters with romanization. Do NOT use double quotes in breakdowns.
        
        EXAMPLES:
        Example 1 (simple - no breakdowns):
        Input: "今日はいい天気ですね。"
        Output: {
          "sentence_number": 1,
          "english": "The weather is nice today, isn't it?",
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
          "english": "I have to submit the report by tomorrow, but I'm feeling reluctant.",
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
          "english": "The proposal was missing the finishing touch, but it's worth trying even if it's like pouring water on a hot stone.",
          "part_to_breakdown_1": "画竜点睛",
          "breakdown_1": "Literally 'dotting the eyes of a painted dragon'. Ga means draw or paint. Ryuu means dragon. Ten means dot. Sei means the pupil of the eye. Meaning: the final crucial touch that makes the whole thing come alive.",
          "part_to_breakdown_2": "を欠く",
          "breakdown_2": "'wo kaku' - to lack/be missing. Often used with abstract nouns to indicate something is absent.",
          "part_to_breakdown_3": "焼け石に水",
          "breakdown_3": "'Yakeishi ni mizu' - pouring water on a hot stone. Means a futile effort that has little to no effect."
        }
        
        Remember: Maintain consistency and context across all sentences in the batch."""
        
        user_prompt = f"""Translate these Japanese sentences to English.

JAPANESE SENTENCES:
{numbered_sentences}

Provide faithful translations that work well together as a coherent passage.
Return a JSON array with translations for each numbered sentence.
Follow the exact output format with sentence_number, english, and up to 3 breakdown pairs (part_to_breakdown_1-3, breakdown_1-3)."""

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
                        
                        # If breakdown exists but part is empty, mark as unspecified
                        if breakdown and not part:
                            part = "[unspecified]"
                        
                        breakdown_data[part_key] = part  # Keep Japanese characters here
                        breakdown_data[breakdown_key] = breakdown  # Cleaned of Japanese chars
                    
                    # Validate English is not empty
                    if not english or len(english) < 3:
                        english = f"[Translation: {orig_text[:80]}...]"
                    
                    results.append({
                        "sentence_number": orig_num,
                        "japanese": orig_text,
                        "english": english,
                        **breakdown_data
                    })
                else:
                    # If no translation found, create placeholder with empty breakdown fields
                    placeholder = {
                        "sentence_number": orig_num,
                        "japanese": orig_text,
                        "english": f"[Translation not provided: {orig_text[:80]}...]",
                    }
                    # Add empty breakdown fields
                    for i in range(1, 4):
                        placeholder[f"part_to_breakdown_{i}"] = ""
                        placeholder[f"breakdown_{i}"] = ""
                    
                    results.append(placeholder)
            
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
            "character_rule": "No Japanese characters in English output, romanization only in breakdowns, ASCII-only output"
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