#!/usr/bin/env python3
"""
DeepSeek Japanese-to-English Translator
Batch-based processing with progress saving
Enhanced breakdown structure
"""

import json
import os
import re
import sys
import time
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
    Returns list of (index, sentence) tuples.
    """
    if not text or not text.strip():
        return []
    
    sentences: List[str] = []
    current = ""
    
    for char in text:
        current += char
        if char in '。！？' and len(current) > 1:
            sentences.append(current.strip())
            current = ""
    
    if current.strip():
        sentences.append(current.strip())
    
    # Number each sentence
    return [(i + 1, s) for i, s in enumerate(sentences) if s]


def clean_japanese_characters(text: str) -> str:
    """Remove Japanese characters from text (for English fields only)."""
    if not text:
        return ""
    return re.sub(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3000-\u303F\uFF00-\uFFEF]', '', text).strip()


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
        
        # Optimized prompt with new breakdown structure
        system_prompt = """You are an expert Japanese-to-English translator.
        
        TASK:
        Translate each Japanese sentence to English, maintaining context across sentences.
        For each sentence, provide:
        1. A faithful English translation
        2. If the sentence contains an advanced expression, grammar point, or obscure cultural reference worth explaining:
           - Identify the specific Japanese word/phrase
           - Provide a concise explanation (max 3-4 sentences)
        
        REQUIREMENTS:
        * Provide translations as close and faithful as possible to the original.
        * If context is missing (e.g., who is the subject), keep it neutral rather than guessing.
        * Only provide breakdown for ONE grammar point or expression per sentence.
        * For simple everyday sentences, leave part_to_breakdown and breakdown empty.
        * English output must NOT contain any Japanese characters.
        * In breakdown, use romanization only (e.g., "hara-guroi" not "腹黒い").
        
        OUTPUT FORMAT:
        Return a JSON array where each object has:
        {
          "sentence_number": (number from input),
          "english": "English translation",
          "part_to_breakdown": "Japanese word/phrase being explained OR empty string",
          "breakdown": "Brief explanation OR empty string"
        }
        
        EXAMPLES:
        Example 1 (simple - no breakdown):
        Input: "今日はいい天気ですね。"
        Output: {
          "sentence_number": 1,
          "english": "The weather is nice today, isn't it?",
          "part_to_breakdown": "",
          "breakdown": ""
        }
        
        Example 2 (advanced expression):
        Input: "彼は腹黒い性格だ。"
        Output: {
          "sentence_number": 2,
          "english": "He has a scheming personality.",
          "part_to_breakdown": "腹黒い",
          "breakdown": "The term (literally 'belly-black') describes someone who is deceitful or manipulative."
        }
        
        Example 3 (advanced grammar point):
        Input: "明日までにレポートを出さなくてはならない。"
        Output: {
          "sentence_number": 3,
          "english": "I have to submit the report by tomorrow.",
          "part_to_breakdown": "出さなくてはならない",
          "breakdown": "This is the '-nakute wa naranai' obligation pattern meaning 'must/have to.' It's generally more formal/written than the conversational '-nakute wa ikenai' variant."
        }
        
        Remember: Maintain consistency and context across all sentences in the batch."""
        
        user_prompt = f"""Translate these Japanese sentences to English.

JAPANESE SENTENCES:
{numbered_sentences}

Provide faithful translations that work well together as a coherent passage.
Return a JSON array with translations for each numbered sentence.
Follow the exact output format with sentence_number, english, part_to_breakdown, and breakdown fields."""

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
                    english = clean_japanese_characters(translation.get("english", ""))
                    part_to_breakdown = translation.get("part_to_breakdown", "")
                    breakdown = clean_japanese_characters(translation.get("breakdown", ""))
                    
                    # Validate English is not empty
                    if not english or len(english) < 3:
                        english = f"[Translation: {orig_text[:80]}...]"
                    
                    # If breakdown exists, part_to_breakdown should not be empty
                    if breakdown and not part_to_breakdown:
                        part_to_breakdown = "[unspecified]"
                    
                    results.append({
                        "sentence_number": orig_num,
                        "japanese": orig_text,
                        "english": english,
                        "part_to_breakdown": part_to_breakdown,
                        "breakdown": breakdown
                    })
                else:
                    # If no translation found, create placeholder
                    results.append({
                        "sentence_number": orig_num,
                        "japanese": orig_text,
                        "english": f"[Translation not provided: {orig_text[:80]}...]",
                        "part_to_breakdown": "",
                        "breakdown": ""
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
                all_sentences[str(global_sentence_counter)] = {
                    "japanese": result["japanese"],
                    "english": result["english"],
                    "part_to_breakdown": result["part_to_breakdown"],
                    "breakdown": result["breakdown"],
                    "original_batch_sentence_num": result.get("sentence_number", 0),
                    "batch_number": batch_num
                }
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
    with_breakdown = sum(1 for s in sentences.values() if s.get("breakdown"))
    breakdown_types = {}
    for s in sentences.values():
        if s.get("part_to_breakdown") and s.get("breakdown"):
            part = s["part_to_breakdown"]
            breakdown_types[part] = breakdown_types.get(part, 0) + 1
    
    output = {
        "metadata": {
            "total_sentences": len(sentences),
            "sentences_with_breakdown": with_breakdown,
            "breakdown_types_count": len(breakdown_types),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "processing_method": "Batch-based translation for context",
            "translation_approach": "Faithful translation, neutral when context missing",
            "breakdown_structure": "part_to_breakdown contains Japanese word/phrase, breakdown contains explanation",
            "character_rule": "No Japanese characters in English output, romanization only in breakdowns"
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
    print("Batch Processing with Enhanced Breakdown Structure")
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
    with_breakdown = sum(1 for s in sentences.values() if s.get("breakdown"))
    
    print("\n" + "=" * 60)
    print("📊 FINAL SUMMARY")
    print("=" * 60)
    print(f"Total sentences: {total}")
    print(f"Sentences with breakdown: {with_breakdown}")
    print(f"Sentences without breakdown: {total - with_breakdown}")
    
    if total > 0:
        breakdown_rate = (with_breakdown / total) * 100
        print(f"Breakdown rate: {breakdown_rate:.1f}%")
    
    # Show examples with new structure
    if sentences:
        print("\n📄 SAMPLE OUTPUT (with breakdown structure):")
        # Find some examples with and without breakdowns
        examples_with_breakdown = []
        examples_without_breakdown = []
        
        for key, s in sentences.items():
            if s.get("breakdown") and len(examples_with_breakdown) < 2:
                examples_with_breakdown.append((key, s))
            elif not s.get("breakdown") and len(examples_without_breakdown) < 1:
                examples_without_breakdown.append((key, s))
            
            if len(examples_with_breakdown) >= 2 and len(examples_without_breakdown) >= 1:
                break
        
        # Show examples
        for key, s in examples_with_breakdown:
            print(f"\n{key}. (WITH BREAKDOWN)")
            print(f"   Japanese: {s['japanese'][:60]}...")
            print(f"   English: {s['english'][:60]}...")
            print(f"   Part: '{s['part_to_breakdown']}'")
            print(f"   Breakdown: {s['breakdown'][:60]}...")
        
        for key, s in examples_without_breakdown:
            print(f"\n{key}. (NO BREAKDOWN)")
            print(f"   Japanese: {s['japanese'][:60]}...")
            print(f"   English: {s['english'][:60]}...")
            print(f"   Part: '' (empty)")
            print(f"   Breakdown: '' (empty)")
    
    # Note about progress file
    if Path(PROGRESS_FILE).exists():
        print(f"\n📁 Progress saved to: {PROGRESS_FILE}")


if __name__ == "__main__":
    main()