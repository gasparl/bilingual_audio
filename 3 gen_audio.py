"""
JP-EN Audio Generator (edge-tts version)

Reads translated_output.json and creates per-sentence MP3 files for
Japanese-first listening practice. Expected sentence fields are:
  japanese, english, english_literal, part_to_breakdown_1..3, breakdown_1..3

Audio order:
1) sentence number in Japanese
2) Japanese sentence
3) English sentence
4) optional breakdown audio
5) optional literal English audio
6) Japanese sentence again, female then male

Requirements:
  pip install edge-tts pydub
  ffmpeg must be installed for pydub MP3 combining/export.
"""

import asyncio
import json
import pathlib
import random
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

from pydub import AudioSegment

try:
    import edge_tts
except ImportError:
    edge_tts = None


# ---------- CONFIG ----------
INPUT_JSON = "translated_output.json"

# Audio processing settings
PAUSE_IN_BREAKDOWN_MS = 500          # pause between Japanese term and English breakdown in part 4
PAUSE_BETWEEN_BREAKDOWNS_MS = 800    # pause between different breakdown pairs
PAUSE_IN_ALTERNATING_MS = 800        # pause between female and male voices in part 6
PAUSE_END_SILENCE_MS = 900           # silence appended to end of final segment files

# edge-tts speech rates
# edge-tts uses strings like "+0%", "-10%", "-25%"
JA_RATE_SENTENCE_NUMBER = "+0%"
JA_RATE_JAPANESE_MALE = "-25%"
JA_RATE_ALTERNATING = "-10%"
EN_RATE = "+0%"

# Robustness settings
MAX_RETRIES = 5
RETRY_BASE_SLEEP = 0.7
RETRY_JITTER = 0.4
MIN_MP3_BYTES = 120

# Voice configurations
JAPANESE_VOICES = {
    "male": ["ja-JP-KeitaNeural"],
    "female": ["ja-JP-NanamiNeural"],
}
ENGLISH_VOICES = [
    "en-US-GuyNeural",
    "en-US-SteffanNeural",
    "en-US-AndrewNeural",
    "en-US-BrianNeural",
]

# ---------- PATHS ----------
BASE_DIR = pathlib.Path(__file__).resolve().parent
AUDIO_OUTPUT_DIR = BASE_DIR / "audio_sentences"
INPUT_JSON_PATH = BASE_DIR / INPUT_JSON


def clean_text(text: str) -> str:
    """Clean text for TTS."""
    if not text:
        return ""
    text = str(text)
    text = text.replace('"', ",")
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def file_has_audio(p: pathlib.Path) -> bool:
    return p.exists() and p.is_file() and p.stat().st_size > MIN_MP3_BYTES


def safe_unlink(file_path: pathlib.Path) -> None:
    """Safely delete a file with error handling."""
    try:
        file_path.unlink(missing_ok=True)
    except Exception:
        pass


def add_end_silence_to_mp3(input_file: pathlib.Path, output_file: Optional[pathlib.Path] = None) -> bool:
    """Add end silence to an MP3 file using pydub."""
    if output_file is None:
        output_file = input_file

    try:
        audio = AudioSegment.from_file(str(input_file), format="mp3")
        silence = AudioSegment.silent(duration=PAUSE_END_SILENCE_MS, frame_rate=audio.frame_rate)
        silence = silence.set_channels(audio.channels).set_sample_width(audio.sample_width)
        audio_with_silence = audio + silence

        temp_file = output_file.with_suffix(f".temp{output_file.suffix}")
        audio_with_silence.export(str(temp_file), format="mp3", bitrate="192k")

        safe_unlink(output_file)
        temp_file.rename(output_file)
        return file_has_audio(output_file)
    except Exception:
        return False


async def generate_mp3_async(text: str, output_file: pathlib.Path, voice_name: str, rate: str) -> bool:
    """Generate one MP3 file with edge-tts."""
    if edge_tts is None:
        return False

    text = clean_text(text)
    if not text:
        return False

    try:
        communicate = edge_tts.Communicate(text=text, voice=voice_name, rate=rate)
        await communicate.save(str(output_file))
        return file_has_audio(output_file)
    except Exception:
        return False


def generate_mp3_once(text: str, output_file: pathlib.Path, voice_name: str, rate: str) -> bool:
    """Synchronous wrapper around edge-tts generation."""
    try:
        return asyncio.run(generate_mp3_async(text, output_file, voice_name, rate))
    except RuntimeError:
        # Defensive fallback for unusual environments with an already-running loop.
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(generate_mp3_async(text, output_file, voice_name, rate))
        finally:
            loop.close()
            asyncio.set_event_loop(None)
    except Exception:
        return False


def generate_mp3_retry(
    text: str,
    output_file: pathlib.Path,
    voice_name: str,
    rate: str = "+0%",
    attempts: int = MAX_RETRIES,
) -> bool:
    """Retry wrapper with exponential backoff."""
    for i in range(1, attempts + 1):
        safe_unlink(output_file)
        ok = generate_mp3_once(text, output_file, voice_name, rate)
        if ok:
            return True

        if i < attempts:
            sleep_s = (RETRY_BASE_SLEEP * (2 ** (i - 1))) + random.uniform(0, RETRY_JITTER)
            time.sleep(sleep_s)

    return False


def test_voice(voice_name: str, test_text: str) -> bool:
    """Test whether an edge-tts voice works."""
    test_file = AUDIO_OUTPUT_DIR / f"_test_{voice_name}.mp3"
    ok = generate_mp3_retry(test_text, test_file, voice_name, "+0%", attempts=2)
    safe_unlink(test_file)
    return ok


def detect_voices() -> Dict[str, str]:
    """Detect usable edge-tts voices from configured candidates."""
    voices: Dict[str, str] = {}

    for gender in ["male", "female"]:
        test_text = "こんにちは。"
        for voice_name in JAPANESE_VOICES[gender]:
            if test_voice(voice_name, test_text):
                voices[f"ja_{gender}"] = voice_name
                print(f"✓ Japanese {gender}: {voice_name}")
                break
        else:
            print(f"✗ Japanese {gender}: Not found")

    for voice_name in ENGLISH_VOICES:
        if test_voice(voice_name, "Hello."):
            voices["en_male"] = voice_name
            print(f"✓ English: {voice_name}")
            break
    else:
        print("✗ English: Not found")

    return voices


class AudioCombiner:
    """Handles MP3 combination with pause insertion."""

    @staticmethod
    def combine_mp3_files(input_files: List[pathlib.Path], output_file: pathlib.Path, pause_ms: int = 0) -> bool:
        """Combine MP3 files with optional pauses."""
        if not input_files:
            return False

        try:
            combined = None

            for idx, input_file in enumerate(input_files):
                if not file_has_audio(input_file):
                    return False

                segment = AudioSegment.from_file(str(input_file), format="mp3")

                if combined is None:
                    combined = segment
                else:
                    if pause_ms > 0:
                        silence = AudioSegment.silent(duration=pause_ms, frame_rate=segment.frame_rate)
                        silence = silence.set_channels(segment.channels).set_sample_width(segment.sample_width)
                        combined += silence
                    combined += segment

            if combined is None:
                return False

            temp_file = output_file.with_suffix(f".temp{output_file.suffix}")
            combined.export(str(temp_file), format="mp3", bitrate="192k")
            safe_unlink(output_file)
            temp_file.rename(output_file)
            return file_has_audio(output_file)
        except Exception:
            return False


class SentenceProcessor:
    """Processes a single sentence with all its audio segments."""

    def __init__(self, sid: str, row: Dict, voices: Dict[str, str], audio_dir: pathlib.Path):
        self.sid = sid
        self.row = row
        self.voices = voices
        self.audio_dir = audio_dir

        self.jp_text = clean_text(row.get("japanese", ""))
        self.en_text = clean_text(row.get("english", ""))
        self.en_literal = clean_text(row.get("english_literal", ""))

        self.errors: List[str] = []
        self.skipped_parts: List[str] = []
        self.created_parts: List[str] = []

    def _extract_part_number(self, filename: str) -> str:
        """Extract part number from filename like '1_sentence_number' or '3_english_translation'."""
        parts = filename.split("_")
        if parts and parts[0].isdigit():
            return parts[0]
        return "?"

    def _get_rate(self, voice_key: str, segment_type: Optional[str]) -> str:
        """Get edge-tts speech rate for a segment."""
        if voice_key.startswith("en_"):
            return EN_RATE

        if segment_type == "sentence_number":
            return JA_RATE_SENTENCE_NUMBER
        if segment_type == "alternating":
            return JA_RATE_ALTERNATING
        return JA_RATE_JAPANESE_MALE

    def generate_segment(
        self,
        text: str,
        filename: str,
        voice_key: str,
        segment_type: Optional[str] = None,
        optional: bool = False,
        add_end_silence: bool = True,
    ) -> Optional[pathlib.Path]:
        """Generate a single MP3 segment."""
        part_num = self._extract_part_number(filename)

        if not text and optional:
            self.skipped_parts.append(part_num)
            return None

        if not text:
            if not optional:
                self.errors.append(part_num)
            return None

        if voice_key not in self.voices:
            if optional:
                self.skipped_parts.append(part_num)
            else:
                self.errors.append(part_num)
            return None

        output_file = self.audio_dir / f"sentence_{self.sid}_{filename}_temp.mp3"
        voice_name = self.voices[voice_key]
        rate = self._get_rate(voice_key, segment_type)

        if generate_mp3_retry(text, output_file, voice_name, rate):
            if add_end_silence and PAUSE_END_SILENCE_MS > 0:
                add_end_silence_to_mp3(output_file)
            return output_file

        if optional:
            self.skipped_parts.append(part_num)
        else:
            self.errors.append(part_num)
        return None

    def process_sentence_number(self) -> Optional[Tuple[pathlib.Path, pathlib.Path]]:
        """Process sentence number segment."""
        num_text = f"{self.sid}番目"
        gender = random.choice(["male", "female"])
        temp_file = self.generate_segment(
            num_text,
            "1_sentence_number",
            f"ja_{gender}",
            "sentence_number",
            add_end_silence=True,
        )
        if temp_file:
            final_file = self.audio_dir / f"sentence_{self.sid}_1_sentence_number.mp3"
            return (temp_file, final_file)
        return None

    def process_japanese_male(self) -> Optional[Tuple[pathlib.Path, pathlib.Path]]:
        """Process Japanese sentence segment."""
        if not self.jp_text:
            self.skipped_parts.append("2")
            return None

        temp_file = self.generate_segment(
            self.jp_text,
            "2_japanese_male",
            "ja_male",
            "japanese_male",
            add_end_silence=True,
        )
        if temp_file:
            final_file = self.audio_dir / f"sentence_{self.sid}_2_japanese_male.mp3"
            return (temp_file, final_file)
        return None

    def process_english_translation(self) -> Optional[Tuple[pathlib.Path, pathlib.Path]]:
        """Process English translation/source sentence segment."""
        if not self.en_text:
            self.errors.append("3")
            return None

        temp_file = self.generate_segment(
            self.en_text,
            "3_english_translation",
            "en_male",
            optional=False,
            add_end_silence=True,
        )
        if temp_file:
            final_file = self.audio_dir / f"sentence_{self.sid}_3_english_translation.mp3"
            return (temp_file, final_file)
        return None

    def process_breakdown(self) -> Optional[pathlib.Path]:
        """Process breakdown combined segment with pauses between pairs."""
        final_file = self.audio_dir / f"sentence_{self.sid}_4_breakdown_combined.mp3"
        breakdown_pairs = self._collect_breakdown_pairs()

        if not breakdown_pairs or "en_male" not in self.voices:
            self.skipped_parts.append("4")
            return None

        temp_files: List[pathlib.Path] = []

        try:
            pair_files: List[pathlib.Path] = []

            for pair_idx, (part_jp, breakdown_en) in enumerate(breakdown_pairs, 1):
                temp_jp = self.audio_dir / f"sentence_{self.sid}_4_{pair_idx}_jp_temp.mp3"
                temp_en = self.audio_dir / f"sentence_{self.sid}_4_{pair_idx}_en_temp.mp3"
                pair_file = self.audio_dir / f"sentence_{self.sid}_4_{pair_idx}_pair_temp.mp3"

                temp_files.extend([temp_jp, temp_en, pair_file])

                if not generate_mp3_retry(part_jp, temp_jp, self.voices["ja_male"], JA_RATE_JAPANESE_MALE):
                    self.errors.append("4")
                    return None

                if not generate_mp3_retry(breakdown_en, temp_en, self.voices["en_male"], EN_RATE):
                    self.errors.append("4")
                    return None

                if not AudioCombiner.combine_mp3_files([temp_jp, temp_en], pair_file, pause_ms=PAUSE_IN_BREAKDOWN_MS):
                    self.errors.append("4")
                    return None

                pair_files.append(pair_file)

            if not AudioCombiner.combine_mp3_files(pair_files, final_file, pause_ms=PAUSE_BETWEEN_BREAKDOWNS_MS):
                self.errors.append("4")
                return None

            if PAUSE_END_SILENCE_MS > 0:
                add_end_silence_to_mp3(final_file)

            if file_has_audio(final_file):
                self.created_parts.append("4")
                return final_file

            self.errors.append("4")
            return None

        except Exception:
            self.errors.append("4")
            return None
        finally:
            for temp_file in temp_files:
                safe_unlink(temp_file)

    def process_english_literal(self) -> Optional[Tuple[pathlib.Path, pathlib.Path]]:
        """Process literal English segment."""
        # Keep original behavior: only create literal audio if part 4 was created.
        if not self.en_literal or "4" not in self.created_parts:
            self.skipped_parts.append("5")
            return None

        temp_file = self.generate_segment(
            self.en_literal,
            "5_english_literal",
            "en_male",
            optional=True,
            add_end_silence=True,
        )
        if temp_file:
            final_file = self.audio_dir / f"sentence_{self.sid}_5_english_literal.mp3"
            return (temp_file, final_file)
        return None

    def process_alternating(self) -> Optional[pathlib.Path]:
        """Process Japanese alternating female -> male segment."""
        final_file = self.audio_dir / f"sentence_{self.sid}_6_japanese_alternating.mp3"

        if not self.jp_text:
            self.skipped_parts.append("6")
            return None

        temp6a = self.audio_dir / f"sentence_{self.sid}_6a_temp.mp3"
        temp6b = self.audio_dir / f"sentence_{self.sid}_6b_temp.mp3"

        try:
            if not generate_mp3_retry(self.jp_text, temp6a, self.voices["ja_female"], JA_RATE_ALTERNATING):
                self.errors.append("6")
                return None

            if not generate_mp3_retry(self.jp_text, temp6b, self.voices["ja_male"], JA_RATE_ALTERNATING):
                self.errors.append("6")
                return None

            if not AudioCombiner.combine_mp3_files([temp6a, temp6b], final_file, pause_ms=PAUSE_IN_ALTERNATING_MS):
                self.errors.append("6")
                return None

            if PAUSE_END_SILENCE_MS > 0:
                add_end_silence_to_mp3(final_file)

            return final_file if file_has_audio(final_file) else None

        finally:
            safe_unlink(temp6a)
            safe_unlink(temp6b)

    def _collect_breakdown_pairs(self) -> List[Tuple[str, str]]:
        """Collect all breakdown pairs from the row data."""
        pairs: List[Tuple[str, str]] = []

        old_part = clean_text(self.row.get("part_to_breakdown", ""))
        old_breakdown = clean_text(self.row.get("breakdown", ""))
        if old_part and old_breakdown:
            pairs.append((old_part, old_breakdown))

        for i in range(1, 4):
            part_key = f"part_to_breakdown_{i}"
            breakdown_key = f"breakdown_{i}"
            part_jp = clean_text(self.row.get(part_key, ""))
            breakdown_en = clean_text(self.row.get(breakdown_key, ""))
            if part_jp and breakdown_en:
                pairs.append((part_jp, breakdown_en))

        return pairs


def count_sentence_files(sentence_id: str, audio_dir: pathlib.Path) -> int:
    """Count MP3 files for a specific sentence."""
    return len(list(audio_dir.glob(f"sentence_{sentence_id}_*.mp3")))


def sentence_has_required_files(sentence_id: str, audio_dir: pathlib.Path) -> bool:
    """Check if all required MP3 files exist for a sentence."""
    required_patterns = [
        f"sentence_{sentence_id}_1_sentence_number.mp3",
        f"sentence_{sentence_id}_2_japanese_male.mp3",
        f"sentence_{sentence_id}_3_english_translation.mp3",
        f"sentence_{sentence_id}_6_japanese_alternating.mp3",
    ]
    return all((audio_dir / pattern).exists() for pattern in required_patterns)


def sort_sentence_ids(sentences: Dict) -> List[str]:
    """Sort numeric sentence ids numerically, non-numeric ids last."""
    return sorted(sentences.keys(), key=lambda x: int(x) if str(x).isdigit() else float("inf"))


def main() -> None:
    print("=" * 60)
    print("JP-EN Audio Generator (edge-tts)")
    print("=" * 60)

    if edge_tts is None:
        print("ERROR: edge-tts is not installed.")
        print("Install it with: pip install edge-tts")
        sys.exit(1)

    AUDIO_OUTPUT_DIR.mkdir(exist_ok=True)

    print("Detecting voices...")
    voices = detect_voices()

    if not voices.get("ja_male") or not voices.get("ja_female"):
        print("ERROR: Need both Japanese voices")
        sys.exit(1)

    if not voices.get("en_male"):
        print("ERROR: Need an English voice")
        sys.exit(1)

    if not INPUT_JSON_PATH.exists():
        print(f"ERROR: {INPUT_JSON} not found")
        sys.exit(1)

    with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
        root = json.load(f)

    sentences = root.get("sentences", {})
    if not sentences:
        print("ERROR: No sentences found")
        sys.exit(1)

    print(f"Loaded {len(sentences)} sentences")
    sentence_ids = sort_sentence_ids(sentences)

    print("\nUsing settings:")
    print("  Voices:")
    print(f"    Japanese male: {voices.get('ja_male')}")
    print(f"    Japanese female: {voices.get('ja_female')}")
    print(f"    English: {voices.get('en_male')}")
    print("  Japanese speech rates:")
    print(f"    Sentence number: {JA_RATE_SENTENCE_NUMBER}")
    print(f"    Japanese male: {JA_RATE_JAPANESE_MALE}")
    print(f"    Alternating: {JA_RATE_ALTERNATING}")
    print("  Pauses:")
    print(f"    Breakdown (Part 4): {PAUSE_IN_BREAKDOWN_MS}ms between components")
    print(f"    Between breakdown pairs: {PAUSE_BETWEEN_BREAKDOWNS_MS}ms")
    print(f"    Alternating (Part 6): {PAUSE_IN_ALTERNATING_MS}ms between voices")
    print(f"    End silence on ALL parts (1-6): {PAUSE_END_SILENCE_MS}ms")
    print("\nStarting generation...")

    start_time = time.time()
    success_count = 0

    for idx, sid in enumerate(sentence_ids, 1):
        row = sentences[sid]

        if idx > 1:
            avg = (time.time() - start_time) / idx
            remaining = avg * (len(sentence_ids) - idx)
            if remaining >= 3600:
                eta = f"{remaining / 3600:.1f}h"
            elif remaining >= 60:
                eta = f"{remaining / 60:.1f}m"
            else:
                eta = f"{remaining:.0f}s"
        else:
            eta = "calc..."

        print(f"[{idx:3d}/{len(sentence_ids)}] Sentence {sid} - ETA: {eta}", end=" ")

        processor = SentenceProcessor(sid, row, voices, AUDIO_OUTPUT_DIR)
        segments_to_rename: List[Tuple[pathlib.Path, pathlib.Path]] = []

        # Parts 1-3
        for processor_func in [
            processor.process_sentence_number,
            processor.process_japanese_male,
            processor.process_english_translation,
        ]:
            result = processor_func()
            if result:
                segments_to_rename.append(result)

        # Part 4 must run before part 5 because it updates created_parts.
        processor.process_breakdown()

        # Part 5
        result = processor.process_english_literal()
        if result:
            segments_to_rename.append(result)

        # Part 6
        processor.process_alternating()

        # Rename temp MP3 files to final names.
        for temp_path, final_path in segments_to_rename:
            try:
                if temp_path.exists():
                    if final_path.exists():
                        safe_unlink(final_path)
                    temp_path.rename(final_path)
            except Exception:
                pass

        # Clean up remaining temp files.
        for temp_file in AUDIO_OUTPUT_DIR.glob(f"sentence_{sid}_*temp*.mp3"):
            safe_unlink(temp_file)

        count_files = count_sentence_files(sid, AUDIO_OUTPUT_DIR)

        errors_sorted = sorted(set(processor.errors))
        skips_sorted = sorted(set(processor.skipped_parts))

        status_parts = []
        if errors_sorted:
            status_parts.append(f"ERROR: {','.join(errors_sorted)}")
        if skips_sorted:
            status_parts.append(f"[o: {','.join(skips_sorted)}]")

        print(f"✓{count_files} " + " ".join(status_parts) if status_parts else f"✓{count_files}")

        if sentence_has_required_files(sid, AUDIO_OUTPUT_DIR):
            success_count += 1

        time.sleep(0.1)

    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Time: {total_time / 60:.1f} minutes")
    print(f"Sentences: {len(sentences)}")
    print(f"Success rate: {success_count / len(sentences) * 100:.1f}%")
    print(f"Avg per sentence: {total_time / len(sentences):.1f}s")
    print(f"\nOutput directory: {AUDIO_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
