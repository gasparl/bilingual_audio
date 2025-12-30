import json
import subprocess
import pathlib
import re
import time
import sys
import random
import datetime
import wave
from pydub import AudioSegment
from typing import Dict, List, Optional, Tuple

# ---------- CONFIG ----------
BAL4WEB = r"C:\Program Files (x86)\Balabolka\bal4web\bal4web.exe"
INPUT_JSON = "translated_output.json"
LANG_CODE_JA = "ja-JP"
LANG_CODE_EN = "en-US"

# Retry behavior (minimal, robust)
MAX_RETRIES = 5
RETRY_BASE_SLEEP = 0.7      # seconds
RETRY_JITTER = 0.4          # seconds
MIN_WAV_BYTES = 120         # quick sanity threshold

# Voice configurations
JAPANESE_VOICES = {
    "male": {"candidates": ["Keita", "ja-JP-KeitaNeural"]},
    "female": {"candidates": ["Nanami", "ja-JP-NanamiNeural"]}
}

ENGLISH_VOICES = ["Steffan", "en-US-GuyNeural", "Andrew2", "Brian"]

# ---------- PATHS ----------
BASE_DIR = pathlib.Path(__file__).resolve().parent
AUDIO_OUTPUT_DIR = BASE_DIR / "audio_sentences"
INPUT_JSON_PATH = BASE_DIR / INPUT_JSON

# ---------- TEXT CLEANING ----------
def clean_text(text: str) -> str:
    """Clean text for TTS - essential cleaning only."""
    if not text:
        return ""
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def file_has_audio(p: pathlib.Path) -> bool:
    return p.exists() and p.is_file() and p.stat().st_size > MIN_WAV_BYTES

# ---------- WAV TO MP3 CONVERSION ----------
def convert_sentence_wavs_to_mp3(sentence_id: str, audio_dir: pathlib.Path):
    """Convert all WAV files for a specific sentence to MP3."""
    
    # Find all WAV files for this sentence
    wav_files = list(audio_dir.glob(f"sentence_{sentence_id}_*.wav"))
    
    for wav_file in wav_files:
        try:
            mp3_file = wav_file.with_suffix('.mp3')
            audio = AudioSegment.from_wav(str(wav_file))
            audio.export(str(mp3_file), format="mp3", bitrate="192k")
            wav_file.unlink(missing_ok=True)
        except Exception:
            pass  # Silently fail - keep WAV if conversion fails

# ---------- TTS FUNCTIONS ----------
def test_voice(voice_name: str, lang_code: str) -> Optional[Dict]:
    """Test if a voice works with simple text."""
    test_text = "こんにちは" if lang_code == LANG_CODE_JA else "Hello"
    test_file = AUDIO_OUTPUT_DIR / "_test.wav"

    args = [
        BAL4WEB, "-s", "microsoft", "-l", lang_code,
        "-n", voice_name, "-t", test_text, "-w", str(test_file)
    ]

    try:
        subprocess.run(
            args, check=True, capture_output=True,
            text=True, timeout=10, encoding="utf-8"
        )
        if file_has_audio(test_file):
            test_file.unlink(missing_ok=True)
            return {"name": voice_name, "lang": lang_code}
    except Exception:
        pass

    test_file.unlink(missing_ok=True)
    return None

def generate_audio_once(text: str, output_file: pathlib.Path, voice_params: Dict) -> bool:
    """Single attempt to generate audio."""
    if not text or not text.strip():
        return False

    text = clean_text(text)
    if not text:
        return False

    voice_name = voice_params["name"]
    lang_code = voice_params["lang"]

    # Attempt 1: with language code (most deterministic)
    args_lang = [
        BAL4WEB, "-s", "microsoft", "-l", lang_code,
        "-n", voice_name, "-t", text, "-w", str(output_file)
    ]

    try:
        subprocess.run(
            args_lang, check=True, capture_output=True,
            text=True, timeout=30, encoding="utf-8"
        )
        if file_has_audio(output_file):
            return True
    except subprocess.CalledProcessError:
        # Attempt 2: without language code (fallback)
        args_nolang = [
            BAL4WEB, "-s", "microsoft",
            "-n", voice_name, "-t", text, "-w", str(output_file)
        ]
        try:
            subprocess.run(
                args_nolang, check=True, capture_output=True,
                text=True, timeout=30, encoding="utf-8"
            )
            return file_has_audio(output_file)
        except Exception:
            return False
    except Exception:
        return False

    return False  # Explicit return if we reach here

def generate_audio_retry(text: str, output_file: pathlib.Path, voice_params: Dict,
                         attempts: int = MAX_RETRIES) -> bool:
    """Retry wrapper for transient failures with exponential backoff + jitter."""
    for i in range(1, attempts + 1):
        output_file.unlink(missing_ok=True)  # avoid accepting partial output
        ok = generate_audio_once(text, output_file, voice_params)
        if ok:
            return True

        if i < attempts:
            sleep_s = (RETRY_BASE_SLEEP * (2 ** (i - 1))) + random.uniform(0, RETRY_JITTER)
            time.sleep(sleep_s)

    return False

# ---------- MAIN PROCESSING ----------
def main():
    print("=" * 60)
    print("Japanese Audio Generator")
    print("=" * 60)

    if not pathlib.Path(BAL4WEB).exists():
        print(f"ERROR: bal4web.exe not found at {BAL4WEB}")
        sys.exit(1)

    AUDIO_OUTPUT_DIR.mkdir(exist_ok=True)

    # Detect voices
    print("Detecting voices...")
    voices: Dict[str, Dict] = {}

    for gender in ["male", "female"]:
        for voice_name in JAPANESE_VOICES[gender]["candidates"]:
            voice = test_voice(voice_name, LANG_CODE_JA)
            if voice:
                voices[f"ja_{gender}"] = voice
                print(f"✓ Japanese {gender}: {voice_name}")
                break
        else:
            print(f"✗ Japanese {gender}: Not found")

    for voice_name in ENGLISH_VOICES:
        voice = test_voice(voice_name, LANG_CODE_EN)
        if voice:
            voices["en_male"] = voice
            print(f"✓ English: {voice_name}")
            break
    else:
        print("✗ English: Not found")

    if not voices.get("ja_male") or not voices.get("ja_female"):
        print("ERROR: Need both Japanese voices")
        sys.exit(1)

    # Load sentences
    print("\nLoading JSON...")
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

    sentence_ids = sorted(
        sentences.keys(),
        key=lambda x: int(x) if x.isdigit() else float("inf")
    )

    print("\nStarting generation...")
    start_time = time.time()
    success_count = 0

    for idx, sid in enumerate(sentence_ids, 1):
        row = sentences[sid]

        elapsed = time.time() - start_time
        if idx > 1:
            avg = elapsed / idx
            remaining = avg * (len(sentence_ids) - idx)
            eta = f"{remaining/60:.1f}m" if remaining > 60 else f"{remaining:.0f}s"
        else:
            eta = "calc..."

        print(f"[{idx:3d}/{len(sentence_ids)}] Sentence {sid} - ETA: {eta}", end=" ")

        files_to_rename: List[Tuple[pathlib.Path, pathlib.Path]] = []
        errors: List[str] = []
        skipped_parts: List[str] = []
        breakdown_combined_created = False
        alternating_created = False

        # 1. Sentence number
        num_text = f"{sid}番目"
        gender = random.choice(["male", "female"])
        voice = voices[f"ja_{gender}"]
        temp1 = AUDIO_OUTPUT_DIR / f"sentence_{sid}_1_temp.wav"
        final1 = AUDIO_OUTPUT_DIR / f"sentence_{sid}_1_sentence_number.wav"

        if generate_audio_retry(num_text, temp1, voice):
            files_to_rename.append((temp1, final1))
        else:
            errors.append("1")

        # 2. Japanese sentence (male)
        jp_text = clean_text(row.get("japanese", ""))
        temp2 = AUDIO_OUTPUT_DIR / f"sentence_{sid}_2_temp.wav"
        final2 = AUDIO_OUTPUT_DIR / f"sentence_{sid}_2_japanese_male.wav"

        if jp_text:
            if generate_audio_retry(jp_text, temp2, voices["ja_male"]):
                files_to_rename.append((temp2, final2))
            else:
                errors.append("2")
        else:
            skipped_parts.append("2")

        # 3. English translation
        en_text = clean_text(row.get("english", ""))
        temp3 = AUDIO_OUTPUT_DIR / f"sentence_{sid}_3_temp.wav"
        final3 = AUDIO_OUTPUT_DIR / f"sentence_{sid}_3_english_translation.wav"

        if en_text and "en_male" in voices:
            if generate_audio_retry(en_text, temp3, voices["en_male"]):
                files_to_rename.append((temp3, final3))
            else:
                errors.append("3")
        else:
            if not en_text:
                skipped_parts.append("3")

        # 4. Breakdown parts (combined)
        part_jp = clean_text(row.get("part_to_breakdown", ""))
        breakdown_en = clean_text(row.get("breakdown", ""))
        final4 = AUDIO_OUTPUT_DIR / f"sentence_{sid}_4_breakdown_combined.wav"

        if part_jp and breakdown_en and "en_male" in voices:
            temp4a = AUDIO_OUTPUT_DIR / f"sentence_{sid}_4a_temp.wav"
            temp4b = AUDIO_OUTPUT_DIR / f"sentence_{sid}_4b_temp.wav"

            part_ok = generate_audio_retry(part_jp, temp4a, voices["ja_male"])
            breakdown_ok = generate_audio_retry(breakdown_en, temp4b, voices["en_male"])

            if part_ok and breakdown_ok:
                try:
                    with wave.open(str(temp4a), "rb") as w1:
                        f1 = w1.readframes(w1.getnframes())
                        params = w1.getparams()

                    with wave.open(str(temp4b), "rb") as w2:
                        f2 = w2.readframes(w2.getnframes())

                    with wave.open(str(final4), "wb") as out:
                        out.setparams(params)
                        out.writeframes(f1 + f2)

                    temp4a.unlink(missing_ok=True)
                    temp4b.unlink(missing_ok=True)
                    breakdown_combined_created = True
                except Exception:
                    errors.append("4c")
            else:
                if not part_ok:
                    errors.append("4a")
                if not breakdown_ok:
                    errors.append("4b")
        else:
            if not part_jp or not breakdown_en:
                skipped_parts.append("4")

        # 5. Japanese alternating (female + male repeat)
        final5 = AUDIO_OUTPUT_DIR / f"sentence_{sid}_5_japanese_alternating.wav"

        if jp_text:
            temp5a = AUDIO_OUTPUT_DIR / f"sentence_{sid}_5a_temp.wav"

            if generate_audio_retry(jp_text, temp5a, voices["ja_female"]):
                male_audio: Optional[pathlib.Path] = None

                # If part 2 temp exists (before rename), use it; else generate dedicated male temp
                if file_has_audio(temp2):
                    male_audio = temp2
                else:
                    temp5b = AUDIO_OUTPUT_DIR / f"sentence_{sid}_5b_temp.wav"
                    if generate_audio_retry(jp_text, temp5b, voices["ja_male"]):
                        male_audio = temp5b
                    else:
                        errors.append("5b")
                        male_audio = None

                try:
                    if male_audio:
                        with wave.open(str(temp5a), "rb") as w1:
                            f1 = w1.readframes(w1.getnframes())
                            params = w1.getparams()

                        with wave.open(str(male_audio), "rb") as w2:
                            f2 = w2.readframes(w2.getnframes())

                        with wave.open(str(final5), "wb") as out:
                            out.setparams(params)
                            out.writeframes(f1 + f2)

                        # Only delete male temp if it is the dedicated 5b temp
                        if male_audio.name.endswith("5b_temp.wav"):
                            male_audio.unlink(missing_ok=True)
                    else:
                        temp5a.rename(final5)

                    temp5a.unlink(missing_ok=True)
                    alternating_created = True
                except Exception:
                    errors.append("5c")
            else:
                errors.append("5a")
        else:
            skipped_parts.append("5")

        # Rename temp files to final names (overwrite-safe)
        for temp_path, final_path in files_to_rename:
            try:
                if temp_path.exists():
                    if final_path.exists():
                        final_path.unlink(missing_ok=True)
                    temp_path.rename(final_path)
            except Exception:
                pass

        # Clean up any remaining temp files
        for temp_file in AUDIO_OUTPUT_DIR.glob(f"*{sid}*temp*.wav"):
            temp_file.unlink(missing_ok=True)

        # Convert WAV files to MP3 for this sentence
        convert_sentence_wavs_to_mp3(sid, AUDIO_OUTPUT_DIR)

        # Count created files for this sentence (now counting MP3 files)
        count_files = 0
        if (AUDIO_OUTPUT_DIR / f"sentence_{sid}_1_sentence_number.mp3").exists():
            count_files += 1
        if (AUDIO_OUTPUT_DIR / f"sentence_{sid}_2_japanese_male.mp3").exists():
            count_files += 1
        if (AUDIO_OUTPUT_DIR / f"sentence_{sid}_3_english_translation.mp3").exists():
            count_files += 1
        if (AUDIO_OUTPUT_DIR / f"sentence_{sid}_4_breakdown_combined.mp3").exists():
            count_files += 1
        if (AUDIO_OUTPUT_DIR / f"sentence_{sid}_5_japanese_alternating.mp3").exists():
            count_files += 1

        if errors:
            print(f"✓{count_files} ✗{len(errors)}", end="")
            if skipped_parts:
                print(f" -{len(skipped_parts)}", end="")
            print()
        elif skipped_parts:
            print(f"✓{count_files} -{len(skipped_parts)}")
        else:
            print(f"✓{count_files}")

        # Count as success if we have at least 1,2,3,5
        if (AUDIO_OUTPUT_DIR / f"sentence_{sid}_1_sentence_number.mp3").exists() and \
           (AUDIO_OUTPUT_DIR / f"sentence_{sid}_2_japanese_male.mp3").exists() and \
           (AUDIO_OUTPUT_DIR / f"sentence_{sid}_3_english_translation.mp3").exists() and \
           (AUDIO_OUTPUT_DIR / f"sentence_{sid}_5_japanese_alternating.mp3").exists():
            success_count += 1

        time.sleep(0.1)

    # Final summary
    total_time = time.time() - start_time
    final_files = list(AUDIO_OUTPUT_DIR.glob("sentence_*.mp3"))

    file_counts = {str(i): 0 for i in range(1, 6)}
    for f in final_files:
        match = re.search(r"sentence_\d+_(\d+)_", f.name)
        if match:
            part = match.group(1)
            if part in file_counts:
                file_counts[part] += 1

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Time: {total_time/60:.1f} minutes")
    print(f"Sentences: {len(sentences)}")
    print(f"Success rate: {success_count/len(sentences)*100:.1f}%")
    print(f"Avg per sentence: {total_time/len(sentences):.1f}s")
    print(f"\nFiles created: {len(final_files)}")
    print("File breakdown:")
    for part_num in sorted(file_counts.keys()):
        desc = {
            "1": "sentence_number",
            "2": "japanese_male",
            "3": "english_translation",
            "4": "breakdown_combined",
            "5": "japanese_alternating",
        }.get(part_num, f"part_{part_num}")
        print(f"  Part {part_num} ({desc}): {file_counts[part_num]}")
    print(f"\nOutput: {AUDIO_OUTPUT_DIR}")

if __name__ == "__main__":
    main()