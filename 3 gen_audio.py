import json
import subprocess
import pathlib
import re
import time
import sys
import random
import wave
from pydub import AudioSegment
from typing import Dict, List, Optional, Tuple


# ---------- CONFIG ----------
BAL4WEB = r"C:\Program Files (x86)\Balabolka\bal4web\bal4web.exe"
INPUT_JSON = "translated_output_part.json"
LANG_CODE_JA = "ja-JP"
LANG_CODE_EN = "en-US"

# Audio processing settings
PAUSE_IN_BREAKDOWN_MS = 500      # pause between Japanese term and English breakdown in part 4
PAUSE_BETWEEN_BREAKDOWNS_MS = 800  # Pause between different breakdown pairs
PAUSE_IN_ALTERNATING_MS = 800    # pause between female and male voices in part 6
PAUSE_END_SILENCE_MS = 900       # Silence appended to end of ALL final segment files (parts 1-6)

# Japanese speech rates for different segments
JA_SPEECH_RATE_SENTENCE_NUMBER = 1
JA_SPEECH_RATE_JAPANESE_MALE = 0.75
JA_SPEECH_RATE_ALTERNATING = 0.90

FORCE_SAMPLE_RATE_KHZ = 24       # Output sample rate (24kHz) for WAV consistency

# Robustness settings
MAX_RETRIES = 5
RETRY_BASE_SLEEP = 0.7
RETRY_JITTER = 0.4
MIN_WAV_BYTES = 120
FALLBACK_TARGET_SAMPLE_RATE_HZ = 24000
FALLBACK_TARGET_CHANNELS = 1
FALLBACK_TARGET_SAMPLE_WIDTH_BYTES = 2

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

def clean_text(text: str) -> str:
    """Clean text for TTS."""
    if not text:
        return ""
    text = text.replace('"', ',')
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def file_has_audio(p: pathlib.Path) -> bool:
    return p.exists() and p.is_file() and p.stat().st_size > MIN_WAV_BYTES


def safe_unlink(file_path: pathlib.Path) -> None:
    """Safely delete a file with error handling."""
    try:
        file_path.unlink(missing_ok=True)
    except Exception:
        pass


def convert_sentence_wavs_to_mp3(sentence_id: str, audio_dir: pathlib.Path) -> None:
    """Convert WAV files to MP3."""
    for wav_file in audio_dir.glob(f"sentence_{sentence_id}_*.wav"):
        try:
            audio = AudioSegment.from_wav(str(wav_file))
            audio.export(str(wav_file.with_suffix(".mp3")), format="mp3", bitrate="192k")
            safe_unlink(wav_file)
        except Exception:
            pass


def add_end_silence_to_wav(input_file: pathlib.Path, output_file: Optional[pathlib.Path] = None) -> bool:
    """Add end silence to a WAV file using pydub."""
    if output_file is None:
        output_file = input_file
    
    try:
        audio = AudioSegment.from_wav(str(input_file))
        silence = AudioSegment.silent(duration=PAUSE_END_SILENCE_MS,
                                       frame_rate=audio.frame_rate)
        silence = silence.set_channels(audio.channels).set_sample_width(audio.sample_width)
        audio_with_silence = audio + silence
        
        # Export to temporary file first, then replace
        temp_file = output_file.with_suffix(f".temp{output_file.suffix}")
        audio_with_silence.export(str(temp_file), format="wav")
        
        # Replace original with new file
        safe_unlink(output_file)
        temp_file.rename(output_file)
        return True
    except Exception:
        return False


def test_voice(voice_name: str, lang_code: str) -> Optional[Dict]:
    """Test if a voice works."""
    test_text = "こんにちは" if lang_code == LANG_CODE_JA else "Hello"
    test_file = AUDIO_OUTPUT_DIR / "_test.wav"

    args = [BAL4WEB, "-s", "microsoft", "-l", lang_code,
            "-n", voice_name, "-t", test_text, "-w", str(test_file),
            "-fr", str(FORCE_SAMPLE_RATE_KHZ)]

    try:
        subprocess.run(args, check=True, capture_output=True, text=True, timeout=10, encoding="utf-8")
        if file_has_audio(test_file):
            safe_unlink(test_file)
            return {"name": voice_name, "lang": lang_code}
    except Exception:
        pass

    safe_unlink(test_file)
    return None


class TTSGenerator:
    """Handles TTS generation with segment-specific settings."""
    
    @staticmethod
    def _get_speech_rate(lang_code: str, segment_type: str) -> Optional[float]:
        """Get speech rate based on language and segment type."""
        if lang_code != LANG_CODE_JA:
            return None
            
        rates = {
            "sentence_number": JA_SPEECH_RATE_SENTENCE_NUMBER,
            "japanese_male": JA_SPEECH_RATE_JAPANESE_MALE,
            "alternating": JA_SPEECH_RATE_ALTERNATING,
        }
        return rates.get(segment_type, JA_SPEECH_RATE_JAPANESE_MALE)
    
    @staticmethod
    def build_args(text: str, output_file: pathlib.Path, voice_params: Dict, 
                   include_lang: bool, segment_type: str = None, add_end_silence: bool = True) -> List[str]:
        """Build command-line arguments for bal4web."""
        voice_name = voice_params["name"]
        lang_code = voice_params["lang"]

        args = [BAL4WEB, "-s", "microsoft"]
        if include_lang:
            args.extend(["-l", lang_code])

        args.extend(["-n", voice_name, "-t", text, "-w", str(output_file),
                     "-fr", str(FORCE_SAMPLE_RATE_KHZ)])
        
        # Add end silence when requested
        if add_end_silence and PAUSE_END_SILENCE_MS > 0:
            args.extend(["-se", str(PAUSE_END_SILENCE_MS)])

        # Apply speech rate for Japanese segments
        if speech_rate := TTSGenerator._get_speech_rate(lang_code, segment_type):
            args.extend(["-r", str(speech_rate)])

        return args


def generate_audio_once(text: str, output_file: pathlib.Path, voice_params: Dict, 
                        segment_type: str = None, add_end_silence: bool = True) -> bool:
    """Single attempt to generate audio with segment-specific settings."""
    if not text or not text.strip():
        return False

    text = clean_text(text)
    if not text:
        return False

    # Attempt with language code
    args_lang = TTSGenerator.build_args(text, output_file, voice_params, 
                                       include_lang=True, segment_type=segment_type,
                                       add_end_silence=add_end_silence)
    try:
        subprocess.run(args_lang, check=True, capture_output=True, text=True, timeout=30, encoding="utf-8")
        if file_has_audio(output_file):
            return True
    except subprocess.CalledProcessError:
        # Fallback without language code
        args_nolang = TTSGenerator.build_args(text, output_file, voice_params,
                                            include_lang=False, segment_type=segment_type,
                                            add_end_silence=add_end_silence)
        try:
            subprocess.run(args_nolang, check=True, capture_output=True, text=True, timeout=30, encoding="utf-8")
            return file_has_audio(output_file)
        except Exception:
            return False
    except Exception:
        return False

    return False


def generate_audio_retry(text: str, output_file: pathlib.Path, voice_params: Dict, 
                         segment_type: str = None, add_end_silence: bool = True,
                         attempts: int = MAX_RETRIES) -> bool:
    """Retry wrapper with exponential backoff."""
    for i in range(1, attempts + 1):
        safe_unlink(output_file)
        ok = generate_audio_once(text, output_file, voice_params, segment_type, add_end_silence)
        if ok:
            return True

        if i < attempts:
            sleep_s = (RETRY_BASE_SLEEP * (2 ** (i - 1))) + random.uniform(0, RETRY_JITTER)
            time.sleep(sleep_s)

    return False


class AudioCombiner:
    """Handles audio file combination with pause insertion."""
    
    @staticmethod
    def _wav_params_tuple(w: wave.Wave_read) -> Tuple[int, int, int, int]:
        """Get WAV parameters as a comparable tuple."""
        params = w.getparams()
        return (params.nchannels, params.sampwidth, params.framerate, 1 if params.comptype == "NONE" else 0)
    
    @staticmethod
    def combine_fast_with_silence(input_files: List[pathlib.Path], output_file: pathlib.Path, pause_ms: int) -> bool:
        """Fast WAV concatenation with silence insertion."""
        if not input_files:
            return False

        try:
            first_params = None
            all_frames = b""
            nchannels = sampwidth = framerate = 0

            for idx, input_file in enumerate(input_files):
                if not file_has_audio(input_file):
                    return False

                with wave.open(str(input_file), "rb") as w:
                    if first_params is None:
                        first_params = w.getparams()
                        nchannels, sampwidth, framerate, is_pcm = AudioCombiner._wav_params_tuple(w)
                        if is_pcm != 1:
                            return False
                    else:
                        current_params = AudioCombiner._wav_params_tuple(w)
                        if current_params != (nchannels, sampwidth, framerate, 1):
                            return False

                    # Insert silence between files
                    if pause_ms > 0 and idx > 0:
                        silence_frames = int(framerate * (pause_ms / 1000.0))
                        if silence_frames > 0:
                            silence_bytes = b"\x00" * (silence_frames * nchannels * sampwidth)
                            all_frames += silence_bytes

                    all_frames += w.readframes(w.getnframes())

            if first_params and all_frames:
                with wave.open(str(output_file), "wb") as out:
                    out.setparams(first_params)
                    out.writeframes(all_frames)
                return file_has_audio(output_file)
        except Exception:
            return False
        return False
    
    @staticmethod
    def combine_with_pydub(input_files: List[pathlib.Path], output_file: pathlib.Path, pause_ms: int) -> bool:
        """Robust fallback using pydub."""
        if not input_files:
            return False

        try:
            combined = None
            for idx, f in enumerate(input_files):
                if not file_has_audio(f):
                    return False

                seg = AudioSegment.from_wav(str(f))
                seg = seg.set_frame_rate(FALLBACK_TARGET_SAMPLE_RATE_HZ)
                seg = seg.set_channels(FALLBACK_TARGET_CHANNELS)
                seg = seg.set_sample_width(FALLBACK_TARGET_SAMPLE_WIDTH_BYTES)

                if combined is None:
                    combined = seg
                else:
                    if pause_ms > 0:
                        silence = AudioSegment.silent(duration=pause_ms,
                                                       frame_rate=FALLBACK_TARGET_SAMPLE_RATE_HZ)
                        silence = silence.set_channels(FALLBACK_TARGET_CHANNELS).set_sample_width(FALLBACK_TARGET_SAMPLE_WIDTH_BYTES)
                        combined += silence
                    combined += seg

            if combined:
                combined.export(str(output_file), format="wav")
                return file_has_audio(output_file)
        except Exception:
            pass
        return False
    
    @staticmethod
    def combine_wav_files(input_files: List[pathlib.Path], output_file: pathlib.Path, pause_ms: int = 0) -> bool:
        """Combine WAV files with optional pauses."""
        if not input_files:
            return False

        return (AudioCombiner.combine_fast_with_silence(input_files, output_file, pause_ms) or
                AudioCombiner.combine_with_pydub(input_files, output_file, pause_ms))


class SentenceProcessor:
    """Processes a single sentence with all its audio segments."""
    
    def __init__(self, sid: str, row: Dict, voices: Dict, audio_dir: pathlib.Path):
        self.sid = sid
        self.row = row
        self.voices = voices
        self.audio_dir = audio_dir
        
        self.jp_text = clean_text(row.get("japanese", ""))
        self.en_text = clean_text(row.get("english", ""))
        self.en_literal = clean_text(row.get("english_literal", ""))
        
        self.errors = []
        self.skipped_parts = []
        self.created_parts = []  # Track which parts were successfully created
    
    def _extract_part_number(self, filename: str) -> str:
        """Extract part number from filename like '1_sentence_number' or '3_english_translation'."""
        parts = filename.split('_')
        if parts and parts[0].isdigit():
            return parts[0]
        return "?"
    
    def generate_segment(self, text: str, filename: str, voice_key: str, 
                        segment_type: str = None, optional: bool = False,
                        add_end_silence: bool = True) -> Optional[pathlib.Path]:
        """Generate a single audio segment."""
        part_num = self._extract_part_number(filename)
        
        if not text and optional:
            self.skipped_parts.append(part_num)
            return None
            
        if not text:
            # If text is empty and NOT optional, it's an error
            if not optional:
                self.errors.append(part_num)
            return None
            
        temp_file = self.audio_dir / f"sentence_{self.sid}_{filename}_temp.wav"
        
        if voice_key not in self.voices:
            if optional:
                self.skipped_parts.append(part_num)
                return None
            self.errors.append(part_num)
            return None
        
        if generate_audio_retry(text, temp_file, self.voices[voice_key], 
                               segment_type, add_end_silence):
            return temp_file
        else:
            if optional:
                self.skipped_parts.append(part_num)
            else:
                self.errors.append(part_num)
            return None
    
    def process_sentence_number(self) -> Optional[Tuple[pathlib.Path, pathlib.Path]]:
        """Process sentence number segment."""
        num_text = f"{self.sid}番目"
        gender = random.choice(["male", "female"])
        temp_file = self.generate_segment(num_text, "1_sentence_number", 
                                         f"ja_{gender}", "sentence_number", 
                                         add_end_silence=True)
        if temp_file:
            final_file = self.audio_dir / f"sentence_{self.sid}_1_sentence_number.wav"
            return (temp_file, final_file)
        return None
    
    def process_japanese_male(self) -> Optional[Tuple[pathlib.Path, pathlib.Path]]:
        """Process Japanese male segment."""
        if not self.jp_text:
            self.skipped_parts.append("2")
            return None
            
        temp_file = self.generate_segment(self.jp_text, "2_japanese_male", 
                                         "ja_male", "japanese_male",
                                         add_end_silence=True)
        if temp_file:
            final_file = self.audio_dir / f"sentence_{self.sid}_2_japanese_male.wav"
            return (temp_file, final_file)
        return None
    
    def process_english_translation(self) -> Optional[Tuple[pathlib.Path, pathlib.Path]]:
        """Process English translation segment."""
        if not self.en_text:
            # English translation is REQUIRED, not optional
            self.errors.append("3")
            return None
            
        temp_file = self.generate_segment(self.en_text, "3_english_translation", 
                                         "en_male", optional=False,
                                         add_end_silence=True)
        if temp_file:
            final_file = self.audio_dir / f"sentence_{self.sid}_3_english_translation.wav"
            return (temp_file, final_file)
        return None
    
    def process_breakdown(self) -> Optional[pathlib.Path]:
        """Process breakdown combined segment with pauses between pairs."""
        final_file = self.audio_dir / f"sentence_{self.sid}_4_breakdown_combined.wav"
        breakdown_pairs = self._collect_breakdown_pairs()
        
        if not breakdown_pairs or "en_male" not in self.voices:
            self.skipped_parts.append("4")
            return None
        
        try:
            combined = None
            
            for pair_idx, (part_jp, breakdown_en) in enumerate(breakdown_pairs, 1):
                # Generate Japanese part WITHOUT end silence
                temp_jp = self.audio_dir / f"sentence_{self.sid}_4_{pair_idx}_jp_temp.wav"
                if not generate_audio_retry(part_jp, temp_jp, self.voices["ja_male"], 
                                           "japanese_male", add_end_silence=False):
                    self.errors.append("4")
                    return None
                
                # Generate English breakdown WITHOUT end silence
                temp_en = self.audio_dir / f"sentence_{self.sid}_4_{pair_idx}_en_temp.wav"
                if not generate_audio_retry(breakdown_en, temp_en, self.voices["en_male"],
                                           add_end_silence=False):
                    self.errors.append("4")
                    safe_unlink(temp_jp)
                    return None
                
                # Load and normalize audio
                jp_audio = AudioSegment.from_wav(str(temp_jp))
                en_audio = AudioSegment.from_wav(str(temp_en))
                
                jp_audio = jp_audio.set_frame_rate(FALLBACK_TARGET_SAMPLE_RATE_HZ)
                jp_audio = jp_audio.set_channels(FALLBACK_TARGET_CHANNELS)
                jp_audio = jp_audio.set_sample_width(FALLBACK_TARGET_SAMPLE_WIDTH_BYTES)
                
                en_audio = en_audio.set_frame_rate(FALLBACK_TARGET_SAMPLE_RATE_HZ)
                en_audio = en_audio.set_channels(FALLBACK_TARGET_CHANNELS)
                en_audio = en_audio.set_sample_width(FALLBACK_TARGET_SAMPLE_WIDTH_BYTES)
                
                # Combine Japanese + pause + English for this pair
                pair_combined = jp_audio + AudioSegment.silent(duration=PAUSE_IN_BREAKDOWN_MS,
                                                                frame_rate=FALLBACK_TARGET_SAMPLE_RATE_HZ) + en_audio
                
                # Add pause between breakdown pairs (except after the last one)
                if combined is None:
                    combined = pair_combined
                else:
                    # Add pause between this pair and previous one
                    pause_between = AudioSegment.silent(duration=PAUSE_BETWEEN_BREAKDOWNS_MS,
                                                         frame_rate=FALLBACK_TARGET_SAMPLE_RATE_HZ)
                    combined = combined + pause_between + pair_combined
                
                # Cleanup temp files
                safe_unlink(temp_jp)
                safe_unlink(temp_en)
            
            if combined:
                # Add end silence to the final combined file
                if PAUSE_END_SILENCE_MS > 0:
                    end_silence = AudioSegment.silent(duration=PAUSE_END_SILENCE_MS,
                                                       frame_rate=FALLBACK_TARGET_SAMPLE_RATE_HZ)
                    combined = combined + end_silence
                
                # Export final file
                combined.export(str(final_file), format="wav")
                if file_has_audio(final_file):
                    self.created_parts.append("4")  # Track that part 4 was created
                    return final_file
        
        except Exception:
            self.errors.append("4")
        
        return None
    
    def process_english_literal(self) -> Optional[Tuple[pathlib.Path, pathlib.Path]]:
        """Process English literal translation segment."""
        # Only create english_literal if part 4 (breakdown) was successfully created
        if not self.en_literal or "4" not in self.created_parts:
            self.skipped_parts.append("5")
            return None
            
        temp_file = self.generate_segment(self.en_literal, "5_english_literal", 
                                         "en_male", optional=True,
                                         add_end_silence=True)
        if temp_file:
            final_file = self.audio_dir / f"sentence_{self.sid}_5_english_literal.wav"
            return (temp_file, final_file)
        return None
    
    def process_alternating(self) -> Optional[pathlib.Path]:
        """Process Japanese alternating segment."""
        final_file = self.audio_dir / f"sentence_{self.sid}_6_japanese_alternating.wav"
        
        if not self.jp_text:
            self.skipped_parts.append("6")
            return None
        
        # Generate female version WITHOUT end silence
        temp6a = self.audio_dir / f"sentence_{self.sid}_6a_temp.wav"
        if not generate_audio_retry(self.jp_text, temp6a, self.voices["ja_female"], 
                                   "alternating", add_end_silence=False):
            self.errors.append("6")
            return None
        
        # Generate male version WITHOUT end silence (fresh, with alternating speed)
        temp6b = self.audio_dir / f"sentence_{self.sid}_6b_temp.wav"
        if not generate_audio_retry(self.jp_text, temp6b, self.voices["ja_male"], 
                                   "alternating", add_end_silence=False):
            self.errors.append("6")
            safe_unlink(temp6a)
            return None
        
        # Combine with the TOTAL pause specified in PAUSE_IN_ALTERNATING_MS
        if AudioCombiner.combine_wav_files([temp6a, temp6b], final_file, 
                                          pause_ms=PAUSE_IN_ALTERNATING_MS):
            # Add end silence to the final combined file
            if PAUSE_END_SILENCE_MS > 0:
                add_end_silence_to_wav(final_file)
            
            safe_unlink(temp6a)
            safe_unlink(temp6b)
            return final_file
        else:
            self.errors.append("6")
            safe_unlink(temp6a)
            safe_unlink(temp6b)
            return None
    
    def _collect_breakdown_pairs(self) -> List[Tuple[str, str]]:
        """Collect all breakdown pairs from the row data."""
        pairs = []
        
        # Old format
        old_part = clean_text(self.row.get("part_to_breakdown", ""))
        old_breakdown = clean_text(self.row.get("breakdown", ""))
        if old_part and old_breakdown:
            pairs.append((old_part, old_breakdown))
        
        # New format (up to 3 breakdowns)
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
        f"sentence_{sentence_id}_6_japanese_alternating.mp3"
    ]
    return all((audio_dir / pattern).exists() for pattern in required_patterns)


def main():
    print("=" * 60)
    print("Japanese Audio Generator (Final)")
    print("=" * 60)

    if not pathlib.Path(BAL4WEB).exists():
        print(f"ERROR: bal4web.exe not found at {BAL4WEB}")
        sys.exit(1)

    AUDIO_OUTPUT_DIR.mkdir(exist_ok=True)

    # Voice detection
    print("Detecting voices...")
    voices: Dict[str, Dict] = {}

    for gender in ["male", "female"]:
        for voice_name in JAPANESE_VOICES[gender]["candidates"]:
            if voice := test_voice(voice_name, LANG_CODE_JA):
                voices[f"ja_{gender}"] = voice
                print(f"✓ Japanese {gender}: {voice_name}")
                break
        else:
            print(f"✗ Japanese {gender}: Not found")

    for voice_name in ENGLISH_VOICES:
        if voice := test_voice(voice_name, LANG_CODE_EN):
            voices["en_male"] = voice
            print(f"✓ English: {voice_name}")
            break
    else:
        print("✗ English: Not found")

    if not voices.get("ja_male") or not voices.get("ja_female"):
        print("ERROR: Need both Japanese voices")
        sys.exit(1)

    # Load JSON data
    if not INPUT_JSON_PATH.exists():
        print(f"ERROR: {INPUT_JSON} not found")
        sys.exit(1)

    with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
        root = json.load(f)

    if not (sentences := root.get("sentences", {})):
        print("ERROR: No sentences found")
        sys.exit(1)

    print(f"Loaded {len(sentences)} sentences")
    sentence_ids = sorted(sentences.keys(), key=lambda x: int(x) if x.isdigit() else float("inf"))

    print(f"\nUsing settings:")
    print(f"  Japanese speech rates:")
    print(f"    Sentence number: {JA_SPEECH_RATE_SENTENCE_NUMBER}")
    print(f"    Japanese male: {JA_SPEECH_RATE_JAPANESE_MALE}")
    print(f"    Alternating: {JA_SPEECH_RATE_ALTERNATING}")
    print(f"  Pauses:")
    print(f"    Breakdown (Part 4): {PAUSE_IN_BREAKDOWN_MS}ms between components")
    print(f"    Between breakdown pairs: {PAUSE_BETWEEN_BREAKDOWNS_MS}ms")
    print(f"    Alternating (Part 6): {PAUSE_IN_ALTERNATING_MS}ms between voices")
    print(f"    End silence on ALL parts (1-6): {PAUSE_END_SILENCE_MS}ms")
    print("\nStarting generation...")

    start_time = time.time()
    success_count = 0

    for idx, sid in enumerate(sentence_ids, 1):
        row = sentences[sid]

        # Progress tracking with ETA
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

        # Process sentence
        processor = SentenceProcessor(sid, row, voices, AUDIO_OUTPUT_DIR)
        
        # Generate all segments in the correct order
        segments_to_rename = []
        
        # Parts 1-3 (with end silence generated directly)
        for processor_func in [processor.process_sentence_number, 
                               processor.process_japanese_male, 
                               processor.process_english_translation]:
            if result := processor_func():
                segments_to_rename.append(result)
        
        # Part 4 (breakdown) - with pauses between pairs
        # Must be called BEFORE process_english_literal() so created_parts is updated
        if breakdown_file := processor.process_breakdown():
            # File already has final name with end silence
            pass
        
        # Part 5 (english_literal) - only if part 4 was created
        if result := processor.process_english_literal():
            segments_to_rename.append(result)
        
        # Part 6 (alternating) - components without end silence, final file gets end silence
        if alternating_file := processor.process_alternating():
            # File already has final name with end silence
            pass

        # Rename temp files to final names
        for temp_path, final_path in segments_to_rename:
            try:
                if temp_path.exists():
                    if final_path.exists():
                        safe_unlink(final_path)
                    temp_path.rename(final_path)
            except Exception:
                pass

        # Clean up any remaining temp files
        for temp_file in AUDIO_OUTPUT_DIR.glob(f"sentence_{sid}_*temp*.wav"):
            safe_unlink(temp_file)

        # Convert WAV files to MP3 for this sentence
        convert_sentence_wavs_to_mp3(sid, AUDIO_OUTPUT_DIR)

        # Count created files for this sentence
        count_files = count_sentence_files(sid, AUDIO_OUTPUT_DIR)

        # Status output - sort errors/skips for better readability
        errors_sorted = sorted(set(processor.errors))
        skips_sorted = sorted(set(processor.skipped_parts))
        
        status_parts = []
        if errors_sorted:
            status_parts.append(f"ERROR: {','.join(errors_sorted)}")
        if skips_sorted:
            status_parts.append(f"[o: {','.join(skips_sorted)}]")
        
        print(f"✓{count_files} " + " ".join(status_parts) if status_parts else f"✓{count_files}")

        # Count as success if we have all required files
        if sentence_has_required_files(sid, AUDIO_OUTPUT_DIR):
            success_count += 1

        time.sleep(0.1)

    # Final summary
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