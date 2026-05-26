# divide_en.py
import re
from pathlib import Path
from typing import List

# ===================== CONFIGURATION =====================
INPUT_FILE = "en.txt"                 # Your input English text file
OUTPUT_FILE = "divided_output.txt"    # Output will be saved here

# Batch settings
MAX_CHARS_PER_BATCH = 500              # Maximum characters per batch
MIN_PARAGRAPHS_PER_BATCH = 1           # Kept for compatibility/readability
MAX_PARAGRAPHS_PER_BATCH = 10          # Maximum paragraphs/chunks per batch
# =========================================================


print("=" * 50)
print("ENGLISH TEXT DIVIDER")
print("=" * 50)
print(f"Input:  {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")
print(f"Max chars per batch: {MAX_CHARS_PER_BATCH}")
print("=" * 50)


def clean_english_text(text: str) -> str:
    """
    Clean English source text while preserving paragraph structure.

    This replaces common typographic punctuation with simpler ASCII forms,
    normalizes whitespace inside lines, and keeps blank lines so paragraph
    splitting still works.
    """
    if not text:
        return ""

    # Normalize line endings first.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    replacements = {
        "\ufeff": "",     # UTF-8 BOM
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "—": "-",
        "–": "-",
        "―": "-",
        "…": "...",
        "\u00a0": " ",   # non-breaking space
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    cleaned_lines = []
    for line in text.split("\n"):
        # Collapse spaces/tabs inside each line but preserve line breaks.
        line = re.sub(r"[ \t]+", " ", line).strip()
        cleaned_lines.append(line)

    # Collapse 3+ blank lines down to 2 blank lines.
    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    return cleaned_text.strip()


def safe_read_file(file_path: str) -> str:
    """Read file with multiple encoding attempts, with text cleaning."""
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]

    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                raw_content = f.read()
            if encoding != "utf-8":
                print(f"✓ Read with encoding: {encoding}")
            return clean_english_text(raw_content)
        except UnicodeDecodeError:
            continue

    raise ValueError("Cannot read file with any supported encoding")


def split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs based on empty lines."""
    if not text:
        return []

    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    paragraphs = []
    current_para = []

    for line in lines:
        stripped = line.rstrip()

        if not stripped:
            if current_para:
                paragraph = "\n".join(current_para)
                if paragraph.strip():
                    paragraphs.append(paragraph)
                current_para = []
        else:
            current_para.append(line)

    if current_para:
        paragraph = "\n".join(current_para)
        if paragraph.strip():
            paragraphs.append(paragraph)

    return paragraphs


def split_english_sentences(paragraph: str) -> List[str]:
    """
    Split an English paragraph into sentences.

    This is intentionally lightweight and dependency-free. It tries to avoid
    splitting at common abbreviations and decimal numbers, while still handling
    quotes after sentence-ending punctuation.
    """
    if not paragraph.strip():
        return []

    abbreviations = {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st",
        "vs", "etc", "e.g", "i.e", "fig", "no", "vol", "ch",
        "u.s", "u.k", "a.m", "p.m",
    }

    sentences = []
    current = []
    text = paragraph.strip()
    i = 0

    while i < len(text):
        char = text[i]
        current.append(char)

        if char in ".!?":
            prev_char = text[i - 1] if i > 0 else ""
            next_char = text[i + 1] if i + 1 < len(text) else ""

            # Do not split decimal numbers like 3.14.
            if char == "." and prev_char.isdigit() and next_char.isdigit():
                i += 1
                continue

            current_text = "".join(current).strip()
            last_token_match = re.search(r"([A-Za-z](?:[A-Za-z]|\.)*)\.$", current_text)
            last_token = last_token_match.group(1).lower() if last_token_match else ""

            # Do not split after common abbreviations.
            if char == "." and last_token in abbreviations:
                i += 1
                continue

            # Include closing quotes/brackets after punctuation.
            j = i + 1
            while j < len(text) and text[j] in "\"')]}”’":
                current.append(text[j])
                j += 1

            # Split if end of paragraph or followed by whitespace.
            if j >= len(text) or text[j].isspace():
                sentence = "".join(current).strip()
                if sentence:
                    sentences.append(sentence)
                current = []
                i = j
                while i < len(text) and text[i].isspace():
                    i += 1
                continue

        i += 1

    remainder = "".join(current).strip()
    if remainder:
        sentences.append(remainder)

    return sentences


def split_long_paragraph(paragraph: str) -> List[str]:
    """Split a paragraph longer than MAX_CHARS_PER_BATCH."""
    if len(paragraph) <= MAX_CHARS_PER_BATCH:
        return [paragraph]

    # Method 1: Split at English sentence endings.
    sentences = split_english_sentences(paragraph)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        extra_space = 1 if current_chunk else 0
        if current_chunk and len(current_chunk) + extra_space + len(sentence) > MAX_CHARS_PER_BATCH:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Method 2: If still too long, split by line breaks.
    if len(chunks) == 1 and len(chunks[0]) > MAX_CHARS_PER_BATCH:
        lines = chunks[0].split("\n")
        chunks = []
        current_chunk = ""

        for line in lines:
            line = line.strip()
            if current_chunk and len(current_chunk) + len(line) + 1 > MAX_CHARS_PER_BATCH:
                chunks.append(current_chunk.strip())
                current_chunk = line
            else:
                if current_chunk:
                    current_chunk += "\n" + line
                else:
                    current_chunk = line

        if current_chunk:
            chunks.append(current_chunk.strip())

    # Method 3: Final fallback - split by character count, preferably at spaces/punctuation.
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= MAX_CHARS_PER_BATCH:
            final_chunks.append(chunk)
            continue

        start = 0
        while start < len(chunk):
            end = min(start + MAX_CHARS_PER_BATCH, len(chunk))
            piece = chunk[start:end]

            if end < len(chunk):
                split_pos = -1
                for j in range(len(piece) - 1, max(0, len(piece) - 120), -1):
                    if piece[j] in " \n,;:.!?-":
                        split_pos = j + 1
                        break

                if split_pos > 0:
                    piece = piece[:split_pos]
                    end = start + split_pos

            final_chunks.append(piece.strip())
            start = end

    return [chunk for chunk in final_chunks if chunk]


def create_batches(paragraphs: List[str]) -> List[str]:
    """Create batches from paragraphs/chunks - returns list of batch texts."""
    if not paragraphs:
        return []

    batches = []
    current_batch = []
    current_batch_chars = 0

    for paragraph in paragraphs:
        chunks = split_long_paragraph(paragraph) if len(paragraph) > MAX_CHARS_PER_BATCH else [paragraph]

        for chunk in chunks:
            chunk_len = len(chunk)
            need_new_batch = False

            if current_batch:
                # +2 accounts roughly for the paragraph separator inside a batch.
                if current_batch_chars + chunk_len + 2 > MAX_CHARS_PER_BATCH:
                    need_new_batch = True
                elif len(current_batch) >= MAX_PARAGRAPHS_PER_BATCH:
                    need_new_batch = True

            if need_new_batch:
                batch_text = "\n\n".join(current_batch)
                batches.append(batch_text)
                current_batch = [chunk]
                current_batch_chars = chunk_len
            else:
                current_batch.append(chunk)
                current_batch_chars += chunk_len + (2 if len(current_batch) > 1 else 0)

    if current_batch:
        batch_text = "\n\n".join(current_batch)
        batches.append(batch_text)

    return batches


def save_batches(batches: List[str], output_path: str) -> None:
    """Save batches with simple separators."""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, batch_text in enumerate(batches, 1):
            f.write(batch_text)
            if i < len(batches):
                f.write("\n\n---\n\n")


def main() -> bool:
    """Main function."""
    try:
        input_path = Path(INPUT_FILE)
        if not input_path.exists():
            print(f"❌ Error: Input file '{INPUT_FILE}' not found!")
            print(f"   Current directory: {Path.cwd()}")
            return False

        print(f"📖 Reading '{INPUT_FILE}'...")
        text = safe_read_file(INPUT_FILE)

        if not text.strip():
            print("❌ Error: File is empty")
            return False

        print(f"✓ File read and cleaned ({len(text):,} chars)")
        print("  (English punctuation and whitespace normalized)")

        print("🔪 Splitting into paragraphs...")
        paragraphs = split_into_paragraphs(text)

        if not paragraphs:
            print("❌ Error: No paragraphs found")
            return False

        print(f"✓ Found {len(paragraphs)} paragraphs")

        long_paragraphs = sum(1 for p in paragraphs if len(p) > MAX_CHARS_PER_BATCH)
        if long_paragraphs > 0:
            print(f"   {long_paragraphs} paragraphs will be split (> {MAX_CHARS_PER_BATCH} chars)")

        print("📦 Creating batches...")
        batches = create_batches(paragraphs)

        if not batches:
            print("❌ Error: No batches created")
            return False

        print(f"✓ Created {len(batches)} batches")

        print(f"\n💾 Saving to '{OUTPUT_FILE}'...")
        save_batches(batches, OUTPUT_FILE)

        total_chars = sum(len(batch) for batch in batches)
        avg_batch_chars = total_chars // len(batches) if batches else 0
        near_limit = sum(1 for batch in batches if len(batch) > MAX_CHARS_PER_BATCH * 0.9)

        print("\n" + "=" * 50)
        print("✅ COMPLETE")
        print("=" * 50)
        print(f"Batches created: {len(batches)}")
        print(f"Total characters: {total_chars:,}")
        print(f"Average batch size: {avg_batch_chars:,} chars")
        if near_limit > 0:
            print(f"Batches near limit (>90%): {near_limit}")
        print(f"Output file: {OUTPUT_FILE}")
        print("=" * 50)

        if batches:
            print("\n📄 First batch (first 200 chars):")
            preview = batches[0][:200]
            print(f'"""\n{preview}...\n"""')
            print("\nSeparator between batches: '\\n\\n---\\n\\n'")

        return True

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


# ===================== RUN =====================
if __name__ == "__main__":
    print()
    success = main()

    if success:
        print("\n✅ Done!")
    else:
        print("\n❌ Failed!")
