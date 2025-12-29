# txt_divider_clean.py
import re
from pathlib import Path
from typing import List

# ===================== CONFIGURATION =====================
INPUT_FILE = "jp.txt"          # Your input text file
OUTPUT_FILE = "divided_output.txt"  # Output will be saved here

# Batch settings
MAX_CHARS_PER_BATCH = 1000          # Maximum characters per batch
MIN_PARAGRAPHS_PER_BATCH = 1        # Minimum paragraphs per batch
MAX_PARAGRAPHS_PER_BATCH = 10       # Maximum paragraphs per batch
# =========================================================

print("=" * 50)
print("TEXT DIVIDER")
print("=" * 50)
print(f"Input:  {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")
print(f"Max chars per batch: {MAX_CHARS_PER_BATCH}")
print("=" * 50)

def safe_read_file(file_path: str) -> str:
    """Read file with multiple encoding attempts."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        encodings = ['shift_jis', 'euc-jp', 'cp932', 'latin-1', 'utf-8-sig']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                print(f"✓ Read with encoding: {encoding}")
                return content
            except:
                continue
        raise ValueError(f"Cannot read file with any supported encoding")

def split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs based on empty lines."""
    if not text:
        return []
    
    lines = text.replace('\r\n', '\n').replace('\r', '\n').split('\n')
    paragraphs = []
    current_para = []
    
    for line in lines:
        stripped = line.rstrip()
        
        if not stripped:
            if current_para:
                paragraph = '\n'.join(current_para)
                if paragraph.strip():
                    paragraphs.append(paragraph)
                current_para = []
        else:
            current_para.append(line)
    
    if current_para:
        paragraph = '\n'.join(current_para)
        if paragraph.strip():
            paragraphs.append(paragraph)
    
    return paragraphs

def split_long_paragraph(paragraph: str) -> List[str]:
    """Split a paragraph longer than MAX_CHARS_PER_BATCH."""
    if len(paragraph) <= MAX_CHARS_PER_BATCH:
        return [paragraph]
    
    # Method 1: Split at Japanese sentence endings
    sentences = re.split(r'([。！？\n])', paragraph)
    chunks = []
    current_chunk = ""
    
    i = 0
    while i < len(sentences):
        if i + 1 < len(sentences):
            sentence = sentences[i] + sentences[i + 1]
            i += 2
        else:
            sentence = sentences[i]
            i += 1
        
        if current_chunk and len(current_chunk) + len(sentence) > MAX_CHARS_PER_BATCH:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += sentence
            else:
                current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Method 2: If still too long, split by line breaks
    if len(chunks) == 1 and len(chunks[0]) > MAX_CHARS_PER_BATCH:
        lines = chunks[0].split('\n')
        chunks = []
        current_chunk = ""
        
        for line in lines:
            if current_chunk and len(current_chunk) + len(line) + 1 > MAX_CHARS_PER_BATCH:
                chunks.append(current_chunk.strip())
                current_chunk = line
            else:
                if current_chunk:
                    current_chunk += '\n' + line
                else:
                    current_chunk = line
        
        if current_chunk:
            chunks.append(current_chunk.strip())
    
    # Method 3: Final fallback - split by character count
    if len(chunks) == 1 and len(chunks[0]) > MAX_CHARS_PER_BATCH:
        text = chunks[0]
        chunks = []
        for i in range(0, len(text), MAX_CHARS_PER_BATCH):
            chunk = text[i:i + MAX_CHARS_PER_BATCH]
            
            # Try to avoid cutting words
            if i + MAX_CHARS_PER_BATCH < len(text):
                lookback = min(100, len(chunk))
                for j in range(lookback, 0, -1):
                    if chunk[-j] in ' \n。！？、，,.':
                        chunk = chunk[:-j] + '\n'
                        break
            
            chunks.append(chunk.strip())
    
    return chunks

def create_batches(paragraphs: List[str]) -> List[str]:
    """Create batches from paragraphs - returns list of batch texts."""
    if not paragraphs:
        return []
    
    batches = []
    current_batch = []
    current_batch_chars = 0
    
    for paragraph in paragraphs:
        # Split paragraph if it's too long
        if len(paragraph) > MAX_CHARS_PER_BATCH:
            chunks = split_long_paragraph(paragraph)
        else:
            chunks = [paragraph]
        
        for chunk in chunks:
            chunk_len = len(chunk)
            
            # Determine if we need a new batch
            need_new_batch = False
            
            if current_batch:
                if current_batch_chars + chunk_len > MAX_CHARS_PER_BATCH:
                    need_new_batch = True
                elif len(current_batch) >= MAX_PARAGRAPHS_PER_BATCH:
                    need_new_batch = True
            
            if need_new_batch:
                # Join current batch and add to batches
                batch_text = '\n\n'.join(current_batch)
                batches.append(batch_text)
                current_batch = [chunk]
                current_batch_chars = chunk_len
            else:
                current_batch.append(chunk)
                current_batch_chars += chunk_len
    
    # Add the last batch
    if current_batch:
        batch_text = '\n\n'.join(current_batch)
        batches.append(batch_text)
    
    return batches

def save_batches(batches: List[str], output_path: str):
    """Save batches with simple separators."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, batch_text in enumerate(batches, 1):
            f.write(batch_text)
            if i < len(batches):  # Not the last batch
                f.write("\n\n---\n\n")  # Simple separator

def main():
    """Main function."""
    try:
        # Check input file
        input_path = Path(INPUT_FILE)
        if not input_path.exists():
            print(f"❌ Error: Input file '{INPUT_FILE}' not found!")
            print(f"   Current directory: {Path.cwd()}")
            return False
        
        print(f"📖 Reading '{INPUT_FILE}'...")
        
        # Read file
        text = safe_read_file(INPUT_FILE)
        
        if not text.strip():
            print("❌ Error: File is empty")
            return False
        
        print(f"✓ File read ({len(text):,} chars)")
        
        # Split into paragraphs
        print("🔪 Splitting into paragraphs...")
        paragraphs = split_into_paragraphs(text)
        
        if not paragraphs:
            print("❌ Error: No paragraphs found")
            return False
        
        print(f"✓ Found {len(paragraphs)} paragraphs")
        
        # Count long paragraphs (those that will be split)
        long_paragraphs = sum(1 for p in paragraphs if len(p) > MAX_CHARS_PER_BATCH)
        if long_paragraphs > 0:
            print(f"   {long_paragraphs} paragraphs will be split (> {MAX_CHARS_PER_BATCH} chars)")
        
        # Create batches
        print("📦 Creating batches...")
        batches = create_batches(paragraphs)
        
        if not batches:
            print("❌ Error: No batches created")
            return False
        
        print(f"✓ Created {len(batches)} batches")
        
        # Save output
        print(f"\n💾 Saving to '{OUTPUT_FILE}'...")
        save_batches(batches, OUTPUT_FILE)
        
        # Show final stats
        total_chars = sum(len(batch) for batch in batches)
        avg_batch_chars = total_chars // len(batches) if batches else 0
        
        # Count how many batches are near the limit
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
        
        # Show sample of first batch
        if batches:
            print(f"\n📄 First batch (first 200 chars):")
            preview = batches[0][:200]
            print(f'"""\n{preview}...\n"""')
            print(f"\nSeparator between batches: '\\n\\n---\\n\\n'")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

# ===================== RUN =====================
if __name__ == "__main__":
    print()  # Blank line
    success = main()
    
    if success:
        print("\n✅ Done!")
    else:
        print("\n❌ Failed!")
