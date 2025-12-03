import pdfplumber
from pathlib import Path
from tqdm.auto import tqdm
import logging
import re

# Suppress noisy pdfminer warnings (e.g., FontBBox)
logging.getLogger("pdfminer").setLevel(logging.ERROR)


# ---------------------------------------------------
# SMART DE-DUPLICATION (fixes 哈哈利利波波 → 哈利波特)
# ---------------------------------------------------
def is_cjk(ch: str) -> bool:
    """Rudimentary CJK check to distinguish Chinese characters from digits/latin."""
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
        or 0x3400 <= code <= 0x4DBF  # CJK Unified Ideographs Extension A
        or 0x20000 <= code <= 0x2A6DF  # Extension B
        or 0x2A700 <= code <= 0x2B73F  # Extension C
        or 0x2B740 <= code <= 0x2B81F  # Extension D
        or 0x2B820 <= code <= 0x2CEAF  # Extension E
        or 0xF900 <= code <= 0xFAFF  # CJK Compatibility Ideographs
    )


def smart_remove_duplication(line, threshold=0.60):
    """
    Remove duplicated *CJK* characters ONLY when duplication is likely a PDF artifact.
    - Never touch digits (e.g. '11' in '第11章').
    - Never touch pure chapter headings like '第11章'.
    """
    if not line or len(line) <= 1:
        return line

    stripped = line.strip()

    # 🔒 Hard-protect chapter headings like '第11章'
    if re.fullmatch(r"第\d+章", stripped):
        return line

    # 1) Measure duplication ratio, but only for CJK chars
    duplicates = 0
    total_pairs = 0
    i = 0
    while i < len(line) - 1:
        c1, c2 = line[i], line[i + 1]
        if c1 == c2 and is_cjk(c1):
            duplicates += 1
        # Only count pairs that at least involve a CJK char for ratio purposes
        if is_cjk(c1) or is_cjk(c2):
            total_pairs += 1
        i += 2

    if total_pairs == 0:
        # No CJK pairs, nothing to dedup
        return line

    ratio = duplicates / total_pairs

    # If duplication ratio is low, treat as normal text
    if ratio < threshold:
        return line

    # 2) Actually remove duplicated CJK chars, but leave digits/letters alone
    result = []
    i = 0
    while i < len(line):
        if (
            i + 1 < len(line)
            and line[i] == line[i + 1]
            and is_cjk(line[i])  # only collapse CJK duplicates
        ):
            result.append(line[i])
            i += 2
        else:
            result.append(line[i])
            i += 1

    return "".join(result)


# ---------------------------------------------------
# FIX BROKEN CHAPTER NUMBERS (1 + 第 章 → 第1章)
# ---------------------------------------------------
def fix_chapter_numbers(lines):
    """
    Repairs lines like:
        '1'
        '第  章'
    into:
        '第1章'
    """
    fixed = []
    i = 0

    while i < len(lines):
        current = lines[i].strip()

        # If current line is a pure number (chapter index)
        if re.fullmatch(r"\d+", current) and i + 1 < len(lines):
            next_line = lines[i + 1].replace(" ", "").strip()

            # Pattern for missing middle part: "第章"
            if next_line == "第章":
                merged = f"第{current}章"
                fixed.append(merged)
                i += 2
                continue

        # Otherwise keep original
        fixed.append(lines[i])
        i += 1

    return fixed


# ---------------------------------------------------
# CLEAN LINE BREAKS
# ---------------------------------------------------
# Sentence-ending punctuation marks
END_PUNCT = set("。！？，、；：…—?!.,;:」』）)]>”)")

# Regex for "第1章", "第一章", "第十章", etc.
CHAPTER_RE = re.compile(r"^第[一二三四五六七八九十百千0-9]+章")

# Max length (characters) to consider a line as a chapter subtitle / heading
SUBTITLE_MAX_LEN = 30


def should_merge(line: str) -> bool:
    """
    Return True if this line should be merged with the next line.
    Only used for body text (not titles/headings).
    """
    if not line.strip():
        return False
    last_char = line.rstrip()[-1]
    return last_char not in END_PUNCT


def find_first_chapter_header(lines):
    """
    Find the first chapter header block at the beginning (e.g.:
        第1章
        大难不死的男孩
        THE BOY WHO LIVED
    and return (start_index, end_index) of that header block.

    If no chapter is found, returns (None, None).
    """
    n = len(lines)
    for i, raw in enumerate(lines):
        line = raw.strip()
        if CHAPTER_RE.match(line):
            # Include this line + up to two short subtitle lines after it
            j = i
            used_subtitles = 0
            j += 1
            while j < n and used_subtitles < 2:
                s = lines[j].strip()
                if not s:
                    break
                if len(s) <= SUBTITLE_MAX_LEN:
                    used_subtitles += 1
                    j += 1
                else:
                    break
            return i, j - 1
    return None, None


def _clean_line_breaks(lines):
    # Normalize: strip only trailing "\n" here, keep empty lines as ""
    lines = [ln.rstrip("\n") for ln in lines]

    # 1. Detect first chapter header block at the beginning of the book
    first_ch_start, first_ch_end = find_first_chapter_header(lines)

    processed = []

    # If we found a chapter header and it is near the top, treat everything
    # up to the end of that header as "front matter" (title, acknowledgements,
    # foreword, chapter heading), preserving line breaks exactly.
    if first_ch_start is not None:
        header_end = first_ch_end
        # Everything before this header is also front matter
        header_start = 0
        # Copy front matter as-is
        for i in range(header_start, header_end + 1):
            processed.append(lines[i])
        body_start = header_end + 1
    else:
        # No chapter header detected; treat entire file as body
        body_start = 0

    buffer = ""

    i = body_start
    n = len(lines)

    while i < n:
        line = lines[i]

        # 2. Handle chapter headers anywhere in the body (for later chapters)
        stripped = line.strip()
        if CHAPTER_RE.match(stripped):
            # Flush any buffered paragraph before the chapter title
            if buffer:
                processed.append(buffer)
                buffer = ""

            # Output the chapter line itself
            processed.append(line)

            # Also treat following short lines as subtitles/headings
            i += 1
            used_subtitles = 0
            while i < n and used_subtitles < 2:
                nxt = lines[i]
                s = nxt.strip()
                if not s:
                    # Stop subtitles on blank line (but still output it)
                    processed.append(nxt)
                    i += 1
                    break
                if len(s) <= SUBTITLE_MAX_LEN:
                    processed.append(nxt)
                    used_subtitles += 1
                    i += 1
                else:
                    break
            continue

        # 3. Normal body handling with line merging

        # Blank line: terminate current paragraph
        if not stripped:
            if buffer:
                processed.append(buffer)
                buffer = ""
            processed.append("")  # keep blank line
            i += 1
            continue

        if not buffer:
            # Start a new paragraph
            buffer = line
        else:
            if should_merge(buffer):
                # merge lines without introducing a newline
                buffer += line
            else:
                # finish previous paragraph and start a new one
                processed.append(buffer)
                buffer = line

        i += 1

    # Flush last paragraph
    if buffer:
        processed.append(buffer)

    # Join with "\n"
    return "\n".join(processed)


def clean_line_breaks(txt_path: str):
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    result = _clean_line_breaks(lines)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(result)

    print("Done! Output written to", txt_path)


# ---------------------------------------------------
# SPLIT BY CHAPTERS
# ---------------------------------------------------
def split_book_by_chapters(txt_path: str) -> None:
    """
    Split a Chinese novel text file into title + chapter files.

    Rules:
    - Everything before the first occurrence of a line matching r'^第(\d+)章' -> title.txt
    - Each chapter (starting at “第N章”) goes to chN.txt
    - Output folder = same name as input file (no extension), located in the same directory
    """

    input_path = Path(txt_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input text file not found: {txt_path}")

    # Read file safely (UTF-8 recommended)
    try:
        text = input_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # fallback for files in GBK/GB2312 etc.
        text = input_path.read_text(encoding="utf-8", errors="ignore")

    # Keep line endings exactly as original
    lines = text.splitlines(keepends=True)

    # Chapter header pattern
    # Ensures correct matching even if extra spaces appear
    chap_re = re.compile(r"^第\s*(\d+)\s*章")

    title_lines = []
    chapters = []  # (chapter_number, content_string)
    current_chap_num = None
    current_content = []

    in_chapter = False

    for line in lines:
        stripped = line.strip()

        match = chap_re.match(stripped)

        if match:
            # New chapter detected
            chap_num = match.group(1)

            if in_chapter:
                # Save previous chapter content
                chapters.append((current_chap_num, "".join(current_content)))
                current_content = []

            # Begin new chapter
            in_chapter = True
            current_chap_num = chap_num
            current_content.append(line)

        else:
            if in_chapter:
                current_content.append(line)
            else:
                title_lines.append(line)

    # Add last chapter if file ended cleanly after a chapter
    if in_chapter and current_content:
        chapters.append((current_chap_num, "".join(current_content)))

    # Create output directory
    out_dir = input_path.with_suffix("")  # remove .txt to form folder
    out_dir.mkdir(exist_ok=True)

    # Write title.txt if exists
    if title_lines:
        (out_dir / "title.txt").write_text("".join(title_lines), encoding="utf-8")

    # Write each chapter file: ch1.txt, ch2.txt, ...
    for chap_num, content in chapters:
        ch_path = out_dir / f"ch{chap_num}.txt"
        ch_path.write_text(content, encoding="utf-8")

# ---------------------------------------------------
# MAIN EXTRACTION FUNCTION
# ---------------------------------------------------
def pdf_to_txt(pdf_path):
    pdf_path = Path(pdf_path)
    txt_path = pdf_path.with_suffix(".txt")

    with pdfplumber.open(pdf_path) as pdf, open(txt_path, "w", encoding="utf-8") as f:
        total_pages = len(pdf.pages)

        for page in tqdm(pdf.pages, total=total_pages, desc="Extracting pages"):
            raw_text = page.extract_text() or ""
            lines = raw_text.split("\n")

            # Step 1: smart de-duplication
            lines = [smart_remove_duplication(line) for line in lines]

            # Step 2: fix chapter numbers
            lines = fix_chapter_numbers(lines)

            # Write out
            f.write("\n".join(lines) + "\n")

    clean_line_breaks(txt_path=txt_path)
    print("txt file saved to:", txt_path)
    split_book_by_chapters(txt_path=txt_path)





if __name__ == "__main__":
    for i in range(1, 8):
        p = f"/home/bo/workspace/whisper/tasks/HP1/text_zh/{i}.pdf"
        pdf_to_txt(p)
