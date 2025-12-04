import os
import re
import sys

def fix_leading_letter_space(text: str) -> str:
    """
    Fix cases like:
        'M r. and Mrs.' -> 'Mr. and Mrs.'
        'H arry woke'   -> 'Harry woke'
    Only at the start of lines.
    """
    pattern = re.compile(r'^([A-Za-z])\s+([a-z])')

    fixed_lines = []
    for line in text.splitlines(keepends=True):
        fixed_lines.append(pattern.sub(r'\1\2', line))
    return ''.join(fixed_lines)


def split_chapters(file_path: str) -> None:
    # Read entire file
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Robust chapter header matcher:
    # Matches:
    #   - CHAPTER ONE
    #   - CHAPTER TWENTY NINE
    #   - CHAPTER 19
    #   - CHAPTER NINETEEN - The Lion and the Serpent
    #   - Chapter 3: Spinner’s End
    #   - Chapter 4 - Will And Won’t
    chapter_pattern = re.compile(
        r'^(?P<title>('
        r'CHAPTER\s+(?:[A-Z]+|\d+)(?:\s+[A-Z]+)*'              # CHAPTER NINETEEN / CHAPTER 5 / CHAPTER TWENTY FOUR
        r'(?:\s*[\-–—:]\s*.*)?'                                # optional dash + title
        r'|'
        r'Chapter\s+\d+\s*[\-–—:]\s*.*'                        # Chapter 2: Spinner’s End
        r'))\s*$',
        re.MULTILINE
    )

    matches = list(chapter_pattern.finditer(text))
    if not matches:
        print("No chapter headings found. Check patterns.")
        return

    base_dir = os.path.dirname(os.path.abspath(file_path))
    out_dir = os.path.join(base_dir, "chapters")
    os.makedirs(out_dir, exist_ok=True)

    for idx, match in enumerate(matches):
        chapter_number = idx + 1
        start = match.start()

        # End of chapter is start of next match, or end of text
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)

        chapter_text = text[start:end].lstrip("\n")

        # Fix “M r.” → “Mr.” types of spacing errors
        chapter_text = fix_leading_letter_space(chapter_text)

        out_path = os.path.join(out_dir, f"ch{chapter_number}.txt")
        with open(out_path, "w", encoding="utf-8") as out_f:
            out_f.write(chapter_text)

        print(f"Saved: {out_path}")





if __name__ == "__main__":
    split_chapters("/home/bo/workspace/transcribe_and_align/data/HP1/text_en/01 Harry Potter and the Philosophers Stone.txt")