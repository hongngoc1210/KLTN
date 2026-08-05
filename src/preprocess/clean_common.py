
import re
import unicodedata
import pandas as pd
from underthesea import ner, word_tokenize


# ==========================================================
# STEP 1. Chuẩn hóa văn bản
# ==========================================================
def clean_text(text: str) -> str:
    if not text:
        return ""

    # Xóa BOM và các ký tự Unicode ẩn
    text = re.sub(r"[\ufeff\u200b\u200c\u200d\u2060\u00ad]", "", text)

    # Xóa ký tự điều khiển (giữ \n)
    text = re.sub(r"[\x00-\x09\x0B-\x1F\x7F]", " ", text)

    # NBSP
    text = text.replace("\xa0", " ")

    # Chuẩn hóa Unicode
    text = unicodedata.normalize("NFC", text)

    lines = []

    for line in text.split("\n"):
        line = re.sub(r"\s+", " ", line).strip()

        if line:
            lines.append(line)

    return "\n".join(lines)



# ==========================================================
# STEP 2. Detect broken encoding
# ==========================================================
BROKEN_RE = re.compile(
    r"[ÑñÖöÛûØøÙù]|"
    r"(?:aø|aù|aï|aû|aõ|"
    r"eø|eù|eï|eû|"
    r"iø|iù|"
    r"oø|où|oï|oû|"
    r"uø|uù|uï|"
    r"öô|öø|öù|öï|"
    r"ñö)"
)


def is_broken_line(line):
    return bool(BROKEN_RE.search(line))


def is_broken_document(text, threshold=0.9):

    lines = [x.strip() for x in text.split("\n") if x.strip()]

    broken = []

    for line in lines:
        if is_broken_line(line):
            broken.append(line)

    # print(f"Broken lines: {len(broken)}/{len(lines)}")

    # for x in broken[:20]:
    #     print(repr(x))

    return len(broken) / len(lines) >= threshold


# ==========================================================
# STEP 3. Remove English
# ==========================================================
def is_english(sentence, threshold=0.8):

    words = re.findall(
        r"[A-Za-zÀ-ỹ]+(?:[-'][A-Za-zÀ-ỹ]+)?",
        sentence
    )

    if not words:
        return False

    english = 0

    for w in words:

        if (
            unicodedata.normalize("NFD", w)
            .encode("ascii", "ignore")
            .decode()
            != w
        ):
            continue

        if re.fullmatch(
            r"[A-Za-z]+(?:[-'][A-Za-z]+)?",
            w
        ):
            english += 1

    return english / len(words) >= threshold



def remove_english_sentences(text):

    out = []

    for line in text.split("\n"):

        line = line.strip()

        if not line:
            continue

        if is_english(line):
            continue

        out.append(line)

    return "\n".join(out)



# ==========================================================
# STEP 4. Remove authors bằng NER
# ==========================================================
def is_author_line(line):

    s = line.strip()

    if not s:
        return False


    try:

        entities = ner(s)


        persons = [
            e
            for e in entities
            if len(e) >= 3
            and e[2] in ("B-PER", "I-PER")
        ]


        # phải có nhiều người
        if len(persons) >= 2:

            # có chỉ số tác giả
            if re.search(r"\d+\s*\*?", s):

                # dòng tác giả thường ngắn
                if len(s.split()) <= 50:
                    return True



        # fallback cho tên nước ngoài
        # Yukio Sato3, Susumu Takamatsu2
        if (
            s.count(",") >= 2
            and re.search(r"\d+", s)
            and len(s.split()) <= 50
        ):
            return True


    except Exception:
        pass


    return False



def remove_authors(lines):

    out = []

    for line in lines:

        if is_author_line(line):
            continue

        out.append(line.strip())

    return out




# ==========================================================
# STEP 5. Filter metadata
# ==========================================================
CATEGORY_PATTERNS = [
    r"^Khoa học Tự nhiên.*",
    r"^Khoa học Kỹ thuật.*",
    r"^Khoa học Nông nghiệp.*",
    r"^Khoa học Xã hội.*",
    r"^Khoa học Y.*",
]


def filter_lines(lines):

    out = []

    for s in lines:

        s = s.strip()

        if not s:
            continue


        if re.fullmatch(r"\d+", s):
            continue


        if re.fullmatch(
            r"\d+\(\d+\)\s+\d+\.\d{4}",
            s
        ):
            continue


        if any(
            re.match(p, s, re.I)
            for p in CATEGORY_PATTERNS
        ):
            continue


        patterns = [
            r"^Ngày nhận bài",
            r"^Received",
            r"^\*?Tác giả liên hệ",
            r"^Corresponding author",
            r"^Chỉ số phân loại",
            r"^Classification number",
            r"^DOI:",
            r"^\d+\s*(Khoa|Trường|Viện|Học viện)",
            r"^\d+\s*(Department|Faculty|School|Institute|University)",
        ]


        if any(
            re.match(p, s, re.I)
            for p in patterns
        ):
            continue


        out.append(s)


    return out



# ==========================================================
# STEP 6. References
# ==========================================================
def remove_references(lines):

    out = []

    for line in lines:

        if re.match(
            r"^TÀI LIỆU THAM KHẢO$",
            line,
            re.I
        ):
            break


        if re.match(
            r"^REFERENCES$",
            line,
            re.I
        ):
            break


        out.append(line)


    return out



# ==========================================================
# STEP 7. Merge OCR lines
# ==========================================================
def merge_lines(lines):

    text = ""

    for line in lines:


        if re.match(
            r"^\d+(\.\d+)*\.?\s+",
            line
        ):
            text += "\n\n" + line
            continue


        if not text:

            text = line
            continue


        if text.endswith(
            (".", "!", "?", ":")
        ):
            text += "\n\n" + line

        else:

            text += " " + line


    return text.strip()

# legal

def remove_header_legal(lines):
    LEGAL_DOC_TYPES = [
        "HIẾN PHÁP",
        "BỘ LUẬT",
        "LUẬT",
        "PHÁP LỆNH",
        "NGHỊ QUYẾT",
        "LỆNH",
        "QUYẾT ĐỊNH",
        "NGHỊ ĐỊNH",
        "THÔNG TƯ LIÊN TỊCH",
        "NGHỊ ĐỊNH LIÊN TỊCH",
        "THÔNG TƯ",
        "CHỈ THỊ",
    ]

    for i, line in enumerate(lines):
        line_norm = re.sub(r"\s+", " ", line).strip().upper()

        if any(line_norm.startswith(doc) for doc in LEGAL_DOC_TYPES):
            return lines[i:]

    return lines


# ==========================================================
# STEP. Remove OCR paragraphs
# ==========================================================

VN_WORD = re.compile(r"^[A-Za-zÀ-ỹ]+$")


def is_bad_ocr(paragraph,
               min_tokens=20,
               valid_ratio=0.6):

    tokens = word_tokenize(paragraph)

    words = [
        t.replace("_", "")
        for t in tokens
        if VN_WORD.fullmatch(t.replace("_", ""))
    ]

    if len(words) < min_tokens:
        return False

    valid = 0

    for w in words:

        if len(w) <= 1:
            continue

        if re.search(r"\d", w):
            continue

        valid += 1

    return valid / len(words) < valid_ratio


def has_bad_ocr(text):

    paragraphs = re.split(r"\n\s*\n", text)

    for para in paragraphs:

        if is_bad_ocr(para):
            return True

    return False

# ==========================================================
# STEP. Remove page numbers
# ==========================================================

def remove_page_numbers(lines):

    out = []

    for line in lines:

        if re.fullmatch(r"\s*\d+\s*", line):
            continue

        out.append(line)

    return out

# ==========================================================
# STEP. Remove separator lines
# ==========================================================

def remove_separator_lines(lines):

    out = []

    for line in lines:

        if re.fullmatch(r"[_\-=\.\*]{3,}", line.strip()):
            continue

        out.append(line)

    return out

from difflib import SequenceMatcher

def remove_duplicate_paragraphs(lines, threshold=0.98):

    paragraphs = []
    paragraph = []

    for line in lines:
        if line.strip():
            paragraph.append(line)
        else:
            if paragraph:
                paragraphs.append("\n".join(paragraph))
                paragraph = []

    if paragraph:
        paragraphs.append("\n".join(paragraph))

    kept = []

    for para in paragraphs:

        norm = re.sub(r"\s+", " ", para).strip()

        duplicated = False

        for old in kept:

            if SequenceMatcher(None, norm, old).ratio() >= threshold:
                duplicated = True
                break

        if not duplicated:
            kept.append(norm)

    return "\n\n".join(kept).split("\n")

# ==========================================================
# Pipeline
# ==========================================================
def clean_vjst(text):

    text = clean_text(text)


    if is_broken_document(text, 0.4):
        return None


    text = remove_english_sentences(text)


    lines = text.split("\n")


    # remove author trước
    lines = remove_authors(lines)


    # filter metadata
    lines = filter_lines(lines)


    # remove reference
    lines = remove_references(lines)


    return merge_lines(lines)

def clean_legal(text):
    text = clean_text(text)
    
    if is_broken_document(text):
        return 
    
    text = remove_english_sentences(text)
    
    lines = text.split("\n")
    
    # remove header
    lines = remove_header_legal(lines)
    
    lines = remove_duplicate_paragraphs(lines)

    return "\n".join(lines)



def clean_education(text):
    text = clean_text(text)
    
    text = remove_english_sentences(text)
    
    if is_bad_ocr(text):
        return 
    
    lines = text.split("\n")
    
    lines = remove_page_numbers(lines)
    lines = remove_separator_lines(lines)
    lines = remove_duplicate_paragraphs(lines)
    
    return "\n".join(lines)

def clean_news_finance(text):
    text = clean_text(text)

    text = remove_english_sentences(text)

    lines = text.split("\n")

    lines = remove_duplicate_paragraphs(lines)

    # Xóa dòng dưới 2 ký tự
    lines = [
        line for line in lines
        if len(line.strip()) >= 2
    ]

    return "\n".join(lines)