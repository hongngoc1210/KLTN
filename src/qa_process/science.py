# -*- coding: utf-8 -*-
"""
Chunk văn bản (context QA) cho bài báo khoa học tiếng Việt, phục vụ bài toán
hallucination detection. Chia theo cấu trúc IMRAD của bài báo: Tóm tắt / Từ
khóa / Đặt vấn đề / Phương pháp / Kết quả / Thảo luận / Kết luận / Tài liệu
tham khảo.

Quy tắc:
  - Mỗi chunk nên nằm trong khoảng [min_words, max_words] từ (mặc định 30-128).
  - Không cắt ngang 1 "phần" (Tóm tắt/Đặt vấn đề/.../Kết luận) nếu bản thân
    nó không thể chia nhỏ hơn nữa (VD 1 đoạn liền mạch) -> nếu đoạn đó >
    max_words thì vẫn giữ nguyên làm 1 chunk (chấp nhận vượt ngưỡng trên).
  - Bảng biểu ("Bảng N. ...") và chú thích hình ("Hình N. ...") bị loại bỏ
    khỏi văn bản trước khi chunk, vì đây là nội dung phi văn bản/rời rạc,
    không phù hợp cho QA dạng văn xuôi.
  - Các đơn vị nội dung liền mạch, có liên hệ chặt với nhau (VD 1 danh sách
    các bước "- Bước 1:, - Bước 2:, ..."): nếu phần còn lại cuối danh sách
    chỉ còn 1-2 đơn vị ngắn thì được GỘP vào chunk liền trước (đã có sẵn
    3-4 đơn vị) thay vì tách ra thành 1 chunk riêng, kể cả khi việc gộp
    khiến chunk vượt quá max_words (chấp nhận vượt ngưỡng trong giới hạn
    tail_merge_factor).
  - Nếu 1 chunk sau khi ghép < min_words từ thì gộp với chunk liền kề
    (ưu tiên gộp lùi vào chunk trước, nếu là chunk đầu/cuối thì gộp với
    chunk còn lại duy nhất).
"""
import re
import json


class SciPaperChunker:
    """Chunk các bài báo khoa học tiếng Việt theo cấu trúc IMRAD.

    Có thể tái sử dụng: khởi tạo 1 lần với các ngưỡng mong muốn, rồi gọi
    chunk_document()/chunk_records()/process_file() nhiều lần.
    """

    # Tách câu nhỏ trong nội bộ 1 phần khi nó quá dài
    SENT_SPLIT_RE = re.compile(r'(?<=[\.;])\s+(?=[A-ZÀ-Ỹ0-9])')

    # Chuỗi dấu chấm/gạch dưới/gạch ngang/phẩy/hai chấm lặp lại nhiều lần
    # (VD đường viền bảng, chỗ trống để điền...) -> chuẩn hoá về 1 token
    # duy nhất để không bị đếm nhầm thành hàng nghìn "từ".
    FILLER_RE = re.compile(r'(?:[\.\_\-\,\:]\s*){5,}')

    # Chú thích bảng/hình: chiếm trọn 1 đoạn/dòng riêng trong văn bản trích
    # xuất từ PDF (kết thúc bằng \n hoặc \n\n).
    TABLE_CAPTION_RE = re.compile(r'Bảng\s*\d+\.[^\n]*')
    FIGURE_CAPTION_RE = re.compile(r'Hình\s*\d+\.[^\n]*')

    WORD_RE = re.compile(r'\S+')
    EXCESS_NEWLINES_RE = re.compile(r'\n{3,}')
    EXCESS_SPACES_RE = re.compile(r'[ \t]{2,}')

    # Đơn vị dạng danh sách bước / gạch đầu dòng - dùng để nhận diện các
    # đơn vị "cùng nhóm" khi quyết định có nên gộp phần đuôi hay không.
    LIST_ITEM_RE = re.compile(
        r'^\s*(?:[-•*]\s*)?(?:Bước\s*\d+|[0-9]+[\.\)])', re.IGNORECASE)

    # Ranh giới các phần IMRAD. Do văn bản trích xuất từ PDF thường dính
    # liền tiêu đề phần với câu đầu tiên của nội dung (không xuống dòng
    # riêng), không thể dùng regex "đầu dòng" -> tìm theo LẦN XUẤT HIỆN ĐẦU
    # TIÊN của mỗi tên phần, đúng theo thứ tự IMRAD.
    PAPER_SECTION_PATTERNS = [
        ('tom_tat', re.compile(r'\bTóm tắt\b')),
        ('tu_khoa', re.compile(r'\bTừ kh(?:óa|oá)\b')),
        ('dat_van_de', re.compile(r'\b(?:Đặt vấn đề|Mở đầu|Giới thiệu|Tổng quan)\b')),
        ('phuong_phap', re.compile(
            r'\b(?:Đối tượng và phương pháp(?: nghiên cứu)?|'
            r'Vật liệu và phương pháp(?: nghiên cứu)?|'
            r'Phương pháp nghiên cứu|Phương pháp)\b')),
        ('ket_qua', re.compile(
            r'\bKết quả(?: nghiên cứu)?(?: và (?:thảo luận|bàn luận))?\b')),
        ('thao_luan', re.compile(r'\b(?:Thảo luận|Bàn luận)\b')),
        ('ket_luan', re.compile(r'\bKết luận\b')),
        ('tai_lieu_tk', re.compile(r'\bTài liệu tham khảo\b')),
    ]

    def __init__(self, min_words=30, max_words=128,
                 tail_merge_factor=1.6, tail_merge_ratio=0.5,
                 table_min_digit_tokens=3, table_prose_run=6,
                 table_resume_window=700):
        """
        min_words / max_words: ngưỡng độ dài chunk mong muốn.
        tail_merge_factor: chunk đuôi được gộp có thể vượt max_words tối đa
            tail_merge_factor lần (VD 1.6 -> tối đa ~1.6*max_words từ).
        tail_merge_ratio: 1 chunk được coi là "đuôi nhỏ, nên gộp" nếu độ
            dài của nó < tail_merge_ratio * max_words.
        table_*: tham số cho thuật toán heuristic nhận diện phần dữ liệu
            bảng (xem _find_table_resume).
        """
        self.min_words = min_words
        self.max_words = max_words
        self.tail_merge_factor = tail_merge_factor
        self.tail_merge_ratio = tail_merge_ratio
        self.table_min_digit_tokens = table_min_digit_tokens
        self.table_prose_run = table_prose_run
        self.table_resume_window = table_resume_window

    # ------------------------------------------------------------------
    # Tiền xử lý văn bản
    # ------------------------------------------------------------------
    def normalize_text(self, text):
        text = self.FILLER_RE.sub(' [...] ', text)
        text = self.EXCESS_NEWLINES_RE.sub('\n\n', text)
        text = self.EXCESS_SPACES_RE.sub(' ', text)
        return text

    def word_count(self, text):
        return len(text.split())

    def _find_table_resume(self, text, start):
        """Tìm vị trí văn bản "hồi phục" trở lại văn xuôi bình thường sau 1
        chú thích bảng, dựa trên heuristic: dữ liệu bảng chứa nhiều token
        có chữ số/%, còn văn xuôi là 1 chuỗi từ liên tục không có chữ số.
        Trả về None nếu không tìm thấy dữ liệu bảng thực sự (VD chú thích
        không có dữ liệu đi kèm, hoặc bảng đã bị cắt/không trích xuất
        được) - khi đó chỉ nên xoá dòng chú thích, giữ nguyên phần sau.
        """
        sub = text[start:start + self.table_resume_window]
        words = list(self.WORD_RE.finditer(sub))
        consumed_digits = 0
        prose_run = 0
        for i, w in enumerate(words):
            tok = w.group()
            has_digit = bool(re.search(r'\d', tok))
            alpha_len = len(re.sub(r'[^\wÀ-ỹ]', '', tok, flags=re.UNICODE))
            is_prose = (not has_digit) and tok != '%' and alpha_len >= 2
            if has_digit or tok == '%':
                consumed_digits += 1
                prose_run = 0
            elif is_prose:
                prose_run += 1
                if (consumed_digits >= self.table_min_digit_tokens
                        and prose_run >= self.table_prose_run):
                    resume_word = words[i - (self.table_prose_run - 1)]
                    return start + resume_word.start()
            else:
                prose_run = 0
        return None

    def remove_tables_and_figures(self, text):
        """Xoá toàn bộ chú thích hình ("Hình N. ...") và bảng biểu ("Bảng
        N. ..." kèm dữ liệu bảng đi sau nó) khỏi văn bản."""
        # 1) Chú thích hình: luôn chỉ là 1 dòng chú thích, không có dữ liệu
        #    theo sau -> xoá thẳng.
        text = self.FIGURE_CAPTION_RE.sub('', text)

        # 2) Bảng: xoá chú thích + phần dữ liệu bảng đi kèm (nếu có).
        pos = 0
        while True:
            m = self.TABLE_CAPTION_RE.search(text, pos)
            if not m:
                break
            resume = self._find_table_resume(text, m.end())
            cut_end = resume if resume is not None else m.end()
            text = text[:m.start()] + text[cut_end:]
            pos = m.start()

        return self.normalize_text(text)

    # ------------------------------------------------------------------
    # Chia theo cấu trúc IMRAD
    # ------------------------------------------------------------------
    def split_into_paper_sections(self, text):
        """Tìm ranh giới các phần IMRAD theo lần xuất hiện đầu tiên của mỗi
        tên phần, rồi cắt văn bản tại các vị trí đó. Mỗi phần thu được là 1
        đơn vị KHÔNG được phép cắt rời tuỳ tiện."""
        cursor = 0
        boundaries = []
        for _, pat in self.PAPER_SECTION_PATTERNS:
            m = pat.search(text, cursor)
            if m:
                boundaries.append(m.start())
                cursor = m.end()
        boundaries = sorted(set(boundaries))
        if not boundaries:
            return [text]
        if boundaries[0] != 0:
            boundaries = [0] + boundaries

        blocks = []
        for i, b in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
            block = text[b:end].strip()
            if block:
                blocks.append(block)
        return blocks

    def get_atomic_units(self, text):
        """Chia 1 khối văn bản thành các đơn vị nhỏ nhất có thể (câu/dòng),
        thử lần lượt các cấp độ chia mịn dần. Nếu không còn cách chia nào
        hợp lý, trả về nguyên khối - đây là đơn vị KHÔNG THỂ chia nhỏ hơn."""
        # cấp 1: theo câu
        sentences = self.SENT_SPLIT_RE.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) > 1:
            return sentences

        # cấp 2: theo dòng (VD danh sách các bước, mỗi dòng 1 đơn vị)
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if len(lines) > 1:
            return lines

        # không chia được nữa -> nguyên khối là 1 đơn vị nguyên tử
        return [text]

    def _is_list_item(self, unit):
        return bool(self.LIST_ITEM_RE.match(unit))

    def _tail_merge(self, chunks):
        """Gộp các chunk "đuôi nhỏ" (VD chỉ còn 1-2 bước ngắn) vào chunk
        liền trước thay vì để đứng riêng, kể cả khi vượt max_words - miễn
        không vượt quá max_words * tail_merge_factor. Áp dụng lặp từ cuối
        danh sách chunk trở lên."""
        if len(chunks) < 2:
            return chunks

        small_threshold = self.max_words * self.tail_merge_ratio
        hard_cap = self.max_words * self.tail_merge_factor

        merged = list(chunks)
        i = len(merged) - 1
        while i > 0:
            tail_wc = self.word_count(merged[i])
            prev_wc = self.word_count(merged[i - 1])
            if tail_wc < small_threshold and (prev_wc + tail_wc) <= hard_cap:
                merged[i - 1] = merged[i - 1] + '\n' + merged[i]
                del merged[i]
            i -= 1
        return merged

    def split_long_section(self, section_text):
        """Nếu 1 phần dài hơn max_words, chia nhỏ theo câu/dòng rồi ghép
        (greedy) các đơn vị nguyên tử lại thành chunk trong ngưỡng. Sau đó
        gộp lại các chunk "đuôi nhỏ" thuộc cùng 1 nhóm nội dung liền mạch
        (VD danh sách các bước) để tránh tách rời không cần thiết."""
        if self.word_count(section_text) <= self.max_words:
            return [section_text]

        units = self.get_atomic_units(section_text)
        if len(units) <= 1:
            # thực sự không thể chia nhỏ hơn -> giữ nguyên dù vượt max_words
            return [section_text]

        chunks, buf, buf_wc = [], [], 0
        for unit in units:
            wc = self.word_count(unit)
            if wc > self.max_words:
                if buf:
                    chunks.append('\n'.join(buf))
                    buf, buf_wc = [], 0
                chunks.extend(self.split_long_section(unit))
                continue
            if buf and buf_wc + wc > self.max_words:
                chunks.append('\n'.join(buf))
                buf, buf_wc = [], 0
            buf.append(unit)
            buf_wc += wc
        if buf:
            chunks.append('\n'.join(buf))

        return self._tail_merge(chunks)

    def merge_small_chunks(self, chunks):
        """Gộp các chunk < min_words từ vào chunk liền kề (ưu tiên chunk
        trước)."""
        if not chunks:
            return chunks
        merged = [chunks[0]]
        for c in chunks[1:]:
            if self.word_count(merged[-1]) < self.min_words:
                merged[-1] = merged[-1] + '\n' + c
            else:
                merged.append(c)
        # chunk cuối cùng có thể vẫn < min_words -> gộp ngược vào chunk trước
        if len(merged) > 1 and self.word_count(merged[-1]) < self.min_words:
            merged[-2] = merged[-2] + '\n' + merged[-1]
            merged = merged[:-1]
        # chunk đầu tiên (không có chunk trước để gộp trong vòng lặp trên)
        if len(merged) > 1 and self.word_count(merged[0]) < self.min_words:
            merged[1] = merged[0] + '\n' + merged[1]
            merged = merged[1:]
        return merged

    # ------------------------------------------------------------------
    # API chính
    # ------------------------------------------------------------------
    def chunk_document(self, text):
        text = self.remove_tables_and_figures(text)
        section_blocks = self.split_into_paper_sections(text)
        raw_chunks = []
        for section in section_blocks:
            raw_chunks.extend(self.split_long_section(section))
        return self.merge_small_chunks(raw_chunks)

    def chunk_records(self, records):
        """records: list[{'id':..., 'content':..., 'domain':...}]
        Trả về list các chunk kèm id nguồn + chỉ số chunk."""
        out = []
        for rec in records:
            chunks = self.chunk_document(rec['content'])
            for i, ch in enumerate(chunks):
                out.append({
                    'chunk_id': f"{rec['id']}_chunk{i}",
                    'doc_id': rec['id'],
                    'domain': rec.get('domain'),
                    'chunk_index': i,
                    'n_chunks': len(chunks),
                    'word_count': self.word_count(ch),
                    'content': ch,
                })
        return out

    def process_file(self, in_path, out_path=None, verbose=True):
        """Đọc file JSON records, chunk toàn bộ, in thống kê, và lưu ra
        out_path (nếu có). Trả về danh sách chunk."""
        with open(in_path, encoding='utf-8') as f:
            data = json.load(f)

        chunks = self.chunk_records(data)

        if verbose:
            wcs = [c['word_count'] for c in chunks]
            under = sum(1 for w in wcs if w < self.min_words)
            over = sum(1 for w in wcs if w > self.max_words)
            print(f"Tổng số văn bản: {len(data)}")
            print(f"Tổng số chunk:   {len(chunks)}")
            print(f"Trung bình chunk/văn bản: {len(chunks) / len(data):.2f}")
            print(f"Độ dài chunk (từ) min/max/avg: "
                  f"{min(wcs)}/{max(wcs)}/{sum(wcs) / len(wcs):.1f}")
            print(f"Số chunk < {self.min_words} từ: {under}")
            print(f"Số chunk > {self.max_words} từ (không thể/không nên "
                  f"chia nhỏ hơn): {over}")

        if out_path:
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)

        return chunks


if __name__ == '__main__':
    import sys
    in_path = sys.argv[1] if len(sys.argv) > 1 else 'data/science_clean.json'
    out_path = sys.argv[2] if len(sys.argv) > 2 else 'data/chunking/science_chunks.json'

    chunker = SciPaperChunker(min_words=30, max_words=128)
    chunker.process_file(in_path, out_path)