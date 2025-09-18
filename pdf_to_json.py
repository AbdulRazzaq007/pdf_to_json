#!/usr/bin/env python3
"""
pdf_to_json.py

Robust PDF -> structured JSON extractor.
- Paragraphs, Tables, Charts (with OCR fallback)
- Preserves page-level hierarchy
- Maps tables/charts to nearest heading
- Cleans footers, filters junk, fixes glued numbers

Usage:
    python pdf_to_json.py input.pdf [output.json]
"""

import os
import sys
import re
import json
import gc
import logging
import atexit
from statistics import median
from typing import List, Dict, Any, Optional

import pdfplumber
import fitz  # PyMuPDF (used only when helpful)
from PIL import Image
import pytesseract

# Try to import camelot (may fail on some environments)
try:
    import camelot
    _HAS_CAMELOT = True
except Exception:
    camelot = None
    _HAS_CAMELOT = False

# Setup
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
gc.collect()
atexit.register(lambda: None)

# --------- Config / Regex ----------
NUM_RE = re.compile(r'[-+]?\d{1,3}(?:[,\d]*)(?:\.\d+)?%?')  # matches numbers like 1,234.56, 10%, 12.3
FOOTER_PATTERNS = [
    r"mutual fund investments are subject to market risks",
    r"page\s*\|\s*\d+",
    r"page\s*\d+",
    r"all rights reserved",
    r"copyright",
]
AXIS_LONG_RE = re.compile(r'^(?:FY\d{2}\b[\s\-]*){4,}', re.IGNORECASE)
JSON_LIKE_STARTS = ('{', '}', '[', ']', '"')

# thresholds
CHAR_GAP_WORD_THRESHOLD_RATIO = 0.6  # if gap > ratio * avg_font_size => insert space
PARAGRAPH_VERTICAL_GAP = 10  # pts, vertical gap to break paragraphs
NUMERIC_LINE_GROUP_GAP = 18  # pts, gap to split numeric table groups

# --------- Utilities ----------
def safe_strip(s: Optional[str]) -> str:
    return (s or "").strip()

def should_skip_line(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    low = t.lower()
    for p in FOOTER_PATTERNS:
        if re.search(p, low):
            return True
    if t.startswith(JSON_LIKE_STARTS):
        return True
    if AXIS_LONG_RE.match(t.replace(',', ' ')):
        return True
    # very small garbage
    if len(t) <= 1:
        return True
    return False

def is_numeric_line(text: str) -> bool:
    """Return True if line likely contains tabular numeric data (>=2 numeric tokens)."""
    tokens = NUM_RE.findall(text)
    return len(tokens) >= 2

def label_and_numbers(text: str) -> List[str]:
    """Return [label, num1, num2, ...] or just numeric list if no label found."""
    nums = NUM_RE.findall(text)
    if not nums:
        return []
    first_match = NUM_RE.search(text)
    if first_match:
        label = text[:first_match.start()].strip()
        if label:
            return [label] + nums
    return nums

def unique_output_filename(path: str) -> str:
    """If path exists, return path_1, path_2, ..."""
    base, ext = os.path.splitext(path)
    if not os.path.exists(path):
        return path
    counter = 1
    while True:
        candidate = f"{base}_{counter}{ext}"
        if not os.path.exists(candidate):
            return candidate
        counter += 1

# --------- Text line building with spacing fix ----------
def group_chars_into_lines_with_spacing(page) -> List[Dict[str, Any]]:
    """
    Use page.chars (pdfplumber) to build lines and insert spaces when horizontal gap
    between adjacent characters is large relative to font size.
    Returns list of {'text','top','avg_size','bottom'}.
    """
    # Fallback: if no char-level info, use extract_text splitlines
    if not getattr(page, "chars", None):
        txt = page.extract_text() or ""
        lines = []
        for i, ln in enumerate(txt.splitlines()):
            if not ln.strip():
                continue
            lines.append({"text": ln.rstrip(), "top": i * 12.0, "avg_size": 10.0, "bottom": i * 12.0 + 10.0})
        return lines

    groups = {}
    for ch in page.chars:
        top = round(ch.get("top", 0.0), 1)
        groups.setdefault(top, []).append(ch)

    line_objs = []
    for top in sorted(groups.keys()):
        chars = sorted(groups[top], key=lambda c: c.get("x0", 0))
        if not chars:
            continue
        sizes = [c.get("size", 0.0) for c in chars if c.get("size", 0.0)]
        avg_size = float(sum(sizes)) / len(sizes) if sizes else 10.0
        pieces = []
        prev_x1 = None
        for ch in chars:
            x0 = ch.get("x0", 0.0)
            x1 = ch.get("x1", 0.0)
            txt = ch.get("text", "")
            if prev_x1 is None:
                pieces.append(txt)
            else:
                gap = x0 - prev_x1
                if gap > max(CHAR_GAP_WORD_THRESHOLD_RATIO * avg_size, 0.6):
                    # treat as space boundary
                    pieces.append(" " + txt)
                else:
                    pieces.append(txt)
            prev_x1 = x1
        line_text = "".join(pieces).strip()
        heights = [c.get("height", avg_size * 1.2) for c in chars]
        height = max(heights) if heights else avg_size * 1.2
        line_objs.append({"text": line_text, "top": top, "avg_size": avg_size, "bottom": top + height})
    return line_objs

# --------- Paragraphs, headings, numeric lines detection ----------
def detect_paragraphs_headings_numeric(page) -> (List[Dict], List[Dict], List[Dict]):
    """
    Analyze a page and return:
      paragraphs: [{'type':'paragraph','section':..,'sub_section':..,'text':..,'top':..},...]
      headings: [{'text':..,'top':..}, ...]
      numeric_lines: [{'text':..,'top':..,'parsed':[...]}, ...]
    """
    lines = group_chars_into_lines_with_spacing(page)
    if not lines:
        return [], [], []

    sizes = [l["avg_size"] for l in lines if l.get("avg_size")]
    median_size = median(sizes) if sizes else 10.0
    heading_thresh = max(2.0, median_size * 0.2)

    paragraphs = []
    headings = []
    numeric_lines = []

    current_section = None
    current_sub = None
    buffer = []
    buffer_top = None
    last_bottom = None

    for ln in lines:
        text = safe_text = (ln.get("text") or "").strip()
        if should_skip_line(text):
            continue

        # heading detection heuristics
        is_allcaps_short = text.isupper() and len(text.split()) <= 8
        is_big_font = ln.get("avg_size", 0) >= median_size + heading_thresh
        is_number_heading = bool(re.match(r'^\s*\d+(\.\d+)*\s+[A-Za-z]', text))

        # sub-section detection: title case short line
        is_subsection = text.istitle() and len(text.split()) <= 8

        if is_big_font or is_allcaps_short or is_number_heading:
            # flush paragraph buffer
            if buffer:
                paragraphs.append({"type": "paragraph", "section": current_section,
                                   "sub_section": current_sub, "text": " ".join(buffer), "top": buffer_top})
                buffer = []
            # register heading
            headings.append({"text": text, "top": ln["top"]})
            current_section = text
            current_sub = None
            last_bottom = ln["bottom"]
            continue

        if is_subsection:
            current_sub = text
            continue

        # numeric table row candidate
        if is_numeric_line(text):
            parsed = label_and_numbers(text)
            numeric_lines.append({"text": text, "top": ln["top"], "parsed": parsed})
            last_bottom = ln["bottom"]
            continue

        # paragraph grouping by vertical gap
        if not buffer:
            buffer = [text]
            buffer_top = ln["top"]
            last_bottom = ln["bottom"]
        else:
            gap = ln["top"] - (last_bottom or ln["top"])
            if gap <= max(PARAGRAPH_VERTICAL_GAP, ln.get("avg_size", 10) * 0.7):
                buffer.append(text)
                last_bottom = ln["bottom"]
            else:
                paragraphs.append({"type": "paragraph", "section": current_section,
                                   "sub_section": current_sub, "text": " ".join(buffer), "top": buffer_top})
                buffer = [text]
                buffer_top = ln["top"]
                last_bottom = ln["bottom"]

    # flush final buffer
    if buffer:
        paragraphs.append({"type": "paragraph", "section": current_section,
                           "sub_section": current_sub, "text": " ".join(buffer), "top": buffer_top})

    return paragraphs, headings, numeric_lines

# --------- Table extraction strategies ----------
def extract_tables_camelot(pdf_path: str, page_number: int) -> List[Dict[str, Any]]:
    """Try Camelot lattice/stream; return list of table dicts or [] if not available/failed."""
    tables = []
    if not _HAS_CAMELOT:
        return tables
    for flavor in ("lattice", "stream"):
        try:
            logging.info("Trying Camelot (%s) on page %d", flavor, page_number)
            tlist = camelot.read_pdf(pdf_path, pages=str(page_number), flavor=flavor, strip_text="\n")
            for t in tlist:
                # convert dataframe to list of lists if possible
                try:
                    df = t.df.fillna("")
                    data = df.values.tolist()
                except Exception:
                    data = getattr(t, "data", []) or []
                if any(any(str(cell).strip() for cell in row) for row in data):
                    tables.append({"type": "table", "section": None, "description": f"Camelot({flavor})", "table_data": data, "bbox": None})
            if tables:
                return tables
        except Exception as e:
            logging.debug("Camelot(%s) error on page %d: %s", flavor, page_number, e)
            continue
    return tables

def extract_tables_pdfplumber(page) -> List[Dict[str, Any]]:
    out = []
    try:
        for t in page.find_tables():
            try:
                data = t.extract()
            except Exception:
                data = page.extract_table() or []
            if data and any(any(str(cell).strip() for cell in row) for row in data):
                out.append({"type": "table", "section": None, "description": "pdfplumber", "table_data": data, "bbox": getattr(t, "bbox", None)})
    except Exception as e:
        logging.debug("pdfplumber find_tables error: %s", e)
    return out

def numeric_rows_to_tables(numeric_lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not numeric_lines:
        return []
    rows = []
    tables = []
    numeric_lines = sorted(numeric_lines, key=lambda x: x["top"])
    last_top = numeric_lines[0]["top"]
    for nl in numeric_lines:
        if rows and (nl["top"] - last_top) > NUMERIC_LINE_GROUP_GAP:
            if rows:
                tables.append({"type": "table", "section": None, "description": "inferred_numeric", "table_data": rows, "bbox": None})
            rows = []
        rows.append(nl.get("parsed") or [nl.get("text")])
        last_top = nl["top"]
    if rows:
        tables.append({"type": "table", "section": None, "description": "inferred_numeric", "table_data": rows, "bbox": None})
    return tables

# --------- Chart / image OCR extraction ----------
def extract_charts_ocr(page, page_number: int) -> List[Dict[str, Any]]:
    """
    Use pdfplumber page.images via page.to_image().crop(bbox) and pytesseract OCR to attempt to parse
    chart labels and numeric points into chart_data list-of-lists.
    """
    charts = []
    images = getattr(page, "images", []) or []
    for idx, img in enumerate(images, start=1):
        bbox = (img.get("x0"), img.get("top"), img.get("x1"), img.get("bottom"))
        ocr_text = ""
        chart_data = []
        try:
            pil = page.to_image(resolution=200).crop(bbox).original
            ocr_text = pytesseract.image_to_string(pil)
        except Exception:
            # fallback to using fitz rendering then OCR
            try:
                doc = fitz.open(page.pdf.stream.name)
                p = doc[page.page_number - 1]
                mat = fitz.Matrix(2.0, 2.0)
                pix = p.get_pixmap(matrix=mat, clip=fitz.Rect(bbox))
                pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_text = pytesseract.image_to_string(pil)
                doc.close()
            except Exception as e:
                logging.debug("Chart OCR fallback failed: %s", e)
                ocr_text = ""

        # try to parse lines into label-num rows
        for ln in (ocr_text or "").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            nums = NUM_RE.findall(ln)
            if nums:
                first = NUM_RE.search(ln)
                label = ln[:first.start()].strip() if first else ""
                if label:
                    chart_data.append([label] + nums)
                else:
                    chart_data.append(nums)
            else:
                # keep non-numeric lines as possible labels
                chart_data.append([ln])

        charts.append({"type": "chart", "section": None, "description": (ocr_text or "").strip() or f"chart_p{page_number}_img{idx}", "table_data": [], "chart_data": chart_data, "top": bbox[1] if bbox and len(bbox) > 1 else None})
    return charts

# --------- cleanup & mapping helpers ----------
def dedupe_and_clean_tables(tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for t in tables:
        td = t.get("table_data", [])
        # normalize rows: turn rows into list of stripped strings, drop empty rows
        norm = []
        for r in td:
            if isinstance(r, (list, tuple)):
                cells = [str(c).strip() for c in r if str(c).strip() != ""]
                if cells:
                    norm.append(cells)
            else:
                s = str(r).strip()
                if s:
                    norm.append([s])
        if not norm:
            continue
        # skip trivial duplicated tables
        key = json.dumps(norm, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        t["table_data"] = norm
        out.append(t)
    return out

def nearest_heading_text(headings: List[Dict[str, Any]], top: Optional[float]) -> Optional[str]:
    if not headings or top is None:
        return None
    candidates = [h for h in headings if h["top"] <= top]
    if not candidates:
        return None
    best = max(candidates, key=lambda h: h["top"])
    return best.get("text")

# --------- main pipeline ----------
def pdf_to_json(pdf_path: str, output_path: str = "output.json") -> Dict[str, Any]:
    output = {"pages": []}
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(pdf_path)

    logging.info("Opening PDF: %s", pdf_path)
    with pdfplumber.open(pdf_path) as pdf:
        page_count = len(pdf.pages)
        logging.info("Total pages: %d", page_count)

        for pno in range(1, page_count + 1):
            page = pdf.pages[pno - 1]
            logging.info("Processing page %d / %d", pno, page_count)

            paragraphs, headings, numeric_lines = detect_paragraphs_headings_numeric(page)

            # 1) Try Camelot first (if available)
            tables = []
            if _HAS_CAMELOT:
                try:
                    tables = extract_tables_camelot(pdf_path, pno)
                except Exception as e:
                    logging.debug("Camelot top-level error: %s", e)
                    tables = []

            # 2) Fallback pdfplumber tables
            if not tables:
                try:
                    tables = extract_tables_pdfplumber(page)
                except Exception as e:
                    logging.debug("pdfplumber tables error: %s", e)
                    tables = []

            # 3) If still none, try inferred numeric tables
            if not tables and numeric_lines:
                tables = numeric_rows_to_tables(numeric_lines)

            # 4) OCR page fallback to extract numeric rows if nothing found
            if not tables:
                try:
                    page_img = page.to_image(resolution=200).original
                    ocr_text = pytesseract.image_to_string(page_img)
                    ocr_rows = []
                    for ln in ocr_text.splitlines():
                        ln = ln.strip()
                        if not ln:
                            continue
                        nums = NUM_RE.findall(ln)
                        if len(nums) >= 2:
                            first = NUM_RE.search(ln)
                            label = ln[:first.start()].strip() if first else ""
                            ocr_rows.append([label] + nums if label else nums)
                    if ocr_rows:
                        tables = [{"type": "table", "section": None, "description": "ocr_page_table", "table_data": ocr_rows}]
                except Exception as e:
                    logging.debug("Page OCR fallback error: %s", e)

            # Clean/dedupe tables
            tables = dedupe_and_clean_tables(tables or [])

            # Map tables to nearest heading where possible
            for t in tables:
                tb_top = None
                # try bbox top if exists
                bbox = t.get("bbox")
                if bbox and isinstance(bbox, (list, tuple)) and len(bbox) >= 2:
                    try:
                        tb_top = float(bbox[1])
                    except Exception:
                        tb_top = None
                # else try numeric line top or first paragraph top
                if tb_top is None:
                    if numeric_lines:
                        tb_top = numeric_lines[0].get("top")
                    elif paragraphs:
                        tb_top = paragraphs[0].get("top")
                sec = nearest_heading_text(headings, tb_top)
                if sec:
                    t["section"] = sec
                # remove bbox from output
                t.pop("bbox", None)

            # Extract charts via OCR (images on page)
            charts = []
            try:
                charts = extract_charts_ocr(page, pno)
            except Exception as e:
                logging.debug("Chart OCR error: %s", e)
                charts = []

            # Map chart sections
            for c in charts:
                c_top = c.get("top")
                sec = nearest_heading_text(headings, c_top)
                if sec:
                    c["section"] = sec
                c.pop("top", None)

            # Build final content list: interleave by vertical position to approximate original order
            # collect all items with position: paragraphs have 'top', tables try to compute top from first row or paragraph
            items_with_pos = []
            for par in paragraphs:
                if par.get("text") and par.get("text").strip():
                    items_with_pos.append({"pos": par.get("top", 0), "item": {"type": "paragraph", "section": par.get("section"), "sub_section": par.get("sub_section"), "text": par.get("text")}})

            for t in tables:
                # estimate pos
                pos = None
                if isinstance(t.get("table_data"), list) and t.get("table_data"):
                    # no exact top available; use heading top if mapped else 0
                    pos = 0
                items_with_pos.append({"pos": pos or 0, "item": {"type": "table", "section": t.get("section"), "description": t.get("description"), "table_data": t.get("table_data")}})

            for c in charts:
                items_with_pos.append({"pos": c.get("chart_data")[0][0] if c.get("chart_data") and isinstance(c.get("chart_data")[0], list) and len(c.get("chart_data")[0])>0 else 0, "item": {"type": "chart", "section": c.get("section"), "description": c.get("description"), "table_data": c.get("table_data", []), "chart_data": c.get("chart_data", [])}})

            # sort items (paragraphs first tendency by using pos then by type)
            items_sorted = sorted(items_with_pos, key=lambda x: (x["pos"] if x["pos"] is not None else 0))
            content = [it["item"] for it in items_sorted]

            output["pages"].append({"page_number": pno, "content": content})

    # ensure unique output filename so we don't overwrite unless user specified
    output_path = unique_output_filename(output_path) if os.path.exists(output_path) else output_path
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logging.info("Extraction complete. JSON saved to: %s", output_path)
    return output

# --------- CLI ----------
def main():
    if len(sys.argv) < 2:
        print("Usage: python pdf_to_json.py <input.pdf> [output.json]")
        sys.exit(1)
    pdf_in = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "output.json"
    pdf_to_json(pdf_in, out)

if __name__ == "__main__":
    main()


