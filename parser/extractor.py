from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LTTextLine

def extract_layout_info(pdf_path):
    layout_data = []
    for page_num, page_layout in enumerate(extract_pages(pdf_path), start=1):
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                for line in element:
                    if not isinstance(line, LTTextLine):
                        continue

                    font_sizes = []
                    fonts = []
                    for obj in line:
                        if isinstance(obj, LTChar):
                            font_sizes.append(obj.size)
                            fonts.append(obj.fontname)

                    if not font_sizes:
                        continue

                    avg_font = sum(font_sizes) / len(font_sizes)
                    text = line.get_text().strip()
                    font_name = fonts[0] if fonts else ""
                    bbox = line.bbox  # (x0, y0, x1, y1)

                    layout_data.append({
                        "text": text,
                        "font_size": avg_font,
                        "font_name": font_name,
                        "page": page_num,
                        "position": bbox
                    })
    return layout_data
