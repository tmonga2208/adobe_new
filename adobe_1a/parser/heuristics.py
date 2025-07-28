import re

def is_heading(text_obj, avg_font_size, median_font_size):
    text = text_obj["text"]
    font_name = text_obj["font_name"]
    bbox = text_obj["position"]
    y_pos = bbox[1]
    
    # Heuristics
    is_large_font = text_obj["font_size"] > median_font_size
    is_bold = "Bold" in font_name or "bold" in font_name
    is_short = len(text) < 100
    is_numbered = bool(re.match(r"^(\d+\.?)+\s+[A-Z]", text))
    is_capitalized = text == text.upper() and len(text) <= 10

    if not text or len(text.split()) < 2:
        return False

    return is_large_font or is_bold or is_numbered or is_capitalized or (is_short and y_pos > 600)

def classify_heading_level(text_obj, sorted_sizes):
    size = text_obj["font_size"]
    if size >= sorted_sizes[0]:
        return "H1"
    elif len(sorted_sizes) > 1 and size >= sorted_sizes[1]:
        return "H2"
    else:
        return "H3"
