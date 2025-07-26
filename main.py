import os
import json
from statistics import median
from parser.extractor import extract_layout_info
from parser.heuristics import is_heading, classify_heading_level

INPUT_DIR = "C:/Users/hp/Desktop/adobe_new/input"
OUTPUT_DIR = "C:/Users/hp/Desktop/adobe_new/output"

def process_pdf(pdf_path):
    layout_info = extract_layout_info(pdf_path)
    font_sizes = [obj["font_size"] for obj in layout_info]
    if not font_sizes:
        return {"title": "", "outline": []}
    
    median_size = median(font_sizes)
    size_ranks = sorted(set(font_sizes), reverse=True)

    title = ""
    headings = []

    for obj in layout_info:
        if not title or obj["font_size"] > max(font_sizes):
            title = obj["text"]
        if is_heading(obj, obj["font_size"], median_size):
            level = classify_heading_level(obj, size_ranks)
            if obj["text"] != title:
                headings.append({
                    "level": level,
                    "text": obj["text"],
                    "page": obj["page"]
                })

    return {
        "title": title,
        "outline": headings
    }

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(INPUT_DIR, filename)
            result = process_pdf(pdf_path)
            json_name = os.path.splitext(filename)[0] + ".json"
            with open(os.path.join(OUTPUT_DIR, json_name), "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

if __name__ == "__main__":
    run()
