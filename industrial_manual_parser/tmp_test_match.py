import fitz
import json

pdf_path = "data/manuals/esp32_datasheet_en.pdf"
chunks_file = "data/chunks.json"

with open(chunks_file, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Find chunk
target_chunk = next(c for c in chunks if "Power-up and Reset" in c["text"])

doc = fitz.open(pdf_path)
page_texts = {}
for i, page in enumerate(doc):
    page_texts[i + 1] = page.get_text().lower()

def get_trigrams(text):
    words = text.split()
    return set(" ".join(words[i:i+3]) for i in range(len(words)-2))

def get_best_page(chunk_text):
    chunk_grams = get_trigrams(chunk_text.lower())
    
    best_page = 1
    best_score = 0
    scores = []
    
    for page_num, p_text in page_texts.items():
        p_grams = get_trigrams(p_text)
        score = len(chunk_grams.intersection(p_grams))
        scores.append((page_num, score))
        if score > best_score:
            best_score = score
            best_page = page_num
            
    scores.sort(key=lambda x: x[1], reverse=True)
    print("Top 5 pages:", scores[:5])
    return best_page, best_score

p, s = get_best_page(target_chunk["text"])
print(f"Target mapped to page {p} with score {s}")
