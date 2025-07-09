import os
import json

input_path = os.path.join("docs", "data", "korean_language_rag_V1.0_train.json")
output_path = os.path.join("docs", "data", "train.jsonl")

# JSON 파일 열기
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# JSONL 파일로 저장
with open(output_path, "w", encoding="utf-8") as f_out:
    for entry in data:
        json.dump(entry, f_out, ensure_ascii=False)
        f_out.write("\n")

print(f" 변환 완료! → {output_path}")
