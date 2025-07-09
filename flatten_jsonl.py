import json

input_path = "docs/data/train.jsonl"
output_path = "docs/data/train_flat.jsonl"

with open(input_path, "r", encoding="utf-8") as infile, \
     open(output_path, "w", encoding="utf-8") as outfile:
    
    count = 0  # 몇 줄 처리했는지 확인용

    for line in infile:
        try:
            data = json.loads(line)
            input_text = data["input"]["question"]
            output_text = data["output"]["answer"]
            flattened = {"input": input_text, "output": output_text}
            outfile.write(json.dumps(flattened, ensure_ascii=False) + "\n")
            count += 1
        except Exception as e:
            print(f" 에러 on line: {e}")

    print(f" {count}줄 변환 완료됨")
