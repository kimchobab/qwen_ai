import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training


def run_training():
    #  1. 모델과 토크나이저 설정
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    #  2. 4bit 양자화 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    #  3. 모델 로딩 및 LoRA 준비
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    #  4. LoRA 설정 적용
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    #  5. 학습 데이터 불러오기
    data_path = "docs/data/train_flat.jsonl"
    raw_dataset = load_dataset("json", data_files={"train": data_path})

    #  6. 전처리 함수 정의 (input과 output 합치기)
    def tokenize(example):
        input_text = example["input"]
        output_text = example["output"]
        if isinstance(input_text, list):  # 혹시라도 리스트면 join 처리
            input_text = " ".join(input_text)
        if isinstance(output_text, list):
            output_text = " ".join(output_text)

        prompt = input_text + "\n" + output_text
        tokens = tokenizer(prompt, max_length=512, padding="max_length", truncation=True)
        return {
            "input_ids": tokens["input_ids"],
            "labels": tokens["input_ids"].copy(),
        }

    #  7. 전처리 및 dataset 변환 (78개 문제 방지)
    tokenized = raw_dataset["train"].map(tokenize, remove_columns=raw_dataset["train"].column_names)
    tokenized_dataset = Dataset.from_dict(tokenized[:])  # Truncating 방지

    print(f" 전체 전처리 완료: {len(tokenized_dataset)}개")

    #  8. 학습 설정
    training_args = TrainingArguments(
        output_dir="./checkpoints_lora",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=100,
        save_total_limit=1,
        fp16=True,
        report_to="none",
        remove_unused_columns=False,
        max_steps=-1,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    #  9. Trainer로 학습
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print(" Qwen LoRA 학습 시작")
    trainer.train()
    print(" 학습 완료")


if __name__ == "__main__":
    run_training()
