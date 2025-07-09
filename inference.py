import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from peft import PeftModel

base_model = "Qwen/Qwen2.5-1.5B-Instruct"
lora_path = "./checkpoints_lora/checkpoint-1866"

#  tokenizer 설정
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.chat_template = None                         #  chat 프롬프트 해제
tokenizer.use_default_system_prompt = False            #  system prompt 제거

#  4bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

#  base model + LoRA 적용
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(model, lora_path)
model.eval()

print(" 적용된 PEFT 계층:", model.peft_config)

#  테스트 문장들
sentences = [
    "우동이 불을 것 같아 걱정이다.",
    "또한 가름한 얼굴에 초승달 같은 눈썹.",
    "모비는 1670년 현종 11년에 후손인 김려와 연양 현감이 세웠다.",
    "저장할까요 또는 취소할까요?",
    "나는 밥을 먹고 학교를 갔습니다",
    "그는 아무말도 않고 떠났습니다"
]

#  문장 교정
for i, text in enumerate(sentences, start=1):
    prompt = (
        f"다음 문장에서 어문 규범에 맞지 않는 부분을 고치고, "
        f"그 이유를 **한국어로만** 설명하시오. 영어, 중국어 등 다른 언어를 사용하지 마시오.\n\n{text}"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n[문장 {i}] 입력: {text}\n→ 응답:\n{decoded.strip()}")
