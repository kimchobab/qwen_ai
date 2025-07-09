from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import settings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,                     # 8비트 양자화
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
)

#  모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(
    settings.QWEN_MODEL,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    settings.QWEN_MODEL,
    quantization_config=bnb_config,       # 양자화 config 주입
    device_map="auto",                    # 자동으로 GPU 배치
    trust_remote_code=True
)
