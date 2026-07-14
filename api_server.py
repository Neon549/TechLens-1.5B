import sys, torch, time, re
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import uvicorn
import json

sys.path.insert(0, 'src')
from techlens.prompts.template import build_system_prompt, serialize_input
from techlens.schemas.output import parse_model_output, validate_output

BASE = 'D:/models/models/Qwen--Qwen3-1.7B/snapshots/master'
SFT = 'D:/code/ProjectExample/TechLens-1.5B/models/adapters/techlens-1.7b-sft'
DPO = 'D:/code/ProjectExample/TechLens-1.5B/models/adapters/techlens-1.7b-dpo'

print('加载模型...')
tok = AutoTokenizer.from_pretrained(SFT)
base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map='cuda')
model = PeftModel.from_pretrained(base, DPO)
model.eval()
print('模型加载完成')

app = FastAPI(title='TechLens API', description='A股技术面分析本地推理服务')

class AnalyzeRequest(BaseModel):
    history_result: str
    price_result: str
    kdj_result: str
    stock_code: str

class StockCodeRequest(BaseModel):
    stock_code: str

def run_inference(history_result, price_result, kdj_result, stock_code):
    sample_input = {'history': history_result, 'price': price_result, 'kdj': kdj_result}
    system = build_system_prompt()
    user = serialize_input(sample_input, stock_code)
    msgs = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
    text_input = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text_input, return_tensors='pt').input_ids.to('cuda')
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=256, do_sample=False, pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    latency = time.time() - t0
    obj, err, fenced = parse_model_output(text)
    return obj, err, latency

@app.get('/health')
def health():
    return {'status': 'ok', 'model': 'techlens-1.7b-dpo'}

@app.post('/analyze')
def analyze(req: AnalyzeRequest):
    obj, err, latency = run_inference(
        req.history_result, req.price_result, req.kdj_result, req.stock_code)
    if err:
        return {'success': False, 'error': err, 'latency_s': latency}
    errs = validate_output(obj)
    return {'success': True, 'result': obj, 'valid': not errs, 'latency_s': round(latency, 2)}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8088)
