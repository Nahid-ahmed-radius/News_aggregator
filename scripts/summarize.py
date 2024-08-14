import sys
import re
from transformers import GPT2LMHeadModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("flax-community/gpt2-bengali")
model = GPT2LMHeadModel.from_pretrained("faridulreza/gpt2-bangla-summurizer")

model.to("cuda")

BEGIN_TOKEN = "<।summary_begin।>"
END_TOKEN = " <।summary_end।>"
BEGIN_TOKEN_ALT = "<।sum_begin।>"
END_TOKEN_ALT = " <।sum_end।>"
SUMMARY_TOKEN = "<।summary।>"

def processTxt(txt):
    txt = re.sub(r"।", "। ", txt)
    txt = re.sub(r",", ", ", txt)
    txt = re.sub(r"!", "। ", txt)
    txt = re.sub(r"\?", "। ", txt)
    txt = re.sub(r"\"", "", txt)
    txt = re.sub(r"'", "", txt)
    txt = re.sub(r"’", "", txt)
    txt = re.sub(r"’", "", txt)
    txt = re.sub(r"‘", "", txt)
    txt = re.sub(r";", "। ", txt)
    txt = re.sub(r"\s+", " ", txt)
    return txt

def index_of(val, in_text, after=0):
    try:
        return in_text.index(val, after)
    except ValueError:
        return -1

def summarize(txt):
    txt = processTxt(txt.strip())
    txt = BEGIN_TOKEN + txt + SUMMARY_TOKEN

    inputs = tokenizer(txt, max_length=800, truncation=True, return_tensors="pt")
    inputs.to("cuda")
    output = model.generate(inputs["input_ids"], max_length=len(txt) + 220, pad_token_id=tokenizer.eos_token_id)

    txt = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

    start = index_of(SUMMARY_TOKEN, txt) + len(SUMMARY_TOKEN)

    if start == len(SUMMARY_TOKEN) - 1:
        return "No Summary!"

    end = index_of(END_TOKEN, txt, start)
    if end == -1:
        end = index_of(END_TOKEN_ALT, txt, start)
    if end == -1:
        end = index_of(BEGIN_TOKEN, txt, start)
    if end == -1:
        return txt[start:].strip()

    txt = txt[start:end].strip()
    end = index_of(SUMMARY_TOKEN, txt)
    if end == -1:
        return txt
    else:
        return txt[:end].strip()

if __name__ == "__main__":
    text = sys.argv[1]
    summary = summarize(text)
    print(summary)
