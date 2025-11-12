import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import PeftModel, PeftConfig

st.set_page_config(page_title="Pseudo-code to Python", layout="wide")
st.title("🖥️ Pseudo-code to Python Generator")

@st.cache_resource(show_spinner=True)
def load_model(model_path="./pseudo-to-python-final"):
    config = PeftConfig.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'additional_special_tokens': ['<|pseudo|>', '<|python|>', '<|end|>']})

    base_model = GPT2LMHeadModel.from_pretrained(config.base_model_name_or_path)
    base_model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()
st.success("Model loaded successfully!")

def generate_code(pseudocode):
    input_text = f"<|pseudo|>{pseudocode}<|python|>"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=3,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            early_stopping=True
        )
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
    if "<|python|>" in full_output:
        python_code = full_output.split("<|python|>")[1]
        if "<|end|>" in python_code:
            python_code = python_code.split("<|end|>")[0]
        return python_code.strip()
    return full_output.replace(input_text, "").strip()

st.subheader("Enter your pseudocode:")
user_input = st.text_area("Pseudocode input", height=200)

if st.button("Generate Python Code"):
    if not user_input.strip():
        st.warning("Please enter pseudocode!")
    else:
        with st.spinner("Generating code..."):
            generated = generate_code(user_input)
            st.subheader("Generated Python Code:")
            st.code(generated, language="python")
