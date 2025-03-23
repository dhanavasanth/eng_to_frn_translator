from transformers import MarianMTModel, MarianTokenizer
import streamlit as st

def load_marian_model(src_lang='en', tgt_lang='fr'):
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

# Load the pre-trained MarianMT model
tokenizer, model = load_marian_model()

model.save_pretrained("saved_marian_model")
tokenizer.save_pretrained("saved_marian_model")

def translate_text(text, tokenizer, model):
    # Tokenize and generate output
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    translation = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translation

tokenizer = MarianTokenizer.from_pretrained("saved_marian_model")
model = MarianMTModel.from_pretrained("saved_marian_model")


st.title("Multiword Translation")
st.write("Enter text to translate in English to French:")

input_text = st.text_area("Input Text")

if st.button("Translate"):
    translated_text = translate_text(input_text, tokenizer, model)
    st.write(f"Translated Text: :green[{translated_text}]")