from transformers import MarianMTModel, MarianTokenizer

def load_marian_model(src_lang='en', tgt_lang='fr'):
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

# Load the pre-trained MarianMT model
tokenizer, model = load_marian_model()

model.save_pretrained("saved_marian_model")
tokenizer.save_pretrained("saved_marian_model")

print("Model saved successfully.")