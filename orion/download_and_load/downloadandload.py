# deepseek_runner.py
# -------------------
# This script loads and runs the DeepSeek-R1 model locally using Hugging Face Transformers.
# It’s written for a MacBook (M2 Pro) but works on CUDA GPUs too — just change "mps" → "cuda".

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 🔹 The name of the pretrained model on Hugging Face.
# This will automatically download model weights and configs the first time you run it.
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Detect best available device and choose a compatible dtype
# MPS does not support bfloat16; prefer float16 on MPS/CUDA, float32 on CPU
DEVICE = (
    "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
)
DTYPE = torch.float16 if DEVICE in ("mps", "cuda") else torch.float32

# Keep singletons to avoid re-loading the model on every request
_MODEL = None
_TOKENIZER = None

# -------------------------------------------------------------------------
# 1️⃣ Load model + tokenizer
# -------------------------------------------------------------------------
def load_model():
    """
    Load the model/tokenizer once with a dtype compatible with the selected device.
    We explicitly avoid BF16 on MPS by forcing float16.
    """
    global _MODEL, _TOKENIZER
    if _MODEL is not None and _TOKENIZER is not None:
        return _MODEL, _TOKENIZER

    print("🔄 Loading DeepSeek model...")

    # Explicitly control dtype and device; avoid BF16 on MPS
    _MODEL = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        device_map=None,
    )
    _MODEL.to(DEVICE)

    _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Ensure pad token exists to silence warnings/errors during generation
    if _TOKENIZER.pad_token is None:
        _TOKENIZER.pad_token = _TOKENIZER.eos_token

    print(f"✅ Model and tokenizer loaded successfully on {DEVICE} with dtype {DTYPE}.")
    return _MODEL, _TOKENIZER


# -------------------------------------------------------------------------
# 2️⃣ Generate a response to a text prompt
# -------------------------------------------------------------------------
def ask(prompt: str):
    """
    Sends a text prompt to the model and returns its generated response.
    Steps:
      1. Convert the text into model-readable tensors (tokenization).
      2. Run model.generate() to predict the next tokens.
      3. Decode the tokens back into readable text.
    """

    # Load model + tokenizer
    model, tokenizer = load_model()

    # Tokenize and move tensors to the same device as the model
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # Run generation
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode the output token IDs back into a string
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response_text


# -------------------------------------------------------------------------
# 3️⃣ Run this file directly (not imported)
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Example research query to test your model
    question = "who's the president of the United States?"

    # Ask the model
    response = ask(question)

    # Print the output
    print("\n🧠 DeepSeek response:\n")
    print(response)
