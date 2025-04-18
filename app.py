import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# Hugging Face Token (🔐 Required for Mistral)
# -----------------------------
HUGGINGFACE_TOKEN = "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"  # Replace with your token

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_models():
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    mistral_tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        use_auth_token=HUGGINGFACE_TOKEN,
        trust_remote_code=True
    )
    mistral_model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_auth_token=HUGGINGFACE_TOKEN
    )

    return blip_processor, blip_model, mistral_tokenizer, mistral_model

blip_processor, blip_model, mistral_tokenizer, mistral_model = load_models()

# -----------------------------
# Generate Emotional Comment
# -----------------------------
def generate_emotional_comment(image: Image, emotion_label: str, mode: str):
    # Step 1: Get caption from BLIP
    blip_inputs = blip_processor(image, return_tensors="pt").to(blip_model.device)
    caption_ids = blip_model.generate(**blip_inputs)
    caption = blip_processor.decode(caption_ids[0], skip_special_tokens=True)

    # Step 2: Prompt Mistral
    if mode == "Short":
        prompt = f"""<s>[INST] Here's a description of an image: "{caption}".
The person in the image is feeling "{emotion_label}". Write a **one-line**, expressive, emotional comment that reflects this vibe.
Be human-like, creative, and fun. [/INST]"""
        max_tokens = 40
    else:
        prompt = f"""<s>[INST] Here's a description of an image: "{caption}".
The person in the image is feeling "{emotion_label}". Write a short **paragraph** that's expressive, vivid, and emotional.
Capture the atmosphere and the feeling. Be poetic and engaging. [/INST]"""
        max_tokens = 100

    # Step 3: Generate comment
    inputs = mistral_tokenizer(prompt, return_tensors="pt").to(mistral_model.device)
    output = mistral_model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.9,
        top_p=0.95
    )
    decoded = mistral_tokenizer.decode(output[0], skip_special_tokens=True)
    comment = decoded.split('[/INST]')[-1].strip()
    return caption, comment

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Emotional Comment Bot 🎭", layout="centered")
st.title("📸 Emotional Comment Generator")
st.caption("Upload an image + pick an emotion → get an expressive comment ✨")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Your Image", use_column_width=True)

    emotion = st.selectbox("Select Emotion 🎭", [
        "very happy", "slightly happy", "sad", "very sad", "angry",
        "excited", "sarcastic", "calm", "hopeful", "anxious",
        "confused", "grateful", "romantic", "playful", "proud"
    ])

    mode = st.radio("Comment Style ✍️", ["Short", "Paragraph"])

    if st.button("Generate Emotional Comment 🚀"):
        with st.spinner("Thinking..."):
            caption, comment = generate_emotional_comment(image, emotion, mode)
        st.subheader("🖼️ Image Caption")
        st.write(caption)
        st.subheader("💬 Emotional Comment")
        st.success(comment)
