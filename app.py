import streamlit as st
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch

st.set_page_config(page_title="Emotion Analysis", page_icon="🎭", layout="centered")

st.title("🎭 Emotion Analysis App")
st.write("Enter text in English and the model will detect the emotion.")

@st.cache_resource
def load_model():
    try:
        model = DistilBertForSequenceClassification.from_pretrained("./emotion_model_final")
        tokenizer = DistilBertTokenizerFast.from_pretrained("./emotion_model_final")
        model.eval()
        return model, tokenizer
    except Exception as e:
        st.error("Model folder './emotion_model_final' was not found or could not be loaded.")
        st.exception(e)
        return None, None

model, tokenizer = load_model()

emotion_map = {
    0: ("Sadness", "😢", "#636EFA"),
    1: ("Joy", "😊", "#00CC96"),
    2: ("Love", "❤️", "#EF553B"),
    3: ("Anger", "😠", "#FF6692"),
    4: ("Fear", "😨", "#AB63FA"),
    5: ("Surprise", "😲", "#FFA15A")
}

user_input = st.text_area("Your text:", height=150, placeholder="Type a sentence in English...")

if st.button("🔍 Analyze"):
    if model is None or tokenizer is None:
        st.stop()

    if user_input.strip():
        with st.spinner("Analyzing..."):
            inputs = tokenizer(
                user_input,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            inputs = {k: v for k, v in inputs.items() if k != "token_type_ids"}

            with torch.no_grad():
                outputs = model(**inputs)

            probs = torch.softmax(outputs.logits, dim=1)[0]
            pred = torch.argmax(probs).item()
            score = probs[pred].item()

            name, emoji, color = emotion_map.get(pred, ("Unknown", "❓", "#808080"))

            st.markdown(
                f"""
                <div style="
                    background-color:{color}22;
                    border: 1px solid {color}55;
                    padding: 20px;
                    border-radius: 12px;
                    margin-top: 20px;
                ">
                    <h2 style="color:{color}; margin-bottom:10px;">{emoji} {name}</h2>
                    <p style="font-size:18px; margin:0;">
                        Confidence: <b>{score*100:.1f}%</b>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(score)

            st.subheader("Top-3 predictions")
            top_probs, top_indices = torch.topk(probs, 3)

            for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
                label_name, label_emoji, _ = emotion_map[idx]
                st.write(f"{label_emoji} **{label_name}** — {prob*100:.1f}%")
    else:
        st.warning("⚠️ Please enter some text.")