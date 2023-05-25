# importing required libraries

import streamlit as st
import torch
from PIL import Image
from transformers import AutoTokenizer, ViTFeatureExtractor, VisionEncoderDecoderModel
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# initialising the pre-trained model taken from HuggingFace

device='cpu'
encoder_checkpoint = "nlpconnect/vit-gpt2-image-captioning"
decoder_checkpoint = "nlpconnect/vit-gpt2-image-captioning"
model_checkpoint = "nlpconnect/vit-gpt2-image-captioning"
feature_extractor = ViTFeatureExtractor.from_pretrained(encoder_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(decoder_checkpoint)
model = VisionEncoderDecoderModel.from_pretrained(model_checkpoint).to(device)

# predict function to convert the image into its caption through the pre-built model

def predict(image,max_length=100):
    image = image.convert('RGB')
    image = feature_extractor(image, return_tensors="pt").pixel_values.to(device)
    clean_text = lambda x: x.replace('<|endoftext|>','').split('\n')[0]
    caption_ids = model.generate(image, max_length = max_length)[0]
    caption_text = clean_text(tokenizer.decode(caption_ids))
    return caption_text 

def convert_to_hashtags(sentence):
    # Tokenize the sentence into words
    tokens = word_tokenize(sentence)

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]

    # Convert words to hashtags
    hashtags = ['#' + token for token in filtered_tokens]
    common_hash = ['#instagood', '#instagram', '#like4like', '#picoftheday', '#instadaily', '#like', '#instalike']
    hashtags.extend(common_hash)
    hashtags = [*set(hashtags)]
    return hashtags



def main():
    st.title("Image Caption and Hashtag Generator")
    st.write("Upload an image and get the predicted caption and hastags.")

    # Image upload
    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        st.divider()

        # Make prediction
        predicted_label = predict(image)
        st.subheader("Predicted Caption/Description:")
        st.write(f"{predicted_label}")

        st.divider()

        prediction_hash = convert_to_hashtags(predicted_label)
        ten_hash = prediction_hash[:10]
        st.subheader("Predicted Hashtags:")
        for hash in ten_hash:
            st.write(f"{hash}")

if __name__ == "__main__":
    main()