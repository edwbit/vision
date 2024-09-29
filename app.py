import streamlit as st
import base64
from groq import Groq

# Constants for the Groq API
GROQ_API_KEY = "GROQ_API_KEY"  # Replace with your actual API key
MODEL_NAME = "llama-3.2-11b-vision-preview"  # Replace with the correct model

# Function to encode the image as base64
def encode_image(image):
    return base64.b64encode(image.read()).decode('utf-8')

# Streamlit app
def main():
    st.title("AI Image Analysis")
    
    # Upload an image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        
        # Encode the image to base64
        base64_image = encode_image(uploaded_image)
        
        # Call Groq API to analyze the image
        if st.button("Analyze Image"):
            client = Groq(api_key=GROQ_API_KEY)  # Using the constant for API key

            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What's in this image?"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ],
                model=MODEL_NAME,  # Using the constant for model
            )

            # Display AI's response
            response = chat_completion.choices[0].message.content
            st.write("AI Response:", response)

if __name__ == "__main__":
    main()
