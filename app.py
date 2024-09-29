import streamlit as st
import base64
from groq import Groq

# Constants
MODEL_NAME = "llama-3.2-11b-vision-preview"  # Replace with the correct model

# Function to encode the image as base64
def encode_image(image):
    return base64.b64encode(image.read()).decode('utf-8')

# Streamlit app
def main():
    st.title("AI Image Analysis")
    
    # Check if the API key is already in session state
    if 'api_key' not in st.session_state:
        st.session_state['api_key'] = ''

    # Input box for the API key if not already provided
    if not st.session_state['api_key']:
        st.session_state['api_key'] = st.text_input("Enter your Groq API Key", type="password")
    
    # If the API key is entered, continue with the app
    if st.session_state['api_key']:
        # Upload an image
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            # Display the uploaded image
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

            # Encode the image to base64
            base64_image = encode_image(uploaded_image)

            # Call Groq API to analyze the image
            if st.button("Analyze Image"):
                client = Groq(api_key=st.session_state['api_key'])  # Use the user's API key

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
                    model=MODEL_NAME,  # Use the constant for the model name
                )

                # Display AI's response
                response = chat_completion.choices[0].message.content
                st.write("AI Response:", response)
    else:
        st.warning("Please provide your API key to continue.")

if __name__ == "__main__":
    main()
