import streamlit as st
import base64
from groq import Groq

# Constants
MODEL_NAME = "llava-v1.5-7b-4096-preview"  # Replace with the correct model
TEMPERATURE = 1  # Adjust as needed for randomness
MAX_TOKENS = 1024  # Limit the response length
TOP_P = 1  # Set to 1 to consider all tokens
STREAM = False  # Disable token-by-token streaming for now
STOP = None  # No stop sequence for now

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

            # Prompt for the user to ask questions about the image
            user_prompt = st.text_input("Enter your prompt to ask about this image", placeholder="e.g., What's in this image?")

            # Encode the image to base64
            base64_image = encode_image(uploaded_image)

            # Use the default Streamlit button
            if st.button("Submit Prompt"):
                if user_prompt:
                    client = Groq(api_key=st.session_state['api_key'])  # Use the user's API key

                    # Create chat completion with the prompt and additional parameters
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": user_prompt},  # Use the user's prompt
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
                        temperature=TEMPERATURE,  # Adjust output randomness
                        max_tokens=MAX_TOKENS,  # Limit the number of tokens in response
                        top_p=TOP_P,  # Control nucleus sampling
                        stream=STREAM,  # Set streaming behavior
                        stop=STOP  # Optional stop sequence
                    )

                    # Display AI's response
                    response = chat_completion.choices[0].message.content
                    st.write("Vision:", response)
                else:
                    st.warning("Please enter a prompt to submit.")
    else:
        st.warning("Please provide your API key to continue.")

if __name__ == "__main__":
    main()
