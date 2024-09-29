import streamlit as st
import base64
from groq import Groq

# Constants
MODEL_NAME = "llama-3.2-11b-vision-preview"  # Use the correct model
TEMPERATURE = 1  # Adjust as needed for randomness
MAX_TOKENS = 1024  # Limit the response length
TOP_P = 1  # Set to 1 to consider all tokens
STREAM = False  # Disable token-by-token streaming for now
STOP = None  # No stop sequence for now
UPLOAD_LIMIT_MB = 20  # Upload limit in MB

# Function to encode the image as base64
def encode_image(image):
    return base64.b64encode(image.read()).decode('utf-8')

# Streamlit app
def main():
    st.title("AI Image Analysis")

    # Check if the API key is already in session state
    if 'api_key' not in st.session_state:
        st.session_state['api_key'] = ''
    
    if 'api_key_entered' not in st.session_state:
        st.session_state['api_key_entered'] = False

    # Input box for the API key if not already provided
    if not st.session_state['api_key_entered']:
        api_key_input = st.text_input("Enter your Groq API Key", type="password")

        # Update session state if the API key is entered
        if api_key_input:
            st.session_state['api_key'] = api_key_input  # Store the API key in session state
            st.session_state['api_key_entered'] = True  # Set flag to indicate API key has been entered
            st.success("API key entered successfully!")

    # If the API key is entered, continue with the app
    if st.session_state['api_key_entered']:
        # Upload an image
        uploaded_image = st.file_uploader("Upload an image (max size: 20 MB)", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            # Check if the uploaded image exceeds the size limit
            if uploaded_image.size > UPLOAD_LIMIT_MB * 1024 * 1024:
                st.warning("The uploaded image exceeds the 20MB limit. Please upload a smaller image.")
            else:
                # Display the uploaded image
                st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

                # Encode the image to base64
                base64_image = encode_image(uploaded_image)

                # Initialize Groq client
                client = Groq(api_key=st.session_state['api_key'])

                # Create chat completion for initial image analysis
                if st.button("Analyze Image"):
                    # Create chat completion for image analysis
                    initial_analysis = client.chat.completions.create(
                        messages=[
                            {
                                "role": "user",
                                "content": "Please analyze this image."
                            },
                            {
                                "role": "user",
                                "content": f"data:image/jpeg;base64,{base64_image}"
                            }
                        ],
                        model=MODEL_NAME,
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                        top_p=TOP_P,
                        stream=STREAM,
                        stop=STOP
                    )

                    # Display the AI response to the image
                    initial_response = initial_analysis.choices[0].message.content
                    st.write("AI Initial Analysis:", initial_response)

if __name__ == "__main__":
    main()
