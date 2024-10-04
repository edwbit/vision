import streamlit as st
import base64
from groq import Groq, AuthenticationError

# Constants
TEMPERATURE = 1
MAX_TOKENS = 1024
TOP_P = 1
STREAM = False
STOP = None
UPLOAD_LIMIT_MB = 20

# Function to encode the image as base64
def encode_image(image):
    return base64.b64encode(image.read()).decode('utf-8')

# Streamlit app
def main():
    st.title("AI Image Analysis")
    st.caption("Developed by Edwin Bitco with AI assistance.")
    
    # Initialize session state for API key
    if 'api_key' not in st.session_state:
        st.session_state['api_key'] = ''
    
    if 'api_key_entered' not in st.session_state:
        st.session_state['api_key_entered'] = False

    # Input box for the API key if not already provided
    if not st.session_state['api_key_entered']:
        api_key_input = st.text_input("Enter your Groq API Key", type="password")

        # Update session state if the API key is entered
        if api_key_input:
            st.session_state['api_key'] = api_key_input
            st.session_state['api_key_entered'] = True
            st.success("API key entered successfully!")

    # If the API key is entered, continue with the app
    if st.session_state['api_key_entered']:
        # Upload an image
        uploaded_image = st.file_uploader("Upload an image (max size: 20 MB)", type=["jpg", "jpeg", "png"])

        # Model selection
        model_options = ["llava-v1.5-7b-4096-preview", "llama-3.2-11b-vision-preview"]
        selected_model = st.selectbox("**Select Model**", model_options)

        # Input box for user's instruction (now using st.text_input)
        user_instruction = st.text_input("Enter your instruction for the AI to analyze the image:")

        # OK button to trigger the analysis
        if st.button("OK"):
            if uploaded_image is not None and user_instruction:
                # Check if the uploaded image exceeds the size limit
                if uploaded_image.size > UPLOAD_LIMIT_MB * 1024 * 1024:
                    st.warning("The uploaded image exceeds the 20MB limit. Please upload a smaller image.")
                else:
                    # Display the uploaded image
                    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

                    # Encode the image to base64
                    base64_image = encode_image(uploaded_image)

                    try:
                        # Initialize Groq client
                        client = Groq(api_key=st.session_state['api_key'])

                        # Use the selected model from the dropdown
                        MODEL_NAME = selected_model

                        # Create chat completion based on user's instruction and the uploaded image
                        analysis = client.chat.completions.create(
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": user_instruction},
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/jpeg;base64,{base64_image}",
                                            },
                                        },
                                    ],
                                }
                            ],
                            model=MODEL_NAME,
                            temperature=TEMPERATURE,
                            max_tokens=MAX_TOKENS,
                            top_p=TOP_P,
                            stream=STREAM,
                            stop=STOP
                        )

                        # Display the AI's response
                        response = analysis.choices[0].message.content
                        st.write("AI Response:", response)
                    
                    except AuthenticationError:
                        st.error("Authentication failed. Please check your API key.")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")
            else:
                st.warning("Please upload an image and provide an instruction before clicking OK.")

if __name__ == "__main__":
    main()
