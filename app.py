import streamlit as st
import base64
from groq import Groq

# Constants
MODEL_NAME = "llama-3.2-11b-vision-preview"  # Replace with the correct model
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
    
    # Check if message history is in session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []

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

            # Initialize the Groq client
            client = Groq(api_key=st.session_state['api_key'])  # Use the user's API key

            # Analyze the image immediately after upload
            if len(st.session_state.messages) == 0:
                initial_analysis = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Please analyze this image."},
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

                # Display the initial AI response to the image
                initial_response = initial_analysis.choices[0].message.content
                st.write("AI Initial Analysis:", initial_response)

                # Append the initial analysis to message history
                st.session_state.messages.append({"role": "ai", "content": initial_response})

            # Display conversation history
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.write("User:", message["content"])
                else:
                    st.write("AI:", message["content"])

            # Allow the user to ask follow-up questions
            user_prompt = st.text_input("Ask a follow-up question about this image", placeholder="e.g., Can you tell me more?")

            # Handle follow-up questions
            if st.button("Submit Question"):
                if user_prompt:
                    # Append user question to message history
                    st.session_state.messages.append({"role": "user", "content": user_prompt})

                    # Make API call with the updated message history
                    follow_up_analysis = client.chat.completions.create(
                        messages=st.session_state.messages + [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": user_prompt},
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

                    # Get and display AI's response to the follow-up question
                    follow_up_response = follow_up_analysis.choices[0].message.content
                    st.write("AI Follow-up Response:", follow_up_response)

                    # Append AI response to message history
                    st.session_state.messages.append({"role": "ai", "content": follow_up_response})
                else:
                    st.warning("Please enter a follow-up question.")
    else:
        st.warning("Please provide your API key to continue.")

if __name__ == "__main__":
    main()
