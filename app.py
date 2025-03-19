import streamlit as st
import bot_code  # Import the functions from bot_code.py

st.title("Physics AI Bot")

# User input
user_input = st.text_input("Ask me a physics question:")

if st.button("Get Answer"):
    if user_input:
        response = bot_code.ask_question(user_input)  # Call the function from bot_code.py
        st.write("**Response:**", response)
    else:
        st.warning("Please enter a question!")
