from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from chains import start_chain
import json


st.title("English Session Generator 📘")
st.markdown("Fill in the form below to generate an explanation and examples.")

with st.form("session_form"):
    tense = st.selectbox("Select the tense", ["past", "present", "future"])
    modifier = st.selectbox("Select the modifier", ["perfect", "simple", "progressive", "perfect progressive"])
    topic = st.text_input("Topic (e.g., travel, education, work)", "travel")
    submitted = st.form_submit_button("Generate Lesson")

if submitted:
    with st.spinner("Generating..."):
        try:
            result = start_chain.invoke({
                "tense": tense,
                "modifier": modifier,
                "topic": topic
            })
            raw_output = result["start_output"]
            try:
                parsed = json.loads(raw_output)
            except json.JSONDecodeError:
                st.error(f"❌ Error parsing JSON output. Please check the input values: {raw_output}")
                parsed = {}

            st.success("Session Generated ✅")
            st.subheader("Explanation")
            st.write(parsed["explanation"])

            st.subheader("Examples")
            for ex in parsed["examples"]:
                st.markdown(f"- {ex}")
        except Exception as e:
            st.error(f"❌ Error: {e}")