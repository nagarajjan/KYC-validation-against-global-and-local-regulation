import streamlit as st
from core_kyc_logic import kyc_app

st.title("KYC Automation with LangGraph")

# Initialize session state for multi-turn conversation
if "kyc_messages" not in st.session_state:
    st.session_state.kyc_messages = []

# Streamlit UI elements
country = st.selectbox("Select Customer's Country", ["Global", "Singapore"])
uploaded_file = st.file_uploader("Upload KYC Document (Text)", type="txt")
run_kyc = st.button("Run KYC Process")

if run_kyc and uploaded_file is not None:
    # Read the content of the uploaded file
    document_content = uploaded_file.read().decode("utf-8")
    
    initial_state = {
        "query": document_content,
        "country": country.lower(),
        "customer_data": {},
        "validation_status": [],
        "iterations": 0,
        "messages": []
    }
    
    # Run the LangGraph
    st.session_state.kyc_messages.append("---KYC Process Initiated---")
    
    # Stream the graph execution output to the UI
    for s in kyc_app.stream(initial_state):
        step_name = list(s.keys())[0]
        step_output = s[step_name]
        
        # Display relevant information from each step
        st.session_state.kyc_messages.append(f"**{step_name}:**")
        if "messages" in step_output:
            st.session_state.kyc_messages.extend(step_output["messages"])
        if "customer_data" in step_output:
            st.session_state.kyc_messages.append(f"**Extracted Data:**\n{step_output['customer_data']}")
        if "validation_status" in step_output:
            for status in step_output["validation_status"]:
                st.session_state.kyc_messages.append(status)
        
        st.session_state.kyc_messages.append("---")
        
    st.session_state.kyc_messages.append("---KYC Process Finished---")

# Display the conversation
for message in st.session_state.kyc_messages:
    st.markdown(message)
