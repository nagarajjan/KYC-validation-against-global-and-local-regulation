import os
import operator
from typing import TypedDict, List
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langgraph.graph import StateGraph, START, END # ConditionalEdge removed here

# 1. State definition for LangGraph
class KycProcessState(TypedDict):
    """Represents the state of the KYC process."""
    query: str
    country: str
    customer_data: dict
    validation_status: List[str]
    iterations: int
    messages: List[str]
    country_rules: str

# 2. Local RAG setup
def setup_vector_store():
    # Ingest regulations from text files
    global_loader = TextLoader("kyc_rules_global.txt")
    singapore_loader = TextLoader("kyc_rules_singapore.txt")
    
    docs = global_loader.load() + singapore_loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    splits = text_splitter.split_documents(docs)

    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
    return vectorstore

vectorstore = setup_vector_store()
retriever = vectorstore.as_retriever()
local_llm = Ollama(model="llama3") # Use any local LLM via Ollama

# 3. Define the nodes for the LangGraph
def get_rules_node(state: KycProcessState):
    """Retrieves relevant global and country-specific rules."""
    print("---Fetching KYC Rules---")
    country = state.get("country", "global").lower()
    
    global_rules = retriever.invoke("Global KYC regulations")
    country_rules = []
    if country == "singapore":
        country_rules = retriever.invoke("Singapore KYC regulations")
        
    rules_text = "\n".join([doc.page_content for doc in global_rules + country_rules])
    
    return {
        "messages": state["messages"] + [f"Fetched KYC rules for {country}."],
        "country_rules": rules_text
    }

def extract_data_node(state: KycProcessState):
    """Uses LLM to extract data from document text."""
    print("---Extracting Data with LLM---")
    data_prompt = PromptTemplate.from_template(
        "Based on the following document text, extract the customer's name, DOB, address, and nationality. "
        "Return the information in a JSON format. If information is missing, state so.\n\n"
        "Document Text: {query}"
    )
    chain = data_prompt | local_llm
    extracted_text = chain.invoke({"query": state["query"]})
    
    import json
    customer_data = {}
    try:
        # Extract the JSON block from the LLM's raw text output
        json_start = extracted_text.find('{')
        json_end = extracted_text.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            json_str = extracted_text[json_start:json_end]
            customer_data = json.loads(json_str)
        else:
            customer_data = {"error": "LLM failed to produce valid JSON."}
    except json.JSONDecodeError:
        customer_data = {"error": f"LLM produced invalid JSON: {extracted_text}"}
        
    return {
        "messages": state["messages"] + ["Extracted customer data."],
        "customer_data": customer_data
    }
    
def validate_data_node(state: KycProcessState):
    """Uses LLM to validate extracted data against rules."""
    print("---Validating Data with LLM---")
    validation_prompt = PromptTemplate.from_template(
        "You are a KYC compliance officer. Validate the following customer data against the rules provided. "
        "Focus on all global rules and specific Singapore rules if applicable. "
        "Report any validation failures concisely.\n\n"
        "KYC Rules:\n{rules}\n\n"
        "Customer Data:\n{data}\n\n"
        "Validation Result:"
    )
    
    data_str = str(state["customer_data"])
    rules_str = state.get("country_rules", "")
    
    chain = validation_prompt | local_llm
    validation_result = chain.invoke({"rules": rules_str, "data": data_str})
    
    if "failure" in validation_result.lower() or "not compliant" in validation_result.lower():
        status = [f"Validation failed: {validation_result}"]
    else:
        status = ["Validation successful."]
        
    return {
        "messages": state["messages"] + [f"Validation check: {status}", validation_result],
        "validation_status": status
    }

def risk_assessment_node(state: KycProcessState):
    """Assess customer risk based on validation results."""
    print("---Assessing Risk---")
    if any("failed" in s for s in state["validation_status"]):
        risk_level = "High Risk: Needs Enhanced Due Diligence (EDD)."
    else:
        risk_level = "Low Risk: Meets basic compliance."
    
    return {
        "messages": state["messages"] + [risk_level]
    }

def check_for_errors(state: KycProcessState):
    """Conditional edge logic."""
    if any("failed" in s for s in state["validation_status"]):
        return "re_check"
    else:
        return "end_process"
        
# 4. Build the LangGraph workflow
def build_kyc_graph():
    workflow = StateGraph(KycProcessState)
    
    workflow.add_node("get_rules", get_rules_node)
    workflow.add_node("extract_data", extract_data_node)
    workflow.add_node("validate_data", validate_data_node)
    workflow.add_node("risk_assessment", risk_assessment_node)
    
    workflow.add_edge(START, "get_rules")
    workflow.add_edge("get_rules", "extract_data")
    workflow.add_edge("extract_data", "validate_data")
    
    workflow.add_conditional_edges(
        "validate_data",
        check_for_errors,
        {
            "re_check": "extract_data", # In a real app, this would be a human-in-the-loop step
            "end_process": "risk_assessment"
        }
    )
    workflow.add_edge("risk_assessment", END)
    
    return workflow.compile()

kyc_app = build_kyc_graph()