import os

# Path configurations
JD_PATH = './dataset/jd'
RESUME_PATH = './dataset/resume'
DOCUMENTS_PATH = './documents.pkl'
TEMP_UPLOAD_DIR = './temp_uploads'
INDEX_PATH = './resume_index'
LORA_PATHS = {
    'meta-llama/Llama-3.2-1B-Instruct': 'weights/llama3-1B',
    'meta-llama/Llama-3.2-3B-Instruct': 'weights/llama3-3B',
    'meta-llama/Llama-3.1-8B-Instruct': 'weights/llama3-8B',
}
DOMAINS = [s.replace('.json', '') for s in os.listdir(JD_PATH)] if os.path.exists(JD_PATH) else []

# Model configurations
QUERY_CLASSIFIER_MODEL = "facebook/bart-large-mnli"
METADATA_MODEL = 'meta-llama/Llama-3.2-1B-Instruct'
AGENT_MODEL = 'meta-llama/Llama-3.2-1B-Instruct'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
EMBEDDING_DIM = 384
RAG_K_THRESHOLD = 5
DEFAULT_LLM_MODEL = 'meta-llama/Llama-3.1-8B-Instruct'
AVAILABLE_LLM_MODELS = [
    'meta-llama/Llama-3.1-8B-Instruct', 
    'meta-llama/Llama-3.2-3B-Instruct',
    'meta-llama/Llama-3.2-1B-Instruct'
]

# Memory management
MEMORY_CHUNK_SIZE = 1000  # Number of documents to process at once
MEMORY_LOGGING_INTERVAL = 5  # Log memory usage after processing this many documents

# UI configurations
APP_TITLE = "Resume Assistant"
APP_ICON = "favicon.ico"
WELCOME_MESSAGE = """
### Introduction 🎉

Resume Assistant is a RAG-based tool that enables HR professionals to efficiently identify top candidates from thousands of applications.

As an experimental version, the project is set up with 4-bit quantized 8B and smaller-parameter versions of LLaMA 3 Instruct as the LLMs. 
Furthermore, we fine-tune these LLMs on a custom QA dataset and make them optional models for use. 
The resume database is also set by default to include 2,400+ resumes from various jobs and domains.

(Reference: [Resume Database](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset), [QA Dataset](https://resources.workable.com/job-descriptions)).

### Getting Started 📝

1, Add your Huggingface API Key.

2, Choose the LLM model.

Note: We recommend using LLaMA 3.1 8B Instruct as the LLM to get the most out of our tool. In that case, a GPU with 24 GB of VRAM is required.
"""
FAQ_MESSAGE = """
### FAQs ❓

**1, How does the chatbot work?**

The system classifies the user query to decide whether to toggle retriever mode on or off. 
If retriever mode is on, the user query then goes through a RAG Fusion pipeline to retrieve the most suitable resumes. 
Otherwise, the chatbot will simply answer the user query based on its knowledge and the chat history.

**2, What if I want to set a custom RAG Top-k?**

You can choose your preference LLM model, but not 'RAG Top-k'.
As an experimental version, we see that fixed 'RAG Top-k' to 5 is best for displaying on Streamlit UI.

**3, How can I use my resume(s)?**

You can upload your PDF resume(s) to replace our default resume database by using the 'Upload' button. 
(Note that it might take some time to index your data.)
"""
ABOUT_MESSAGE = """
### About 🎊

This project serve as a capstone project of HUST - IT4772E (NLP) subject.
"""
API_KEY_INFO = "Please add your Huggingface API key to continue. For more information, please refer to this [page](https://huggingface.co/docs/hub/security-tokens)."
API_KEY_ERROR = "The API key is invalid or expired! For more information, please refer to this [page](https://huggingface.co/docs/hub/security-tokens)."
NO_INDEX_WARNING = "No document index available. Please upload a resume or check if the documents.pkl file is available."
