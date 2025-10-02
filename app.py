#code giúp build giao diện web bằng streamlit
import streamlit as st
import time #dùng để tính toán thời gian xử lý
import os
from streamlit_modal import Modal #modal để có thể hiển thị cửa sổ popup trong streamlit
from langchain_core.messages import AIMessage, HumanMessage #tạo prompt cho AI
from huggingface_hub.utils import HfHubHTTPError #xử lý lỗi từ huggingface
from huggingface_hub import HfApi #tương tác với huggingface hub

from config import (
    EMBEDDING_MODEL, EMBEDDING_DIM, DOCUMENTS_PATH, TEMP_UPLOAD_DIR,
    APP_TITLE, APP_ICON, WELCOME_MESSAGE, FAQ_MESSAGE, ABOUT_MESSAGE,
    API_KEY_INFO, API_KEY_ERROR, NO_INDEX_WARNING, DEFAULT_LLM_MODEL,
    AVAILABLE_LLM_MODELS, INDEX_PATH
) #import các biến cấu hình từ file config.py

from utils import load_documents_list, extract_doc, initialize_index, set_seed, load_index #import các hàm từ file utils.py
import chatbot as chatbot_verbosity #import module chatbot và đổi tên thành chatbot_verbosity để tránh trùng tên với biến chatbot trong file này
from chatbot import ChatBot #import class ChatBot từ file chatbot.py
from retriever import ResumeRetriever #import class ResumeRetriever từ file retriever.py
from llama_index.core import Document #set seed cho việc random

set_seed(42) #đặt seed để đảm bảo tính tái lập

# Ensure temp upload directory exists
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True) #tạo thư mục tạm để lưu file upload

# Set up the page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
) #tạo tên và ảnh của trang web
st.title(APP_TITLE)

#session state: dictionary global để lưu state (các biến quan trọng cần giữ) của các session trong streamlit
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [AIMessage(content=WELCOME_MESSAGE)] #nếu chưa có lịch sử chat => khởi tạo với tin nhắn chào mừng

if "embedding_model" not in st.session_state: #nếu chưa có model embedding trong session_state
    # Explicitly initialize the embedding model to avoid OpenAI default
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding #import model embedding từ llama_index
    from llama_index.core import Settings #import Settings từ llama_index để cấu hình
    
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL) #load model embedding
    Settings.embed_model = embed_model #cấu hình model embedding cho Settings
    
    st.session_state["embedding_model"] = EMBEDDING_MODEL #lưu model embedding vào session_state
    st.session_state["embedding_dim"] = EMBEDDING_DIM #lưu kích thước embedding vào session_state
    

if "documents" not in st.session_state: #nếu chưa có documents trong session_state
    try:
        document_list = load_documents_list(pkl_path=DOCUMENTS_PATH) #load danh sách documents từ file pickle
        st.session_state["documents"] = document_list #lưu documents vào session_state
        st.session_state["faiss_index"] = load_index(index_path=INDEX_PATH) #load faiss index từ file
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}") #hiển thị lỗi
        st.session_state["documents"] = [] #reset lại documents
        st.session_state["faiss_index"] = None

if "rag_pipeline" not in st.session_state and st.session_state.get("faiss_index") is not None: #nếu chưa có rag_pipeline trong session_state và faiss_index đã được load thành công
    st.session_state["rag_pipeline"] = ResumeRetriever(st.session_state["faiss_index"]) #khởi tạo rag_pipeline với faiss_index

if "cur_resume_list" not in st.session_state: #nếu chưa có cur_resume_list trong session_state
    st.session_state["cur_resume_list"] = [] #khởi tạo cur_resume_list rỗng để lưu trữ danh sách resume hiện tại


def upload_file(): #hàm này được gọi khi người dùng upload file
    modal = Modal(title="Upload File Error", key="upload_error", max_width=500) #tạo một modal để hiển thị lỗi khi upload file
    if st.session_state["uploaded_file"]: #nếu có file được upload
        try:
            # Create a unique filename in the temp directory
            filename = st.session_state['uploaded_file'].name #lấy tên file
            tmp_file = os.path.join(TEMP_UPLOAD_DIR, f"{filename}") #tạo đường dẫn tạm cho file
            
            # Save the uploaded file
            with open(tmp_file, "wb") as f:
                f.write(st.session_state["uploaded_file"].getvalue()) #lưu file vào đường dẫn tạm
        except Exception as e:
            with modal.container():
                st.markdown("Failed to upload your file! Here is the error:") #nếu lỗi khi upload file => hiển thị lỗi trong modal
                st.error(e)
        else:
            try:
                # Extract document text with memory-efficient processing
                uploaded_doc = extract_doc(tmp_file) #trích xuất văn bản từ file pdf
                
                # Clean up the temp file after extraction
                if os.path.exists(tmp_file):
                    os.remove(tmp_file) #xóa file tạm sau khi trích xuất xong
                    
            except Exception as e: #nếu trong quá trình trích xuất file có lỗi
                with modal.container():
                    st.markdown("Failed to extract your file! Here is the error:") #hiển thị lỗi trong modal
                    st.error(e)
                # Clean up on error too
                if os.path.exists(tmp_file): #nếu file tạm vẫn tồn tại => xóa nó đi
                    os.remove(tmp_file)
            else: #nếu không có lỗi nào xảy ra (try thực hiện được) => tiến hành tạo index với file được upload
                with st.toast('Indexing your file, ready in a moment...'):
                    # Create document with metadata
                    document = Document(
                        text=uploaded_doc, 
                        metadata={"file_name": st.session_state["uploaded_file"].name}
                    )
                    
                    # Clear previous documents to free memory
                    if "documents" in st.session_state:
                        st.session_state["documents"] = None
                    if "faiss_index" in st.session_state:
                        st.session_state["faiss_index"] = None
                    
                    # Force garbage collection to free memory
                    import gc
                    gc.collect()
                    
                    # Create new index with the single document
                    st.session_state["documents"] = [document]
                    st.session_state["faiss_index"] = initialize_index(
                        docs=st.session_state["documents"],
                        model_name=st.session_state["embedding_model"],
                        model_dim=st.session_state["embedding_dim"],
                    )
                    st.session_state["rag_pipeline"] = ResumeRetriever(st.session_state["faiss_index"])
                    st.success('Successfully indexed your file!')
    else: #nếu không có file nào được upload
        try:
            # Load documents from the configured path
            documents = load_documents_list(pkl_path=DOCUMENTS_PATH) #load danh sách documents từ file pickle
            
            # clear các document và index trước đó để giải phóng bộ nhớq
            if "documents" in st.session_state:
                st.session_state["documents"] = None
            if "faiss_index" in st.session_state:
                st.session_state["faiss_index"] = None
                
            # Force garbage collection
            import gc
            gc.collect()
            
            # set lại documents và index trong session_state
            st.session_state["documents"] = documents #lưu documents vào session_state
            st.session_state["faiss_index"] = load_index(index_path=INDEX_PATH) #lưu faiss index vào session_state
            st.session_state["rag_pipeline"] = ResumeRetriever(st.session_state["faiss_index"]) #khởi tạo ResumeRetriever với faiss_index của các documents
        except Exception as e:
            st.error(f"Error loading documents: {str(e)}") #nếu có lỗi khi load documents => hiển thị lỗi

# Kiểm tra tính hợp lệ của Huggingface API key. Trả về True nếu hợp lệ, ngược lại trả về False
def check_hf_api(api_token: str):
    api = HfApi()
    try:
        user_info = api.whoami(token=api_token) #whoami: lấy thông tin người dùng dựa trên token
        return True 
    except HfHubHTTPError as e:
        return False

#hàm để xóa lịch sử chat và danh sách resume hiện tại
def clear_message():
    st.session_state["cur_resume_list"] = []
    st.session_state["chat_history"] = [AIMessage(content=WELCOME_MESSAGE)]


# Sidebar configuration
with st.sidebar: #tạo một sidebar trong giao diện web
    st.markdown("# Configuration ⚙️")

    st.text_input("Huggingface API Key", type="password", key="api_key") #input để nhập hugging face api key vào
    st.selectbox( #tạo một dropdown để chọn model llm
        label="LLM Model",
        options=AVAILABLE_LLM_MODELS,
        placeholder=DEFAULT_LLM_MODEL,
        key="llm_selection",
    )
    st.checkbox("Fine-tune Version", key="finetune") #tạo một checkbox để chọn có dùng model đã fine-tune hay không
    st.file_uploader("Upload resumes", type=["pdf"], key="uploaded_file", on_change=upload_file) #tạo một nút để upload file pdf, on change: khi có file được upload sẽ gọi hàm upload_file
    st.button("Clear conversation", on_click=clear_message)

    st.divider() #tạo một kẻ ngang
    st.markdown(FAQ_MESSAGE) #hiển thị câu hỏi thường gặp (FAQ)

    st.divider() #tạo một kẻ ngang
    st.markdown(ABOUT_MESSAGE) #hiển thị thông tin về dự án

# tạo một input để nhập câu hỏi người dùng
user_query = st.chat_input("Type your message here...")

# hiển thị message
for message in st.session_state["chat_history"]: #đi qua từng message trong lịch sử chat
    if isinstance(message, AIMessage): #nếu message là của AI
        with st.chat_message("AI"):
            st.write(message.content) #hiển thị nội dung message
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"): #hiển thị message của người dùng
            st.write(message.content)
    else:
        with st.chat_message("AI"): 
            message[0].render(*message[1:])

# API key validation
if not st.session_state["api_key"]: #nếu chưa có api key trong session_state
    st.info(API_KEY_INFO) #hiển thị thông báo yêu cầu nhập api key
    st.stop()

if not check_hf_api(st.session_state["api_key"]): #nếu api key không hợp lệ
    st.error(API_KEY_ERROR) #hiển thị lỗi api key không hợp lệ
    st.stop()

# Main chat logic
if st.session_state["faiss_index"]: #nếu faiss_index đã được load thành công
    #khởi tạo resume_retriver và chatbot
    retriever = st.session_state["rag_pipeline"]
    chatbot = ChatBot(
        path=st.session_state["llm_selection"],
        fine_tune=st.session_state["finetune"]
    ) #khởi tạo chatbot

    # Flow xử lý câu hỏi người dùng
    if user_query is not None and user_query != "": #neu có câu hỏi từ người dùng
        with st.chat_message("Human"): #hiển thị message của người dùng
            st.markdown(user_query)
            st.session_state["chat_history"].append(HumanMessage(content=user_query)) #lưu message của người dùng vào lịch sử chat

        with st.chat_message("AI"):
            start = time.time()
            with st.spinner("Thinking..."): #hiển thị spinner với chữ "Thinking..." trong khi chatbot đang xử lý
                document_list = retriever.retrieve_docs(user_query, chatbot) #retrieve docs dựa trên câu hỏi người dùng và llm trong chatbot
                query_type = retriever.metadata["query_type"] #lấy loại truy vấn từ metadata của retriever
                st.session_state["cur_resume_list"] = document_list #lưu dánh sách cv được retrieve vào cur_resume_list trong session_state
                stream_message = chatbot.generate_message_stream(
                    question=user_query,
                    docs=document_list,
                    history=st.session_state["chat_history"],
                    prompt_cls=query_type
                ) #tạo một luồng phản hồi từ chatbot dựa trên câu hỏi, tài liệu được retrieve, lịch sử chat và loại prompt
            end = time.time()

            response = st.write_stream(stream_message) #hiển thị luồng phản hồi từ chatbot

            retriever_message = chatbot_verbosity #biến retriever_message để hiển thị thông tin về việc retrieve tài liệu
            retriever_message.render(document_list, retriever.metadata, float(end - start)) #hiển thị thông tin về việc retrieve tài liệu
            
            #lưu lịch sử chat
            st.session_state["chat_history"].append(AIMessage(content=response))
            #lưu thông tin về việc retrieve tài liệu vào lịch sử chat
            st.session_state["chat_history"].append((retriever_message, document_list, retriever.metadata, end - start))
else:
    st.warning(NO_INDEX_WARNING)
