#file này dùng để sử dụng model
import torch
import streamlit as st
import numpy as np
import os
from config import RAG_K_THRESHOLD, LORA_PATHS
from peft import PeftConfig, PeftModel #thư viện peft để load model peft
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, pipeline #load các model trên hugging face
from transformers.utils.logging import set_verbosity_info, set_verbosity_error #set_verbosity_info để hiện lỗi chi tiết hơn
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline #load model hugging face
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage #laod các message
from huggingface_hub import login #login hugging face

# Set verbosity for more detailed error messages
set_verbosity_info()


def load_llm(path: str, temperature=0.1, max_new_tokens=2048, fine_tune=False): #load model dựa vào path của adapter và trả lại một chatgenerativemodel trong langchain
    """Load a Hugging Face language model with quantization"""
    # Login to Hugging Face Hub with API token if available
    if "api_key" in st.session_state and st.session_state["api_key"]: # nếu có api key được lưu trong session_state
        login(token=st.session_state["api_key"], write_permission=False) #đăng nhập vào hugging face với api key đó
        os.environ["HUGGINGFACE_TOKEN"] = st.session_state["api_key"] #lưu api key vào biến môi trường huggingface_token
    
    model, tokenizer = None, None
    if fine_tune: #nếu ta đặt fine_tune = True
        try:
            path_map = LORA_PATHS[path] #lấy đường dẫn weights từ config.py dựa vào path truyền vào (weigts này là adapter của model đã được fine-tune)
            config = PeftConfig.from_pretrained(path_map) #load adapters cho base model

            # Configure 4-bit quantization for better compatibility
            #sử dụng cấu hình 4-bit để load model nhẹ hơn
            quantization_config = BitsAndBytesConfig( #cấu hình 4-bit
                load_in_4bit=True, #tải model ở chế độ 4-bitq (int4)
                bnb_4bit_compute_dtype=torch.float16, #sử dụng kiểu dữ liệu float16 để tính toán
                bnb_4bit_use_double_quant=True, #sử dụng double quantization để giảm lỗi lượng tử hóa
                bnb_4bit_quant_type="nf4" #loại quantization nf4 (normal float 4-bit)
            )

            #load model gốc sử dụng quantization 
            base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path, #đường dẫn model gốc
                quantization_config=quantization_config, #cấu hình quantization
                device_map="auto", #tự động phân bố gpu/cpu
                torch_dtype=torch.float16,
                token=st.session_state.get("api_key", None) #lấy token là api key của người dùng được lưu trong session_state
            )

            # peftmodel: giúp gắn adapters vào base_model
            model = PeftModel.from_pretrained(base_model, path_map) 
            model = model.merge_and_unload() #hợp nhất trọng số của lora adapter vào base model và giải phóng bộ nhớ
            
            # load tokenizer của model
            tokenizer = AutoTokenizer.from_pretrained(path_map, token=st.session_state.get("api_key", None))
            if tokenizer.pad_token is None: #nếu tokenizer không có pad_token thì gán pad_token = eos_token
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            print(f"Error loading fine-tuned model: {e}")
            # nếu bị lỗi thì thử load model không sử dụng quantization
            try:
                base_model = AutoModelForCausalLM.from_pretrained(
                    config.base_model_name_or_path, #load base model được lưu trong config của adapter
                    device_map="auto",
                    torch_dtype=torch.float16,
                    token=st.session_state.get("api_key", None)
                )
                model = PeftModel.from_pretrained(base_model, path_map) #load adapter của model
                model = model.merge_and_unload() #merge adapter vào model
                tokenizer = AutoTokenizer.from_pretrained(path_map, token=st.session_state.get("api_key", None)) #load tokenizer của model
                if tokenizer.pad_token is None: #nếu tokenizer không có pad token thì gán pad token
                    tokenizer.pad_token = tokenizer.eos_token
                print("Loaded fine-tuned model without quantization")
            except Exception as fallback_error: #nếu vẫn báo lỗi thì trả lại modle
                print(f"Fallback loading for fine-tuned model also failed: {fallback_error}")
                raise RuntimeError(f"Failed to load fine-tuned model: {str(e)}. Fallback also failed: {str(fallback_error)}")

    else: #nếu không phải là fine-tune => sử dụng mô hình base
        # sử dụng quantization load model nhẹ hơn
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model = AutoModelForCausalLM.from_pretrained( #load model sử dụng quantization
                path,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                token=st.session_state.get("api_key", None)
            )
        except Exception as e: #nếu vẫn lỗi => load model không sử dụng quantization
            print(f"Error loading model with quantization: {e}")
            model = AutoModelForCausalLM.from_pretrained(
                path,
                device_map="auto",
                torch_dtype=torch.float16,
                token=st.session_state.get("api_key", None)
            )

        # Load tokenizer của model
        tokenizer = AutoTokenizer.from_pretrained(path, token=st.session_state.get("api_key", None))
        if tokenizer.pad_token is None: #thêm pad_token là eos token nếu pad token trống
            tokenizer.pad_token = tokenizer.eos_token

    # Pipeline configuration
    pipe = pipeline( #pipeline: giúp đơn giản hóa việc sử dụng model => có thể đưa trực tiếp vào input vào model và nhận được output luôn (không cần tokenizer)
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        return_full_text=False
    )

    llm = HuggingFacePipeline(pipeline=pipe) #wrapper của langchain giúp biến một pipeline trong huggingface sang llm của langchain (có thể dùng invoke như một llm bthg)
    # Explicitly set the model_id to ensure it's not None
    llm_model = ChatHuggingFace(llm=llm, model_id=path) #biến llm thành một  chatgenerativemodel
    return llm_model #trả lại một chatgenerativemodel trong langchain

#hàm render này sẽ hiển thị thông tin về việc retrieve tài liệu giúp ta hiểu hơn về hệ thống
def render(document_list: list, retriever_metadata: dict, time_elapsed: float):
    #st.expander: tạo một khung expander có thể thu gọn và mở rọng 
    retriever_message = st.expander(f"Verbosity") #tạo một khung có thể thu gọn và mở rộng
    #message_map: ánh xạ các loại truy vấn thành các mô tả hành động của hệ thống
    message_map = { #
        "retrieve_applicant_jd": "**A job description is detected**. The system defaults to using RAG.",
        "retrieve_applicant_id": "**Applicant IDs are provided**. The system defaults to using exact ID retrieval.",
        "no_retrieve": "**No retrieval is required for this task**. The system will utilize chat history to answer."
    }

    #hiển thị bên trong retriever_message
    with retriever_message:
        #hiển thị tổng thời gian
        st.markdown(f"Total time elapsed: {np.round(time_elapsed, 3)} seconds")
        #hiển thị mô tả hành động của hệ thống dựa vào loại truy vấn
        st.markdown(f"{message_map[retriever_metadata['query_type']]}")

        #nếu loại truy vấn là retrieve_applicant_jd
        if retriever_metadata["query_type"] == "retrieve_applicant_jd":
            st.markdown(f"Returning top {RAG_K_THRESHOLD} most similar resumes.")

            #tạo cột cho từng cv retrieve được dựa vào truy vấn => in ra từng cv cụ thể (hiện lên mỗi cột)
            button_columns = st.columns([float(1 / RAG_K_THRESHOLD) for _ in range(RAG_K_THRESHOLD)], gap="small")
            for index, document in enumerate(document_list[:RAG_K_THRESHOLD]):
                with button_columns[index], st.popover(f"Resume {index + 1}"):
                    st.markdown(document)

            #hiển thị thông tin chi tiết về truy vấn
            st.markdown(f"**Extracted query**:\n`{retriever_metadata['extracted_input']}`\n") #hiển thị thông tin được llm trích xuất từ truy vấn
            st.markdown(f"**Generated questions**:\n`{retriever_metadata['subqueries_list']}`") #hiển thị các truy vấn phụ tạo ra từ truy vấn chính
            st.markdown(f"**Document re-ranking scores**:\n`{retriever_metadata['retrieved_docs_with_scores']}`") #hiển thị điểm số của các cv được retrieve

        #nếu loại truy vấn là retrieve_applicant_id
        elif retriever_metadata["query_type"] == "retrieve_applicant_id":
            st.markdown(f"Using the ID to retrieve.")

            #tạo cột cho từng cv retrieve được dựa vào truy vấn => in ra từng cv cụ thể (hiện lên mỗi cột)
            button_columns = st.columns([float(1 / RAG_K_THRESHOLD) for _ in range(RAG_K_THRESHOLD)], gap="small")
            for index, document in enumerate(document_list[:RAG_K_THRESHOLD]):
                with button_columns[index], st.popover(f"Resume {index + 1}"):
                    st.markdown(document)

            st.markdown(f"**Extracted query**:\n`{retriever_metadata['extracted_input']}`\n")

#lớp chatbot: là một lớp để tạo một chatbot sử dụng llm đã load
class ChatBot():
    def __init__(self, path: str, fine_tune: bool = False):
        """Initialize the chatbot with a language model"""
        self.llm = load_llm(
            path=path,
            temperature=0.1,
            fine_tune=fine_tune
        ) #khởi tạo llm bằng hàm load_llm ở trên

    
    def generate_subquestions(self, question: str): #hàm generate subquestions: áp dụng multiquery retrieval để tách truy vấn chính thành các truy vấn phụ => giúp retrieve hiệu quả hơn (là một bước của rag fusion)
        """Generate subqueries from a job description to improve retrieval"""
        system_message = SystemMessage(content="""
            You are an expert in talent acquisition. Separate this job description into 3-4 more focused aspects for efficient resume retrieval.
            Make sure every single relevant aspect of the query is covered in at least one query. You may choose to remove irrelevant information that doesn't contribute to finding resumes such as the expected salary of the job, the ID of the job, the duration of the contract, etc.
            Only use the information provided in the initial query. Do not make up any requirements of your own.
            Put each result in one line, separated by a linebreak.
        """) #định nghĩa prompt cho hệ thống

        user_message = HumanMessage(content=f"""
            Generate 3 to 4 sub-queries based on this initial job description:
            {question}
        """) #định nghĩa prompt người dùng

        #oneshot example: ví dụ mẫu để huấn luyện zero-shot cho llm
        oneshot_example = HumanMessage(content="""
            Generate 3 to 4 sub-queries based on this initial job description:

            Wordpress Developer
            We are looking to hire a skilled WordPress Developer to design and implement attractive and functional websites and Portals for our Business and Clients. You will be responsible for both back-end and front-end development including the implementation of WordPress themes and plugins as well as site integration and security updates.
            To ensure success as a WordPress Developer, you should have in-depth knowledge of front-end programming languages, a good eye for aesthetics, and strong content management skills. Ultimately, a top-class WordPress Developer can create attractive, user-friendly websites that perfectly meet the design and functionality specifications of the client.
            WordPress Developer Responsibilities:
            Meeting with clients to discuss website design and function.
            Designing and building the website front-end.
            Creating the website architecture.
            Designing and managing the website back-end including database and server integration.
            Generating WordPress themes and plugins.
            Conducting website performance tests.
            Troubleshooting content issues.
            Conducting WordPress training with the client.
            Monitoring the performance of the live website.
            WordPress Developer Requirements:
            Bachelors degree in Computer Science or a similar field.
            Proven work experience as a WordPress Developer.
            Knowledge of front-end technologies including CSS3, JavaScript, HTML5, and jQuery.
            Knowledge of code versioning tools including Git, Mercurial, and SVN.
            Experience working with debugging tools such as Chrome Inspector and Firebug.
            Good understanding of website architecture and aesthetics.
            Ability to project manage.
            Good communication skills.
            Contract length: 12 months
            Expected Start Date: 9/11/2020
            Job Types: Full-time, Contract
            Salary: 12,004.00 - 38,614.00 per month
            Schedule:
            Flexible shift
            Experience:
            Wordpress: 3 years (Required)
            web designing: 2 years (Required)
            total work: 3 years (Required)
            Education:
            Bachelor's (Preferred)
            Work Remotely:
            Yes
        """)

        oneshot_response = AIMessage(content="""
            1. WordPress Developer Skills:
              - WordPress, front-end technologies (CSS3, JavaScript, HTML5, jQuery), debugging tools (Chrome Inspector, Firebug), code versioning tools (Git, Mercurial, SVN).
              - Required experience: 3 years in WordPress, 2 years in web designing.

            2. WordPress Developer Responsibilities:
              - Meeting with clients for website design discussions.
              - Designing website front-end and architecture.
              - Managing website back-end including database and server integration.
              - Generating WordPress themes and plugins.
              - Conducting website performance tests and troubleshooting content issues.
              - Conducting WordPress training with clients and monitoring live website performance.

            3. WordPress Developer Other Requirements:
              - Education requirement: Bachelor's degree in Computer Science or similar field.
              - Proven work experience as a WordPress Developer.
              - Good understanding of website architecture and aesthetics.
              - Ability to project manage and strong communication skills.

            4. Skills and Qualifications:
              - Degree in Computer Science or related field.
              - Proven experience in WordPress development.
              - Proficiency in front-end technologies and debugging tools.
              - Familiarity with code versioning tools.
              - Strong communication and project management abilities.
        """)

        response = self.llm.invoke([system_message, oneshot_example, oneshot_response, user_message]) #đưa prompt, oneshot vào llm để lấy phản hồi
        result = response.content.split("\n\n") #result là list các truy vấn phụ được tách ra từ phản hồi của llm
        return result #trả lại list các truy vấn phụ

    def generate_message_stream(self, question: str, docs: list, history: list, prompt_cls: str): #hàm generate_message_stream: tạo một luồng phản hồi dựa vào truy vấn, tài liệu được retrieve, lịch sử chat và loại prompt
        """Generate a streaming response to a user query with context from retrieved documents"""
        context = "\n\n".join(doc for doc in docs) #nối các tài liệu được retrieve thành một chuỗi ngăn cách bởi 2 dòng trống

        if prompt_cls == "retrieve_applicant_jd": #nếu loại truy vấn là retrieve_applicant_jd
            system_message = SystemMessage(content="""
              You are an expert in talent acquisition that helps determine the best candidate among multiple suitable resumes.
              Use the following pieces of context to determine the best resume given a job description.
              You should provide some detailed explanations for the best resume choice.
              Because there can be applicants with similar names, use the applicant ID to refer to resumes in your response.
              If you don't know the answer, just say that you don't know, do not try to make up an answer.
            """) #prompt bắt llm tìm ứng viên phù hợp nhất trong các cv được retrieve dựa vào mô tả công việc

            user_message = HumanMessage(content=f"""
              Chat history: {history}
              Context: {context}
              Question: {question}
            """) #đưa lịch sử chat, tài liệu được retrieve và truy vấn vào prompt người dùng
        
        #nếu loại truy vấn là retrieve applicant_id
        elif prompt_cls == "retrieve_applicant_id":
            system_message = SystemMessage(content="""
              You are an expert in talent acquisition that helps analyze any resumes.
              Use the following pieces of context to analyze the given resumes.
              You should provide some detailed explanations for any questions related to those resumes.
              If there are no questions about those resumes and the user only ask for retrival, you can just summarize the given resumes.
              Because there can be applicants with similar names, use the applicant ID to refer to resumes in your response.
              If you don't know the answer, just say that you don't know, do not try to make up an answer.
            """) #prompt bắt llm phân tích các cv được retrieve dựa vào id ứng viên

            user_message = HumanMessage(content=f"""
              Chat history: {history}
              Context: {context}
              Question: {question}
            """) #đưa lịch sử chat, tài liệu được retrieve và truy vấn vào prompt người dùng
        else: #nếu prompt không được xác định là 1 trong 2 loại trên => trả lời bất cứ câu hỏi nào liên quand dến job
            system_message = SystemMessage(content="""
              You are an expert in talent acquisition that can answer any job-related concepts, requirements or questions.
              You may use the following pieces of context to answer your question.
              You can only use chat history if the question mention about information in the chat history.
              In that case, do not mention in your response that you are provided with a chat history.
              If you don't know the answer, just say that you don't know, do not try to make up an answer.
            """)

            user_message = HumanMessage(content=f"""
              Chat history: {history}
              Context: {context}
              Question: {question}
            """)

        stream = self.llm.stream([system_message, user_message])
        return stream #trả lại một luồng phản hồi. Lưu ý: đây là một hàm sinh phản hổi cho một câu hỏi cụ thể chứ chưa phải là hàm sinh phản hồi cho cả cuộc hội thoại
    
