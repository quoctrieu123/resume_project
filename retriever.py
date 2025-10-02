#retriever.py: file này dùng để định nghĩa các hàm và lớp liên quan đến việc truy xuất tài liệu từ chỉ mục (index)
import xml.etree.ElementTree as ET # thư viện để phân tích cú pháp XML
from typing import List # thư viện để sử dụng kiểu List
from pydantic import BaseModel, Field #thư viện để định nghĩa các mô hình dữ liệu
from langchain.agents import tool #tạo các tools để sử dụng trong các agents
from langchain_core.prompts.chat import ChatPromptTemplate #tạo prompt
from transformers import pipeline #sử dụng các mô hình đã được huấn luyện trước

from llama_index.core import Document #Document: object đại diện cho một document
from config import RAG_K_THRESHOLD, QUERY_CLASSIFIER_MODEL # các hằng số cấu hình từ file config.py

#thông thường, một document có thể được chia thành nhiều nodes để dễ dàng quản lý và truy xuất
def get_document_nodes(index, doc_id: str) -> List[Document]: #hàm này lấy tất cả các nodes từ một document dựa trên ID của nó
    """Get all nodes from a document by its ID"""
    # Get all nodes from the index
    all_nodes = index.docstore.docs.values() #lấy tất cả các nodes  từ chỉ mục

    # Filter nodes that belong to the specified document
    doc_nodes = [node for node in all_nodes if node.ref_doc_id == doc_id] #chỉ lấy các nodes có ref_doc_id trùng với doc_id

    if not doc_nodes:
        print(f"No nodes found for document ID: {doc_id}") #nếu không tìm thấy node nào, in ra thông báo và trả về None
        return None

    print(f"Found {len(doc_nodes)} nodes for document ID: {doc_id}")
    return doc_nodes


class ResumeID(BaseModel):
    """List of applicant IDs to retrieve resumes for"""
    id_list: List[str] = Field(description="List of applicant IDs to retrieve resumes for") #ResumeID có định dạng là một danh sách các ID ứng viên


class JobDescription(BaseModel):
    """Description of a job to retrieve similar resumes for"""
    job_description: str = Field(description="Description of a job to retrieve similar resumes for") #JobDescription có định dạng là một chuỗi mô tả công việc


class Retriever: #truy xuất documents từ index
    def __init__(self, index): #index: chỉ mục chứa các documents
        self.index = index

    def __reciprocal_rank_fusion__(self, document_rank_list: list[dict], k=50): #Recipal Rank Fusion (RRF): xếp hạng lại các tài liệu được truy xuất dựa trên truy vấn con
        """Implement reciprocal rank fusion for re-ranking results"""
        fused_scores = {}
        for doc_list in document_rank_list:
            for rank, (doc, _) in enumerate(doc_list.items()):
                if doc not in fused_scores:
                    fused_scores[doc] = 0
                fused_scores[doc] += 1 / (rank + k)
        reranked_results = {doc: score for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}
        return reranked_results #rerank lại các docuemnts trong document_rank_list và trả về một dictionary với doc_id và điểm số đã được xếp hạng lại

    def __retrieve_docs_id__(self, question: str, k=50): #truy xuất các documents IDs dựa trên một truy vấn
        """Retrieve document IDs based on a query"""
        retriever = self.index.as_retriever(similarity_top_k=k)
        results = retriever.retrieve(question)
        docs_score = {str(res.node.ref_doc_id): res.score for res in results}
        return docs_score

    def retrieve_id_and_rerank(self, subquestion_list: list): #truy xuất và xếp hạng lại các documents dựa trên nhiều truy vấn con
        """Retrieve and rerank documents based on multiple subqueries"""
        document_rank_list = []
        for subquestion in subquestion_list:
            document_rank_list.append(self.__retrieve_docs_id__(subquestion, RAG_K_THRESHOLD))
        reranked_documents = self.__reciprocal_rank_fusion__(document_rank_list)
        return reranked_documents


class ResumeRetriever(Retriever): #ResumeRetriever kế thừa từ lớp Retriever và thêm các chức năng cụ thể để truy xuất hồ sơ ứng viên dựa trên ID ứng viên hoặc mô tả công việc
    def __init__(self, index):
        super(ResumeRetriever, self).__init__(index)

        #định nghĩa prompt cho LLM
        #prompt này hướng dẫn LLM quyết định gọi tools dựa trên các loại truy vấn khác nhau không và trả lại final answer hoặc tool call trong định dạng XML
        #cần trả về định dạng XML tại vì nó dễ dàng để phân tích cú pháp và trích xuất thông tin cần thiết  
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in talent acquisition. Respond with an XML string in the following format:
            <response>
                <type>final_answer|tool_call</type>
                <tool_name>retrieve_applicant_id|retrieve_applicant_jd|null</tool_name>
                <tool_input>...</tool_input>
                <output>...</output>
            </response>

            IMPORTANT INSTRUCTIONS:
            1. For queries that ask for retrieve or find suitable resumes based on the given job descriptions, job requirements or job postings use:
               - <type>tool_call</type>
               - <tool_name>retrieve_applicant_jd</tool_name>
               - <tool_input>[FULL JOB DESCRIPTION]</tool_input>
            
            2. For queries that specifically mention applicant IDs or resume IDs, use:
               - <type>tool_call</type>
               - <tool_name>retrieve_applicant_id</tool_name>
               - <tool_input>[LIST OF IDS]</tool_input>
            
            3. For queries that don't require retrieve resume, use:
               - <type>final_answer</type>
               - <tool_name>null</tool_name>
               - <output>[YOUR ANSWER]</output>
            
            4. If you don't know the answer, just say that in the <output> field.
            
            5. NEVER respond without using this exact XML format."""),
            ("user", "{input}")
        ])
        #self.metadata: lưu trữ thông tin về loại truy vấn, đầu vào đã trích xuất, danh sách truy vấn con và các tài liệu đã truy xuất cùng với điểm số của chúng
        self.metadata = {
            "query_type": "no_retrieve",
            "extracted_input": "",
            "subqueries_list": [],
            "retrieved_docs_with_scores": {}
        }
    
    #hàm retrieve_docs: truy xuất tài liệu dựa trên loại câu hỏi (truy vấn)
    def retrieve_docs(self, question: str, llm):
        """Retrieve documents based on the question type"""

        #tool để truy xuất hồ sơ ứng viên dựa trên danh sách ID ứng viên
        #args_schema = ResumeID: định nghĩa kiểu dữ liệu đầu vào cho tool này
        @tool(args_schema=ResumeID) 
        def retrieve_applicant_id(id_list: list):
            """Retrieve resumes for applicants in the id_list"""
            retrieved_resumes = []

            for id_element in id_list:
                try:
                    resume_nodes = get_document_nodes(self.index, id_element)
                    file_name = resume_nodes[0].metadata["file_name"]
                    resume_with_id = "Applicant ID: " + id_element + " | File Name: " + file_name + "\n" + ' '.join(
                        [node.text for node in resume_nodes])
                    retrieved_resumes.append(resume_with_id)
                except Exception:
                    return []
            return retrieved_resumes


        #tool để truy xuất hồ sơ ứng viên dựa trên mô tả công việc
        @tool(args_schema=JobDescription)
        def retrieve_applicant_jd(job_description: str):
            """Retrieve similar resumes given a job description"""
            # Generate subqueries for RAG Fusion approach
            subqueries_list = [job_description]
            subqueries_list += llm.generate_subquestions(question)

            self.metadata["subqueries_list"] = subqueries_list
            retrieved_ids = self.retrieve_id_and_rerank(subqueries_list)
            self.metadata["retrieved_docs_with_scores"] = retrieved_ids

            # Retrieve documents with the IDs
            retrieved_resumes = []
            for doc_id in list(retrieved_ids.keys())[:RAG_K_THRESHOLD]:
                try:
                    resume_nodes = get_document_nodes(self.index, doc_id)
                    file_name = resume_nodes[0].metadata["file_name"]
                    resume_with_id = "Applicant ID: " + doc_id + " | File Name: " + file_name + "\n" + ' '.join(
                        [node.text for node in resume_nodes])
                    retrieved_resumes.append(resume_with_id)
                except Exception as e:
                    print(f"Error retrieving document {doc_id}: {str(e)}")

            return retrieved_resumes
        #ta không cần phải gắn tool vào llm vì llm chỉ có nhiệm vụ chỉ khi nào cần dùng tool và nó sẽ để hàm router thực thi tools

        #hàm router: phân tích phàn hồi llm => thực hiện tool nếu cần
        def router(res: str):
            try:
                # Parse XML response
                root = ET.fromstring(res.strip())
                response_type = root.find("type").text
                tool_name = root.find("tool_name").text
                tool_input = root.find("tool_input").text
                output = root.find("output").text

                if response_type == "final_answer":
                    return output or ""

                if response_type == "tool_call":
                    # Update metadata
                    self.metadata["query_type"] = tool_name
                    self.metadata["extracted_input"] = tool_input

                    # Map tools
                    toolbox = {
                        "retrieve_applicant_id": retrieve_applicant_id,
                        "retrieve_applicant_jd": retrieve_applicant_jd
                    }

                    if tool_name not in toolbox:
                        raise ValueError(f"Unknown tool: {tool_name}")

                    # Execute tool
                    return toolbox[tool_name].run(tool_input)

                raise ValueError("Invalid response type")

            except ET.ParseError:
                # Treat invalid XML as final answer
                return res
            except Exception as e:
                return f"Error: {str(e)}"

        # Gọi llm và trả lại router
        messages = self.prompt.format_messages(input=question)
        response = llm.llm.invoke(messages).content #gọi llm (nhận vào câu hỏi và trả về phản hồi xem có quyết định dùng tool hay không)
        return router(response) #trả về kết quả của hàm router
