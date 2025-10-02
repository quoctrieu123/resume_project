import os
import json
import logging
import gc
import psutil
from typing import List, Tuple, Dict, Any
from datetime import datetime
from tqdm import tqdm
from pydantic import BaseModel, Field
from langchain_core.output_parsers.base import BaseOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from llama_index.core import Document

from config import DOMAINS, METADATA_MODEL, MEMORY_CHUNK_SIZE, MEMORY_LOGGING_INTERVAL
from utils import extract_doc
from chatbot import load_llm


class CustomMetadataParser(BaseOutputParser):
    def parse(self, text):
        """
        Parse text that contains schema information and extract the example data.
        """
        try:
            print(text)
            # Try to parse the text as JSON
            data = json.loads(text)
            return data
        except json.JSONDecodeError as e:
            # Handle cases where output isn't valid JSON
            raise ValueError(f"Failed to parse JSON: {e}")
        except Exception as e:
            raise ValueError(f"Failed to extract metadata: {e}")


def get_metadata_chain(model_name: str = METADATA_MODEL):
    # Get LLM model
    metadata_model = load_llm(model_name, temperature=0.0)

    # Use the custom parser
    metadata_parser = CustomMetadataParser()

    # Create a simplified prompt that just asks for the information
    metadata_template = """
    Extract information from this resume:
    {resume}

    IMPORTANT FORMATTING INSTRUCTIONS:
    1. Return ONLY valid JSON without any comments or explanations
    2. Do not include explanatory comments in the JSON
    3. Do not wrap the JSON in markdown code blocks
    4. The JSON must have the EXACT following structure with these exact field names:
      - yob: The person's year of birth as a string (default value: "")
      - domain: The person's main working domain and must be in {domains} (default value: "")
      - education: Array of education entries with institution, degree, gpa, and dates (default value: [])
      - experience: Array of work experiences with company, title, dates (default value: [])
      - skills: Array of skill strings (default value: [])
      - languages: Array of language strings (default value: [string of input text's language (e.g. "English")])

    Example of CORRECTLY formatted response:
    {{
      "yob": "1990",
      "domain": "Engineering",
      "education": [
        {{
          "institution": "Example University",
          "degree": "B.S. Computer Science",
          "gpa": "3.8/4.0",
          "dates": "2010-2014"
        }}
      ],
      "experience": [
        {{
          "company": "Example Corp",
          "title": "Software Engineer",
          "dates": "2014-2018",
        }}
      ],
      "skills": ["Python", "JavaScript"],
      "languages": ["English"]
    }}

    DO NOT include the schema information in your response, only the data.
    """

    metadata_prompt = PromptTemplate(
        template=metadata_template,
        input_variables=["resume"],
        partial_variables={"domains": DOMAINS},
    )

    metadata_chain = metadata_prompt | metadata_model | metadata_parser
    return metadata_chain


def extract_metadata_content(resume_path: str, metadata_model: str) -> Tuple[List[Document], List[str]]:
    """Extract metadata from resumes with memory-efficient batch processing"""
    # Chain for extracting metadata
    metadata_chain = get_metadata_chain(metadata_model)

    # Set up logging
    log_filename = f"resume_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    # Initialize result containers
    documents = []
    unprocessed_documents = []  # contain only file names

    # Get list of all resumes
    resumes = os.listdir(resume_path)
    total_resumes = len(resumes)
    logger.info(f"Starting extraction for {total_resumes} resumes")

    # Process in batches to manage memory
    batch_size = min(MEMORY_CHUNK_SIZE, total_resumes)
    num_batches = (total_resumes + batch_size - 1) // batch_size  # Ceiling division
    
    # Log initial memory usage
    mem_info = psutil.Process(os.getpid()).memory_info()
    logger.info(f"Initial memory usage: {mem_info.rss / (1024 * 1024):.2f} MB")
    
    # Process each batch
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_resumes)
        batch_resumes = resumes[start_idx:end_idx]
        
        logger.info(f"Processing batch {batch_idx + 1}/{num_batches} (items {start_idx+1}-{end_idx} of {total_resumes})")
        
        # Process each resume in the batch
        batch_documents = []
        loop = tqdm(enumerate(batch_resumes), desc=f"Batch {batch_idx+1}", total=len(batch_resumes), position=0, leave=False)
        
        for i, resume in loop:
            global_idx = start_idx + i  # Index in the full dataset
            try:
                # Get resume path
                cur_resume_path = os.path.join(resume_path, resume)
                logger.info(f"Processing [{global_idx + 1}/{total_resumes}]: {resume}")

                # Extract document text
                doc_content = extract_doc(cur_resume_path)
                logger.info(f"Extracted content from {resume}, length: {len(doc_content)} chars")

                # Extract metadata
                logger.info(f"Starting metadata extraction for {resume}")
                metadata = metadata_chain.invoke({"resume": doc_content})
                metadata["file_name"] = resume

                # Create document and add to batch
                logger.info(f"Successfully extracted metadata for {resume}")
                batch_documents.append(Document(text=doc_content, metadata=metadata))

                # Free memory for the document content
                doc_content = None
                
                # Log memory usage periodically
                if (i + 1) % MEMORY_LOGGING_INTERVAL == 0:
                    mem_info = psutil.Process(os.getpid()).memory_info()
                    logger.info(f"Memory usage after {global_idx + 1} documents: {mem_info.rss / (1024 * 1024):.2f} MB")
                    # Force garbage collection
                    gc.collect()

            except Exception as e:
                unprocessed_documents.append(resume)
                logger.error(f"Error processing {resume}: {str(e)}")
                # Continue with the next file rather than crashing
                continue
        
        # Add batch documents to the main list
        documents.extend(batch_documents)
        
        # Clear batch data and force garbage collection
        batch_documents = None
        gc.collect()
        
        # Log memory after batch
        mem_info = psutil.Process(os.getpid()).memory_info()
        logger.info(f"Memory usage after batch {batch_idx + 1}: {mem_info.rss / (1024 * 1024):.2f} MB")

    # Log completion
    logger.info(f"Completed processing {len(documents)} out of {total_resumes} resumes")
    logger.info(f"Failed to process {len(unprocessed_documents)} resumes")
    
    # Final garbage collection
    gc.collect()
    
    return documents, unprocessed_documents
