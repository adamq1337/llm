import os
from langchain_community.document_loaders import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch

class ClinicalTrialsLoader:
    def __init__(self, file_path):
        self.loader = CSVLoader(file_path=file_path)
        
    def load_documents(self):
        # Load documents and enhance metadata
        documents = self.loader.load()
        for doc in documents:
            # Extract key metadata for easier searching
            metadata = {
                'nct_number': doc.metadata.get('NCT Number', ''),
                'study_title': doc.metadata.get('Study Title', ''),
                'conditions': doc.metadata.get('Conditions', ''),
                'study_status': doc.metadata.get('Study Status', ''),
                'study_type': doc.metadata.get('Study Type', '')
            }
            doc.metadata.update(metadata)
        return documents


# Load LLaMA Model
class LLaMAWrapper:
    def __init__(self, model_path):
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.model = LlamaForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )

    def __call__(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(inputs["input_ids"], max_length=200)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# Create Embedding and Retrieval System
def setup_clinical_trials_qa(documents, model_path):
    # Initialize embeddings (Sentence Transformers)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create vector store
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # Initialize LLaMA model
    llm = LLaMAWrapper(model_path)
    
    # Create retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
    )
    
    return qa_chain, vectorstore


# Interactive Chatbot
def clinical_trials_chatbot(file_path, model_path):
    # Load documents
    loader = ClinicalTrialsLoader(file_path)
    documents = loader.load_documents()
    
    # Setup QA chain
    qa_chain, vectorstore = setup_clinical_trials_qa(documents, model_path)
    
    # Interactive chat loop
    while True:
        query = input("Ask about clinical trials (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        
        # Get response
        response = qa_chain.run(query)
        print("\nResponse:", response)
        
        # Optional: Show source documents
        print("\nRelevant Studies:")
        results = vectorstore.similarity_search(query)
        for doc in results:
            print(f"- {doc.metadata['study_title']} (NCT: {doc.metadata['nct_number']})")


# Usage
if __name__ == "__main__":
    file_path = "metadata.csv"
    model_path = "/path/to/llama-3.1"
    clinical_trials_chatbot(file_path, model_path)
