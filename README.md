requirements:

python3 -m venv venv && \ source venv/bin/activate && \ pip install --upgrade pip && \ pip install torch transformers accelerate langchain faiss-cpu && \ python -c "from transformers import LlamaForCausalLM, LlamaTokenizer; \ model_name='YOUR_MODEL_NAME'; \ tokenizer = LlamaTokenizer.from_pretrained(model_name, cache_dir='./llama_model'); \ model = LlamaForCausalLM.from_pretrained(model_name, cache_dir='./llama_model')" && \ echo "Setup complete. Activate the venv with: source venv/bin/activate"
