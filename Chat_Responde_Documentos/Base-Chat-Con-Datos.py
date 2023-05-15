import os
from langchain import  PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
os.environ['TRANSFORMERS_CACHE'] = '/mnt/hdd/IA/Txt2Txt/modelos'
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig , pipeline
from langchain.llms import HuggingFacePipeline

from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import IFixitLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains import VectorDBQA
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings


import gradio as gr


import numpy as np

BASE_MODEL = "MBZUAI/LaMini-Flan-T5-783M"
LORA_WEIGHTS = ""
TOKENIZER = "MBZUAI/LaMini-Flan-T5-783M"

# BASE_MODEL = "bigscience/T0_3B"
# LORA_WEIGHTS = ""
# TOKENIZER = "bigscience/T0_3B"

device = "cpu"

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, model_max_length = 1024)
model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL,
    device_map={"": device},
)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=1024,
    temperature=0.6,
    top_p=0.95,
    device=-1,
    repetition_penalty= 1.2 
)

local_llm = HuggingFacePipeline(pipeline=pipe)


emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2",model_kwargs={'device': 'cpu'} )
    
loader = DirectoryLoader('./data', glob="**/*.txt")
documents = loader.load()
print (f"Found {len(documents)} document")
#print (f"{data.page_content}")
print (f"You have {len(documents)} document")
# Get your splitter ready
loaderIFixit = IFixitLoader("https://www.ifixit.com/Guide/Nintendo+DS+Lite+Disassembly/86279")
iFixitDoc = loaderIFixit.load()
#concatente document and data
documents = documents + iFixitDoc
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
# Split your docs into texts
texts = text_splitter.split_documents(documents)

# Embedd your texts
db = Chroma.from_documents(texts, emb, persist_directory="./data/index")
   
# Init your retriever. Asking for just 1 document back
#retriever = db.as_retriever()
conversation =  VectorDBQA.from_chain_type(llm=local_llm, chain_type="stuff", vectorstore=db)


def evaluate(
    instruction
):
    generation_output = conversation({"query": instruction})
    return generation_output["result"]


def add_text(history, text):
    history = history + [(text, None)]
    return history, ""

def add_file(history, file):
    history = history + [((file.name,), None)]
    return history

def bot(history):
    history[-1][1] = evaluate(history[-1][0])
    return history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot([], elem_id="chatbot").style(height=750)

    with gr.Row():
        with gr.Column(scale=0.85):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter, or upload an image",
            ).style(container=False)
        with gr.Column(scale=0.15, min_width=0):
            btn = gr.UploadButton("📁", file_types=["image", "video", "audio"])

    txt.submit(add_text, [chatbot, txt], [chatbot, txt]).then(
        bot, chatbot, chatbot
    )
    btn.upload(add_file, [chatbot, btn], [chatbot]).then(
        bot, chatbot, chatbot
    )

demo.launch()



