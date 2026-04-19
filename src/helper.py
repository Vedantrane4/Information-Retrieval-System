import os
import time

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_EMBEDDING_MODEL = os.getenv(
    "GOOGLE_EMBEDDING_MODEL",
    "gemini-embedding-001",
)
GOOGLE_CHAT_MODEL = os.getenv("GOOGLE_CHAT_MODEL", "gemini-2.5-flash")

if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


class RetryingGoogleEmbeddings(Embeddings):
    def __init__(self, model, max_retries=3, retry_delay=2):
        self.embeddings = GoogleGenerativeAIEmbeddings(model=model)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _run_with_retries(self, operation):
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return operation()
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        raise RuntimeError(f"Google embedding failed: {last_error}") from last_error

    def embed_documents(self, texts):
        vectors = []
        for text in texts:
            vectors.extend(
                self._run_with_retries(
                    lambda: self.embeddings.embed_documents([text], batch_size=1)
                )
            )
        return vectors

    def embed_query(self, text):
        return self._run_with_retries(lambda: self.embeddings.embed_query(text))


class ChatMemory:
    def __init__(self):
        self.messages = []


class ConversationMemory:
    def __init__(self):
        self.chat_memory = ChatMemory()


class ConversationalPdfChain:
    def __init__(self, vector_store):
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        self.llm = ChatGoogleGenerativeAI(model=GOOGLE_CHAT_MODEL, temperature=0.3)
        self.memory = ConversationMemory()

    def invoke(self, inputs):
        question = inputs["question"]
        docs = self.retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)

        history = "\n".join(
            f"{message.type}: {message.content}"
            for message in self.memory.chat_memory.messages
        )
        prompt = (
            "Answer the question using only the PDF context below. "
            "If the answer is not in the context, say you do not know.\n\n"
            f"Chat history:\n{history}\n\n"
            f"PDF context:\n{context}\n\n"
            f"Question: {question}"
        )

        response = self.llm.invoke(prompt)
        answer = response.content

        self.memory.chat_memory.messages.append(HumanMessage(content=question))
        self.memory.chat_memory.messages.append(AIMessage(content=answer))

        return {
            "answer": answer,
            "chat_history": self.memory.chat_memory.messages,
        }


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def get_text_chunks(text):
    if not text or not text.strip():
        return []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    return text_splitter.split_text(text)


def get_vector_store(text_chunks):
    if not text_chunks:
        raise ValueError("No text chunks were created from the uploaded PDFs.")

    embeddings = RetryingGoogleEmbeddings(model=GOOGLE_EMBEDDING_MODEL)
    embedded_chunks = embeddings.embed_documents(text_chunks)

    if len(embedded_chunks) != len(text_chunks):
        raise ValueError(
            "Embedding count mismatch: "
            f"created {len(embedded_chunks)} embeddings for {len(text_chunks)} chunks."
        )

    return FAISS.from_embeddings(
        text_embeddings=zip(text_chunks, embedded_chunks),
        embedding=embeddings,
    )


def get_conversational_chain(vector_store):
    return ConversationalPdfChain(vector_store)