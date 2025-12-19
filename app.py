
import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from typing import List, Optional


from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


os.environ["GOOGLE_API_KEY"] ="AIzaSyCewBhcR00_HKv3B1JXUFJ8MbD-nR1COXQ"

load_dotenv()

APP_DIR = Path(__file__).parent
UPLOAD_DIR = APP_DIR / "uploads"
INDEX_DIR = APP_DIR / "course_index"
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
INDEX_DIR.mkdir(exist_ok=True, parents=True)


GEMINI_EMB_MODEL = "models/text-embedding-004"  
GEMINI_CHAT_MODEL = "gemini-2.0-flash"


COLL_CURRICULUM = "curriculum_gemini_768"
COLL_PAST = "past_subjects_gemini_768"


CHUNK_SIZE = 900
CHUNK_OVERLAP = 200

st.set_page_config(page_title="Gemini Course Recommender", layout="wide")
st.title(" Course & Elective Recommender — Gemini Only")

st.markdown("""
This app recommends **future subjects and electives** using the **Google Gemini** model.

### Upload:
-  **Past subjects (optional)** — courses already completed by the student  
-  **Curriculum / electives (recommended)** — syllabus or program guide  
""")


with st.sidebar:
    st.header(" Settings")
    persist_index = st.checkbox("Persist embeddings to disk", value=True)
    force_rebuild = st.checkbox("Force rebuild vector index", value=False)


col1, col2 = st.columns(2)
with col1:
    past_files = st.file_uploader("Upload Past Subjects PDF(s) (optional)", type=["pdf"], accept_multiple_files=True)
with col2:
    curriculum_files = st.file_uploader("Upload Curriculum / Electives PDF(s)", type=["pdf"], accept_multiple_files=True)

st.subheader(" Student Profile")
profile = st.text_area(
    "Describe the student's interests, strengths, and career goals (e.g., loves AI, prefers data analytics, avoid hardware-heavy subjects, etc.)",
    height=150,
)

def save_uploaded_files(files, subfolder: str) -> List[Path]:
    """Save uploaded files to uploads/<subfolder>/"""
    folder = UPLOAD_DIR / subfolder
    folder.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for file in files or []:
        dest = folder / file.name
        with open(dest, "wb") as f:
            f.write(file.getvalue())
        saved_paths.append(dest)
    return saved_paths


def split_pdfs(paths: List[Path]):
    """Split PDFs into text chunks."""
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    for p in paths:
        loader = PyPDFLoader(str(p))
        pages = loader.load()
        docs.extend(splitter.split_documents(pages))
    return docs


def build_or_open_index(embedder, collection_name, docs=None):
    """Create or open Chroma index for given document type."""
    try:
        vs = Chroma(persist_directory=str(INDEX_DIR), embedding_function=embedder, collection_name=collection_name)
        return vs
    except Exception:
        if docs:
            vs = Chroma.from_documents(docs, embedder, persist_directory=str(INDEX_DIR), collection_name=collection_name)
            if persist_index:
                vs.persist()
            return vs
        return None


def retrieve_docs(vs: Optional[Chroma], query: str, k: int = 6):
    """Retrieve top relevant documents."""
    if not vs:
        return []
    try:
        retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": k})
        try:
            return retriever.invoke(query)
        except Exception:
            return retriever.get_relevant_documents(query)
    except Exception:
        return []


def make_recommend_prompt(context_text: str, profile: str) -> str:
    """Prompt to generate course recommendations."""
    return (
        "You are an experienced academic advisor. Based on the syllabus and curriculum below, "
        "recommend 5 suitable subjects or electives for the student.\n"
        "For each recommendation include:\n"
        "1. Course name\n"
        "2. Why it fits the student's interests/goals\n"
        "3. Confidence level (Low/Medium/High)\n\n"
        f"Context:\n{context_text}\n\n"
        f"Student profile:\n{profile}\n\n"
        "Respond in clear bullet points."
    )


def make_explain_prompt(course: str, context: str, profile: str) -> str:
    """Prompt to explain why a specific course is recommended."""
    return (
        f"You are an academic advisor. The student asked why the following course is recommended:\n\n"
        f"Course: {course}\n\n"
        f"Supporting syllabus:\n{context}\n\n"
        f"Student profile:\n{profile}\n\n"
        "Explain briefly (3-6 sentences): how this course aligns with the student's background and goals, "
        "and what they will gain from it."
    )



try:
    embedder = GoogleGenerativeAIEmbeddings(model=GEMINI_EMB_MODEL)
    llm = ChatGoogleGenerativeAI(model=GEMINI_CHAT_MODEL)
except Exception as e:
    st.error(f"Failed to initialize Gemini models: {e}")
    st.stop()


saved_past = save_uploaded_files(past_files, "past")
saved_curriculum = save_uploaded_files(curriculum_files, "curriculum")


docs_past = split_pdfs(saved_past) if saved_past else []
docs_curriculum = split_pdfs(saved_curriculum) if saved_curriculum else []

st.info(f"Past: {len(saved_past)} file(s), Curriculum: {len(saved_curriculum)} file(s) uploaded.")


with st.spinner("Building / Loading Chroma vector indexes..."):
    vs_curriculum = build_or_open_index(embedder, COLL_CURRICULUM, docs_curriculum if (docs_curriculum and force_rebuild) else docs_curriculum)
    vs_past = build_or_open_index(embedder, COLL_PAST, docs_past if (docs_past and force_rebuild) else docs_past)



st.markdown("---")
st.subheader("Recommend Future Subjects & Electives")

if st.button("Generate Recommendations"):
    if not profile.strip():
        st.error("Please provide the student's interests and goals.")
        st.stop()

    with st.spinner("Retrieving syllabus context and generating recommendations..."):
       
        context_docs = []
        context_docs += retrieve_docs(vs_curriculum, profile, k=6)
        context_docs += retrieve_docs(vs_past, profile, k=4)
        snippets = [d.page_content[:1500] for d in context_docs if hasattr(d, "page_content")]

        context_text = "\n\n---\n\n".join(snippets) if snippets else "(no syllabus context found)"

        
        try:
            prompt = make_recommend_prompt(context_text, profile)
            resp = llm.invoke(prompt)
            result = getattr(resp, "content", None) or str(resp)
            st.markdown("##  Recommended Courses / Electives")
            st.write(result)
        except Exception as e:
            st.error(f"Gemini generation error: {e}")

       
        if snippets:
            with st.expander(" Show syllabus snippets used"):
                for i, s in enumerate(snippets[:6], 1):
                    st.markdown(f"**Snippet {i}:**\n\n{s[:800]}{'...' if len(s)>800 else ''}")


st.markdown("---")
st.subheader("Ask for Explanation")
course_name = st.text_input("Enter a recommended course name to explain why it suits the student:")
if st.button("Explain This Course"):
    if not course_name.strip():
        st.warning("Enter a course name first.")
    else:
        with st.spinner("Retrieving supporting syllabus and generating explanation..."):
            docs = retrieve_docs(vs_curriculum, course_name, k=6)
            context_text = "\n\n---\n\n".join([d.page_content[:1200] for d in docs])
            try:
                prompt = make_explain_prompt(course_name, context_text, profile)
                resp = llm.invoke(prompt)
                answer = getattr(resp, "content", None) or str(resp)
                st.markdown(f"###  Explanation for **{course_name}**")
                st.write(answer)
            except Exception as e:
                st.error(f"Gemini explanation error: {e}")
