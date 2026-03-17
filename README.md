# Gemini Course Recommender

An AI-powered academic advisor that recommends future subjects and electives based on a student's past performance, interests, and the provided curriculum. Using Google's Gemini models (Gemini 2.0 Flash) and RAG (Retrieval-Augmented Generation) with ChromaDB, it provides personalized course suggestions with human-like explanations.

##  Key Features

- **Personalized Recommendations**: Suggests subjects based on student-specific interests, career goals, and past academic history.
- **RAG Architecture**: Uses PDF-based curriculum guides and syllabus documents to ground its recommendations in actual course data.
- **Explanation Engine**: Allows students to ask "Why this course?" and receive a contextualized justification.
- **Support for PDFs**: Upload past subject transcripts and new curriculum/program guides directly.
- **Vector Index Persistence**: Optionally saves embeddings to disk for faster subsequent loads.

##  Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **AI Framework**: [LangChain](https://www.langchain.com/)
- **Model Provider**: [Google Gemini (Generative AI)](https://ai.google.dev/)
- **Vector Database**: [ChromaDB](https://www.trychroma.com/)
- **Document Processing**: `pypdf`, `RecursiveCharacterTextSplitter`

##  Setup & Installation

### Prerequisites

- Python 3.9 or higher
- A Google API Key (Generate one at [Google AI Studio](https://aistudio.google.com/))

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/prasannad20011121/Course-recommendation.git
   cd Course-recommendation
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**:
   Create a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```
   *Alternatively, you can paste the API key directly in `app.py` at line 15 (less recommended).*

##  Usage

1. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```
2. **Upload Files**:
   - Upload PDFs of courses already completed (Past Subjects).
   - Upload the syllabus or program guide (Curriculum / Electives).
3. **Describe the Student**:
   - Enter career goals or specific interests (e.g., "Interested in Machine Learning, wants to work as a Data Scientist").
4. **Generate Recommendations**:
   - Click "Generate Recommendations" to see the top 5 suggested subjects.
5. **Ask for Explanations**:
   - Paste a recommended course name in the "Ask for Explanation" section to understand the rationale.

##  Project Structure

- `app.py`: Main application logic containing Streamlit UI and LangChain/Gemini integration.
- `requirements.txt`: List of Python dependencies.
- `uploads/`: Automated folder for temporary storage of uploaded PDFs.
- `course_index/`: Local directory for persisted ChromaDB vector stores.

##  How it Works (RAG Flow)

1. **Ingest**: PDF documents are loaded and split into text chunks.
2. **Embed**: Chunks are converted into vector embeddings using `text-embedding-004`.
3. **Store**: Vectors are stored in separate ChromaDB collections for "Past Subjects" and "Curriculum".
4. **Retrieve**: When a recommendation is requested, the system retrieves the most relevant snippets from the vector store based on the student's profile.
5. **Generate**: The retrieved context and student profile are passed to `gemini-2.0-flash` to generate the final recommendations.
