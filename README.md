# YouTube-Q&A-AI-Project
🎥 YouTube Q&A Assistant using LangChain + Google Gemini
This is an interactive Streamlit web app that allows users to ask questions about the content of any YouTube video — powered by LangChain, Google Gemini (via Generative AI APIs), and FAISS vector search.

💡 Just paste a YouTube video ID and ask a question. The app fetches the transcript, chunks it, embeds it, and retrieves the most relevant sections to answer your query accurately using RAG (Retrieval-Augmented Generation).

🚀 Features<br>
✅ Extracts YouTube video transcripts via youtube-transcript-api

✅ Chunks and embeds transcript using LangChain + GoogleGenerativeAIEmbeddings

✅ Stores embeddings in FAISS vector store

✅ Retrieves top-K relevant chunks

✅ Answers questions using gemini-1.5-flash model

✅ Interactive and modern Streamlit UI

✅ Sidebar controls for chunking, retrieval, and overlap

✅ Markdown-based output with balloon animation on success

🧰 Tech Stack
Component	Technology
Frontend	Streamlit
LLM Backend	Google Gemini via LangChain
Embeddings	Google Generative AI Embeddings
Vector DB	FAISS
Prompting	LangChain PromptTemplate
Transcripts	youtube-transcript-apis

📄 License
MIT License © [ANMOL PRAKASH]
