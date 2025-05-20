from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough, RunnableSequence
from langchain_core.output_parsers import StrOutputParser

import streamlit as st

load_dotenv()

# Configure page
st.set_page_config(page_title="YouTube Q&A AI", layout="centered")

st.title("🎯 YouTube Q&A Assistant")
st.text("Ask questions from any YouTube video's transcript using Google Gemini and LangChain.")

# --- Sidebar Settings ---
with st.sidebar:
    st.header("⚙️ Settings")
    chunk_size = st.slider("Chunk Size", 500, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, step=50)
    top_k = st.slider("Top K Chunks to Retrieve", 1, 10, 4)
    st.markdown("---")
    st.info("Tip: Use short and specific questions for better results.")

# --- User Inputs ---
video_id = st.text_input("🎬 Enter YouTube Video ID (e.g., `dQw4w9WgXcQ`):")
question = st.text_input("❓ Ask your question about the video:")

if st.button("🚀 Submit"):
    if not video_id or not question:
        st.error("Please enter both a video ID and a question.")
    else:
        try:
            with st.spinner("🔍 Retrieving transcript and processing..."):
                # Step 1: Fetch transcript
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
                transcript = " ".join(chunk["text"] for chunk in transcript_list)

                # Optional: Show transcript
                with st.expander("📄 View Full Transcript"):
                    st.write(transcript)

                # Step 2: Split
                splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                chunks = splitter.create_documents([transcript])

                # Step 3: Embedding & Vector Store
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                vector_store = FAISS.from_documents(chunks, embeddings)
                retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": top_k})

                # Step 4: Prompt Template
                template = PromptTemplate(
                    template="""
                        You are a helpful assistant.
                        Answer ONLY from the provided transcript context.
                        If the context is insufficient, just say you don't know.
                        
                        Context:
                        {context}
                        
                        Question:
                        {question} """,
                    input_variables=["context", "question"]
                )

                # Step 5: Chain
                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)

                parallel_chain = RunnableParallel({
                    "context": RunnableSequence(retriever, RunnableLambda(format_docs)),
                    "question": RunnablePassthrough()
                })

                model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
                parser = StrOutputParser()

                final_chain = RunnableSequence(parallel_chain, template, model, parser)

                # Step 6: Generate answer
                response = final_chain.invoke(question)

            # Step 7: Display result
            st.subheader("🧠 Answer")
            st.markdown(response)

            st.balloons()
            st.success("Done!")

        except Exception as e:
            st.error(f"❌ Error: {e}")
