from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough, RunnableSequence
from langchain_core.output_parsers import StrOutputParser

import streamlit as st

load_dotenv()

st.title("Youtube Q&A Application")

video_id = st.text_input("Enter YouTube video id:")
question = st.text_input("Ask the Question:")

if st.button("Submit"):
    try:
        # Step 1a - Indexing (Document Ingestion)
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        # st.write(transcript)
        # st.success("Transcript Download Successful")

        # Step 1b - Indexing (Text Splitting)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript])
        # st.write(chunks)

        # Step 1c & 1d - Indexing (Embedding Generation and Storing in Vector Store)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_documents(chunks, embeddings)

        # Step 2 - Retrieval
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        # Step 3 - Augmentation
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

        template = PromptTemplate(template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.
        {context}
        Question: {question}
        """, input_variables=['context', 'question'])

        # retrieved_docs = retriever.invoke(question)

        # make context
        # context = "\n\n".join(doc.page_content for doc in retrieved_docs)

        # prompt = template.invoke({"context": context, "question": question})


        # Step 4: Text Generation
        # response = model.invoke(prompt)
        # st.write(response.content)

        def format_docs(retrieved_docs):
            context = "\n\n".join(doc.page_content for doc in retrieved_docs)
            return context

        # Chain Generation
        parallel_chain = RunnableParallel({
            "context": RunnableSequence(retriever, RunnableLambda(format_docs)),
            "question": RunnablePassthrough()
        })
        parser = StrOutputParser()

        final_chain =RunnableSequence(parallel_chain, template, model, parser)

        response = final_chain.invoke(question)

        st.write(response)

        st.success("Done")

    except Exception as e:
        st.warning(f"No captions available for this video. \n{e}.")