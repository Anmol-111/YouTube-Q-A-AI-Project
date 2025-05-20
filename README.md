<h1>🎥 YouTube Q&A Assistant using LangChain + Google Gemini</h1>

  <div class="section">
    <p>
      This is an interactive <strong>Streamlit</strong> web app that allows users to ask questions about the content of any YouTube video — powered by <strong>LangChain</strong>, <strong>Google Gemini</strong> (via Generative AI APIs), and <strong>FAISS</strong> vector search.
    </p>
    <p>
      💡 Just paste a YouTube video ID and ask a question. The app fetches the transcript, chunks it, embeds it, and retrieves the most relevant sections to answer your query accurately using <strong>RAG (Retrieval-Augmented Generation)</strong>.
    </p>
  </div>

  <div class="section">
    <h2>🚀 Features</h2>
    <ul>
      <li>✅ Extracts YouTube video transcripts via <code>youtube-transcript-api</code></li>
      <li>✅ Chunks and embeds transcript using <code>LangChain</code> + <code>GoogleGenerativeAIEmbeddings</code></li>
      <li>✅ Stores embeddings in <code>FAISS</code> vector store</li>
      <li>✅ Retrieves top-K relevant chunks</li>
      <li>✅ Answers questions using <strong>gemini-1.5-flash</strong> model</li>
      <li>✅ Interactive and modern <strong>Streamlit</strong> UI</li>
      <li>✅ Sidebar controls for chunking, retrieval, and overlap</li>
      <li>✅ Markdown-based output with 🎈 balloon animation on success</li>
    </ul>
  </div>

  <div class="section">
    <h2>🧰 Tech Stack</h2>
    <table>
      <tr>
        <th>Component</th>
        <th>Technology</th>
      </tr>
      <tr>
        <td>Frontend</td>
        <td>Streamlit</td>
      </tr>
      <tr>
        <td>LLM Backend</td>
        <td>Google Gemini via LangChain</td>
      </tr>
      <tr>
        <td>Embeddings</td>
        <td>Google Generative AI Embeddings</td>
      </tr>
      <tr>
        <td>Vector DB</td>
        <td>FAISS</td>
      </tr>
      <tr>
        <td>Prompting</td>
        <td>LangChain PromptTemplate</td>
      </tr>
      <tr>
        <td>Transcripts</td>
        <td>youtube-transcript-api</td>
      </tr>
    </table>
  </div>

  <div class="section">
    <h2>📄 License</h2>
    <p>MIT License © <strong>ANMOL PRAKASH</strong></p>
  </div>
