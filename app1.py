import os, dotenv, base64
import streamlit as st
import asyncio
import edge_tts
from streamlit_mic_recorder import speech_to_text
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from langchain_cohere.embeddings import CohereEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import tempfile

# Load environment variables
dotenv.load_dotenv()

os.environ['GROQ_API_KEY'] = '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'
os.environ['GOOGLE_API_KEY'] = "AIzaSyB4bdbCaHraBKMqmnjkqfr_CPlF3UKmU90"     ## add your api key here
os.environ['COHERE_API_KEY'] = "k8PvagpN1xDFqXJ2mt7xbSrpNLPyigsSbLAiJNLh"
# Available voices for Text-to-Speech
voices = {
    "William":"en-AU-WilliamNeural",
    "James":"en-PH-JamesNeural",
    "Jenny":"en-US-JennyNeural",
    "US Guy":"en-US-GuyNeural",
    "Sawara":"hi-IN-SwaraNeural",
}

st.set_page_config(page_title="Skillup Academy ChatBot", layout="wide", page_icon="./assets/logo-.jpeg")

# Title
st.markdown("""
    <h1 style='text-align: center;'>
        <span style='color: #fcfcfc;'>SkillUp</span> 
        <span style='color: #00c220;'>Academy</span>
        <span style='color: #fcfcfc;'>Chatbot</span>
    </h1>
""", unsafe_allow_html=True)

# Streamlit setup
with st.sidebar:
    st.image("assets/logo.png", use_column_width=True)
    st.markdown("## Skillup Academy ChatBot")
    st.write("This bot can answer questions related to Skillup academy courses, outlines, and instructors.")
    st.divider()

# Load vectorstore only once
if "vectorstore" not in st.session_state:
    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
    st.session_state["vectorstore"] = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = [
        {"role":"assistant", "content":"Hey there! How can I assist you today?"}
    ]

def format_docs(docs):
    return "\n\n".join(
        [f'Document {i+1}:\n{doc.page_content}\n'
         f'Source: {doc.metadata.get("source", "Unknown")}\n'
         f'Category: {doc.metadata.get("category", "Unknown")}\n'
         f'Instructor: {doc.metadata.get("instructor", "N/A")}\n-------------'
         for i, doc in enumerate(docs)]
    )

# Reset conversation
def reset_conversation():
    st.session_state.pop('chat_history')
    st.session_state['chat_history'] = [
        {"role":"assistant", "content":"Hey there! How can I assist you today about sillkup academy?"}
    ]

def rag_qa_chain(question, retriever, chat_history):
    llm = ChatGroq(model="llama-3.1-70b-versatile")
    output_parser = StrOutputParser()

    # System prompt to contextualize the question
    contextualize_q_system_prompt = """Given a chat history and the latest user question which might reference context in the chat history,
    formulate a standalone question which can be understood without the chat history. If the original question is in Roman Urdu or Hingish language,
    then translate it to accurate English. Do NOT answer the question, just reformulate and translate it if needed and otherwise return it as is."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    
    contextualize_q_chain = contextualize_q_prompt | llm | output_parser

    qa_system_prompt =  """You are a helpful assistant for Q&A about 'Skillup Academy,' which offers the following courses:
    - Python Programming for Data Science
    - C++ Programming
    - AI and Machine Learning
    - Web Development (Frontend, Backend)
    - App Development (Kotlin)
    - Digital Marketing
    - Graphic Designing
    - English Speaking
    - Sketching
    - Office Automation

    The user will ask questions related to the academy's courses, course outlines, and instructors.
    Use the retrieved documents as context below to answer the user's question.If the user ask about the course fee give them the contect number of academy and ask them to contect on this number and get the information. If the information is not available in the context, politely apologize and inform the user that you don't know.
    If the question is irrelevant to the academy, politely steer the conversation back to the academy's courses. DO NOT answer irrelevant questions.

    Keep your responses concise and in English ONLY.

    Retrieved Documents (Context):
    ------------
    {context}
    ------------
    """
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    final_llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.5)
    rag_chain = (
        RunnablePassthrough.assign(
            context=contextualize_q_chain | retriever | format_docs
        )
        | prompt
        | final_llm
        | output_parser
    )
    
    return rag_chain.stream({"question": question, "chat_history": chat_history})

# Generate the speech from text
async def generate_speech(text, voice):
    communicate = edge_tts.Communicate(text, voice)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        await communicate.save(temp_file.name)
        temp_file_path = temp_file.name
    return temp_file_path

# Get audio player
def get_audio_player(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        return f'<audio autoplay="true" src="data:audio/mp3;base64,{b64}">'
        
# Text-to-Speech function which automatically plays the audio
def generate_voice(text, voice):
    text_to_speak = (text).translate(str.maketrans('', '', '#-*_😊👋😄😁🥳👍🤩😂😎')) # Removing special chars and emojis
    with st.spinner("Generating voice response..."):
        temp_file_path = asyncio.run(generate_speech(text_to_speak, voice)) 
        audio_player_html = get_audio_player(temp_file_path)  # Create an audio player
        st.markdown(audio_player_html, unsafe_allow_html=True)
        os.unlink(temp_file_path)

# Sidebar voice option selection
if st.sidebar.toggle("Enable Voice Response"):
    voice_option = st.sidebar.selectbox("Choose a voice for response:", options=list(voices.keys()), key="voice_response")

# Dividing the main interface into two parts
col1, col2 = st.columns([1, 5])

# Displaying chat history
for message in st.session_state.chat_history:
    avatar = "assets/user.png" if message["role"] == "user" else "assets/assistant.png"
    with col2:
        st.chat_message(message["role"], avatar=avatar).write(message["content"])


# Handle voice or text input
with col1:
    st.button("Reset", use_container_width=True, on_click=reset_conversation)

    with st.spinner("Converting speech to text..."):
        text = speech_to_text(language="ur", just_once=True, key="STT", use_container_width=True)


query = st.chat_input("Type your question")

# Generate the response
if text or query:
    col2.chat_message("user", avatar="assets/user.png").write(text if text else query)
    
    st.session_state.chat_history.append({"role": "user", "content": text if text else query})

    # Generate response
    with col2.chat_message("assistant", avatar="assets/assistant.png"):
        try:
            response = st.write_stream(rag_qa_chain(question=text if text else query,
                                retriever=st.session_state["vectorstore"].as_retriever(search_kwargs={"k": 6}),
                                chat_history=st.session_state.chat_history))
        
            # Add response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"An internal error occurred: {e}")

    # Generate voice response if the user has enabled it
    if "voice_response" in st.session_state and st.session_state.voice_response:
        response_voice = st.session_state.voice_response
        generate_voice(response, voices[response_voice])
