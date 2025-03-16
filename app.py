# import os
# import streamlit as st
# from langchain_groq import ChatGroq
# from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
# from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
# from langchain.agents import initialize_agent, AgentType
# from langchain.callbacks import StreamlitCallbackHandler
# from dotenv import load_dotenv
# load_dotenv()

# # prepairing tools
# wiki_wrap = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
# wiki = WikipediaQueryRun(api_wrapper=wiki_wrap)
# arxiv_wrap = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
# arxiv = ArxivQueryRun(api_wrapper=arxiv_wrap)
# search = DuckDuckGoSearchRun(name='Search')
# tools = [wiki, arxiv, search]

# # UI App
# st.title('The Surf: An AI-Agent With Search')

# # Sidebar 
# st.sidebar.title("Settings")
# api_key = st.sidebar.text_input("Enter your GROQ API key", type='password')

# if 'messages' not in st.session_state:
#     st.session_state['messages']=[{
#         'role': 'assistant',
#         'content':'Hi,Im a chatbot who can search the web. How can I help you?'
#     }]

# for msg in st.session_state.messages:
#     st.chat_message(msg['role']).write(msg['content'])


# if prompt:= st.chat_input(placeholder='What is machine learning...?'):
#     st.session_state.messages.append({'role':'user','content':prompt})
#     st.chat_message('user').write(prompt)

#     if not api_key:
#         st.warning('Please enter your GROQ API Key in the sidebar.')
#         st.stop()
#     llm = ChatGroq(groq_api_key=api_key, model_name='Llama3-8b-8192',streaming=True)
    
#     search_agent = initialize_agent(
#         tools, 
#         llm, 
#         agent=AgentType.OPENAI_FUNCTIONS, 
#         handling_parsing_errors=True)
    
#     with st.chat_message('assistant'):
#         st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
#         response=search_agent.invoke(st.session_state.messages,callbacks=[st_cb])
#         st.session_state.messages.append({'role':'assistant',"content":response})
#         st.write(response)


import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv

load_dotenv()

# Preparing tools
wiki_wrap = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrap)
arxiv_wrap = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrap)
search = DuckDuckGoSearchRun(name='Search')
tools = [wiki, arxiv, search]

# UI App
st.title('The Surf: An AI-Agent With Search')

# Sidebar 
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your GROQ API key", type='password')

# Check if API key is provided
if not api_key:
    st.warning("Please enter your GROQ API key in the sidebar.")
    st.stop()

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state['messages'] = [{
        'role': 'assistant',
        'content': 'Hi, I'm a chatbot who can search the web. How can I help you?'
    }]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

# Handling user input
if prompt := st.chat_input(placeholder='What is machine learning...?'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    st.chat_message('user').write(prompt)
    
    # Initialize LLM
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name='Llama3-8b-8192',
        streaming=True
    )
    
    # Initialize search agent
    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        handle_parsing_errors=True
    )
    
    # Generate response
    with st.chat_message('assistant'):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        try:
            # Fix 1: Pass only the prompt instead of the entire message history
            response = search_agent.invoke(
                {"input": prompt},
                callbacks=[st_cb]
            )
            
            # Fix 2: Extract the actual response content
            response_content = response.get('output', response.get('response', str(response)))
            
            # Fix 3: Store and display the response content as a string
            st.session_state.messages.append({
                'role': 'assistant',
                'content': str(response_content)
            })
            st.write(str(response_content))
            
        except Exception as e:
            error_message = f"Error: {str(e)}"
            st.error(error_message)
            st.session_state.messages.append({
                'role': 'assistant',
                'content': "Sorry, I couldn't process that request properly."
            })