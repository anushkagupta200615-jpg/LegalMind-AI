import streamlit as st
import tempfile
import os

from agents.orchestrator import run_legalmind

st.set_page_config(title='LegalMind AI', page_icon='■', layout='wide')

st.title('LegalMind AI')
st.subheader('Autonomous Legal Research, Analysis & Drafting')

with st.sidebar:
    st.selectbox('Mode', ['Analyze Legal Issue', 'Research Indian Law', 'Draft Legal Notice', 'Summarize Contract', 'Reply to Notice'], key='mode')
    uploaded_file = st.file_uploader('Upload Document (optional)', type=['pdf'])
    
    api_key = st.text_input('OpenAI API Key', type='password')
    tavily_key = st.text_input('Tavily API Key', type='password')
    
    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key
    if tavily_key:
        os.environ['TAVILY_API_KEY'] = tavily_key

user_query = st.text_area('Enter your query or issue description here:')

if st.button('Run LegalMind'):
    if not os.environ.get('OPENAI_API_KEY'):
        st.error('Please enter your OpenAI API key in the sidebar or .env file.')
    else:
        file_path = None
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                tmp.write(uploaded_file.read())
                file_path = tmp.name
        
        with st.spinner('Agents working...'):
            try:
                # Based on the selected mode, we can optionally prefix the query to guide the classifier
                prefix = f"[{st.session_state.mode}] "
                result = run_legalmind(prefix + user_query, file_path)
                st.success(result)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            finally:
                if file_path and os.path.exists(file_path):
                    os.unlink(file_path)

with st.expander('How it works'):
    st.markdown("""
    **5-Agent Architecture**:
    1. **Orchestrator Agent**: Plans task sequence and routes user input.
    2. **Document Agent**: Parses uploaded PDFs and extracts text.
    3. **Research Agent**: Uses web search to find relevant case laws.
    4. **Analysis Agent**: Matches extracted text against local vector DB of Bare Acts.
    5. **Drafting Agent**: Auto-generates legal notices and replies using templates.
    """)
