import streamlit as st
import yaml
import google.generativeai as genai
import json
from xai_sdk import Client
from xai_sdk.chat import user, system, assistant
import copy
from datetime import datetime
import os
import pandas as pd
import io
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
from typing import Dict, List, Optional, Any
import uuid
import time
from pathlib import Path
from io import StringIO
from PyPDF2 import PdfReader


# --- CONFIGURATION ---
AGENTS_FILE = 'agents.yaml'
GEMINI_MODEL = 'gemini-2.5-flash'
GROK_MODEL = 'grok-3-mini'
# --- THEMES ---
THEMES = {
    "Deep Ocean": {
        "primary": "#0A2463", "secondary": "#3E92CC", "accent": "#1E96FC",
        "bg": "#001233", "text": "#E8F1F2", "card_bg": "#0D3B66",
        "gradient": "linear-gradient(135deg, #0A2463 0%, #1E96FC 100%)"
    },
    "Alps Forest": {
        "primary": "#2D4A2B", "secondary": "#5A8C5A", "accent": "#8BC34A",
        "bg": "#1A2F1A", "text": "#E8F5E9", "card_bg": "#2E5C2E",
        "gradient": "linear-gradient(135deg, #2D4A2B 0%, #8BC34A 100%)"
    },
    "Fendi Casa Luxury": {
        "primary": "#1C1C1C", "secondary": "#C9A96E", "accent": "#E8D5B7",
        "bg": "#0A0A0A", "text": "#F5F5DC", "card_bg": "#2D2D2D",
        "gradient": "linear-gradient(135deg, #1C1C1C 0%, #C9A96E 100%)"
    }
}

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Multi-Agent Analysis System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CONFIGURATION FUNCTIONS
# =============================================================================

def make_editable(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure dataframe is editable by st.data_editor (no MultiIndex)."""
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(c) for c in col]).strip() for col in df.columns.values]
    return df

def configure_gemini() -> Optional[genai.GenerativeModel]:
    """Configure Gemini API using environment variables or secrets."""
    api_key = None
    
    if hasattr(st, 'secrets') and 'GEMINI_API_KEY' in st.secrets:
        api_key = st.secrets['GEMINI_API_KEY']
    elif 'GEMINI_API_KEY' in os.environ:
        api_key = os.environ['GEMINI_API_KEY']
    elif 'gemini_api_key' in st.session_state:
        api_key = st.session_state.gemini_api_key
    
    if not api_key:
        return None
    
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(GEMINI_MODEL)
    except Exception as e:
        st.error(f"Failed to configure Gemini API: {e}")
        return None

def configure_grok() -> Optional[Client]:
    """Configure Grok/XAI API using environment variables or secrets."""
    api_key = None
    
    if hasattr(st, 'secrets') and 'GROK_API_KEY' in st.secrets:
        api_key = st.secrets['GROK_API_KEY']
    elif 'GROK_API_KEY' in os.environ:
        api_key = os.environ['GROK_API_KEY']
    elif 'grok_api_key' in st.session_state:
        api_key = st.session_state.grok_api_key
    
    if not api_key:
        return None
    
    try:
        return Client(api_key=api_key, timeout=3600)
    except Exception as e:
        st.error(f"Failed to configure Grok API: {e}")
        return None

@st.cache_data
def load_agent_config() -> Dict[str, Dict[str, Any]]:
    """Load the specialized AI agent configuration from agents.yaml."""
    if not os.path.exists(AGENTS_FILE):
        st.error(f"FATAL: {AGENTS_FILE} not found. Creating default configuration...")
        create_default_config()
    
    try:
        with open(AGENTS_FILE, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            squads = defaultdict(dict)
            
            for agent in config.get('agents', []):
                squad_name = agent.get('squad', 'Unassigned')
                agent_name = agent.get('name', agent.get('id', 'Unknown'))
                squads[squad_name][agent_name] = agent
            
            return dict(squads)
    except Exception as e:
        st.error(f"Failed to load {AGENTS_FILE}: {e}")
        st.stop()

def create_default_config():
    """Create a default agents.yaml with minimal configuration."""
    default_config = {
        'agents': [
            {
                'id': 'data_summarizer',
                'name': 'Data Summarizer',
                'squad': 'Data Analysis Team',
                'category': 'data_analysis',
                'description': 'Analyzes structured data and extracts key insights.',
                'system_prompt': 'You are a data analyst. Analyze the provided data and summarize key trends, patterns, and insights.',
                'params': {'temperature': 0.3, 'top_p': 0.9, 'max_output_tokens': 3072}
            }
        ]
    }
    
    with open(AGENTS_FILE, 'w', encoding='utf-8') as f:
        yaml.dump(default_config, f, allow_unicode=True)

# =============================================================================
# SESSION STATE
# =============================================================================

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    defaults = {
        'theme': 'Deep Ocean',
        'squads': {},
        'selected_squad': None,
        'selected_agent_name': None,
        'selected_agent_config': None,
        'gemini_client': None,
        'grok_client': None,
        'selected_model': None,
        'current_document': {'name': '', 'content': ''},
        'current_df': None,
        'datasets': {},
        'active_dataset_names': [],
        'analysis_result': None,
        'workflow_results': [],
        'comparison_results': [],
        'workflow_instruction': '',
        'workflow_plan': None,
        'workflow_agents': [],
        'workflow_history': [],
        'chat_history': [],
        'conversation_context': None,
        'document_analysis_result': None,  # Added for Tab 7
        'multi_dataset_result': None,  # Added for Tab 8
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# =============================================================================
# AGENT EXECUTION ENGINE
# =============================================================================

class AgentExecutor:
    """Handles agent execution with support for both Gemini and Grok."""
    
    def __init__(self, model_name: str, client: Any):
        self.model_name = model_name
        self.client = client
        self.is_grok = 'grok' in model_name.lower()
        self.is_gemini = 'gemini' in model_name.lower()
    
    def execute(self, agent_config: Dict, content: str, 
                context: Optional[List] = None) -> Dict:
        """Execute an agent with given content and optional conversation context."""
        
        if not self.client:
            return {'status': 'error', 'error': 'Client not initialized'}
        
        params = agent_config.get('params', {})
        temperature = params.get('temperature', 0.5)
        top_p = params.get('top_p', 0.9)
        max_tokens = params.get('max_output_tokens', 8192)
        
        try:
            if self.is_gemini:
                return self._execute_gemini(agent_config, content, temperature, 
                                           top_p, max_tokens)
            elif self.is_grok:
                return self._execute_grok(agent_config, content, temperature, 
                                         top_p, max_tokens, context)
            else:
                return {'status': 'error', 'error': 'Unknown model type'}
                
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _execute_gemini(self, agent_config: Dict, content: str,
                       temperature: float, top_p: float, 
                       max_tokens: int) -> Dict:
        """Execute using Gemini API."""
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens
        )
        
        system_prompt = agent_config.get('system_prompt', '')
        prompt = f"""**Role:** {agent_config.get('name', 'Agent')}
**System Instructions:** {system_prompt}
**Task:** Analyze the following content.

**Content:**
---
{content}
---"""
        
        response = self.client.generate_content(prompt, 
                                               generation_config=generation_config)
        
        return {
            'status': 'success',
            'result': response.text,
            'agent_name': agent_config.get('name'),
            'agent_id': agent_config.get('id'),
            'squad': agent_config.get('squad'),
            'model': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'params': agent_config.get('params', {})
        }
    
    def _execute_grok(self, agent_config: Dict, content: str,
                     temperature: float, top_p: float, max_tokens: int,
                     context: Optional[List] = None) -> Dict:
        """Execute using Grok/XAI SDK with conversation context support."""
        
        chat = self.client.chat.create(model=self.model_name)
        
        system_prompt = agent_config.get('system_prompt', '')
        if system_prompt:
            chat.append(system(system_prompt))
        
        if context:
            for msg in context:
                if msg['role'] == 'user':
                    chat.append(user(msg['content']))
                elif msg['role'] == 'assistant':
                    chat.append(assistant(msg['content']))
        
        chat.append(user(f"--- Content to Analyze ---\n\n{content}"))
        
        response = chat.sample()
        
        return {
            'status': 'success',
            'result': response.content,
            'agent_name': agent_config.get('name'),
            'agent_id': agent_config.get('id'),
            'squad': agent_config.get('squad'),
            'model': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'params': agent_config.get('params', {}),
            'chat_context': chat
        }

# =============================================================================
# WORKFLOW ORCHESTRATION
# =============================================================================

class WorkflowOrchestrator:
    """Orchestrates multi-agent workflows with context preservation."""
    
    def __init__(self, executor: AgentExecutor):
        self.executor = executor
        self.results = []
        self.context = []
    
    def execute_workflow(self, agents: List[Dict], initial_content: str,
                        progress_callback=None) -> List[Dict]:
        """Execute a sequence of agents, passing results between them."""
        
        current_content = initial_content
        self.results = []
        self.context = []
        
        for idx, agent in enumerate(agents):
            if progress_callback:
                progress_callback(idx, len(agents), agent['name'])
            
            result = self.executor.execute(
                agent_config=agent,
                content=current_content,
                context=self.context
            )
            
            self.results.append(result)
            
            if result['status'] == 'success':
                self.context.append({
                    'role': 'user',
                    'content': current_content
                })
                self.context.append({
                    'role': 'assistant',
                    'content': result['result']
                })
                
                current_content = result['result']
            else:
                break
        
        return self.results

# =============================================================================
# UI RENDERING
# =============================================================================

def apply_theme():
    """Apply the selected theme CSS."""
    theme = THEMES[st.session_state.theme]
    st.markdown(f"""
    <style>
        :root {{
            --primary-color: {theme['primary']};
            --secondary-color: {theme['secondary']};
            --accent-color: {theme['accent']};
            --bg-color: {theme['bg']};
            --text-color: {theme['text']};
            --card-bg: {theme['card_bg']};
        }}
        
        .stApp {{
            background: {theme['gradient']};
            color: {theme['text']};
        }}
        
        .main .block-container {{
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }}
        
        .agent-card {{
            background: {theme['card_bg']};
            padding: 20px;
            border-radius: 15px;
            border: 2px solid {theme['secondary']};
            margin: 10px 0;
            transition: all 0.3s ease;
        }}
        
        .agent-card:hover {{
            transform: translateX(5px);
            border-color: {theme['accent']};
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        }}
        
        .metric-card {{
            background: {theme['card_bg']};
            padding: 20px;
            border-radius: 15px;
            border-left: 4px solid {theme['accent']};
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            margin: 10px 0;
        }}
    </style>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render the sidebar with configuration options."""
    with st.sidebar:
        st.header("🎨 Theme")
        selected_theme = st.selectbox(
            "Select Interface Theme",
            options=list(THEMES.keys()),
            index=list(THEMES.keys()).index(st.session_state.theme),
            key="theme_selector"
        )
        if selected_theme != st.session_state.theme:
            st.session_state.theme = selected_theme
            st.rerun()
        
        st.divider()
        st.header("🔑 API Configuration")
        
        gemini_client = configure_gemini()
        grok_client = configure_grok()
        
        if not gemini_client:
            gemini_key = st.text_input("Gemini API Key", type="password", 
                                      key="gemini_key_input")
            if gemini_key:
                st.session_state.gemini_api_key = gemini_key
                st.rerun()
        
        if not grok_client:
            grok_key = st.text_input("Grok API Key", type="password", 
                                    key="grok_key_input")
            if grok_key:
                st.session_state.grok_api_key = grok_key
                st.rerun()
        
        st.session_state.gemini_client = gemini_client
        st.session_state.grok_client = grok_client
        
        st.divider()
        st.header("🤖 Model Selection")
        
        available_models = []
        if gemini_client:
            available_models.append(GEMINI_MODEL)
        if grok_client:
            available_models.append(GROK_MODEL)
        
        if available_models:
            st.session_state.selected_model = st.selectbox(
                "Select Model",
                options=available_models
            )
        else:
            st.warning("Please configure at least one API key.")
            st.session_state.selected_model = None
        
        st.divider()
        st.header("📊 Agent Squads")
        
        squads = load_agent_config()
        st.session_state.squads = squads
        
        squad_names = list(squads.keys())
        if squad_names:
            selected_squad = st.selectbox(
                "Select Squad",
                options=squad_names,
                key="squad_selector"
            )
            st.session_state.selected_squad = selected_squad
            
            if selected_squad in squads:
                st.markdown(f"**Agents in {selected_squad}:**")
                for agent_name in squads[selected_squad].keys():
                    st.markdown(f"- {agent_name}")

def render_analysis_tab():
    """Render the main analysis tab."""
    st.header("🔬 Agent Analysis Workspace")
    
    if not st.session_state.selected_squad:
        st.warning("Please select a squad from the sidebar.")
        return
    
    squad = st.session_state.squads.get(st.session_state.selected_squad, {})
    agent_names = list(squad.keys())
    
    if not agent_names:
        st.warning("No agents found in selected squad.")
        return
    
    selected_agent_name = st.selectbox(
        "Select Agent",
        options=agent_names,
        key="agent_selector"
    )
    
    agent_config = squad[selected_agent_name]
    st.session_state.selected_agent_config = agent_config
    
    st.markdown(f"""
    <div class="metric-card">
        <h4>🤖 {agent_config['name']}</h4>
        <p><strong>Squad:</strong> {agent_config.get('squad', 'N/A')}</p>
        <p><strong>Category:</strong> {agent_config.get('category', 'N/A')}</p>
        <p><strong>Description:</strong> {agent_config.get('description', 'N/A')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 📄 Input")
        content = st.text_area("Enter content to analyze", height=300)
        
        if content:
            st.session_state.current_document['content'] = content
    
    with col2:
        st.markdown("#### ⚙️ Parameters")
        params = agent_config.get('params', {})
        
        temp = st.slider("Temperature", 0.0, 1.0, 
                        float(params.get('temperature', 0.5)), 0.05)
        top_p = st.slider("Top P", 0.0, 1.0, 
                         float(params.get('top_p', 0.9)), 0.05)
        max_tokens = st.number_input("Max Tokens", 1024, 32768,
                                     int(params.get('max_output_tokens', 8192)))
        
        agent_config['params'] = {
            'temperature': temp,
            'top_p': top_p,
            'max_output_tokens': max_tokens
        }
    
    if st.button("🚀 Execute Agent", type="primary", 
                disabled=not (content and st.session_state.selected_model)):
        
        client = (st.session_state.gemini_client 
                 if 'gemini' in st.session_state.selected_model 
                 else st.session_state.grok_client)
        
        executor = AgentExecutor(st.session_state.selected_model, client)
        
        with st.spinner(f"Agent '{agent_config['name']}' is analyzing..."):
            result = executor.execute(agent_config, content)
            st.session_state.analysis_result = result
            st.rerun()
    
    if st.session_state.analysis_result:
        st.divider()
        st.header("📊 Results")
        
        result = st.session_state.analysis_result
        if result['status'] == 'success':
            col1, col2, col3 = st.columns(3)
            col1.metric("Agent", result['agent_name'])
            col2.metric("Model", result['model'])
            col3.metric("Time", result['timestamp'].split('T')[1][:8])
            
            st.markdown(f"""
            <div class="metric-card">
                {result['result']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error(f"Error: {result['error']}")

# =============================================================================
# UTILITY FUNCTIONS FOR TABS 3-8
# =============================================================================

def parse_text_to_df(raw_text: str):
    """Try to parse pasted text into a DataFrame (json / csv / table)."""
    try:
        obj = json.loads(raw_text)
        if isinstance(obj, list):
            return pd.json_normalize(obj)
        elif isinstance(obj, dict):
            try:
                return pd.DataFrame(obj)
            except Exception:
                return pd.json_normalize(obj)
    except Exception:
        pass

    try:
        df = pd.read_csv(io.StringIO(raw_text))
        return df
    except Exception:
        pass

    try:
        df = pd.read_table(io.StringIO(raw_text))
        return df
    except Exception:
        pass

    return pd.DataFrame({"text": raw_text.splitlines()})

def load_uploaded_file_to_df(uploaded_file):
    """Load an uploaded file (txt, csv, json) into DataFrame."""
    if uploaded_file is None:
        return None
    content = uploaded_file.getvalue().decode("utf-8")
    try:
        return parse_text_to_df(content)
    except Exception:
        return pd.DataFrame({"text": content.splitlines()})

def df_to_json_pretty(df: pd.DataFrame) -> str:
    try:
        return df.to_json(orient="records", force_ascii=False, indent=2)
    except Exception:
        return json.dumps(df.to_dict(orient="records"), indent=2)

def json_text_to_df(json_text: str):
    try:
        parsed = json.loads(json_text)
        return pd.json_normalize(parsed)
    except Exception as e:
        try:
            return pd.read_json(io.StringIO(json_text))
        except Exception:
            raise e

def extract_text_from_pdf(file, pages=None):
    """Extract text from PDF file."""
    try:
        reader = PdfReader(file)
        text = ""
        if pages:
            for p in pages:
                if p < len(reader.pages):
                    text += reader.pages[p].extract_text() or ""
        else:
            for page in reader.pages:
                text += page.extract_text() or ""
        if not text.strip():
            return "[OCR required but not available in this environment]"
        return text
    except Exception as e:
        return f"Error extracting text: {e}"

# =============================================================================
# TAB 3: DATA ANALYSIS
# =============================================================================

def data_analysis_tab_ui():
    """Render Tab 3 - Data Analysis (single dataset flow)."""
    st.header("📊 Data Analysis")
    st.info("Upload or paste a dataset (txt/csv/json). Edit the table, convert to JSON, and run an agent for analysis.")

    with st.expander("📁 1) Upload or Paste Data", expanded=True):
        col1, col2 = st.columns([1, 1])
        with col1:
            uploaded = st.file_uploader("Upload file (csv/json/txt)", type=["csv", "json", "txt"], key="tab3_uploader")
        with col2:
            pasted = st.text_area("Or paste data here", height=160, key="tab3_pasted")

    df = None
    if uploaded:
        df = load_uploaded_file_to_df(uploaded)
        st.session_state.current_df = df
    elif pasted and pasted.strip():
        df = parse_text_to_df(pasted)
        st.session_state.current_df = df
    else:
        df = st.session_state.get("current_df", None)

    if df is None:
        st.warning("⚠️ No data loaded yet. Please upload a file or paste data above.")
        return

    st.divider()
    st.markdown("#### 2) Preview & Edit Table")
    df = make_editable(df)
    edited = st.data_editor(df, num_rows="dynamic", use_container_width=True, key="tab3_editor")
    st.session_state.current_df = edited

    st.divider()
    st.markdown("#### 3) Convert to JSON")
    json_text = df_to_json_pretty(edited)
    edited_json = st.text_area("Editable JSON representation", value=json_text, height=220, key="tab3_json")

    if st.button("✅ Apply JSON edits to table", key="tab3_apply_json"):
        try:
            new_df = json_text_to_df(edited_json)
            st.session_state.current_df = new_df
            st.success("JSON applied successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to parse JSON: {e}")

    st.divider()
    st.markdown("#### 4) Select Agent & Configure")
    
    if not st.session_state.squads:
        st.error("No agents loaded. Please ensure agents.yaml exists.")
        return

    flat_agents = {}
    for sname, agents in st.session_state.squads.items():
        for aname, cfg in agents.items():
            flat_agents[f"{sname} / {aname}"] = cfg

    selected_agent_key = st.selectbox("Choose an agent", options=list(flat_agents.keys()), key="tab3_agent_select")
    agent_cfg = copy.deepcopy(flat_agents[selected_agent_key])

    with st.expander("⚙️ Agent Configuration", expanded=False):
        agent_cfg['system_prompt'] = st.text_area("System Prompt", value=agent_cfg.get('system_prompt', ''), height=150, key="tab3_prompt")
        
        params = agent_cfg.get('params', {})
        col1, col2, col3 = st.columns(3)
        with col1:
            p_temp = st.slider("Temperature", 0.0, 1.0, float(params.get('temperature', 0.3)), 0.05, key="tab3_temp")
        with col2:
            p_top_p = st.slider("Top P", 0.0, 1.0, float(params.get('top_p', 0.9)), 0.05, key="tab3_tp")
        with col3:
            p_max = st.number_input("Max Tokens", 256, 65536, int(params.get('max_output_tokens', 3072)), 256, key="tab3_max")
        agent_cfg['params'] = {'temperature': p_temp, 'top_p': p_top_p, 'max_output_tokens': p_max}

    st.divider()
    if st.button("🚀 Run Agent Analysis", type="primary", key="tab3_run"):
        if not st.session_state.selected_model:
            st.error("No model selected in sidebar. Please configure API keys and choose a model.")
        else:
            client = st.session_state.gemini_client if 'gemini' in st.session_state.selected_model else st.session_state.grok_client
            executor = AgentExecutor(st.session_state.selected_model, client)
            
            # Build content for agent
            content_for_agent = f"""DATA (JSON format):
{edited_json}

Please analyze this dataset and provide:
1. Summary statistics and key findings
2. Data quality assessment
3. Insights and patterns
4. Recommendations for further analysis or visualizations
"""
            
            with st.spinner(f"Agent '{agent_cfg['name']}' is analyzing..."):
                result = executor.execute(agent_cfg, content_for_agent, context=None)
                st.session_state.analysis_result = result
                st.rerun()

    if st.session_state.analysis_result:
        res = st.session_state.analysis_result
        if res.get('status') == 'success':
            st.divider()
            st.success("✅ Analysis Complete!")
            agent_text = res.get('result', '')
            
            with st.expander("📄 View Full Report", expanded=True):
                st.markdown(agent_text)
            
            st.download_button(
                label="💾 Download Report (Markdown)",
                data=agent_text,
                file_name=f"data_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
        else:
            st.error(f"❌ Agent Error: {res.get('error')}")

# =============================================================================
# TAB 4: COMPARISON
# =============================================================================

def comparison_tab_ui():
    """Render Tab 4 - Comparison (multiple datasets)."""
    st.header("📈 Dataset Comparison")
    st.info("Upload or paste multiple datasets to compare them side-by-side with agent analysis.")

    uploaded_list = st.file_uploader(
        "Upload multiple files (csv/json/txt)", 
        type=["csv", "json", "txt"], 
        accept_multiple_files=True,
        key="tab4_uploader"
    )
    pasted_multi = st.text_area(
        "Or paste multiple datasets separated by '---' (three dashes)", 
        height=160,
        key="tab4_pasted"
    )

    datasets = {}
    
    if uploaded_list:
        for f in uploaded_list:
            try:
                df = load_uploaded_file_to_df(f)
                datasets[f.name] = df
            except Exception as e:
                st.warning(f"Failed to parse {f.name}: {e}")

    if pasted_multi and pasted_multi.strip():
        parts = [p.strip() for p in pasted_multi.split("\n---\n") if p.strip()]
        for idx, part in enumerate(parts):
            try:
                df = parse_text_to_df(part)
                datasets[f"pasted_{idx+1}"] = df
            except Exception as e:
                st.warning(f"Failed to parse pasted dataset #{idx+1}: {e}")

    if not datasets:
        st.warning("⚠️ No datasets loaded yet. Please upload files or paste data above.")
        return

    st.divider()
    st.markdown(f"#### 📊 Loaded Datasets ({len(datasets)})")
    
    edited_datasets = {}
    json_map = {}
    
    for name, df in datasets.items():
        with st.expander(f"📁 {name} ({df.shape[0]} rows × {df.shape[1]} cols)"):
            df = make_editable(df)
            edited = st.data_editor(df, key=f"cmp_editor_{name}", num_rows="dynamic", use_container_width=True)
            edited_datasets[name] = edited
            
            jtext = df_to_json_pretty(edited)
            new_json = st.text_area(f"JSON for {name}", value=jtext, key=f"cmp_json_{name}", height=180)
            json_map[name] = new_json

    if st.button("✅ Apply all JSON edits", key="tab4_apply_json"):
        applied = {}
        errors = []
        for name, jtext in json_map.items():
            try:
                newdf = json_text_to_df(jtext)
                applied[name] = newdf
            except Exception as e:
                errors.append(f"{name}: {e}")
        
        if errors:
            st.error("Some JSONs failed to parse:\n" + "\n".join(errors))
        else:
            st.success("All JSONs applied successfully!")
            st.session_state.datasets = applied
            st.rerun()

    st.divider()
    st.markdown("#### 🤖 Select Agent for Comparative Analysis")
    
    if not st.session_state.squads:
        st.error("No agents loaded. Please ensure agents.yaml exists.")
        return

    flat_agents = {}
    for sname, agents in st.session_state.squads.items():
        for aname, cfg in agents.items():
            flat_agents[f"{sname} / {aname}"] = cfg

    selected_agent_key = st.selectbox("Choose comparative agent", options=list(flat_agents.keys()), key="tab4_agent_select")
    agent_cfg = copy.deepcopy(flat_agents[selected_agent_key])

    with st.expander("⚙️ Agent Configuration", expanded=False):
        agent_cfg['system_prompt'] = st.text_area(
            "System Prompt", 
            value=agent_cfg.get('system_prompt', ''),
            height=150,
            key="tab4_prompt"
        )
        
        params = agent_cfg.get('params', {})
        col1, col2, col3 = st.columns(3)
        with col1:
            p_temp = st.slider("Temperature", 0.0, 1.0, float(params.get('temperature', 0.3)), 0.05, key="cmp_temp")
        with col2:
            p_top_p = st.slider("Top P", 0.0, 1.0, float(params.get('top_p', 0.9)), 0.05, key="cmp_tp")
        with col3:
            p_max = st.number_input("Max Tokens", 256, 65536, int(params.get('max_output_tokens', 3072)), 256, key="cmp_max")
        agent_cfg['params'] = {'temperature': p_temp, 'top_p': p_top_p, 'max_output_tokens': p_max}

    st.divider()
    if st.button("🚀 Run Comparative Analysis", type="primary", key="tab4_run"):
        if not st.session_state.selected_model:
            st.error("No model selected in sidebar.")
        else:
            client = st.session_state.gemini_client if 'gemini' in st.session_state.selected_model else st.session_state.grok_client
            executor = AgentExecutor(st.session_state.selected_model, client)
            
            assembled = {}
            for name, jtext in json_map.items():
                try:
                    assembled[name] = json.loads(jtext)
                except Exception:
                    assembled[name] = jtext
            
            content_for_agent = f"""COMPARE THESE DATASETS:

{json.dumps(assembled, ensure_ascii=False, indent=2)}

Please provide a comprehensive comparative analysis including:
1. Schema comparison (columns, data types)
2. Statistical comparison (distributions, ranges, missing values)
3. Key differences and similarities
4. Potential join keys or relationships
5. Recommendations for merging or further analysis
6. Suggested visualizations for comparison
"""
            
            with st.spinner("Running comparative analysis..."):
                result = executor.execute(agent_cfg, content_for_agent, context=None)
                st.session_state.comparison_results = result
                st.rerun()

    if st.session_state.comparison_results:
        res = st.session_state.comparison_results
        if res.get('status') == 'success':
            st.divider()
            st.success("✅ Comparative Analysis Complete!")
            agent_text = res.get('result', '')
            
            with st.expander("📄 View Full Comparison Report", expanded=True):
                st.markdown(agent_text)
            
            st.download_button(
                label="💾 Download Comparison Report",
                data=agent_text,
                file_name=f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
        else:
            st.error(f"❌ Agent Error: {res.get('error')}")

# =============================================================================
# TAB 5: DOCUMENT OCR (LEGACY)
# =============================================================================

def tab5_ui():
    """Legacy Tab 5 - Document OCR."""
    st.header("📄 Document OCR & Analysis (Legacy)")
    st.info("⚠️ This is a legacy tab. Please use Tab 7 for improved document analysis.")
    
    st.markdown("This tab has been superseded by **Tab 7: Document Scientist**")
    
    if st.button("Go to Tab 7", key="tab5_redirect"):
        st.info("Please click on the 'Document Scientist' tab above.")

# =============================================================================
# TAB 6: MULTI-DATASET (LEGACY)
# =============================================================================

def tab6_ui():
    """Legacy Tab 6 - Multi-dataset Analysis."""
    st.header("📊 Multi-dataset Analysis (Legacy)")
    st.info("⚠️ This is a legacy tab. Please use Tab 8 for improved multi-dataset analysis.")
    
    st.markdown("This tab has been superseded by **Tab 8: Deep Learning Scientist**")
    
    if st.button("Go to Tab 8", key="tab6_redirect"):
        st.info("Please click on the 'Deep Learning Scientist' tab above.")

# =============================================================================
# TAB 7: DOCUMENT SCIENTIST
# =============================================================================

def tab7_ui():
    """Render Tab 7 - Document Scientist (Improved OCR & Analysis)."""
    st.header("📄 Document Scientist")
    st.info("Upload or paste a document (TXT/PDF). Extract text, edit it, and run agent analysis.")
    
    with st.expander("📁 1) Upload or Paste Document", expanded=True):
        uploaded = st.file_uploader("Upload TXT/PDF", type=["txt", "pdf"], key="tab7_uploader")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            pasted = st.text_area("Or paste text/markdown here", height=160, key="tab7_pasted")
        with col2:
            pages_str = st.text_input(
                "PDF pages (optional)", 
                placeholder="e.g., 0,2,5-7",
                help="Leave blank for all pages"
            )

    content = ""
    if uploaded:
        if uploaded.type == "application/pdf" or uploaded.name.endswith('.pdf'):
            pages = None
            if pages_str:
                try:
                    pages = []
                    for part in pages_str.split(','):
                        part = part.strip()
                        if '-' in part:
                            start, end = map(int, part.split('-'))
                            pages.extend(range(start, end + 1))
                        else:
                            pages.append(int(part))
                except ValueError:
                    st.warning("Invalid page range format. Using all pages.")
                    pages = None
            
            with st.spinner("Extracting text from PDF..."):
                content = extract_text_from_pdf(uploaded, pages)
        else:
            content = uploaded.getvalue().decode("utf-8")
    elif pasted:
        content = pasted

    if not content:
        st.warning("⚠️ No document loaded yet. Please upload a file or paste content above.")
        return

    st.divider()
    st.markdown("#### 2) Review & Edit Extracted Content")
    edited_content = st.text_area(
        "Editable Content", 
        content, 
        height=300, 
        key="tab7_content_area",
        help="Edit the extracted text before analysis"
    )

    st.divider()
    st.markdown("#### 3) Select Agent & Configure")
    
    if not st.session_state.squads:
        st.error("No agents loaded. Please ensure agents.yaml exists.")
        return

    flat_agents = {}
    for sname, agents in st.session_state.squads.items():
        for aname, cfg in agents.items():
            flat_agents[f"{sname} / {aname}"] = cfg

    selected_agent_key = st.selectbox("Choose an agent", options=list(flat_agents.keys()), key="tab7_agent_select")
    agent_cfg = copy.deepcopy(flat_agents[selected_agent_key])

    with st.expander("⚙️ Agent Configuration", expanded=False):
        agent_cfg['system_prompt'] = st.text_area(
            "System Prompt", 
            value=agent_cfg.get('system_prompt', ''),
            height=150,
            key="tab7_prompt"
        )
        
        params = agent_cfg.get('params', {})
        col1, col2, col3 = st.columns(3)
        with col1:
            p_temp = st.slider("Temperature", 0.0, 1.0, float(params.get('temperature', 0.3)), 0.05, key="tab7_temp")
        with col2:
            p_top_p = st.slider("Top P", 0.0, 1.0, float(params.get('top_p', 0.9)), 0.05, key="tab7_top_p")
        with col3:
            p_max = st.number_input("Max Tokens", 256, 65536, int(params.get('max_output_tokens', 4096)), 256, key="tab7_max")
        agent_cfg['params'] = {'temperature': p_temp, 'top_p': p_top_p, 'max_output_tokens': p_max}

    st.divider()
    if st.button("🚀 Run Document Analysis", type="primary", key="tab7_run"):
        if not st.session_state.selected_model:
            st.error("Please select a model from the sidebar.")
        else:
            client = st.session_state.gemini_client if 'gemini' in st.session_state.selected_model else st.session_state.grok_client
            executor = AgentExecutor(st.session_state.selected_model, client)
            
            content_for_agent = f"""DOCUMENT CONTENT:

{edited_content}

Please analyze this document and provide:
1. Summary of main topics and themes
2. Key entities (people, organizations, locations, dates)
3. Important insights and findings
4. Action items or recommendations (if applicable)
5. Questions for further investigation
"""
            
            with st.spinner(f"Agent '{agent_cfg['name']}' is analyzing the document..."):
                result = executor.execute(agent_cfg, content_for_agent, context=None)
                st.session_state.document_analysis_result = result
                st.rerun()

    if st.session_state.document_analysis_result:
        res = st.session_state.document_analysis_result
        if res.get('status') == 'success':
            st.divider()
            st.success("✅ Document Analysis Complete!")
            agent_text = res.get('result', '')
            
            with st.expander("📄 View Full Analysis", expanded=True):
                st.markdown(agent_text)
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="💾 Download Analysis Report",
                    data=agent_text,
                    file_name=f"document_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
            with col2:
                st.download_button(
                    label="💾 Download Original Content",
                    data=edited_content,
                    file_name=f"document_content_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        else:
            st.error(f"❌ Agent Error: {res.get('error')}")

# =============================================================================
# TAB 8: DEEP LEARNING SCIENTIST
# =============================================================================

def tab8_ui():
    """Render Tab 8 - Deep Learning Scientist (Advanced Multi-dataset Analysis)."""
    st.header("🧠 Deep Learning Scientist")
    st.info("Upload multiple datasets for comprehensive deep learning and ML analysis with advanced insights.")

    with st.expander("📁 1) Upload Multiple Datasets", expanded=True):
        uploaded = st.file_uploader(
            "Upload datasets (CSV, JSON, TXT)", 
            type=["csv", "json", "txt"], 
            accept_multiple_files=True, 
            key="tab8_uploader"
        )

    datasets = {}
    if uploaded:
        for file in uploaded:
            dsid = file.name
            try:
                datasets[dsid] = load_uploaded_file_to_df(file)
            except Exception as e:
                st.warning(f"Could not load {file.name}: {e}")

    if not datasets:
        st.warning("⚠️ Please upload one or more datasets to begin analysis.")
        return

    st.divider()
    st.markdown(f"#### 📊 Dataset Overview ({len(datasets)} datasets loaded)")
    
    json_map = {}
    summary_stats = []
    
    for dsid, df in datasets.items():
        with st.expander(f"📁 {dsid} ({df.shape[0]} rows × {df.shape[1]} cols)"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.dataframe(df.head(10), use_container_width=True)
            
            with col2:
                st.markdown("**Quick Stats:**")
                st.metric("Rows", df.shape[0])
                st.metric("Columns", df.shape[1])
                st.metric("Missing", df.isnull().sum().sum())
            
            json_data = df_to_json_pretty(df)
            edited_json = st.text_area(
                f"Editable JSON for {dsid}", 
                json_data, 
                height=200, 
                key=f"json_edit_{dsid}"
            )
            json_map[dsid] = edited_json
            
            summary_stats.append({
                "Dataset": dsid,
                "Rows": df.shape[0],
                "Columns": df.shape[1],
                "Missing Values": df.isnull().sum().sum()
            })

    st.divider()
    st.markdown("#### 📈 Dataset Summary")
    summary_df = pd.DataFrame(summary_stats)
    st.dataframe(summary_df, use_container_width=True)

    st.divider()
    st.markdown("#### 🤖 Select Agent for Deep Learning Analysis")
    
    if not st.session_state.squads:
        st.error("No agents loaded. Please ensure agents.yaml exists.")
        return

    flat_agents = {}
    for sname, agents in st.session_state.squads.items():
        for aname, cfg in agents.items():
            flat_agents[f"{sname} / {aname}"] = cfg

    selected_agent_key = st.selectbox("Choose an agent", options=list(flat_agents.keys()), key="tab8_agent_select")
    agent_cfg = copy.deepcopy(flat_agents[selected_agent_key])

    with st.expander("⚙️ Agent Configuration", expanded=False):
        agent_cfg['system_prompt'] = st.text_area(
            "System Prompt", 
            value=agent_cfg.get('system_prompt', ''),
            height=150,
            key="tab8_prompt",
            help="Customize the agent's instructions for deep learning analysis"
        )
        
        params = agent_cfg.get('params', {})
        col1, col2, col3 = st.columns(3)
        with col1:
            p_temp = st.slider("Temperature", 0.0, 1.0, float(params.get('temperature', 0.3)), 0.05, key="tab8_temp")
        with col2:
            p_top_p = st.slider("Top P", 0.0, 1.0, float(params.get('top_p', 0.9)), 0.05, key="tab8_top_p")
        with col3:
            p_max = st.number_input("Max Tokens", 256, 65536, int(params.get('max_output_tokens', 8192)), 256, key="tab8_max")
        agent_cfg['params'] = {'temperature': p_temp, 'top_p': p_top_p, 'max_output_tokens': p_max}

    st.divider()
    if st.button("🚀 Run Deep Learning Analysis", type="primary", key="tab8_run"):
        if not st.session_state.selected_model:
            st.error("Please select a model from the sidebar.")
        else:
            client = st.session_state.gemini_client if 'gemini' in st.session_state.selected_model else st.session_state.grok_client
            executor = AgentExecutor(st.session_state.selected_model, client)
            
            assembled_payload = {}
            for dsid, json_text in json_map.items():
                try:
                    assembled_payload[dsid] = json.loads(json_text)
                except json.JSONDecodeError:
                    st.warning(f"Could not parse JSON for {dsid}. Sending as raw text.")
                    assembled_payload[dsid] = json_text
            
            content_for_agent = f"""DEEP LEARNING ANALYSIS REQUEST

You are a Deep Learning Scientist. Analyze the following collection of datasets and provide comprehensive insights.

DATASETS:
{json.dumps(assembled_payload, indent=2, ensure_ascii=False)}

DATASET SUMMARY:
{summary_df.to_markdown()}

Please provide:
1. **Data Exploration**: Comprehensive overview of each dataset
2. **Feature Engineering**: Suggestions for new features and transformations
3. **Relationships**: Cross-dataset relationships and potential joins
4. **ML Opportunities**: Specific machine learning and deep learning approaches
5. **Model Recommendations**: Suitable models (classification, regression, clustering, etc.)
6. **Training Strategy**: Data splitting, validation approach, hyperparameter tuning
7. **Advanced Techniques**: Transfer learning, ensemble methods, neural architecture suggestions
8. **Visualization Ideas**: Key plots and dashboards for insights
9. **Implementation Roadmap**: Step-by-step plan for building ML pipeline
10. **Potential Challenges**: Data quality issues, class imbalance, etc.
"""
            
            with st.spinner(f"Agent '{agent_cfg['name']}' is running deep analysis... This may take a moment."):
                result = executor.execute(agent_cfg, content_for_agent, context=None)
                st.session_state.multi_dataset_result = result
                st.rerun()

    if st.session_state.multi_dataset_result:
        res = st.session_state.multi_dataset_result
        if res.get('status') == 'success':
            st.divider()
            st.success("✅ Deep Learning Analysis Complete!")
            agent_text = res.get('result', '')
            
            with st.expander("📄 View Full Deep Learning Report", expanded=True):
                st.markdown(agent_text)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button(
                    label="💾 Download Full Report",
                    data=agent_text,
                    file_name=f"deep_learning_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
            with col2:
                st.download_button(
                    label="💾 Download Dataset Summary",
                    data=summary_df.to_csv(index=False),
                    file_name=f"dataset_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            with col3:
                all_json = json.dumps(assembled_payload, indent=2, ensure_ascii=False)
                st.download_button(
                    label="💾 Download Combined JSON",
                    data=all_json,
                    file_name=f"combined_datasets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        else:
            st.error(f"❌ Agent Error: {res.get('error')}")

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    initialize_session_state()
    apply_theme()
    render_sidebar()
    
    st.title("🤖 Multi-Agent Analysis System")
    st.markdown("*Powered by AI agents for comprehensive data and document analysis*")
    
    tabs = st.tabs([
        "🔬 Analysis",
        "📚 Agent Library",
        "🔗 Workflow",
        "📊 Data Analysis",
        "📈 Comparison",
        "📄 Document Scientist",
        "🧠 Deep Learning Scientist"
    ])
    
    with tabs[0]:
        render_analysis_tab()
    
    with tabs[1]:
        st.header("📚 Agent Library")
        st.info("Browse all available agents organized by squad.")
        
        if st.session_state.squads:
            for squad_name, agents in st.session_state.squads.items():
                with st.expander(f"**{squad_name}** ({len(agents)} agents)", expanded=False):
                    for agent_name, agent in agents.items():
                        st.markdown(f"""
                        <div class="agent-card">
                            <h4>{agent['name']}</h4>
                            <p><strong>ID:</strong> {agent['id']}</p>
                            <p><strong>Category:</strong> {agent.get('category', 'N/A')}</p>
                            <p>{agent.get('description', 'No description')}</p>
                        </div>
                        """, unsafe_allow_html=True)
    
    with tabs[2]:
        st.header("🔗 Multi-Agent Workflow")
        st.info("Create workflows that chain multiple agents together.")
        
        if st.session_state.selected_squad:
            squad = st.session_state.squads[st.session_state.selected_squad]
            
            st.markdown("#### Build Workflow")
            num_agents = st.number_input("Number of agents in workflow", 
                                        min_value=2, max_value=5, value=2)
            
            workflow_agents = []
            for i in range(num_agents):
                agent_name = st.selectbox(
                    f"Agent {i+1}",
                    options=list(squad.keys()),
                    key=f"workflow_agent_{i}"
                )
                workflow_agents.append(squad[agent_name])
            
            content = st.text_area("Initial Content", height=200)
            
            if st.button("🚀 Execute Workflow", type="primary"):
                client = (st.session_state.gemini_client 
                         if 'gemini' in st.session_state.selected_model 
                         else st.session_state.grok_client)
                
                executor = AgentExecutor(st.session_state.selected_model, client)
                orchestrator = WorkflowOrchestrator(executor)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(idx, total, agent_name):
                    progress_bar.progress((idx + 1) / total)
                    status_text.text(f"Stage {idx+1}/{total}: {agent_name}")
                
                results = orchestrator.execute_workflow(
                    workflow_agents, content, update_progress
                )
                
                st.session_state.workflow_results = results
                st.rerun()
            
            if st.session_state.workflow_results:
                st.divider()
                st.header("📊 Workflow Results")
                
                for idx, result in enumerate(st.session_state.workflow_results):
                    with st.expander(f"Stage {idx+1}: {result['agent_name']}", expanded=(idx == len(st.session_state.workflow_results) - 1)):
                        if result['status'] == 'success':
                            st.success(f"✅ Completed")
                            st.markdown(result['result'])
                        else:
                            st.error(f"❌ Failed: {result['error']}")
        else:
            st.warning("Please select a squad from the sidebar to build workflows.")
    
    with tabs[3]:
        data_analysis_tab_ui()
    
    with tabs[4]:
        comparison_tab_ui()
    
    with tabs[5]:
        tab7_ui()
    
    with tabs[6]:
        tab8_ui()

if __name__ == "__main__":
    main()
