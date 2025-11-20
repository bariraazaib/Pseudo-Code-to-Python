import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Page configuration
st.set_page_config(
    page_title="Pseudocode to Python Converter by barirazaib",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    
    .main-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .stTextArea textarea {
        border-radius: 15px;
        border: 2px solid #e0e0e0;
        font-family: 'Courier New', monospace;
        font-size: 15px;
        padding: 1rem;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
    
    .stButton>button {
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 700;
        font-size: 18px;
        transition: all 0.3s;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        color: white;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
    }
    
    .code-output {
        background: #1e1e1e;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .example-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s;
        border-left: 5px solid #667eea;
    }
    
    .example-card:hover {
        transform: translateX(8px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.2);
    }
    
    .info-box {
        background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: #333;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        text-align: center;
        border-top: 4px solid #667eea;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #667eea;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    .stDownloadButton>button {
        border-radius: 20px;
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.6rem 1.5rem;
    }
    
    .stDownloadButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(17, 153, 142, 0.3);
    }
    
    .stRadio > label {
        font-weight: 600;
        color: #667eea;
        font-size: 1.1rem;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #999;
        margin-top: 3rem;
        border-top: 2px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the fine-tuned model from Hugging Face Hub"""
    try:
        model_name = "mustehsannisarrao/pseudocode-to-python"
        
        with st.spinner("🔄 Loading AI Model..."):
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
        
        return tokenizer, model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

def pseudocode_to_python(pseudocode, tokenizer, model):
    """Convert pseudocode to Python code"""
    prompt = f"""Translate the following pseudocode to Python code:

Pseudocode:
{pseudocode}

Python Code:
"""
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2
        )
    
    # Decode output
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the Python code part
    if "Python Code:" in full_output:
        python_code = full_output.split("Python Code:")[-1].strip()
    else:
        python_code = full_output[len(prompt):].strip()
    
    return python_code

# Initialize session state
if 'selected_example' not in st.session_state:
    st.session_state.selected_example = ""

# Header
st.markdown('<div class="main-title">🐍 Pseudocode → Python</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">✨ Transform your pseudocode into clean, executable Python code with AI</div>', unsafe_allow_html=True)

# Load model
tokenizer, model = load_model()

if tokenizer and model:
    # Sidebar with examples
    with st.sidebar:
        st.markdown('<div class="section-header">📚 Example Library</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <strong>🎯 How to use:</strong><br>
            Select an example below to auto-fill the input area, or write your own pseudocode!
        </div>
        """, unsafe_allow_html=True)
        
        examples = {
            "🔢 Simple Variable": "x = 5\nprint x",
            "🔁 For Loop": "FOR i FROM 1 TO 5\n    PRINT i\nENDFOR", 
            "❓ Conditional": "IF score > 50 THEN\n    PRINT 'Pass'\nELSE\n    PRINT 'Fail'\nENDIF",
            "📊 Array Sum": "numbers = [1, 2, 3, 4, 5]\nsum = 0\nFOR i FROM 0 TO 4\n    sum = sum + numbers[i]\nENDFOR\nprint sum",
            "🔍 Find Maximum": "numbers = [3, 7, 2, 9, 1]\nmax = numbers[0]\nFOR num IN numbers\n    IF num > max THEN\n        max = num\n    ENDIF\nENDFOR\nprint max"
        }
        
        selected_example = st.radio(
            "Choose an example:",
            options=list(examples.keys()),
            key="example_selector"
        )
        
        if selected_example:
            st.session_state.selected_example = examples[selected_example]
        
        st.markdown("---")
        
        # Model Info
        st.markdown('<div class="section-header">🤖 Model Info</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #667eea; margin: 0;">GPT-Based</h3>
            <p style="font-size: 13px; color: #666; margin-top: 0.5rem;">Fine-tuned Transformer</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; margin-top: 1rem;">
            <strong>✨ Features:</strong><br>
            • Context-aware conversion<br>
            • Multiple syntax support<br>
            • Optimized output<br>
            • Fast generation
        </div>
        """, unsafe_allow_html=True)

    # Main content area
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<div class="section-header">📝 Input Pseudocode</div>', unsafe_allow_html=True)
        
        pseudocode = st.text_area(
            "",
            height=300,
            value=st.session_state.selected_example,
            placeholder="""Example:
FOR i FROM 1 TO 10
    IF i MOD 2 = 0 THEN
        PRINT i
    ENDIF
ENDFOR""",
            key="pseudocode_input",
            label_visibility="collapsed"
        )
        
        # Stats
        lines = len(pseudocode.split('\n'))
        chars = len(pseudocode)
        
        stat_col1, stat_col2 = st.columns(2)
        with stat_col1:
            st.markdown(f"""
            <div class="metric-card">
                <h2 style="color: #667eea; margin: 0;">{lines}</h2>
                <p style="font-size: 14px; color: #666; margin: 0;">Lines</p>
            </div>
            """, unsafe_allow_html=True)
        
        with stat_col2:
            st.markdown(f"""
            <div class="metric-card">
                <h2 style="color: #764ba2; margin: 0;">{chars}</h2>
                <p style="font-size: 14px; color: #666; margin: 0;">Characters</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="section-header">🚀 Python Output</div>', unsafe_allow_html=True)
        
        if st.button("✨ Convert to Python", type="primary", use_container_width=True):
            if pseudocode.strip():
                with st.spinner("🔮 Converting your pseudocode..."):
                    try:
                        import time
                        start_time = time.time()
                        
                        python_code = pseudocode_to_python(pseudocode, tokenizer, model)
                        
                        end_time = time.time()
                        
                        st.markdown("### ✅ Generated Code")
                        st.code(python_code, language="python", line_numbers=True)
                        
                        # Metrics
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("⏱️ Time", f"{(end_time - start_time):.2f}s")
                        with col_b:
                            st.metric("📏 Lines", len(python_code.split('\n')))
                        
                        # Download button
                        st.download_button(
                            label="📥 Download Python File",
                            data=python_code,
                            file_name="converted_code.py",
                            mime="text/x-python",
                            use_container_width=True
                        )
                        
                        st.success("✅ Conversion completed successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Conversion failed: {e}")
            else:
                st.warning("⚠️ Please enter some pseudocode first!")
        
        else:
            # Placeholder
            st.markdown("""
            <div style="background: #f8f9fa; padding: 2rem; border-radius: 15px; text-align: center; border: 2px dashed #667eea;">
                <h3 style="color: #667eea;">🎯 Ready to Convert</h3>
                <p style="color: #666;">Enter your pseudocode and click the button above to generate Python code</p>
            </div>
            """, unsafe_allow_html=True)

else:
    st.error("❌ Failed to load the model. Please try refreshing the page.")

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <strong>Built with 🐍 Python + Streamlit</strong><br>
    Model by <a href="https://huggingface.co/mustehsannisarrao" target="_blank" style="color: #667eea;">barirazaib</a>
</div>
""", unsafe_allow_html=True)
