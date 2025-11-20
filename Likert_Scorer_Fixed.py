"""
Streamlit NLP Likert Scorer - PROFESSIONAL VERSION
Clean, professional UI with pie chart visualization
"""

import streamlit as st
import pandas as pd
import torch
import os
from datetime import datetime
import plotly.express as px

st.set_page_config(
    page_title="Psychological Likert Scorer", 
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Model path
MODEL_PATH = "./likert_model_finetuned"

# HARDCODED QUESTIONS
QUESTIONS = {
    "Personality Assessment": [
        "I enjoy being the center of attention in social situations",
        "I prefer detailed planning over spontaneous decisions",
        "I feel comfortable expressing my emotions openly",
        "I tend to worry about future possibilities",
        "I enjoy taking risks and trying new experiences"
    ],
    "Work Psychology": [
        "Remote work significantly improves my productivity",
        "I prefer collaborative projects over individual tasks",
        "Tight deadlines help me perform better",
        "I am satisfied with my current work-life balance",
        "Feedback from colleagues helps me improve my work"
    ],
    "Mental Health": [
        "I generally feel optimistic about my future",
        "I have effective coping strategies for stress",
        "I feel comfortable discussing mental health topics",
        "Regular exercise significantly improves my mood",
        "I have a strong support system of friends/family"
    ],
    "Product Feedback": [
        "This product meets all my expectations",
        "The user interface is intuitive and easy to use",
        "Customer support was responsive and helpful",
        "I would recommend this product to others",
        "The price represents good value for money"
    ]
}

@st.cache_resource
def load_model():
    """Load fine-tuned model with proper validation"""
    
    if not os.path.exists(MODEL_PATH):
        st.error("Fine-tuned model not found. Please run training scripts first.")
        st.stop()
    
    try:
        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
        return tokenizer, model, True
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

# Load model
tokenizer, model, is_finetuned = load_model()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

def get_score_color(score):
    """Get color for score visualization"""
    colors = {
        1: "#ff6b6b",  # Red for strong disagreement
        2: "#ffa500",  # Orange for disagreement  
        3: "#ffd93d",  # Yellow for neutral
        4: "#6bcf7f",  # Light green for agreement
        5: "#4ecdc4"   # Teal for strong agreement
    }
    return colors.get(score, "#000000")

def predict_likert(text: str) -> dict:
    """Predict Likert score"""
    
    if not text or text.strip() == "":
        return {
            'likert_score': 3,
            'confidence': 0.0,
            'message': "No text provided",
            'all_probabilities': [0.2, 0.2, 0.2, 0.2, 0.2]
        }
    
    inputs = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][predicted_class].item()
    
    likert_score = predicted_class + 1
    
    all_probs = probs[0].cpu().numpy()
    
    messages = {
        1: "Strongly Disagree",
        2: "Disagree",
        3: "Neutral", 
        4: "Agree",
        5: "Strongly Agree"
    }
    
    return {
        'likert_score': likert_score,
        'confidence': confidence,
        'message': messages.get(likert_score, "Unknown"),
        'all_probabilities': all_probs
    }

# ========== CLEAN PROFESSIONAL UI ==========

# Header
st.title("Psychological Likert Scale Analyzer")
st.markdown("AI-Powered Sentiment Scoring for Research & Assessment")
st.markdown("*Automatically score free-text responses on a standardized 1-5 Likert scale*")

# Sidebar
with st.sidebar:
    st.markdown("### Study Configuration")
    
    category = st.selectbox("Domain", list(QUESTIONS.keys()))
    selected_question = st.selectbox("Assessment Item", QUESTIONS[category])
    
    st.markdown("---")
    st.markdown("### Response Mode")
    mode = st.radio("Input Format", ["Single response", "Batch responses"], index=1)
    
    st.markdown("---")
    st.markdown("### Display Options")
    show_confidence = st.checkbox("Show confidence scores", value=True)
    show_probabilities = st.checkbox("Show probability distribution", value=False)
    show_visualizations = st.checkbox("Show visualizations", value=True)

# Main Content Area
col1, col2 = st.columns([2, 1])

with col1:
    # FIXED: Question Display - This will now properly show the selected question
    st.markdown("### Current Assessment Item")
    
    # This container will dynamically update with the selected question
    question_container = st.container()
    with question_container:
        st.markdown(
            f'<div style="background-color:#f8f9fa; padding:20px; border-radius:8px; border-left: 4px solid #2E86AB; margin:10px 0;">'
            f'<span style="font-size:16px; color:#333;">{selected_question}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    st.markdown("### Participant Responses")
    
    if st.button("Load Example Responses"):
        example_responses = {
            "Personality Assessment": "I absolutely love being the center of attention\nI prefer to stay in the background\nIt depends on the situation",
            "Work Psychology": "Remote work has transformed my productivity\nI struggle to focus when working from home\nI prefer a hybrid approach",
            "Mental Health": "I consistently feel optimistic about life\nI often struggle with negative thoughts\nMy mood varies day by day",
            "Product Feedback": "This product exceeded all my expectations\nIt failed to meet basic requirements\nIt was average with some issues"
        }
        example_text = example_responses.get(category, "I strongly agree with this statement")
    else:
        example_text = ""
    
    if mode == "Single response":
        response = st.text_input(
            "Enter response:",
            placeholder="e.g., I strongly agree with this statement...",
            value=example_text if example_text else ""
        )
        responses = [response.strip()] if response.strip() else []
    else:
        response = st.text_area(
            "Enter responses (one per line):", 
            height=150,
            placeholder="I strongly agree with this...\nI have some reservations...\nIt depends on the context...",
            value=example_text
        )
        responses = [r.strip() for r in response.splitlines() if r.strip()]

with col2:
    # Clean Scoring Guide
    st.markdown("### Scoring Guide")
    
    for score, interpretation, color in [
        (5, "Strongly Agree", "#4ecdc4"),
        (4, "Agree", "#6bcf7f"), 
        (3, "Neutral", "#ffd93d"),
        (2, "Disagree", "#ffa500"),
        (1, "Strongly Disagree", "#ff6b6b")
    ]:
        st.markdown(
            f"<div style='background-color:{color}20; padding:6px; border-radius:4px; border-left:3px solid {color}; margin:2px 0; font-size:14px;'>"
            f"<strong>Score {score}:</strong> {interpretation}"
            f"</div>",
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    st.markdown("### Analysis Tips")
    st.markdown("""
    - Be specific and clear
    - Use natural language
    - Avoid mixed signals
    - Consider context carefully
    """)

# Analysis Button
if st.button("Analyze Responses", type="primary", use_container_width=True):
    if not responses:
        st.error("Please enter at least one response to analyze.")
    else:
        rows = []
        confidence_scores = []
        
        with st.spinner("Analyzing responses..."):
            progress_bar = st.progress(0)
            
            for i, response_text in enumerate(responses):
                result = predict_likert(response_text)
                
                row = {
                    "Response": response_text,
                    "Score": result['likert_score'],
                    "Interpretation": result['message'],
                    "Confidence": result['confidence']
                }
                
                if show_probabilities:
                    for score_idx, prob in enumerate(result['all_probabilities'], 1):
                        row[f'P(Score {score_idx})'] = f"{prob:.3f}"
                
                rows.append(row)
                confidence_scores.append(result['confidence'])
                progress_bar.progress((i + 1) / len(responses))
        
        df = pd.DataFrame(rows)
        
        st.markdown("## Analysis Results")
        
        # Summary Statistics
        st.markdown("### Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_score = df['Score'].mean()
            st.metric("Mean Score", f"{avg_score:.2f}")
        
        with col2:
            st.metric("Median Score", f"{df['Score'].median():.1f}")
        
        with col3:
            st.metric("Std Deviation", f"{df['Score'].std():.2f}")
        
        with col4:
            st.metric("Total Responses", len(df))
        
        if show_confidence:
            st.metric("Average Confidence", f"{df['Confidence'].mean():.1%}")
        
        # Results Table
        st.markdown("### Detailed Scoring")
        display_columns = ['Response', 'Score', 'Interpretation']
        if show_confidence:
            display_columns.append('Confidence')
        if show_probabilities:
            prob_columns = [col for col in df.columns if col.startswith('P(Score')]
            display_columns.extend(prob_columns)
        
        st.dataframe(df[display_columns], use_container_width=True)
        
        # Visualizations
        if show_visualizations and len(responses) > 1:
            st.markdown("### Visualizations")
            
            score_counts = df['Score'].value_counts().sort_index()
            score_counts = score_counts.reindex([1, 2, 3, 4, 5], fill_value=0)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie Chart Visualization
                st.markdown("**Score Distribution**")
                fig_pie = px.pie(
                    values=score_counts.values, 
                    names=score_counts.index, 
                    title="Likert Score Distribution",
                    color=score_counts.index.astype(str),
                    color_discrete_map={
                        '1': '#ff6b6b', '2': '#ffa500', '3': '#ffd93d', 
                        '4': '#6bcf7f', '5': '#4ecdc4'
                    }
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                if show_confidence:
                    st.markdown("**Confidence Distribution**")
                    st.bar_chart(pd.DataFrame({'Confidence': df['Confidence']}))
        
        # Export
        st.markdown("### Export Results")
        export_data = df.copy()
        export_data.insert(0, 'Question_Category', category)
        export_data.insert(1, 'Assessment_Item', selected_question)
        export_data.insert(2, 'Analysis_Date', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        csv = export_data.to_csv(index=False)
        
        st.download_button(
            label="Download Analysis (CSV)",
            data=csv,
            file_name=f"likert_analysis_{category.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("### About This Tool")
st.markdown("""
This psychological assessment tool uses fine-tuned transformer models to automatically score 
free-text responses on a standardized 1-5 Likert scale. Designed for research integrity 
and consistent scoring across multiple respondents.
""")