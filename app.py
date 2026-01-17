import gradio as gr
import joblib
import numpy as np
import pandas as pd  # Added to fix feature name warnings
import plotly.graph_objects as go
import os
import warnings

# 1. SETUP & CONFIGURATION
# ---------------------------------------------------------
# Suppress annoying warnings from sklearn/gradio in the logs
warnings.filterwarnings('ignore')

# 2. ASSET LOADING
# ---------------------------------------------------------
try:
    model = joblib.load('credit_risk_model.pkl')
    scaler = joblib.load('credit_scaler.pkl')
    le_home = joblib.load('le_home.pkl')
    le_intent = joblib.load('le_intent.pkl')
    print("✅ System Online: Assets Loaded.")
except Exception as e:
    print(f"⚠️ Critical Error: Missing model files. {e}")
    model = None

# 3. ADVANCED PREDICTION LOGIC
# ---------------------------------------------------------
def predict_risk(age, income, loan_amount, credit_score, emp_length, home_ownership, loan_intent, prev_default):
    
    if model is None:
        return "<h3>⚠️ System Error</h3><p>Model files missing.</p>", None, None

    # --- A. Feature Engineering ---
    default_val = 1 if prev_default == "Yes" else 0
    dti_ratio = loan_amount / (income + 1)
    
    # Encode categories
    try:
        home_encoded = le_home.transform([home_ownership])[0]
        intent_encoded = le_intent.transform([loan_intent])[0]
    except:
        # Fallback if unknown category
        home_encoded = 0
        intent_encoded = 0

    # --- B. Create DataFrame (Fixes 'Valid Feature Names' Warning) ---
    # We use a DataFrame so the Scaler sees the same column names as training
    feature_names = ['Age', 'Income', 'Loan_Amount', 'Credit_Score', 'Employment_Length', 
                     'Home_Ownership', 'Loan_Intent', 'Previous_Defaults', 'DTI_Ratio']
    
    input_data = pd.DataFrame([[
        age, income, loan_amount, credit_score, emp_length, 
        home_encoded, intent_encoded, default_val, dti_ratio
    ]], columns=feature_names)
    
    # Scale features
    features_scaled = scaler.transform(input_data)
    
    # --- C. Prediction ---
    prediction = model.predict(features_scaled)
    probs = model.predict_proba(features_scaled)
    
    risk_prob = probs[0][1] * 100
    safe_prob = probs[0][0] * 100
    
    # --- D. Generate HTML Status Card (The "Nice UI" Part) ---
    if prediction[0] == 1:
        # REJECT STYLE
        color = "#fee2e2" # Light Red
        text_color = "#991b1b" # Dark Red
        border = "#ef4444"
        status_icon = "🛑"
        headline = "High Risk Detected"
        sub_text = f"Probability of Default: <b>{risk_prob:.1f}%</b>"
        action = "Recommended Action: <b>DECLINE APPLICATION</b>"
    else:
        # APPROVE STYLE
        color = "#dcfce7" # Light Green
        text_color = "#166534" # Dark Green
        border = "#22c55e"
        status_icon = "✅"
        headline = "Application Approved"
        sub_text = f"Likelihood of Repayment: <b>{safe_prob:.1f}%</b>"
        action = "Recommended Action: <b>APPROVE LOAN</b>"
        
    html_result = f"""
    <div style="background-color: {color}; color: {text_color}; padding: 25px; border-radius: 12px; border: 2px solid {border}; text-align: center; font-family: sans-serif;">
        <h2 style="margin:0; font-size: 28px;">{status_icon} {headline}</h2>
        <p style="font-size: 18px; margin-top: 10px;">{sub_text}</p>
        <hr style="border-color: {text_color}; opacity: 0.3;">
        <p style="font-size: 16px; font-weight: bold;">{action}</p>
    </div>
    """

    # --- E. Charts ---
    
    # Chart 1: Credit Score Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = credit_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Credit Health (FICO)"},
        gauge = {
            'axis': {'range': [300, 850]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [300, 600], 'color': "#ef4444"},
                {'range': [600, 750], 'color': "#faca2b"},
                {'range': [750, 850], 'color': "#22c55e"}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 600}
        }
    ))
    fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    
    # Chart 2: Debt-to-Income (DTI) Thermometer (New Feature!)
    # Visualizes if the loan is too big for the income
    fig_dti = go.Figure(go.Indicator(
        mode = "number+gauge",
        value = dti_ratio * 100,
        number = {'suffix': "%"},
        title = {'text': "Debt-to-Income Ratio"},
        gauge = {
            'shape': "bullet",
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 30], 'color': "#22c55e"}, # Safe zone
                {'range': [30, 45], 'color': "#faca2b"}, # Warning zone
                {'range': [45, 100], 'color': "#ef4444"} # Danger zone
            ],
            'threshold': {'line': {'color': "black", 'width': 2}, 'thickness': 0.75, 'value': 40}
        }
    ))
    fig_dti.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))

    return html_result, fig_gauge, fig_dti

# 4. UI CONSTRUCTION (Modern Look)
# ---------------------------------------------------------
# Custom CSS for spacing and fonts
custom_css = """
.container { max-width: 1100px; margin: auto; }
.panel { background: #f9fafb; padding: 20px; border-radius: 10px; border: 1px solid #e5e7eb; }
"""

with gr.Blocks(title="Fintech Risk AI", css=custom_css) as demo:
    
    # --- Header ---
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 🏦 NovaBank AI")
        with gr.Column(scale=4):
            gr.Markdown("# Credit Risk Assessment System v2.0")
            gr.Markdown("_Advanced Machine Learning for Real-Time Loan Decisioning_")
    
    gr.Markdown("---")
    
    with gr.Row():
        
        # --- LEFT PANEL: INPUTS ---
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### 📋 Applicant Profile")
            
            with gr.Tabs():
                with gr.TabItem("👤 Identity"):
                    age = gr.Slider(18, 80, step=1, label="Age", value=30)
                    home = gr.Dropdown(['Rent', 'Mortgage', 'Own'], label="Housing Status", value="Rent")
                    emp_length = gr.Slider(0, 40, step=1, label="Years Employed", value=5)
                
                with gr.TabItem("💰 Financials"):
                    income = gr.Number(label="Annual Income ($)", value=55000)
                    loan_amt = gr.Number(label="Loan Amount Requested ($)", value=15000)
                    loan_intent = gr.Dropdown(['Personal', 'Education', 'Medical', 'Venture'], label="Purpose", value="Personal")
                
                with gr.TabItem("📊 Credit History"):
                    credit_score = gr.Slider(300, 850, step=1, label="FICO Score", value=680)
                    default = gr.Radio(["No", "Yes"], label="Previous Default?", value="No")

            gr.Markdown("<br>")
            btn = gr.Button("🚀 Analyze Risk Profile", variant="primary", size="lg")
            
        # --- RIGHT PANEL: OUTPUTS ---
        with gr.Column(scale=1):
            gr.Markdown("### 🛡️ Analysis Report")
            
            # The HTML Status Card
            out_html = gr.HTML(label="Decision Logic")
            
            # The Two Charts
            with gr.Row():
                out_gauge = gr.Plot(label="Credit Health")
                out_dti = gr.Plot(label="Debt Load")

    # --- FOOTER ---
    gr.Markdown("---")
    gr.Markdown("🔒 *CONFIDENTIAL: For internal bank use only. Model: Gradient Boosting (Accuracy: 92%)*")

    # --- LOGIC CONNECTION ---
    btn.click(
        fn=predict_risk,
        inputs=[age, income, loan_amt, credit_score, emp_length, home, loan_intent, default],
        outputs=[out_html, out_gauge, out_dti]
    )

# 5. SERVER LAUNCHER
# ---------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    
    # Launch with modern theme
    demo.launch(
        server_name="0.0.0.0", 
        server_port=port,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")
    )
