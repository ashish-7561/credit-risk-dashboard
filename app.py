import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
import warnings

# 1. SETUP
warnings.filterwarnings('ignore')

# 2. ASSET LOADING
try:
    model = joblib.load('credit_risk_model.pkl')
    scaler = joblib.load('credit_scaler.pkl')
    le_home = joblib.load('le_home.pkl')
    le_intent = joblib.load('le_intent.pkl')
    print("✅ System Online.")
except Exception as e:
    print(f"⚠️ Error: {e}")
    model = None

# 3. LOGIC & VISUALIZATION
def predict_risk(age, income, loan_amount, credit_score, emp_length, home_ownership, loan_intent, prev_default):
    
    if model is None:
        return "<h3>⚠️ System Error</h3>", None, None

    # --- Feature Engineering ---
    default_val = 1 if prev_default == "Yes" else 0
    dti_ratio = loan_amount / (income + 1)
    
    try:
        home_encoded = le_home.transform([home_ownership])[0]
        intent_encoded = le_intent.transform([loan_intent])[0]
    except:
        home_encoded = 0
        intent_encoded = 0

    feature_names = ['Age', 'Income', 'Loan_Amount', 'Credit_Score', 'Employment_Length', 
                     'Home_Ownership', 'Loan_Intent', 'Previous_Defaults', 'DTI_Ratio']
    
    input_data = pd.DataFrame([[
        age, income, loan_amount, credit_score, emp_length, 
        home_encoded, intent_encoded, default_val, dti_ratio
    ]], columns=feature_names)
    
    features_scaled = scaler.transform(input_data)
    
    # --- Prediction ---
    prediction = model.predict(features_scaled)
    probs = model.predict_proba(features_scaled)
    
    risk_prob = probs[0][1] * 100
    safe_prob = probs[0][0] * 100
    
    # --- UI GENERATION (The Major Fix) ---
    if prediction[0] == 1:
        # HIGH RISK - Dark Red Card
        card_color = "linear-gradient(135deg, #450a0a 0%, #7f1d1d 100%)" # Deep Red Gradient
        border_color = "#f87171" # Light Red Border
        icon = "🛑"
        title = "HIGH RISK DETECTED"
        prob_text = f"Default Probability: <span style='color: #fca5a5; font-size: 24px; font-weight: bold;'>{risk_prob:.1f}%</span>"
        action = "RECOMMENDATION: DECLINE"
    else:
        # LOW RISK - Dark Green Card
        card_color = "linear-gradient(135deg, #052e16 0%, #14532d 100%)" # Deep Green Gradient
        border_color = "#4ade80" # Light Green Border
        icon = "✅"
        title = "LOAN APPROVED"
        prob_text = f"Repayment Score: <span style='color: #86efac; font-size: 24px; font-weight: bold;'>{safe_prob:.1f}%</span>"
        action = "RECOMMENDATION: APPROVE"

    html_result = f"""
    <div style="background: {card_color}; padding: 25px; border-radius: 15px; border: 1px solid {border_color}; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
        <h2 style="margin:0; color: white; font-size: 24px; letter-spacing: 1px;">{icon} {title}</h2>
        <p style="margin-top: 15px; color: #e5e7eb; font-size: 16px;">{prob_text}</p>
        <hr style="border-color: rgba(255,255,255,0.2); margin: 15px 0;">
        <p style="color: white; font-weight: bold; font-size: 18px; margin: 0;">{action}</p>
    </div>
    """

    # --- DARK MODE CHARTS ---
    
    # Chart 1: Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = credit_score,
        title = {'text': "FICO Score", 'font': {'color': 'white'}},
        number = {'font': {'color': 'white'}},
        gauge = {
            'axis': {'range': [300, 850], 'tickcolor': "white"},
            'bar': {'color': "#00bcd4"}, # Cyan needle
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'steps': [
                {'range': [300, 600], 'color': "#7f1d1d"}, # Dark Red
                {'range': [600, 750], 'color': "#ca8a04"}, # Dark Yellow
                {'range': [750, 850], 'color': "#14532d"}  # Dark Green
            ],
            'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': 600}
        }
    ))
    # Make transparent and dark text
    fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, height=250, margin=dict(t=40, b=20, l=20, r=20))
    
    # Chart 2: DTI Bullet
    fig_dti = go.Figure(go.Indicator(
        mode = "number+gauge",
        value = dti_ratio * 100,
        number = {'suffix': "%", 'font': {'color': 'white'}},
        title = {'text': "Debt Load (DTI)", 'font': {'color': 'white'}},
        gauge = {
            'shape': "bullet",
            'axis': {'range': [0, 100], 'tickcolor': "white"},
            'bar': {'color': "white"}, # White marker
            'bgcolor': "rgba(255,255,255,0.1)",
            'steps': [
                {'range': [0, 30], 'color': "#14532d"}, # Green
                {'range': [30, 45], 'color': "#ca8a04"}, # Yellow
                {'range': [45, 100], 'color': "#7f1d1d"} # Red
            ],
            'threshold': {'line': {'color': "cyan", 'width': 2}, 'thickness': 0.75, 'value': 40}
        }
    ))
    fig_dti.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, height=250, margin=dict(t=40, b=20, l=20, r=20))

    return html_result, fig_gauge, fig_dti

# 4. DARK UI LAYOUT
# ---------------------------------------------------------
# Force a dark theme via CSS
custom_css = """
body { background-color: #0f172a; color: white; }
.gradio-container { background-color: #0f172a !important; border: none; }
.panel { background-color: #1e293b !important; border: 1px solid #334155 !important; border-radius: 10px; padding: 20px; }
label { color: #e2e8f0 !important; }
span { color: #e2e8f0 !important; }
"""

with gr.Blocks(theme=gr.themes.Base(), css=custom_css, title="Fintech Risk AI") as demo:
    
    with gr.Row():
        gr.Markdown("# 🏦 NovaBank Risk Terminal", elem_id="header")
    
    with gr.Row():
        
        # LEFT: CONTROLS
        with gr.Column(scale=1, elem_classes="panel"):
            gr.Markdown("### 📝 Applicant Data")
            
            with gr.Tabs():
                with gr.TabItem("👤 Profile"):
                    age = gr.Slider(18, 80, label="Age", value=30)
                    home = gr.Dropdown(['Rent', 'Mortgage', 'Own'], label="Housing", value="Rent")
                    emp = gr.Slider(0, 40, label="Exp (Yrs)", value=5)
                
                with gr.TabItem("💰 Income"):
                    income = gr.Number(label="Income ($)", value=55000)
                    loan = gr.Number(label="Loan ($)", value=15000)
                    intent = gr.Dropdown(['Personal', 'Education', 'Medical', 'Venture'], label="Purpose", value="Personal")
                
                with gr.TabItem("💳 History"):
                    score = gr.Slider(300, 850, label="FICO Score", value=680)
                    default = gr.Radio(["No", "Yes"], label="Past Default?", value="No")

            gr.Markdown("<br>")
            btn = gr.Button("RUN ANALYSIS", variant="primary")
            
        # RIGHT: DASHBOARD
        with gr.Column(scale=2):
            # Status Card
            out_html = gr.HTML()
            
            # Charts Row
            with gr.Row():
                with gr.Column(elem_classes="panel"):
                    out_gauge = gr.Plot()
                with gr.Column(elem_classes="panel"):
                    out_dti = gr.Plot()

    btn.click(
        fn=predict_risk,
        inputs=[age, income, loan, score, emp, home, intent, default],
        outputs=[out_html, out_gauge, out_dti]
    )

# 5. DEPLOY
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
