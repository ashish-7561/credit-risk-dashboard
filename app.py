import gradio as gr
import joblib
import numpy as np
import plotly.graph_objects as go
import os  # Required for Cloud Deployment (Render)

# ==========================================
# 1. ASSET LOADING
# ==========================================
# We load the trained model, scaler, and label encoders.
# Uses a try-except block to prevent crashing if files are missing during setup.
try:
    model = joblib.load('credit_risk_model.pkl')
    scaler = joblib.load('credit_scaler.pkl')
    le_home = joblib.load('le_home.pkl')
    le_intent = joblib.load('le_intent.pkl')
    print("✅ All Model Assets Loaded Successfully!")
except Exception as e:
    print(f"⚠️ Warning: Could not load model files. Error: {e}")
    model = None

# ==========================================
# 2. PREDICTION ENGINE
# ==========================================
def predict_credit_risk(age, income, loan_amount, credit_score, emp_length, home_ownership, loan_intent, prev_default):
    
    # Check if model exists before running
    if model is None:
        return "⚠️ Error: Model files not found. Please ensure .pkl files are uploaded.", None

    # --- A. Data Preprocessing ---
    
    # 1. Handle "Previous Default" (Yes/No -> 1/0)
    default_val = 1 if prev_default == "Yes" else 0
    
    # 2. Calculate Debt-to-Income Ratio (DTI)
    # Adding +1 to income to avoid division by zero errors
    dti_ratio = loan_amount / (income + 1)
    
    # 3. Encode Categorical Strings (Home & Intent)
    # We use [0] because the encoder returns a list
    home_encoded = le_home.transform([home_ownership])[0]
    intent_encoded = le_intent.transform([loan_intent])[0]
    
    # 4. Create Feature Array
    # IMPORTANT: The order MUST match the training columns exactly:
    # ['Age', 'Income', 'Loan_Amount', 'Credit_Score', 'Employment_Length', 'Home_Ownership', 'Loan_Intent', 'Previous_Defaults', 'DTI_Ratio']
    features = np.array([[age, income, loan_amount, credit_score, emp_length, home_encoded, intent_encoded, default_val, dti_ratio]])
    
    # 5. Scale the features using the saved scaler
    features_scaled = scaler.transform(features)
    
    # --- B. Model Inference ---
    prediction = model.predict(features_scaled)
    probs = model.predict_proba(features_scaled)
    
    # Extract probabilities
    risk_prob = probs[0][1] * 100 # Probability of Default (Risk)
    safe_prob = probs[0][0] * 100 # Probability of Repayment (Safe)
    
    # --- C. Generate Output Report ---
    if prediction[0] == 1:
        # High Risk Case
        title = "🚨 HIGH RISK APPLICATION"
        desc = (
            f"**Default Probability:** {risk_prob:.1f}%\n"
            f"**Debt-to-Income Ratio:** {dti_ratio:.2f}\n\n"
            "**⚠️ RECOMMENDATION:** \n"
            "This applicant shows strong signs of default risk. \n"
            "Action: **REJECT** application or require 50% collateral."
        )
    else:
        # Low Risk Case
        title = "✅ LOW RISK - APPROVED"
        desc = (
            f"**Repayment Probability:** {safe_prob:.1f}%\n"
            f"**Debt-to-Income Ratio:** {dti_ratio:.2f}\n\n"
            "**👍 RECOMMENDATION:** \n"
            "This applicant has a solid financial profile. \n"
            "Action: **APPROVE** standard loan terms."
        )
        
    # --- D. Visual Gauge Chart (Plotly) ---
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = credit_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Applicant Credit Health"},
        gauge = {
            'axis': {'range': [300, 850]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [300, 600], 'color': "#ef4444"},  # Red (Risk)
                {'range': [600, 750], 'color': "#eab308"},  # Yellow (Caution)
                {'range': [750, 850], 'color': "#22c55e"}   # Green (Good)
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 600 # The "Danger Zone" mark
            }
        }
    ))
    
    # Clean up the chart look
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    
    return f"# {title}\n\n{desc}", fig

# ==========================================
# 3. GRADIO UI LAYOUT
# ==========================================
# Custom CSS for a professional "Banking" look
custom_css = """
.container { max-width: 1200px; margin: auto; }
.gr-button-primary { background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%); border: none; }
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), css=custom_css, title="Credit Risk AI") as demo:
    
    # Header
    with gr.Row():
        gr.Markdown(
            """
            # 🏦 Enterprise Credit Risk System
            ### 🤖 Advanced ML Default Prediction Engine
            """
        )
    
    # Main Content
    with gr.Row():
        
        # --- Left Column: Inputs ---
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### 📝 Application Form")
            gr.Markdown("Enter applicant financial details below.")
            
            with gr.Group():
                age = gr.Slider(21, 65, step=1, label="Applicant Age", value=30)
                
                with gr.Row():
                    income = gr.Number(label="Annual Income ($)", value=55000)
                    loan_amt = gr.Number(label="Loan Amount Requested ($)", value=15000)
                
                credit_score = gr.Slider(300, 850, step=1, label="Credit Score (FICO)", value=680)
                emp_length = gr.Slider(0, 40, step=1, label="Employment Length (Years)", value=5)
                
                home = gr.Dropdown(['Rent', 'Mortgage', 'Own'], label="Home Ownership Status", value="Rent")
                intent = gr.Dropdown(['Personal', 'Education', 'Medical', 'Venture'], label="Loan Purpose", value="Personal")
                default = gr.Radio(["No", "Yes"], label="Has Previous Default?", value="No")
            
            btn = gr.Button("🚀 Run Risk Analysis", variant="primary", size="lg")
            
        # --- Right Column: Results ---
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Decision Dashboard")
            
            # Text Report
            out_text = gr.Markdown("Waiting for input...")
            
            # Gauge Chart
            out_plot = gr.Plot(label="Credit Health Indicator")

    # Connect the Button to the Logic
    btn.click(
        fn=predict_credit_risk,
        inputs=[age, income, loan_amt, credit_score, emp_length, home, intent, default],
        outputs=[out_text, out_plot]
    )

# ==========================================
# 4. DEPLOYMENT CONFIG (CRITICAL FOR CLOUD)
# ==========================================
if __name__ == "__main__":
    # This block allows the app to run on Render, Heroku, or locally.
    # It reads the PORT environment variable (required by cloud) or defaults to 7860.
    port = int(os.environ.get("PORT", 7860))
    
    # Launch with specific server settings
    demo.launch(server_name="0.0.0.0", server_port=port)
