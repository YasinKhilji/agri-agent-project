import streamlit as st
import json
import warnings
import os
import pandas as pd
import plotly.express as px
import re
import numpy as np
from datetime import datetime

# --- 1. Import Core Agent Logic and Executor Tool ---
try:
    # Import all functions from the executor
    from executor_tool import (
        run_crop_stage_analysis, 
        summarize_all_fields, 
        load_resources, 
        simulate_new_day,
        get_historical_stage,  # <-- NEW
        DATA_FILE,
        FEATURES_PUNJAB
    )
    from agent import run_planner
    
    warnings.filterwarnings("ignore")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    
    # Load all ML resources (TF Model, Scaler, Encoder) on startup
    load_resources()
    
except Exception as e:
    st.error(f"FATAL ERROR: Could not load core agent resources.")
    st.error(f"Error: {e}")
    st.stop()

# --- 2. RAG Dictionary (Actionable Insights) ---
@st.cache_data
def load_insights():
    """
    Retrieves the actionable insights from the external JSON knowledge base.
    """
    try:
        with open("insights.json", 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Knowledge Base file 'insights.json' not found.")
        return {}
    except json.JSONDecodeError:
        st.error("Error decoding 'insights.json'. Please check for syntax errors.")
        return {}

# Load the insights once
ACTIONABLE_INSIGHTS = load_insights()


# --- 3. Streamlit Page Configuration ---
st.set_page_config(page_title="Agri-Agent Demo", layout="wide")
st.title("🌱 Agri-Agent: Proactive Crop Monitoring")
st.markdown("---")
st.caption("Try: 'How are my crops?', 'Check Field 3', 'Show map for Field 9', 'Status for Field 7 on 2024-10-15'")

# --- 4. Agent Logic Wrapper ---
def process_user_query(user_input):
    """
    Handles the full Plan -> Execute cycle.
    """
    planner_json_output = run_planner(user_input)
    
    try:
        command = json.loads(planner_json_output)
    except json.JSONDecodeError:
        return {"type": "error", "content": f"Planner Error: Invalid JSON. Output: {planner_json_output}"}

    action = command.get("action")
    
    if action == "predict_crop_stage":
        field_id = command.get("parameters", {}).get("field_id", "Field_1")
        result_dict = run_crop_stage_analysis(field_id=field_id)
        
        if result_dict["status"] == "error":
            return {"type": "error", "content": f"Executor Error: {result_dict['message']}"}
        else:
            return {"type": "analysis", "content": result_dict}

    elif action == "generate_health_map":
        field_id = command.get("parameters", {}).get("field_id", "Field_1")
        result_dict = run_crop_stage_analysis(field_id=field_id)
        
        if result_dict["status"] == "error":
            return {"type": "error", "content": f"Executor Error: {result_dict['message']}"}
        
        stage = result_dict["stage"]
        fig = generate_infield_health_map(stage, field_id)
        return {"type": "map", "content": fig, "stage": stage, "field_id": field_id}
        
    elif action == "summarize_all_fields":
        summary_data = summarize_all_fields()
        if isinstance(summary_data, str):
            return {"type": "error", "content": f"Executor Error: {summary_data}"}
        else:
            return {"type": "summary", "content": summary_data}

    # --- NEW: Handle Historical Action ---
    elif action == "get_historical_stage":
        field_id = command.get("parameters", {}).get("field_id", "Field_1")
        date = command.get("parameters", {}).get("date")
        
        if not date:
             return {"type": "error", "content": "Planner Error: No date was provided for historical lookup."}
             
        result_dict = get_historical_stage(field_id=field_id, target_date_str=date)
        
        if result_dict["status"] == "error":
            return {"type": "error", "content": f"Executor Error: {result_dict['message']}"}
        else:
            return {"type": "historical_analysis", "content": result_dict}

    elif action == "error":
        error_message = command.get("parameters", {}).get("message", "I can't help with that.")
        return {"type": "error", "content": error_message}
    
    else:
        return {"type": "error", "content": "Unknown agent action."}

def generate_infield_health_map(stage, field_id):
    """
    Simulates a 10x10 spatial heatmap based on the field's overall stage.
    """
    if stage == "Peak":
        base_health = np.random.normal(loc=0.9, scale=0.05, size=(10, 10))
        base_health[2:4, 2:4] = 0.6
    elif stage == "Senescence":
        base_health = np.random.normal(loc=0.4, scale=0.1, size=(10, 10))
    elif stage == "Vegetative":
        base_health = np.random.normal(loc=0.7, scale=0.08, size=(10, 10))
    else:
        base_health = np.random.rand(10, 10)
        
    simulated_map_data = np.clip(base_health, 0, 1)
    
    fig = px.imshow(
        simulated_map_data,
        color_continuous_scale='RdYlGn',
        range_color=[0,1],
        title=f"Simulated In-Field Health Map for {field_id} (Stage: {stage})",
        labels=dict(color="Health (NDVI)")
    )
    fig.update_layout(coloraxis_colorbar=dict(title="Health"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    
    return fig

# --- 5. Field Comparison and Visualization (Sidebar) ---

@st.cache_data
def get_csv_data():
    df = pd.read_csv(DATA_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def create_field_comparison_plot(df, field_ids, feature='ndvi_normalized'):
    """Generates a Plotly chart comparing features of selected fields over time."""
    try:
        comparison_data = df[df['field_id'].isin(field_ids)].copy()
        if comparison_data.empty: return None
        
        fig = px.line(
            comparison_data, 
            x="Date", y=feature, color='field_id', 
            title=f"Time Series Comparison: {feature.upper()}",
            labels={feature: feature.upper(), "Date": "Time Step"}
        )
        fig.update_layout(legend_title_text='Field ID')
        return fig
    except Exception:
        return None

with st.sidebar:
    st.header("Visualizations & Comparison")
    
    if st.button("Simulate Next Day", use_container_width=True):
        with st.spinner("Simulating new data and updating CSV..."):
            sim_result = simulate_new_day()
            get_csv_data.clear() 
            st.success(sim_result)
            
    st.markdown("---")
    
    try:
        df = get_csv_data()
        all_field_ids = sorted(df['field_id'].unique().tolist())
        selected_fields = st.multiselect(
            "Select Fields for Comparison:", options=all_field_ids,
            default=all_field_ids[:2] if len(all_field_ids) >= 2 else all_field_ids, max_selections=3)
        feature_options = ['ndvi_normalized', 'evi_normalized', 'savi_normalized', 'Temperature_C']
        selected_feature = st.selectbox("Select Feature to Plot:", options=feature_options)
        
        if selected_fields:
            plot = create_field_comparison_plot(df, selected_fields, selected_feature)
            if plot: st.plotly_chart(plot, use_container_width=True)
    except Exception as e:
        st.error(f"Could not load data for sidebar: {e}")

# --- 6. Main Chat & Proactive Alerting Logic ---

if "messages" not in st.session_state:
    st.session_state.messages = []
if "field_states" not in st.session_state:
    st.session_state.field_states = {}

def check_for_state_changes():
    print("Agent: Proactively checking field states...")
    summary_data = summarize_all_fields()
    if isinstance(summary_data, str): return

    new_states = {}
    for item in summary_data:
        field_id = item["Field ID"]
        new_stage = item["Predicted Stage"]
        
        if field_id in st.session_state.field_states:
            old_stage = st.session_state.field_states[field_id]
            if new_stage != old_stage and old_stage != "N/A":
                st.toast(f"🔔 **ALERT:** `{field_id}` has changed from '{old_stage}' to **'{new_stage}'**!")
        
        new_states[field_id] = new_stage
        
    st.session_state.field_states = new_states

check_for_state_changes()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["type"] == "text":
            st.markdown(message["content"])
        elif message["type"] == "summary_table":
            st.dataframe(message["content"])
        elif message["type"] == "analysis_card":
            st.markdown(f"**Field:** `{message['field_id']}`")
            st.metric(label="Predicted Stage", value=message['stage'])
            st.progress(message['confidence_val'], text=f"Confidence: {message['confidence_text']}")
            st.info(message['insight'])
            with st.expander("Click to see data used for this prediction"):
                st.markdown("The LSTM model made this prediction based on the following 4-feature data sequence (most recent data last):")
                df_data = pd.DataFrame(message['data_used'], columns=FEATURES_PUNJAB)
                st.dataframe(df_data)
        elif message["type"] == "map":
            st.markdown(f"Here is the simulated health map for **{message['field_id']}** (Stage: {message['stage']}):")
            st.plotly_chart(message["content"], use_container_width=True)
        # --- NEW: Display for Historical Analysis ---
        elif message["type"] == "historical_card":
            st.markdown(f"**Historical Analysis for `{message['field_id']}`**")
            st.markdown(f"*(Data from {message['closest_date']} is the closest to your request of {message['req_date']})*")
            st.metric(label="Predicted Stage", value=message['stage'])
            st.progress(message['confidence_val'], text=f"Confidence: {message['confidence_text']}")
            st.info(message['insight'])
            with st.expander("Click to see data used for this prediction"):
                st.markdown("The LSTM model made this prediction based on the following 4-feature data sequence (ending on the closest date):")
                df_data = pd.DataFrame(message['data_used'], columns=FEATURES_PUNJAB)
                st.dataframe(df_data)


# Process new user input
if prompt := st.chat_input("Ask about crop health (e.g., 'Check Field 5')"):
    
    st.session_state.messages.append({"role": "user", "type": "text", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Agent is running Plan & Execute cycle..."):
        agent_output = process_user_query(prompt)
    
    with st.chat_message("assistant"):
        if agent_output['type'] == 'error':
            response_content = f"⚠️ {agent_output['content']}"
            st.session_state.messages.append({"role": "assistant", "type": "text", "content": response_content})
            st.markdown(response_content)
        
        elif agent_output['type'] == 'summary':
            response_content = "Here is the summary for all fields:"
            df_summary = pd.DataFrame(agent_output['content'])
            st.session_state.messages.append({"role": "assistant", "type": "text", "content": response_content})
            st.session_state.messages.append({"role": "assistant", "type": "summary_table", "content": df_summary})
            st.markdown(response_content)
            st.dataframe(df_summary)
            
        elif agent_output['type'] == 'analysis':
            try:
                data = agent_output['content']
                field_id = data['field_id']
                stage = data['stage']
                confidence_val = data['confidence'] / 100.0
                confidence_text = f"{data['confidence']:.1f}%"
                insight = ACTIONABLE_INSIGHTS.get(stage, "No specific insight available.")
                data_used = data['data_used']

                st.session_state.messages.append({
                    "role": "assistant", "type": "analysis_card",
                    "field_id": field_id, "stage": stage,
                    "confidence_val": confidence_val, "confidence_text": confidence_text,
                    "insight": insight,
                    "data_used": data_used
                })
                
                st.markdown(f"**Field:** `{field_id}`")
                st.metric(label="Predicted Stage", value=stage)
                st.progress(confidence_val, text=f"Confidence: {confidence_text}")
                st.info(insight)

                with st.expander("Click to see data used for this prediction"):
                    st.markdown("The LSTM model made this prediction based on the following 4-feature data sequence (most recent data last):")
                    # Create a DataFrame to display the data neatly
                    df_data = pd.DataFrame(data_used, columns=FEATURES_PUNJAB)
                    st.dataframe(df_data)
                
            except Exception as e:
                st.error(f"Error parsing analysis dict: {e}")
                
        elif agent_output['type'] == 'map':
            fig = agent_output['content']
            stage = agent_output['stage']
            field_id = agent_output['field_id']
            
            st.session_state.messages.append({
                "role": "assistant", "type": "map",
                "content": fig, "field_id": field_id, "stage": stage
            })
            
            st.markdown(f"Here is the simulated health map for **{field_id}** (Stage: {stage}):")
            st.plotly_chart(fig, use_container_width=True)

        # --- NEW: Handle Historical Analysis ---
        elif agent_output['type'] == 'historical_analysis':
            try:
                data = agent_output['content']
                field_id = data['field_id']
                stage = data['stage']
                confidence_val = data['confidence'] / 100.0
                confidence_text = f"{data['confidence']:.1f}%"
                insight = ACTIONABLE_INSIGHTS.get(stage, "No specific insight available.")
                req_date = data['requested_date']
                closest_date = data['closest_date_in_data']
                data_used = data['data_used']
                
                st.session_state.messages.append({
                    "role": "assistant", "type": "historical_card",
                    "field_id": field_id, "stage": stage,
                    "confidence_val": confidence_val, "confidence_text": confidence_text,
                    "insight": insight,
                    "req_date": req_date,
                    "closest_date": closest_date,
                    "data_used": data_used
                    
                })
                
                # Display the structured data
                st.markdown(f"**Historical Analysis for `{field_id}`**")
                st.markdown(f"*(Data from {closest_date} is the closest to your request of {req_date})*")
                st.metric(label="Predicted Stage", value=stage)
                st.progress(confidence_val, text=f"Confidence: {confidence_text}")
                st.info(insight)

                with st.expander("Click to see data used for this prediction"):
                            st.markdown("The LSTM model made this prediction based on the following 4-feature data sequence (ending on the closest date):")
                            df_data = pd.DataFrame(data_used, columns=FEATURES_PUNJAB)
                            st.dataframe(df_data)
                
            except Exception as e:
                st.error(f"Error parsing historical analysis dict: {e}")