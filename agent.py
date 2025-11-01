import json
import warnings
import os
import time

# Import our specialized executor tool
try:
    from executor_tool import run_crop_stage_analysis, model as executor_model
except ImportError:
    print("FATAL: executor_tool.py not found.")
    print("Please make sure executor_tool.py is in the same directory.")
    exit()

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- 1. Define the Rule-Based Planner Agent (Updated for Date-Based Retrieval) ---

def run_planner(user_prompt):
    """
    Rule-based planner that now understands historical date queries.
    """
    import re
    import time
    
    print("Agent: Thinking... (Running Rule-Based Planner)")
    time.sleep(0.5) 
    
    prompt_lower = user_prompt.lower()
    
    # --- Entity Extraction ---
    field_match = re.search(r'field[_\s]?(\d+)', prompt_lower)
    field_id = f"Field_{field_match.group(1).strip()}" if field_match else "Field_1" 

    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', prompt_lower)
    target_date = date_match.group(1) if date_match else None

    # --- Keyword Definitions ---
    analysis_keywords = [
        "farm", "crop", "stage", "analysis", "growth", "health", "field",
        "what", "which", "tell", "show", "check", "how's", "how are"
    ]
    summary_phrases = [
        "how are my crops", "how are crops", "how's the farm", "summarize all fields", 
        "show all crops", "farm status", "check my crops"
    ]
    map_keywords = ["map", "heatmap", "spatial health"]
    error_keywords = ["weather", "joke", "time", "who are you", "hello", "hi", "rain", "fertilizer"]

    # --- NEW LOGIC ORDER (Most specific rules first) ---

    # Rule 1: Check for known error keywords.
    if any(keyword in prompt_lower for keyword in error_keywords):
        return "{\"action\": \"error\", \"parameters\": {\"message\": \"I am an agriculture agent. I can only provide crop health and stage information.\"}}"

    # Rule 2: Check for map-specific requests.
    if any(keyword in prompt_lower for keyword in map_keywords):
        return f'{{"action": "generate_health_map", "parameters": {{"field_id": "{field_id}"}}}}'
    
    # NEW Rule 3: Check for a historical date query.
    if target_date and any(keyword in prompt_lower for keyword in analysis_keywords):
        return f'{{"action": "get_historical_stage", "parameters": {{"field_id": "{field_id}", "date": "{target_date}"}}}}'

    # Rule 4: Check for specific field analysis (current data).
    if field_match and any(keyword in prompt_lower for keyword in analysis_keywords):
        return f'{{"action": "predict_crop_stage", "parameters": {{"field_id": "{field_id}"}}}}'
    
    # Rule 5: Check for a general summary request (current data).
    if any(phrase in prompt_lower for phrase in summary_phrases):
        return "{\"action\": \"summarize_all_fields\", \"parameters\": {}}"
    
    # Rule 6: Default fallback
    return "{\"action\": \"error\", \"parameters\": {\"message\": \"I'm sorry, I can only help with summaries or specific field status. For historical data, please ask using a YYYY-MM-DD date format.\"}}"

# --- 2. Define the Main Agent Loop ---

def main():
    print("\n" + "="*50)
    print("🌱 Agri-Agent Initialized (Rule-Based Planner) 🌱")
    print("="*50)
    
    if executor_model is None:
        print("WARNING: The Executor's LSTM model failed to load.")
        print("The agent will only be able to return errors.")
    else:
        print("Planner and Executor are both loaded. Ready for commands.")
        
    print("\nType 'exit' or 'quit' to end the session.")

    while True:
        print("\n" + "-"*50)
        # 1. Get User Input
        user_input = input("You: ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("\nAgri-Agent: Shutting down. Goodbye!")
            break
            
        # 2. REASON/PLAN: Call the Planner
        planner_json_output = run_planner(user_input)
        
        try:
            command = json.loads(planner_json_output)
            print(f"Agent: Planner produced command: {command}")
        except Exception as e:
            # This should never happen with our new reliable planner
            print(f"Agent: Planner produced invalid JSON. Output: {planner_json_output}")
            print("Agent: I'm sorry, I had trouble understanding that.")
            continue

        # 3. EXECUTE: Parse the command and call the right tool
        if command.get("action") == "predict_crop_stage":
            # Call our executor_tool.py function
            print("Agent: Executing crop stage analysis...")
            # Pass the field_id extracted from the Planner's JSON command
            field_id_to_check = command.get("parameters", {}).get("field_id", "Field_1")
            result = run_crop_stage_analysis(field_id=field_id_to_check)
            print(f"\nAgri-Agent: {result}")
            
        elif command.get("action") == "error":
            # The planner decided the query was invalid
            error_message = command.get("parameters", {}).get("message", "I can't help with that.")
            print(f"\nAgri-Agent: {error_message}")
            
        else:
            print("\nAgri-Agent: I'm not sure how to handle that action.")


if __name__ == "__main__":
    main()