# executor_tool.py

# --- Imports ---
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import os
import warnings

# Suppress TensorFlow/Keras warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='keras')
tf.get_logger().setLevel('ERROR')

print("Executor Tool: Initializing...")

# --- Global Variables ---
model = None
scaler = None
encoder = None
stage_map = None
MODEL_FILE = 'crop_stage_lstm_model.h5'
DATA_FILE = 'advanced_labeled_dataset.csv'
# Define the exact features the Punjab LSTM model was trained on
# Make sure this list is 100% correct based on your notebook
# Example - Use your actual feature names!
FEATURES_PUNJAB = ['ndvi_normalized', 'evi_normalized', 'savi_normalized', 'Temperature_C']
TARGET_PUNJAB = 'growth_stage'

# --- Load Model and Preprocessors ---
def load_resources():
    """Loads the ML model, scaler, and encoder."""
    global model, scaler, encoder, stage_map
    try:
        if not os.path.exists(MODEL_FILE):
            print(f"Executor Tool: ERROR - Model file '{MODEL_FILE}' not found.")
            return False
        if not os.path.exists(DATA_FILE):
            print(f"Executor Tool: ERROR - Data file '{DATA_FILE}' not found for scaler/encoder.")
            return False

        model = tf.keras.models.load_model(MODEL_FILE)

        # Re-create scaler/encoder from the original Punjab data
        df_punjab = pd.read_csv(DATA_FILE)

        # Fit the scaler only on the feature columns used during training
        scaler = MinMaxScaler()
        scaler.fit(df_punjab[FEATURES_PUNJAB])

        # Fit the encoder on the target column
        encoder = LabelEncoder()
        encoder.fit(df_punjab[TARGET_PUNJAB])
        stage_map = {i: cls for i, cls in enumerate(encoder.classes_)} # e.g., {0: 'Peak', 1: 'Senescence', ...}

        print("Executor Tool: Model and preprocessors loaded successfully.")
        return True

    except Exception as e:
        print(f"Executor Tool: ERROR loading model or preprocessors - {e}")
        return False

def run_crop_stage_analysis(field_id="Field_1"):
    """
    Runs the crop stage prediction and returns a structured dictionary.
    """
    global model, scaler, encoder, stage_map, DATA_FILE, FEATURES_PUNJAB
    
    if model is None or scaler is None or encoder is None:
        return {"status": "error", "message": "Model or preprocessors not loaded."}

    print(f"Executor Tool: Received request for Field ID '{field_id}'. Running analysis...")

    try:
        df_punjab = pd.read_csv(DATA_FILE)
        sequence_length = model.input_shape[1]
        num_model_features = model.input_shape[2] # 6
        num_csv_features = len(FEATURES_PUNJAB) # 4

        field_data = df_punjab[df_punjab['field_id'] == field_id].sort_values(by='Date').reset_index(drop=True)
        
        if len(field_data) < sequence_length:
            return {"status": "error", "message": f"Not enough data for {field_id} (need {sequence_length} timesteps)."}

        sequence_raw_4_features = field_data[FEATURES_PUNJAB].tail(sequence_length).values
        scaled_features_2d = scaler.transform(sequence_raw_4_features)
        
        final_sequence_for_lstm = np.zeros((1, sequence_length, num_model_features))
        final_sequence_for_lstm[0, :, :num_csv_features] = scaled_features_2d

        # --- Make Prediction ---
        predictions = model.predict(final_sequence_for_lstm, verbose=0)
        prediction_encoded = np.argmax(predictions, axis=1)[0]
        prediction_decoded = stage_map.get(prediction_encoded, "Unknown Stage")
        confidence = float(np.max(predictions) * 100) # Converted to float

        print(f"Executor Tool: Analysis complete for {field_id}.")
        
        # --- Return the full dictionary ---
        return {
            "status": "success",
            "field_id": field_id,
            "stage": prediction_decoded,
            "confidence": confidence,
            "data_used": sequence_raw_4_features.tolist() # The XAI data
        }

    except Exception as e:
        import traceback
        print(f"Executor Tool: Error during prediction for {field_id} - {e}")
        # traceback.print_exc()
        return {"status": "error", "message": f"Error running analysis for {field_id}."}

def summarize_all_fields():
    """
    Runs analysis on ALL fields and returns a summary.
    """
    global model, scaler, encoder, stage_map, DATA_FILE
    
    print("Executor Tool: Received request for full summary...")
    
    if model is None:
        return "Error: Model not loaded."
        
    try:
        df_punjab = pd.read_csv(DATA_FILE)
        all_field_ids = sorted(df_punjab['field_id'].unique().tolist())
        
        sequence_length = model.input_shape[1]
        num_model_features = model.input_shape[2] # 6
        num_csv_features = len(FEATURES_PUNJAB) # 4
        
        results = []
        
        for field_id in all_field_ids:
            field_data = df_punjab[df_punjab['field_id'] == field_id].sort_values(by='Date')
            
            if len(field_data) < sequence_length:
                results.append({"Field ID": field_id, "Predicted Stage": "N/A", "Confidence": 0, "Status": f"Not enough data (need {sequence_length})"})
                continue

            # Get latest sequence
            sequence_raw_4_features = field_data[FEATURES_PUNJAB].tail(sequence_length).values
            scaled_features_2d = scaler.transform(sequence_raw_4_features)
            
            # Build 6-feature input
            final_sequence_for_lstm = np.zeros((1, sequence_length, num_model_features))
            final_sequence_for_lstm[0, :, :num_csv_features] = scaled_features_2d
            # Last 2 features (simulated) remain 0

            # Make Prediction
            predictions = model.predict(final_sequence_for_lstm, verbose=0)
            prediction_encoded = np.argmax(predictions, axis=1)[0]
            prediction_decoded = stage_map.get(prediction_encoded, "Unknown Stage")
            confidence = np.max(predictions) * 100
            
            results.append({"Field ID": field_id, "Predicted Stage": prediction_decoded, "Confidence": f"{confidence:.1f}%"})

        print("Executor Tool: Full summary complete.")
        # Return a list of dictionaries, which we will turn into a table
        return results

    except Exception as e:
        return f"Error creating summary: {e}"
def simulate_new_day():
    """
    Loads the CSV, simulates a new day of data for all fields, and appends it.
    This makes the agent's data dynamic for the demo.
    """
    try:
        df = pd.read_csv(DATA_FILE)
        
        # Get the most recent date
        last_date = pd.to_datetime(df['Date']).max()
        new_date = last_date + pd.Timedelta(days=7) # Add 7 days
        
        all_field_ids = df['field_id'].unique()
        new_rows = []
        
        for field_id in all_field_ids:
            # Get the last row of data for this field
            last_row = df[df['field_id'] == field_id].iloc[-1].copy()
            
            # Update the date
            last_row['Date'] = new_date.strftime('%Y-%m-%d')
            
            # Simulate a slight increase in indices (basic "growth")
            # This is a very simple simulation
            last_row['ndvi_normalized'] = min(last_row['ndvi_normalized'] * 1.05, 1.0) # 5% growth, capped at 1.0
            last_row['evi_normalized'] = min(last_row['evi_normalized'] * 1.05, 1.0)
            last_row['savi_normalized'] = min(last_row['savi_normalized'] * 1.05, 1.0)
            
            # Simulate a slight change in temp
            last_row['Temperature_C'] = last_row['Temperature_C'] + (np.random.rand() - 0.5) * 2 # change by +/- 1 degree
            
            new_rows.append(last_row)
            
        # Append all new rows to the main dataframe
        new_df = pd.DataFrame(new_rows)
        combined_df = pd.concat([df, new_df], ignore_index=True)
        
        # Save the updated CSV
        combined_df.to_csv(DATA_FILE, index=False)
        
        return f"Successfully simulated new data for {len(all_field_ids)} fields for date {new_date.date()}."

    except Exception as e:
        return f"Error during simulation: {e}"

def get_historical_stage(field_id, target_date_str):
    """
    Runs crop stage prediction for a specific field ID at a specific historical date.
    """
    global model, scaler, encoder, stage_map, DATA_FILE, FEATURES_PUNJAB
    
    if model is None:
        return {"status": "error", "message": "Model or preprocessors not loaded."}

    print(f"Executor Tool: Received historical request for {field_id} on {target_date_str}...")

    try:
        df_punjab = pd.read_csv(DATA_FILE)
        df_punjab['Date'] = pd.to_datetime(df_punjab['Date']) # Convert date column
        
        sequence_length = model.input_shape[1]
        num_model_features = model.input_shape[2] # 6
        num_csv_features = len(FEATURES_PUNJAB) # 4

        # 1. Filter data
        field_data = df_punjab[df_punjab['field_id'] == field_id].sort_values(by='Date').reset_index(drop=True)
        if field_data.empty:
            return {"status": "error", "message": f"No data found for {field_id}."}

        # 2. Find closest date
        target_date = pd.to_datetime(target_date_str)
        date_diff = (field_data['Date'] - target_date).abs()
        closest_index = date_diff.idxmin() 
        closest_date_in_data = field_data.loc[closest_index, 'Date']
        
        # 3. Check for enough data
        if closest_index < (sequence_length - 1):
            return {"status": "error", "message": f"Not enough historical data for {field_id} before {closest_date_in_data.date()}. Need at least {sequence_length} data points."}

        # 4. Get historical sequence
        start_index = closest_index - sequence_length + 1
        end_index = closest_index + 1 
        
        sequence_slice_df = field_data.iloc[start_index:end_index]
        sequence_raw_4_features = sequence_slice_df[FEATURES_PUNJAB].values
        
        if sequence_raw_4_features.shape[0] != sequence_length:
             return {"status": "error", "message": "Data slicing error during historical lookup."}

        # 5. Scale and build
        scaled_features_2d = scaler.transform(sequence_raw_4_features)
        
        final_sequence_for_lstm = np.zeros((1, sequence_length, num_model_features))
        final_sequence_for_lstm[0, :, :num_csv_features] = scaled_features_2d

        # 6. Make Prediction
        predictions = model.predict(final_sequence_for_lstm, verbose=0)
        prediction_encoded = np.argmax(predictions, axis=1)[0]
        prediction_decoded = stage_map.get(prediction_encoded, "Unknown Stage")
        confidence = float(np.max(predictions) * 100) # Converted to float

        print(f"Executor Tool: Historical analysis complete for {field_id}.")
        
        # --- Return the full dictionary ---
        return {
            "status": "success",
            "field_id": field_id,
            "stage": prediction_decoded,
            "confidence": confidence,
            "requested_date": target_date_str,
            "closest_date_in_data": closest_date_in_data.strftime('%Y-%m-%d'),
            "data_used": sequence_raw_4_features.tolist() # The XAI data
        }

    except Exception as e:
        return {"status": "error", "message": f"Error running historical analysis: {e}"}
# --- Load resources when the script is imported/run ---
load_resources()

# --- Optional: Test the function directly ---
if __name__ == '__main__':
    # Make sure these files are in the same directory as executor_tool.py:
    # 1. crop_stage_lstm_model.h5
    # 2. advanced_labeled_dataset.csv
    print("\n--- Testing executor_tool.py directly ---")
    if model: # Only run test if model loaded successfully
        output = run_crop_stage_analysis(location="Punjab Test Field")
        print("Direct Test Output:", output)
    else:
        print("Skipping direct test because model failed to load.")