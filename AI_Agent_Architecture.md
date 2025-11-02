# Agri-Agent: AI Agent Architecture Document

**Author:** Khilji Mohammed Yasin
**Project:** Submission for the I'mbesideyou Data Scientist Internship (2026)

---

## 1. Project Goal

The **Agri-Agent** is a proactive, multi-agent AI system designed to automate the manual university task of monitoring crop health and growth stages using time-series data. It is built to understand natural language, execute specialized data science tasks, and provide actionable, interpretable insights to a user.

## 2. Core Architecture: Planner-Executor Pattern

This system is built using a **Planner-Executor** multi-agent architecture. This design is crucial as it separates the "reasoning" (understanding user intent) from the "doing" (running complex data science models).

* **The Planner (`agent.py`):** Acts as the "brain." Its sole responsibility is to receive an unstructured natural language query from the user (e.g., "show me a map of field 3") and translate it into a strict, machine-readable JSON command.

* **The Executor (`executor_tool.py`):** Acts as the "specialist tool." It is a collection of functions that the Planner can call. It receives the JSON command, executes the correct function (e.g., `run_crop_stage_analysis`), and returns the result.



## 3. Information & Tool-Use Flow

The agent operates in a continuous "Reason -> Plan -> Execute" loop.

1.  **Reasoning:** The user's query (e.g., "what was the stage for field 7 on 2024-10-15") is sent to the **Planner Agent**.
2.  **Planning:** The Planner analyzes the intent. It extracts key entities (like `Field_7` and `2024-10-15`) and determines the correct tool to use. It generates a specific JSON command:
    ```json
    {
      "action": "get_historical_stage",
      "parameters": {
        "field_id": "Field_7",
        "date": "2024-10-15"
      }
    }
    ```
3.  **Execution:** The main application routes this command to the **Executor Agent**. The Executor calls its internal `get_historical_stage()` function, which:
    a. Loads the `advanced_labeled_dataset.csv`.
    b. Finds the data row closest to the requested date.
    c. Grabs the historical sequence of data *before* that date.
    d. Runs the pre-trained Keras LSTM model on this sequence.
    e. Returns a structured Python dictionary with the prediction.
4.  **Response Generation:** The main application receives this dictionary and performs **Retrieval-Augmented Generation (RAG)** by retrieving the correct advice from `insights.json`. It then formats this complex data into a user-friendly "analysis card" in the UI.

## 4. Implemented Tools (Executor Capabilities)

The Executor agent has four distinct tools that the Planner can call:

* **`predict_crop_stage(field_id)`:** Gets the *current* stage for a single field using the latest data.
* **`summarize_all_fields()`:** Runs the model on all 9 fields and returns a full summary.
* **`get_historical_stage(field_id, date)`:** Runs the model on a *past* data sequence.
* **`simulate_new_day()`:** A utility tool that appends new simulated, stage-aware data to the CSV to demonstrate the agent's dynamic capabilities.