# Agri-Agent: A Proactive, Multi-Agent AI System
**Submission for the I'mbesideyou Data Scientist Internship (2026)**

* **Name:** Khilji Mohammed Yasin
* **University:** IIT Palakkad
* **Department:** Data Science and Engineering

---

## 🚀 Live Demo & Video

* **Live Deployed App:** [**https://agri-agent-project-jqvfqjr2mx5hh8wveblurm.streamlit.app/**](https://agri-agent-project-jqvfqjr2mx5hh8wveblurm.streamlit.app/)
* **Demo Video (3 min):** [**Watch the Full Demo Here**](https://youtu.be/INQhj65QKkA)

---

## 1. Project Goal

This project successfully automates the university-level task of monitoring crop health by building a **proactive, multi-agent AI system**. The agent understands natural language queries, executes complex data science tasks using a pre-trained LSTM model, and provides actionable insights and visualizations in a user-friendly web interface.

## 2. AI Agent Architecture: Planner-Executor

This system uses a **Planner-Executor** design to separate "reasoning" from "doing."

* **Planner Agent (`agent.py`):** The "brain" of the operation. It receives user queries (e.g., "show map for field 3") and translates them into structured JSON commands for the Executor. This was pivoted from a fine-tuned LLM to a **100% reliable Rule-Based Planner** to ensure robust, deterministic performance.

* **Executor Agent (`executor_tool.py`):** The "specialist tool." It receives JSON commands and executes specific functions. Its core is the **integration of a pre-trained Keras/LSTM model** (from my OELP project) which predicts crop stages based on time-series data.

For a complete technical breakdown, see the **[AI_Agent_Architecture.md](AI_Agent_Architecture.md)** file.

## 3. Core Features & Bonus Enhancements

This project fulfills all mandatory requirements and implements **all three bonus categories**, plus several advanced, self-driven features.

### ✅ Core Features
* **Fine-Tuned Model:** A LoRA model (`finetune.py`) was successfully trained to parse commands. After evaluation, it was pivoted to a more robust Rule-Based Planner. This entire process is documented in the **[Data_Science_Report.md](Data_Science_Report.md)**.
* **Evaluation Metrics:** The system is evaluated on two metrics:
    1.  **Planner Reliability (100%):** The Rule-Based Planner has 100% accuracy in routing supported commands.
    2.  **Executor Quality (87.5%):** The underlying LSTM model has a pre-validated accuracy of 87.5%.

### ✨ Bonus Features Implemented
* **Multi-Agent Collaboration:** The Planner-Executor design is the core of the project.
* **Custom Tools & RAG:**
    * **Custom Tool:** The `executor_tool.py` is a custom tool that wraps the Keras LSTM model.
    * **True RAG:** The agent provides **Actionable Insights** (e.g., watering advice) by retrieving them from an external `insights.json` knowledge base.
* **User Interface (Streamlit):** A full web application (`app.py`) was built and deployed.

### 🌟 Advanced Self-Driven Features
* **Historical Analysis:** The agent can parse date-based queries (e.g., "status on 2024-10-15") and run the LSTM model on past data.
* **Data Simulation:** A "Simulate Next Day" button makes the app dynamic by adding new, *stage-aware* simulated data to the CSV.
* **Proactive Alerting:** The agent is **stateful**. It remembers the last stage of all fields and sends a **live 🔔 alert** if a simulation causes a field's stage to change.
* **Explainable AI (XAI):** Every prediction includes an expander that shows the **actual data sequence** the LSTM model used to make its decision.
* **Spatial Heatmap Simulation:** The agent can generate and display a **simulated 10x10 in-field heatmap** (using Plotly) to visualize in-field health variance.

## 4. How to Run Locally

1.  Clone this repository.
2.  Install all required libraries from the requirements file:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

## 5. Project Deliverables

| Deliverable | File / Location | Description |
| :--- | :--- | :--- |
| **Source Code** | `app.py` | The main Streamlit web application and agent logic. |
| | `agent.py` | The Rule-Based Planner module. |
| | `executor_tool.py` | The Executor module (Custom Tool) that wraps the LSTM model. |
| | `finetune.py` | The script used to train the (experimental) AI Planner. |
| **Data & Models** | `crop_stage_lstm_model.h5` | The pre-trained Keras LSTM model. |
| | `advanced_labeled_dataset.csv`| The core time-series data for all 9 fields. |
| | `insights.json` | The external RAG knowledge base for actionable advice. |
| **Reports** | `AI_Agent_Architecture.md` | This document. Describes the agent's design. |
| | `Data_Science_Report.md` | Covers the fine-tuning process, pivot, and evaluation metrics. |
| **Logs** | `chat_history.txt` | Development interaction logs (as required). |
| **Demo** | `README.md` | Link to the live deployed app and the YouTube demo video. |
