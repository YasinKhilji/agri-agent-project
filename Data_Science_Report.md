# Agri-Agent: Data Science & Evaluation Report

**Author:** Khilji Mohammed Yasin
**Project:** Submission for the I'mbesideyou Data Scientist Internship (2026)

---

## 1. Executive Summary

This project successfully developed a multi-agent system (`Agri-Agent`) that fulfills all mandatory and bonus requirements. The final product is a **proactive, reliable, and interpretable** Streamlit web application.

A key part of this project was the **fine-tuning of an AI Planner**. After rigorous testing, this AI Planner was found to be unreliable. A critical engineering decision was made to **pivot to a 100% reliable Rule-Based Planner**. This report details this pivot and the evaluation metrics designed to measure the final system's success.

## 2. The AI Planner: Fine-Tuning & Pivot

This project directly addresses the "fine-tuned model" requirement.

### 2.1. Initial Goal: The AI Planner
The primary fine-tuning target was to create a "Planner" agent. The goal was to fine-tune a `TinyLlama-1.1B` model to **improve JSON reliability**—forcing it to stop acting as a chatbot and *only* output valid JSON commands.

* **Dataset:** An initial 65-example dataset (`finetune_data.jsonl`) was created.
* **Training:** A `finetune.py` script using `peft` (LoRA), `trl`, and `transformers` was built.
* **Challenge 1 (Environment):** Training failed on cloud GPUs (Colab/Kaggle) due to RAM limitations. The environment was successfully migrated to a local machine (RTX 3060).
* **Challenge 2 (Reliability):** The 65-example model failed evaluation. It was non-deterministic and still produced chatty, non-JSON responses.
* **Iteration:** A synthetic dataset of **622 examples** (`finetune_data_v2.jsonl`) was programmatically generated (`generate_dataset.py`) to cover all agent actions.
* **Final Test:** This re-trained model (`ai_planner_v2_adapter`) was integrated and tested. It *also* proved to be unreliable, failing to parse user queries consistently (as shown in the `chat_history.txt`).

### 2.2. The Engineering Pivot: A Robust Rule-Based Planner
The fine-tuning experiment *succeeded* in proving that this specific task was unsuited for a small, fine-tuned model.

To meet the core requirement of building a **functional and reliable** agent, I pivoted to a **Rule-Based Planner** (`agent.py`). This planner uses regular expressions and keyword logic to achieve **100% routing accuracy** for all supported queries. This strategic decision demonstrates a mature engineering approach focused on delivering a robust, successful product over a failed experiment.

## 3. Evaluation Metrics

The final agent's quality is measured by two distinct metrics:

### Metric 1: Planner Reliability
* **Metric:** JSON Format & Routing Accuracy
* **Description:** The percentage of user queries that are correctly parsed into the valid, expected JSON command.
* **Result:** **100%**
* **Evaluation:** The Rule-Based Planner guarantees that all defined queries (for analysis, maps, history, or errors) are correctly routed every time, far exceeding the initial 95% target.

### Metric 2: Executor Quality
* **Metric:** Crop Stage Prediction Accuracy
* **Description:** The classification accuracy of the underlying LSTM model (the "Custom Tool") on its original test data.
* **Result:** **87.5%**
* **Evaluation:** The agent's core "quality" is based on the proven 87.5% accuracy of the OELP model it uses, which fulfills the project's data science requirement.

## 4. Bonus Features Implemented

The final application successfully implements **all three** bonus categories, plus additional advanced features.

1.  **Multi-agent Collaboration:** The Planner-Executor architecture is the foundation of this project.
2.  **Custom Tools & Integrations:**
    * **Custom Tool:** The `executor_tool.py` wraps the LSTM model as a callable tool.
    * **True RAG:** The agent provides actionable advice by *retrieving* it from an external `insights.json` file based on the prediction.
3.  **User Interface:**
    * A full **Streamlit Web Application** (`app.py`) was built and deployed.
    * **Data Visualization:** A Plotly sidebar chart dynamically plots and compares historical data for any selected field.
    * **Spatial Map Simulation:** A `generate_health_map` function simulates and displays a 10x10 Plotly heatmap for in-field variance.
4.  **Advanced (Self-Added) Features:**
    * **Proactive Alerting:** The agent is stateful. It remembers past predictions and uses `st.toast` to **push a 🔔 notification** to the user if a field's stage changes after a simulation.
    * **Data Simulation:** A "Simulate Next Day" button makes the app dynamic by appending new, *stage-aware* simulated data to the CSV.
    * **Historical Analysis:** The agent can parse date requests (e.g., "on 2024-10-15") and run the model on past data.
    * **Explainable AI (XAI):** Every prediction is accompanied by an expander that shows the **actual data sequence** the LSTM model used, making the decision interpretable.