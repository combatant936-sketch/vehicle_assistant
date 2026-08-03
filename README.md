# Vehicle Diagnostic Assistant

A Retrieval-Augmented Generation (RAG) based Vehicle Diagnostic Assistant. This application allows users to ask questions about vehicle issues and receive expert diagnostic advice based on a comprehensive database of vehicle symptoms, causes, and diagnostic steps.

## Features

- **RAG-based Q&A**: Uses `minsearch` to index vehicle issue data and retrieves relevant context for questions, which is then fed to an LLM (`openai/gpt-oss-120b` via Groq) for accurate answers.
- **Automated Relevance Evaluation**: Every answer generated is automatically evaluated by an LLM as `RELEVANT`, `PARTLY_RELEVANT`, or `NON_RELEVANT` and the explanation is recorded.
- **Feedback Loop**: Users can submit +1 (thumbs up) or -1 (thumbs down) feedback for each conversation.
- **Analytics & Monitoring**: Integrated with PostgreSQL to store conversations, metrics (tokens, response time, OpenAI cost), and feedback. A setup script automatically provisions a Grafana dashboard for real-time analytics.
- **Containerized Environment**: Complete Docker Compose setup including the FastAPI app, PostgreSQL database, and Grafana dashboard.

## Evaluation Criteria (For Reviewers)

As part of the RAG pipeline, the system evaluates the generated answers based on the following criteria:

- **Relevance**: Evaluated as `NON_RELEVANT`, `PARTLY_RELEVANT`, or `RELEVANT`.
- **Relevance Explanation**: A generated explanation detailing why the answer falls into the assigned relevance category.
- **Performance Metrics**: 
  - **Response Time**: How long the full RAG pipeline took to respond.
  - **Token Usage**: Detailed tracking of prompt tokens, completion tokens, and total tokens.
  - **Cost**: Estimated cost of the LLM operations (both RAG generation and Evaluation).
- **User Feedback**: Tracking implicit user satisfaction through the `/feedback` endpoint (+1 / -1).

The evaluation logic can be found in `project/rag.py` under the `evaluate_relevance` function, and all metrics are saved to the `conversations` table in PostgreSQL.

## Prerequisites

Before running the project, ensure you have the following installed:

- **Docker** and **Docker Compose**
- **Git**

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/combatant936-sketch/vehicle_assistant.git
   cd vehicle-assistant
   ```

2. **Environment Variables**
   Create a `.env` file in the root directory based on the `.env.example` file provided:
   ```bash
   cp .env.example .env
   ```
   Open the `.env` file and populate the necessary keys. Note that `GROQ_API_KEY` is required for the LLM to function.
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   GRAFANA_ADMIN_USER=admin
   GRAFANA_ADMIN_PASSWORD=admin
   POSTGRES_HOST=postgres
   POSTGRES_DB=vehicle_assistant
   POSTGRES_USER=user
   POSTGRES_PASSWORD=password
   POSTGRES_PORT=5432
   ```

3. **Data Preparation**
   The application expects a CSV file containing the vehicle data. Place the dataset at `data/data.csv` (as referenced by the `DATA_PATH` environment variable).

4. **Running the Application**
   Use Docker Compose to build and start the services:
   ```bash
   docker-compose up --build
   ```

   This will start three containers:
   - **PostgreSQL**: Running on port `5432`
   - **FastAPI App**: Running on port `8001` (Mapped from `8000` in the container)
   - **Grafana**: Running on port `3000`

## Initializing Analytics (Grafana & Database)

Once the containers are running, you need to run the `init_db()` function and the Grafana initialization script to set up tables, datasources, and import the dashboard. 

Run the following commands in a new terminal window while the containers are running:

**Initialize the Database Tables:**
```bash
docker-compose exec app python -c "from project.db import init_db; init_db()"
```

**Initialize Grafana Dashboard:**
```bash
docker-compose exec app python project/init.py
```

## Usage

### 1. Ask a Question
Send a POST request to the `/question` endpoint to interact with the RAG system.

**Endpoint**: `http://localhost:8001/question`

**Request Body**:
```json
{
  "question": "My car's engine is misfiring and the check engine light is on. What could be the issue?"
}
```

**Response**:
```json
{
  "conversation_id": "uuid-string",
  "question": "My car's engine is misfiring...",
  "answer": "Based on the context, the issue could be..."
}
```

### 2. Provide Feedback
Send a POST request to the `/feedback` endpoint to provide feedback on the generated answer.

**Endpoint**: `http://localhost:8001/feedback`

**Request Body**:
```json
{
  "conversation_id": "uuid-string",
  "feedback": 1
}
```
*(Feedback can be `1` for positive, or `-1` for negative)*

### 3. View Analytics
Navigate to `http://localhost:3000` in your browser.
Login using the `GRAFANA_ADMIN_USER` and `GRAFANA_ADMIN_PASSWORD` you set in the `.env` file.
You will find the pre-configured Dashboard that tracks API performance, token usage, cost, and relevance metrics.

## Project Structure

- `project/app.py`: The FastAPI application containing the API endpoints.
- `project/rag.py`: Implements the RAG logic (Retrieval from `minsearch`, LLM generation, and Evaluation).
- `project/ingest.py`: Loads the vehicle dataset from `data/data.csv` and builds the `minsearch` index.
- `project/db.py`: Database connection and SQL queries to save conversations and feedback.
- `project/init.py`: Setup script for Grafana Service Accounts, Datasources, and Dashboards.
- `project/dashboard.json`: The Grafana dashboard layout.
- `docker-compose.yaml`: Docker orchestrator file.
- `Dockerfile`: Defines the FastAPI application container.
- `pyproject.toml`: Python dependencies and project settings (using `uv`).
- `Untitled.ipynb`: Jupyter Notebook used for synthetic data generation, search index prototyping, and offline RAG evaluation.

## Data Generation & Prototyping

The root directory contains the [Untitled.ipynb](file:///d:/vehicle-assistant/Untitled.ipynb) notebook, which outlines the design, generation, and testing workflow:
1. **Synthetic Data Generation**: Uses structured output parsing (`VehicleIssueDataset` schema) and Groq (`openai/gpt-oss-120b`) to generate a dataset of 50 diverse vehicle diagnostic issues. The results are saved to [data/data.csv](file:///d:/vehicle-assistant/data/data.csv).
2. **Search Index Prototyping**: Prototypes indexing the dataset using `minsearch` and testing text query matching.
3. **RAG Pipeline Experimentation**: Simulates user questions, constructs prompt templates, and runs test completions to refine the generator response.
4. **Evaluation**: Samples the RAG pipeline's response quality on test questions using an LLM-as-a-judge system that rates relevance (`RELEVANT`, `PARTLY_RELEVANT`, or `NON_RELEVANT`). The evaluations are recorded and exported to a CSV file.

