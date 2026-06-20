# Vortex-Auditor

## AI-Powered Document Auditor & Citation Compliance Engine

Vortex-NLP is a scalable AI-powered document auditing platform that automates compliance verification and citation validation for regulatory and policy documents. Built using FastAPI, PyMuPDF, SQLite, SQLAlchemy, and local Large Language Models through Ollama, the system analyzes uploaded documents, retrieves relevant guideline context, performs reasoning-driven audits, and generates evidence-backed compliance reports.

The platform combines semantic retrieval using SentenceTransformers embeddings with Llama 3.1-powered reasoning to deliver transparent, citation-aware audit results while maintaining complete local execution without reliance on external cloud AI services.

---

## Features

### Document Processing

* PDF ingestion and parsing using PyMuPDF
* Structured text extraction
* Multi-page document support
* Efficient document normalization

### Semantic Retrieval

* Context-aware text chunking
* SentenceTransformers embeddings generation
* High-precision semantic search
* Relevant guideline retrieval

### AI-Powered Auditing

* Llama 3.1 reasoning through Ollama
* Requirement validation
* Policy compliance analysis
* Logical consistency checks

### Citation Verification

* Evidence-backed findings
* Citation traceability
* Source-reference validation
* Hallucination reduction mechanisms

### Report Generation

* Structured audit reports
* Compliance summaries
* Detailed findings and observations
* Context-supported conclusions

### Persistence Layer

* SQLite database integration
* SQLAlchemy ORM
* Audit metadata management
* Transaction-safe operations

---

## System Architecture

```text
                    ┌─────────────────────┐
                    │ Uploaded PDF Files  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     PyMuPDF Parser  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Guideline Chunker   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ SentenceTransformer │
                    │    Embeddings       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Semantic Retriever  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Llama 3.1 via      │
                    │      Ollama         │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
      Classifier         Reasoner       Citation Checker
              │                │                │
              └────────────────┼────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │   Report Builder    │
                    └──────────┬──────────┘
                               ▼
                    Compliance Audit Report
```

---

## Technology Stack

### Backend

* FastAPI
* Python 3.10+
* Uvicorn

### AI & NLP

* Ollama v0.30.06
* Llama 3.1
* SentenceTransformers
* Semantic Retrieval Pipeline

### Document Processing

* PyMuPDF

### Database

* SQLite
* SQLAlchemy ORM

### Data Validation

* Pydantic

---

## Repository Structure

```text
├── data/
│   ├── raw/
│   └── processed/
│
├── app/
│   ├── main.py
│   ├── config.py
│
│   ├── db/
│   │   ├── model.py
│   │   └── session.py
│
│   ├── routers/
│   │   ├── audit.py
│   │   └── upload.py
│
│   ├── schemas/
│   │   └── audit.py
│
│   └── services/
│       ├── pdf_parser.py
│       ├── guideline_chunker.py
│       ├── retriever.py
│       ├── classifier.py
│       ├── reasoner.py
│       ├── citation_checker.py
│       ├── statement_store.py
│       └── report_builder.py
│
├── requirements.txt
├── .env
└── .gitignore
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/sushantgarde/Vortex-NLP.git
cd Vortex-NLP
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment

Windows:

```bash
.venv\Scripts\activate
```

Linux/Mac:

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment

Create a `.env` file:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1

DATABASE_URL=sqlite:///audit.db
```

### Start Ollama

```bash
ollama run llama3.1
```

### Run the Application

```bash
uvicorn app.main:app --reload
```

---

## Workflow

1. Upload regulatory documents.
2. Extract text using PyMuPDF.
3. Chunk content into semantic sections.
4. Generate SentenceTransformers embeddings.
5. Retrieve relevant guideline context.
6. Execute Llama 3.1 reasoning through Ollama.
7. Validate citations and supporting evidence.
8. Generate a compliance audit report.

---

## Example Use Cases

* Regulatory Compliance Audits
* Clinical Trial Protocol Validation
* Internal Policy Reviews
* Governance Documentation Analysis
* Risk Assessment Workflows
* Citation Verification Systems

---

## Future Enhancements

* FAISS/ChromaDB vector database integration
* Multi-document reasoning
* Explainable AI audit scoring
* Enterprise authentication and RBAC
* Multi-language document support
* Cloud deployment support
* Interactive audit dashboard

---

## Author

**Sushant Dattatray Garde**

B.Tech Computer Science and Engineering

Nutan College of Engineering and Research

GitHub: https://github.com/sushantgarde

**Vedant Vinod Sankpal**

B.Tech Computer Science and Engineering

Nutan College of Engineering and Research

GitHub: https://github.com/vedant2004x

---

## License

This project is licensed under the MIT License.
