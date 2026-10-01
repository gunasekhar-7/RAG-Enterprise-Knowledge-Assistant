# RAG Enterprise Knowledge Assistant

Enterprise-grade Generative AI platform for querying organizational knowledge (HR policies, technical SOPs, security manuals, compliance guides) with grounded answers and inline source citations.

---

## 🛠️ Technology Stack

- **Backend:** Python 3.11+, FastAPI (async), SQLAlchemy 2.0 (async), Pydantic v2
- **Vector Search & DB:** PostgreSQL 16 with `pgvector` extension
- **Cache & Async Queue:** Redis 7, Celery 5
- **Object Storage:** MinIO (local) / AWS S3 (production)
- **RAG Kernel:** PyMuPDF, LangChain text-splitters, rank-bm25 (sparse), LiteLLM, FlashRank
- **Frontend:** React 18, Vite, TypeScript, TailwindCSS, Zustand
- **Containerization & Tooling:** Docker Compose, Ruff, Mypy, Pytest

---

## 🚀 Quickstart Guide (Windows / PowerShell)

### 1. Initialize the Environment
Run the setup command to create the virtual environment, install Python backend dependencies, and install Node.js frontend packages:

```powershell
.\dev.ps1 setup
```

### 2. Start Local Infrastructure
Start PostgreSQL (with `pgvector`), Redis, and MinIO object storage in Docker:

```powershell
.\dev.ps1 infra-up
```

Verify containers are healthy:
```powershell
.\dev.ps1 infra-status
```

### 3. Run the Backend API
```powershell
.\dev.ps1 run-backend
```
- API Root: `http://localhost:8000`
- Interactive OpenAPI Docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### 4. Run the Frontend (New Terminal)
```powershell
.\dev.ps1 run-frontend
```
- Web Application: `http://localhost:5173`

### 5. Run Tests & Linter
```powershell
.\dev.ps1 test
.\dev.ps1 lint
```

---

## 📋 Infrastructure Ports & Consoles

| Service | Port | Console / URL | Credentials |
|---|---|---|---|
| **FastAPI Backend** | `8000` | http://localhost:8000/docs | N/A |
| **React Frontend** | `5173` | http://localhost:5173 | N/A |
| **PostgreSQL (pgvector)** | `5432` | `localhost:5432` | user: `postgres`, password: `postgres`, db: `rag_knowledge_db` |
| **Redis** | `6379` | `localhost:6379` | N/A |
| **MinIO S3 API** | `9000` | http://localhost:9000 | user: `minioadmin`, pass: `minioadmin` |
| **MinIO Web Console** | `9001` | http://localhost:9001 | user: `minioadmin`, pass: `minioadmin` |

---

## 🧪 Validation & Health Checks

Run pre-flight validation checklist at any time:
```powershell
.\dev.ps1 check
```
