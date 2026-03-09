# OCR Extraction & Classification API

## 🏗️ Project Structure

```
ocr-extraction/
├── backend/
│   ├── api/
│   │   └── routers/
│   │       ├── extraction_router.py    # OCR extraction endpoints
│   │       ├── classification_router.py # Document classification
│   │       └── prompt_router.py        # Prompt management
│   ├── core/
│   │   └── config.py                   # Configuration settings
│   ├── services/
│   │   ├── ocr_processor.py           # OCR processing logic
│   │   └── classification.py          # Document classification
│   ├── assets/
│   │   └── sap_fields.csv            # SAP field mappings
│   ├── database.py                    # MongoDB operations
│   ├── models.py                      # Pydantic models
│   └── main.py                        # FastAPI application
├── docker-compose.yml                 # Docker services
├── Dockerfile                         # Container definition
├── requirements.txt                   # Python dependencies
├── run.py                             # Application runner
└── README.md                          # This file
```

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/AI-Exploration-Initiatives/ocr-extraction.git
cd ocr-extraction
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
Create a `.env` file in the root directory:
```env
# API Keys
MISTRAL_API_KEY=your_mistral_api_key_here
GOOGLE_API_KEY=your_gemini_api_key_here
HF_API_KEY=your_hugging_face_key

# Database
MONGODB_URI=urUri
```

## Quick Start

### 1. Start the Development Server
```bash
python run.py
```

The API will be available at: `http://localhost:8080`

### 2. API Documentation
Visit `http://localhost:8080/docs` for interactive API documentation (Swagger UI)

### Supported File Types

- PDF documents (.pdf)

## Frontend
### 1. Change directory to sap-pal
```
cd sap-pal
```

### 2. Install the dependencies 
```
npm install
```

### 3. Start the server
```
npm run dev
```