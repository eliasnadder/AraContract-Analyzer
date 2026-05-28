AraContract Analyzer — Software Requirements Specificationv1.0 | 2025

# AraContract Analyzer

# ﻲﻋﺎﻧطﺻﻻا ءﺎﻛذﻟﺎﺑ ﺔﯾﺑرﻌﻟا ﺔﯾرﺎﺟﺗﻟا دوﻘﻌﻟا لﯾﻠﺣﺗ مﺎظﻧ

## Software Requirements Specification (SRS)

### Version 1.0

Project Name AraContract Analyzer Course Project 2 — AI Specialization Damascus University — Computer University Engineering Team Size 5 Members One Academic Semester (~4 Duration months) Date 2025

Damascus University — Computer Engineering, AI SpecializationPage N


AraContract Analyzer — Software Requirements Specificationv1.0 | 2025

# 1. Introduction

### 1.1 Purpose

This document defines the Software Requirements Specification (SRS) for AraContract Analyzer, an AI-powered platform designed to analyze Arabic commercial contracts. The system automatically extracts and classifies contract clauses, assesses risk levels, generates executive summaries, and enables users to ask natural language questions about their contracts. This SRS serves as the technical and functional reference for the development team throughout the project lifecycle.

### 1.2 Problem Statement

In Syria and the broader Arab world, small and medium business owners routinely sign commercial contracts without fully understanding their content. Legal consultation is expensive and time-consuming, while manual reading requires legal expertise most individuals lack. This creates significant risk of signing unfavorable or harmful contractual terms. AraContract Analyzer addresses this gap by providing an accessible, intelligent first-level contract review tool — functioning as a free preliminary legal advisor.

### 1.3 Scope

The system accepts Arabic commercial contract documents (PDF or scanned image), processes them through an NLP pipeline, and delivers:

- Automatic clause segmentation and type classification
- Risk level assessment per clause (High / Medium / Low)
- Highlighted and color-coded clause visualization
- Automatic warnings for high-risk clauses
- Executive summary of the full contract
- Interactive Q&A chat interface (RAG-powered)
- Optional: side-by-side comparison of two contracts
### 1.4 Definitions and Abbreviations

Term / Definition Abbreviation Natural Language Processing — computational processing of human NLP language Retrieval-Augmented Generation — technique combining document RAG retrieval with LLM generation Large Language Model — a neural model trained on large text corpora LLM (e.g., Qwen, Llama) Process of further training a pre-trained model on a smaller domain-specific Fine-tuning dataset Arabic BERT variant by NYU Abu Dhabi, optimized for Modern Standard CAMeLBERT Arabic text Facebook AI Similarity Search — vector database for fast semantic FAISS similarity lookups

Damascus University — Computer Engineering, AI SpecializationPage N


AraContract Analyzer — Software Requirements Specificationv1.0 | 2025

Term / Definition Abbreviation Optical Character Recognition — converting scanned images to OCR machine-readable text A distinct section or article within a contract, representing a single obligation Clause or condition SRS Software Requirements Specification — this document

## 1.5 References

- CUAD: Contract Understanding Atticus Dataset — atticusprojectai.org/cuad
- CAMeL-Lab/bert-base-arabic-camelbert-msa — Hugging Face
- sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 — Hugging Face
- LangChain Documentation — python.langchain.com
- FAISS Library — github.com/facebookresearch/faiss
Damascus University — Computer Engineering, AI SpecializationPage N


AraContract Analyzer — Software Requirements Specificationv1.0 | 2025

# 2. Overall Description

## 2.1 Product Perspective

AraContract Analyzer is a standalone web application with no dependency on external legal databases or paid APIs (all models run locally). It is structured as a client-server application with a React frontend, a FastAPI backend, and a local AI inference layer.

## 2.2 System Architecture Overview

Layer Components Frontend React.js — file upload, clause viewer, risk highlights, chat interface Backend API FastAPI (Python) — orchestrates all pipeline components Text Extraction PyMuPDF (digital PDF) + Tesseract Arabic (scanned/OCR) Clause Segmentation Rule-based Arabic regex segmenter Classifier CAMeLBERT fine-tuned on annotated Arabic contract clauses RAG Engine FAISS vector store + multilingual sentence embeddings + local LLM LLM Qwen2.5-7B (local) — summary generation and Q&A answers

Full pipeline data flow: PDF/Image Upload ↓ PyMuPDF / Tesseract Raw Arabic Text ↓ Regex Segmenter List of Clauses ↓ CAMeLBERT Classifier Clause Type + Risk Level per Clause ↓ FAISS Embedding (parallel) Vector Store (for Q&A) ↓ FastAPI Response React Frontend: Highlighted Clauses + Chat Interface

## 2.3 User Classes and Characteristics

- Primary User: Small/medium business owner — uploads contracts, reads analysis, asks
questions. No technical background required.

- Secondary User: Legal assistant or paralegal — uses the system to speed up initial
contract review.

- Administrator: Project team member — manages deployment and model updates.
## 2.4 Operating Environment

- Backend: Python 3.10+, runs on a university server or local machine (GPU preferred for
LLM inference)

- Frontend: Modern web browser (Chrome, Firefox, Edge)
- Storage: Local filesystem for uploaded contracts and vector stores (no cloud required)
- Model inference: CPU-compatible, GPU-accelerated when available
Damascus University — Computer Engineering, AI SpecializationPage N


AraContract Analyzer — Software Requirements Specificationv1.0 | 2025

### 2.5 Constraints

- All models must be open-source and locally deployable (no paid API dependency)
- System must handle Arabic text including both Modern Standard Arabic and formal
contract language

- Scanned PDFs must be processable via OCR (quality-dependent limitation)
- Development timeline: one academic semester (~4 months)
- Team size: 5 developers with mixed AI/backend/frontend experience
Damascus University — Computer Engineering, AI SpecializationPage N


AraContract Analyzer — Software Requirements Specificationv1.0 | 2025

# 3. Functional Requirements

### FR-1: Document Upload and Text Extraction

- The system shall accept PDF files up to 20 MB.
- The system shall extract text from digital PDFs using PyMuPDF.
- The system shall apply Arabic OCR (Tesseract) to scanned PDFs or image-based
contracts.

- The system shall detect whether a PDF is digital or scanned and select the appropriate
extraction method automatically.

- Extracted text must be returned in proper Arabic Unicode (RTL preserved).
### FR-2: Clause Segmentation

- The system shall split the extracted text into individual clauses.
،ًﻻوأ ًﺎﯾﻧﺎﺛ

- Segmentation shall recognize Arabic article/clause markers: ،دﻧﺑﻟا ،ةدﺎﻣﻟا، and
decimal numbering patterns.

- Each clause must be a minimum of 30 characters to be included.
- The system shall return an ordered list of clause text strings.
### FR-3: Clause Classification

- The system shall classify each clause into one of 8 predefined types.
Clause Type (Arabic) Description ﻲﻟﺎﻣ / ﻊﻓد Payment terms, amounts, schedules ءﺎﮭﺗﻧا / ةدﻣ Contract duration and expiry ءﺎﮭﻧإ / ﺦﺳﻓ Termination conditions and rights تﺎﺿﯾوﻌﺗ / تﺎﻣارﻏ Penalties, compensation, and damages لوﻷا فرطﻟا تﺎﻣازﺗﻟا Obligations of Party A ﻲﻧﺎﺛﻟا فرطﻟا تﺎﻣازﺗﻟا Obligations of Party B تﺎﻋازﻧ ﺔﯾوﺳﺗ Dispute resolution and jurisdiction ﺔﻣﺎﻋ مﺎﻛﺣأ General provisions and definitions

### FR-4: Risk Assessment

- The system shall assign a risk level to each classified clause: High (red), Medium
(yellow), or Low (green).

- High-risk clauses shall include an explanatory warning text (e.g., 'Unilateral termination
right — unfavorable to Party B').

- Risk classification shall be part of the fine-tuned model output (multi-label: type + risk).
### FR-5: Clause Visualization

- The frontend shall display each clause in a card with its type badge and risk color
indicator.

- High-risk clauses shall display a warning icon and warning message.
- The user shall be able to filter clauses by type or risk level.
Damascus University — Computer Engineering, AI SpecializationPage N


AraContract Analyzer — Software Requirements Specificationv1.0 | 2025

### FR-6: Executive Summary

- The system shall generate a 3-5 sentence Arabic summary of the full contract.
- The summary shall highlight: contract parties, main obligations, duration, and any
detected high-risk items.

- Summary generation shall be performed by the local LLM via a structured prompt.
### FR-7: RAG-powered Q&A

- The system shall build a FAISS vector store from the contract text after upload.
- Users shall be able to submit free-text Arabic questions about the contract.
- The system shall retrieve the 3 most relevant text chunks and pass them to the LLM with
the question.

- The LLM shall answer based solely on retrieved contract content (no hallucination from
general knowledge).

- The Q&A session shall persist for the duration of the user session.
### FR-8: Contract Comparison (Optional / Bonus Feature)

- The system shall accept two contract PDFs simultaneously.
- The system shall identify and display differences in clause types, risk levels, and key
terms between the two contracts.

- This feature is marked as optional and will be implemented if time permits.
Damascus University — Computer Engineering, AI SpecializationPage N


AraContract Analyzer — Software Requirements Specificationv1.0 | 2025

# 4. Non-Functional Requirements

Category Requirement Details

Full contract analysis < 30 seconds for a Performance Response time 10-page contract on CPU

Performance Q&A latency Answer generated in < 10 seconds

Accuracy Classifier F1 Target weighted F1 >= 0.80 on held-out test set

All UI text and contract display must be Usability Arabic RTL right-to-left

System available during demonstration and Reliability Uptime evaluation sessions

Uploaded files stored temporarily and deleted Security File handling after session

No mandatory internet connection after initial Portability Local-first model download

Each pipeline stage implemented as an Maintainability Modularity independent module

Damascus University — Computer Engineering, AI SpecializationPage N


AraContract Analyzer — Software Requirements Specificationv1.0 | 2025

# 5. Data Collection and Dataset

### 5.1 Why a Dataset is Needed

The RAG Q&A component requires no training data — it retrieves directly from the uploaded contract. However, the clause classifier (FR-3, FR-4) requires a supervised dataset of labeled Arabic contract clauses to be fine-tuned. Without this, the system cannot automatically identify clause types or risk levels.

### 5.2 Dataset Sources

5.2.1 Reference Dataset (English — for structural guidance)

- CUAD Dataset: 510 English commercial contracts with 13,000+ annotated spans
covering 41 clause types.

- Usage: Study clause taxonomy and annotation methodology; use as a
translation/adaptation reference.

- Source: atticusprojectai.org/cuad (free, open license)
5.2.2 Arabic Contract Sources

- Google Search: query patterns such as 'دﯾروﺗ دﻘﻋ جذوﻣﻧ PDF'، 'ﺔﯾرﺎﺟﺗ تﺎﻣدﺧ دﻘﻋ' ،'يروﺳ رﺎﺟﯾإ دﻘﻋ
ﻲﺑرﻋ'

- Legal Websites: al-manassa-alqanouniyya.com, qanoonak.com — contain downloadable
Arabic contract templates

- Government Sources: Syrian Ministry of Commerce, Arab arbitration bodies — publish
standard contract forms

- Team-generated contracts: Using real contract templates as a base, the team generates
variations to increase volume 5.2.3 Target Collection Volume Contract Type Number of Contracts Estimated Clauses Supply / دﯾروﺗ 25-30 80-100 Service / تﺎﻣدﺧ 20-25 60-80 Rental / رﺎﺟﯾإ 20-25 60-80 Employment / لﻣﻋ 15-20 50-70 General Commercial 10-15 30-50 TOTAL 90-115 contracts 280-380 labeled clauses

### 5.3 Annotation Process

5.3.1 Annotation Tool: Doccano Doccano is a free, open-source text annotation platform that runs locally in the browser. Team members label each clause by selecting its type and risk level through a point-and-click interface. pip install doccano doccano init doccano createuser --username admin --password admin123 doccano runserver # opens at http://localhost:8000

Damascus University — Computer Engineering, AI SpecializationPage N


AraContract Analyzer — Software Requirements Specificationv1.0 | 2025

5.3.2 Annotation Label Schema Each clause is annotated with two labels:

- clause_type: one of 8 types (see FR-3 table)
- risk_level: high / medium / low
- risk_reason: short Arabic text explaining why a clause is high/medium risk (optional for
low-risk) 5.3.3 Final Dataset Format { "ﻖﺒﺴﻣ رﺎﻌﺷإ نود ﺖﻗو يأ ﻲﻓ ﺪﻘﻌﻟا اﺬﻫ ءﺎﻬﻧإ لوﻷا فﺮﻄﻠﻟ ﻖﺤﻳ", "text": "ﺦﺴﻓ", "clause_type": "risk_level": "high", "ﻲﻧﺎﺜﻟا فﺮﻄﻠﻟ ﺐﺳﺎﻨﻣ ﺮﻴﻏ رﺎﻌﺷإ نوﺪﺑ يدﺎﺣأ ﺦﺴﻓ" "risk_reason": — } 5.3.4 Team Annotation Distribution

- Each of the 5 team members annotates 60-80 clauses independently
- 10% of clauses are double-annotated to measure inter-annotator agreement (target
Cohen's Kappa >= 0.75)

- Disagreements are resolved by team discussion
- Total annotation time estimate: 2-3 weeks (Week 2-4 of the project)
### 5.4 Dataset Split

Split Percentage Approx. Size Training 70% 196-266 clauses Validation 15% 42-57 clauses Test 15% 42-57 clauses

Damascus University — Computer Engineering, AI SpecializationPage N


AraContract Analyzer — Software Requirements Specificationv1.0 | 2025

# 6. Technical Implementation

## 6.1 Technology Stack

Component Technology Justification Lightweight, fast, preserves Arabic Text Extraction PyMuPDF Unicode; used in prior projects OCR Tesseract Arabic Free, local, no API cost Best available Arabic BERT for formal Base NLP Model CAMeLBERT MSA text (contracts, legal) Industry standard, good Arabic support, Fine-tuning HuggingFace Transformers Colab-compatible Supports Arabic, lightweight, high Embeddings multilingual-MiniLM-L12-v2 quality sentence vectors Vector Store FAISS Local, no cloud, fast similarity search RAG Orchestration LangChain Mature library, good RAG abstractions LLM Open-source, multilingual including Qwen2.5-7B (local) (Q&A+Summary) Arabic, runs on CPU Team is experienced; async support; Backend FastAPI (Python) auto-generates API docs Component-based; good RTL support; Frontend React.js widely documented

## 6.2 Module Descriptions

### 6.2.1 Text Extraction Module

Accepts uploaded PDF, detects whether it is digital or scanned, and returns raw Arabic text. Digital PDFs processed with PyMuPDF; scanned documents converted to images via pdf2image then passed through pytesseract with Arabic language pack.

### 6.2.2 Clause Segmentation Module

(،ًﻻوأ Uses a regex-based Arabic segmenter that splits text on article/clause markers ،دﻧﺑﻟا ،ةدﺎﻣﻟا ًﺎﯾﻧﺎﺛ، decimal patterns). Returns an ordered list of clause strings, filtering out any segment shorter than 30 characters.

### 6.2.3 Classifier Module

CAMeLBERT fine-tuned as a sequence classifier. Input: single clause text. Output: clause_type (8-class) + risk_level (3-class). Fine-tuning performed on Google Colab using the annotated dataset. The final model is saved locally and loaded by FastAPI at startup.

### 6.2.4 RAG Module

At upload time, the full contract text is chunked (500-character windows, 50-character overlap) and embedded using the multilingual sentence transformer. Vectors are stored in a FAISS index keyed to the user's session. At query time, the top-3 most similar chunks are retrieved and passed to the LLM with a strict context-only prompt to prevent hallucination.

### 6.2.5 LLM Module

Damascus University — Computer Engineering, AI SpecializationPage N


AraContract Analyzer — Software Requirements Specificationv1.0 | 2025

Qwen2.5-7B loaded via the transformers library with 4-bit quantization (bitsandbytes) to reduce memory requirements. Used for two tasks: (1) generating the contract executive summary from the classifier output and full text, (2) answering user questions in the RAG pipeline. 6.2.6 FastAPI Backend Exposes two primary endpoints: POST /analyze — accepts PDF, returns clause list with types, risks, and summary POST /ask/{session} — accepts question string, returns LLM answer from RAG Additional utility endpoints: GET /health, DELETE /session/{id} 6.2.7 React Frontend Single-page application with three main views: (1) Upload screen, (2) Analysis results screen with colored clause cards and filter controls, (3) Chat interface for Q&A. Supports Arabic RTL layout throughout.

Damascus University — Computer Engineering, AI SpecializationPage N


AraContract Analyzer — Software Requirements Specificationv1.0 | 2025

# 7. Team Roles and Responsibilities

# Role Responsibilities Key Deliverables

Trained model, Clause segmentation, CAMeLBERT evaluation report, 1 NLP Engineer (Lead) fine-tuning, classifier evaluation segmentation module

RAG module, Q&A FAISS vector store, RAG pipeline, endpoint, summary 2 RAG / AI Engineer LLM integration, prompt engineering generation

REST API, server FastAPI server, API design, file deployment, API 3 Backend Engineer handling, session management documentation

React UI, RTL layout, clause card Complete web frontend, 4 Frontend Engineer components, chat interface UX design

Contract collection, Doccano Labeled dataset

annotation, dataset management, (CSV/JSON), test suite, 5 Data & QA Engineer

system testing final presentation

Damascus University — Computer Engineering, AI SpecializationPage N


AraContract Analyzer — Software Requirements Specificationv1.0 | 2025

# 8. Project Timeline

Month Phase Key Activities Milestone

Finalize SRS, collect 90-115 Arabic contracts, set up Doccano, begin Dataset v1 Month 1 Planning & Data annotation, set up development ready environments

Complete annotation, fine-tune CAMeLBERT on Colab, build RAG pipeline, Working AI Month 2 Core AI integrate Qwen2.5, develop text extraction pipeline and segmentation modules

Build FastAPI backend, develop React Functional frontend, integrate all modules end-to-end, Month 3 Integration prototype internal testing

System testing, classifier evaluation (F1 score), UI polishing, optional contract Final Month 4 Testing & Delivery comparison feature, prepare final delivery presentation and demo

Damascus University — Computer Engineering, AI SpecializationPage N


AraContract Analyzer — Software Requirements Specificationv1.0 | 2025

# 9. Evaluation Criteria

### 9.1 Classifier Evaluation

- Primary metric: Weighted F1 Score (target >= 0.80)
- Secondary: Per-class precision and recall to identify weak clause types
- Confusion matrix analysis for misclassification patterns
- Evaluation tool: sklearn.metrics.classification_report
### 9.2 RAG Evaluation

- Manual evaluation: team creates 30 test questions with ground-truth answers from
known contracts

- Scoring: answer correctness (0-2 scale) assessed by two team members independently
- Target: >= 80% of answers rated as correct or partially correct
### 9.3 System Evaluation

- End-to-end processing time for a 10-page contract (target < 30 seconds)
- User experience evaluation: usability walkthrough with 3-5 non-technical users
- OCR quality measurement on a sample of 10 scanned contracts
Damascus University — Computer Engineering, AI SpecializationPage N


AraContract Analyzer — Software Requirements Specificationv1.0 | 2025

# 10. Risks and Mitigation

Risk Severity Mitigation Strategy

Start annotation in Week 1; use data

Insufficient annotated data for augmentation (paraphrasing) if needed; fall back High to prompt-based classification if dataset < 150 fine-tuning

samples

Pre-process images (denoise, contrast Poor OCR quality on Arabic enhancement); filter out contracts with OCR Medium scanned contracts confidence < 60%

Qwen2.5-7B too slow / Use 4-bit quantization; or switch to Gemini API memory-intensive on team Medium free tier as fallback hardware

Clause segmentation errors Manual review of segmentation output on 20 Medium affecting classifier sample contracts; refine regex rules iteratively

All code documented; each member Team member drops out or cross-trained on one other module; GitHub with Low becomes unavailable clear commit history

Contract dialect variation (Gulf Focus dataset on Syrian and Levantine Low vs. Levantine legal style) contracts; note dialect limitation in final report

End of Document

AraContract Analyzer — SRS v1.0

Damascus University — Computer Engineering, AI SpecializationPage N
