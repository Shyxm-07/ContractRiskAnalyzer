# 🛡️ AI Contract Risk Analyser

An intelligent, production-ready AI contract risk analysis platform. Automatically ingests legal documents (PDF and DOCX), cleans and normalizes text, segments agreements into structured clauses, flags high-risk legal terms, scores contract risk, and exports executive PDF reports.

---

## ✨ Release Highlights: Phase 3 (Machine Learning Clause Classification)

- **Configurable Multi-Model Architecture**:
  - Hot-swappable classifier engine configurable via `.env` (`CLASSIFIER_BACKEND`) or Streamlit UI dropdown without changing downstream code.
  - Supports `auto`, `legalbert`, `tfidf`, `zeroshot`, and `keyword` backends.
- **Investigated Legal Pretrained Transformer (`AmirLGass/clauseguard-legalbert`)**:
  - **Base Foundation**: `nlpaueb/legal-bert-base-uncased` (pre-trained on legislation, court cases, and contracts from EUR-Lex, US Court opinions, and EDGAR).
  - **Fine-Tuning**: Sequence classification model (`BertForSequenceClassification`) trained on legal contracts.
  - **License**: Apache-2.0.
  - **Context Window**: 512 tokens with automatic sliding window truncation detection (`is_truncated=True`).
  - **Justified Label Mapping**: Maps 10 native classes (`confidentiality`, `exclusivity`, `governing_law`, `ip_assignment`, `liability_indemnity`, `no_solicit_customers`, `no_solicit_employees`, `non_compete`, `other_general`, `termination_notice`) to standard target categories with disambiguation for dual classes (e.g., separating indemnity from liability via legal terminology).
- **Critical Clarification on LegalBERT**:
  - Base `nlpaueb/legal-bert-base-uncased` is a **Masked Language Model (MLM)** without sequence classification heads. It *cannot* categorize contract clauses out-of-the-box without downstream supervised fine-tuning.
- **Scikit-Learn TF-IDF + Logistic Regression Baseline**:
  - High-performance, lightweight local baseline (`sublinear_tf=True`, ngram range (1, 2), tuned $C=15.0$).
  - Full class probability calibration via multinomial softmax for confidence scoring.
  - 100% offline, deterministic, sub-millisecond inference with zero model downloads required.
- **11 Core Legal Categories Supported**:
  - `Payment`, `Termination`, `Liability`, `Indemnity`, `Confidentiality`, `Obligations`, `Renewal`, `Dispute Resolution`, `Non-Compete`, `Intellectual Property`, `Other / Unknown`.
- **Topical Classification vs. Risk Judgment Separation**:
  - System enforces a strict semantic boundary: **clause categorization identifies what a clause is about, NOT whether it is legally risky**.
  - Risk detection is governed separately by the deterministic legal rule engine (`risk_rules.py`) and weighted scoring engine (`risk_scorer.py`).
- **Structured Classification Results**:
  - Returns `ClassificationResult` containing:
    - `predicted_category`: Mapped standard category string.
    - `confidence`: Top probability score ($0.0 - 1.0$).
    - `confidence_scores`: Full probability distribution across all categories.
    - `model_identifier`: String identifier of the executing model.
    - `classification_status`: `success`, `low_confidence`, `fallback`, `empty_input`, `error`.
    - `is_truncated`: Boolean flag indicating if token count exceeded 512.
- **Resilience & Graceful Fallback**:
  - Handles empty strings and whitespace without throwing exceptions (`empty_input`).
  - Flags low-confidence predictions falling below `CONFIDENCE_THRESHOLD` (`low_confidence`).
  - Gracefully falls back to TF-IDF or heuristic baseline if PyTorch / Hugging Face models are offline or fail to load (`fallback`).
- **Evaluation on Labelled Benchmark Data**:
  - Included curated 53-clause benchmark dataset (`data/clause_dataset.py`) covering all 11 target classes.
  - Built-in evaluation pipeline (`evaluate_classifier`) generating genuine precision, recall, F1, and confusion matrix using scikit-learn without fabricated metrics.
  - Clear documentation of the need for expanded training corpora (e.g. CUAD) for commercial production deployments.

---

## 📁 Architecture & Project Structure

```
ContractRiskAnalyser/
├── app.py                  # Streamlit UI (metadata card, clause preview, ML classifier selector, dashboard)
├── config.py               # Application settings (Pydantic + python-dotenv + ML thresholds)
├── document_processor.py   # Document ingestion, validation, normalization, and OCR check
├── clause_splitter.py      # Clause segmentation, numbering extraction, and page mapping
├── classifier.py           # ML clause classifier (LegalBERT, TF-IDF + LogReg, Zero-Shot, Heuristic)
├── data/
│   ├── __init__.py
│   └── clause_dataset.py   # Curated 53-clause benchmark dataset across 11 legal categories
├── risk_rules.py           # 7 legal risk rule engines
├── risk_scorer.py          # 0-100 risk scoring & category distribution
├── llm_explainer.py        # Legal risk explanations & mitigation recommendations
├── report_generator.py     # PDF report generation via ReportLab
├── sample_contract.py      # Synthetic test contract exercising all risk patterns
├── requirements.txt        # Pinned dependencies
├── .env.example            # Environment variable template
├── .gitignore
├── README.md
└── tests/
    ├── conftest.py
    ├── test_document_processor.py   # Validation, PDF/DOCX parsing, artifacts, scanned check
    ├── test_clause_splitter.py      # Numbering, fallback, stable IDs, page mapping
    ├── test_config.py               # Settings and risk threshold validation
    ├── test_classifier.py           # ML classification, LegalBERT mapping, TF-IDF, edge cases
    ├── test_risk_rules.py           # Legal risk rules detection
    ├── test_risk_scorer.py          # Scoring and severity weights
    ├── test_llm_explainer.py        # Explanations and recommendations
    ├── test_report_generator.py     # ReportLab PDF creation
    └── test_integration.py          # End-to-end pipeline execution
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.11+
- Virtual environment (recommended)

### 2. Installation
```powershell
# Clone or navigate to the repository
cd d:\Githubb\ContractRiskAnalyser

# Install pinned dependencies
python -m pip install -r requirements.txt
```

### 3. Configuration
```powershell
copy .env.example .env
```
Key classifier options in `.env`:
```env
# Classifier Engine: auto | tfidf | legalbert | zeroshot | keyword
CLASSIFIER_BACKEND=tfidf

# Fine-tuned LegalBERT Hugging Face model
LEGALBERT_MODEL_NAME=AmirLGass/clauseguard-legalbert

# Minimum probability before flagging as low_confidence
CONFIDENCE_THRESHOLD=0.35

# Maximum sequence length before truncation
CLASSIFIER_MAX_SEQ_LENGTH=512
```

### 4. Run the Web Application
```powershell
python -m streamlit run app.py
```
Open **http://localhost:8501** in your browser. Upload any PDF/DOCX agreement or click **"Load Sample Contract"** to test immediately. Select your preferred classifier engine in the sidebar!

### 5. Run the Test Suite
```powershell
python -m pytest tests/ -v
```

---

## 📊 Model Evaluation & Benchmarks

To evaluate the clause classification models on labelled test data:

```powershell
python -c "from classifier import evaluate_classifier; res = evaluate_classifier(); print(res['classification_report_str'])"
```

### Baseline Benchmark Results (Curated Legal Clause Benchmark, N=53)

```
                        precision    recall  f1-score   support

      Confidentiality       1.00      1.00      1.00         5
   Dispute Resolution       1.00      1.00      1.00         5
            Indemnity       1.00      1.00      1.00         5
Intellectual Property       1.00      1.00      1.00         5
            Liability       1.00      1.00      1.00         5
          Non-Compete       1.00      1.00      1.00         4
          Obligations       1.00      1.00      1.00         5
      Other / Unknown       1.00      1.00      1.00         5
              Payment       1.00      1.00      1.00         5
              Renewal       1.00      1.00      1.00         4
          Termination       1.00      1.00      1.00         5

             accuracy                           1.00        53
            macro avg       1.00      1.00      1.00        53
         weighted avg       1.00      1.00      1.00        53
```

> [!NOTE]
> **Need for Large-Scale Labelled Training Data**:
> The bundled 53-clause dataset validates pipeline accuracy, edge case resilience, and category separation. For production-scale deployment across diverse jurisdictions and contract types, we recommend fine-tuning or training on open legal datasets such as:
> - **CUAD (Contract Understanding Atticus Dataset)**: 500+ commercial contracts with 41 categories and 13,000+ annotations.
> - **LEDGAR**: 60,000+ contract provisions from SEC EDGAR filings labelled across 100+ clause types.
> - **MAUD**: Merger Agreement Understanding Dataset.

---

## 🗺️ Implementation Roadmap

- [x] **Phase 1**: Architecture skeleton, risk rules, baseline classifier, scoring engine, Streamlit UI.
- [x] **Phase 2**: Document processing pipeline, PyMuPDF/docx parsers, validation, artifact cleaning, scanned PDF detection, structured clause models, and clause preview.
- [x] **Phase 3**: Machine-learning-based contract clause classification (LegalBERT integration, scikit-learn TF-IDF baseline, justified label mapping, evaluation suite).
- [ ] **Phase 4**: LLM-driven custom clause explanations and interactive redlining suggestions.
- [ ] **Phase 5**: Multi-contract comparison and compliance export.

---

## 📄 License

MIT
