# 🔍 TruthLens — AI Article Verification & Originality Suite

> **4th Year AI Capstone Project**  
> An intelligent system that verifies factual claims in articles using live web evidence, detects hallucinations, analyzes text similarity vs. plagiarism, and generates a transparent Multi-Signal Trust Score with exportable dossiers.
>
> 🔗 **Repository**: [https://github.com/noobcoder1982/4th-year-project.git](https://github.com/noobcoder1982/4th-year-project.git)

---

## ⚡ Quick Start for Teammates & Evaluators

Want to get running in 5 minutes? Check our complete step-by-step guide:
👉 **[TEAMMATE SETUP & TESTING GUIDE (Click here)](./TEAMMATE_SETUP_AND_TESTING_GUIDE.md)**

```bash
# 1. Clone repository
git clone https://github.com/noobcoder1982/4th-year-project.git
cd 4th-year-project

# 2. Set up virtual environment
python -m venv .venv
.venv\Scripts\activate       # On Windows
# source .venv/bin/activate  # On macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch web application
python app.py
```
Open **`http://localhost:7860`** in your browser.

---

## 🌟 Key Features

| Feature | Description | File |
|---|---|---|
| 📄 **Universal Input Ingestion** | Extract clean text from pasted text, **PDFs**, **Word (.docx)**, and live **Article URLs**. | [`input_processor.py`](./input_processor.py) |
| 🧠 **Discourse Memory Engine** | Resolves pronouns (*"It"*, *"He"*, *"They"*, *"This mission"*) and links cross-sentence context. | [`discourse_memory.py`](./discourse_memory.py) |
| 🌐 **Multi-Provider Web Retrieval** | Live search across DuckDuckGo, Yahoo, Bing, and Wikipedia API with domain authority scoring. | [`app.py`](./app.py) |
| 🎓 **Fine-Tuned Qwen 2.5 (3B)** | QLoRA-adapted 3-billion parameter causal LM trained on verification reasoning. | [`qwen-fact-checker/`](./qwen-fact-checker/) |
| ≈ **Similarity vs. Plagiarism** | Differentiates between natural topical overlap (shared vocabulary) and verbatim copying. | [`similarity_analyzer.py`](./similarity_analyzer.py), [`plagiarism_detector.py`](./plagiarism_detector.py) |
| ⚡ **Hallucination Detection** | Flags unsupported, contradicted, and ungrounded statements. | [`trust_engine.py`](./trust_engine.py) |
| 🏆 **Transparent Trust Score** | 0–100 composite score based on Fact Check (35%), Credibility (25%), Grounding (25%), and Originality (15%). | [`trust_engine.py`](./trust_engine.py) |
| 📊 **Exportable Dossier** | 1-click downloads of complete verification reports in **Markdown (.md)** and **JSON (.json)** formats. | [`report_generator.py`](./report_generator.py) |

---

## 📂 Project Architecture

```
4th-year-project/
│
├── app.py                      # 🌟 THE MAIN APP: Starts the interactive web interface (Gradio UI).
├── discourse_memory.py         # 🧠 DISCOURSE ENGINE: Resolves pronouns, maintains memory, augments search.
├── input_processor.py          # 📄 FILE READER: Parses PDFs, Word docs, text files, and live article URLs.
├── similarity_analyzer.py      # ≈ SIMILARITY ENGINE: Analyzes topical overlap without misclassifying as plagiarism.
├── plagiarism_detector.py      # 🔍 PLAGIARISM AUDIT: Exact n-gram, lexical overlap, and semantic matching.
├── trust_engine.py             # ⚡ TRUST & GROUNDING: Detects hallucinations and computes 0–100 Trust Score.
├── report_generator.py         # 📊 REPORT DOSSIER: Generates downloadable JSON, Markdown, and HTML reports.
├── ui_theme.css                # 🎨 MODERN UI THEME: High-contrast typography (Plus Jakarta Sans & JetBrains Mono).
├── requirements.txt            # 📦 DEPENDENCIES: Full list of required Python packages.
├── TEAMMATE_SETUP_AND_TESTING_GUIDE.md # 🚀 STEP-BY-STEP TUTORIAL for group members and testing.
├── TruthLens_Comprehensive_Project_Notes_and_Viva_Guide.pdf # 📚 Comprehensive viva notes and revision guide.
│
├── qwen-fact-checker/          # 🧠 TRAINED AI MODEL: LoRA fine-tuned adapter weights for Qwen 2.5 3B.
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
│
├── tests/                      # 🧪 AUTOMATED TEST SUITE
│   ├── test_discourse_memory.py       # Pronoun resolution & discourse context tests
│   ├── test_similarity_analyzer.py   # Topical similarity vs plagiarism tests
│   ├── test_trust_engine.py          # Hallucination grounding & trust scoring tests
│   ├── test_report_generator.py      # Report generation & export tests
│   ├── test_plagiarism.py            # Plagiarism detection tests
│   └── test_phase4_pipeline.py       # Complete end-to-end integration test
│
└── training/                   # 🎓 FINE-TUNING & DATA PREPARATION
    ├── train_qwen.py                 # QLoRA fine-tuning script
    ├── prepare_dataset.py            # Dataset formatter
    ├── train.jsonl                   # Training examples
    └── test.jsonl                    # Validation examples
```

---

## 🧪 Running Automated Tests

Verify all system modules in one go:

```bash
# Discourse Memory Engine
python tests/test_discourse_memory.py

# Similarity Analyzer
python tests/test_similarity_analyzer.py

# Trust Score & Hallucination Engine
python tests/test_trust_engine.py

# Report Generator & Export Formats
python tests/test_report_generator.py

# End-to-End Pipeline Integration
python tests/test_phase4_pipeline.py
```

---

## 🎓 Model Fine-Tuning Details

TruthLens uses **Qwen 2.5 3B Instruct** fine-tuned with **QLoRA (Quantized Low-Rank Adaptation)**:
* **Quantization**: 4-bit NormalFloat (NF4) with double quantization.
* **LoRA Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
* **Hardware**: Trained on NVIDIA Laptop GPU (RTX 5050 with 8GB VRAM) in ~12 minutes.
* **Adapter Size**: Only **57 MB** (`qwen-fact-checker/adapter_model.safetensors`), allowing instant deployment without massive downloads.

---

## 📄 License & Attribution

Developed as a 4th Year B.Tech Computer Science & Engineering Capstone Project.  
Base Model: [Qwen 2.5 3B Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) by Alibaba Cloud (Apache 2.0).
