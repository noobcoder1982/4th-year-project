# 🚀 TruthLens — Teammate Setup, Installation & Testing Guide

Welcome to **TruthLens**! This guide is written so that **any teammate in our group of 4** can clone this project, install all dependencies, start the interactive web application, and test every single feature without any technical headaches.

---

## 📌 Project Overview in 30 Seconds

**TruthLens** is an end-to-end AI system that verifies factual claims in written articles:
1. **Reads Any Content**: Paste text, upload **PDFs**, **Word documents (.docx)**, **Text files (.txt)**, or enter live **URLs**.
2. **Breaks Down Sentences**: Extracts individual testable atomic claims.
3. **Discourse Memory Engine**: Tracks pronouns (*"It"*, *"He"*, *"They"*, *"This mission"*) across sentences so context is never lost.
4. **Live Multi-Source Web Retrieval**: Queries DuckDuckGo, Yahoo, Bing, and Wikipedia for credible sources.
5. **AI Verification**: Our fine-tuned **Qwen 2.5 (3B)** model audits claims against retrieved evidence to label them **TRUE**, **FALSE**, or **UNCERTAIN**.
6. **Multi-Signal Trust Score (0–100)**: Evaluates fact verification, source credibility, hallucination grounding, and content originality.
7. **Exportable Dossier**: Generates downloadable **Markdown (.md)** and **JSON (.json)** verification reports.

---

## ⚙️ Step 1: System Requirements & Prerequisites

Before starting, ensure you have:
1. **Python 3.10, 3.11, or 3.12** installed.
   * Check your version in terminal:
     ```bash
     python --version
     ```
2. **Git** installed.
   * Check in terminal:
     ```bash
     git --version
     ```
3. **Hardware Recommendation**:
   * **Recommended**: NVIDIA GPU with 6GB+ VRAM (e.g. RTX 3050, 3060, 4050, 4060, 5050) with CUDA enabled.
   * **CPU Mode**: Supported (Qwen runs in 4-bit or 16-bit, but GPU is much faster).

---

## 📥 Step 2: Clone the GitHub Repository

Open your terminal (PowerShell, Command Prompt, or Terminal) and run:

```bash
git clone https://github.com/noobcoder1982/4th-year-project.git
cd 4th-year-project
```

---

## 🐍 Step 3: Create & Activate a Virtual Environment

Keep your system Python clean by creating an isolated virtual environment:

### On Windows (PowerShell or Command Prompt):
```powershell
python -m venv .venv
.venv\Scripts\activate
```
*(If PowerShell says scripts are disabled, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*

### On macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` appear at the beginning of your command line.

---

## 📦 Step 4: Install PyTorch & Dependencies

### Option A: If you have an NVIDIA GPU (Recommended):
Install PyTorch with CUDA 12.4 support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### Option B: If you are running on CPU only:
```bash
pip install torch torchvision
```

### Next, install all project requirements:
```bash
pip install -r requirements.txt
```

This installs:
* `transformers`, `peft`, `bitsandbytes`, `accelerate` (AI model & LoRA adapter runtime)
* `gradio` (Interactive web UI)
* `ddgs`, `requests`, `beautifulsoup4`, `lxml` (Web search & scraping)
* `pypdf`, `python-docx` (PDF and Word document parsing)
* `pytest`, `reportlab` (Testing & reporting)

---

## 🧪 Step 5: Run Automated Tests to Verify Setup

Run the built-in automated test suites to ensure everything is functioning 100%:

```bash
# 1. Test the Discourse Memory & Pronoun Resolution Engine
python tests/test_discourse_memory.py

# 2. Test Topical Similarity vs. Plagiarism Analysis
python tests/test_similarity_analyzer.py

# 3. Test Hallucination Grounding & Transparent Trust Engine
python tests/test_trust_engine.py

# 4. Test Report Generation & File Exports (Markdown, JSON, HTML)
python tests/test_report_generator.py

# 5. Test Full Pipeline Integration
python tests/test_phase4_pipeline.py
```

If all tests output `PASSED`, your environment is completely configured!

---

## 🖥️ Step 6: Launch TruthLens

Start the web application:

```bash
python app.py
```

### What to Expect on Startup:
1. The terminal displays `LOADING TRUTHLENS`.
2. It loads `Qwen/Qwen2.5-3B-Instruct` base model in 4-bit precision.
3. It loads the fine-tuned LoRA adapter from `./qwen-fact-checker/`.
4. It displays:
   ```text
   Running on local URL: http://127.0.0.1:7860
   ```
5. Open your web browser and navigate to:
   👉 **`http://localhost:7860`**

---

## 🎯 Step 7: How to Test Each Feature (Demo & Viva Walkthrough)

Here are ready-to-use test prompts to demonstrate every capability of TruthLens:

### Test Case 1: Fact-Checking & Discourse Memory Link (TRUE Claim with Pronoun)
* **Input Mode**: Direct Text
* **Paste this article**:
  ```text
  NASA successfully launched the James Webb Space Telescope in December 2021. 
  It reached the Sun-Earth L2 Lagrange point approximately 30 days later. 
  The telescope began transmitting unprecedented deep space infrared imagery to Earth.
  ```
* **Click**: `Verify Article & Audit Originality`
* **What to observe**:
  * **Discourse Memory Pill**: Sentence 2 shows a glowing `🧠 Discourse Memory Link: Resolved pronoun "It" → NASA / James Webb Space Telescope`.
  * **Search Query**: Shows how the search engine was queried with the resolved antecedent instead of an ambiguous pronoun.
  * **Verdicts**: `TRUE` (Supported by NASA/ESA evidence).
  * **Trust Score**: Grade A (High Trust, 90%+).

---

### Test Case 2: Detecting Misinformation & Hallucinations (FALSE Claim)
* **Input Mode**: Direct Text
* **Paste this article**:
  ```text
  The Sun revolves directly around the Earth every twenty-four hours. 
  Humans first walked on Mars in 1965 aboard an Apollo mission.
  ```
* **What to observe**:
  * **Verdicts**: `FALSE` (Refuted with rose red badges).
  * **Hallucination Alert**: Statement flagged with `⚡ CONTRADICTED (High Hallucination Risk)`.
  * **Trust Score**: Grade D/F (`LIKELY MISLEADING`).

---

### Test Case 3: Testing File Uploads (PDF & Word Documents)
1. Switch to the **Upload Document** tab.
2. Drag and drop any `.pdf`, `.docx`, or `.txt` file (e.g. an essay, Wikipedia printout, or news draft).
3. Click `Verify Article`.
4. The system extracts structured text, validates the file format, and streams the verification claim by claim.

---

### Test Case 4: Testing Article URLs (Live Web Ingestion)
1. Switch to the **Article URL** tab.
2. Enter any news article or blog link (e.g., a BBC, Reuters, or Wikipedia URL).
3. The system extracts the main readable body text, strips navigation ads, and performs claim verification.

---

### Test Case 5: Originality Audit vs. Text Similarity
1. Enable the **Plagiarism Audit** toggle.
2. Optional: Paste a reference document or syllabus excerpt into the Reference Document box.
3. Observe the distinction between:
   * **Topical Similarity**: Blue pill indicating natural shared terminology (e.g., scientific terms like *atmosphere, carbon, exoplanet*).
   * **Plagiarism**: Yellow/Orange pill indicating verbatim copying or heavy paraphrasing.

---

### Test Case 6: Download Verification Dossier
1. Scroll to the bottom of the results page.
2. Look at the **Comprehensive Verification Dossier** card.
3. Click **⬇ Export Markdown (.md)** or **⬇ Export JSON (.json)**.
4. Your browser will instantly download the complete structured verification report with all claims, confidence values, source URLs, and credibility scores.

---

## 🛠️ Common Troubleshooting

| Issue / Error | Cause | Instant Fix |
|---|---|---|
| `torch not compiled with CUDA enabled` | PyTorch installed without CUDA wheels | Run: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124` |
| `CUDA out of memory` | GPU VRAM is full from another program | Close other GPU applications or verify `BitsAndBytesConfig(load_in_4bit=True)` is enabled in `app.py`. |
| `triton not found` warning | Windows does not natively support Triton | This is a normal warning on Windows; PyTorch runs seamlessly without Triton. |
| DuckDuckGo transient timeout | Network latency or rate limiting | The app automatically retries and falls back to Yahoo/Bing/Wikipedia providers. |
| Port 7860 already in use | Another Gradio app is running | Run on another port: in `app.py`, change `demo.launch(server_port=7861)`. |

---

## 👥 Project Team Viva Checklist

When presenting this project to evaluators:
1. **Explain the Architecture**: Ingestion $\rightarrow$ Discourse Memory $\rightarrow$ Multi-Provider Retrieval $\rightarrow$ Qwen 2.5 3B (QLoRA) $\rightarrow$ Trust Score & Dossier.
2. **Highlight Novel Contributions**:
   * **Discourse Memory Engine**: Solves sentence-isolation blindness by resolving pronouns and anchoring article themes.
   * **Separation of Similarity vs. Plagiarism**: Doesn't falsely penalize standard domain terminology.
   * **Transparent Trust Score**: 4-pillar transparent formula instead of a black-box percentage.
3. **Show Live Exports**: Demonstrate the 1-click Markdown/JSON download and the comprehensive PDF notes guide ([`TruthLens_Comprehensive_Project_Notes_and_Viva_Guide.pdf`](./TruthLens_Comprehensive_Project_Notes_and_Viva_Guide.pdf)).
