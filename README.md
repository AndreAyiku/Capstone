# Can AI Move You?
## EVALUATING THE EMOTIONAL DETECTION AND INFLUENCE EFFECTS OF OPEN-WEIGHT LARGE LANGUAGE MODELS ON HUMAN EMOTIONAL STATES

**Author:** Andre Ayiku - Ashesi University, Class of 2026  
**Student ID:** 84272026  
**Year:** 2026

---

## Project Overview

This project investigates whether open-weight large language models (LLMs) can detectably shift human emotional states through a structured, AI-mediated conversation. It is simultaneously:

- A **research study** measuring AI emotional influence using validated psychometric tools
- A **comparative model evaluation** across 5 open-weight LLMs

The system runs participants through a five-phase protocol and measures before-and-after PANAS scores to quantify the emotional impact of AI-generated narrative influence.

---

## Five-Phase Protocol

| Phase | Name | Description |
|-------|------|-------------|
| 1 | Casual Chat | 8-turn rapport-building; model extracts user's emotional situation |
| 2 | Baseline Assessment | DASS-21 + PANAS administered adaptively via RAG-based question selection |
| 3 | Influence Phase | AI applies directional influence anchored to a single real-world news story |
| 4 | Re-Assessment | Full 20-item PANAS re-administered |
| 5 | Comparison | Before/after Δ Positive Affect and Δ Negative Affect computed and displayed |

---

## Key Results

- Statistically significant PANAS shifts measured across 10 human sessions + 50 AI-simulated sessions
- Largest recorded effect: **−39 Positive Affect / +24 Negative Affect** in a single session (negative influence)
- Influence success rate and effect size varied across model architectures
- Larger parameter models showed stronger and more consistent emotional influence

---

## Models Evaluated

| Model | Parameters |
|-------|-----------|
| gemma-2-9b-it | 9B |
| gemma-3-4b-it | 4B |
| llama-3.1-8b-instruct | 8B |
| llama-3.2-3b-instruct | 3B |
| mistral-7b-instruct | 7B |

Model weights are **not** included in this repository.

---

## Psychometric Instruments

- **DASS-21** — Depression Anxiety Stress Scales (21 items, 3 subscales)
- **PANAS** — Positive and Negative Affect Schedule (20 items, 2 subscales)

A 0–10 user-facing scale is used throughout for consistency and clarity. Responses are automatically converted to official DASS and PANAS scoring scales.

---

## Project Structure

```
emotional_chatbot_project/
├── RAG_setup Emotionoal State (3).ipynb  # Main research notebook (primary deliverable)
├── human_session_analysis.ipynb          # Statistical analysis notebook
├── requirements.txt                      # Python dependencies
│
├── backend/
│   └── chatbot_complete.py               # EmotionalStateChatbot class (all 5 phases)
│   
│
├── data/
│   ├── news_stories.py                   # 10 happy + 10 sad real news stories
│   └── model_manager.py                  # Multi-model loading and switching
│
├── test_results/                         # CSV exports of collected data
└── tables/                               # JSON exports from Supabase
```

---

## Quick Start

### Run in Jupyter

1. Open `RAG_setup Emotionoal State (3).ipynb`
2. Run Cells 1–5 to load models and initialise the chatbot
3. Run Cell 6 for the full study loop
4. Run Cell 7 for fast-testing (auto-fills 8 casual turns)


---

## Ethical Safeguards

| Safeguard | Implementation |
|-----------|---------------|
| Informed consent | Disclaimer shown before influence phase begins |
| STOP safe word | Typing `STOP` exits the study at any time without penalty |
| Debrief | Full study purpose revealed after completion |
| Data anonymisation | Sessions identified by UUID; no names stored |
| Screening only | DASS-21 used for screening, not clinical diagnosis |

---

## Requirements

- Python 3.8+
- CUDA GPU with 16GB+ VRAM (for running LLM inference)
- Jupyter Notebook / JupyterLab

Key Python packages: `torch`, `transformers`, `chromadb`, `sentence-transformers`

---

## License

This project was developed as a capstone project at Ashesi University. All code is the original work of the author. The DASS-21 and PANAS instruments are used under academic fair use for non-commercial research.
