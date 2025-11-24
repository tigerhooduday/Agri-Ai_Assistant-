# Intelligent Agricultural Assistant

## Overview
This project builds an AI assistant that detects crop diseases from images and gives treatment advice grounded in an indexed knowledge base (RAG). It combines a CNN-based image classifier with a retrieval-augmented language model to produce context-aware, citation-backed guidance for farmers and agronomists.

## Why this approach
- Computer vision identifies *what* is wrong with a plant (disease classification + heatmap).
- RAG retrieves domain documents (extension guides, research) to answer *how to fix it*, reducing hallucination and providing sources.

## Features
- Upload crop image → receive disease label(s), confidences, Grad-CAM heatmap.
- Ask natural-language questions — answers include citations to KB sources.
- Integrated flow: "What's wrong?" → "How to treat?" with context-aware follow-ups.

## Quickstart
1. Create environment:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
