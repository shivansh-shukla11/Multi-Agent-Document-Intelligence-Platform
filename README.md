# Multi-Agent Document Intelligence Platform

A Generative AI system that ingests documents (PDF/TXT), builds a semantic
search index over them, and answers user questions using a **team of
cooperating agents** instead of a single RAG chain:

```
User Query
   │
   ▼
┌─────────────┐     simple lookup      ┌────────────────┐
│ Router Agent│ ─────────────────────► │ Retriever Agent │
└─────────────┘                        └────────┬────────┘
   │ complex / multi-step                        │
   ▼                                              ▼
┌─────────────────┐                     ┌──────────────────┐
│ (future) Planner │                    │ Summarizer Agent  │
└─────────────────┘                     └────────┬──────────┘
                                                  │
                                                  ▼
                                         ┌──────────────────┐
                                         │  Verifier Agent   │
                                         │ (checks claims    │
                                         │  against sources) │
                                         └────────┬──────────┘
                                                  │
                                                  ▼
                                          Final answer + sources


