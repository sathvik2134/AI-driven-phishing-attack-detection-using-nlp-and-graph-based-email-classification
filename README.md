#  AI Driven Phishing Attack Detection using NLP & Graph Based Email Classification

![Python](https://img.shields.io/badge/Python-3.9-blue?style=flat-square)
![Flask](https://img.shields.io/badge/Backend-Flask-green?style=flat-square)
![PyTorch](https://img.shields.io/badge/DeepLearning-PyTorch-red?style=flat-square)
![TensorFlow](https://img.shields.io/badge/DeepLearning-TensorFlow-orange?style=flat-square)
![Transformers](https://img.shields.io/badge/NLP-RoBERTa-yellow?style=flat-square)
![ONNX](https://img.shields.io/badge/Inference-ONNX_Runtime-purple?style=flat-square)
![Explainable AI](https://img.shields.io/badge/XAI-Enabled-teal?style=flat-square)
![Cybersecurity](https://img.shields.io/badge/Domain-CyberSecurity-black?style=flat-square)

---

##  Project Overview

This project presents a **hybrid artificial intelligence cybersecurity system** designed to detect phishing emails using semantic understanding, communication behavior analysis, and adversarial learning.

Unlike traditional spam filters that rely on keywords or blacklists, this system performs **context-aware threat detection** by combining:

- Natural Language Processing (email intent)
- URL behavior analysis (link structure)
- Graph modeling (communication relationships)
- Adversarial learning (zero-day attack detection)

The system acts like an automated security analyst capable of understanding both **what the email says and how it behaves**.

---

##  Problem Statement

Traditional phishing detection techniques fail because modern attackers:

- Mimic writing style of real employees
- Use newly registered domains
- Dynamically generate phishing content
- Avoid blacklisted keywords
- Impersonate trusted entities

Therefore, detection must go beyond surface-level patterns and analyze:

> **Language + Behavior + Context + Relationships**

---

##  Objectives

The system aims to:

- Detect sophisticated phishing emails
- Reduce false positives
- Identify spoofed domains
- Detect zero-day attacks
- Provide explainable detection reasoning
- Enable real-time monitoring

---

##  Core Concept — Multi-Modal Intelligence

Instead of a single classifier, the system uses multiple intelligence layers:

| Intelligence Layer | What It Understands |
|-------------------|-------------------|
| NLP | Meaning and intent of the email |
| URL Analysis | Malicious link characteristics |
| Graph Analysis | Suspicious communication patterns |
| GAN | Previously unseen attacks |
| Score Fusion | Final decision making |
| Explainable AI | Reason behind classification |

---

##  High Level Architecture

            ┌─────────────────────────┐
            │       User Input        │
            │ Email / URL / Metadata  │
            └─────────────┬───────────┘
                          │
                          ▼
            ┌─────────────────────────┐
            │     Preprocessing       │
            │ Cleaning & Tokenizing   │
            └─────────────┬───────────┘
                          │
    ┌─────────────────────┼─────────────────────┐
    ▼                     ▼                     ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ NLP Model │ │ URL Analyzer │ │ Graph Model │
│ (RoBERTa) │ │ Lexical/WHOIS│ │ Relationships│
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
│ │ │
└─────────────┬──────┴─────────────┬────┘
▼ ▼
┌─────────────────────────────┐
│ Score Fusion Engine │
└─────────────┬───────────────┘
▼
┌─────────────────────────────┐
│ Explainable AI Layer │
└─────────────┬───────────────┘
▼
┌─────────────────────────────┐
│ Dashboard & REST API Layer │
└─────────────────────────────┘


---

##  End-to-End Workflow

User submits email or URL
↓
Text preprocessing
↓
Parallel multi-modal analysis
├── Semantic understanding (Transformer NLP)
├── URL behavior analysis
└── Communication graph analysis
↓
GAN robustness validation
↓
Hybrid score fusion
↓
Confidence + reasoning output
↓
Stored in monitoring dashboard


---

##  Why Hybrid Detection?

A phishing email may look legitimate textually but malicious behaviorally.

Example:

| Email Text | Domain | Result |
|---------|------|------|
| Looks safe | Newly registered | Phishing |
| Urgent payment | Trusted domain | Legitimate |
| Clean text | Suspicious sender pattern | Phishing |

Single models fail — hybrid reasoning succeeds.

---
## Machine Learning Pipeline

The detection engine is implemented as a staged machine learning pipeline where each module contributes a different type of intelligence signal.  
Instead of relying on a single classifier, the system aggregates semantic, structural, and behavioral evidence before making a final decision.

Pipeline stages:

1. Data ingestion
2. Text preprocessing
3. Feature engineering
4. Multi-model inference
5. Score fusion
6. Explainability

---

## 1. Data Ingestion

The system accepts three primary inputs:

- Email body text
- URLs embedded inside email
- Sender metadata

Example metadata used:
- Sender email address
- Domain name
- Recipient address
- Timestamp
- Header patterns

This information allows the system to reason about communication authenticity instead of only message content.

---

## 2. Text Preprocessing (NLP Preparation)

The email text passes through a preprocessing stage before entering the transformer model.

Steps performed:

### Cleaning
- Remove HTML tags
- Normalize whitespace
- Remove special artifacts
- Lowercasing

### Tokenization
Text is converted into tokens using the RoBERTa tokenizer.

### Stopword Handling
Common words that carry little semantic meaning are filtered when generating auxiliary features (not for transformer input).

### TF-IDF Representation
A parallel sparse representation is generated for additional statistical signals.

---

## 3. Feature Engineering

The system extracts three independent feature groups.

---

### A. Semantic Features (NLP)

Model: RoBERTa Transformer

Purpose: Understand intent and deception patterns

The model captures:
- urgency phrases
- impersonation tone
- payment requests
- credential harvesting patterns
- abnormal sentence structures

Unlike keyword matching, contextual embeddings allow detection even if attackers paraphrase the message.

---

### B. URL Behavioral Features

The system analyzes embedded URLs using lexical, structural, and network attributes.

#### Lexical Features
- URL length
- Number of dots
- Special characters (@, -, _)
- Presence of IP address
- Suspicious keywords in path

#### Structural Features
- Directory depth
- Redirection patterns
- Query parameter entropy

#### Domain Features
- Domain age
- WHOIS registration information
- SSL presence
- Hosting mismatch indicators

These signals detect malicious links even when text appears legitimate.

---

### C. Graph Relationship Features

A communication graph is constructed:

Nodes:
- senders
- receivers
- domains

Edges:
- communication frequency
- historical interaction
- trust relationships

Anomalies detected:
- new sender impersonating trusted user
- unusual communication path
- abnormal domain clusters
- sudden behavior change

This enables detection of social engineering attacks.

---

## 4. Hybrid Model Inference

Each modality produces an independent probability score:

| Module | Output |
|------|------|
| RoBERTa NLP | semantic phishing probability |
| URL classifier | link risk probability |
| Graph analyzer | behavior anomaly score |

---

## 5. Adversarial Robustness using GAN

Problem:
Traditional classifiers fail on unseen phishing styles.

Solution:
A Generative Adversarial Network generates synthetic phishing samples.

### Generator
Creates realistic phishing variations.

### Discriminator
Learns to distinguish real vs fake phishing samples.

Effect:
- improves generalization
- detects zero-day attacks
- reduces overfitting

---

## 6. Score Fusion Decision Engine

The final decision is not taken by a single model.

Instead a fusion function combines all signals:

Final Score =
Weighted(Semantic) +
Weighted(URL) +
Weighted(Graph)

Decision thresholds:

| Score | Classification |
|-----|------|
| < 0.4 | Legitimate |
| 0.4 – 0.7 | Suspicious |
| > 0.7 | Phishing |

This reduces false positives and false negatives.

---

## 7. Explainable AI Layer

The system provides interpretable output instead of only a label.

Output includes:

- phishing probability
- suspicious keywords
- risky domain indicators
- abnormal sender behavior
- reasoning summary

This allows security analysts to trust the model decision.

---

## Why This Pipeline Works

Single-model detection fails because phishing attacks manipulate only one dimension at a time.

The hybrid pipeline succeeds because it validates:

- what the email says
- where the link goes
- who sent it
- whether the behavior is normal

The attack must bypass all layers simultaneously, making evasion significantly harder.

## Backend System Architecture

The application layer exposes the machine learning pipeline as a production style service.  
The backend is responsible for request handling, model invocation, data storage, monitoring and response formatting.

The backend is implemented using Flask and organized into modular services.

High level backend flow:

Client Request → API Layer → Preprocessing → ML Inference → Score Fusion → Explanation → Response → Logging

---

## API Design

The system exposes REST APIs that allow integration with external applications or a web interface.

### 1. Scan Email

POST /scan/email

Input:
- email_text
- sender
- receiver
- timestamp

Process:
1. Preprocessing
2. Feature extraction
3. Model inference
4. Score fusion
5. Explanation generation

Output:
- classification
- probability
- explanation
- suspicious indicators

---

### 2. Scan URL

POST /scan/url

Input:
- url

Process:
1. URL feature extraction
2. Domain lookup
3. Classification

Output:
- risk score
- domain indicators
- classification

---

### 3. History Logs

GET /logs

Returns previously scanned results including timestamps and classifications.

---

### 4. System Status

GET /health

Used for monitoring and deployment health checks.

---

## Database Layer

The system uses SQLite for persistent storage.

Stored information:

- scan id
- email content hash
- sender address
- url extracted
- classification result
- probability score
- timestamp

Purpose:
- audit tracking
- monitoring attacks over time
- dashboard visualization

---

## Dashboard Monitoring System

The dashboard provides a visual interface for analysis.

Features:

- real time scan results
- phishing probability visualization
- suspicious keyword highlighting
- historical trend analysis
- risk distribution graphs

The dashboard acts as a lightweight Security Operations Center view.

---

## Explainability Interface

Each prediction is accompanied by reasoning signals.

Displayed to the user:

- suspicious phrases detected
- abnormal domain behavior
- sender anomaly
- model confidence

This helps analysts verify automated decisions.

---

## Model Serving and Inference Optimization

Deep learning models are optimized for real-time use.

Techniques used:

- ONNX runtime conversion for faster inference
- CPU optimized execution
- reduced latency predictions
- batching disabled for instant response

Average response time target: sub-second inference.

---

## Deployment Architecture

The application can run locally or on a production server.

Recommended deployment stack:

Flask Application  
→ Gunicorn WSGI Server  
→ Nginx Reverse Proxy  

Benefits:
- concurrency handling
- secure routing
- production stability

---

## Performance Considerations

The system is designed for scalability and reliability.

Strategies:

- lightweight database queries
- optimized transformer inference
- independent module execution
- minimal blocking operations

The architecture allows handling multiple scan requests without blocking the application.

---

## Logging and Monitoring

The system records all scans for monitoring purposes.

Logs include:

- timestamp
- input type
- classification result
- probability score

This enables detection trend analysis and security auditing.

---

## Security Considerations

To prevent misuse:

- input sanitization
- restricted file handling
- no raw email storage (hashing used)
- API endpoint validation

The system is designed as an analysis tool, not a mail relay, reducing attack surface.

## Dataset Information

The system uses email phishing datasets containing:

- legitimate emails
- phishing emails
- malicious URLs

Data preprocessing performed:

- cleaning
- normalization
- balancing
- feature selection

GAN augmentation is used to create synthetic phishing samples to improve generalization.

---

## Model Evaluation

The model is evaluated using standard classification metrics.

### Metrics Used

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

### Performance Goals

- High recall (detect maximum phishing)
- Low false positives
- Fast inference time

---

## Detection Capabilities

The system can identify:

- credential harvesting emails
- fake payment requests
- spoofed domains
- impersonation attempts
- suspicious communication behavior
- unseen phishing variations

---

## Advantages Over Traditional Filters

| Traditional Filters | Proposed System |
|-------------------|--------------|
| keyword matching | contextual understanding |
| blacklist checking | behavior analysis |
| single classifier | multi-model reasoning |
| no explanations | explainable output |
| weak to new attacks | robust to zero-day attacks |

---


## License

2025 9th International Conference on Computational System and Information Technology for Sustainable Solutions (CSITSS)






















