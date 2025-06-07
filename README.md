# HSN Code Validation and Suggestion Agent

## Overview

This Python-based agent utilizes **Google's Agent Development Kit (ADK)** to perform intelligent operations on Harmonized System of Nomenclature (HSN) codes. It supports:

- ✅ HSN Code Format and Hierarchy Validation  
- 🔍 Semantic Search and Code Suggestions  
- 🧠 AI-Powered TF-IDF Matching  
- 🧰 Modular Tools for Lookup, Suggestion, and Validation  
- 🧪 CLI-based Testing Interface

> **Note**: Ensure `HSN_SAC.xlsx` (the HSN master data) is present in the working directory.

---

## Features

### 1. **Validation Engine**
- Validates the HSN code format (length & digit-only).
- Checks existence in the master dataset.
- Validates hierarchical consistency (parent codes at 2, 4, 6 levels).

### 2. **Suggestion Engine**
- Suggests top 5 HSN codes using TF-IDF vector similarity.
- Adds reasoning and top-level category for each match.

### 3. **Data Handler Tool**
- Provides lookup, search, and hierarchical traversal functionality.

### 4. **Interactive CLI**
- Validate a single code
- Suggest based on product description
- Batch validate multiple codes

---

## Installation

### Prerequisites

Ensure the following packages are installed: numpy pandas scikit-learn

### Screenshots

![Screenshot 2025-06-07 120159](https://github.com/user-attachments/assets/9a7268d5-9e14-47d7-b801-661aa26e85ac)

![Screenshot 2025-06-07 120221](https://github.com/user-attachments/assets/771c5baa-9487-4b18-9b8a-2e4ebeabfbbf)
![Screenshot 2025-06-07 120309](https://github.com/user-attachments/assets/6cc118a8-5e12-405e-9b99-1218c9d1c86b)
![Screenshot 2025-06-07 120256](https://github.com/user-attachments/assets/d0fca1c6-d458-40e5-b1bd-793fa581bab1)
