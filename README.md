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

Ensure the following packages are installed:

```bash
pip install pandas numpy scikit-learn openpyxl


## 🖼️ **Screenshots**
![Screenshot 2025-06-07 120309](https://github.com/user-attachments/assets/83580b97-02f5-476d-bfda-cff656ef41a4)
![Screenshot 2025-06-07 120256](https://github.com/user-attachments/assets/89044edb-3398-4ae1-83db-598e8757829b)
![Screenshot 2025-06-07 120221](https://github.com/user-attachments/assets/f96f4406-c943-4dcc-a1ca-b6e1750038a7)
![Screenshot 2025-06-07 120159](https://github.com/user-attachments/assets/946903c4-78e3-42f5-9e43-11d856bb17d5)

Author Vaisnav R R
