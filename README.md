# Ecometricx Take Home - Data Scientist

# Read PDF Report and Notebook
[View Notebook](rag_ecometrics.ipynb) \
[View HTML Report](rag_ecometrics.html) \
[View PDF Report](rag_ecometrics.pdf)

## You will be given a PDF document that contains both textual and graphical data. Your task is to:

* Extract the textual and graphical information from the PDF pages.
* Convert the extracted graphical data (such as charts or graphs) into a structured, queryable format.
* Implement a system where users can ask questions and receive meaningful responses based on the extracted data.


## Requirements:
* Document your approach and display your results in a Jupyter notebook (.ipynb)
* Your solution should allow users to query both the extracted text and any data that was derived from the graphical elements (such as tables).
* Provide brief explanations of your approach, choices made, and any challenges you encountered.

# Solution Overview

To tackle the the requirements listed above I will leverage a **Multimodal RAG** (Retrieval Augmented Generation). An overall architecture is shown below:
![architecture overview](diagrams/architecture%20overview.png)

The diagram starts with the input PDF file. The unstructured Python library is used to extract both textual and image content from the document. Text and image summarization models then generate summaries and captions based on the extracted content. These outputs are stored in a vector database, enabling retrieval in response to user queries. Additionally, a general-purpose LLM can be used to generate answers based on the top documents retrieved from the vector database. This entire multimodal RAG workflow is made possible through extensive use of the LangChain framework, which orchestrates the components for document loading, embedding, retrieval, and response generation.

---

# Environment Setup
**Python**

- Using Python version 3.9.11
- Create virtual environment and activate \
`python -m venv .venv` \
`.venv/Scripts/activate`

- Install dependencies \
`pip install -r requirements.txt`



**Unstructured Python Library dependency to properly read PDF**
- Download [Poppler](https://github.com/oschwartz10612/poppler-windows/releases) extract and put bin directory to your `PATH`
- Download and install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) put bin directory to your `PATH`

**Ollama Setup (Will act as LLM generator)**
- Install [Ollama](https://ollama.com/download)
- Get a model \
`ollama pull <model_name>` \
e.g. `ollama pull gemma2`
- List models \
`ollama list`

---

# Directory Structure

- `\data\` - Contains the singular PDF file
- `\diagrams\` - Contains supplementary images and drawio diagram
- `helper.py` – Contains utility functions to keep the notebook clean and organized
- `rag_ecometricx.ipynb` - Main jupyter notebook
- `README.md` - README
- `requirements.txt` - Python dependencies