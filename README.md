# Help Website Q&A Agent

This AI-powered question-answering agent can process documentation from help websites and accurately answer user queries about product features, integrations, and functionality.

## Table of Contents
- [Setup Instructions](#setup-instructions)
- [Dependencies](#dependencies)
- [Design decisions](#design-decisions)
- [Known limitations](#known-limitations)

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/reyanalam/AI-powered-question-answering-agent.git
```
### 2. Create and activate a virtual environment
```
conda create -p venv python=3.12
conda activate venv/
```
### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. To run the system 
```
python semantic_search.py
```

Input the URL (e.g., https://help.zluri.com/) and hit enter.

### 6. Ask questions 

Type the question in the terminal and hit enter.

### 7. Testing
```
python test.py
```

Enter the following URL - https://help.zluri.com/ 


## Dependencies

Following are the dependencies 

```
python=3.12
requests
beautifulsoup4 
urllib3 
tldextract
spacy
numpy
```


## Design decisions

### Tech Stack Choice

* 1. *Python*: Chosen for its rich ecosystem in NLP and seamless compatibility with most AI libraries.

* 2. *requests*: Used for making robust HTTP requests to fetch help site content efficiently.

* 3. *beautifulsoup4*: Ideal for parsing and extracting meaningful content (like headers, paragraphs) from the raw HTML of help pages.

* 4. *urllib3* & *tldextract*: Ensure clean URL parsing, domain extraction, and management of redirects or relative links.

* 5. *spaCy*: Chosen for preprocessing of the raw text and for creating vector embeddings.


### Architecture

* 1. *Web Crawling*:  Implemented a custom class to crawl through documentation pages of a website starting from a base_url, extract meaningful content (like lists, paragraphs, and tables), and optionally follow internal links to crawl sub-pages up to a defined max_depth.

This ensure focusing only on relevant documentation pages, extracts structured, human-readable content, and recursively follows internal links with control.

* 2. *Preprocessing*: Used large model of Spacy to effectively preprocess the raw text. 

Spacy provides variety of models, among which large is chosed because it supports vectors for english words.

* 3. *Chunking*: Deviced custom chunking logic. Each chunk is contains tokens (words) that are semantically similar. This is done via calculating the cosine similarity and setting a threshold for it. Once the similarity drops below the threshold the token would not be added to the previous chunk and a new chunk will start.

* 4. *Search*: Embedded the chunks and query using Spacy large model. It computes the vector for each token (word) and then takes an average of all the vectors.

Top 3 chunks are then returned.

### Known limitations

* 1. The agent cannot scrap the dynamic content of the website that uses JavaScript.

* 2. Scalability is problem for this agent as it uses traditional NLP techniques.

### Suggestions for improvements

* 1. Chunking startegy can be modified. Langchain provides various chunking technique than can be used to create relevant chunks.

* 2. Pre-trained transformers can be used for creating contextual embeddings. This will result it accurate response.

* 3. Effective search algorithms can be used instead of using cosine similarity.