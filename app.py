import os
import re
import tempfile
from typing import List, Dict, Optional
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridUpdateMode
from pdfminer.high_level import extract_text
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Streamlit config must be first
st.set_page_config(layout="wide", page_title="PDF to Knowledge Graph")

# Initialize NLP tools
try:
    nlp = spacy.load("en_core_web_sm")
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
except Exception as e:
    st.error(f"Failed to initialize NLP tools: {e}")
    st.stop()

# Initialize Neo4j connection (optional)
neo4j_enabled = False
graph = None
try:
    from py2neo import Graph, Node, Relationship
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    
    graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    graph.run("MATCH (n) RETURN n LIMIT 1")  # Test connection
    neo4j_enabled = True
except Exception as e:
    st.warning(f"Neo4j connection failed (some features disabled). Error: {e}")

# Initialize Ollama (optional)
ollama_enabled = False
try:
    import ollama
    ollama.list()  # Test connection
    ollama_enabled = True
except Exception as e:
    st.warning(f"Ollama connection failed (some features disabled). Error: {e}")

# Main app
st.title("PDF to Knowledge Graph Generator")

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF file."""
    try:
        text = extract_text(pdf_path)
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def preprocess_text(text: str) -> str:
    """Preprocess text by removing noise and normalizing."""
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = ' '.join(text.split())
    return text

def extract_entities(text: str) -> List[Dict]:
    """Extract entities from text using spaCy."""
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "WORK_OF_ART"]:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
    return entities

def extract_relations(text: str, entities: List[Dict]) -> List[Dict]:
    """Extract relationships between entities."""
    doc = nlp(text)
    relations = []
    
    for sent in doc.sents:
        sent_entities = [e for e in entities if e["start"] >= sent.start_char and e["end"] <= sent.end_char]
        
        if len(sent_entities) >= 2:
            for token in sent:
                if token.dep_ in ("nsubj", "dobj", "pobj", "attr", "prep"):
                    subject = token.text if token.dep_ == "nsubj" else token.head.text
                    obj = token.head.text if token.dep_ == "nsubj" else token.text
                    
                    subj_entity = next((e for e in sent_entities if subject.lower() in e["text"].lower()), None)
                    obj_entity = next((e for e in sent_entities if obj.lower() in e["text"].lower()), None)
                    
                    if subj_entity and obj_entity:
                        relations.append({
                            "source": subj_entity["text"],
                            "target": obj_entity["text"],
                            "relation": token.dep_,
                            "sentence": sent.text
                        })
    return relations

def build_knowledge_graph(entities: List[Dict], relations: List[Dict]):
    """Build knowledge graph in Neo4j."""
    if not neo4j_enabled:
        st.error("Neo4j connection not available")
        return
    
    try:
        graph.run("MATCH (n) DETACH DELETE n")
        nodes = {}
        
        for entity in entities:
            if entity["text"] not in nodes:
                node = Node(entity["label"], name=entity["text"])
                graph.create(node)
                nodes[entity["text"]] = node
        
        for relation in relations:
            if relation["source"] in nodes and relation["target"] in nodes:
                rel = Relationship(nodes[relation["source"]], relation["relation"].upper(), nodes[relation["target"]])
                graph.create(rel)
                
        st.success("Knowledge graph built successfully!")
    except Exception as e:
        st.error(f"Error building knowledge graph: {e}")

def visualize_knowledge_graph():
    """Visualize the knowledge graph."""
    if not neo4j_enabled:
        st.error("Neo4j connection not available")
        return
    
    try:
        query = """
        MATCH (n)-[r]->(m)
        RETURN n.name as source, type(r) as relationship, m.name as target
        LIMIT 100
        """
        result = graph.run(query).data()
        
        if not result:
            st.warning("No relationships found in the knowledge graph")
            return
        
        df = pd.DataFrame(result)
        AgGrid(df, key='knowledge_graph', update_mode=GridUpdateMode.NO_UPDATE)
        
        st.subheader("Knowledge Graph Relationships")
        for row in result:
            st.write(f"{row['source']} -- {row['relationship']} --> {row['target']}")
    except Exception as e:
        st.error(f"Error visualizing knowledge graph: {e}")

def query_knowledge_graph(question: str) -> str:
    """Use LLM to answer questions."""
    if not neo4j_enabled or not ollama_enabled:
        return "Required services (Neo4j/Ollama) not available"
    
    try:
        query = """
        MATCH (n)-[r]->(m)
        WHERE toLower(n.name) CONTAINS toLower($keyword) OR toLower(m.name) CONTAINS toLower($keyword)
        RETURN n.name as source, type(r) as relationship, m.name as target
        LIMIT 10
        """
        
        keywords = [token.text for token in nlp(question) if not token.is_stop and token.is_alpha]
        if not keywords:
            return "No relevant information found"
        
        graph_data = []
        for keyword in keywords[:3]:
            result = graph.run(query, keyword=keyword).data()
            graph_data.extend(result)
        
        if not graph_data:
            return "No relevant information found"
        
        context = "Knowledge Graph Information:\n"
        for row in graph_data:
            context += f"- {row['source']} is related to {row['target']} via {row['relationship']}\n"
        
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        response = ollama.generate(model='llama2', prompt=prompt)
        return response['response']
    except Exception as e:
        return f"Error querying knowledge graph: {e}"

def main():
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        text = extract_text_from_pdf(tmp_path)
        os.unlink(tmp_path)
        
        if text:
            with st.expander("View extracted text"):
                st.text(text[:2000] + "...")
            
            with st.spinner("Processing PDF..."):
                entities = extract_entities(text)
                relations = extract_relations(text, entities)
                
                if entities and relations:
                    st.success(f"Extracted {len(entities)} entities and {len(relations)} relationships")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Entities")
                        AgGrid(pd.DataFrame(entities), key='entities')
                    with col2:
                        st.subheader("Relationships")
                        AgGrid(pd.DataFrame(relations), key='relations')
                    
                    if neo4j_enabled:
                        if st.button("Build Knowledge Graph"):
                            with st.spinner("Building..."):
                                build_knowledge_graph(entities, relations)
                                visualize_knowledge_graph()
                    
                    if neo4j_enabled and ollama_enabled:
                        st.subheader("Ask Questions")
                        question = st.text_input("Enter your question:")
                        if question:
                            with st.spinner("Searching..."):
                                answer = query_knowledge_graph(question)
                                st.info(answer)
                else:
                    st.warning("No entities or relationships found")

if __name__ == "__main__":
    main()