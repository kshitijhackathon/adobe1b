"""
Round 1B: Enhanced Persona-Driven Document Intelligence
Modified to accept JSON input files with PDF document lists
"""

import os
import json
import sys
import re
import argparse
import gc
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Generator
from collections import defaultdict
import fitz 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer

def setup_nltk():
    """Setup NLTK data with error handling"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
        except:
            print("Warning: Could not download NLTK punkt tokenizer")
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        try:
            nltk.download('stopwords', quiet=True)
        except:
            print("Warning: Could not download NLTK stopwords")

setup_nltk()

class MemoryEfficientDocumentProcessor:
    def __init__(self, chunk_size: int = 50):
        """Initialize with memory optimization settings"""
        self.chunk_size = chunk_size  
        self.stemmer = PorterStemmer()
        
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
                'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                'further', 'then', 'once'
            }
    
    def extract_sections_generator(self, pdf_path: str) -> Generator[Dict, None, None]:
        """Memory-efficient section extraction using generator"""
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            print(f"Processing {total_pages} pages from {os.path.basename(pdf_path)}")
            
            current_section = None
            
            for chunk_start in range(0, total_pages, self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, total_pages)
                
                for page_num in range(chunk_start, chunk_end):
                    try:
                        page = doc[page_num]
                        blocks = page.get_text("dict")["blocks"]
                        
                        for block in blocks:
                            if block.get("type") == 0:  
                                for line in block["lines"]:
                                    for span in line["spans"]:
                                        text = span["text"].strip()
                                        size = span["size"]
                                        flags = span["flags"]
                                        
                                        if not text or len(text) < 2:
                                            continue
                                        
                                        is_heading = self._is_heading(text, size, flags)
                                        
                                        if is_heading and len(text) > 2:
                                            if current_section and len(current_section["content"].strip()) > 50:
                                                yield current_section
                                            
                                            current_section = {
                                                "document": os.path.basename(pdf_path),
                                                "page_number": page_num + 1,
                                                "section_title": self._clean_heading(text),
                                                "content": "",
                                                "importance_rank": -1,
                                                "word_count": 0
                                            }
                                        else:
                                            if current_section:
                                                current_section["content"] += " " + text
                                                current_section["word_count"] += len(text.split())
                                                
                                                if current_section["word_count"] > 2000:
                                                    yield current_section
                                                    current_section = {
                                                        "document": os.path.basename(pdf_path),
                                                        "page_number": page_num + 1,
                                                        "section_title": f"{current_section['section_title']} (continued)",
                                                        "content": "",
                                                        "importance_rank": -1,
                                                        "word_count": 0
                                                    }
                        
                        page = None
                        
                    except Exception as e:
                        print(f"Error processing page {page_num}: {str(e)}")
                        continue
                
                # Force garbage collection after each chunk
                gc.collect()
            
            if current_section and len(current_section["content"].strip()) > 50:
                yield current_section
            
            doc.close()
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
    
    def _is_heading(self, text: str, size: float, flags: int) -> bool:
        """Enhanced heading detection with multiple heuristics"""
        if size >= 14 or (flags & 2**4):  
            return True
        
        heading_patterns = [
            r'^\d+\.?\s+[A-Z]',           
            r'^\d+\.\d+\.?\s+[A-Z]',      
            r'^[IVX]+\.?\s+[A-Z]',        
            r'^[A-Z][A-Z\s]{2,}$',        
            r'^Chapter\s+\d+',            
            r'^Section\s+\d+',            
            r'^Part\s+[IVX\d]+',          
            r'^Appendix\s+[A-Z]?',        
            r'^\d+\s+[A-Z][a-z]+',        
            r'^[A-Z][a-z]+:$',            
        ]
        
        for pattern in heading_patterns:
            if re.match(pattern, text.strip()):
                return True
        
        words = text.split()
        if len(words) >= 2:
            capitalized = sum(1 for word in words if word and word[0].isupper())
            if capitalized / len(words) > 0.7:
                return True
        
        if len(text) < 100 and text[0].isupper() and not text.endswith('.'):
            return True
        
        return False
    
    def _clean_heading(self, text: str) -> str:
        """Clean and normalize heading text"""
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^\w\s\-:()]+$', '', text)
        
        return text
    
    def preprocess_text(self, text: str) -> str:
        """Optimized text preprocessing"""
        if not text:
            return ""
        
        
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = text.lower()
        
        words = text.split()
        
        filtered_words = []
        for word in words:
            if len(word) > 2 and word not in self.stop_words:
                if word.endswith('ing'):
                    word = word[:-3]
                elif word.endswith('ed'):
                    word = word[:-2]
                elif word.endswith('s') and len(word) > 3:
                    word = word[:-1]
                filtered_words.append(word)
        
        return ' '.join(filtered_words)
    
    def calculate_relevance_scores_batch(self, sections: List[Dict], persona: str, job: str) -> List[Dict]:
        """Memory-efficient relevance calculation with batching"""
        if not sections:
            return sections
        
        print(f"Calculating relevance for {len(sections)} sections...")
        
        query = f"{persona} {job}"
        query_processed = self.preprocess_text(query)
        
        batch_size = 100
        all_scores = []
        
        for i in range(0, len(sections), batch_size):
            batch_sections = sections[i:i + batch_size]
            
            documents = []
            for section in batch_sections:
                content = f"{section['section_title']} {section['content'][:1000]}"  
                processed_content = self.preprocess_text(content)
                documents.append(processed_content)
            
            if not documents:
                continue
            
            all_docs = documents + [query_processed]
            
            try:
                vectorizer = TfidfVectorizer(
                    max_features=500,  
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95,
                    stop_words='english'
                )
                tfidf_matrix = vectorizer.fit_transform(all_docs)
                
                query_vector = tfidf_matrix[-1]
                doc_vectors = tfidf_matrix[:-1]
                
                similarities = cosine_similarity(query_vector, doc_vectors).flatten()
                all_scores.extend(similarities)
                
                del tfidf_matrix, vectorizer, doc_vectors, query_vector
                gc.collect()
                
            except Exception as e:
                print(f"Error in batch {i//batch_size}: {str(e)}")
                all_scores.extend([0.1] * len(batch_sections))
        
        for i, section in enumerate(sections):
            if i < len(all_scores):
                section['relevance_score'] = float(all_scores[i])
            else:
                section['relevance_score'] = 0.0
        
        sections.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        for i, section in enumerate(sections):
            section['importance_rank'] = i + 1
        
        return sections
    
    def extract_subsections(self, sections: List[Dict], top_k: int = 10) -> List[Dict]:
        """Enhanced subsection extraction with better text summarization"""
        subsections = []
        
        for section in sections[:top_k]:
            content = section['content']
            
            if not content.strip():
                continue
            
            try:
                sentences = sent_tokenize(content)
            except:
                sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 10]
            
            if not sentences:
                continue
            
            key_sentences = []
            
            if sentences:
                key_sentences.append(sentences[0])
            
            title_words = set(section['section_title'].lower().split())
            scored_sentences = []
            
            for i, sent in enumerate(sentences[1:6]):  
                sent_words = set(sent.lower().split())
                overlap = len(title_words.intersection(sent_words))
                scored_sentences.append((i + 1, sent, overlap, len(sent.split())))
            
            scored_sentences.sort(key=lambda x: (x[2], x[3]), reverse=True)
            
            for _, sent, _, _ in scored_sentences[:2]:
                if sent not in key_sentences and len(sent.strip()) > 20:
                    key_sentences.append(sent)
            
            remaining_sentences = [s for s in sentences if s not in key_sentences]
            remaining_sentences.sort(key=lambda x: len(x.split()), reverse=True)
            
            for sent in remaining_sentences[:2]:
                if len(key_sentences) < 4 and len(sent.strip()) > 30:
                    key_sentences.append(sent)
            
            refined_sentences = key_sentences[:3]
            refined_text = ' '.join(refined_sentences)
            
            if len(refined_text) > 300:
                refined_text = refined_text[:297] + "..."
            
            if len(refined_text.strip()) < 50:
                refined_text = content[:200] + "..." if len(content) > 200 else content
            
            subsections.append({
                "document": section["document"],
                "page_number": section["page_number"],
                "section_title": section["section_title"],
                "importance_rank": section["importance_rank"],
                "refined_text": refined_text.strip()
            })
        
        return subsections

def load_json_input(json_path: str) -> Dict:
    """Load and parse JSON input file"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading JSON file {json_path}: {str(e)}")
        return None

def extract_pdf_paths(json_data: Dict, base_path: str = None) -> List[Path]:
    """Extract PDF file paths from JSON data"""
    pdf_files = []
    
    if 'documents' not in json_data:
        print("Warning: 'documents' key not found in JSON input")
        return pdf_files
    
    base_dir = Path(base_path) if base_path else Path.cwd()
    
    for doc in json_data['documents']:
        if 'filename' in doc:
            filename = doc['filename']
            if filename.lower().endswith('.pdf'):
                pdf_path = base_dir / filename
                if pdf_path.exists():
                    pdf_files.append(pdf_path)
                else:
                    print(f"Warning: PDF file not found: {pdf_path}")
    
    return pdf_files

def extract_persona_and_job(json_data: Dict) -> Tuple[str, str]:
    """Extract persona and job information from JSON data"""
    persona = ""
    job = ""
    
    # Extract persona
    if 'persona' in json_data:
        if isinstance(json_data['persona'], dict):
            persona = json_data['persona'].get('role', '')
        else:
            persona = str(json_data['persona'])
    
    # Extract job to be done
    if 'job_to_be_done' in json_data:
        if isinstance(json_data['job_to_be_done'], dict):
            job = json_data['job_to_be_done'].get('task', '')
        else:
            job = str(json_data['job_to_be_done'])
    
    return persona, job

def main():
    parser = argparse.ArgumentParser(description='Enhanced Persona-Driven Document Intelligence - JSON Input')
    parser.add_argument('--input', required=True, help='Input JSON file containing document list and persona information')
    parser.add_argument('--pdf_dir', help='Directory containing PDF files (default: same directory as JSON file)')
    parser.add_argument('--output', default='output.json', help='Output JSON file')
    parser.add_argument('--top_k', type=int, default=15, help='Number of top sections to analyze')
    parser.add_argument('--chunk_size', type=int, default=25, help='Pages per chunk for memory management')
    
    args = parser.parse_args()
    
    print("=== Enhanced Round 1B: Persona-Driven Document Intelligence (JSON Input) ===")
    
    # Load JSON input
    json_data = load_json_input(args.input)
    if not json_data:
        print("Failed to load JSON input file. Exiting.")
        return
    
    # Extract persona and job information
    persona, job = extract_persona_and_job(json_data)
    
    if not persona or not job:
        print("Warning: Persona or job information missing from JSON input")
        print(f"Persona: {persona}")
        print(f"Job: {job}")
    
    print(f"Persona: {persona}")
    print(f"Job: {job}")
    print(f"Processing chunks of {args.chunk_size} pages")
    
    # Determine PDF directory
    json_path = Path(args.input)
    pdf_base_dir = args.pdf_dir if args.pdf_dir else json_path.parent
    
    # Extract PDF file paths
    pdf_files = extract_pdf_paths(json_data, pdf_base_dir)
    
    if not pdf_files:
        print(f"No PDF files found. Check that:")
        print(f"1. The JSON file contains a 'documents' array with PDF filenames")
        print(f"2. PDF files exist in: {pdf_base_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s) to process...")
    
    processor = MemoryEfficientDocumentProcessor(chunk_size=args.chunk_size)
    
    all_sections = []
    processed_documents = []
    
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        section_count = 0
        
        try:
            for section in processor.extract_sections_generator(str(pdf_file)):
                all_sections.append(section)
                section_count += 1
                
                if section_count % 50 == 0:
                    print(f"  Extracted {section_count} sections...")
            
            processed_documents.append(pdf_file.name)
            print(f"  Completed: {section_count} sections extracted")
            
        except Exception as e:
            print(f"  Error processing {pdf_file.name}: {str(e)}")
            continue
    
    print(f"\nTotal sections extracted: {len(all_sections)}")
    
    if not all_sections:
        print("No sections extracted. Exiting.")
        return
    
    print("\nCalculating relevance scores...")
    ranked_sections = processor.calculate_relevance_scores_batch(all_sections, persona, job)
    
    print(f"\nExtracting top {args.top_k} subsections...")
    subsections = processor.extract_subsections(ranked_sections, args.top_k)
    
    # Create output with same format as original
    output = {
        "metadata": {
            "input_documents": sorted(processed_documents),
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": [
            {
                "document": section["document"],
                "section_title": section["section_title"],
                "importance_rank": section["importance_rank"],
                "page_number": section["page_number"]
            }
            for section in ranked_sections[:args.top_k] 
        ],
        "subsection_analysis": [
            {
                "document": section["document"],
                "refined_text": section["refined_text"],
                "page_number": section["page_number"]
            }
            for section in subsections
        ]
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== Results ===")
    print(f"Results written to: {args.output}")
    print(f"Documents processed: {len(processed_documents)}")
    print(f"Total sections: {len(all_sections)}")
    print(f"Top sections analyzed: {len(subsections)}")
    
    print(f"\nTop 10 most relevant sections:")
    for i, section in enumerate(ranked_sections[:10]):
        score = section.get('relevance_score', 0.0)
        print(f"{i+1:2d}. [{section['document']}] {section['section_title'][:60]}... (Score: {score:.3f})")
    
    print("\nProcessing completed successfully!")

if __name__ == "__main__":
    main()