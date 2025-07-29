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
    print("=== Enhanced Round 1B: Persona-Driven Document Intelligence ===")
    print()

    # FIXED PATHS
    input_dir = Path("app/input")
    output_dir = Path("app/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "output.json"

    # Find JSON input file in app/input
    input_json_files = list(input_dir.glob("*.json"))
    if not input_json_files:
        print("No JSON input file found in app/input directory.")
        return
    
    if len(input_json_files) > 1:
        print(f"Multiple JSON files found, using the first: {input_json_files[0].name}")

    json_input_path = input_json_files[0]
    print(f"Using input JSON: {json_input_path}")

    # Load JSON input
    json_data = load_json_input(str(json_input_path))
    if not json_data:
        print("Failed to load JSON input file. Exiting.")
        return

    # Extract persona and job information from JSON
    persona, job = extract_persona_and_job(json_data)
    if not persona or not job:
        print("Error: Persona or job information missing from JSON input")
        return

    print(f"✓ Persona: {persona}")
    print(f"✓ Job to be Done: {job}")
    print()

    chunk_size = 25
    print(f"Processing PDF pages in chunks of {chunk_size} pages")

    # Look for PDF files based on JSON document list
    pdf_files = extract_pdf_paths(json_data, input_dir)
    if not pdf_files:
        print(f"No PDF files found in {input_dir} as specified in JSON.")
        return

    print(f"Found {len(pdf_files)} PDF file(s) to process...")

    processor = MemoryEfficientDocumentProcessor(chunk_size=chunk_size)

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

    top_k = 15
    print(f"\nExtracting top {top_k} subsections...")
    subsections = processor.extract_subsections(ranked_sections, top_k)

    # Output file will be app/output/output.json
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
            for section in ranked_sections[:top_k] 
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

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n=== Results ===")
    print(f"Results written to: {output_path}")
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
