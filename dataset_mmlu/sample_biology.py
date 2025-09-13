import json
import random
from pathlib import Path

def sample_questions(input_file: str, output_file: str, subject_name: str, sample_size: int = 100):
    """Sample random questions from dataset"""
    
    with open(input_file, 'r', encoding='utf-8') as f:
        all_questions = json.load(f)
    
    print(f"Loaded {len(all_questions)} total {subject_name} questions")
    
    if len(all_questions) < sample_size:
        print(f"Warning: Only {len(all_questions)} questions available, using all")
        sampled_questions = all_questions
    else:
        random.seed(42)  # For reproducible sampling
        sampled_questions = random.sample(all_questions, sample_size)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sampled_questions, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(sampled_questions)} {subject_name} questions to {output_file}")

def sample_biology_questions(sample_size: int = 100):
    input_file = "mmlu_pro_sections/biology_json"
    output_file = "mmlu_pro_sections/biology_sample_100.json"
    sample_questions(input_file, output_file, "biology", sample_size)

def sample_law_questions(sample_size: int = 100):
    input_file = "mmlu_pro_sections/law_json"
    output_file = "mmlu_pro_sections/law_sample_100.json"
    sample_questions(input_file, output_file, "law", sample_size)

if __name__ == "__main__":
    # Sample biology questions
    sample_biology_questions(100)
  
    # Sample law questions  
    sample_law_questions(100)

