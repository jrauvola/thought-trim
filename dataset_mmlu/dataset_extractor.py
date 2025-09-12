import json
import argparse
from pathlib import Path
from typing import Dict, List, Set
import pandas as pd
from datasets import load_dataset

class MMLUDatasetExtractor:
    AVAILABLE_SUBJECTS = {
        "biology": ["biology"],
        "chemistry": ["chemistry"], 
        "health": ["health"],
        "law": ["law"],
        "physics": ["physics"],
        "math": ["math"],
        "computer_science": ["computer_science"],
        "engineering": ["engineering"],
        "economics": ["economics"],
        "psychology": ["psychology"]
    }
    
    def __init__(self, dataset_name: str = "TIGER-Lab/MMLU-Pro"):
        self.dataset_name = dataset_name
        self.dataset = None
        
    def load_dataset(self) -> None:
        print(f"Loading {self.dataset_name}")
        self.dataset = load_dataset(self.dataset_name)
        print(f"Loaded {len(self.dataset['test'])} questions")
    
    def get_available_categories(self) -> Set[str]:
        if not self.dataset:
            raise RuntimeError("Dataset not loaded")
        categories = set()
        for split_name, split_data in self.dataset.items():
            for item in split_data:
                if 'category' in item:
                    categories.add(item['category'])
        return categories
    
    def extract_subjects(self, subjects: List[str], split: str = 'test') -> Dict[str, List[Dict]]:
        if not self.dataset:
            raise RuntimeError("Dataset not loaded")
        if split not in self.dataset:
            raise ValueError(f"Split '{split}' not found")
        
        unknown_subjects = set(subjects) - set(self.AVAILABLE_SUBJECTS.keys())
        if unknown_subjects:
            raise ValueError(f"Unknown subjects: {unknown_subjects}")
        
        target_categories = set()
        for subject in subjects:
            target_categories.update(self.AVAILABLE_SUBJECTS[subject])
        
        print(f"Extracting: {', '.join(subjects)}")
        
        extracted_data = {subject: [] for subject in subjects}
        split_data = self.dataset[split]
        
        for item in split_data:
            category = item.get('category', '')
            for subject, categories in self.AVAILABLE_SUBJECTS.items():
                if subject in subjects and category in categories:
                    question_data = {
                        'question_id': item.get('question_id', len(extracted_data[subject])),
                        'question': item.get('question', ''),
                        'options': item.get('options', []),
                        'answer': item.get('answer', ''),
                        'answer_index': item.get('answer_index', -1),
                        'category': category,
                        'src': item.get('src', ''),
                        'pred': item.get('pred', ''),
                        'cot_content': item.get('cot_content', '')
                    }
                    extracted_data[subject].append(question_data)
                    break
        
        total_extracted = sum(len(questions) for questions in extracted_data.values())
        print(f"Extracted {total_extracted} questions")
        
        for subject, questions in extracted_data.items():
            print(f"   {subject}: {len(questions)}")
        
        return extracted_data
    
    def save_data(self, data: Dict[str, List[Dict]], output_dir: str = "./mmlu_pro_sections", 
                  formats: List[str] = ["json"]) -> Dict[str, List[str]]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        saved_files = {format_type: [] for format_type in formats}
        
        for subject, questions in data.items():
            if not questions:
                continue
                
            for format_type in formats:
                if format_type == "json":
                    file_path = output_path / f"{subject}_json"
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(questions, f, indent=2, ensure_ascii=False)
                    saved_files["json"].append(str(file_path))
                elif format_type == "csv":
                    file_path = output_path / f"{subject}.csv"
                    pd.DataFrame(questions).to_csv(file_path, index=False)
                    saved_files["csv"].append(str(file_path))
                elif format_type == "parquet":
                    file_path = output_path / f"{subject}.parquet"
                    pd.DataFrame(questions).to_parquet(file_path, index=False)
                    saved_files["parquet"].append(str(file_path))
        
        return saved_files
    
    def extract_and_save(self, subjects: List[str], output_dir: str = "./mmlu_pro_sections",
                        formats: List[str] = ["json"], split: str = 'test') -> Dict[str, List[str]]:
        if not self.dataset:
            self.load_dataset()
        extracted_data = self.extract_subjects(subjects, split)
        saved_files = self.save_data(extracted_data, output_dir, formats)
        
        print(f"\nSaved to: {output_dir}")
        for format_type, files in saved_files.items():
            if files:
                print(f"   {format_type}: {len(files)} files")
        
        return saved_files
    
    def print_sample(self, subject: str, num_samples: int = 3) -> None:
        if not self.dataset:
            self.load_dataset()
            
        extracted_data = self.extract_subjects([subject])
        questions = extracted_data.get(subject, [])
        
        if not questions:
            print(f"No questions for {subject}")
            return
        
        print(f"\n{subject} samples ({min(num_samples, len(questions))} of {len(questions)}):")
        
        for i, q in enumerate(questions[:num_samples], 1):
            print(f"\nQ{i}: {q['question']}")
            for j, option in enumerate(q['options']):
                letter = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'][j]
                marker = "✓" if letter == q['answer'] else " "
                print(f"   {marker} {letter}) {option}")
            print(f"Answer: {q['answer']}")

def main():
    parser = argparse.ArgumentParser(description="MMLU Pro Dataset Extractor")
    
    parser.add_argument("--subjects", nargs="+", help="Subjects to extract")
    parser.add_argument("--output_dir", default="./mmlu_pro_sections", help="Output directory")
    parser.add_argument("--formats", nargs="+", default=["json"], choices=["json", "csv", "parquet"], help="Output formats")
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--list_categories", action="store_true", help="List categories")
    parser.add_argument("--sample", type=str, help="Show samples for subject")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of samples")
    
    args = parser.parse_args()
    
    extractor = MMLUDatasetExtractor()
    
    try:
        if args.list_categories:
            extractor.load_dataset()
            categories = extractor.get_available_categories()
            print("Available categories:")
            for category in sorted(categories):
                print(f"  {category}")
            return 0
        
        if args.sample:
            extractor.print_sample(args.sample, args.num_samples)
            return 0
        
        if not args.subjects:
            print("ERROR: Specify subjects with --subjects")
            return 1
        
        saved_files = extractor.extract_and_save(args.subjects, args.output_dir, args.formats, args.split)
        print(f"\nComplete!")
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

if __name__ == "__main__":
    exit(main())