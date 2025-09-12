import sys
import os
import json
import random
import re
import time
import argparse
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# Add parent directory to path for ollama_client
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ollama_client import OllamaClient

@dataclass
class ModelConfig:
    name: str
    type: str
    options: Dict[str, Any]
    description: str = ""

class ModelEvaluator:
    MODELS = {
        "qwen2.5:1.5b": ModelConfig(
            name="qwen2.5:1.5b",
            type="non-reasoning", 
            options={},
            description="Qwen 2.5 1.5B Base Model"
        ),
        "qwen2.5:1.5b-instruct": ModelConfig(
            name="qwen2.5:1.5b-instruct",
            type="non-reasoning",
            options={},
            description="Qwen 2.5 1.5B Instruct Model"
        ),
        "qwen3:1.7b": ModelConfig(
            name="qwen3:1.7b", 
            type="reasoning",
            options={"think": True},
            description="Qwen 3 1.7B Reasoning Model"
        ),
        "yasserrmd/OpenReasoning-Nemotron:1.5b": ModelConfig(
            name="yasserrmd/OpenReasoning-Nemotron:1.5b",
            type="reasoning",
            options={},
            description="OpenReasoning Nemotron 1.5B"
        ),
        "deepseek-r1:1.5b": ModelConfig(
            name="deepseek-r1:1.5b",
            type="reasoning", 
            options={},
            description="DeepSeek R1 1.5B Reasoning Model"
        )
    }
    
    def __init__(self, data_path: str = "./mmlu_pro_sections/biology_json"):
        self.client = OllamaClient()
        self.data_path = Path(data_path)
        self.stats = {"json_success": 0, "json_failed": 0, "extraction_methods": {}, "retry_success": 0}
        
    def load_questions(self, sample_size: Optional[int] = None) -> List[Dict]:
        with open(self.data_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        if sample_size and sample_size < len(questions):
            random.shuffle(questions)
            questions = questions[:sample_size]
        return questions
    
    def create_dynamic_schema(self, question_data: Dict, model_type: str) -> Dict:
        num_options = len(question_data.get('options', []))
        valid_answers = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'][:num_options]
        
        if model_type == "reasoning":
            return {
                "type": "object",
                "properties": {
                    "thinking": {"type": "string"},
                    "answer": {"type": "string", "enum": valid_answers},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["thinking", "answer"],
                "additionalProperties": False
            }
        else:
            return {
                "type": "object", 
                "properties": {"answer": {"type": "string", "enum": valid_answers}},
                "required": ["answer"],
                "additionalProperties": False
            }
    
    def create_prompt(self, question_data: Dict, question_id: int, schema: Dict, model_type: str) -> str:
        question = question_data['question']
        options = question_data['options']
        
        option_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        formatted_options = []
        valid_answers = []
        
        for i, option in enumerate(options[:len(option_labels)]):
            formatted_options.append(f"{option_labels[i]}) {option}")
            valid_answers.append(option_labels[i])
        
        if model_type == "reasoning":
            return f"""CRITICAL: Respond with JSON matching this schema EXACTLY.

Schema: {json.dumps(schema)}

Question {question_id}: {question}

Options:
{chr(10).join(formatted_options)}

Return ONLY valid JSON: {{"thinking": "...", "answer": "{valid_answers[0]}"}}"""
        else:
            return f"""CRITICAL: Return ONLY JSON.

Question: {question}

Options:
{chr(10).join(formatted_options)}

Return JSON: {{"answer": "X"}} where X is one of: {', '.join(valid_answers)}"""
    
    def extract_answer_robust(self, text: str, valid_options: List[str]) -> Tuple[str, str]:
        # JSON parsing
        try:
            data = json.loads(text)
            if "answer" in data:
                answer = str(data["answer"]).upper()
                if answer in valid_options:
                    self.stats["extraction_methods"]["json_valid"] = self.stats["extraction_methods"].get("json_valid", 0) + 1
                    return answer, "json_valid"
        except:
            pass
        
        # JSON patterns
        for pattern in [r'"answer"\s*:\s*"([A-J])"', r'answer["\s]*:\s*["\s]*([A-J])["\s]*']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and match.group(1).upper() in valid_options:
                self.stats["extraction_methods"]["json_pattern"] = self.stats["extraction_methods"].get("json_pattern", 0) + 1
                return match.group(1).upper(), "json_pattern"
        
        # Answer statements
        for pattern in [r'(?:final|my|the)\s+answer\s+is\s+([A-J])', r'answer:\s*([A-J])', r'choose\s+([A-J])']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                for match in reversed(matches):
                    if match.upper() in valid_options:
                        self.stats["extraction_methods"]["answer_statement"] = self.stats["extraction_methods"].get("answer_statement", 0) + 1
                        return match.upper(), "answer_statement"
        
        # Standalone letters
        for option in valid_options:
            if re.search(rf'\b{option}\b', text):
                self.stats["extraction_methods"]["standalone_letter"] = self.stats["extraction_methods"].get("standalone_letter", 0) + 1
                return option, "standalone_letter"
        
        # First valid letter
        for char in text:
            if char.upper() in valid_options:
                self.stats["extraction_methods"]["first_letter"] = self.stats["extraction_methods"].get("first_letter", 0) + 1
                return char.upper(), "first_letter"
        
        return "UNKNOWN", "failed"
    
    def query_with_retries(self, model_config: ModelConfig, prompt: str, schema: Dict, max_retries: int = 3) -> Dict:
        for attempt in range(max_retries):
            try:
                eval_options = {
                    "temperature": 0.3 if attempt == 0 else 0.5,
                    "top_p": 1.0,
                    "num_predict": 500,
                    "format": schema
                }
                eval_options.update(model_config.options)
                
                response = self.client.generate_response(model_config.name, prompt, eval_options)
                
                if "error" not in response:
                    if attempt > 0:
                        self.stats["retry_success"] += 1
                    return {"success": True, "response": response.get("response", ""), "attempt": attempt + 1}
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    return {"success": False, "error": str(e), "attempt": attempt + 1}
                time.sleep(0.5)
        
        return {"success": False, "error": "Max retries exceeded"}
    
    def evaluate_model(self, model_name: str, questions: List[Dict], verbose: bool = True) -> Dict:
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_config = self.MODELS[model_name]
        
        if not self.client.check_model_exists(model_config.name):
            raise RuntimeError(f"Model {model_config.name} not found")
        
        if verbose:
            print(f"\n{model_config.name} ({model_config.type})")
            print("-" * 50)
        
        results = {
            "model_name": model_name,
            "model_type": model_config.type, 
            "total_questions": len(questions),
            "correct": 0,
            "incorrect": 0,
            "errors": 0,
            "responses": [],
            "json_success": 0,
            "extraction_methods": {}
        }
        
        for i, question_data in enumerate(questions, 1):
            # Create dynamic schema for this question
            schema = self.create_dynamic_schema(question_data, model_config.type)
            
            # Create explicit prompt
            prompt = self.create_prompt(question_data, i, schema, model_config.type)
            
            # Get valid answers
            num_options = len(question_data['options'])
            valid_answers = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'][:num_options]
            
            # Query with retries
            result = self.query_with_retries(model_config, prompt, schema)
            
            if not result["success"]:
                if verbose:
                    print(f"Q{i:3d}: ❌ API ERROR")
                results["errors"] += 1
                continue
            
            raw_output = result["response"]
            
            # Extract answer
            predicted_answer, extraction_method = self.extract_answer_robust(raw_output, valid_answers)
            
            # Track extraction method
            results["extraction_methods"][extraction_method] = results["extraction_methods"].get(extraction_method, 0) + 1
            
            # Track JSON success
            if extraction_method == "json_valid":
                results["json_success"] += 1
                self.stats["json_success"] += 1
            else:
                self.stats["json_failed"] += 1
            
            # Check correctness
            correct_answer = question_data["answer"].upper()
            is_correct = predicted_answer == correct_answer
            
            if is_correct:
                results["correct"] += 1
                if verbose:
                    print(f"Q{i:3d}: CORRECT {predicted_answer}")
            else:
                results["incorrect"] += 1
                if verbose:
                    print(f"Q{i:3d}: WRONG {predicted_answer} vs {correct_answer}")
            
            # Store response data
            results["responses"].append({
                "question_id": i,
                "question": question_data["question"][:100] + "..." if len(question_data["question"]) > 100 else question_data["question"],
                "correct_answer": correct_answer,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "extraction_method": extraction_method,
                "attempt": result.get("attempt", 1)
            })
        
        # Calculate final metrics
        total_answered = results["correct"] + results["incorrect"]
        if total_answered > 0:
            results["accuracy"] = results["correct"] / total_answered
            results["json_success_rate"] = results["json_success"] / total_answered
        else:
            results["accuracy"] = 0.0
            results["json_success_rate"] = 0.0
        
        if verbose:
            print(f"\nResults: {results['correct']}/{total_answered} = {results['accuracy']:.1%}")
        
        return results
    
    def evaluate_models(self, model_names: List[str], sample_size: Optional[int] = None, 
                       verbose: bool = True) -> Dict[str, Any]:
        questions = self.load_questions(sample_size)
        if verbose:
            print(f"Loaded {len(questions)} questions")
        
        results = {
            "evaluation_summary": {
                "total_questions": len(questions),
                "models_evaluated": len(model_names),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "model_results": {}
        }
        
        for model_name in model_names:
            try:
                model_result = self.evaluate_model(model_name, questions, verbose)
                results["model_results"][model_name] = model_result
            except Exception as e:
                if verbose:
                    print(f"ERROR {model_name}: {e}")
                results["model_results"][model_name] = {"error": str(e)}
        
        # Add global statistics
        results["global_stats"] = {
            "json_success_rate": self.stats["json_success"] / (self.stats["json_success"] + self.stats["json_failed"]) if (self.stats["json_success"] + self.stats["json_failed"]) > 0 else 0,
            "retry_successes": self.stats["retry_success"],
            "extraction_methods": self.stats["extraction_methods"]
        }
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_dir: str = "./results") -> Tuple[str, str]:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # JSON results
        json_file = output_path / f"evaluation_detailed_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Summary
        txt_file = output_path / f"evaluation_summary_{timestamp}.txt"
        with open(txt_file, 'w') as f:
            f.write("MMLU BIOLOGY MODEL EVALUATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            summary = results["evaluation_summary"]
            f.write(f"Date: {summary['timestamp']}\n")
            f.write(f"Questions: {summary['total_questions']}\n")
            f.write(f"Models: {summary['models_evaluated']}\n\n")
            
            f.write("ACCURACY RESULTS:\n")
            f.write("-" * 20 + "\n")
            
            model_results = results["model_results"]
            sorted_models = sorted(
                [(name, result.get("accuracy", 0)) for name, result in model_results.items() if "error" not in result],
                key=lambda x: x[1], reverse=True
            )
            
            for model_name, accuracy in sorted_models:
                result = model_results[model_name]
                f.write(f"{model_name}: {accuracy:.1%} ({result['correct']}/{result['correct'] + result['incorrect']})\n")
            
            global_stats = results["global_stats"]
            f.write(f"\nJSON Success Rate: {global_stats['json_success_rate']:.1%}\n")
            f.write(f"Retries: {global_stats['retry_successes']}\n")
            
            f.write(f"\nExtraction Methods:\n")
            for method, count in global_stats['extraction_methods'].items():
                f.write(f"  {method}: {count}\n")
        
        return str(json_file), str(txt_file)

def main():
    parser = argparse.ArgumentParser(description="MMLU Model Evaluator")
    parser.add_argument("--models", nargs="+", default=["qwen3:1.7b"], help="Models to evaluate")
    parser.add_argument("--sample_size", type=int, help="Number of questions")
    parser.add_argument("--output_dir", default="./results", help="Output directory")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode")
    parser.add_argument("--list_models", action="store_true", help="List models")
    
    args = parser.parse_args()
    evaluator = ModelEvaluator()
    
    if args.list_models:
        print("Available models:")
        for name, config in evaluator.MODELS.items():
            print(f"  {name} - {config.description} ({config.type})")
        return
    
    try:
        results = evaluator.evaluate_models(args.models, args.sample_size, not args.quiet)
        json_file, txt_file = evaluator.save_results(results, args.output_dir)
        
        print(f"\nComplete!")
        print(f"Detailed: {json_file}")  
        print(f"Summary: {txt_file}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
