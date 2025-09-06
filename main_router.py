from typing import Dict, List, Any
from ollama_client import OllamaClient
from thinking_extractor import ThinkingExtractor
from text_segmenter import TextSegmenter, SegmentationType


class ReasoningProcessor:
    def __init__(self, ollama_base_url: str = "http://localhost:11434"):
        self.ollama_client = OllamaClient(ollama_base_url)
        self.thinking_extractor = ThinkingExtractor()
        self.text_segmenter = TextSegmenter()
    
    def process_question(self, question: str, model: str, segmentation_type: str = "sentence") -> Dict[str, Any]:
        result = {
            "question": question,
            "model": model,
            "segmentation_type": segmentation_type,
            "success": False,
            "error": None,
            "raw_response": None,
            "thinking_sections": [],
            "segmented_thinking": [],
            "total_segments": 0
        }
        
        try:
            if segmentation_type.lower() not in ["sentence", "thought"]:
                raise ValueError(f"Invalid segmentation type: {segmentation_type}")
            
            seg_type = SegmentationType.SENTENCE if segmentation_type.lower() == "sentence" else SegmentationType.THOUGHT
            
            if not self.ollama_client.check_model_exists(model):
                raise ValueError(f"Model '{model}' not found")
            
            raw_response = self.ollama_client.generate_response(model, question)
            if "error" in raw_response:
                result["error"] = raw_response["error"]
                return result
            
            result["raw_response"] = raw_response
            thinking_sections = self.thinking_extractor.extract_from_response(raw_response)
            result["thinking_sections"] = thinking_sections
            
            if not thinking_sections:
                result["success"] = True
                return result
            
            segmented_thinking = self.text_segmenter.segment_thinking_sections(thinking_sections, seg_type)
            result["segmented_thinking"] = segmented_thinking
            result["total_segments"] = sum(len(segments) for segments in segmented_thinking)
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def get_available_models(self) -> List[str]:
        models_response = self.ollama_client.list_models()
        if "error" in models_response:
            return []
        return [model["name"] for model in models_response.get("models", [])]


if __name__ == "__main__":
    processor = ReasoningProcessor()
    models = processor.get_available_models()
    if models:
        result = processor.process_question("Why is the sky blue?", models[0])
        print(f"Success: {result['success']}")
        print(f"Thinking sections: {len(result['thinking_sections'])}")
        print(f"Total segments: {result['total_segments']}")
    else:
        print("No models available")