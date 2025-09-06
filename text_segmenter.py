import re
from typing import List
from enum import Enum


class SegmentationType(Enum):
    SENTENCE = "sentence"
    THOUGHT = "thought"


class TextSegmenter:
    def __init__(self):
        self.sentence_endings = r'[.!?]+(?:\s|$)'
        self.thought_indicators = [
            r'(?:First|Second|Third|Next|Then|Finally|Additionally|Moreover|Furthermore|However|But|Although|Therefore|Thus|So|In conclusion|To summarize)',
            r'(?:Let me|I need to|I should|We can|We need to|We should)',
            r'(?:Because|Since|Given that|Considering|If|When|While)',
            r'(?:This means|This suggests|This indicates|This shows)',
        ]
    
    def segment_by_sentence(self, text: str) -> List[str]:
        sentences = re.split(self.sentence_endings, text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]
    
    def segment_by_thought(self, text: str) -> List[str]:
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        thoughts = []
        for paragraph in paragraphs:
            if len(paragraph) > 200:
                pattern = '|'.join(f'({indicator})' for indicator in self.thought_indicators)
                parts = re.split(f'({pattern})', paragraph, flags=re.IGNORECASE)
                current_thought = ""
                for part in parts:
                    if part and part.strip():
                        if re.match(pattern, part, re.IGNORECASE):
                            if current_thought.strip():
                                thoughts.append(current_thought.strip())
                            current_thought = part
                        else:
                            current_thought += " " + part if current_thought else part
                if current_thought.strip():
                    thoughts.append(current_thought.strip())
            else:
                thoughts.append(paragraph)
        return [t for t in thoughts if len(t.strip()) > 10]
    
    def segment_text(self, text: str, segmentation_type: SegmentationType) -> List[str]:
        if segmentation_type == SegmentationType.SENTENCE:
            return self.segment_by_sentence(text)
        elif segmentation_type == SegmentationType.THOUGHT:
            return self.segment_by_thought(text)
        else:
            raise ValueError(f"Unknown segmentation type: {segmentation_type}")
    
    def segment_thinking_sections(self, thinking_sections: List[str], segmentation_type: SegmentationType) -> List[List[str]]:
        return [self.segment_text(section, segmentation_type) for section in thinking_sections]