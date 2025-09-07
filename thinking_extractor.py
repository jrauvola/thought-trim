import re
from typing import List, Dict


class ThinkingExtractor:
    def __init__(self):
        self.thinking_patterns = [
            r'<thinking>(.*?)</thinking>',
            r'<think>(.*?)</think>',
            r'\*thinking\*(.*?)\*\/thinking\*',
            r'Let me think about this\.\.\.(.*?)(?=\n\n|\nNow|$)',
            r'I need to think through this\.\.\.(.*?)(?=\n\n|\nSo|$)',
            r'Thinking:(.*?)(?=\n\n|\nAnswer:|$)',
            r'My reasoning:(.*?)(?=\n\n|\nConclusion:|$)',
        ]

    def extract_thinking_sections(self, text: str) -> List[str]:
        thinking_sections = []
        for pattern in self.thinking_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                cleaned = re.sub(r'\n\s*\n', '\n\n', match.strip())
                if cleaned and len(cleaned.strip()) > 10:
                    thinking_sections.append(cleaned)
        return thinking_sections

    def extract_from_response(self, response_dict: Dict) -> List[str]:
        if "thinking" in response_dict:
          return [response_dict["thinking"]]

        if "error" in response_dict:
            return []
        response_text = response_dict.get("response", "")
        if not response_text:
            return []
        return self.extract_thinking_sections(response_text)
