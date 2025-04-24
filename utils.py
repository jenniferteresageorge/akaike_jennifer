import re
from typing import List, Dict, Tuple
from datetime import datetime

class PIIDetector:
    """
    Class for detecting and masking Personally Identifiable Information (PII) in text.
    Uses regular expressions and pattern matching to identify PII entities.
    """
    
    def __init__(self):
        # Compile regex patterns for different PII types
        self.patterns = {
            "full_name": re.compile(r'\b([A-Z][a-z]+(\s[A-Z][a-z]+)+)\b'),
            "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "phone_number": re.compile(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
            "dob": re.compile(r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4})\b'),
            "aadhar_num": re.compile(r'\b\d{4}[ -]?\d{4}[ -]?\d{4}\b'),
            "credit_debit_no": re.compile(r'\b(?:\d[ -]*?){13,16}\b'),
            "cvv_no": re.compile(r'\b\d{3,4}\b'),
            "expiry_no": re.compile(r'\b(0[1-9]|1[0-2])[-/]\d{2}\b')
        }
        
    def detect_pii(self, text: str) -> List[Dict]:
        """
        Detect all PII entities in the given text.
        
        Args:
            text: Input text to scan for PII
            
        Returns:
            List of dictionaries containing PII entities with their positions and types
        """
        entities = []
        
        for entity_type, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                start, end = match.span()
                entity_value = match.group()
                
                # Additional validation for specific entity types
                if entity_type == "credit_debit_no" and not self._validate_luhn(entity_value):
                    continue
                if entity_type == "dob" and not self._validate_date(entity_value):
                    continue
                
                entities.append({
                    "position": [start, end],
                    "classification": entity_type,
                    "entity": entity_value
                })
        
        # Sort entities by start position to handle masking in order
        entities.sort(key=lambda x: x["position"][0])
        return entities
    
    def mask_pii(self, text: str, entities: List[Dict]) -> Tuple[str, List[Dict]]:
        """
        Mask detected PII entities in the text.
        
        Args:
            text: Original text containing PII
            entities: List of detected PII entities
            
        Returns:
            Tuple of (masked_text, list_of_masked_entities)
        """
        masked_text = text
        offset = 0
        masked_entities = []
        
        for entity in entities:
            start, end = entity["position"]
            entity_type = entity["classification"]
            original_value = entity["entity"]
            
            # Adjust positions based on previous replacements
            adj_start = start + offset
            adj_end = end + offset
            
            # Create masked token
            masked_token = f"[{entity_type}]"
            
            # Replace the entity with masked token
            masked_text = masked_text[:adj_start] + masked_token + masked_text[adj_end:]
            
            # Update offset for next replacement
            offset += len(masked_token) - (end - start)
            
            # Store masked entity info
            masked_entities.append({
                "position": [start, end],
                "classification": entity_type,
                "entity": original_value
            })
        
        return masked_text, masked_entities
    
    def _validate_luhn(self, card_number: str) -> bool:
        """Validate credit card number using Luhn algorithm."""
        # Remove non-digit characters
        card_number = re.sub(r'[^0-9]', '', card_number)
        
        if not card_number.isdigit() or len(card_number) < 13 or len(card_number) > 19:
            return False
        
        digits = list(map(int, card_number))
        checksum = digits[-1]
        total = 0
        
        for i, digit in enumerate(digits[:-1]):
            if i % 2 == 0:
                digit *= 2
                if digit > 9:
                    digit -= 9
            total += digit
        
        return (total * 9) % 10 == checksum
    
    def _validate_date(self, date_str: str) -> bool:
        """Validate date of birth."""
        try:
            # Try to parse different date formats
            for fmt in ('%m/%d/%Y', '%m-%d-%Y', '%d/%m/%Y', '%d-%m-%Y', 
                        '%b %d, %Y', '%B %d, %Y'):
                try:
                    datetime.strptime(date_str, fmt)
                    return True
                except ValueError:
                    continue
            return False
        except:
            return False
