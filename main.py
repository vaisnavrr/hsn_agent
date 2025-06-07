"""
HSN Code Validation and Suggestion Agent
Built with Google's Agent Development Kit (ADK)
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from pathlib import Path

# ADK Imports
from google.adk.agents import LlmAgent, BaseAgent
from google.adk.tools.base_tool import BaseTool
from google.adk.sessions import Session, InMemorySessionService

class HSNDataManager:
    """Handles HSN master data loading and management"""
    
    def __init__(self, excel_path: str = "HSN_SAC.xlsx"):
        self.excel_path = excel_path
        self.hsn_lookup = {}
        self.hsn_trie = {}
        self.description_index = {}
        self.length_groups = {2: [], 4: [], 6: [], 8: []}
        self.tfidf_vectorizer = None
        self.description_vectors = None
        self.descriptions_list = []
        self.load_data()
    
    def load_data(self):
        """Load and preprocess HSN master data"""
        try:
            # Load Excel file
            df = pd.read_excel(self.excel_path)
            
            # Clean and validate data
            df['HSNCode'] = df['HSNCode'].astype(str).str.strip()
            df['Description'] = df['Description'].astype(str).str.strip()
            
            # Remove invalid entries
            df = df[df['HSNCode'].str.isdigit()]
            df = df[df['HSNCode'].str.len().isin([2, 4, 6, 8])]
            
            # Build lookup structures
            for _, row in df.iterrows():
                hsn_code = row['HSNCode']
                description = row['Description']
                
                self.hsn_lookup[hsn_code] = description
                self.length_groups[len(hsn_code)].append(hsn_code)
                self.descriptions_list.append(description)
            
            # Build hierarchical trie
            self._build_trie()
            
            # Build TF-IDF vectors for suggestions
            self._build_tfidf_index()
            
            print(f"Loaded {len(self.hsn_lookup)} HSN codes successfully")
            
        except Exception as e:
            print(f"Error loading HSN data: {e}")
            raise
    
    def _build_trie(self):
        """Build trie structure for hierarchical validation"""
        for hsn_code in self.hsn_lookup:
            current = self.hsn_trie
            
            # Build path for each level (2, 4, 6, 8 digits)
            for length in [2, 4, 6, 8]:
                if len(hsn_code) >= length:
                    prefix = hsn_code[:length]
                    if prefix not in current:
                        current[prefix] = {}
                    current = current[prefix]
    
    def _build_tfidf_index(self):
        """Build TF-IDF index for semantic search"""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        
        # Preprocess descriptions
        cleaned_descriptions = [
            self._clean_text(desc) for desc in self.descriptions_list
        ]
        
        self.description_vectors = self.tfidf_vectorizer.fit_transform(cleaned_descriptions)
    
    def _clean_text(self, text: str) -> str:
        """Clean text for better matching"""
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


class HSNDataHandler(BaseTool):
    """Tool for handling HSN data operations"""
    
    name = "hsn_data_handler"
    description = "Handles HSN master data operations including lookup and search"
    
    def __init__(self, data_manager: HSNDataManager):
        super().__init__(
            name="hsn_data_handler",
            description="Performs HSN code lookup, search, and hierarchical operations"
        )
        self.data_manager = data_manager
    
    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        """Execute data handling operations"""
        if action == "lookup":
            return self._lookup_code(kwargs.get('code', ''))
        elif action == "search":
            return self._search_descriptions(kwargs.get('query', ''))
        elif action == "get_hierarchy":
            return self._get_hierarchy(kwargs.get('code', ''))
        else:
            return {"error": f"Unknown action: {action}"}
    
    def _lookup_code(self, hsn_code: str) -> Dict[str, Any]:
        """Look up HSN code in master data"""
        hsn_code = str(hsn_code).strip()
        
        if hsn_code in self.data_manager.hsn_lookup:
            return {
                "found": True,
                "code": hsn_code,
                "description": self.data_manager.hsn_lookup[hsn_code]
            }
        else:
            return {"found": False, "code": hsn_code}
    
    def _search_descriptions(self, query: str) -> List[Dict[str, Any]]:
        """Search descriptions for matching HSN codes"""
        if not query.strip():
            return []
        
        # Clean query
        cleaned_query = self.data_manager._clean_text(query)
        
        # Transform query to TF-IDF vector
        query_vector = self.data_manager.tfidf_vectorizer.transform([cleaned_query])
        
        # Calculate similarity scores
        similarity_scores = cosine_similarity(
            query_vector, 
            self.data_manager.description_vectors
        ).flatten()
        
        # Get top matches
        top_indices = similarity_scores.argsort()[-10:][::-1]
        
        results = []
        for idx in top_indices:
            if similarity_scores[idx] > 0.1:  # Minimum similarity threshold
                hsn_code = list(self.data_manager.hsn_lookup.keys())[idx]
                description = self.data_manager.descriptions_list[idx]
                
                results.append({
                    "hsn_code": hsn_code,
                    "description": description,
                    "confidence": float(similarity_scores[idx])
                })
        
        return results[:5]  # Return top 5 matches
    
    def _get_hierarchy(self, hsn_code: str) -> Dict[str, str]:
        """Get hierarchical breakdown of HSN code"""
        hsn_code = str(hsn_code).strip()
        hierarchy = {}
        
        for length in [2, 4, 6, 8]:
            if len(hsn_code) >= length:
                prefix = hsn_code[:length]
                if prefix in self.data_manager.hsn_lookup:
                    hierarchy[prefix] = self.data_manager.hsn_lookup[prefix]
        
        return hierarchy


class ValidationEngine(BaseTool):
    """Tool for HSN code validation"""
    
    def __init__(self, data_manager: HSNDataManager):
        super().__init__(
            name="validation_engine",
            description="Validates HSN codes using multiple validation rules"
        )
        self.data_manager = data_manager
    
    def execute(self, hsn_code: str) -> Dict[str, Any]:
        """Execute comprehensive HSN code validation"""
        hsn_code = str(hsn_code).strip()
        
        result = {
            "hsn_code": hsn_code,
            "format_validation": self._validate_format(hsn_code),
            "existence_validation": self._validate_existence(hsn_code),
            "hierarchy_validation": self._validate_hierarchy(hsn_code),
            "overall_status": "invalid"
        }
        
        # Determine overall status
        if (result["format_validation"]["valid"] and 
            result["existence_validation"]["valid"] and 
            result["hierarchy_validation"]["valid"]):
            result["overall_status"] = "valid"
        
        return result
    
    def _validate_format(self, hsn_code: str) -> Dict[str, Any]:
        """Validate HSN code format"""
        validation = {
            "valid": False,
            "errors": []
        }
        
        # Check if numeric
        if not hsn_code.isdigit():
            validation["errors"].append("HSN code must contain only digits")
            return validation
        
        # Check length
        if len(hsn_code) not in [2, 4, 6, 8]:
            validation["errors"].append(f"Invalid length: {len(hsn_code)}. Must be 2, 4, 6, or 8 digits")
            return validation
        
        # Check for valid range (basic check)
        if not (hsn_code.startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))):
            validation["errors"].append("Invalid HSN code format")
            return validation
        
        validation["valid"] = True
        return validation
    
    def _validate_existence(self, hsn_code: str) -> Dict[str, Any]:
        """Validate if HSN code exists in master data"""
        validation = {
            "valid": False,
            "description": None
        }
        
        if hsn_code in self.data_manager.hsn_lookup:
            validation["valid"] = True
            validation["description"] = self.data_manager.hsn_lookup[hsn_code]
        
        return validation
    
    def _validate_hierarchy(self, hsn_code: str) -> Dict[str, Any]:
        """Validate hierarchical structure of HSN code"""
        validation = {
            "valid": True,
            "hierarchy": {},
            "missing_parents": []
        }
        
        # Check each level of hierarchy
        for length in [2, 4, 6]:
            if len(hsn_code) > length:
                parent_code = hsn_code[:length]
                if parent_code in self.data_manager.hsn_lookup:
                    validation["hierarchy"][parent_code] = self.data_manager.hsn_lookup[parent_code]
                else:
                    validation["missing_parents"].append(parent_code)
                    validation["valid"] = False
        
        return validation


class SuggestionEngine(BaseTool):
    """Tool for HSN code suggestions"""
    
    name = "suggestion_engine"
    description = "Suggests HSN codes based on product descriptions"
    
    def __init__(self, data_manager: HSNDataManager):
        super().__init__(
            name="suggestion_engine",
            description="Generates HSN code suggestions based on product descriptions"
        )
        self.data_manager = data_manager
    
    def execute(self, query: str, max_suggestions: int = 5) -> List[Dict[str, Any]]:
        """Generate HSN code suggestions for given product description"""
        if not query.strip():
            return []
        
        # Use data handler to search
        data_handler = HSNDataHandler(self.data_manager)
        results = data_handler._search_descriptions(query)
        
        # Enhance results with additional metadata
        enhanced_results = []
        for result in results[:max_suggestions]:
            enhanced_result = result.copy()
            enhanced_result["match_reason"] = self._generate_match_reason(query, result["description"])
            enhanced_result["category"] = self._get_category(result["hsn_code"])
            enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    def _generate_match_reason(self, query: str, description: str) -> str:
        """Generate explanation for why this HSN code was suggested"""
        query_words = set(query.lower().split())
        desc_words = set(description.lower().split())
        common_words = query_words.intersection(desc_words)
        
        if common_words:
            return f"Matched keywords: {', '.join(list(common_words)[:3])}"
        else:
            return "Semantic similarity match"
    
    def _get_category(self, hsn_code: str) -> str:
        """Get top-level category for HSN code"""
        if len(hsn_code) >= 2:
            category_code = hsn_code[:2]
            return self.data_manager.hsn_lookup.get(category_code, "Unknown category")
        return "Unknown category"
    

from pydantic import PrivateAttr

class HSNValidationAgent(LlmAgent):
    _data_manger: HSNDataManager = PrivateAttr()
    """Main HSN Validation and Suggestion Agent"""
    def __init__(self, excel_path: str = "HSN_SAC.xlsx"):
        # Initialize data manager
        data_manager = HSNDataManager(excel_path)

        tools = [
            HSNDataHandler(data_manager),
            ValidationEngine(data_manager),
            SuggestionEngine(data_manager)
        ]
        
        # Initialize parent LlmAgent
        super().__init__(
            name="hsn_validation_agent",
            model="gemini-pro",
            tools=tools
        )
        self._data_manager = data_manager
    
    def _get_agent_instructions(self) -> str:
        """Get comprehensive instructions for the agent"""
        return """
        You are an expert HSN (Harmonized System Nomenclature) Code validation and suggestion agent.
        
        Your capabilities include:
        1. Validating HSN codes for format, existence, and hierarchical correctness
        2. Suggesting appropriate HSN codes based on product descriptions
        3. Providing detailed explanations and hierarchical breakdowns
        4. Handling both single codes and batch processing
        
        When validating HSN codes:
        - Check format (must be 2, 4, 6, or 8 digits)
        - Verify existence in master database
        - Validate hierarchical structure
        - Provide clear status and error messages
        
        When suggesting HSN codes:
        - Analyze product descriptions semantically
        - Provide multiple relevant suggestions with confidence scores
        - Explain why each code was suggested
        - Include category information
        
        Always be helpful, accurate, and provide clear explanations.
        If you're unsure about something, ask for clarification.
        """
    
    async def validate_hsn_code(self, hsn_code: str) -> Dict[str, Any]:
        """Validate a single HSN code"""
        validation_tool = next(tool for tool in self.tools if tool.name == "validation_engine")
        data_tool = next(tool for tool in self.tools if tool.name == "hsn_data_handler")
        
        # Get validation results
        validation_result = validation_tool.execute(hsn_code)
        
        # Get hierarchy if valid
        hierarchy = {}
        if validation_result["overall_status"] == "valid":
            hierarchy = data_tool.execute("get_hierarchy", code=hsn_code)
        
        # Format response
        response = {
            "hsn_code": hsn_code,
            "status": validation_result["overall_status"],
            "validation_details": validation_result,
            "hierarchy": hierarchy
        }
        
        # Add suggestions for invalid codes
        if validation_result["overall_status"] == "invalid":
            suggestion_tool = next(tool for tool in self.tools if tool.name == "suggestion_engine")
            # Try to find similar codes
            partial_matches = []
            if validation_result["format_validation"]["valid"]:
                # Look for similar codes
                for length in [len(hsn_code) - 2, len(hsn_code) + 2]:
                    if 2 <= length <= 8:
                        similar_code = hsn_code[:length] if length < len(hsn_code) else hsn_code + "00"
                        lookup_result = data_tool.execute("lookup", code=similar_code)
                        if lookup_result["found"]:
                            partial_matches.append({
                                "hsn_code": similar_code,
                                "description": lookup_result["description"],
                                "confidence": 0.7
                            })
            
            response["suggestions"] = partial_matches[:3]
        
        return response
    
    async def suggest_hsn_codes(self, product_description: str) -> Dict[str, Any]:
        """Suggest HSN codes for a product description"""
        suggestion_tool = next(tool for tool in self.tools if tool.name == "suggestion_engine")
        
        suggestions = suggestion_tool.execute(product_description, max_suggestions=5)
        
        return {
            "query": product_description,
            "suggestions": suggestions,
            "total_found": len(suggestions)
        }
    
    async def batch_validate(self, hsn_codes: List[str]) -> List[Dict[str, Any]]:
        """Validate multiple HSN codes"""
        results = []
        for code in hsn_codes:
            result = await self.validate_hsn_code(code)
            results.append(result)
        
        return {
            "batch_results": results,
            "summary": {
                "total": len(results),
                "valid": sum(1 for r in results if r["status"] == "valid"),
                "invalid": sum(1 for r in results if r["status"] == "invalid")
            }
        }


# CLI Interface for testing
import asyncio
async def main():
    """Main function for testing the agent"""
    print("HSN Code Validation Agent - Testing Interface")
    print("=" * 50)
    
    try:
        # Initialize agent
        agent = HSNValidationAgent()
        
        while True:
            print("\nOptions:")
            print("1. Validate HSN Code")
            print("2. Suggest HSN Codes")
            print("3. Batch Validate")
            print("4. Exit")
            
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                hsn_code = input("Enter HSN Code to validate: ").strip()
                result = await agent.validate_hsn_code(hsn_code)
                print(json.dumps(result, indent=2))
            
            elif choice == "2":
                description = input("Enter product description: ").strip()
                result = await agent.suggest_hsn_codes(description)
                print(json.dumps(result, indent=2))
            
            elif choice == "3":
                codes_input = input("Enter HSN codes (comma-separated): ").strip()
                codes = [code.strip() for code in codes_input.split(",")]
                result = await agent.batch_validate(codes)
                print(json.dumps(result, indent=2))
            
            elif choice == "4":
                print("Goodbye!")
                break
            
            else:
                print("Invalid choice. Please try again.")
    
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure HSN_SAC.xlsx is available in the current directory")


if __name__ == "__main__":
    asyncio.run(main())