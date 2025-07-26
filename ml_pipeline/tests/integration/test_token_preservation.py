#!/usr/bin/env python3
"""
Test script for token preservation functionality
Verifies EV-specific terminology is properly preserved during tokenization
"""

import json
import logging
from typing import List, Dict, Any
from src.data_processing.token_preservation import (
    TokenPreservation, 
    create_ev_tokenizer, 
    tokenize_ev_documents, 
    analyze_token_preservation
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TokenPreservationTester:
    """Test token preservation functionality"""
    
    def __init__(self):
        self.tokenizer, self.preservation = create_ev_tokenizer()
        self.test_documents = self._create_test_documents()
    
    def _create_test_documents(self) -> List[Dict[str, Any]]:
        """Create test documents with EV-specific terminology"""
        
        test_docs = [
            {
                "id": "doc_001",
                "content": "The CCS2 charging standard supports up to 350kW power output and uses the OCPP protocol for communication. The ISO15118 standard enables Plug&Charge functionality.",
                "expected_terms": ["CCS2", "350kW", "OCPP", "ISO15118", "Plug&Charge"]
            },
            {
                "id": "doc_002", 
                "content": "Tesla Supercharger stations provide 250kW charging with Type2 connectors. The battery capacity is 75kWh with a range of 300km.",
                "expected_terms": ["Tesla Supercharger", "250kW", "Type2", "75kWh"]
            },
            {
                "id": "doc_003",
                "content": "CHAdeMO charging supports bidirectional V2G (Vehicle-to-Grid) functionality. The smart grid integration enables load balancing and peak shaving.",
                "expected_terms": ["CHAdeMO", "V2G", "smart_grid", "load_balancing", "peak_shaving"]
            },
            {
                "id": "doc_004",
                "content": "The carbon footprint of EV charging depends on renewable energy sources. CO2 emissions are reduced by 80% compared to ICE vehicles.",
                "expected_terms": ["carbon_footprint", "CO2_emissions", "renewable_energy"]
            },
            {
                "id": "doc_005",
                "content": "Power quality and efficiency are crucial for EV charging infrastructure. Voltage drop and cable losses must be minimized for optimal performance.",
                "expected_terms": ["power_quality", "efficiency", "voltage_drop", "cable_losses"]
            }
        ]
        
        return test_docs
    
    def test_basic_tokenization(self) -> bool:
        """Test basic tokenization functionality"""
        try:
            logger.info("🧪 Testing basic tokenization...")
            
            test_text = "CCS2 charging at 350kW"
            tokens = self.preservation.tokenize_with_preservation(
                test_text,
                truncation=True,
                max_length=64,
                padding="max_length"
            )
            
            # Verify tokenization worked
            assert "input_ids" in tokens
            assert "attention_mask" in tokens
            assert len(tokens["input_ids"]) > 0
            
            logger.info("  ✅ Basic tokenization passed")
            return True
            
        except Exception as e:
            logger.error(f"  ❌ Basic tokenization failed: {e}")
            return False
    
    def test_term_preservation(self) -> bool:
        """Test that EV terms are preserved as single tokens"""
        try:
            logger.info("🔋 Testing EV term preservation...")
            
            # Test key EV terms
            ev_terms = ["CCS2", "CHAdeMO", "OCPP", "ISO15118", "Plug&Charge", "350kW", "75kWh"]
            
            preserved_count = 0
            for term in ev_terms:
                verification = self.preservation.verify_token_preservation(term)
                
                if verification["overall_score"] == 1.0:
                    preserved_count += 1
                    logger.info(f"    ✅ {term} - Preserved")
                else:
                    logger.warning(f"    ⚠️ {term} - Not fully preserved")
            
            preservation_rate = preserved_count / len(ev_terms)
            logger.info(f"  📊 Term preservation rate: {preservation_rate:.1%} ({preserved_count}/{len(ev_terms)})")
            
            # Expect at least 80% preservation
            return preservation_rate >= 0.8
            
        except Exception as e:
            logger.error(f"  ❌ Term preservation test failed: {e}")
            return False
    
    def test_document_tokenization(self) -> bool:
        """Test tokenization of full documents"""
        try:
            logger.info("📄 Testing document tokenization...")
            
            # Tokenize test documents
            tokenized_docs = tokenize_ev_documents(
                self.test_documents, 
                self.tokenizer, 
                self.preservation
            )
            
            # Verify all documents were processed
            assert len(tokenized_docs) == len(self.test_documents)
            
            # Check preservation scores
            avg_preservation_score = sum(doc["preservation_score"] for doc in tokenized_docs) / len(tokenized_docs)
            logger.info(f"  📊 Average preservation score: {avg_preservation_score:.2f}")
            
            # Log individual document scores
            for doc in tokenized_docs:
                logger.info(f"    📄 {doc['id']}: {doc['preservation_score']:.2f}")
            
            # Expect average score > 0.7
            return avg_preservation_score > 0.7
            
        except Exception as e:
            logger.error(f"  ❌ Document tokenization test failed: {e}")
            return False
    
    def test_preservation_analysis(self) -> bool:
        """Test preservation analysis functionality"""
        try:
            logger.info("📊 Testing preservation analysis...")
            
            # Analyze preservation across documents
            analysis = analyze_token_preservation(self.test_documents, self.preservation)
            
            # Verify analysis structure
            required_keys = ["total_documents", "total_terms_found", "total_terms_preserved", "overall_preservation_rate"]
            for key in required_keys:
                assert key in analysis
            
            # Log analysis results
            logger.info(f"  📈 Analysis Results:")
            logger.info(f"    Total documents: {analysis['total_documents']}")
            logger.info(f"    Total terms found: {analysis['total_terms_found']}")
            logger.info(f"    Total terms preserved: {analysis['total_terms_preserved']}")
            logger.info(f"    Overall preservation rate: {analysis['overall_preservation_rate']:.2%}")
            
            # Log category analysis
            logger.info(f"  📋 Category Analysis:")
            for category, stats in analysis["category_analysis"].items():
                if stats["found"] > 0:
                    rate = stats["preserved"] / stats["found"]
                    logger.info(f"    {category}: {rate:.2%} ({stats['preserved']}/{stats['found']})")
            
            # Expect overall preservation rate > 0.6
            return analysis["overall_preservation_rate"] > 0.6
            
        except Exception as e:
            logger.error(f"  ❌ Preservation analysis test failed: {e}")
            return False
    
    def test_custom_terms(self) -> bool:
        """Test adding custom terms"""
        try:
            logger.info("➕ Testing custom term addition...")
            
            # Add custom EV terms
            custom_terms = ["FastNed", "Ionity", "ElectrifyAmerica", "EVgo"]
            added_count = self.preservation.add_custom_terms(custom_terms, "charging_networks")
            
            # Verify terms were added
            assert added_count > 0
            
            # Test preservation of custom terms
            test_text = "FastNed and Ionity provide high-speed charging"
            verification = self.preservation.verify_token_preservation(test_text)
            
            # Check if custom terms are found
            found_terms = [term_info["term"] for term_info in verification["terms_found"]]
            custom_terms_found = [term for term in custom_terms if term in found_terms]
            
            logger.info(f"  📊 Custom terms found: {len(custom_terms_found)}/{len(custom_terms)}")
            
            # Expect at least some custom terms to be found
            return len(custom_terms_found) > 0
            
        except Exception as e:
            logger.error(f"  ❌ Custom terms test failed: {e}")
            return False
    
    def test_config_export_import(self) -> bool:
        """Test configuration export and import"""
        try:
            logger.info("💾 Testing config export/import...")
            
            # Export configuration
            export_path = "test_preservation_config.json"
            self.preservation.export_preservation_config(export_path)
            
            # Create new preservation instance
            new_preservation = TokenPreservation(self.tokenizer, "electric_vehicles")
            
            # Import configuration
            new_preservation.load_preservation_config(export_path)
            
            # Verify configuration was loaded
            assert len(new_preservation.special_terms) > 0
            assert len(new_preservation.term_mappings) > 0
            
            logger.info("  ✅ Config export/import passed")
            return True
            
        except Exception as e:
            logger.error(f"  ❌ Config export/import test failed: {e}")
            return False
    
    def test_statistics(self) -> bool:
        """Test preservation statistics"""
        try:
            logger.info("📈 Testing preservation statistics...")
            
            # Get statistics
            stats = self.preservation.get_preservation_statistics()
            
            # Verify statistics structure
            required_keys = ["domain", "total_terms", "preserved_terms", "preservation_rate"]
            for key in required_keys:
                assert key in stats
            
            # Log statistics
            logger.info(f"  📊 Statistics:")
            logger.info(f"    Domain: {stats['domain']}")
            logger.info(f"    Total terms: {stats['total_terms']}")
            logger.info(f"    Preserved terms: {stats['preserved_terms']}")
            logger.info(f"    Preservation rate: {stats['preservation_rate']:.2%}")
            
            # Log category statistics
            logger.info(f"  📋 Category Statistics:")
            for category, cat_stats in stats["categories"].items():
                logger.info(f"    {category}: {cat_stats['preservation_rate']:.2%} ({cat_stats['preserved']}/{cat_stats['total']})")
            
            # Expect reasonable preservation rate
            return stats["preservation_rate"] > 0.5
            
        except Exception as e:
            logger.error(f"  ❌ Statistics test failed: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all token preservation tests"""
        logger.info("🚀 Token Preservation Test Suite")
        logger.info("=" * 50)
        
        tests = [
            ("Basic Tokenization", self.test_basic_tokenization),
            ("Term Preservation", self.test_term_preservation),
            ("Document Tokenization", self.test_document_tokenization),
            ("Preservation Analysis", self.test_preservation_analysis),
            ("Custom Terms", self.test_custom_terms),
            ("Config Export/Import", self.test_config_export_import),
            ("Statistics", self.test_statistics),
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            logger.info(f"\n🔍 Running: {test_name}")
            try:
                results[test_name] = test_func()
            except Exception as e:
                logger.error(f"  ❌ Test failed with exception: {e}")
                results[test_name] = False
        
        # Summary
        logger.info("\n" + "=" * 50)
        logger.info("📋 Test Results Summary:")
        
        passed = sum(results.values())
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"  {status} {test_name}")
        
        logger.info(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        return results

def main():
    """Main test function"""
    logger.info("🔋 EV Token Preservation Test Suite")
    logger.info("=" * 50)
    
    # Create tester
    tester = TokenPreservationTester()
    
    # Run tests
    results = tester.run_all_tests()
    
    # Exit with appropriate code
    if all(results.values()):
        logger.info("🎉 All token preservation tests passed!")
        return 0
    else:
        logger.error("❌ Some token preservation tests failed!")
        return 1

if __name__ == "__main__":
    exit(main()) 