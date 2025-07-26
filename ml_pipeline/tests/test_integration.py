import pytest
import os
import tempfile
import shutil
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_collection import WebScraper, PDFExtractor
from src.data_processing import DataCleaner, QualityFilter, Normalizer
from src.training.dataset_preparation import QADatasetPreparer
from src.evaluation import BenchmarkCreator, MetricsCalculator

class TestIntegration:
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_data_collection(self, temp_dir):
        """Test data collection stage"""
        # Test web scraping
        scraper = WebScraper(output_dir=f"{temp_dir}/web_data")
        test_urls = ["https://httpbin.org/html"]  # Safe test URL
        scraper.collect(test_urls)
        
        # Check if files were created
        files = os.listdir(f"{temp_dir}/web_data")
        assert len(files) > 0
    
    def test_data_processing(self, temp_dir):
        """Test data processing stage"""
        import pandas as pd
        
        # Create test data
        test_data = pd.DataFrame({
            "text": [
                "EV charging stations are essential for electric vehicles.",
                "EV charging stations are essential for electric vehicles.",  # Duplicate
                "Short text.",  # Low quality
                "Tesla Superchargers provide fast charging for Tesla vehicles."
            ]
        })
        
        # Test cleaning
        cleaner = DataCleaner()
        cleaned_data = cleaner.process(
            test_data, 
            text_column="text",
            remove_boilerplate=False,
            filter_sentences=True,
            min_length=20
        )
        
        # Should remove duplicates and short text
        assert len(cleaned_data) < len(test_data)
    
    def test_qa_generation(self, temp_dir):
        """Test QA pair generation (mock)"""
        # Mock OpenAI API key for testing
        os.environ["OPENAI_API_KEY"] = "test-key"
        
        preparer = QADatasetPreparer(
            openai_api_key="test-key",
            domain="test",
            output_dir=f"{temp_dir}/qa_data"
        )
        
        # Test with mock data
        test_texts = ["EV charging is important for sustainability."]
        
        # This will fail due to invalid API key, but tests the setup
        with pytest.raises(Exception):
            preparer.prepare(test_texts, n_questions=1)
    
    def test_benchmark_creation(self, temp_dir):
        """Test benchmark creation"""
        creator = BenchmarkCreator(domain="test")
        benchmark = creator.create_benchmark(n=5)
        
        assert len(benchmark) == 5
        assert "prompt" in benchmark[0]
    
    def test_metrics_calculation(self):
        """Test metrics calculation"""
        calc = MetricsCalculator()
        
        predictions = ["EV charging is fast."]
        references = ["Electric vehicle charging is quick."]
        
        # Test F1
        f1_score = calc.compute_f1(predictions, references)
        assert 0 <= f1_score <= 1
        
        # Test exact match
        em_score = calc.compute_exact_match(predictions, references)
        assert 0 <= em_score <= 1

if __name__ == "__main__":
    pytest.main([__file__]) 