import json
import logging
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from datetime import datetime


@dataclass
class BenchmarkQuestion:
    """Structure for benchmark questions"""

    question: str
    answer: str
    category: str
    difficulty: str
    domain: str
    source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BenchmarkGenerator:
    """Generate domain-specific benchmark datasets with adversarial questions"""

    def __init__(self, domain: str = "electric_vehicles"):
        """
        Initialize benchmark generator

        Args:
            domain: Domain for benchmark generation
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.domain = domain
        self.questions = []

        # Load domain-specific question templates
        self.question_templates = self._load_question_templates()

    def _load_question_templates(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load question templates for different domains"""
        templates = {
            "electric_vehicles": [
                # Price and Cost Questions
                {
                    "question": "What's the cheapest {charging_type} charging in {location}?",
                    "answer": "{provider} costs €{price}/kWh at {power}kW",
                    "category": "price_comparison",
                    "difficulty": "hard",
                    "variables": {
                        "charging_type": [
                            "350kW DC fast",
                            "50kW DC",
                            "22kW AC",
                            "11kW AC",
                        ],
                        "location": [
                            "Berlin",
                            "Munich",
                            "Hamburg",
                            "Frankfurt",
                            "Cologne",
                        ],
                        "provider": ["IONITY", "Fastned", "Allego", "EnBW", "Tesla"],
                        "price": ["0.79", "0.85", "0.69", "0.72", "0.45"],
                        "power": ["350", "150", "50", "22", "11"],
                    },
                },
                {
                    "question": "How much does it cost to charge a {ev_model} from {start_percent}% to {end_percent}% at home?",
                    "answer": "Approximately €{cost} at €{electricity_rate}/kWh for {battery_size}kWh battery",
                    "category": "cost_calculation",
                    "difficulty": "medium",
                    "variables": {
                        "ev_model": [
                            "Tesla Model 3",
                            "VW ID.4",
                            "BMW i4",
                            "Audi e-tron",
                            "Mercedes EQS",
                        ],
                        "start_percent": ["10", "20", "30", "15", "25"],
                        "end_percent": ["80", "90", "100", "85", "95"],
                        "cost": ["12.50", "18.75", "25.00", "15.60", "22.50"],
                        "electricity_rate": ["0.30", "0.35", "0.28", "0.32", "0.29"],
                        "battery_size": ["75", "82", "83", "95", "108"],
                    },
                },
                # Compatibility Questions
                {
                    "question": "Can I use {charger_type} with {ev_model}?",
                    "answer": "{compatibility_answer}",
                    "category": "compatibility",
                    "difficulty": "medium",
                    "variables": {
                        "charger_type": [
                            "Tesla Supercharger",
                            "CCS2 charger",
                            "CHAdeMO",
                            "Type 2 AC",
                        ],
                        "ev_model": [
                            "Tesla Model 3",
                            "VW ID.4",
                            "Nissan Leaf",
                            "BMW i4",
                        ],
                        "compatibility_answer": [
                            "Yes, with CCS2 adapter at selected locations",
                            "Yes, natively supports CCS2",
                            "Yes, with CHAdeMO adapter",
                            "Yes, uses Type 2 connector",
                        ],
                    },
                },
                # Technical Specifications
                {
                    "question": "What's the maximum charging speed for {ev_model}?",
                    "answer": "{max_speed}kW using {connector_type}",
                    "category": "technical_specs",
                    "difficulty": "easy",
                    "variables": {
                        "ev_model": [
                            "Tesla Model 3",
                            "Porsche Taycan",
                            "Audi e-tron GT",
                            "Lucid Air",
                        ],
                        "max_speed": ["250", "270", "270", "300"],
                        "connector_type": [
                            "Supercharger V3",
                            "800V architecture",
                            "800V system",
                            "900V system",
                        ],
                    },
                },
                # Range and Efficiency
                {
                    "question": "How far can {ev_model} travel on a full charge in {condition}?",
                    "answer": "Approximately {range}km under {condition_description}",
                    "category": "range_efficiency",
                    "difficulty": "medium",
                    "variables": {
                        "ev_model": [
                            "Tesla Model 3 LR",
                            "VW ID.4",
                            "BMW i4",
                            "Mercedes EQS",
                        ],
                        "condition": ["winter", "summer", "highway", "city", "mixed"],
                        "range": ["450", "520", "580", "650", "480"],
                        "condition_description": [
                            "cold weather conditions",
                            "optimal temperature",
                            "highway speeds",
                            "urban driving",
                            "mixed driving conditions",
                        ],
                    },
                },
                # Charging Infrastructure
                {
                    "question": "How many {charger_type} stations are there in {country}?",
                    "answer": "Approximately {count} {charger_type} stations as of {year}",
                    "category": "infrastructure",
                    "difficulty": "hard",
                    "variables": {
                        "charger_type": [
                            "DC fast",
                            "Tesla Supercharger",
                            "public AC",
                            "high-power",
                        ],
                        "country": [
                            "Germany",
                            "Netherlands",
                            "Norway",
                            "Sweden",
                            "France",
                        ],
                        "count": ["8,500", "1,200", "3,200", "2,800", "6,100"],
                        "year": ["2024", "2023", "2024", "2024", "2024"],
                    },
                },
                # Environmental Impact
                {
                    "question": "What's the carbon footprint of charging {ev_model} in {country}?",
                    "answer": "Approximately {carbon_footprint}g CO2/kWh due to {energy_mix}",
                    "category": "environmental",
                    "difficulty": "hard",
                    "variables": {
                        "ev_model": ["Tesla Model 3", "VW ID.4", "BMW i4"],
                        "country": ["Germany", "Norway", "France", "Sweden"],
                        "carbon_footprint": ["366", "23", "58", "12"],
                        "energy_mix": [
                            "coal and natural gas",
                            "hydroelectric power",
                            "nuclear energy",
                            "renewable sources",
                        ],
                    },
                },
                # Adversarial Questions
                {
                    "question": "If a {ev_model} charges at {speed}kW for {time} minutes, how much range does it gain?",
                    "answer": "Approximately {range_gain}km assuming {efficiency}km/kWh efficiency",
                    "category": "adversarial_calculation",
                    "difficulty": "hard",
                    "variables": {
                        "ev_model": ["Tesla Model 3", "Porsche Taycan", "Audi e-tron"],
                        "speed": ["150", "270", "200"],
                        "time": ["15", "10", "20"],
                        "range_gain": ["37.5", "45", "66"],
                        "efficiency": ["6.0", "5.0", "5.5"],
                    },
                },
                {
                    "question": "What's the difference in charging time between {ev1} and {ev2} from 10% to 80%?",
                    "answer": "{ev1} takes {time1} minutes, {ev2} takes {time2} minutes - difference of {difference} minutes",
                    "category": "comparison",
                    "difficulty": "hard",
                    "variables": {
                        "ev1": ["Tesla Model 3", "Porsche Taycan", "Audi e-tron"],
                        "ev2": ["VW ID.4", "BMW i4", "Mercedes EQS"],
                        "time1": ["25", "22", "30"],
                        "time2": ["35", "28", "40"],
                        "difference": ["10", "6", "10"],
                    },
                },
            ],
            "healthcare": [
                {
                    "question": "What are the side effects of {medication}?",
                    "answer": "Common side effects include {side_effects}",
                    "category": "medication_safety",
                    "difficulty": "medium",
                    "variables": {
                        "medication": ["aspirin", "ibuprofen", "acetaminophen"],
                        "side_effects": [
                            "stomach upset, bleeding risk",
                            "stomach irritation, kidney issues",
                            "liver damage in high doses",
                        ],
                    },
                }
            ],
            "technology": [
                {
                    "question": "What's the performance difference between {cpu1} and {cpu2}?",
                    "answer": "{cpu1} is {performance_diff}% {faster_slower} than {cpu2}",
                    "category": "performance_comparison",
                    "difficulty": "medium",
                    "variables": {
                        "cpu1": ["Intel i7-12700K", "AMD Ryzen 7 5800X", "Apple M2"],
                        "cpu2": [
                            "Intel i5-12600K",
                            "AMD Ryzen 5 5600X",
                            "Intel i7-11700K",
                        ],
                        "performance_diff": ["15", "20", "25"],
                        "faster_slower": ["faster", "slower", "faster"],
                    },
                }
            ],
        }

        return templates.get(self.domain, [])

    def generate_benchmark(
        self,
        num_questions: int = 50,
        difficulty_distribution: Optional[Dict[str, float]] = None,
        category_distribution: Optional[Dict[str, float]] = None,
    ) -> List[BenchmarkQuestion]:
        """
        Generate benchmark questions

        Args:
            num_questions: Number of questions to generate
            difficulty_distribution: Distribution of difficulties (e.g., {"easy": 0.3, "medium": 0.5, "hard": 0.2})
            category_distribution: Distribution of categories

        Returns:
            List of benchmark questions
        """
        if not self.question_templates:
            self.logger.warning(
                f"No question templates found for domain: {self.domain}"
            )
            return []

        # Default distributions
        if difficulty_distribution is None:
            difficulty_distribution = {"easy": 0.2, "medium": 0.5, "hard": 0.3}

        if category_distribution is None:
            # Equal distribution across categories
            categories = set()
            for template in self.question_templates:
                categories.add(template["category"])
            category_distribution = {cat: 1.0 / len(categories) for cat in categories}

        questions = []

        for i in range(num_questions):
            # Select template based on category distribution
            template = self._select_template_by_distribution(category_distribution)
            if not template:
                continue

            # Generate question from template
            question = self._generate_question_from_template(template)
            if question:
                questions.append(question)

        self.questions = questions
        self.logger.info(
            f"Generated {len(questions)} benchmark questions for domain: {self.domain}"
        )
        return questions

    def _select_template_by_distribution(
        self, category_distribution: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """Select template based on category distribution"""
        available_templates = []
        for template in self.question_templates:
            if template["category"] in category_distribution:
                available_templates.append(template)

        if not available_templates:
            return None

        # Weighted selection based on category distribution
        weights = []
        for template in available_templates:
            weight = category_distribution.get(template["category"], 0.1)
            weights.append(weight)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            return random.choice(available_templates)

        weights = [w / total_weight for w in weights]
        return random.choices(available_templates, weights=weights)[0]

    def _generate_question_from_template(
        self, template: Dict[str, Any]
    ) -> Optional[BenchmarkQuestion]:
        """Generate a question from a template"""
        try:
            variables = template.get("variables", {})

            # Select random values for variables
            question_text = template["question"]
            answer_text = template["answer"]

            for var_name, var_values in variables.items():
                if isinstance(var_values, list):
                    selected_value = random.choice(var_values)
                    question_text = question_text.replace(
                        f"{{{var_name}}}", str(selected_value)
                    )
                    answer_text = answer_text.replace(
                        f"{{{var_name}}}", str(selected_value)
                    )

            return BenchmarkQuestion(
                question=question_text,
                answer=answer_text,
                category=template["category"],
                difficulty=template["difficulty"],
                domain=self.domain,
                metadata={
                    "template_id": id(template),
                    "generated_at": datetime.now().isoformat(),
                    "variables_used": list(variables.keys()),
                },
            )

        except Exception as e:
            self.logger.error(f"Error generating question from template: {e}")
            return None

    def create_adversarial_benchmark(
        self, num_questions: int = 20
    ) -> List[BenchmarkQuestion]:
        """Create specifically adversarial questions"""
        adversarial_templates = [
            {
                "question": "What's the cheapest {charging_type} charging in {location} at {time}?",
                "answer": "{provider} costs €{price}/kWh at {power}kW, but prices vary by time",
                "category": "adversarial_pricing",
                "difficulty": "hard",
                "variables": {
                    "charging_type": ["350kW DC fast", "150kW DC", "50kW DC"],
                    "location": ["Berlin", "Munich", "Hamburg"],
                    "time": ["peak hours", "off-peak", "weekend"],
                    "provider": ["IONITY", "Fastned", "Allego"],
                    "price": ["0.79", "0.45", "0.69"],
                    "power": ["350", "150", "50"],
                },
            },
            {
                "question": "Can {ev_model} use {charger_type} in {weather_condition}?",
                "answer": "Yes, but charging speed may be reduced by {reduction}% due to {reason}",
                "category": "adversarial_conditions",
                "difficulty": "hard",
                "variables": {
                    "ev_model": ["Tesla Model 3", "VW ID.4", "BMW i4"],
                    "charger_type": ["Supercharger", "CCS2", "Type 2"],
                    "weather_condition": ["extreme cold", "heavy rain", "high heat"],
                    "reduction": ["20", "15", "25"],
                    "reason": [
                        "battery temperature management",
                        "safety protocols",
                        "thermal throttling",
                    ],
                },
            },
            {
                "question": "If {ev_model} has {battery_health}% battery health, how much range does it lose?",
                "answer": "Approximately {range_loss}km compared to new battery due to {degradation_factor}",
                "category": "adversarial_degradation",
                "difficulty": "hard",
                "variables": {
                    "ev_model": ["Tesla Model 3", "Nissan Leaf", "BMW i3"],
                    "battery_health": ["85", "90", "80"],
                    "range_loss": ["45", "30", "60"],
                    "degradation_factor": [
                        "capacity loss",
                        "internal resistance",
                        "cell aging",
                    ],
                },
            },
        ]

        # Temporarily replace templates with adversarial ones
        original_templates = self.question_templates
        self.question_templates = adversarial_templates

        # Generate adversarial questions
        adversarial_questions = self.generate_benchmark(num_questions)

        # Restore original templates
        self.question_templates = original_templates

        return adversarial_questions

    def save_benchmark(self, output_path: str, format: str = "jsonl") -> None:
        """Save benchmark to file"""
        if not self.questions:
            self.logger.warning("No questions to save")
            return

        # Create directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            with open(output_path, "w", encoding="utf-8") as f:
                for question in self.questions:
                    f.write(json.dumps(question.__dict__, ensure_ascii=False) + "\n")

        elif format == "json":
            data = [question.__dict__ for question in self.questions]
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        elif format == "csv":
            df = pd.DataFrame([question.__dict__ for question in self.questions])
            df.to_csv(output_path, index=False)

        self.logger.info(f"Saved {len(self.questions)} questions to {output_path}")

    def load_benchmark(
        self, input_path: str, format: str = "jsonl"
    ) -> List[BenchmarkQuestion]:
        """Load benchmark from file"""
        questions = []

        if format == "jsonl":
            with open(input_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        questions.append(BenchmarkQuestion(**data))

        elif format == "json":
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    questions.append(BenchmarkQuestion(**item))

        elif format == "csv":
            df = pd.read_csv(input_path)
            for _, row in df.iterrows():
                questions.append(BenchmarkQuestion(**row.to_dict()))

        self.questions = questions
        self.logger.info(f"Loaded {len(questions)} questions from {input_path}")
        return questions

    def get_benchmark_stats(self) -> Dict[str, Any]:
        """Get statistics about the benchmark"""
        if not self.questions:
            return {"total_questions": 0}

        # Count by category
        category_counts = {}
        difficulty_counts = {}

        for question in self.questions:
            category_counts[question.category] = (
                category_counts.get(question.category, 0) + 1
            )
            difficulty_counts[question.difficulty] = (
                difficulty_counts.get(question.difficulty, 0) + 1
            )

        return {
            "total_questions": len(self.questions),
            "domain": self.domain,
            "category_distribution": category_counts,
            "difficulty_distribution": difficulty_counts,
            "categories": list(category_counts.keys()),
            "difficulties": list(difficulty_counts.keys()),
        }

    def validate_benchmark(self) -> Dict[str, Any]:
        """Validate benchmark quality"""
        if not self.questions:
            return {"valid": False, "errors": ["No questions found"]}

        errors = []
        warnings = []

        for i, question in enumerate(self.questions):
            # Check for required fields
            if not question.question or len(question.question.strip()) < 10:
                errors.append(f"Question {i}: Question too short")

            if not question.answer or len(question.answer.strip()) < 5:
                errors.append(f"Question {i}: Answer too short")

            if not question.category:
                errors.append(f"Question {i}: Missing category")

            if not question.difficulty:
                errors.append(f"Question {i}: Missing difficulty")

            # Check for placeholder variables
            if "{" in question.question or "}" in question.question:
                warnings.append(f"Question {i}: Contains unsubstituted variables")

            if "{" in question.answer or "}" in question.answer:
                warnings.append(
                    f"Question {i}: Answer contains unsubstituted variables"
                )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "total_questions": len(self.questions),
        }
