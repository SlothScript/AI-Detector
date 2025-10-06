#!/usr/bin/env python3
"""
AI Data Gathering Script using Ollama and smollm2:135m
Generates AI text samples and saves them to aiData.txt
"""

import subprocess
import sys
import os
import re
import random

class OllamaAIDataGenerator:
    """Generates AI text using Ollama with smollm2:135m model"""
    
    def __init__(self, model="smollm2:135m"):
        self.model = model
        self.output_file = "aiData.txt"
        
        # Prompt generation components
        self.story_subjects = [
            "a young detective", "an elderly gardener", "a curious child",
            "a wandering artist", "a brilliant scientist", "a retired teacher",
            "a street musician", "a small-town librarian", "a space explorer",
            "a deep-sea diver", "a mountain climber", "a time traveler",
            "a talented chef", "a wildlife photographer", "a software engineer"
        ]
        
        self.story_discoveries = [
            "a hidden door", "a mysterious letter", "an ancient artifact",
            "a secret society", "a forgotten technology", "a portal",
            "a talking animal", "a magical object", "a conspiracy",
            "hidden abilities", "a lost civilization", "a time capsule"
        ]
        
        self.story_settings = [
            "a quiet suburban town", "a bustling metropolis", "a remote village",
            "an abandoned facility", "a futuristic city", "a space station",
            "an underwater colony", "a desert outpost", "a floating island"
        ]
        
        self.technical_topics = [
            "blockchain technology", "neural networks", "quantum computing",
            "renewable energy", "genetic engineering", "nanotechnology",
            "virtual reality", "augmented reality", "robotics",
            "cloud computing", "Internet of Things", "autonomous vehicles"
        ]
        
        self.philosophical_concepts = [
            "free will", "consciousness", "reality", "mortality",
            "truth", "knowledge", "morality", "identity",
            "time", "purpose", "happiness", "justice"
        ]
        
        self.discussion_topics = [
            "climate change", "AI ethics", "digital privacy",
            "future of work", "space exploration", "genetic modification",
            "social media", "education reform", "healthcare innovation",
            "income inequality", "mental health", "sustainable living"
        ]
        
        self.creative_scenarios = [
            "humans can read minds", "gravity stops working",
            "everyone dreams the same dream", "animals can speak",
            "time flows backward", "memories can be transferred",
            "emotions are visible as colors", "the internet disappears",
            "sleep becomes unnecessary", "music has physical effects"
        ]
        
    def check_ollama_installed(self):
        """Check if Ollama is installed and accessible"""
        try:
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                print(f"✓ Ollama is installed: {result.stdout.strip()}")
                return True
            else:
                print("✗ Ollama command failed")
                return False
        except FileNotFoundError:
            print("✗ Ollama is not installed or not in PATH")
            return False
        except Exception as e:
            print(f"✗ Error checking Ollama: {e}")
            return False
    
    def check_model_available(self):
        """Check if the model is available"""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if self.model in result.stdout:
                print(f"✓ Model '{self.model}' is available")
                return True
            else:
                print(f"✗ Model '{self.model}' not found")
                print("Available models:")
                print(result.stdout)
                print(f"\nTo install, run: ollama pull {self.model}")
                return False
        except Exception as e:
            print(f"✗ Error checking models: {e}")
            return False
    
    def generate_text(self, prompt, max_tokens=500):
        """Generate text using Ollama"""
        try:
            print(f"\nGenerating text for prompt: '{prompt[:50]}...'")
            
            process = subprocess.Popen(
                ["ollama", "run", self.model, prompt],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            try:
                stdout, stderr = process.communicate(timeout=60)
                
                if process.returncode == 0:
                    print("✓ Text generated successfully")
                    return stdout.strip()
                else:
                    print(f"✗ Generation failed: {stderr}")
                    return None
                    
            except subprocess.TimeoutExpired:
                process.kill()
                print("✗ Generation timed out")
                return None
                
        except Exception as e:
            print(f"✗ Error generating text: {e}")
            return None
    
    def generate_dynamic_prompt(self):
        """Generate a random prompt dynamically"""
        prompt_types = [
            "story", "technical", "philosophical", "discussion",
            "creative", "descriptive", "comparison", "tutorial"
        ]
        
        prompt_type = random.choice(prompt_types)
        
        if prompt_type == "story":
            subject = random.choice(self.story_subjects)
            discovery = random.choice(self.story_discoveries)
            setting = random.choice(self.story_settings)
            return f"Write a short story about {subject} who discovers {discovery} in {setting}."
        
        elif prompt_type == "technical":
            topic = random.choice(self.technical_topics)
            styles = [
                f"Explain {topic} in simple terms.",
                f"Describe how {topic} works.",
                f"What is {topic} and why is it important?",
                f"Discuss the future of {topic}."
            ]
            return random.choice(styles)
        
        elif prompt_type == "philosophical":
            concept = random.choice(self.philosophical_concepts)
            styles = [
                f"Write a philosophical reflection on {concept}.",
                f"What is the nature of {concept}?",
                f"Explore the concept of {concept}."
            ]
            return random.choice(styles)
        
        elif prompt_type == "discussion":
            topic = random.choice(self.discussion_topics)
            styles = [
                f"Discuss the implications of {topic}.",
                f"What are the challenges of {topic}?",
                f"Analyze the impact of {topic} on society."
            ]
            return random.choice(styles)
        
        elif prompt_type == "creative":
            scenario = random.choice(self.creative_scenarios)
            return f"Write a story about a world where {scenario}."
        
        elif prompt_type == "descriptive":
            subjects = [
                "a bustling marketplace", "a serene forest", "a high-tech lab",
                "an ancient library", "a space colony", "a mountain peak",
                "a desert oasis", "a cyberpunk city"
            ]
            return f"Describe {random.choice(subjects)} in vivid detail."
        
        elif prompt_type == "comparison":
            comparisons = [
                ("AI", "human intelligence"),
                ("urban living", "rural living"),
                ("books", "digital media"),
                ("renewable energy", "fossil fuels")
            ]
            item1, item2 = random.choice(comparisons)
            return f"Compare and contrast {item1} and {item2}."
        
        else:  # tutorial
            activities = [
                "start a garden", "learn a language", "build a robot",
                "write a novel", "create digital art", "practice mindfulness"
            ]
            return f"Provide a guide on how to {random.choice(activities)}."
    
    def generate_dataset(self, num_samples=10, use_dynamic=False):
        """Generate multiple AI text samples"""
        
        static_prompts = [
            "Write a short story about a person discovering something mysterious in their town.",
            "Explain the concept of artificial intelligence in simple terms.",
            "Describe a futuristic city and what life would be like there.",
            "Write about the importance of environmental conservation.",
            "Create a narrative about time travel and its consequences.",
            "Discuss the role of technology in modern education.",
            "Write a philosophical reflection on the nature of consciousness.",
            "Describe a day in the life of an astronaut on Mars.",
            "Explain how machine learning algorithms work.",
            "Write about the evolution of human communication.",
            "Create a story about a scientist making an important discovery.",
            "Discuss the ethical implications of genetic engineering.",
            "Write about a world where robots and humans coexist.",
            "Describe the process of innovation and creativity.",
            "Write a narrative about exploring the depths of the ocean.",
            "Discuss the impact of social media on society.",
            "Write about the power of human resilience and adaptation.",
            "Create a story set in a post-apocalyptic world.",
            "Explain the principles of quantum mechanics simply.",
            "Write about the relationship between art and technology.",
            "Describe life on a generation ship traveling to another star.",
            "Write about the discovery of a new form of mathematics.",
            "Explain the concept of emergence in complex systems.",
            "Create a story about the last day before an asteroid impact.",
            "Discuss the philosophy of effective altruism.",
            "Write about memory and how it shapes identity.",
            "Describe a post-scarcity society.",
            "Explain the butterfly effect and chaos theory.",
            "Write about first contact with alien intelligence.",
            "Discuss the ethics of artificial general intelligence.",
            "Create a story about someone who can see the future.",
            "Explain neuroplasticity and brain adaptation.",
            "Write about the role of failure in science.",
            "Describe a world without written language.",
            "Discuss creativity and constraints.",
            "Write about terraforming Mars.",
            "Explain quantum entanglement.",
            "Create a story in a vertical city reaching the clouds.",
            "Discuss automation's impact on employment.",
            "Write about the nature and purpose of dreams.",
            "Describe the experience of synesthesia.",
            "Explain how vaccines work at the molecular level.",
            "Write a story about living in a simulation.",
            "Discuss universal basic income.",
            "Create a narrative about the last human in an AI world.",
            "Explain game theory principles.",
            "Write about language evolution.",
            "Describe a day in a medieval monastery.",
            "Discuss ethics of human enhancement.",
            "Write about rediscovering lost knowledge."
        ]
        
        generated_texts = []
        
        if use_dynamic:
            print("Using dynamic prompt generation...")
            prompts = [self.generate_dynamic_prompt() for _ in range(num_samples)]
            num_to_generate = num_samples
        else:
            prompts = static_prompts
            num_to_generate = min(num_samples, len(prompts))
        
        for i in range(num_to_generate):
            if use_dynamic:
                prompt = prompts[i]
            else:
                prompt = prompts[i % len(prompts)]
                
            print(f"\n{'='*60}")
            print(f"Sample {i+1}/{num_to_generate}")
            print(f"{'='*60}")
            
            text = self.generate_text(prompt)
            
            if text:
                generated_texts.append(text)
                print(f"✓ Generated {len(text)} characters")
            else:
                print("✗ Failed to generate text for this prompt")
        
        return generated_texts
    
    def split_into_sentences(self, text):
        """Split text into sentences using regex"""
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$'
        sentences = re.split(sentence_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def save_to_file(self, texts, append=True, split_sentences=True):
        """Save generated texts to aiData.txt"""
        try:
            mode = 'a' if append else 'w'
            total_sentences = 0
            
            with open(self.output_file, mode, encoding='utf-8') as f:
                if append and os.path.exists(self.output_file) and os.path.getsize(self.output_file) > 0:
                    f.write("\n")
                
                for text in texts:
                    if split_sentences:
                        sentences = self.split_into_sentences(text)
                        total_sentences += len(sentences)
                        for sentence in sentences:
                            f.write(sentence)
                            f.write("\n")
                    else:
                        f.write(text)
                        f.write("\n")
            
            if split_sentences:
                print(f"\n✓ Successfully saved {len(texts)} samples ({total_sentences} sentences) to {self.output_file}")
            else:
                print(f"\n✓ Successfully saved {len(texts)} samples to {self.output_file}")
            
            file_size = os.path.getsize(self.output_file)
            print(f"✓ File size: {file_size:,} bytes ({file_size/1024:.2f} KB)")
            
        except Exception as e:
            print(f"✗ Error saving to file: {e}")

def main():
    """Main function"""
    print("="*60)
    print("AI Data Generator using Ollama (smollm2:135m)")
    print("="*60)
    
    generator = OllamaAIDataGenerator()
    
    print("\nChecking prerequisites...")
    if not generator.check_ollama_installed():
        print("\nPlease install Ollama from: https://ollama.ai")
        sys.exit(1)
    
    if not generator.check_model_available():
        print(f"\nPlease install the model by running:")
        print(f"  ollama pull {generator.model}")
        sys.exit(1)
    
    try:
        num_samples = int(input("\nHow many text samples to generate? (default: 10): ") or "10")
    except ValueError:
        num_samples = 10
    
    prompt_choice = input("\nUse dynamic prompt generation? (y/n, default: n): ").lower() or "n"
    use_dynamic = prompt_choice == "y"
    
    append_choice = input("\nAppend to existing aiData.txt? (y/n, default: y): ").lower() or "y"
    append = append_choice == "y"
    
    print(f"\nGenerating {num_samples} samples...")
    
    texts = generator.generate_dataset(num_samples, use_dynamic=use_dynamic)
    
    if texts:
        print(f"\n✓ Successfully generated {len(texts)} text samples")
        generator.save_to_file(texts, append=append, split_sentences=True)
        print("\n" + "="*60)
        print("Generation complete!")
        print("="*60)
    else:
        print("\n✗ No texts were generated successfully")
        sys.exit(1)

if __name__ == "__main__":
    main()
