import os
import time
import json
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables from .env file
load_dotenv()

class ResponseQuality(Enum):
    """Enumeration for response quality levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INVALID = "invalid"

@dataclass
class Config:
    """Configuration settings for the prompt chaining system"""
    planning_max_tokens: int = 1000
    output_max_tokens: int = 2000
    temperature: float = 0.7
    max_retries: int = 3
    retry_base_delay: float = 1.0  # seconds
    min_plan_length: int = 50
    min_response_length: int = 200
    min_paragraphs: int = 2
    quality_thresholds: Dict[str, float] = None

    def __post_init__(self):
        self.quality_thresholds = {
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4
        }

@dataclass
class ValidationResult:
    """Container for validation results"""
    is_valid: bool
    quality: ResponseQuality
    score: float
    details: Dict[str, Any]

# Azure OpenAI Configuration
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_base = os.getenv("AZURE_OPENAI_API_BASE")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
mini_deployment_name = os.getenv("AZURE_OPENAI_MINI_DEPLOYMENT_NAME")

# Initialize configuration
config = Config()

# Configure OpenAI client
client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=api_base
)

class PromptValidationError(Exception):
    """Exception raised for invalid user prompts"""
    pass

class ResponseValidationError(Exception):
    """Exception raised for invalid AI responses"""
    pass

def validate_user_prompt(prompt: str) -> None:
    """
    Validate the user's input prompt.
    
    Args:
        prompt: The user's input prompt
        
    Raises:
        PromptValidationError: If the prompt is invalid
    """
    if not prompt or not prompt.strip():
        raise PromptValidationError("Prompt cannot be empty")
    
    if len(prompt.strip()) < 10:
        raise PromptValidationError("Prompt is too short")
    
    if len(prompt.strip()) > 1000:
        raise PromptValidationError("Prompt exceeds maximum length of 1000 characters")

def calculate_response_metrics(text: str) -> Dict[str, float]:
    """
    Calculate various metrics for response quality assessment.
    
    Args:
        text: The text to analyze
        
    Returns:
        Dict containing calculated metrics
    """
    words = text.split()
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    metrics = {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "paragraph_count": len(paragraphs),
        "avg_words_per_sentence": len(words) / max(len(sentences), 1),
        "avg_sentences_per_paragraph": len(sentences) / max(len(paragraphs), 1)
    }
    
    return metrics

def assess_response_quality(metrics: Dict[str, float]) -> Tuple[ResponseQuality, float]:
    """
    Assess the quality of a response based on its metrics.
    
    Args:
        metrics: Dictionary of response metrics
        
    Returns:
        Tuple of (ResponseQuality, quality_score)
    """
    # Calculate a quality score based on metrics
    score = 0.0
    
    # Word count scoring
    if metrics["word_count"] >= 200:
        score += 0.3
    elif metrics["word_count"] >= 100:
        score += 0.2
    
    # Paragraph structure scoring
    if metrics["paragraph_count"] >= config.min_paragraphs:
        score += 0.3
    
    # Sentence complexity scoring
    if 10 <= metrics["avg_words_per_sentence"] <= 20:
        score += 0.2
    
    # Paragraph organization scoring
    if 2 <= metrics["avg_sentences_per_paragraph"] <= 5:
        score += 0.2
    
    # Determine quality level
    if score >= config.quality_thresholds["high"]:
        quality = ResponseQuality.HIGH
    elif score >= config.quality_thresholds["medium"]:
        quality = ResponseQuality.MEDIUM
    elif score >= config.quality_thresholds["low"]:
        quality = ResponseQuality.LOW
    else:
        quality = ResponseQuality.INVALID
    
    return quality, score

def print_messages(messages: List[Dict[str, str]]) -> None:
    """
    Print formatted messages for debugging purposes.
    
    Args:
        messages: List of message dictionaries containing role and content
    """
    print("\nPrompt sent to model:")
    for message in messages:
        print(f"{message['role'].capitalize()}: {message['content']}")
    print()

def print_validation_result(stage: str, result: ValidationResult) -> None:
    """
    Print formatted validation results.
    
    Args:
        stage: Name of the validation stage
        result: ValidationResult object
    """
    print(f"\n{stage} Validation Results:")
    print(f"Valid: {result.is_valid}")
    print(f"Quality: {result.quality.value}")
    print(f"Score: {result.score:.2f}")
    print("Details:")
    print(json.dumps(result.details, indent=2))
    print()

class ExecutionStats:
    """Track execution statistics and performance metrics"""
    def __init__(self):
        self.start_time = time.time()
        self.stage_times = {}
        self.total_tokens = 0
        self.stage_tokens = {}
        self.retries = {}
        self.validation_results = {}
    
    def start_stage(self, stage_name: str) -> None:
        """Record the start of a new stage"""
        self.stage_times[stage_name] = {"start": time.time()}
        self.retries[stage_name] = 0
    
    def end_stage(self, stage_name: str, tokens_used: int) -> None:
        """Record the end of a stage"""
        if stage_name in self.stage_times:
            self.stage_times[stage_name]["end"] = time.time()
            self.stage_times[stage_name]["duration"] = (
                self.stage_times[stage_name]["end"] - 
                self.stage_times[stage_name]["start"]
            )
        self.stage_tokens[stage_name] = tokens_used
        self.total_tokens += tokens_used
    
    def increment_retry(self, stage_name: str) -> None:
        """Increment retry counter for a stage"""
        self.retries[stage_name] = self.retries.get(stage_name, 0) + 1
    
    def add_validation_result(self, stage_name: str, result: ValidationResult) -> None:
        """Record validation result for a stage"""
        self.validation_results[stage_name] = result
    
    def get_summary(self) -> Dict[str, Any]:
        """Get execution summary"""
        total_duration = time.time() - self.start_time
        return {
            "total_duration": total_duration,
            "total_tokens": self.total_tokens,
            "stage_durations": {
                name: stats["duration"] 
                for name, stats in self.stage_times.items()
                if "duration" in stats
            },
            "stage_tokens": self.stage_tokens,
            "retries": self.retries,
            "validation_results": {
                name: {
                    "quality": result.quality.value,
                    "score": result.score,
                    "details": result.details
                }
                for name, result in self.validation_results.items()
            }
        }

def handle_api_error(e: Exception, stage: str) -> str:
    """
    Handle API-related errors with appropriate error messages.
    
    Args:
        e: The exception that occurred
        stage: The stage where the error occurred
        
    Returns:
        A user-friendly error message
    """
    if "rate_limit" in str(e).lower():
        return f"Rate limit exceeded during {stage}. Please try again in a few seconds."
    elif "context_length" in str(e).lower():
        return f"Input too long for {stage}. Please try a shorter prompt."
    elif "authentication" in str(e).lower():
        return f"Authentication error during {stage}. Please check your API credentials."
    else:
        return f"An error occurred during {stage}: {str(e)}"

def process_response(response: str, min_length: int = 0) -> str:
    """
    Process and clean up the response text.
    
    Args:
        response: Raw response text
        min_length: Minimum required length
        
    Returns:
        Processed response text
        
    Raises:
        ResponseValidationError: If response is too short
    """
    if not response:
        raise ResponseValidationError("Empty response received")
    
    # Remove excessive whitespace
    processed = "\n".join(line.strip() for line in response.splitlines())
    processed = " ".join(processed.split())
    
    # Ensure minimum length
    if min_length > 0 and len(processed) < min_length:
        raise ResponseValidationError(
            f"Response too short: {len(processed)} chars (minimum: {min_length})"
        )
    
    return processed

def planning_stage(user_prompt: str) -> Optional[str]:
    """
    Uses gpt-4o-mini to create a plan for addressing the user prompt.
    
    Args:
        user_prompt: The user's input prompt
        
    Returns:
        Plan as a string, or None if planning fails
    """
    try:
        validate_user_prompt(user_prompt)
        
        print("\n" + "="*50)
        print("PLANNING STAGE (using gpt-4o-mini)")
        print("="*50)
        
        planning_prompt = [
            {"role": "system", "content": "You are an AI assistant that helps with task planning and decomposition."},
            {"role": "user", "content": f"Create a concise, structured plan to address the following task: '{user_prompt}'. Provide 3-5 main steps, each with a brief description of its importance and 2-3 key considerations. Ensure all points are fully completed within the given space."}
        ]
        print_messages(planning_prompt)
        
        start_time = time.time()
        planning_response = client.chat.completions.create(
            model=mini_deployment_name,
            messages=planning_prompt,
            max_tokens=config.planning_max_tokens
        )
        end_time = time.time()
        
        plan = planning_response.choices[0].message.content.strip()
        print("Planning Stage Output:")
        print(plan)
        print(f"\nExecution time: {end_time - start_time:.2f} seconds")
        print(f"Tokens used: {planning_response.usage.total_tokens}")
        print()
        
        # Validate plan
        metrics = calculate_response_metrics(plan)
        quality, score = assess_response_quality(metrics)
        result = ValidationResult(
            is_valid=True,
            quality=quality,
            score=score,
            details=metrics
        )
        print_validation_result("Planning", result)
        
        if result.is_valid:
            return plan
        else:
            raise ResponseValidationError("Invalid plan generated")
    
    except PromptValidationError as e:
        print(f"Invalid user prompt: {e}")
        return None
    except ResponseValidationError as e:
        print(f"Invalid plan generated: {e}")
        return None
    except Exception as e:
        print(f"An error occurred during the planning stage: {e}")
        return None

def final_output_stage(user_prompt: str, plan: str) -> Optional[str]:
    """
    Uses gpt-4o to generate the final, detailed response based on the initial prompt and the plan.
    
    Args:
        user_prompt: The user's input prompt
        plan: The plan generated by the planning stage
        
    Returns:
        Final response as a string, or None if final output generation fails
    """
    try:
        print("\n" + "="*50)
        print("FINAL OUTPUT STAGE (using gpt-4o)")
        print("="*50)
        
        final_prompt = [
            {"role": "system", "content": "You are an AI assistant that provides comprehensive and detailed responses to complex tasks."},
            {"role": "user", "content": f"Initial task: {user_prompt}\n\nBased on the following plan, provide a detailed and comprehensive response to the initial task. Ensure your response is well-structured, addresses all aspects of the plan, and completes any truncated points:\n\n{plan}"}
        ]
        print_messages(final_prompt)
        
        start_time = time.time()
        final_response = client.chat.completions.create(
            model=deployment_name,
            messages=final_prompt,
            max_tokens=config.output_max_tokens
        )
        end_time = time.time()
        
        final_content = final_response.choices[0].message.content.strip()
        print("Final Output:")
        print(final_content)
        print(f"\nExecution time: {end_time - start_time:.2f} seconds")
        print(f"Tokens used: {final_response.usage.total_tokens}")
        print()
        
        # Validate final response
        metrics = calculate_response_metrics(final_content)
        quality, score = assess_response_quality(metrics)
        result = ValidationResult(
            is_valid=True,
            quality=quality,
            score=score,
            details=metrics
        )
        print_validation_result("Final Output", result)
        
        if result.is_valid:
            return final_content
        else:
            raise ResponseValidationError("Invalid final response generated")
    
    except ResponseValidationError as e:
        print(f"Invalid final response generated: {e}")
        return None
    except Exception as e:
        print(f"An error occurred during the final output stage: {e}")
        return None

def prompt_chaining(user_prompt: str) -> str:
    """
    Performs a two-stage process: planning with gpt-4o-mini and final output generation with gpt-4o.
    
    Args:
        user_prompt: The user's input prompt
        
    Returns:
        Final response as a string, or error message
    """
    stats = ExecutionStats()
    
    try:
        print(f"\nUser Prompt: {user_prompt}\n")
        
        # Stage 1: Planning with retries
        stats.start_stage("planning")
        plan = None
        for attempt in range(config.max_retries):
            try:
                plan = planning_stage(user_prompt)
                if plan:
                    plan = process_response(plan, config.min_plan_length)
                    break
                stats.increment_retry("planning")
                print(f"Planning attempt {attempt + 1} failed, retrying...")
            except Exception as e:
                error_msg = handle_api_error(e, "planning")
                print(error_msg)
                if attempt < config.max_retries - 1:
                    time.sleep(config.retry_base_delay * (2 ** attempt))
        
        if plan is None:
            return "Failed to generate a valid plan after multiple attempts."
        
        # Stage 2: Final Output Generation with retries
        stats.start_stage("output")
        final_response = None
        for attempt in range(config.max_retries):
            try:
                final_response = final_output_stage(user_prompt, plan)
                if final_response:
                    final_response = process_response(
                        final_response, 
                        config.min_response_length
                    )
                    break
                stats.increment_retry("output")
                print(f"Output generation attempt {attempt + 1} failed, retrying...")
            except Exception as e:
                error_msg = handle_api_error(e, "output generation")
                print(error_msg)
                if attempt < config.max_retries - 1:
                    time.sleep(config.retry_base_delay * (2 ** attempt))
        
        if final_response is None:
            return "Failed to generate valid final output after multiple attempts."
        
        # Print execution summary
        print("\nExecution Summary:")
        print(json.dumps(stats.get_summary(), indent=2))
        
        return final_response

    except Exception as e:
        error_message = f"Critical error during prompt chaining: {e}"
        print(error_message)
        return error_message

# Example usage
if __name__ == "__main__":
    user_prompt = "How should a healthcare organization starting an Intelligent Automation and Generative AI program structure their approach?"
    enhanced_response = prompt_chaining(user_prompt)
    print("\nEnhanced Response:")
    print(enhanced_response)
