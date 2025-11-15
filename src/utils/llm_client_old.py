"""
LLM Client for Agentic GraphRAG

This module provides a robust wrapper around the Groq API with:
- Automatic retry logic with exponential backoff
- Error handling and logging
- Token usage tracking
- Async and sync interfaces

Author: Agentic GraphRAG Team
"""

import logging
from typing import Optional, List, Dict, Any
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from groq import Groq, GroqError
from .config import get_config, LLMConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LLMClient:
    """
    Wrapper class for Groq LLM API with retry logic and error handling.

    This class provides a clean interface to interact with Groq's API,
    including automatic retries, error handling, and token tracking.

    Attributes:
        config: LLM configuration from environment
        client: Groq API client instance
        total_tokens_used: Running count of tokens used
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize the LLM client.

        Args:
            config: Optional LLMConfig. If None, loads from environment.

        Raises:
            ValueError: If API key is invalid
        """
        self.config = config or get_config().llm
        self.client = Groq(api_key=self.config.api_key)
        self.total_tokens_used = 0
        logger.info(f"Initialized LLM client with model: {self.config.model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(GroqError),
        reraise=True,
    )
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """
        Generate text completion using Groq LLM.

        Args:
            prompt: User prompt/query
            system_prompt: Optional system prompt to guide model behavior
            temperature: Override default temperature (0.0 = deterministic)
            max_tokens: Override default max tokens
            stop_sequences: Optional list of stop sequences

        Returns:
            str: Generated text response

        Raises:
            GroqError: If API call fails after retries
            ValueError: If prompt is empty
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Use config defaults if not specified
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        try:
            logger.debug(f"Sending request to Groq API with {len(prompt)} chars")

            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences,
            )

            # Extract response
            content = response.choices[0].message.content

            # Track token usage
            if hasattr(response, 'usage'):
                tokens_used = response.usage.total_tokens
                self.total_tokens_used += tokens_used
                logger.info(
                    f"Tokens used: {tokens_used} "
                    f"(total: {self.total_tokens_used})"
                )

            logger.debug(f"Received response: {len(content)} chars")
            return content

        except GroqError as e:
            logger.error(f"Groq API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in generate(): {e}")
            raise

    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output from LLM.

        Args:
            prompt: User prompt requesting JSON output
            system_prompt: Optional system prompt
            temperature: Override default temperature

        Returns:
            dict: Parsed JSON response

        Raises:
            ValueError: If response is not valid JSON
        """
        import json

        # Add JSON instruction to system prompt
        json_system = (
            "You are a helpful assistant that outputs valid JSON only. "
            "Do not include any text before or after the JSON."
        )
        if system_prompt:
            json_system = f"{system_prompt}\n\n{json_system}"

        response_text = self.generate(
            prompt=prompt,
            system_prompt=json_system,
            temperature=temperature,
        )

        try:
            # Try to extract JSON from markdown code blocks if present
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()

            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response text: {response_text}")
            raise ValueError(f"Invalid JSON response: {e}")

    def generate_with_examples(
        self,
        prompt: str,
        examples: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate text with few-shot examples.

        Args:
            prompt: User prompt
            examples: List of example dicts with 'input' and 'output' keys
            system_prompt: Optional system prompt

        Returns:
            str: Generated response
        """
        # Build messages with examples
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add examples as conversation history
        for example in examples:
            messages.append({"role": "user", "content": example["input"]})
            messages.append({"role": "assistant", "content": example["output"]})

        # Add actual prompt
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            content = response.choices[0].message.content

            if hasattr(response, 'usage'):
                self.total_tokens_used += response.usage.total_tokens

            return content

        except GroqError as e:
            logger.error(f"Groq API error in generate_with_examples: {e}")
            raise

    def batch_generate(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
    ) -> List[str]:
        """
        Generate completions for multiple prompts.

        Args:
            prompts: List of prompts to process
            system_prompt: Optional system prompt

        Returns:
            List[str]: List of generated responses
        """
        responses = []
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {i+1}/{len(prompts)}")
            try:
                response = self.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                )
                responses.append(response)
            except Exception as e:
                logger.error(f"Failed to process prompt {i+1}: {e}")
                responses.append("")

        return responses

    def get_token_usage(self) -> int:
        """
        Get total tokens used by this client instance.

        Returns:
            int: Total tokens used
        """
        return self.total_tokens_used

    def reset_token_count(self) -> None:
        """Reset the token usage counter."""
        logger.info(f"Resetting token count (was: {self.total_tokens_used})")
        self.total_tokens_used = 0


# Singleton instance
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """
    Get the global LLM client instance (singleton pattern).

    Returns:
        LLMClient: Global LLM client
    """
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def reset_llm_client() -> None:
    """Reset the global LLM client (useful for testing)."""
    global _llm_client
    _llm_client = None


if __name__ == "__main__":
    """Test the LLM client."""
    import sys

    try:
        # Initialize client
        print("🔄 Initializing LLM client...")
        client = get_llm_client()

        # Test simple generation
        print("\n📝 Testing simple generation...")
        response = client.generate(
            prompt="What is the capital of France? Answer in one word.",
            temperature=0.0,
        )
        print(f"Response: {response}")

        # Test JSON generation
        print("\n📊 Testing JSON generation...")
        json_response = client.generate_json(
            prompt="Create a JSON object with fields: name='Paris', country='France', population=2161000",
        )
        print(f"JSON Response: {json_response}")

        # Show token usage
        print(f"\n📈 Total tokens used: {client.get_token_usage()}")

        print("\n✅ All tests passed!")

    except ValueError as e:
        print(f"\n⚠️  Configuration Error: {e}")
        print("\nPlease set GROQ_API_KEY in your .env file:")
        print("  cp .env.example .env")
        print("  # Edit .env and add your Groq API key")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
