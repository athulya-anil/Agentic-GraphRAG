"""
Multi-Provider LLM Client for Agentic GraphRAG

This module provides a unified interface for multiple LLM providers:
- Google Gemini (gemini-2.0-flash-exp)
- Groq (llama-3.3-70b-versatile)

Features:
- Automatic retry logic with exponential backoff
- Error handling and logging
- Token usage tracking
- Provider-agnostic interface
- Caching support

Author: Agentic GraphRAG Team
"""

import logging
import os
import hashlib
import json
from typing import Optional, List, Dict, Any, Literal
from abc import ABC, abstractmethod
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.total_tokens_used = 0

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> tuple[str, int]:
        """
        Generate completion from messages.

        Returns:
            tuple: (response_text, tokens_used)
        """
        pass


class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider implementation."""

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        super().__init__(api_key, model, temperature, max_tokens)

        try:
            import google.generativeai as genai
            self.genai = genai
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model)
            logger.info(f"Initialized Gemini provider with model: {model}")
        except ImportError:
            raise ImportError(
                "google-generativeai not installed. "
                "Install it with: pip install google-generativeai"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini: {e}")

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> tuple[str, int]:
        """Generate completion using Gemini."""
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        try:
            # Convert messages to Gemini format
            prompt_parts = []
            for msg in messages:
                role = msg["role"]
                content = msg["content"]

                if role == "system":
                    prompt_parts.append(f"Instructions: {content}\n")
                elif role == "user":
                    prompt_parts.append(f"User: {content}\n")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}\n")

            prompt = "\n".join(prompt_parts)

            # Generate with Gemini
            generation_config = {
                "temperature": temp,
                "max_output_tokens": max_tok,
            }

            response = self.client.generate_content(
                prompt,
                generation_config=generation_config
            )

            content = response.text

            # Estimate tokens (Gemini doesn't always provide usage)
            # Rough estimate: 1 token ≈ 4 chars
            tokens_used = (len(prompt) + len(content)) // 4

            self.total_tokens_used += tokens_used

            return content, tokens_used

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise


class GroqProvider(BaseLLMProvider):
    """Groq provider implementation."""

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        super().__init__(api_key, model, temperature, max_tokens)

        try:
            from groq import Groq
            self.client = Groq(api_key=api_key)
            logger.info(f"Initialized Groq provider with model: {model}")
        except ImportError:
            raise ImportError(
                "groq not installed. Install it with: pip install groq"
            )

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> tuple[str, int]:
        """Generate completion using Groq."""
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                max_tokens=max_tok,
            )

            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else 0

            self.total_tokens_used += tokens_used

            return content, tokens_used

        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise


class UnifiedLLMClient:
    """
    Unified LLM client with multi-provider support and caching.

    Supports:
    - Google Gemini (gemini, gemini-2.0-flash-exp)
    - Groq (llama-3.3-70b-versatile)

    Features:
    - Automatic provider selection
    - Response caching
    - Token tracking
    - Retry logic
    """

    def __init__(
        self,
        provider: Literal["gemini", "groq"] = "gemini",
        enable_cache: bool = True,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize unified LLM client.

        Args:
            provider: LLM provider to use
            enable_cache: Enable response caching
            cache_dir: Directory for cache storage
        """
        self.provider_name = provider
        self.enable_cache = enable_cache
        self.cache_dir = cache_dir or Path("data/cache/llm")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize provider
        self.provider = self._initialize_provider(provider)

        logger.info(f"Initialized UnifiedLLMClient with provider: {provider}")

    def _initialize_provider(self, provider: str) -> BaseLLMProvider:
        """Initialize the specified provider."""
        if provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY", "")
            model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
            temperature = float(os.getenv("GEMINI_TEMPERATURE", "0.0"))
            max_tokens = int(os.getenv("GEMINI_MAX_TOKENS", "2048"))

            if not api_key:
                raise ValueError("GEMINI_API_KEY not set in environment")

            return GeminiProvider(api_key, model, temperature, max_tokens)

        elif provider == "groq":
            api_key = os.getenv("GROQ_API_KEY", "")
            model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            temperature = float(os.getenv("GROQ_TEMPERATURE", "0.0"))
            max_tokens = int(os.getenv("GROQ_MAX_TOKENS", "2048"))

            if not api_key:
                raise ValueError("GROQ_API_KEY not set in environment")

            return GroqProvider(api_key, model, temperature, max_tokens)

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _get_cache_key(self, messages: List[Dict[str, str]], temperature: float) -> str:
        """Generate cache key from messages and temperature."""
        cache_input = json.dumps(messages, sort_keys=True) + str(temperature)
        return hashlib.md5(cache_input.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[str]:
        """Retrieve response from cache."""
        if not self.enable_cache:
            return None

        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                    logger.debug(f"Cache hit: {cache_key}")
                    return cached["response"]
            except Exception as e:
                logger.warning(f"Failed to read cache: {e}")

        return None

    def _save_to_cache(self, cache_key: str, response: str) -> None:
        """Save response to cache."""
        if not self.enable_cache:
            return

        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump({"response": response}, f)
            logger.debug(f"Cached response: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate text completion.

        Args:
            prompt: User prompt/query
            system_prompt: Optional system prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            str: Generated text response
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Check cache
        temp = temperature if temperature is not None else self.provider.temperature
        cache_key = self._get_cache_key(messages, temp)

        cached_response = self._get_from_cache(cache_key)
        if cached_response:
            return cached_response

        # Generate new response
        try:
            logger.debug(f"Sending request to {self.provider_name} with {len(prompt)} chars")

            response, tokens_used = self.provider.generate(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            logger.info(
                f"Tokens used: {tokens_used} "
                f"(total: {self.provider.total_tokens_used})"
            )

            # Cache response
            self._save_to_cache(cache_key, response)

            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output.

        Args:
            prompt: User prompt requesting JSON output
            system_prompt: Optional system prompt
            temperature: Override default temperature

        Returns:
            dict: Parsed JSON response
        """
        import json

        # Add JSON instruction
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
            # Extract JSON from markdown if present
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
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"Response: {response_text}")
            raise ValueError(f"Invalid JSON response: {e}")

    def batch_generate(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
    ) -> List[str]:
        """
        Generate completions for multiple prompts.

        Args:
            prompts: List of prompts
            system_prompt: Optional system prompt

        Returns:
            List[str]: List of responses
        """
        responses = []
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {i+1}/{len(prompts)}")
            try:
                response = self.generate(prompt=prompt, system_prompt=system_prompt)
                responses.append(response)
            except Exception as e:
                logger.error(f"Failed to process prompt {i+1}: {e}")
                responses.append("")

        return responses

    def get_token_usage(self) -> int:
        """Get total tokens used."""
        return self.provider.total_tokens_used

    def reset_token_count(self) -> None:
        """Reset token usage counter."""
        logger.info(f"Resetting token count (was: {self.provider.total_tokens_used})")
        self.provider.total_tokens_used = 0

    def clear_cache(self) -> None:
        """Clear the response cache."""
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            logger.info("Cache cleared")


# Singleton instance
_unified_client: Optional[UnifiedLLMClient] = None


def get_llm_client(provider: Optional[str] = None, enable_cache: bool = True) -> UnifiedLLMClient:
    """
    Get the global unified LLM client instance (singleton).

    Args:
        provider: Override provider (gemini or groq)
        enable_cache: Enable response caching

    Returns:
        UnifiedLLMClient: Global LLM client
    """
    global _unified_client

    if provider:
        # Override provider
        return UnifiedLLMClient(provider=provider, enable_cache=enable_cache)

    if _unified_client is None:
        # Use provider from environment
        default_provider = os.getenv("LLM_PROVIDER", "gemini")
        _unified_client = UnifiedLLMClient(provider=default_provider, enable_cache=enable_cache)

    return _unified_client


def reset_llm_client() -> None:
    """Reset the global LLM client."""
    global _unified_client
    _unified_client = None


if __name__ == "__main__":
    """Test the unified LLM client."""
    import sys

    try:
        # Test Gemini
        print("🔄 Testing Gemini provider...")
        gemini_client = get_llm_client(provider="gemini")

        response = gemini_client.generate(
            prompt="What is the capital of France? Answer in one word.",
            temperature=0.0,
        )
        print(f"Gemini Response: {response}")
        print(f"Tokens used: {gemini_client.get_token_usage()}")

        # Test Groq
        print("\n🔄 Testing Groq provider...")
        groq_client = get_llm_client(provider="groq")

        response = groq_client.generate(
            prompt="What is 2+2? Answer with just the number.",
            temperature=0.0,
        )
        print(f"Groq Response: {response}")
        print(f"Tokens used: {groq_client.get_token_usage()}")

        # Test caching
        print("\n🔄 Testing cache...")
        response2 = gemini_client.generate(
            prompt="What is the capital of France? Answer in one word.",
            temperature=0.0,
        )
        print(f"Cached response: {response2}")

        print("\n✅ All tests passed!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
