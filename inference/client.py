"""PrismClient: async client for prism-lora multi-adapter vLLM inference."""

import asyncio
from openai import AsyncOpenAI

from configs.config import (
    BASE_MODEL,
    JUDGE_ADAPTER,
    POET_ADAPTER,
    JUDGE_SYSTEM_PROMPT,
    POET_SYSTEM_PROMPT,
    BASE_SYSTEM_PROMPT,
    VLLM_BASE_URL,
)


class PrismClient:
    """Async client for multi-adapter LoRA inference via vLLM."""

    def __init__(self, base_url=VLLM_BASE_URL, api_key="dummy"):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def judge(self, old_memory: str, new_fact: str, temperature: float = 0.1) -> str:
        """Send a memory conflict query to the Judge LoRA adapter."""
        user_msg = f"旧记忆：{old_memory}\n新事实：{new_fact}\n请判断新事实与旧记忆的关系，并决定处理策略。"
        response = await self.client.chat.completions.create(
            model=JUDGE_ADAPTER,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=temperature,
            max_tokens=256,
        )
        return response.choices[0].message.content

    async def poet(self, prompt: str, temperature: float = 0.7) -> str:
        """Send a poetry generation query to the Poet LoRA adapter."""
        response = await self.client.chat.completions.create(
            model=POET_ADAPTER,
            messages=[
                {"role": "system", "content": POET_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=256,
        )
        return response.choices[0].message.content

    async def base_model(self, prompt: str, temperature: float = 0.1) -> str:
        """Send a query to the base model without any LoRA adapter."""
        response = await self.client.chat.completions.create(
            model=BASE_MODEL,
            messages=[
                {"role": "system", "content": BASE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=256,
        )
        return response.choices[0].message.content

    async def list_models(self) -> list:
        """List available models on the vLLM server."""
        models = await self.client.models.list()
        return [m.id for m in models.data]

    # Synchronous wrappers for convenience
    def judge_sync(self, old_memory: str, new_fact: str, temperature: float = 0.1) -> str:
        return asyncio.run(self.judge(old_memory, new_fact, temperature))

    def poet_sync(self, prompt: str, temperature: float = 0.7) -> str:
        return asyncio.run(self.poet(prompt, temperature))

    def base_model_sync(self, prompt: str, temperature: float = 0.1) -> str:
        return asyncio.run(self.base_model(prompt, temperature))


if __name__ == "__main__":
    async def demo():
        client = PrismClient()
        models = await client.list_models()
        print(f"Available models: {models}")

        # Judge: conflict case
        result = await client.judge("张三喜欢吃苹果", "张三不喜欢吃苹果")
        print(f"\n[Judge - conflict] {result}")

        # Judge: no conflict case
        result = await client.judge("张三喜欢吃苹果", "张三喜欢吃香蕉")
        print(f"\n[Judge - no conflict] {result}")

        # Poet
        result = await client.poet("写一首关于秋天的七言绝句，风格要求：意境深远。")
        print(f"\n[Poet] {result}")

    asyncio.run(demo())