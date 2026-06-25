"""PrismClient: async client for prism-lora multi-adapter vLLM inference."""

import asyncio
from openai import AsyncOpenAI

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
JUDGE_ADAPTER = "judge"
POET_ADAPTER = "poet"

JUDGE_SYSTEM_PROMPT = (
    "你是一个记忆冲突检测专家。给定旧记忆和新事实，你需要判断它们是否"
    "在同一维度上存在冲突。如果冲突则输出UPDATE并用新事实替换旧记忆，"
    "如果不冲突则输出KEEP让两条记忆共存。请以JSON格式输出："
    "{\"decision\": \"UPDATE/KEEP\", \"reason\": \"...\", "
    "\"updated_memory\": \"...\"}"
)

POET_SYSTEM_PROMPT = (
    "你是一位精通古诗词的创作大师，擅长根据要求创作符合格律和意境的古典诗词。"
    "你的创作严格遵守古典诗词的体裁规范，包括字数、行数和押韵。"
)

BASE_SYSTEM_PROMPT = "你是一个有用的AI助手。"


class PrismClient:
    """Async client for multi-adapter LoRA inference via vLLM."""

    def __init__(self, base_url="http://localhost:8000/v1", api_key="dummy"):
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