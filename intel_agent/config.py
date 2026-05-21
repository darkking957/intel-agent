from pydantic import Field,HttpUrl
from pydantic_settings import BaseSettings,SettingsConfigDict
from pathlib import Path

_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_PATH,
        env_file_encoding="utf-8",
        case_sensitive=False, # 大小写不敏感
        extra="ignore",       # 忽略 .env 里多余的变量
    )

    deepseek_api_key: str = Field(..., description="DeepSeek API Key，必填")
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"

    # 备用：Anthropic
    anthropic_api_key: str | None = Field(default=None, description="Anthropic API Key，可选")

    # LangSmith
    langsmith_api_key: str | None = None
    langsmith_project: str = "intel-agent"

    # 数据库
    database_url: str = "postgresql://postgres:secret@localhost:5432/intel"

    # 应用配置
    log_level: str = "INFO"
    max_concurrent_llm_calls: int = 5  # asyncio.Semaphore 上限

settings = Settings()