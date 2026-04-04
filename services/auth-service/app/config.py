import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=r"C:\OpenHealth\.env", override=True)


class Config:
    SQLALCHEMY_DATABASE_URI: str = os.getenv(
        "DATABASE_URL",
        "postgresql://openhealth:openhealth@postgres:5432/openhealth",
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY: str = os.getenv("JWT_SECRET", "CHANGE-ME-in-production")

    REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379/0")

    JWT_SECRET: str = os.getenv("JWT_SECRET", "CHANGE-ME-in-production")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "15"))
    REFRESH_TOKEN_EXPIRE_DAYS: int = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

    BCRYPT_COST_FACTOR: int = int(os.getenv("BCRYPT_COST_FACTOR", "12"))
