import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    # Supabase Postgres connection string (psycopg2 sync driver)
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:[PASSWORD]@db.[PROJECT_REF].supabase.co:5432/postgres",
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = os.getenv("JWT_SECRET", "CHANGE-ME-in-production")

    UPSTASH_REDIS_REST_URL = os.getenv("UPSTASH_REDIS_REST_URL", "")
    UPSTASH_REDIS_REST_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")

    JWT_SECRET = os.getenv("JWT_SECRET", "CHANGE-ME-in-production")
    JWT_ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "15"))
    REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

    BCRYPT_COST_FACTOR = int(os.getenv("BCRYPT_COST_FACTOR", "12"))
