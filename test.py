from dotenv import load_dotenv
import os
load_dotenv()
print("✅ PGVECTOR_URL =", os.getenv("PGVECTOR_URL"))
print("✅ AUTH_TOKEN =", os.getenv("AUTH_TOKEN"))
