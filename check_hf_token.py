#!/usr/bin/env python3
"""Quick check of HF token and MedGemma access."""
from huggingface_hub import HfApi
import os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv('HF_TOKEN')

if not token:
    print('❌ No HF_TOKEN found in .env')
    exit(1)

print(f'Token found: {token[:10]}...')
api = HfApi()

try:
    user = api.whoami(token=token)
    print(f'✅ Logged in as: {user["name"]}')
except Exception as e:
    print(f'❌ Token invalid: {e}')
    exit(1)

try:
    info = api.model_info('google/medgemma-4b-it', token=token)
    print(f'✅ MedGemma 4B accessible (license accepted)')
except Exception as e:
    print(f'⚠️  Cannot access MedGemma: {e}')
    print('   Visit https://huggingface.co/google/medgemma-4b-it and accept the license')
