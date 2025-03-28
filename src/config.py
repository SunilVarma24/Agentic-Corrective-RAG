# src/config.py

import os
import yaml

# Set up Gemini & Tavily Search API keys

with open("gemini_key_gtm.yaml", 'r') as file:
    api_creds = yaml.safe_load(file)
os.environ["GOOGLE_API_KEY"] = api_creds['gemini_key']

with open("tavily_key.yaml", 'r') as file:
    api_creds = yaml.safe_load(file)
os.environ["TAVILY_API_KEY"] = api_creds['tavily_key']
