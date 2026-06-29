import os
import requests
from bs4 import BeautifulSoup
from langchain.tools import tool
from tavily import TavilyClient

@tool
def web_search(query: str) -> str:
    """
    Performs a general web search using Tavily.
    Returns the top 3 search results formatted as a string containing title, URL, and a snippet.
    """
    try:
        api_key = os.environ.get('TAVILY_API_KEY')
        if not api_key:
            return "Search unavailable: TAVILY_API_KEY not set in environment."
            
        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, search_depth="basic", max_results=3)
        
        results_str = ""
        for i, result in enumerate(response.get('results', [])):
            results_str += f"{i+1}. {result.get('title')}\nURL: {result.get('url')}\nSnippet: {result.get('content')}\n\n"
            
        return results_str.strip() if results_str else "No results found."
    except Exception:
        return 'Search unavailable'

@tool
def indian_kanoon_search(query: str) -> str:
    """
    Searches Indian Kanoon for Indian case laws and acts.
    Returns the top 5 case/act titles and their URLs.
    """
    try:
        url = f'https://indiankanoon.org/search/?formInput={query}'
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        results = soup.find_all('div', class_='result_title')
        
        results_str = ""
        for i, res in enumerate(results[:5]):
            link = res.find('a')
            if link:
                title = link.text.strip()
                href = "https://indiankanoon.org" + link.get('href')
                results_str += f"{i+1}. {title}\nURL: {href}\n\n"
                
        return results_str.strip() if results_str else "No results found on Indian Kanoon."
    except Exception:
        return 'Search unavailable'
