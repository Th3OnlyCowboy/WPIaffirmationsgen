# https://serpapi.com/search?api_key = 'd88ed400868a960b8a44f0e151aa3a8b390a0c66dd8396c5d22a46bf92e35c17'

from serpapi.google_search_results import GoogleSearchResults

searchParams = {
    "engine": "google",
    "q": "Coffee",
    "location": "Austin, Texas, United States",
    "google_domain": "google.com",
    "gl": "us",
    "hl": "en",
    "api_key": "API_SECRET_KEY"
}

client = GoogleSearchResults(searchParams)
results = client.get_dict()