import requests
import base64
import json

def query_fineweb(query, num_docs):
    request_url = f"https://clueweb22.us/fineweb/search?query={query}&k={num_docs}"
    
    response = requests.get(request_url)
    
    if response.status_code != 200:
        raise Exception(f"Error querying FineWeb: {response.status_code}")
    
    json_data = response.json()
    
    output = []

    results = json_data.get("results", [])
    for returned_document in results:
        # Assuming each document in 'results' is a base64 encoded JSON string
        decoded_result = base64.b64decode(returned_document).decode("utf-8")
        parsed_result = json.loads(decoded_result)
        
        text = parsed_result["text"]
        url = parsed_result["url"]
        
        output.append({"text": text, "url": url})
        
    return output