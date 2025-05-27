import json
from langchain_core.documents import Document 

class JsonParser:
    def __init__(self, json_file: str) :
        self.json_file = json_file
        self.documents = []

    def parse_json(self):
        with open(self.json_file, 'r') as file:
            data = json.load(file)
            for key, item in data.items():
                self.documents.append(Document(page_content=item['text_content'],\
                                                metadata={'url': key, 'title': item['title'], 'description': item['description']}))
 
if __name__ == "__main__":
    parser = JsonParser("/Users/kunalbajaj/VAC/backend/scripts/data_collection/crawl_results/crawl_results_20250526_133954.json")
    parser.parse_json()
    print(parser.documents[0])