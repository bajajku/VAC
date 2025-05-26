
'''
    Retriever class to create and manage retrievers.
'''

class Retriever:
    def __init__(self, vector_database):
        self.vector_database = vector_database

    def create_retriever(self, **kwargs):
        return self.vector_database.as_retriever(**kwargs)
