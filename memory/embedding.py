class EmbeddingDatabase:
    def __init__(self, client):
        self.client = client
        self.collections = self.client.create_collection("all-my-documents")
    
    def _add_collection(self, document, metadata=None, id=None):
        self.collections.add(
            documents = ["document"],
            metadatas=[metadata],
            ids=[id]
        )
    
    def get_collections(self, message, n_results=2):
        results = self.collections.query(
            query_texts=[message],
            n_results=n_results,
        )
        
        return results