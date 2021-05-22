from elasticsearch import Elasticsearch
import requests
import tarfile

# On Windows/Ubuntu try to download the elasticsearch application and run it as a service
# elastic search instance has to be running on the machine. Default port is 9200.
# Call the Elastic Search instance
# Download the client library using pip for python and use it
# Use https://www.oreilly.com/library/view/elasticsearch-the-definitive/9781449358532/
# to explore more on Elasticsearch

elastic_search = Elasticsearch([{'host': 'localhost', 'port': 9200}])


# delete any pre-existing index
# if es.indices.exists(index="myindex"):
#     es.indices.delete(index='myindex', ignore=[400, 404])

def get_file_name():

    """
    This function downloads the required data to test elasticsearch and searching mechanism
    """
    url = 'http://www.cs.cmu.edu/~dbamman/data/booksummaries.tar.gz'
    target_path = 'books.tar.gz'

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        open(target_path, 'wb').write(response.content)

    # open tar zip file file
    file = tarfile.open(target_path)
    # extracting the tar zip file
    file.extractall('.')
    file.close()
    return True

class Indexing:

    def __init__(self, path, elastic_instance):
        self.path = path
        self.elastic_instance = elastic_instance
        self.res = None

    def build_index(self,index_count):
        path = self.path
        count = 1
        for line in open(path, encoding='utf-8'):
            try:
                fields = line.split("\t")
                doc = {'id': fields[0],
                       'title': fields[2],
                       'author': fields[3],
                       'summary': fields[6]
                       }

                self.res = self.elastic_instance.index(index="myindex", id=fields[0], body=doc)
                count = count + 1
                if count == index_count:
                    break
            except:
                pass

    def search_query(self, query):
        result = self.elastic_instance.search(index="myindex", body={"query": {"match": {"summary": query}}})
        print("Your search returned %d results." % result['hits']['total']['value'])
        return result


if __name__ == '__main__':
    file_name = get_file_name()
    instance = Indexing("booksummaries/booksummaries.txt", elastic_search)
    instance.build_index(index_count= 1000)

    while True:
        query = input("Enter your search query: ")
        if query.lower() == "stop":
            break
        res = instance.search_query(query)
        print("Your search returned %d results:" % res['hits']['total']['value'])
        for hit in res["hits"]["hits"]:
            print(hit["_source"]["title"])
            loc = hit["_source"]["summary"].lower().index(query)
            print(hit["_source"]["summary"][:100])
            print(hit["_source"]["summary"][loc - 100:loc + 100])
