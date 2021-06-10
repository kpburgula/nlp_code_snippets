import os
import requests
import tarfile


class FetchData:
    def __init__(self):
        pass

    def download_data(self, url="http://www.cs.cmu.edu/~dbamman/data/booksummaries.tar.gz"):
        """
        This function downloads the required data to test a text summarization technique
        """
        target_path = 'books.tar.gz'
        if not os.path.isfile(target_path):
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                open(target_path, 'wb').write(response.content)

            # open tar zip file file
            file = tarfile.open(target_path)
            # extracting the tar zip file
            file.extractall('.')
            file.close()
        return True

    def prepare_data(self, data_path):
        my_data = {}
        for line in open(data_path, encoding="utf-8"):
            temp = line.split("\t")
            my_data[temp[2]] = temp[6]

        return my_data

    def get_sample_data(self):
        self.download_data()
        return list(self.prepare_data("booksummaries/booksummaries.txt").values())


class Preprocessing:

    def __init__(self):
        pass

    def preprocess(self):
        print('Base method')
        pass
