import unittest
from unittest.mock import patch, MagicMock

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader

from main import TextSplitter, MainApp

class TestTextSplitter(unittest.TestCase):
    def setUp(self):
        self.text_splitter = TextSplitter()

    def test_split_text_empty_data(self):
        data = ""
        result = self.text_splitter.split_text(data)
        self.assertEqual(result, [])

    @patch('main.UnstructuredURLLoader')
    def test_split_text_with_data(self, mock_data):
      url = 'https://www.mammals-locomotion.com/preface.html'
      loader = UnstructuredURLLoader(urls=url)
      mock_data = loader.load()

      text_splitter = TextSplitter()
      result = text_splitter.split_text(mock_data)

      self.assertEqual(result, [])

# class MainApp(unittest.TestCase):
#   @patch('main.st')
#   @patch('main.time')
#   @patch('main.OpenAIEmbeddings')
#   @patch('main.UnstructuredURLLoader')
#   @patch('main.FAISS')
#   def test_run(...):

if __name__ == '__main__':
    unittest.main()
