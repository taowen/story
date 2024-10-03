import unittest
import get_hash_key

def my_func1(key1, _key2, key3):
    pass

class TestFlow(unittest.TestCase):

    def test_get_hash_key(self):
        # _key2 starts with a _ prefix, so it is not included in the hash key
        self.assertEqual(get_hash_key.get_hash_key(my_func1, ('a',), {'_key2': 'b'})[0], 
                         get_hash_key.get_hash_key(my_func1, ('a',), {'_key2': 'c'})[0])

