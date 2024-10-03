import unittest
import get_hash_key

def my_func1(key1: str, _key2: str, key3: str) -> None:
    pass

class TestFlow(unittest.TestCase):

    def test_get_hash_key(self):
        # _key2 starts with a _ prefix, so it is not included in the hash key
        self.assertEqual(get_hash_key.get_hash_key(my_func1, ('a', 'b', 'a'), {})[0], 
                         get_hash_key.get_hash_key(my_func1, ('a', 'c', 'a'), {})[0])
