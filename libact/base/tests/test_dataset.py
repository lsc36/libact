import unittest

import numpy as np

from libact.base.dataset import Dataset


class TestDatasetMethods(unittest.TestCase):

    initial_X = np.arange(15).reshape((5, 3))
    initial_y = np.array([1, 2, None, 1, None])

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, self.assertNdArrayEqual)

    def assertNdArrayEqual(self, a, b, msg=None):
        return np.array_equal(a, b)

    def setup_dataset(self):
        return Dataset(self.initial_X, self.initial_y)

    def callback(self, entry_id, new_label):
        self.cb_index = entry_id
        self.cb_label = new_label

    def test_len(self):
        dataset = self.setup_dataset()
        self.assertEqual(len(dataset), 5)
        self.assertEqual(dataset.len_labeled(), 3)
        self.assertEqual(dataset.len_unlabeled(), 2)

    def test_get_num_of_labels(self):
        dataset = self.setup_dataset()
        self.assertEqual(dataset.get_num_of_labels(), 2)

    def test_append(self):
        dataset = self.setup_dataset()
        # labeled
        dataset.append(np.array([9, 8, 7]), 2)
        last_labeled_entry = [e[-1] for e in dataset.get_labeled_entries()]
        self.assertEqual(last_labeled_entry[0], np.array([9, 8, 7]))
        self.assertEqual(last_labeled_entry[1], 2)
        # unlabeled
        idx = dataset.append(np.array([8, 7, 6]))
        last_unlabeled_entry = [e[-1] for e in dataset.get_unlabeled_entries()]
        self.assertEqual(last_unlabeled_entry[0], idx)
        self.assertEqual(last_unlabeled_entry[1], np.array([8, 7, 6]))

    def test_update(self):
        dataset = self.setup_dataset()
        dataset.on_update(self.callback)
        idx = dataset.append(np.array([8, 7, 6]))
        dataset.update(idx, 2)
        self.assertEqual(self.cb_index, idx)
        self.assertEqual(self.cb_label, 2)
        last_labeled_entry = [e[-1] for e in dataset.get_labeled_entries()]
        self.assertEqual(last_labeled_entry[0], np.array([8, 7, 6]))
        self.assertEqual(last_labeled_entry[1], 2)

    def test_format_sklearn(self):
        dataset = self.setup_dataset()
        X, y = dataset.format_sklearn()
        self.assertEqual(X, self.initial_X[[0, 1, 3]])
        self.assertEqual(y, self.initial_y[[0, 1, 3]])

    def test_get_labeled_entries(self):
        dataset = self.setup_dataset()
        X, y = dataset.get_labeled_entries()
        self.assertEqual(X[0], np.array([0, 1, 2]))
        self.assertEqual(X[1], np.array([3, 4, 5]))
        self.assertEqual(X[2], np.array([9, 10, 11]))
        self.assertEqual(y[0], 1)
        self.assertEqual(y[1], 2)
        self.assertEqual(y[2], 1)

    def test_get_unlabeled_entries(self):
        dataset = self.setup_dataset()
        _, X = dataset.get_unlabeled_entries()
        self.assertTrue(np.array_equal(X[0], np.array([6, 7, 8])))
        self.assertTrue(np.array_equal(X[1], np.array([12, 13, 14])))

    def test_labeled_uniform_sample(self):
        dataset = self.setup_dataset()
        pool_X, pool_y = dataset.get_labeled_entries()
        # with replacement
        dataset_s = dataset.labeled_uniform_sample(10)
        for entry_s in zip(*dataset_s.get_labeled_entries()):
            for entry in zip(pool_X, pool_y):
                if np.array_equal(entry_s[0], entry[0]) and entry_s[1] == entry[1]:
                    break
            else:
                self.fail()
        # without replacement
        dataset_s = dataset.labeled_uniform_sample(3, replace=False)
        used_indexes = set()
        for entry_s in zip(*dataset_s.get_labeled_entries()):
            for idx, entry in enumerate(zip(pool_X, pool_y)):
                if (
                    np.array_equal(entry_s[0], entry[0]) and entry_s[1] == entry[1]
                    and idx not in used_indexes
                ):
                    used_indexes.add(idx)
                    break
            else:
                self.fail()
        with self.assertRaises(ValueError):
            dataset_s = dataset.labeled_uniform_sample(4, replace=False)


if __name__ == '__main__':
    unittest.main()
