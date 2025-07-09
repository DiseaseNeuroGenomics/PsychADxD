import pickle
import numpy as np

class Splits:

    """Code to generate train/test splits, required for model training.
    Splits are chosen so that the same donor is never in both the train and test splits."""

    def __init__(self, metadata_fn: str, save_fn: str, n_splits: int = 10, seed: int = 42):

        np.random.seed(seed)
        self.meta = pickle.load(open(metadata_fn, "rb"))
        self._subset_data()
        self.save_fn = save_fn
        self.n_splits = n_splits
        self.splits = {}

    def _subset_data(self, age_threshold: float = 0.0):

        self.cell_idx = np.where(self.meta["obs"]["Age"] > age_threshold)[0]

    def create_splits(self):

        self.splits = {}
        self.splits_donor = {}
        self._split_by_subjects()
        self._add_indices()

        pickle.dump(self.splits, open(self.save_fn, "wb"))

    def _split_by_subjects(self):

        # we will only include RUSH/MMS subjects in the train set
        subjects = np.unique(self.meta["obs"]["SubID"][self.cell_idx])
        subjects = np.random.permutation(subjects)

        test_subject_splits = np.array_split(subjects, self.n_splits)
        for n in range(self.n_splits):
            train_subject_split = list(set(subjects) - set(test_subject_splits[n]))
            self.splits_donor[n] = {
                "train_subjects": train_subject_split,
                "test_subjects": test_subject_splits[n].tolist(),
            }

    def _add_indices(self):

        for i in range(self.n_splits):

            print(f"Split number {i}")
            self.splits[i] = {}

            idx = [
                n for n in self.cell_idx if
                self.meta["obs"]["SubID"][n] in self.splits_donor[i]["test_subjects"]
            ]
            self.splits[i]["test_idx"] = np.int32(idx)

            idx = [
                n for n in self.cell_idx if
                self.meta["obs"]["SubID"][n] in self.splits_donor[i]["train_subjects"]
            ]
            self.splits[i]["train_idx"] = np.int32(idx)

            print("Size of train subjects/indices, test subjects/indices")
            print((f"Number of train subjects: {len(self.splits_donor[i]['train_subjects'])}, "
                   f"Number of test subjects: {len(self.splits_donor[i]['test_subjects'])}, "
                   f"Number of train indices: {len(self.splits[i]['train_idx'])}, "
                   f"Number of test indices: {len(self.splits[i]['test_idx'])}"))


            idx = set(self.splits_donor[i]["train_subjects"]).intersection(set(self.splits_donor[i]["test_subjects"]))
            print(f"Intersection size between train and test subjects: {len(idx)}")

            idx = set(self.splits[i]["train_idx"]).intersection(set(self.splits[i]["test_idx"]))
            print(f"Intersection size between train and test indices: {len(idx)}")


def create_splits(metadata_fn: str, save_fn: str) -> None:

    splits = Splits(metadata_fn, save_fn)
    splits.create_splits()

