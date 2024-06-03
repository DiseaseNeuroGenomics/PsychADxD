import pickle
import numpy as np

class Splits:

    def __init__(self, metadata_fn: str, save_fn: str, n_splits: int = 10):

        self.meta = pickle.load(open(metadata_fn, "rb"))
        self._subset_data()
        self.save_fn = save_fn
        self.n_splits = n_splits
        self.splits = {}

    def _subset_data(self, age_threshold: float = 65.0):

        # idx = self.meta["obs"]["Age"] >= age_threshold
        self.cell_idx = np.where(self.meta["obs"]["include_training"])[0]

    def create_splits(self):

        self.splits = {}
        self._split_by_subjects()
        self._add_indices()
        pickle.dump(self.splits, open(self.save_fn, "wb"))

    def _split_by_subjects(self):

        # we will only include RUSH/MMS subjects in the train set
        subjects = np.unique(self.meta["obs"]["SubID"][self.cell_idx])
        braak = []
        dementia = []
        for s in subjects:
            idx = np.where(np.array(self.meta["obs"]["SubID"]) == s)[0]
            braak.append(self.meta["obs"]["BRAAK_AD"][idx][0])
            dementia.append(self.meta["obs"]["Dementia"][idx][0])


        progress = False
        while not progress:

            idx_rand = np.random.permutation(len(subjects))
            subjects = subjects[idx_rand]
            braak = np.array(braak)[idx_rand]
            dementia = np.array(dementia)[idx_rand]

            braak_splits = np.array_split(braak, self.n_splits)
            dementia_splits = np.array_split(dementia, self.n_splits)

            counts = []
            for n in range(10):
                idx = np.where((braak_splits[n] == 6) * (dementia_splits[n] == 0))[0]
                print(n, len(idx), len(braak_splits[n]))
                counts.append(len(idx))

            if np.max(np.array(counts)) <= 1:
                progress = True

        test_subject_splits = np.array_split(subjects, self.n_splits)
        for n in range(self.n_splits):
            train_subject_split = list(set(subjects) - set(test_subject_splits[n]))
            self.splits[n] = {
                "train_subjects": train_subject_split,
                "test_subjects": test_subject_splits[n].tolist(),
            }

    def _add_indices(self):

        for i in range(self.n_splits):

            print(f"Split number {i}")

            idx = [
                n for n in self.cell_idx if
                self.meta["obs"]["SubID"][n] in self.splits[i]["test_subjects"]
            ]
            self.splits[i]["test_idx"] = idx

            idx = [
                n for n in self.cell_idx if
                self.meta["obs"]["SubID"][n] in self.splits[i]["train_subjects"]
            ]
            self.splits[i]["train_idx"] = idx

            print("Size of train subjects/indices, test subjects/indices")
            print((f"Number of train subjects: {len(self.splits[i]['train_subjects'])}, "
                   f"Number of test subjects: {len(self.splits[i]['test_subjects'])}, "
                   f"Number of train indices: {len(self.splits[i]['train_idx'])}, "
                   f"Number of test indices: {len(self.splits[i]['test_idx'])}"))


            idx = set(self.splits[i]["train_subjects"]).intersection(set(self.splits[i]["test_subjects"]))
            print(f"Intersection size between train and test subjects: {len(idx)}")

            idx = set(self.splits[i]["train_idx"]).intersection(set(self.splits[i]["test_idx"]))
            print(f"Intersection size between train and test indices: {len(idx)}")


def create_splits(metadata_fn: str, save_fn: str) -> None:

    splits = Splits(metadata_fn, save_fn)
    splits.create_splits()

