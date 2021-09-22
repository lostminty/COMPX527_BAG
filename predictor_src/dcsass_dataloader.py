from torchvision.datasets import VisionDataset
from torchvision import transforms
import re,glob,os,pickle,sys,csv
import numpy as np
from sklearn import preprocessing
import torch
class DCSASS(VisionDataset):
    def __init__(self, directory, transform, pickled_dirname="pickle_jar"):
        super().__init__(directory)

        self.directory = directory
        self.transform = transform
        self.video_files = []
        self.labels = []
        self.positives = []

        if not self.transform:
            raise ValueError(
                "A transformer is required for outputting tensors.")

        regex = re.compile(r".*_x264")
        for file in glob.glob(directory + "/Labels/*.csv"):
            for row in csv.reader(open(file)):
                if not row:
                    continue

                file_base = regex.match(row[0])
                file_base = file_base.group(0) if file_base else None
                if not file_base:
                    continue

                if file_base[0:4] == "oadA":  # Correct a bad file path.
                    file_base = "R" + file_base
                    row[0] = "R" + row[0]

                file_path = f"{directory}/{row[1]}/{file_base}.mp4/{row[0]}.mp4"
                if not os.path.exists(file_path):
                    continue

                self.video_files.append(file_path)
                #self.positives.append(bool(int(row[2])) if row[2] else False)
                self.labels.append(row[1] if row[2] else "normal")

        self.unique_labels = np.unique(self.labels)
        self.encoder = preprocessing.LabelEncoder()
        self.encoder.fit(self.unique_labels)

        self.cur_label = np.array([0] * len(self.unique_labels))
        self.pickled_dirname = pickled_dirname
        if not os.path.exists(pickled_dirname):
            os.mkdir(pickled_dirname)

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, index):
        label = self.labels[index]

        # Avoided unneeded conditional checks for performance reasons, as this
        # is where the bottleneck is.
        # Comment out the pickling block below if you want to disable pickling
        # and instead uncomment the directly below line.
        # video = self.transform(self.video_files[index])

        pickled = f"{self.pickled_dirname}/{index}"
        if os.path.exists(pickled):
            video = pickle.load(open(pickled, "rb"))
        else:
            video = self.transform(self.video_files[index])
            pickle.dump(video, open(pickled, "wb"))

        label_index = self.encoder.transform([label])[0]
        return video, self.label_formatter(label_index, len(self.unique_labels))


    def label_formatter(self,label, num_of_classes):
        label_encoded = [0] * num_of_classes
        label_encoded[label] += 1
        return torch.from_numpy(np.array(label_encoded))

