import os
import numpy as np
import pandas as pd
import glob
from typing import List, Dict, Tuple
from ..utils import download_url, unzip_file
import gzip
import shutil
from pathlib import Path
from tqdm.contrib.concurrent import thread_map
from typing import Union

class ExtraSensoryUser:
    def __init__(self, dataset, uuid: str, feature_file: Path, user_accelerometer_dir: Path,
                 user_gyroscope_dir: Path, store_raw_df: bool = False):
        self.uuid = uuid
        self.user_accelerometer_dir = user_accelerometer_dir
        self.user_gyroscope_dir = user_gyroscope_dir
        self.feature_path = feature_file
        self.dataset = dataset
        self.feature_df = None 
        self.feature_names =  None
        self.label_names = None
        self._raw_df = None
        self.store_raw_df = store_raw_df
        self.parse_features()

    def parse_features(self):
        self.feature_df = pd.read_csv(self.feature_path, compression='gzip').astype({"timestamp": int})
        self.feature_df.set_index("timestamp", inplace=True)
        # read features
        self.feature_names = sorted([
            col for col in self.feature_df.columns 
            if not col.startswith("label") and not col.startswith("timestamp")
        ])
        # read and rename labels
        renames = {
            col: col.replace("label:", "") for col in self.feature_df.columns 
            if col.startswith("label:")
        }
        self.feature_df.rename(columns=renames, inplace=True)
        self.label_names = sorted(list(renames.values()))
        self.feature_df[self.label_names] = self.feature_df[self.label_names].fillna(0).astype(bool)
        # reorder
        all_columns = [col for col in self.feature_df.columns if col not in self.label_names] + self.label_names
        self.feature_df = self.feature_df[all_columns]

    def num_samples(self):
        return len(self.metadata["timestamps"])

    @property
    def raw_measurements(self) -> pd.DataFrame:
        # read only once if store_raw_df is True
        if self._raw_df is not None:
            return self._raw_df
        
        def read_raw(timestamp):
            try:
                acc_file = self.user_accelerometer_dir / f"{timestamp}.m_raw_acc.dat"
                acc_data = pd.read_csv(acc_file, header=None, sep=" ",
                                        names=["accelerometer timestamp", "accelerometer-x",
                                        "accelerometer-y", "accelerometer-z"])
                gyro_file = self.user_gyroscope_dir / f"{timestamp}.m_proc_gyro.dat"
                gyro_data = pd.read_csv(gyro_file, header=None, sep=" ",
                                        names=["gyroscope timestamp", "gyroscope-x",
                                        "gyroscope-y", "gyroscope-z"])
                # labels = self.feature_df.loc[[timestamp], self.label_names]
                data = pd.concat([acc_data, gyro_data], axis=1)
                data["timestamp source"] = timestamp
                labels = self.dataset.user_map[self.uuid].feature_df.loc[[timestamp], self.dataset.user_map[self.uuid].label_names]
                for i in labels.columns:
                    data[i] = labels[i].to_list()*len(data)
                data = data.dropna()
                return data
            except FileNotFoundError:
                return None

        res = thread_map(read_raw, list(self.feature_df.index), desc=f"Reading files from user {self.uuid}...")
        res = [r for r in res if r is not None]
        df =  pd.concat(res).sort_values(by=["timestamp source", "accelerometer timestamp", "gyroscope timestamp"])
        # Just reordering columns. Put "timestamp source" at the beggining
        columns = df.columns.to_list()
        columns = columns[-1:] + columns[:-1]
        df = df[columns]

        if self.store_raw_df:
            self._raw_df = df

        return df

    # For now, suppose 40Hz so 3s window will have 120 samples
    def time_window(self, window_len: int = 120, overlap: int = 0, interpolation: str = None, merge_classes_label: bool = True) -> pd.DataFrame:
        assert overlap < window_len, "Overlap must be less than window length"
        df =  self.raw_measurements
        to_select = [
            "accelerometer-x", "accelerometer-y", "accelerometer-z", 
            "gyroscope-x", "gyroscope-y", "gyroscope-z"
        ]
        other_columns =  [c for c in df.columns if c not in to_select]
        column_names = [
            f"{c}-{i}"
            for i in range(window_len) for c in to_select
        ] + other_columns

        print(f"Generating time frames")

        dfs = []
        for timestamp, grouped_df in df.groupby(by="timestamp source"):
            rows = []
            for i in range(0, len(grouped_df), window_len-overlap):
                window = grouped_df.iloc[i:(i+window_len)]
                if len(window) != window_len:
                    continue
                values = window[to_select].to_numpy().flatten()
                labels = window[other_columns].values[0].flatten()
                rows.append(np.concatenate([values, labels], axis=0))
            dfs.append(pd.DataFrame(rows, columns=column_names))
        df = pd.concat(dfs)

        reordered_column_names = [
            f"{c}-{i}" for c in to_select for i in range(window_len) 
        ] + other_columns

        df = df[reordered_column_names]

        if merge_classes_label:
            merged_labels = []
            for i, row in df[self.label_names].iterrows():
                true_labels = [k for k in sorted(row.keys()) if row[k]]
                true_labels = ", ".join(true_labels)
                merged_labels.append(true_labels)
            df["merged label"] = merged_labels

        return df

class ExtraSensoryDataset:
    # Version 2017 ExtraSensory
    acc_url = "http://extrasensory.ucsd.edu/data/raw_measurements/ExtraSensory.raw_measurements.raw_acc.zip"
    gyr_url = "http://extrasensory.ucsd.edu/data/raw_measurements/ExtraSensory.raw_measurements.proc_gyro.zip"
    label_url = "http://extrasensory.ucsd.edu/data/primary_data_files/ExtraSensory.per_uuid_features_labels.zip"

    # Activity names and codes
    activity_names = {
        0: "OR_indoors",
        1: "LOC_home",
        2: "SITTING",
        3: "PHONE_ON_TABLE",
        4: "LYING_DOWN",
        5: "SLEEPING",
        6: "AT_SCHOOL",
        7: "COMPUTER_WORK",
        8: "OR_standing",
        9: "TALKING",
        10: "LOC_main_workplace",
        11: "WITH_FRIENDS",
        12: "PHONE_IN_POCKET",
        13: "FIX_walking",
        14: "SURFING_THE_INTERNET",
        15: "EATING",
        16: "PHONE_IN_HAND",
        17: "WATCHING_TV",
        18: "OR_outside",
        19: "PHONE_IN_BAG",
        20: "OR_exercise",
        21: "DRIVE_-_I_M_THE_DRIVER",
        22: "WITH_CO-WORKERS",
        23: "IN_CLASS",
        24: "IN_A_CAR",
        25: "IN_A_MEETING",
        26: "BICYCLING",
        27: "COOKING",
        28: "LAB_WORK",
        29: "CLEANING",
        30: "GROOMING",
        31: "TOILET",
        32: "DRIVE_-_I_M_A_PASSENGER",
        33: "DRESSING",
        34: "FIX_restaurant",
        35: "BATHING_-_SHOWER",
        36: "SHOPPING",
        37: "ON_A_BUS",
        38: "AT_A_PARTY",
        39: "DRINKING__ALCOHOL_",
        40: "WASHING_DISHES",
        41: "AT_THE_GYM",
        42: "FIX_running",
        43: "STROLLING",
        44: "STAIRS_-_GOING_UP",
        45: "STAIRS_-_GOING_DOWN",
        46: "SINGING",
        47: "LOC_beach",
        48: "DOING_LAUNDRY",
        49: "AT_A_BAR",
        50: "ELEVATOR"
    }
    
    activity_codes = {
        "OR_indoors": 0,
        "LOC_home": 1,
        "SITTING": 2,
        "PHONE_ON_TABLE": 3,
        "LYING_DOWN": 4,
        "SLEEPING": 5,
        "AT_SCHOOL": 6,
        "COMPUTER_WORK": 7,
        "OR_standing": 8,
        "TALKING": 9,
        "LOC_main_workplace": 10,
        "WITH_FRIENDS": 11,
        "PHONE_IN_POCKET": 12,
        "FIX_walking": 13,
        "SURFING_THE_INTERNET": 14,
        "EATING": 15,
        "PHONE_IN_HAND": 16,
        "WATCHING_TV": 17,
        "OR_outside": 18,
        "PHONE_IN_BAG": 19,
        "OR_exercise": 20,
        "DRIVE_-_I_M_THE_DRIVER": 21,
        "WITH_CO-WORKERS": 22,
        "IN_CLASS": 23,
        "IN_A_CAR": 24,
        "IN_A_MEETING": 25,
        "BICYCLING": 26,
        "COOKING": 27,
        "LAB_WORK": 28,
        "CLEANING": 29,
        "GROOMING": 30,
        "TOILET": 31,
        "DRIVE_-_I_M_A_PASSENGER": 32,
        "DRESSING": 33,
        "FIX_restaurant": 34,
        "BATHING_-_SHOWER": 35,
        "SHOPPING": 36,
        "ON_A_BUS": 37,
        "AT_A_PARTY": 38,
        "DRINKING__ALCOHOL_": 39,
        "WASHING_DISHES": 40,
        "AT_THE_GYM": 41,
        "FIX_running": 42,
        "STROLLING": 43,
        "STAIRS_-_GOING_UP": 44,
        "STAIRS_-_GOING_DOWN": 45,
        "SINGING": 46,
        "LOC_beach": 47,
        "DOING_LAUNDRY": 48,
        "AT_A_BAR": 49,
        "ELEVATOR": 50
    }
    
    def __init__(self, dataset_dir:Path, labels_dir: Path, accelerometer_dir: Path,
                 gyroscope_dir: Path, window: int,
                 store_raw_df: bool = False, download: bool = False):
        self.dataset_dir = dataset_dir
        self.labels_dir = labels_dir
        self.accelerometer_dir = accelerometer_dir
        self.gyroscope_dir = gyroscope_dir
        self._users = None
        self.store_raw_df = store_raw_df
        self.window_len = window
        if download:
            self._download_and_extract()
        self.read_users()
        #self.metadata_df = self._read_metadata()
        
    def _read_metadata(self):
        df = None
        for uuid in self.user_ids:
            try:
                df = self.user_time_window(uuid, window_len=self.window_len)
            except Exception as e:
                print(f"Error processing user {uuid}. {e.__class__.__name__}: {e}")
                continue
        return df

    def _download_and_extract(self):
        # Create directories
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)

        fname_acc = os.path.join(self.dataset_dir, "raw_acc.zip") 
        fname_gyr = os.path.join(self.dataset_dir, "proc_gyr.zip")
        fname_label = os.path.join(self.labels_dir, "features_labels.zip")

        print(f"Downloading acc data to '{fname_acc}'")
        download_url(self.acc_url, fname=fname_acc)
        print(f"Downloading gyr data to '{fname_gyr}'")
        download_url(self.gyr_url, fname=fname_gyr)
        print(f"Downloading label data to '{fname_label}'")
        download_url(self.label_url, fname=fname_label)

        if not os.path.exists(fname_acc):
            print(f"Downloading dataset to '{fname_acc}'")
            download_url(self.dataset_url, fname=fname_acc)
        else:
            print(f"'{fname_acc}' already exists and will not be downloaded again")
        if not os.path.exists(fname_gyr):
            print(f"Downloading dataset to '{fname_gyr}'")
            download_url(self.dataset_url, fname=fname_gyr)
        else:
            print(f"'{fname_gyr}' already exists and will not be downloaded again")
        if not os.path.exists(fname_label):
            print(f"Downloading dataset to '{fname_label}'")
            download_url(self.dataset_url, fname=fname_label)
        else:
            print(f"'{fname_label}' already exists and will not be downloaded again")

        print(f"Unziping acc data to {self.dataset_dir}")
        unzip_file(filename=fname_acc, destination=self.dataset_dir)
        print(f"Removing {fname_acc}")
        os.unlink(fname_acc)
        print("Done!")
        print(f"Unziping gyr data to {self.dataset_dir}")
        unzip_file(filename=fname_gyr, destination=self.dataset_dir)
        print(f"Removing {fname_gyr}")
        os.unlink(fname_gyr)
        print("Done!")
        print(f"Unziping label data to {self.labels_dir}")
        unzip_file(filename=fname_label, destination=self.labels_dir)
        print(f"Removing {fname_label}")
        os.unlink(fname_label)
        print("Done!")

    def _extract_time_series(self):
        # list the label files
        labels_zip_files = os.listdir(self.labels_dir)

        # for each file, get the UUID, create the corresponding folder, move the file to the folder, and then extract it there, deleting all the zip files after the extraction
        for zip_file in labels_zip_files:
            uuid = zip_file.split('.')[0]
            path = os.path.join(self.labels_dir, uuid)
            os.makedirs(path, exist_ok=True)
            target = os.path.join(path, zip_file)
            csv_name = os.path.join(path, uuid + ".csv")
            shutil.move(os.path.join(self.labels_dir, zip_file), target)
            with gzip.open(target, 'rb') as gz_file:
                data = gz_file.read()
                with open(csv_name, 'wb') as out_file:
                    out_file.write(data)
            os.unlink(target)

    def read_users(self):
        # get all files
        label_files = list(self.labels_dir.rglob("*.csv.gz"))
        def read_user(label_file: Path):
            userid = label_file.stem.split(".")[0]

            acc_path = self.accelerometer_dir/userid
            if not acc_path.exists():
                print(f"User {userid} does not have accelerometer files. Skipping")
                return None
            gyro_path = self.gyroscope_dir/userid
            if not gyro_path.exists():
                print(f"User {userid} does not have gyroscope files. Skipping")
                return None

            return ExtraSensoryUser(
                dataset=self,
                uuid=userid, 
                feature_file=label_file, 
                user_accelerometer_dir=acc_path, 
                user_gyroscope_dir=gyro_path,
                store_raw_df=self.store_raw_df
            )

        users = thread_map(read_user, label_files, desc="Reading users....")
        self._users = {u.uuid: u for u in users if u is not None}
    
    @property
    def user_map(self):
        return self._users

    @property
    def all_users(self):
        ids = self.user_ids
        return [self._users[i] for i in ids]

    @property
    def user_ids(self):
        return sorted(list(self._users.keys()))

    def user_raw_measurements(self, userid: str):
        df = self._users[userid].raw_measurements
        df["user"] = userid
        return df

    def user_time_window(self, userid: str, window_len: Union[int, str], overlap: int = 0, interpolation: str = None, merge_classes_label: bool = True):
        user = self._users[userid]
        df = user.time_window(window_len, overlap, interpolation, merge_classes_label)
        df["user"] = userid
        return df
    
    def get_all_user_ids(self) -> List[int]:
        return np.sort(self.metadata_df["user"].unique()).tolist()
    
    def get_all_activity_ids(self) -> List[int]:
        return np.sort(self.metadata_df["class"].unique()).tolist()
    
    def get_all_activity_names(self) -> List[str]:
        return [self.activity_names[i] for i in self.get_all_activity_ids()]
    
    def get_data_iterator(self, users: List[int] = None, activities: List[int] = None, shuffle: bool = False) -> List[pd.DataFrame]:
        # Must select first
        if users is None:
            users = self.get_all_user_ids()
        if activities is None:
            activities = self.get_all_activity_ids()
            
        selecteds = self.metadata_df[
            (self.metadata_df["user"].isin(users)) & 
            (self.metadata_df["class"].isin(activities))
        ]
        
        # Shuffle data
        if shuffle:
            selecteds = selecteds.sample(frac=1)
        
        for i, (row_index, row) in enumerate(selecteds.iterrows()):
            data = self._read_csv_data(row)
            yield data
            
    #def __str__(self):
    #    return f"ExtraSensory Dataset at: '{self.dataset_dir}' ({len(self.metadata_df)} files, {len(self.get_all_user_ids())} users and {len(self.get_all_activity_ids())} activities)"
