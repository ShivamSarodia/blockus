import os
import random
import torch
import numpy as np
from collections import deque

from training.load_games import load_games


class GameDataManager:
    def __init__(self, gamedata_dir, max_window_size):
        self.gamedata_dir = gamedata_dir
        self.max_window_size = max_window_size

        self.loaded_gamedata_paths = set()

        self.gamedata = deque()
        self._window_size = 0
        self._cumulative_window_fed = 0

    def _load_unread_gamedata(self, samples_requested: int):
        """
        Load unread gamedata into memory.
        Read enough files as needed to get the requested number of samples,
        if available.
        """
        if samples_requested <= 0:
            return

        # Get a list of all gamedata files on disk.
        gamedata_paths = [
            os.path.join(self.gamedata_dir, filename)
            for filename in os.listdir(self.gamedata_dir)
            if filename.endswith(".npz")
        ]
        gamedata_paths.sort()

        # Load new files until we've reached the requested number of samples 
        # (or run out of files).
        total_samples_loaded = 0
        paths_to_load = []
        for path in gamedata_paths:
            if total_samples_loaded >= samples_requested:
                break

            if path not in self.loaded_gamedata_paths:
                paths_to_load.append(path)                
                self.loaded_gamedata_paths.add(path)
                total_samples_loaded += int(path.split("_")[-1].split(".")[0])

        if len(paths_to_load) == 0:
            return

        game_data_tuple = load_games(paths_to_load)

        # This can happen if we encounter a bad game file.
        if not game_data_tuple:
            return

        for i in range(len(game_data_tuple[0])):
            self.gamedata.append(
                tuple(
                    game_data_category[i]
                    for game_data_category in game_data_tuple
                )
            )

    def feed_window(self, max_samples_to_feed: int, require_exact_count: bool = False):
        """
        Read up to max_samples_to_feed new samples into the current sample window.
        Drop any old samples that have fallen out of the window.
        Return the number of samples read.
        """
        # Load new samples from disk if possible.
        # E.g. suppose we have 100 samples in gamedata, and the window size is currently 95.
        # If we're asked to feed the window 20 samples, we'll attempt to load at least 15 more
        # samples from disk.
        # 
        # This method would still work correctly if we loaded all samples available on disk, but 
        # it would be prohibitively expensive in the offline training context where we have many
        # samples on disk that we cannot load into memory at once.
        self._load_unread_gamedata(max_samples_to_feed - (len(self.gamedata) - self._window_size))

        # Check how many samples we have beyond the window to see if we can feed into
        # the window the desired amount.
        num_samples_after_window = len(self.gamedata) - self._window_size

        if require_exact_count:
            # If we require an exact count, either feed the requested number of samples 
            # or none at all.
            if num_samples_after_window >= max_samples_to_feed:
                num_samples_to_feed = max_samples_to_feed
            else:
                num_samples_to_feed = 0
        else:
            # Otherwise, feed as many samples as we can up to the requested number.
            num_samples_to_feed = min(num_samples_after_window, max_samples_to_feed)

        # Short circuit the common case where we have no new samples to feed.
        if num_samples_to_feed == 0:
            return 0

        # First, expand the window size to account for the new samples.
        # (This may send the window_size above the limit.)
        self._window_size += num_samples_to_feed

        # Second, if we've exceeded the window size limit, drop old samples
        # and shrink the window size back down.
        num_samples_to_drop = max(0, self._window_size - self.max_window_size)
        for _ in range(num_samples_to_drop):
            self.gamedata.popleft()
        self._window_size -= num_samples_to_drop

        assert self._window_size <= self.max_window_size
        if require_exact_count:
            assert num_samples_to_feed == max_samples_to_feed

        self._cumulative_window_fed += num_samples_to_feed
        return num_samples_to_feed

    def sample(self, size: int):
        """
        Return a batch randomly sampled from the current sample window.
        """
        assert size <= self._window_size, "Cannot sample batch smaller than current window size"

        sample_indices = random.sample(range(len(self.gamedata)), size)

        NUM_CATEGORIES = len(self.gamedata[0])

        batch = [[] for _ in range(NUM_CATEGORIES)]
        for i in sample_indices:
            gamedata_category_tuple = tuple(self.gamedata[i])
            for category_index in range(NUM_CATEGORIES):
                batch[category_index].append(gamedata_category_tuple[category_index])

        return tuple(torch.tensor(np.array(b)) for b in batch)

    def current_window_size(self):
        return self._window_size
    
    def cumulative_window_fed(self):
        return self._cumulative_window_fed
