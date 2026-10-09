# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
import os
import cv2
import json
import datetime
import numpy as np
import time
from .rerun_visualizer import RerunLogger
from queue import Queue, Empty
from threading import Lock, Thread
import logging_mp
import re
logger_mp = logging_mp.getLogger(__name__)

EPISODE_DIRECTORY_PATTERN = re.compile(r"^episode_(\d+)$")

class EpisodeWriter():
    def __init__(self, task_dir, task_goal=None, task_desc = None, task_steps = None, frequency=30, image_size=[640, 480], rerun_log = True, task_before_accept=None, task_after_accept=None):
        """
        image_size: [width, height]
        """
        logger_mp.info("==> EpisodeWriter initializing...\n")
        self.task_dir = task_dir
        self.text = {
            "goal": "Pick up the red cup on the table.",
            "desc": "task description",
            "steps":"step1: do this; step2: do that; ...",
        }
        if task_goal is not None:
            self.text['goal'] = task_goal
        if task_desc is not None:
            self.text['desc'] = task_desc
        if task_steps is not None:
            self.text['steps'] = task_steps

        self.frequency = frequency
        self.image_size = image_size
        self.task_before_accept = task_before_accept
        self.task_after_accept = task_after_accept
        if (task_before_accept is None) != (task_after_accept is None):
            raise ValueError("task_before_accept and task_after_accept must be set together")
        self._task_lock = Lock()
        self._accepted_event = None

        self.rerun_log = rerun_log
        if self.rerun_log:
            logger_mp.info("==> RerunLogger initializing...\n")
            self.rerun_logger = RerunLogger(prefix="online/", IdxRangeBoundary = 60, memory_limit = "300MB")
            logger_mp.info("==> RerunLogger initializing ok.\n")
        
        self.item_id = -1
        self.episode_id = -1
        if os.path.exists(self.task_dir):
            episode_ids = []
            for parent in (self.task_dir, os.path.join(self.task_dir, "rejected")):
                if not os.path.isdir(parent):
                    continue
                for episode_dir in os.listdir(parent):
                    match = EPISODE_DIRECTORY_PATTERN.fullmatch(episode_dir)
                    if match:
                        episode_ids.append(int(match.group(1)))
            self.episode_id = max(episode_ids, default=-1)
            logger_mp.info(f"==> task_dir directory already exist, now self.episode_id is:{self.episode_id}\n")
        else:
            os.makedirs(self.task_dir)
            logger_mp.info(f"==> episode directory does not exist, now create one.\n")
        self.data_info()

        self.is_available = True  # Indicates whether the class is available for new operations
        # Initialize the queue and worker thread
        self.item_data_queue = Queue(-1)
        self.stop_worker = False
        self.need_save = False  # Flag to indicate when save_episode is triggered
        self.worker_thread = Thread(target=self.process_queue)
        self.worker_thread.start()

        logger_mp.info("==> EpisodeWriter initialized successfully.\n")
    
    def is_ready(self):
        return self.is_available

    def accepted_event(self):
        with self._task_lock:
            return None if self._accepted_event is None else dict(self._accepted_event)

    def data_info(self, version='1.0.0', date=None, author=None):
        self.info = {
                "version": "1.0.0" if version is None else version, 
                "date": datetime.date.today().strftime('%Y-%m-%d') if date is None else date,
                "author": "unitree" if author is None else author,
                "image": {"width":self.image_size[0], "height":self.image_size[1], "fps":self.frequency},
                "depth": {"width":self.image_size[0], "height":self.image_size[1], "fps":self.frequency},
                "audio": {"sample_rate": 16000, "channels": 1, "format":"PCM", "bits":16},    # PCM_S16
                "joint_names":{
                    "left_arm":   [],
                    "left_ee":  [],
                    "right_arm":  [],
                    "right_ee": [],
                    "body":       [],
                },

                "tactile_names": {
                    "left_ee": [],
                    "right_ee": [],
                }, 
                "sim_state": ""
            }

 
    def create_episode(self):
        """
        Create a new episode.
        Returns:
            bool: True if the episode is successfully created, False otherwise.
        Note:
            Once successfully created, this function will only be available again after save_episode complete its save task.
        """
        if not self.is_available:
            logger_mp.info("==> The class is currently unavailable for new operations. Please wait until ongoing tasks are completed.")
            return False  # Return False if the class is unavailable

        # Reset episode-related data and create necessary directories
        with self._task_lock:
            self.item_id = -1
            self._accepted_event = None
        self.episode_id = self.episode_id + 1
        
        self.episode_dir = os.path.join(self.task_dir, f"episode_{str(self.episode_id).zfill(4)}")
        self.color_dir = os.path.join(self.episode_dir, 'colors')
        self.depth_dir = os.path.join(self.episode_dir, 'depths')
        self.audio_dir = os.path.join(self.episode_dir, 'audios')
        self.json_path = os.path.join(self.episode_dir, 'data.json')
        self.events_path = os.path.join(self.episode_dir, 'events.jsonl')
        os.makedirs(self.episode_dir, exist_ok=True)
        os.makedirs(self.color_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)
        with open(self.json_path, "w", encoding="utf-8") as f:
            f.write('{\n')
            f.write('"info": ' + json.dumps(self.info, ensure_ascii=False, indent=4) + ',\n')
            f.write('"text": ' + json.dumps(self.text, ensure_ascii=False, indent=4) + ',\n')
            f.write('"data": [\n')
        if self.task_before_accept is not None:
            with open(self.events_path, "w", encoding="utf-8"):
                pass
        self.first_item = True   # Flag to handle commas in JSON array

        if self.rerun_log:
            self.online_logger = RerunLogger(prefix="online/", IdxRangeBoundary = 60, memory_limit="300MB")

        self.is_available = False  # After the episode is created, the class is marked as unavailable until the episode is successfully saved
        logger_mp.info(f"==> New episode created: {self.episode_dir}")
        return True  # Return True if the episode is successfully created

    def task_context_for_timestamp(self, record_monotonic_ns):
        """Return the task label that was active when a sample was timestamped."""
        with self._task_lock:
            accepted_event = self._accepted_event
            if self.task_before_accept is None:
                return {
                    "active_task": self.text.get("goal", ""),
                    "task_phase": "single_task",
                }
            if (
                accepted_event is None
                or record_monotonic_ns < accepted_event["receiver_monotonic_ns"]
            ):
                return {
                    "active_task": self.task_before_accept,
                    "task_phase": "before_item_accepted",
                }
            return {
                "active_task": self.task_after_accept,
                "task_phase": "after_item_accepted",
                "item_accepted_event_id": accepted_event["event_id"],
                "item_accepted_barcode": accepted_event["barcode"],
            }

    def accept_item_event(self, event):
        """Commit one item_accepted transition to the active raw episode."""
        with self._task_lock:
            if self.task_before_accept is None:
                return {
                    "status": "error",
                    "message": "task switching is not configured",
                }
            if self.is_available:
                return {
                    "status": "error",
                    "message": "no active recording episode",
                }
            if self.item_id < 0:
                return {
                    "status": "error",
                    "message": "the active episode has no recorded frames",
                }
            if self._accepted_event is not None:
                return {
                    "status": "ok",
                    "episode_id": self.episode_id,
                    "active_task": self.task_after_accept,
                    "duplicate": True,
                    "duplicate_scope": "episode",
                    "accepted_event_id": self._accepted_event["event_id"],
                    "accepted_barcode": self._accepted_event["barcode"],
                }

            event_record = {
                **event,
                "episode_id": self.episode_id,
                "last_enqueued_frame_index": self.item_id,
                "task_before": self.task_before_accept,
                "task_after": self.task_after_accept,
            }
            with open(self.events_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(event_record, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self._accepted_event = event_record
            return {
                "status": "ok",
                "episode_id": self.episode_id,
                "active_task": self.task_after_accept,
                "receiver_monotonic_ns": event_record["receiver_monotonic_ns"],
                "receiver_wall_time_ns": event_record["receiver_wall_time_ns"],
                "duplicate": False,
            }
        
    def add_item(self, colors, depths=None, states=None, actions=None, tactiles=None, audios=None, sim_state=None, metadata=None):
        # Increment the item ID
        with self._task_lock:
            self.item_id += 1
            item_id = self.item_id
        # Create the item data dictionary
        item_data = {
            'idx': item_id,
            'colors': colors,
            'depths': depths,
            'states': states,
            'actions': actions,
            'tactiles': tactiles,
            'audios': audios,
            'sim_state': sim_state,
            'metadata': metadata,
        }
        # Enqueue the item data
        self.item_data_queue.put(item_data)

    def process_queue(self):
        while not self.stop_worker or not self.item_data_queue.empty():
            # Process items in the queue
            try:
                item_data = self.item_data_queue.get(timeout=1)
                try:
                    self._process_item_data(item_data)
                except Exception as e:
                    logger_mp.info(f"Error processing item_data (idx={item_data['idx']}): {e}")
                self.item_data_queue.task_done()
            except Empty:
                pass
        
            # Check if save_episode was triggered
            if self.need_save and self.item_data_queue.empty():
                self._save_episode()

    def _process_item_data(self, item_data):
        idx = item_data['idx']
        colors = item_data.get('colors', {})
        depths = item_data.get('depths', {})
        audios = item_data.get('audios', {})

        # Save images
        if colors:
            for idx_color, (color_key, color) in enumerate(colors.items()):
                color_name = f'{str(idx).zfill(6)}_{color_key}.jpg'
                if not cv2.imwrite(os.path.join(self.color_dir, color_name), color):
                    logger_mp.info(f"Failed to save color image.")
                item_data['colors'][color_key] = os.path.join('colors', color_name)

        # Save depths
        if depths:
            for idx_depth, (depth_key, depth) in enumerate(depths.items()):
                depth_name = f'{str(idx).zfill(6)}_{depth_key}.jpg'
                if not cv2.imwrite(os.path.join(self.depth_dir, depth_name), depth):
                    logger_mp.info(f"Failed to save depth image.")
                item_data['depths'][depth_key] = os.path.join('depths', depth_name)

        # Save audios
        if audios:
            for mic, audio in audios.items():
                audio_name = f'audio_{str(idx).zfill(6)}_{mic}.npy'
                np.save(os.path.join(self.audio_dir, audio_name), audio.astype(np.int16))
                item_data['audios'][mic] = os.path.join('audios', audio_name)

        # Update episode data
        with open(self.json_path, "a", encoding="utf-8") as f:
            if not self.first_item:
                f.write(",\n")
            f.write(json.dumps(item_data, ensure_ascii=False, indent=4))
            self.first_item = False

        # Log data if necessary
        if self.rerun_log:
            curent_record_time = time.time()
            logger_mp.info(f"==> episode_id:{self.episode_id}  item_id:{idx}  current_time:{curent_record_time}")
            self.rerun_logger.log_item_data(item_data)

    def save_episode(self):
        """
        Trigger the save operation. This sets the save flag, and the process_queue thread will handle it.
        """
        self.need_save = True  # Set the save flag
        logger_mp.info(f"==> Episode saved start...")

    def _save_episode(self):
        """
        Save the episode data to a JSON file.
        """
        with open(self.json_path, "a", encoding="utf-8") as f:
            f.write("\n]\n}")      # Close the JSON array and object

        self.need_save = False     # Reset the save flag
        self.is_available = True   # Mark the class as available after saving
        logger_mp.info(f"==> Episode saved successfully to {self.json_path}.")

    def close(self):
        """
        Stop the worker thread and ensure all tasks are completed.
        """
        self.item_data_queue.join()
        if not self.is_available:  # If self.is_available is False, it means there is still data not saved.
            self.save_episode()
        while not self.is_available:
            time.sleep(0.01)
        self.stop_worker = True
        self.worker_thread.join()
