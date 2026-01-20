import os
import json
from datetime import datetime, timezone, timedelta

KST = timezone(timedelta(hours=9))

class TMPPLogger:
    def __init__(self, log_file):
        if not os.path.exists(log_file):
            with open(log_file, "w") as f:
                json.dump({}, f, indent=4)

        self.start_times = {}
        self.durations = {}
        self.log_file = log_file
        self.save_scene_name = None

        with open(log_file, 'r') as jf:
            self.log_dict = json.load(jf)

        self.cur_dict = {}

    def set_cur_log(self, save_scene_name):
        self.save_scene_name = save_scene_name
        print(f"\n==========================================================\n{self.save_scene_name}\n")

        if save_scene_name in self.log_dict.keys():
            self.cur_dict = self.log_dict[save_scene_name]
        else:
            self.cur_dict = {}
            
    def start(self, name):
        self.cur_dict[name] = {"Start": datetime.now(KST).isoformat()}
        self.log_dict[self.save_scene_name] = self.cur_dict
        with open(self.log_file, 'w') as jf:
            json.dump(self.log_dict, jf, indent=4)
        print(f">>> Start {name}")

    def stop(self, name):
        self.cur_dict[name]["Stop"] = datetime.now(KST).isoformat()

        start_time = datetime.fromisoformat(self.cur_dict[name]["Start"])
        stop_time = datetime.fromisoformat(self.cur_dict[name]["Stop"])

        elapsed = stop_time - start_time
        formatted_time = str(elapsed).split('.')[0]
        self.cur_dict[name]["Duration"] = formatted_time

        print(f'=> Duration of [{name}]: {formatted_time}')

        self.log_dict[self.save_scene_name] = self.cur_dict
        with open(self.log_file, 'w') as jf:
            json.dump(self.log_dict, jf, indent=4)
    
    def is_done(self, name, result_dir=None):
        with open(self.log_file, 'r') as jf:
            self.log_dict = json.load(jf)
        try:
            done = "Duration" in self.log_dict[self.save_scene_name][name]
            # if not os.path.exists(result_dir):
            #     done = False
        except:
            done = False
            # if os.path.exists(result_dir):
            #     shutil.rmtree(result_dir)
        return done