import json
from pathlib import Path
# import sys
# sys.path.append(str(Path(__file__).resolve().parent.parent))

class JSONLLogger:
    def __init__(self, log_file):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)


    def log(self, **kwargs):
        with self.log_file.open("a") as f:
            json.dump(kwargs, f)
            f.write("\n")