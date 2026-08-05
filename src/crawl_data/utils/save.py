import json
import os

def save_json(data, path):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path,"w",encoding="utf8") as f:

        json.dump(data,f,ensure_ascii=False,indent=4)