import json
import os

meta_dir = "data/meta"
FORMATIONS = ["I-Form", "Shotgun", "Trips", "Singleback", "Pistol", "Empty"]

def label_formations():
    for file in sorted(os.listdir(meta_dir)):
        if not file.endswith(".json"):
            continue
        
        path = os.path.join(meta_dir, file)
        with open(path) as f:
            data = json.load(f)

        print(f"\nFrame: {data['frame']}")
        for i, f_name in enumerate(FORMATIONS):
            print(f"{i}: {f_name}")
        idx = int(input("Select formation number: "))
        data["formation"] = FORMATIONS[idx]

        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Saved formation '{FORMATIONS[idx]}' for {data['frame']}")

if __name__ == "__main__":
    label_formations()
