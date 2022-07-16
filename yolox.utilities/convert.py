import json

subdict = {"iscrowd": 0}

with open("./annotations/val.json", "r+") as file:
    data = json.load(file)
    for dict in data["annotations"]:
        dict.update(subdict)
        print(dict)
    file.seek(0)
    json.dump(data, file)
print("val.json edited")

with open("./annotations/train.json", "r+") as file:
    data = json.load(file)
    for dict in data["annotations"]:
        dict.update(subdict)
        print(dict)
    file.seek(0)
    json.dump(data, file)
print("train.json edited")
