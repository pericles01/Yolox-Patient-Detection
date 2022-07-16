import os
import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F


export_dir =  './Door_dataset6000'
temp_dir = './Openimages'

os.mkdir(export_dir)

if len(os.listdir(path=export_dir)) != 0:
    exit('Target directory is not empty!')

fo.config.dataset_zoo_dir = os.path.join(temp_dir, 'zoo')


trainset = foz.load_zoo_dataset("open-images-v6", classes=["Door"], split='train', max_samples=6000,
    dataset_dir=temp_dir, seed=51, shuffle=True, label_types='detections')
trainview = trainset.filter_labels(
    "detections",
    (F("label") == "Door"))
trainview.save()

catview = trainset.view()
catview.info['categories'] = [{
      "supercategory": "null",
      "id": 0,
      "name": "door"
    }]
catview.default_classes = ['Door']
catview.save()

trainset.export(
    export_dir=export_dir,
    dataset_type=fo.types.COCODetectionDataset,
    data_path='images',
    labels_path = "annotations/train.json",
    label_field="detections")

valset = foz.load_zoo_dataset("open-images-v6", classes=["Door"], split='validation', max_samples=300, 
    dataset_dir=temp_dir, seed=51, shuffle=True, label_types ='detections')
valview = valset.filter_labels(
    "detections",
    (F("label") == "Door"))
valview.save()

catview = valset.view()
catview.info['categories'] = [{
      "supercategory": "null",
      "id": 0,
      "name": "door"
    }]
catview.default_classes = ['Door']
catview.save()

valset.export(
    export_dir=export_dir,
    dataset_type=fo.types.COCODetectionDataset,
    data_path='images',
    labels_path = "annotations/val.json",
    label_field="detections")