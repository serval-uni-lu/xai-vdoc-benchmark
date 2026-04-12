import os
import glob
import json
import torch
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class POPEGroundingDataset(Dataset):
    def __init__(
        self, data_path, pope_json_path, pope_type="random", coco_split="val2014"
    ):
        """
        Args:
            data_path (str): Path to your COCO folder (containing 'val2014' and 'annotations')
            pope_json_path (str): Path to the folder with the POPE json file (e.g., 'coco_pope_random.jsonl')
            coco_split (str): 'val2014' depending on which POPE version you downloaded
        """
        self.data_path = data_path
        self.coco_split = coco_split

        self.pope_json_path = pope_json_path
        self.pope_type = pope_type

        # 1. Keep your COCO Instances logic for Ground Truth Masks
        instances_file = os.path.join(
            self.data_path, "annotations", f"instances_{coco_split}.json"
        )
        self.coco_instances = COCO(instances_file)
        cats = self.coco_instances.loadCats(self.coco_instances.getCatIds())
        self.id2name = {cat["id"]: cat["name"] for cat in cats}

        # 2. Load the POPE Questions
        self.pope_data = self._gather_json_files()

    def _gather_json_files(self):
        json_files = glob.glob(os.path.join(self.pope_json_path, "*.json"))

        if not json_files:
            raise ValueError(f"Error: No .json files found in {self.pope_json_path}")

        pope_data = {}
        print(f"Scanning {len(json_files)} POPE json files...")
        for json_file in json_files:
            with open(json_file, "r") as f:
                pope_type = json_file.split("pope_")[-1].split(".json")[0]
                pope_data[pope_type] = []
                for line in f:
                    pope_data[pope_type].append(json.loads(line))
        return pope_data

    def __len__(self):
        return len(self.pope_data[self.pope_type])

    def __getitem__(self, idx):
        item = self.pope_data[self.pope_type][idx]
        img_filename = item["image"]  # e.g., '000000397133.jpg'
        question = item["text"]  # e.g., 'Is there a dog in the image?'
        label = item["label"]  # e.g., 'yes' or 'no'

        # --- 1. Load Image ---
        img_path = os.path.join(self.data_path, self.coco_split, img_filename)
        image = Image.open(img_path).convert("RGB")

        # --- 2. Extract Object Category ---
        # POPE questions are strictly formatted: "Is there a <object> in the image?"
        object_name = (
            question.replace("Is there a ", "").replace(" in the image?", "").strip()
        )

        # --- 3. Get Ground Truth Mask (Your original logic, targeted to the POPE object) ---
        # Extract the numeric ID from the COCO filename (e.g., '000000397133.jpg' -> 397133)
        # img_id = int(img_filename.split('.')[0])
        ann_ids_inst = self.coco_instances.getAnnIds(imgIds=img_filename)
        anns_inst = self.coco_instances.loadAnns(ann_ids_inst)

        # Create a blank mask
        W, H = image.size
        final_mask = np.zeros((H, W), dtype=np.float32)

        # If the label is "yes", extract the mask for the specific object POPE is asking about
        if label == "yes":
            for ann in anns_inst:
                cat_name = self.id2name[ann["category_id"]]

                if cat_name == object_name:
                    instance_mask = self.coco_instances.annToMask(ann)
                    final_mask = np.maximum(final_mask, instance_mask)

        return {
            "image": image,
            "question": question,
            "label": label,
            "object_name": object_name,
            "ground_truth_mask": torch.tensor(
                final_mask
            ),  # Boolean tensor of the object!
            "image_id": img_filename,
            "image_path": img_path,
        }


class POPEOracleDataset(Dataset):
    def __init__(
        self, 
        data_path, 
        pope_json_path, 
        pope_type="random", 
        coco_split="val2014",
        max_samples=500         # <-- ADDED: Easily cap it for Experiment 1.1
    ):
        self.data_path = data_path
        self.coco_split = coco_split
        self.pope_json_path = pope_json_path
        self.pope_type = pope_type

        # 1. Keep your COCO Instances logic
        instances_file = os.path.join(
            self.data_path, "annotations", f"instances_{coco_split}.json"
        )
        self.coco_instances = COCO(instances_file)
        cats = self.coco_instances.loadCats(self.coco_instances.getCatIds())
        self.id2name = {cat["id"]: cat["name"] for cat in cats}

        # 2. Load the POPE Questions (FILTERED FOR "YES" ONLY)
        self.pope_data = self._gather_json_files(max_samples)

    def _gather_json_files(self, max_samples):
        json_files = glob.glob(os.path.join(self.pope_json_path, "*.json"))
        if not json_files:
            raise ValueError(f"Error: No .json files found in {self.pope_json_path}")

        pope_data = {}
        for json_file in json_files:
            with open(json_file, "r") as f:
                pope_type = json_file.split("pope_")[-1].split(".json")[0]
                pope_data[pope_type] = []
                
                for line in f:
                    item = json.loads(line)
                    # KEEP ONLY 'YES' SAMPLES FOR THE ORACLE TEST
                    if item["label"] == "yes":
                        pope_data[pope_type].append(item)
                        
                    # Stop early if we hit our target number of samples
                    if len(pope_data[pope_type]) >= max_samples:
                        break
        return pope_data

    def __len__(self):
        return len(self.pope_data[self.pope_type])

    def __getitem__(self, idx):
        item = self.pope_data[self.pope_type][idx]
        img_filename = item["image"]  
        question = item["text"]  
        label = item["label"]  

        # --- 1. Load Image ---
        img_path = os.path.join(self.data_path, self.coco_split, img_filename)
        image = Image.open(img_path).convert("RGB")

        # --- 2. Extract Object Category ---
        object_name = question.replace("Is there a ", "").replace(" in the image?", "").strip()

        # --- 3. Get Ground Truth Image Mask (BUG FIX) ---
        # FIX: Extract raw integer from filename (Handles both '000123.jpg' and 'COCO_val2014_000123.jpg')
        img_id_str = img_filename.split('_')[-1].split('.')[0]
        img_id = int(img_id_str)
        
        ann_ids_inst = self.coco_instances.getAnnIds(imgIds=img_id)
        anns_inst = self.coco_instances.loadAnns(ann_ids_inst)

        W, H = image.size
        pixel_oracle_mask = np.zeros((H, W), dtype=np.float32)

        for ann in anns_inst:
            cat_name = self.id2name[ann["category_id"]]
            if cat_name == object_name:
                instance_mask = self.coco_instances.annToMask(ann)
                pixel_oracle_mask = np.maximum(pixel_oracle_mask, instance_mask)


        return {
            "image": image,
            "question": question,
            "object_name": object_name,
            "pixel_oracle_mask": torch.tensor(pixel_oracle_mask), 
            "image_id": img_id
        }
