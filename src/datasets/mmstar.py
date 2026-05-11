from torch.utils.data import Dataset

from datasets import load_dataset


class MMStarDataset(Dataset):
    def __init__(self, hf_path="Lin-Chen/MMStar", split="val"):
        """
        Args:
            hf_path (str): The Hugging Face dataset path.
            split (str): The dataset split to load.
        """
        # 1. Load the actual images/data from HF
        self.hf_dataset = load_dataset(hf_path, split=split)

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        

        image = item['image'] #.convert("RGB")
        index = item["index"]
        question = item['question']
        meta_info = item['meta_info']
        label = item["answer"]
        category = item["category"]
        
        return {
            "image": image,
            "question": question,
            "label": label,
            "index": index,
            "metadata": meta_info,
            "category": category,
        }

