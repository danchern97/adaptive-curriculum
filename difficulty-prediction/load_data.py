import os
import pickle
import random
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from collections import Counter
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Add searchformer path for imports
searchformer_path = Path(__file__).parent.parent / "searchformer-main"
if str(searchformer_path) not in sys.path:
    sys.path.insert(0, str(searchformer_path))

# Import searchformer modules
from searchformer.local_storage import LocalClient
from searchformer.trace import TokenizedTrace

class TeacherDataset(object):
    def __init__(self, data_path, model_name):
        self.data_path = data_path
        self.model_name = model_name

        with open(os.path.join(data_path, "data_train.pkl"), "rb") as f:
            self.data_train = pickle.load(f)
        with open(os.path.join(data_path, "data_ref.pkl"), "rb") as f:
            self.data_ref = pickle.load(f)
        
    def load_embeddings(self):
        questions = pd.read_parquet(os.path.join(self.data_path, "questions.parquet"))['problem'].tolist()
        embeddings = torch.load(os.path.join(self.data_path, 'embeddings', f'{self.model_name.replace("/", "_")}.pt'))
       
        if len(questions) != embeddings.shape[0]:
            raise ValueError(f"Number of questions ({len(questions)}) does not match number of embeddings ({embeddings.shape[0]})")
        
        embeddings_dict = {questions[i]: embeddings[i] for i in range(len(questions))}
        print(f"Number of questions: {len(questions)}")
        print(f"Number of embeddings: {len(embeddings_dict)}")
        print(f"Number of duplications: {len(questions)-len(embeddings_dict)}")
        return embeddings_dict

    def load_train_data(self):
        train_questions = []
        train_rewards = []
        group_ids = []
        for model_name, item in self.data_train.items():
            train_questions_group = item['questions']
            train_rewards_group = item['rewards']
            train_questions.extend(train_questions_group)
            train_rewards.extend(train_rewards_group)
            group_ids.extend([model_name]*len(train_questions_group))
        assert len(train_rewards) == len(train_questions), "train_rewards and train_questions must have the same length"
        assert len(group_ids) == len(train_rewards), "group_ids and train_rewards must have the same length"
        
        ref_candidate_questions = [item["questions"] for model_name, item in self.data_ref.items()]
        for i in range(len(ref_candidate_questions)):
            assert ref_candidate_questions[0] == ref_candidate_questions[i], f"Questions in ref_data are not in the same order for all models"
        ref_candidate_questions=ref_candidate_questions[0]
        group_ref_candidate_labels = {model_name:item["rewards"] for model_name, item in self.data_ref.items()}
        return train_questions, train_rewards, group_ids, ref_candidate_questions, group_ref_candidate_labels

    def load_test_data(self,data,test_group_id,ref_size,seed=42):
        group_questions = data[test_group_id]['questions']
        group_rewards = data[test_group_id]['rewards']
        
        test_questions = []
        test_rewards = {}
        test_ref_questions = []
        test_ref_rewards = {}
        
        random.seed(seed)
        all_indices = list(range(len(group_questions)))
        random.shuffle(all_indices)
        
        test_ref_questions = [group_questions[i] for i in all_indices[:ref_size]]
        test_ref_rewards = [group_rewards[i] for i in all_indices[:ref_size]]
        
        test_questions = [group_questions[i] for i in all_indices[ref_size:]]
        test_rewards = [group_rewards[i] for i in all_indices[ref_size:]]
        test_group_ids = [test_group_id]*len(test_questions)
            
        return test_questions, test_rewards, test_group_ids, test_ref_questions, test_ref_rewards

    
class QuestionDataset(Dataset):
    def __init__(self, group_ids, questions, rewards):
        self.group_ids = group_ids
        self.questions = questions
        self.rewards = rewards

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return self.group_ids[idx], self.questions[idx], self.rewards[idx]
    
    
class QuestionEmbeddingDataset(Dataset):
    def __init__(self, group_ids, questions, rewards, embeddings_dict):
        self.group_ids = group_ids
        self.questions = questions
        self.rewards = rewards
        self.embeddings_dict = embeddings_dict  # Dictionary mapping question to its embedding

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        if isinstance(question, list):
            print(type(question), question)
        embedding = self.embeddings_dict.get(question, None)
        return self.group_ids[idx], question, self.rewards[idx], embedding


class SokobanDataset:
    """Dataset class to load Sokoban data from local storage."""
    
    def __init__(self, data_path: str = "local_data", db_name: str = "tokenSeqDB"):
        """
        Initialize Sokoban dataset loader.
        
        Args:
            data_path: Path to local storage data directory
            db_name: Database name (default: tokenSeqDB)
        """
        self.data_path = data_path
        self.db_name = db_name
        self.client = LocalClient(data_path)
        self.db = self.client[db_name]
    
    def list_available_datasets(self) -> List[str]:
        """List all available Sokoban datasets."""
        try:
            # Check what collections exist in the database
            db_path = Path(self.data_path) / self.db_name
            if not db_path.exists():
                return []
            
            collections = []
            for item in db_path.iterdir():
                if item.is_file() and item.suffix == '.json':
                    # Remove .json extension to get collection name
                    collection_name = item.stem
                    # Filter for Sokoban-related collections
                    if any(keyword in collection_name.lower() for keyword in ['seq.train', 'seq.test']):
                        collections.append(collection_name)
            
            return sorted(collections)
        except Exception as e:
            print(f"Error listing datasets: {e}")
            return []
    
    def load_tokenized_traces(self, dataset_name: str, split: str = "train") -> List[TokenizedTrace]:
        """
        Load tokenized traces from a dataset.
        
        Args:
            dataset_name: Name of the dataset (e.g., "sokoban.7-by-7-walls-2-boxes-2.with-box-40k")
            split: Either "train" or "test"
            
        Returns:
            List of TokenizedTrace objects
        """
        collection_name = f"{dataset_name}.seq.{split}"
        collection = self.db[collection_name]
        
        traces = []
        try:
            for doc in collection.find():
                trace = TokenizedTrace.from_dict(doc)
                traces.append(trace)
        except Exception as e:
            print(f"Error loading traces from {collection_name}: {e}")
            return []
        
        return traces
    
    def load_vocabulary(self, dataset_name: str = None) -> List[str]:
        """
        Load vocabulary for a dataset. Since vocabulary is the same for all datasets,
        we can load from any available dataset.
        
        Args:
            dataset_name: Name of the dataset (optional, will use any available if None)
            
        Returns:
            List of vocabulary tokens
        """
        vocab_collection = self.db["vocabulary"]
        try:
            # Get all vocabulary data
            all_vocab_data = vocab_collection.documents
            
            if not all_vocab_data:
                print("No vocabulary data found")
                return []
            
            # If specific dataset requested, try to find it
            if dataset_name:
                # Direct lookup by dataset name
                if dataset_name in all_vocab_data:
                    vocab_entry = all_vocab_data[dataset_name]
                    if isinstance(vocab_entry, dict) and "vocabulary" in vocab_entry:
                        return vocab_entry["vocabulary"]
                
                # MongoDB-style approach with _id lookup
                vocab_doc = vocab_collection.find_one({"_id": dataset_name})
                if vocab_doc and "vocabulary" in vocab_doc:
                    return vocab_doc.get("vocabulary", [])
            
            # Since vocabulary is the same for all datasets, just return the first one we find
            for key, vocab_entry in all_vocab_data.items():
                if isinstance(vocab_entry, dict) and "vocabulary" in vocab_entry:
                    vocabulary = vocab_entry["vocabulary"]
                    if vocabulary:  # Make sure it's not empty
                        return vocabulary
            
            print("No vocabulary found in any dataset")
            
        except Exception as e:
            print(f"Error loading vocabulary: {e}")
            import traceback
            traceback.print_exc()
        
        return []
    
    def get_shared_vocabulary(self) -> List[str]:
        """
        Get the shared vocabulary (same for all datasets).
        
        Returns:
            List of vocabulary tokens
        """
        return self.load_vocabulary()
    
    def get_dataset_stats(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get statistics for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            "dataset_name": dataset_name,
            "train_count": 0,
            "test_count": 0,
            "vocabulary_size": 0,
            "avg_prompt_len": 0,
            "avg_reasoning_len": 0,
            "avg_plan_len": 0
        }
        
        # Count train and test sequences
        try:
            train_meta = self.db[f"{dataset_name}.meta.train"]
            test_meta = self.db[f"{dataset_name}.meta.test"]
            
            stats["train_count"] = train_meta.count_documents({})
            stats["test_count"] = test_meta.count_documents({})
            
            # Get vocabulary size
            vocab = self.load_vocabulary(dataset_name)
            stats["vocabulary_size"] = len(vocab)
            
            # Calculate average lengths from metadata
            train_docs = list(train_meta.find())
            if train_docs:
                stats["avg_prompt_len"] = np.mean([doc.get("prompt_len", 0) for doc in train_docs])
                stats["avg_reasoning_len"] = np.mean([doc.get("reasoning_len", 0) for doc in train_docs])
                stats["avg_plan_len"] = np.mean([doc.get("plan_len", 0) for doc in train_docs])
        
        except Exception as e:
            print(f"Error computing stats for {dataset_name}: {e}")
        
        return stats

