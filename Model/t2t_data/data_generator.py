import os
import tensorflow as tf
# from tensor2tensor.data_generators import text_encoder
# from tensor2tensor.data_generators.generator_utils import generate_files
# from tensor2tensor.utils import registry

from tensorflow import text_encoder
from tensorflow import generate_files
from tensorflow import registry

# Custom problem import
from grammar_problem import GrammarCorrectionProblem

def create_tfrecords(data_dir, vocab_path, output_dir, problem_name, num_shards):
    """
    Generate TFRecords from raw data for a specified Tensor2Tensor problem.

    Args:
        data_dir: Path to the directory containing raw source and target files.
        vocab_path: Path to the SentencePiece vocab model.
        output_dir: Path to the directory where TFRecords will be saved.
        problem_name: Name of the Tensor2Tensor problem.
        num_shards: Number of shards to split the data into.
    """
    # Load the problem
    problem = registry.problem(problem_name)
    
    # Initialize the vocabulary (SentencePiece model)
    vocab = text_encoder.SubwordTextEncoder(vocab_path)
    
    # Generate TFRecords for each dataset split
    for dataset_split in [problem.DatasetSplit.TRAIN, 
                          problem.DatasetSplit.EVAL, 
                          problem.DatasetSplit.TEST]:
        
        # Define paths to source and target files
        split_map = {
            problem.DatasetSplit.TRAIN: ("train.source", "train.target"),
            problem.DatasetSplit.EVAL: ("dev.source", "dev.target"),
            problem.DatasetSplit.TEST: ("test.source", "test.target"),
        }
        source_file, target_file = split_map[dataset_split]
        source_path = os.path.join(data_dir, source_file)
        target_path = os.path.join(data_dir, target_file)
        
        # Check for file existence
        if not os.path.exists(source_path) or not os.path.exists(target_path):
            print(f"Skipping split {dataset_split}: Missing files.")
            continue
        
        # Generate the output filename
        split_name = {
            problem.DatasetSplit.TRAIN: "train",
            problem.DatasetSplit.EVAL: "dev",
            problem.DatasetSplit.TEST: "test",
        }[dataset_split]
        
        output_file = os.path.join(output_dir, f"grammar_correction-{split_name}")
        
        # Generate TFRecords
        print(f"Generating TFRecords for {split_name}...")
        with tf.io.gfile.GFile(source_path, "r") as source_f, \
             tf.io.gfile.GFile(target_path, "r") as target_f:
            
            examples = []
            for src_line, tgt_line in zip(source_f, target_f):
                # Tokenize inputs and targets using the vocab
                inputs = vocab.encode(src_line.strip())
                targets = vocab.encode(tgt_line.strip())
                examples.append({"inputs": inputs, "targets": targets})
            
            # Write the TFRecords in sharded format
            generate_files(
                examples,
                output_prefix=output_file,
                num_shards=num_shards
            )
        print(f"TFRecords for {split_name} saved to {output_file}-00000-of-{num_shards:05d}")

if __name__ == "_main_":
    # Directory paths
    DATA_DIR = "data"
    VOCAB_PATH = "vocab/vocab.model"
    OUTPUT_DIR = "t2t_data"
    PROBLEM_NAME = "grammar_correction_problem"
    NUM_SHARDS = 10  # Number of shards for training data
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate TFRecords
    create_tfrecords(DATA_DIR, VOCAB_PATH, OUTPUT_DIR, PROBLEM_NAME, NUM_SHARDS)