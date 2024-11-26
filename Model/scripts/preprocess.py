import os
from sklearn.model_selection import train_test_split

def preprocess_and_split(data_dir, source_file, target_file, output_dir, split_ratios=(0.8, 0.1, 0.1)):
    # Load data
    with open(source_file, 'r') as src, open(target_file, 'r') as tgt:
        source_data = src.readlines()
        target_data = tgt.readlines()

    assert len(source_data) == len(target_data), "Mismatch in number of source and target sentences."

    # Split data
    train_src, temp_src, train_tgt, temp_tgt = train_test_split(
        source_data, target_data, test_size=split_ratios[1] + split_ratios[2], random_state=42
    )
    val_src, test_src, val_tgt, test_tgt = train_test_split(
        temp_src, temp_tgt, test_size=split_ratios[2] / (split_ratios[1] + split_ratios[2]), random_state=42
    )

    # Save to files
    os.makedirs(output_dir, exist_ok=True)
    for name, data in [("train.source", train_src), ("train.target", train_tgt),
                       ("dev.source", val_src), ("dev.target", val_tgt),
                       ("test.source", test_src), ("test.target", test_tgt)]:
        with open(os.path.join(output_dir, name), 'w') as f:
            f.writelines(data)

if __name__ == "__main__":
    data_dir = "./data"
    output_dir = "./data"
    preprocess_and_split(
        data_dir=data_dir,
        source_file=os.path.join(data_dir, "source.txt"),
        target_file=os.path.join(data_dir, "target.txt"),
        output_dir=output_dir
    )
