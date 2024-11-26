import sentencepiece as spm

def generate_vocab(input_files, model_prefix, vocab_size=111):
    spm.SentencePieceTrainer.train(
        input=','.join(input_files),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=1.0,
        model_type='unigram'  # Options: unigram, bpe, char, word
    )

if __name__ == "__main__":
    input_files = ["data/train.source", "data/train.target"]
    output_prefix = "vocab/vocab"
    generate_vocab(input_files, output_prefix)
