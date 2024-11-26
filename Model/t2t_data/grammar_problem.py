from tensorflow import problem, text_problems
from tensorflow import registry

# from tensor2tensor import problem, text_problems
# from tensor2tensor import registry

@registry.register_problem
class GrammarCorrectionProblem(text_problems.Text2TextProblem):
    @property
    def approx_vocab_size(self):
        return 2**15  
    
    @property
    def is_generate_per_split(self):
        return False

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        dataset_map = {
            'train': ('data/train.source', 'data/train.target'),
            'dev': ('data/dev.source', 'data/dev.target'),
            'test': ('data/test.source', 'data/test.target')
        }
        src_file, tgt_file = dataset_map[dataset_split]
        with open(src_file, 'r') as src, open(tgt_file, 'r') as tgt:
            for source, target in zip(src, tgt):
                yield {"inputs": source.strip(), "targets": target.strip()}
