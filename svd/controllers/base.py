import pandas as pd
from cement import Controller, ex
from cement.utils.version import get_version_banner
from ..core.version import get_version

from svd.utils.viz import token_counts, zipf_log, histogram
from svd.core.mining import get_tokenizer, mine, save_sparse_matrix
from svd.utils.misc import curate
from pathlib import Path


VERSION_BANNER = """
Software Vulnerability Detection %s
%s
""" % (get_version(), get_version_banner())


# WORDS SIZE =


class Base(Controller):
    class Meta:
        label = 'base'

        # text displayed at the top of --help output
        description = 'Software Vulnerability Detection'

        # text displayed at the bottom of --help output
        epilog = 'Usage: svd command1 --foo bar'

        # controller level arguments. ex: 'svd --version'
        arguments = [
            ### add a version banner
            ( [ '-v', '--version' ],
              { 'action'  : 'version',
                'version' : VERSION_BANNER } ),
        ]

    def _default(self):
        """Default action if no sub-command is passed."""

        self.app.args.print_help()

    @ex(
        help='Plots Zipf-log of the dataset',

        arguments=[
            ### add a sample foo option under subcommand namespace
            (['-d', '--dataset'], {'help': 'Dataset path (json format)', 'action': 'store', 'required': True}),
        ],
    )
    def zipf(self):
        """Zipf-log plot."""
        dataset = pd.read_json(self.app.pargs.dataset)
        self.app.log.info(f"Loaded dataset: {self.app.pargs.dataset}")
        tokenizer = get_tokenizer(vocab_size=None)
        self.app.log.info("Tokenizing dataset")
        dataset_tokens = [tokenizer(el) for el in dataset.func.to_list()]
        self.app.log.info("Counting tokens")
        tokens, counts = token_counts(dataset_tokens)
        self.app.log.info("Plotting")
        zipf_log(tokens, counts)

    @ex(
        help='Plots histogram of the occurrences of tokens per function in the dataset',

        arguments=[
            ### add a sample foo option under subcommand namespace
            (['-d', '--dataset'], {'help': 'Dataset path (json format)', 'action': 'store', 'required': True}),
        ],
    )
    def histogram(self):
        dataset = pd.read_json(self.app.pargs.dataset)
        self.app.log.info(f"Loaded dataset: {self.app.pargs.dataset}")
        tokenizer = get_tokenizer(vocab_size=None)
        self.app.log.info("Tokenizing dataset")
        dataset_tokens = [tokenizer(el) for el in dataset.func.to_list()]
        self.app.log.info("Counting tokens in functions")
        funcs_size_in_tokens = [len(row) for row in dataset_tokens]
        self.app.log.info("Plotting")
        histogram(data=funcs_size_in_tokens, interval=(0, 1700), bins_size=200)

    @ex(help='Curate dataset',

        arguments=[
            ### add a sample foo option under subcommand namespace
            (['-d', '--dataset'], {'help': 'Dataset path (csv format)', 'type': str, 'required': True}),
            (['-o', '--output'], {'help': 'Dataset output path (csv format)', 'type': str, 'required': True}),
            (['-is', '--func_size'], {'help': 'Function size in tokens', 'type': int, 'default': 1530}),
        ],
    )
    def curate(self):
        dataset = curate(Path(self.app.pargs.dataset), func_size=self.app.pargs.func_size)
        safe_size = len(dataset[dataset['target'] == 0])
        unsafe_size = len(dataset[dataset['target'] == 1])
        self.app.log.info(f"Total: {len(dataset)}, Safe: {safe_size} , Unsafe: {unsafe_size}")

        dataset.to_csv(self.app.pargs.output)

    @ex(help='Mine features from each function in the dataset',

        arguments=[
            ### add a sample foo option under subcommand namespace
            (['-d', '--dataset'], {'help': 'Dataset path (csv format)', 'type': str, 'required': True}),
            (['-o', '--output'], {'help': 'Path to the output directory', 'type': str, 'required': True}),
            (['-nlp', '--nlp_output'], {'help': 'Path to the extracted nlp output directory', 'type': str,
                                         'required': True}),
            (['-vs', '--vocab_size'], {'help': 'Vocab size.', 'type': int, 'default': 3100})
        ],
    )
    def mine(self):
        dataset = pd.read_csv(self.app.pargs.dataset)
        project = 'devign'
        self.app.log.info(f"Loaded dataset: {self.app.pargs.dataset}")
        features = mine(dataset, vocab_size=self.app.pargs.vocab_size, output_path=self.app.pargs.output,
                        project=project)

        save_sparse_matrix(features, output_dir=self.app.pargs.nlp_output, project_name=project)
