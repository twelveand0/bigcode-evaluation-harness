"""Program Synthesis with Large Language Models
https://arxiv.org/abs/2108.07732

The benchmark consists of around 1,000 crowd-sourced Python programming problems, 
designed to be solvable by entry level programmers, covering programming fundamentals, 
standard library functionality, and so on. Each problem consists of a task description, 
code solution and 3 automated test cases. As described in the paper, a subset of the data
has been hand-verified by the authors.

Homepage:: https://github.com/google-research/google-research/tree/master/mbpp
"""

import re

from evaluate import load
from datasets import load_dataset


from bigcode_eval.base import Task

_CITATION = """
@article{austin2021program,
  title={Program Synthesis with Large Language Models},
  author={Austin, Jacob and Odena, Augustus and Nye, Maxwell and Bosma, Maarten and Michalewski, Henryk and Dohan, David and Jiang, Ellen and Cai, Carrie and Terry, Michael and Le, Quoc and others},
  journal={arXiv preprint arXiv:2108.07732},
  year={2021}
}
"""


class MBPP(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "mbpp"

    def __init__(self):
        super().__init__(
            stop_words=['\nclass', '\nassert', '\n"""', '\nprint', '\nif', '\n<|/'],
            requires_execution=True
        )
        self.dataset = load_dataset(
            'json', 
            data_files=f'local_benchmarks/mbpp/mbpp.jsonl',
            split='train'
        )


    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        ##dataset = self.dataset["test"]
        dataset = self.dataset.select(range(10, 510))
        # the wrong split of mbpp can be loaded with old datasets cache
        assert (
            len(dataset) == 500
        ), "please ensure you have the latest version of MBPP dataset, try deleting its old cache"
        return dataset

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from.
        MBPP prompt is built following to InCoder (Fried et al.) approach
        prompt = docstring that includes one test
        """
        description = doc["text"]
        test_example = doc["test_list"][0]
        prompt = f'"""\n{description}\n{test_example}\n"""\n'
        return prompt

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return "\n".join(doc["test_list"])

    @staticmethod
    def _stop_at_stop_token(decoded_string, stop_tokens):
        """
        Produces the prefix of decoded_string that ends at the first occurrence of
        a stop_token.
        WARNING: the decoded_string *must not* include the prompt, which may have stop tokens
        itself.
        """
        min_stop_index = len(decoded_string)
        for stop_token in stop_tokens:
            stop_index = decoded_string.find(stop_token)
            if stop_index != -1 and stop_index < min_stop_index:
                min_stop_index = stop_index
        return decoded_string[:min_stop_index]

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
        """

        prompt = self.get_prompt(self.get_dataset()[idx])
        generation = generation[len(prompt) :]
        generation = self._pick_code_block(generation)
        return prompt + self._stop_at_stop_token(generation, self.stop_words)

    def _pick_code_block(self, generation):
        lines = generation.split('\n')
        new_generation = ''
        if '```' in generation:
            in_code = False
            for line in lines:
                if not in_code and line.startswith('```'):
                    in_code = True
                    continue
                if in_code:
                    if line.rstrip().endswith('```'):
                        new_generation += line.rstrip()[:-3] + '\n'
                        break
                    new_generation += line + '\n'
        else:
            in_code = False
            for line in lines:
                if not in_code and line.startswith('def '):
                    in_code = True
                    new_generation += line + '\n'
                    continue
                if in_code:
                    if len(line) > 0 and line[0].strip() != '' and not line.startswith('def'):
                        break
                    new_generation += line + '\n'
        return new_generation

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        code_metric = load("code_eval")
        results, _ = code_metric.compute(
            references=references,
            predictions=generations,
        )
        return results
