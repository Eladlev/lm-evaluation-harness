"""
Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge
https://arxiv.org/pdf/1803.05457.pdf

The ARC dataset consists of 7,787 science exam questions drawn from a variety
of sources, including science questions provided under license by a research
partner affiliated with AI2. These are text-only, English language exam questions
that span several grade levels as indicated in the files. Each question has a
multiple choice structure (typically 4 answer options). The questions are sorted
into a Challenge Set of 2,590 “hard” questions (those that both a retrieval and
a co-occurrence method fail to answer correctly) and an Easy Set of 5,197 questions.

Homepage: https://allenai.org/data/arc
"""
from lm_eval.base import MultipleChoiceTask
from typing import Callable, List, Mapping, Optional, Tuple, Union
from lm_eval.base import Request, rf, Task
from lm_eval.metrics import mean

_CITATION = """
@article{Clark2018ThinkYH,
  title={Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge},
  author={Peter Clark and Isaac Cowhey and Oren Etzioni and Tushar Khot and Ashish Sabharwal and Carissa Schoenick and Oyvind Tafjord},
  journal={ArXiv},
  year={2018},
  volume={abs/1803.05457}
}
"""


class AQuA_COT(Task):
    VERSION = 0
    DATASET_PATH = "aqua_rat"
    max_generation_length = 200
    prompt_template = '''Answer the following question step by step.
Q: John found that the average of 15 numbers is 40. If 10 is added to each number then the mean of the
numbers is? Answer Choices: (a) 50 (b) 45 (c) 65 (d) 78 (e) 64
A: If 10 is added to each number, then the mean of the numbers also increases by 10. So the new mean
would be 50. The answer is (a).
Q: If a / b = 3/4 and 8a + 5b = 22,then find the value of a. Answer Choices: (a) 1/2 (b) 3/2 (c) 5/2 (d) 4/2 (e) 7/2
A: If a / b = 3/4, then b = 4a / 3. So 8a + 5(4a / 3) = 22. This simplifies to 8a + 20a / 3 = 22, which means
44a / 3 = 22. So a is equal to 3/2. The answer is (b).
Q: A person is traveling at 20 km/hr and reached his destiny in 2.5 hr then find the distance? Answer Choices:
(a) 53 km (b) 55 km (c) 52 km (d) 60 km (e) 50 km
A: The distance that the person traveled would have been 20 km/hr * 2.5 hrs = 50 km. The answer is (e).
Q: How many keystrokes are needed to type the numbers from 1 to 500? Answer Choices: (a) 1156 (b) 1392 (c) 1480 (d) 1562 (e) 1788
A: There are 9 one-digit numbers from 1 to 9. There are 90 two-digit numbers from 10 to 99. There are 401
three-digit numbers from 100 to 500. 9 + 90(2) + 401(3) = 1392. The answer is (b).'''

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def doc_to_target(self, doc):
        return " " + doc["answer"]

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {"acc": True}

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {"acc": mean}

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        # NOTE: Some `doc["answerKey"]`s are in numeric string format being one
        # of {'1', '2', '3', '4', '5'}. We map them back to letters.
        # num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
        # doc["answerKey"] = num_to_letter.get(doc["answerKey"], doc["answerKey"])
        # out_doc = {
        #     "id": doc["id"],
        #     "query": "Question: " + doc["question"] + "\nAnswer:",
        #     "choices": doc["choices"]["text"],
        #     "gold": ["A", "B", "C", "D", "E"].index(doc["answerKey"]),
        # }
        return doc

    def max_generation_length(self) -> Optional[int]:
        """Denote where the max length of the generation if it is obvious from the task."""
        return self.max_generation_length

    def stop_sequences(self) -> str:
        """Denote where the generation should end based on the few-shot example
        separator.

        NOTE: Override this if you want to use a sequence other than just the
        task's few-shot example separator.
        """
        return ').'

    def construct_requests(self, doc: dict, ctx: str) -> List[Request]:
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        Args:
            doc (dict):
                The document as returned from training_docs, validation_docs, or
                test_docs.
            ctx (str):
                The context string, generated by fewshot_context. This includes
                the natural language description, as well as the few shot examples,
                and the question part of the document for `doc`.
            args (dict):
                The specifics of the context, including number of few shots.

        Returns:
            An iterable of `Request` objects.
        """
        requests = []
        cont_request = rf.cfg_until(ctx, {'until': 'Q:'})
        requests.append(cont_request)
        return requests

    def doc_to_text(self, doc):
        ans = doc['options']
        ans = ['(' + t[:1].lower() + t[1:].lower() for t in ans]
        cur_prompt = 'Q: {}. Answer Choices: {}'.format(doc['question'], ' '.join(ans))
        cur_prompt = self.prompt_template + '\n' + cur_prompt + '\n' + 'A:'
        return cur_prompt

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]

    def process_results(self, doc, results):
        gold = doc['correct'].lower()
        res_template = 'The answer is ({}).'.format(gold)
        acc = 1.0 if results[0][-len(res_template):] == res_template else 0.0

        return {
            "acc": acc,
        }


class COTChallenge(AQuA_COT):
    DATASET_PATH = "aqua_rat"
