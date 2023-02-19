from tqdm import tqdm

class SquadExample(object):
    """
    A single training/test example for the Squad dataset, as loaded from disk.

    Args:
        qas_id: The example's unique identifier
        question_text: The question string
        context_text: The context string
        answer_text: The answer string
        start_position_character: The character position of the start of the answer
        title: The title of the example
        answers: None by default, this is used during evaluation. Holds answers as well as their start positions.
        is_impossible: False by default, set to True if the example has no possible answer.
    """

    def __init__(
        self,
        qas_id,
        question_text,
        context_text,
        answer_text,
        start_position_character,
        title,
        answers=[],
        is_impossible=False,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.answer_text = answer_text
        self.title = title
        self.is_impossible = is_impossible
        self.answers = answers

        self.start_position, self.end_position = 0, 0

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        # Split on whitespace so that different tokens may be attributed to their original position.
        for c in self.context_text:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset

        # Start end end positions only has a value during evaluation.
        if start_position_character is not None and not is_impossible:
            self.start_position = char_to_word_offset[start_position_character]
            self.end_position = char_to_word_offset[
                min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
            ]

def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def _create_examples(input_data, set_type):
    is_training = set_type == "train"
    examples = []
    for entry in tqdm(input_data):
        title = entry["title"]
        for paragraph in entry["paragraphs"]:
            context_text = paragraph["context"]
            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position_character = None
                answer_text = None
                answers = []
                sub_context = ""

                if "is_impossible" in qa:
                    is_impossible = qa["is_impossible"]
                else:
                    is_impossible = False

                if not is_impossible:
                    if is_training:
                        answer = qa["answers"][0]
                        answer_text = answer["text"]
                        start_position_character = answer["answer_start"]
                        end_position_character = start_position_character+len(answer_text)
                        if start_position_character - 500 < 0:
                            start_context_position = 0 
                        else:
                            start_context_position = start_position_character - 500
                            start_position_character = 500
                        end_context_position = len(context_text) if end_position_character+500 > len(context_text) else end_position_character+500
                        sub_context = context_text[start_context_position:end_context_position]                        
                    else:
                        answers = qa["answers"]
                example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=sub_context,
                        answer_text=answer_text,
                        start_position_character=start_position_character,
                        title=title,
                        is_impossible=is_impossible,
                        answers=answers,
                    )

                examples.append(example)
    return examples