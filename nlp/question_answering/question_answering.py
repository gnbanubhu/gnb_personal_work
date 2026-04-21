from transformers import pipeline


# Load question answering pipeline using a pre-trained model
qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2"
)


def answer_question(question: str, context: str) -> dict:
    result = qa_pipeline(question=question, context=context)
    return result


def print_answer(question: str, result: dict) -> None:
    print(f"Question : {question}")
    print(f"Answer   : {result['answer']}")
    print(f"Score    : {result['score']:.4f}")
    print(f"Start    : {result['start']}  |  End: {result['end']}")
    print("-" * 65)


if __name__ == "__main__":

    # Context 1 — Technology
    context_tech = """
    Amazon Web Services (AWS) is a subsidiary of Amazon that provides on-demand
    cloud computing platforms and APIs to individuals, companies, and governments.
    AWS was launched in 2006 and has grown to become the world's most comprehensive
    and broadly adopted cloud platform. Andy Jassy became the CEO of Amazon in 2021,
    while Adam Selipsky serves as the CEO of AWS.
    """

    # Context 2 — Science
    context_science = """
    Albert Einstein was a German-born theoretical physicist who developed the theory
    of relativity, one of the two pillars of modern physics. He was born on March 14,
    1879, in Ulm, Germany. Einstein received the Nobel Prize in Physics in 1921 for
    his discovery of the law of the photoelectric effect. He published his special
    theory of relativity in 1905 and his general theory of relativity in 1915.
    """

    # Context 3 — History
    context_history = """
    The Apollo 11 mission was the first crewed lunar landing mission. It was launched
    by NASA on July 16, 1969, and landed on the Moon on July 20, 1969. Neil Armstrong
    became the first human to walk on the Moon, followed by Buzz Aldrin. Michael Collins
    remained in lunar orbit aboard the command module Columbia while Armstrong and Aldrin
    explored the lunar surface.
    """

    print("=" * 65)
    print("            QUESTION ANSWERING RESULTS")
    print("=" * 65)
    print()

    # Technology questions
    print("Context: AWS / Amazon")
    print("=" * 65)
    tech_questions = [
        "When was AWS launched?",
        "Who is the CEO of AWS?",
        "What does AWS provide?",
    ]
    for question in tech_questions:
        result = answer_question(question, context_tech)
        print_answer(question, result)

    print()

    # Science questions
    print("Context: Albert Einstein")
    print("=" * 65)
    science_questions = [
        "Where was Albert Einstein born?",
        "When did Einstein receive the Nobel Prize?",
        "What theory did Einstein develop?",
    ]
    for question in science_questions:
        result = answer_question(question, context_science)
        print_answer(question, result)

    print()

    # History questions
    print("Context: Apollo 11 Mission")
    print("=" * 65)
    history_questions = [
        "When did Apollo 11 land on the Moon?",
        "Who was the first human to walk on the Moon?",
        "What was the name of the command module?",
    ]
    for question in history_questions:
        result = answer_question(question, context_history)
        print_answer(question, result)
