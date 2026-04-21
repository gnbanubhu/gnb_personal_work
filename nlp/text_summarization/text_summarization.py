from transformers import pipeline


# Load summarization pipeline using a pre-trained model
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)


def summarize(text: str, max_length: int = 130, min_length: int = 30) -> str:
    result = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return result[0]["summary_text"]


def print_summary(title: str, original: str, summary: str) -> None:
    print(f"Topic          : {title}")
    print(f"Original Length: {len(original.split())} words")
    print(f"Summary Length : {len(summary.split())} words")
    print(f"\nSummary:\n{summary}")
    print("-" * 70)
    print()


if __name__ == "__main__":

    # Article 1 — Artificial Intelligence
    text_ai = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed
    to natural intelligence displayed by animals including humans. AI research has been
    defined as the field of study of intelligent agents, which refers to any system
    that perceives its environment and takes actions that maximize its chance of
    achieving its goals. The term "artificial intelligence" had previously been used
    to describe machines that mimic and display human cognitive skills associated with
    the human mind, such as learning and problem-solving. This definition has since
    been rejected by major AI researchers who now describe AI in terms of rationality
    and acting rationally, which does not limit how intelligence can be articulated.
    AI applications include advanced web search engines, recommendation systems used
    by YouTube, Amazon, and Netflix, understanding human speech such as Siri and Alexa,
    self-driving cars such as Waymo, generative or creative tools such as ChatGPT and
    AI art, and competing at the highest level in strategic games such as chess and Go.
    As machines become increasingly capable, tasks considered to require intelligence
    are often removed from the definition of AI, a phenomenon known as the AI effect.
    """

    # Article 2 — Climate Change
    text_climate = """
    Climate change refers to long-term shifts in temperatures and weather patterns.
    These shifts may be natural, such as through variations in the solar cycle.
    But since the 1800s, human activities have been the main driver of climate change,
    primarily due to burning fossil fuels like coal, oil and gas. Burning fossil fuels
    generates greenhouse gas emissions that act like a blanket wrapped around the Earth,
    trapping the sun's heat and raising temperatures. Examples of greenhouse gas emissions
    that are causing climate change include carbon dioxide and methane. These come from
    using gasoline for driving a car or coal for heating a building. Clearing land and
    forests can also release carbon dioxide. Landfills for garbage are a major source
    of methane emissions. Energy, industry, transport, buildings, agriculture and land
    use are among the main emitters. Climate change is already affecting every inhabited
    region across the globe, with human influence contributing to many observed changes
    in weather and climate extremes such as heatwaves, heavy precipitation, droughts,
    and tropical cyclones.
    """

    # Article 3 — Machine Learning
    text_ml = """
    Machine learning (ML) is a branch of artificial intelligence (AI) and computer
    science which focuses on the use of data and algorithms to imitate the way that
    humans learn, gradually improving its accuracy. IBM has a rich history with machine
    learning. One of its own, Arthur Samuel, is credited for coining the term machine
    learning with his research around the game of checkers. Robert Nealey, the self-
    proclaimed checkers master, played the game on an IBM 7094 computer in 1962, and
    he lost to the computer. Compared to what can be done today, this feat seems minor,
    but it's considered a major milestone in the field of artificial intelligence.
    Over the last couple of decades, the technological advances in storage and processing
    power have enabled some innovative products based on machine learning, such as
    Netflix's recommendation engine and self-driving cars. Machine learning is an
    important component of the growing field of data science. Through the use of
    statistical methods, algorithms are trained to make classifications or predictions,
    and to uncover key insights in data mining projects. These insights subsequently
    drive decision making within applications and businesses, ideally impacting key
    growth metrics.
    """

    print("=" * 70)
    print("               TEXT SUMMARIZATION RESULTS")
    print("=" * 70)
    print()

    articles = [
        ("Artificial Intelligence", text_ai),
        ("Climate Change", text_climate),
        ("Machine Learning", text_ml),
    ]

    for title, text in articles:
        summary = summarize(text.strip())
        print_summary(title, text.strip(), summary)
