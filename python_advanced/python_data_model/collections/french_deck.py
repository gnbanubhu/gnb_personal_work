import random
from collections import namedtuple


Card = namedtuple("Card", ["rank", "suit"])

suit_values = {"spades": 3, "hearts": 2, "diamonds": 1, "clubs": 0}


def card_value(card):
    rank_value = FrenchDeck.ranks.index(card.rank)
    return rank_value * len(suit_values) + suit_values[card.suit]


class FrenchDeck:
    ranks = [str(n) for n in range(2, 11)] + list("JQKA")
    suits = "spades diamonds clubs hearts".split()

    def __init__(self):
        self._cards = [Card(rank, suit) for suit in self.suits for rank in self.ranks]

    def __len__(self):
        return len(self._cards)

    def __getitem__(self, position):
        return self._cards[position]


if __name__ == "__main__":
    deck = FrenchDeck()

    print("--- Deck Info ---")
    print(f"Total cards : {len(deck)}")
    print(f"First card  : {deck[0]}")
    print(f"Last card   : {deck[-1]}")

    print("\n--- Random Card ---")
    print(f"Random pick : {random.choice(deck)}")

    print("\n--- Slicing (first 5 cards) ---")
    for card in deck[:5]:
        print(f"  {card}")

    print("\n--- Slicing (every 13th card — Aces) ---")
    for card in deck[12::13]:
        print(f"  {card}")

    print("\n--- Iteration (first 5) ---")
    for card in deck[:5]:
        print(f"  {card}")

    print("\n--- Membership check ---")
    print(f"Card('A', 'spades') in deck : {Card('A', 'spades') in deck}")
    print(f"Card('Z', 'spades') in deck : {Card('Z', 'spades') in deck}")

    print("\n--- Sorted by card value (last 5) ---")
    sorted_deck = sorted(deck, key=card_value)
    for card in sorted_deck[-5:]:
        print(f"  {card}")
