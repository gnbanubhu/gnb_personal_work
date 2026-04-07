import matplotlib.pyplot as plt

categories = ["Python", "Java", "JavaScript", "C++", "Go"]
values = [45, 30, 25, 20, 15]

plt.figure(figsize=(8, 5))
plt.bar(categories, values, color=["steelblue", "coral", "mediumseagreen", "orchid", "goldenrod"])
plt.title("Most Popular Programming Languages")
plt.xlabel("Language")
plt.ylabel("Popularity (%)")
plt.ylim(0, 60)

for i, v in enumerate(values):
    plt.text(i, v + 1, f"{v}%", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("bar_chart.png")
plt.show()
print("Bar chart saved as bar_chart.png")
