from matplotlib import pyplot as plt

friends = [70, 65, 72, 63, 71, 64, 60, 64, 67]
minutes = [175, 170, 205, 120, 220, 130, 105, 145, 190]
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

plt.scatter(friends, minutes)

for label, friend_cnt, minute_cnt in zip(labels, friends, minutes):
    plt.annotate(label, xy=(friend_cnt, minute_cnt), xytext=(10, -10), textcoords='offset points')

plt.annotate('annotate', xy=(72, 205), xytext=(66, 170), fontsize=8,
             arrowprops=dict(facecolor='green', edgecolor='green', headlength=5))
plt.title("Daily Minutes vs. Number of Friends")
plt.xlabel("# of friends")
plt.ylabel("daily minutes spent on the site")
# plt.axis("equal")
plt.show()
