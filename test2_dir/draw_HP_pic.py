import csv
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind,mannwhitneyu

def get_data(filename):
    y_values_from_csv = []
    print(filename)
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            y_values_from_csv.append(float(row[0]))
    return y_values_from_csv

print("input ID of user1 (000)")
User1_name = input()
print("input ID of user2 (000)")
User2_name = input()
print("input ID of user3 (000)")
User3_name = input()


User1_RB=get_data("./"+User1_name+"/"+User1_name+"_y_valuesRB.csv")
User1_HM=get_data("./"+User1_name+"/"+User1_name+"_y_valuesHM.csv")

User2_RB=get_data("./"+User2_name+"/"+User2_name+"_y_valuesRB.csv")
User2_HM=get_data("./"+User2_name+"/"+User2_name+"_y_valuesHM.csv")

User3_RB=get_data("./"+User3_name+"/"+User3_name+"_y_valuesRB.csv")
User3_HM=get_data("./"+User3_name+"/"+User3_name+"_y_valuesHM.csv")

print("hp User1 RB vs HM: mannwhitneyu, p_value",mannwhitneyu(User1_RB, User1_HM))
print("hp User2 RB vs HM: mannwhitneyu, p_value",mannwhitneyu(User2_RB, User2_HM))
print("hp User3 RB vs HM: mannwhitneyu, p_value",mannwhitneyu(User3_RB, User3_HM))

print("hp User1 RB vs HM: t_stat, p_value",ttest_ind(User1_RB, User1_HM))
print("hp User2 RB vs HM: t_stat, p_value",ttest_ind(User2_RB, User2_HM))
print("hp User3 RB vs HM: t_stat, p_value",ttest_ind(User3_RB, User3_HM))

data = [User1_RB, User1_HM, User2_RB, User2_HM, User3_RB, User3_HM]
labels = ['User1', 'User1', 'User2', 'User2', 'User3', 'User3']
colors = ['red', 'green', 'red', 'green', 'red', 'green']

# 各ユーザのHMとRBの距離を少し近づけるために、x軸の位置を調整します
positions = [1.25, 1.75, 3.25, 3.75, 5.25, 5.75]
box = plt.boxplot(data, vert=True, patch_artist=True, positions=positions)

# x軸のラベルをユーザ名だけに設定します
plt.xticks([1.5, 3.5, 5.5], ['User1', 'User2', 'User3'])

# 箱ひげ図の色を設定します
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

# y軸のラベルを変更します
plt.ylabel('Hemodynamic Profile')
plt.grid(True)

# 凡例を追加します
plt.legend(handles=[plt.Line2D([0], [0], color='red', lw=4),
                    plt.Line2D([0], [0], color='green', lw=4)],
           labels=['robot', 'human'])

filename= User1_name+"_"+User2_name+"_"+User3_name+"_HPstatistics_test"
#plt.savefig(filename+".eps")
plt.savefig(filename+".png")
