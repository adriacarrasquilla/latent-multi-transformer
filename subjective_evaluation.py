import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("tkagg")

subj_dir = "./data/subjective/form/"

answers = pd.read_csv(subj_dir + "answers.csv")

answers = answers.drop(columns=['Marca temporal', 'Nombre de usuario', 'Motivo de la elección / Reason of choice (optional)'])
answers = answers.drop(columns=[f'Motivo de la elección / Reason of choice (optional).{i}' for i in range(1,20)])
quest_num = [i+1 for i in range(20)]
answers.columns = quest_num

true_labels = pd.read_csv(subj_dir + "transformer_order.csv")
true_labels = true_labels.melt(id_vars=['question'], value_vars=['single', 'multi'], var_name='column')
true_labels = true_labels.pivot_table(index='question', columns='value', values='column', aggfunc='first')
true_labels.columns = ['A', 'B']


scores = {"single": np.zeros(20), "multi": np.zeros(20), "Ambas / Both": np.zeros(20)}
scores_org = {"A": np.zeros(20), "B": np.zeros(20), "Ambas / Both": np.zeros(20)}
for col in answers:
    for i, row_value in answers[col].items():
        if row_value not in ["A", "B"]:
            scores["Ambas / Both"][col-1] += 1
            scores_org["Ambas / Both"][col-1] += 1
        else:
            scores[true_labels[row_value][col]][col-1] += 1
            scores_org[row_value][col-1] += 1

scores = pd.DataFrame(scores)

overall_pct = (scores.sum()/scores.sum().sum()) * 100
question_pct = scores.apply(lambda x: x / x.sum() * 100, axis=1)


# Pie chart for overall percentages
fig = plt.figure(figsize=(10,10))
plt.pie(overall_pct, labels=scores.columns, autopct='%1.1f%%', textprops={'size': 14})
plt.title('Overall Percentages by Column', fontsize=15)
plt.savefig(subj_dir + "overall_pie.png")

# Bar plot for question percentages
ax = question_pct.plot(kind='bar', figsize=(20,10))
ax.set_title('Percentages of choice per question', fontsize=15)
ax.set_xlabel('Question Number', fontsize=12)
ax.set_ylabel('Choice percentage (%)', fontsize=12)
ax.legend()
plt.savefig(subj_dir + "question_pct.png")


max_counts = question_pct.idxmax(axis=1).value_counts()
print(max_counts)

tmp = question_pct.drop(columns=["Ambas / Both"])
max_counts = tmp.idxmax(axis=1).value_counts()
print(max_counts)
