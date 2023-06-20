import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap


subj_dir = "../data/subjective/form/"
out_dir = "../outputs/evaluation/subjective/"
os.makedirs(out_dir, exist_ok=True)

def get_all_subjective_results():
    answers = pd.read_csv(subj_dir + "answers.csv")

    answers = answers.drop(columns=['Marca temporal', 'Nombre de usuario', 'Motivo de la elección / Reason of choice (optional)'])
    answers = answers.drop(columns=[f'Motivo de la elección / Reason of choice (optional).{i}' for i in range(1,20)])
    quest_num = [i+1 for i in range(20)]
    answers.columns = quest_num

    true_labels = pd.read_csv(subj_dir + "transformer_order.csv")
    true_labels = true_labels.melt(id_vars=['question'], value_vars=['single', 'multi'], var_name='column')
    true_labels = true_labels.pivot_table(index='question', columns='value', values='column', aggfunc='first')
    true_labels.columns = ['A', 'B']


    # Get scores per model (number of choices per image)
    scores = {"single": np.zeros(20), "multi": np.zeros(20), "Ambas / Both": np.zeros(20)}
    for col in answers:
        for i, row_value in answers[col].items():
            if row_value not in ["A", "B"]:
                scores["Ambas / Both"][col-1] += 1
            else:
                scores[true_labels[row_value][col]][col-1] += 1

    scores = pd.DataFrame(scores)

    overall_pct = (scores.sum()/scores.sum().sum()) * 100
    question_pct = scores.apply(lambda x: x / x.sum() * 100, axis=1)


    # Pie chart for overall percentages
    fig = plt.figure(figsize=(5,5))
    plt.pie(overall_pct, labels=scores.columns, autopct='%1.1f%%', textprops={'size': 14})
    plt.title('Overall Voting Percentages', fontsize=15)
    plt.savefig(out_dir + "overall_pie.png")

    # Bar plot for question percentages
    ax = question_pct.plot(kind='bar', figsize=(10,5))
    ax.set_title('Percentages of choice per question', fontsize=15)
    ax.set_xlabel('Question Number', fontsize=12)
    ax.set_ylabel('Choice percentage (%)', fontsize=12)
    ax.legend()
    plt.savefig(out_dir + "question_pct.png")


    # Print how many times each option has been the most voted one
    overall_winner = question_pct.idxmax(axis=1).value_counts().to_frame().transpose()

    # Print how many times each option, excluding "Both", has been the most voted one
    tmp = question_pct.drop(columns=["Ambas / Both"])
    model_winner = tmp.idxmax(axis=1).value_counts().to_frame().transpose()


    # Count how many times each person has voted each option
    scores = [{"single":0, "multi":0, "Ambas / Both":0} for _ in range(len(answers.index))]
    for i, row in answers.iterrows():
        for quest, ans in enumerate(row):
            if ans in ["A", "B"]:
                scores[i][true_labels.iloc[quest][ans]] += 1
            else:
                scores[i]["Ambas / Both"] += 1

    person_winner = pd.DataFrame(scores).idxmax(axis=1).value_counts().to_frame().transpose()

    df = pd.DataFrame(columns=['single', 'multi', 'Ambas / Both'])

    for i, winner in enumerate([overall_winner, model_winner, person_winner]):
        df.loc[i] = [winner[col].iloc[0] if col in winner.columns else 0 for col in df.columns]

    group_names = ['Most chosen\noption', 'Most chosen\nmodel', 'Option preference\nby person']

    # create the bar plot
    ax = df.plot.bar(rot=0, figsize=(6, 4))
    ax.set_ylabel('Times an option was chosen', fontsize=12)
    ax.set_xlabel('Types of comparisons', fontsize=12)
    ax.set_xticklabels(group_names)
    ax.set_title("Counts of winning scenarios (most chosen option)", fontsize=15)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(out_dir + "winners.png")


def summary_image():
    img_dir = "../data/subjective/best/img/"
    txt_dir = "../data/subjective/best/txt/"
    imgs = ["_original.jpg", "_single.jpg", "_multi.jpg"]
    column_titles = [" ", "Original", "Single", "Multi", "Attributes Modified"]

    for part in [0, 1]:
        fig, ax = plt.subplots(11, 5, figsize=(14, 30))
        for c in range(5):
            ax[0, c].text(0.5, 0.05, column_titles[c], ha='center', va='bottom', fontsize=15, fontweight="bold")
            ax[0, c].axis('off')

        for i in range(1,11):
            attrs = []
            with open(txt_dir + f"{i+part*10}_attrs.txt", "r") as f:
                for l in f.readlines():
                    attrs.append(l.split(":")[0])
            attrs = ", ".join(attrs)
            attrs = textwrap.fill(attrs, width=30)

            ax[i, 0].text(0.5, 0.5, f'Question {i+part*10}', ha='center', va='center', fontsize=15)
            ax[i, 0].axis('off')  # Turn off the axis for the row number subplot

            for j in range(3):
                image = plt.imread(img_dir + str(i+part*10) + imgs[j])
                ax[i, j+1].imshow(image)
                ax[i, j+1].axis('off')

            ax[i, 4].text(0, 0.5, attrs, ha='left', va='center', fontsize=15)
            ax[i, 4].axis('off')

        plt.tight_layout(pad=0, h_pad=0, w_pad=0)
        plt.savefig(out_dir + f"form_{part}.jpg")



if __name__ == "__main__":
    summary_image()
