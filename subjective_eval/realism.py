from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
from PIL import Image
import numpy as np
import torch
import pandas as pd

subj_dir = "../data/subjective/form/"

def compute_realism_scores():
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

    question_pct = scores.apply(lambda x: x / x.sum() * 100, axis=1)

    lpips_model = lpips.LPIPS(net='vgg')
    psnr_better = 0
    ssim_better = 0
    lpips_better = 0
    worse = []
    diffs_pnsr = {}
    diffs_ssim = {}
    diffs_lpips = {}
    for i in range(1,21):
        original_image_path = f"../data/subjective/best/img/{i}_original.jpg"
        original_image = Image.open(original_image_path)

        transformed_image_path_1 = f"../data/subjective/best/img/{i}_single.jpg"
        transformed_image_1 = Image.open(transformed_image_path_1)

        transformed_image_path_2 = f"../data/subjective/best/img/{i}_multi.jpg"
        transformed_image_2 = Image.open(transformed_image_path_2)

        original = np.array(original_image)
        t1 = np.array(transformed_image_1)
        t2 = np.array(transformed_image_2)

        original_tensor = torch.tensor(original).unsqueeze(0).permute(0, 3, 1, 2)
        t1_tensor = torch.tensor(t1).unsqueeze(0).permute(0, 3, 1, 2)
        t2_tensor = torch.tensor(t2).unsqueeze(0).permute(0, 3, 1, 2)


        psnr_t1 = peak_signal_noise_ratio(original, t1)
        psnr_t2 = peak_signal_noise_ratio(original, t2)

        ssim_t1 = structural_similarity(original, t1, multichannel=True, channel_axis=2)
        ssim_t2 = structural_similarity(original, t2, multichannel=True, channel_axis=2)

        lpips_t1 = lpips_model.forward(original_tensor, t1_tensor).item()
        lpips_t2 = lpips_model.forward(original_tensor, t2_tensor).item()

        print(f"Results for image {i}")
        print("PSNR single:", psnr_t1)
        print("PSNR multi:", psnr_t2)
        if psnr_t2 > psnr_t1:
            psnr_better += 1
        else:
            worse.append(i)
        diffs_pnsr[i] = psnr_t2-psnr_t1
        print()
        print("SSIM single:", ssim_t1)
        print("SSIM multi:", ssim_t2)
        if ssim_t2 > ssim_t1:
            ssim_better += 1
        else:
            worse.append(i)
        diffs_ssim[i] = ssim_t2-ssim_t1
        print()
        print("LPIPS single:", lpips_t1)
        print("LPIPS multi:", lpips_t2)
        if lpips_t1 > lpips_t2:
            lpips_better += 1
        else:
            worse.append(i)
        diffs_lpips[i] = lpips_t2-lpips_t1
        print()

        question_pct.at[i-1, "PSNR"] = diffs_pnsr[i]
        question_pct.at[i-1, "SSIM"] = diffs_ssim[i]
        question_pct.at[i-1, "LPIPS"] = diffs_lpips[i]

        question_pct.at[i-1, "PSNR_s"] = psnr_t1
        question_pct.at[i-1, "SSIM_s"] = ssim_t1
        question_pct.at[i-1, "LPIPS_s"] = lpips_t1

        question_pct.at[i-1, "PSNR_m"] = psnr_t2
        question_pct.at[i-1, "SSIM_m"] = ssim_t2
        question_pct.at[i-1, "LPIPS_m"] = lpips_t2

    print("\nTimes Multi has better results out of 20:")
    print("PSNR: ", psnr_better)
    print("SSIM: ", ssim_better)
    print("LPIPS: ", lpips_better)

    print("\n we did worse in : ", worse)

    print("Sorted diffs for PSNR: ", sorted(diffs_pnsr.items(), key=lambda x: x[1], reverse=True) )
    print("Sorted diffs for SSIM: ", sorted(diffs_ssim.items(), key=lambda x: x[1], reverse=True) )
    print("Sorted diffs for LPIPS: ", sorted(diffs_lpips.items(), key=lambda x: x[1], reverse=True) )

    print(question_pct)

    question_pct.to_csv("../data/subjective/form/quantitative_cmp.csv")


def check_hardcoded():
    df = pd.read_csv("../data/subjective/form/quantitative_cmp.csv")
    df["LPIPS"] = df["LPIPS"] * -1
    df["pct_diff"] = df["multi"] - df["single"]
    df = df.sort_values('pct_diff', ascending=False)

    print(df)
    print(df[["PSNR", "SSIM", "LPIPS"]].head(5).describe())
    print(df[["PSNR", "SSIM", "LPIPS"]].tail(5).describe())

    print("5 head with multi")
    print(df["LPIPS"].head(5).corr(df["multi"]))
    print(df["PSNR"].head(5).corr(df["multi"]))
    print(df["SSIM"].head(5).corr(df["multi"]))

    print("13 tail with multi")
    print(df["LPIPS"].tail(13).corr(df["multi"]))
    print(df["PSNR"].tail(13).corr(df["multi"]))
    print(df["SSIM"].tail(13).corr(df["multi"]))

    print("overall with multi")
    print(df["LPIPS"].corr(df["multi"]))
    print(df["PSNR"].corr(df["multi"]))
    print(df["SSIM"].corr(df["multi"]))

    print(df[["PSNR_s", "PSNR_m", "SSIM_s", "SSIM_m", "LPIPS_s", "LPIPS_m"]].describe())


if __name__ == "__main__":
    # compute_realism_scores()
    check_hardcoded()
