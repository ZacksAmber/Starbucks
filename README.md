# Starbucks

> [Starbucks Take Home Assignment](https://drive.google.com/file/d/18klca9Sef1Rs6q8DW4l7o349r8B70qXM/view)

An analysis of target marketing, through A/B testing and Machine Learning. Data is from Starbucks' take-home assignment.

---

## TL;DR

In this project, I explore the relationship between promotion strategy and purchase data from Starbucks. I am curious about two questions:

1. Do promotions bring more purchases, leading to revenue increment?
2. If so, how to maximize the purchase, resulting in better IRR and NIR?

Therefore, there are one prerequisite and two main parts:

- Exploratory Data Analysis
- Part I: Applying A/B Testing for answering the 1st question.
- Part II: Implementing Machine Learning for answering the 2nd question.

---

### Guidelines

Part I: A/B Test Flowchart

![General A/B Testing Flowchart](https://raw.githubusercontent.com/ZacksAmber/Flowcharts/main/general_ab_testing.svg)

Part II: Machine Learning

![General Machine Learning Flowchart](https://raw.githubusercontent.com/ZacksAmber/Flowcharts/main/general_ab_testing.svg)

---

## Conclusions

### Exploratory Data Analysis

There are 7 features (`V1` to `V7`), 1 target (`purchase`), and one more feature (`Promotion`), which is the model prediction. In the dataset `../data/training.csv`, `Promotion` was randomly assigned with the value `'Yes'` or `'No'` by A/B testing. Regarding all columns in dataset `training.csv`.

- There is no Missing Values issue.
- There is an Imbalanced Dataset issue and we will handle it in the Machine Learning part.
- There is no Outliers issue.

### Part I: A/B Testing

$$\displaystyle IRR = \frac{purch_{treat}}{cust_{treat}} - \frac{purch_{ctrl}}{cust_{ctrl}}$$
$$\displaystyle H_0: IRR = 0$$
$$\displaystyle H_1: IRR > 0$$
$$\displaystyle \alpha = 0.05$$
$$\displaystyle p-value = 5.55 \times 10^{-36}$$

With the p-value of $5.55 \times 10^{-36}$, which is extremely small, we can reject the null hypothesis, which states that there is no difference between the two groups. And we are more than 99% confident that there is a statistically significant difference in purchase rate between the control and treatment. This means that the promotion has had a significant impact on the purchase rate. Therefore, we can infer that the promotion has been successful in increasing the purchase rate compared to the control group where no promotion was provided. We can recommend the continuation of the promotion strategy to increase sales.

It's important to note that statistical significance does not necessarily imply practical significance. And for this project, there is pre-defined practical significance. But the randomly assigned `Promotion` in `training.csv` shows that, with a `Promotion`, the `purchase` increased by **125%**. It means, by applying Machine Learning for better sending out promotions, we can get a better IRR.

---

### Part II: Machine Learning

My tuned model with proper hyperparameters can significantly increase the IRR and NIR. Therefore, Starbucks should apply this ML model as a promotion strategy for IRR and NIR improvement.

- In comparison to the basic IRR (random promotion strategy), my solution increased IRR by 0.0109, or 114.73%;
- In comparison to the basic NIR (random promotion strategy), my solution increased NIR by 2853.5;
- In comparison to the Starbucks IRR (Starbucks strategy), my solution increased IRR by 0.0016, or 8.51%;
- In comparison to the Starbucks NIR (Starbucks strategy), my solution increased NIR by 329.45, or 173.90%;

![IRR Line Plot](reports/img/irr_line_plot.png)

![NIR Line Plot](reports/img/nir_line_plot.png)

---

## Instructions

### Clone GitHub Repo & Reproduce the Code

> [Pipenv](https://pypi.org/project/pipenv/)
> [GitHub Repo](https://github.com/ZacksAmber/Starbucks)

```sh
# clone repo
git clone https://github.com/ZacksAmber/Starbucks.git

# change dir
cd Starbucks

# make virtual env & activate it
pipenv shell
# install necessary packages
pipenv install
# add the virtualenv as a jupyter kernel
ipython kernel install --name "Starbucks" --user
# run jupyter notebook or lab
jupyter lab


# remove kernel if you would like to
# jupyter kernelspec remove <Starbucks>
```

---

### File Structure

```sh
.
├── LICENSE
├── Pipfile
├── Pipfile.lock  # Pipenv version requirements.txt
├── README.md
├── data  # storing training and test dataset
│   ├── Test.csv
│   └── training.csv
├── notebooks  # storing notebooks
│   ├── Metric_Baseline.ipynb
│   ├── Model_Baseline.ipynb
│   ├── Model_Tuning.ipynb
│   └── Starbucks.ipynb  # main notebook
├── reports  # storing outputs from notebooks/
│   ├── baselines  # storing outputs of baseline
│   │   ├── metric_baseline.csv
│   │   └── model_baseline.csv
│   ├── img  # storing exported images
│   │   ├── irr_line_plot.png
│   │   ├── nir_line_plot.png
│   │   ├── power.png
│   │   ├── purchase_funnel.png
│   │   └── sample_size.png
│   └── tuning_results  # storing tuning result
│       └── model_tuning.csv
├── requirements.txt
└── src
    └── test_results.py
```
