# synthetic data generator

built this for a research internship application. the idea is to generate fake data that has the same statistical properties as real data, useful when you cant share actual data (like patient records).

uses a healthcare dataset as an example.

## what it does

- creates a sample dataset (age, bmi, blood pressure etc.)
- generates synthetic version of it using numpy (no fancy library)
- compares the two using ks-test and some plots

## how to run

```bash
pip install pandas numpy matplotlib seaborn scipy
python synthetic_data_generator.py
```

outputs:
- `real_data.csv` and `synthetic_data.csv`
- `comparison.png` — kde + boxplots
- `categorical.png` — gender/smoker distribution

## why synthetic data

real datasets often cant be shared due to privacy. synthetic data keeps the statistical patterns without exposing actual records. useful for training ml models when data is scarce too.

## planned improvements

- try using a GAN for more realistic generation
- add correlation preservation between columns
- maybe build a small gradio ui

---
2nd year btech aiml project
