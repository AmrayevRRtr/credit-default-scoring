# Data

`raw/credit_default.csv` - UCI "Default of Credit Card Clients" (Yeh, I.C. & Lien, C.H., 2009, *Expert Systems with Applications*), licensed CC BY 4.0.
Copy of the 24,000-row version from https://github.com/pycaret/datasets (the original has 30,000 rows).

Columns: `LIMIT_BAL` credit limit (NT$); `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`; `PAY_1..PAY_6` repayment status for the last 6 months
(-2 no consumption, -1 paid duly, 0 revolving credit, 1..8 months of delay; PAY_1 = most recent month); `BILL_AMT1..6` bill statements;
`PAY_AMT1..6` payments; `default` - default next month (target).
