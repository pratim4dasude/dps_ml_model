import pandas as pd
df = pd.read_csv('data.csv')

df=df[[
    'MONATSZAHL',   # Category
    'AUSPRAEGUNG',    # Accident-type
    'JAHR',   # Year
    'MONAT',  # Month
    'WERT'   # Value
]]

df_after_2020 = df[df['JAHR'] > 2020]

df = df[df['JAHR'] <= 2020]

# print(df)