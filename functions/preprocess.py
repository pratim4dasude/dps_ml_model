import pandas as pd

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df = df[[
        'MONATSZAHL',  # Category
        'AUSPRAEGUNG',  # Accident-type
        'JAHR',  # Year
        'MONAT',  # Month
        'WERT'  # Value
    ]]

    # df_after_2020 = df[df['JAHR'] > 2020]
    # Storing the data that constians up to 2020

    df = df[df['JAHR'] <= 2020]

    #  ----------------  There are some missing data for year 2000's so I add Summe data which means SUM

    df_2000 = df[df['JAHR'] == 2000]
    summary_df = df_2000.groupby(['MONATSZAHL', 'AUSPRAEGUNG'])['WERT'].sum().reset_index()
    summary_df['MONAT'] = 'Summe'
    summary_df['JAHR'] = 2000

    df_updated = pd.concat([df, summary_df], ignore_index=True)
    df = df_updated

    return df


def encode_features(df):


    category_encoding = {
        'Verkehrsunfälle': 0,
        'Alkoholunfälle': 1,
        'Fluchtunfälle': 2
    }

    accident_type_encoding = {
        'insgesamt': 0,
        'Verletzte und Getötete': 1,
        'mit Personenschäden': 2
    }
    # ------ LAbel Encoding But done manually so that i know which value are store in which actual values
    df['MONATSZAHL'] = df['MONATSZAHL'].map(category_encoding)
    df['AUSPRAEGUNG'] = df['AUSPRAEGUNG'].map(accident_type_encoding)

    #  MONAT aka Month  is a string with year attach so last 2 digit is month

    def extract_month(month_str):
        return 0 if month_str == "Summe" else int(month_str[-2:])

    df['MONAT'] = df['MONAT'].apply(extract_month).astype(int)
    df['WERT'] = df['WERT'].astype(int)
    df['JAHR'] = df['JAHR'].astype(int)

    df = df.sort_values(by=['JAHR', 'MONAT'])
    df_filtered = df[df['MONAT'] != 0]

    return df_filtered