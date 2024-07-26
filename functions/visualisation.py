import matplotlib.pyplot as plt
import seaborn as sns

def visualize_data(df_filtered):

    plt.figure(figsize=(12, 8)) # plt.figure(figsize=(1, 5))
    # plt.figure(figsize=(11, 11))
    sns.lineplot(data=df_filtered, x='MONAT', y='WERT', hue='JAHR', palette='tab10', marker='o')
    # sns.lineplot(data=df_filtered, x='Month', y='Value', hue='Year', palette='tab10', marker='o')

    plt.title('Monthly Values from 2000 to 2020')
    plt.xlabel('Month')
    plt.ylabel('Value')
    plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()