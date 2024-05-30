import os
import xml.etree.ElementTree as ET
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt

# Шлях до XML файлу
xml_file_path = os.path.join(os.path.dirname(__file__), "statistics.xml")
xlsx_file_path = os.path.join(os.path.dirname(__file__), "analize.xlsx")
graphs_dir = os.path.join(os.path.dirname(__file__), "graphs")

# Створення директорії для графіків, якщо вона не існує
if not os.path.exists(graphs_dir):
    os.makedirs(graphs_dir)


def parse_xml_to_dataframe(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    data = []
    for initial_block in root.findall('initial_point_count_block'):
        initial_count = int(initial_block.get('count'))

        for target_block in initial_block.findall('all_success_rates'):
            target_count = int(target_block.get('target_vertex_count'))
            for run in target_block.findall('run'):
                run_number = int(run.get('number'))
                success_rate = float(run.find('success_rate').text)

                # Додавання даних в список
                data.append({
                    'Initial Point Count': initial_count,
                    'Target Vertex Count': target_count,
                    'Run Number': run_number,
                    'Success Rate (%)': success_rate
                })

    df = pd.DataFrame(data)
    return df


def save_to_excel(df, file_path):
    df.to_excel(file_path, index=False, engine='openpyxl')
    print(f"Таблиця збережена у файл {file_path}")


def plot_graphs(df):
    # Середній відсоток успішності в залежності від цільової кількості точок
    plt.figure(figsize=(10, 6))
    for target_count in df['Target Vertex Count'].unique():
        subset = df[df['Target Vertex Count'] == target_count]
        avg_success_rates = subset.groupby('Initial Point Count')['Success Rate (%)'].mean()
        plt.plot(avg_success_rates.index, avg_success_rates.values, marker='o', label=f'Target Vertex Count {target_count}')
    plt.xlabel('Initial Point Count')
    plt.ylabel('Average Success Rate (%)')
    plt.title('Average Success Rate vs. Initial Point Count')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(graphs_dir, 'average_success_rate_vs_initial_point_count.png'))

    # Найкращий і найгірший відсоток успішності в залежності від цільової кількості точок
    plt.figure(figsize=(10, 6))
    for target_count in df['Target Vertex Count'].unique():
        subset = df[df['Target Vertex Count'] == target_count]
        best_success_rates = subset.groupby('Initial Point Count')['Success Rate (%)'].max()
        worst_success_rates = subset.groupby('Initial Point Count')['Success Rate (%)'].min()
        plt.plot(best_success_rates.index, best_success_rates.values, marker='o', linestyle='-', label=f'Best Success Rate for Target Vertex Count {target_count}')
        plt.plot(worst_success_rates.index, worst_success_rates.values, marker='o', linestyle='--', label=f'Worst Success Rate for Target Vertex Count {target_count}')
    plt.xlabel('Initial Point Count')
    plt.ylabel('Success Rate (%)')
    plt.title('Best and Worst Success Rate vs. Initial Point Count')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(graphs_dir, 'best_worst_success_rate_vs_initial_point_count.png'))


def main():
    # Парсинг XML файлу в DataFrame
    df = parse_xml_to_dataframe(xml_file_path)

    # Виведення таблиці
    print(tabulate(df, headers='keys', tablefmt='pretty', showindex=False))

    # Збереження таблиці у Excel
    save_to_excel(df, xlsx_file_path)

    # Побудова графіків
    plot_graphs(df)
    print("Графіки збережені як зображення в папці 'graphs'")


if __name__ == "__main__":
    main()
