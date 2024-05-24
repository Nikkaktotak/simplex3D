import os
import xml.etree.ElementTree as ET
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt

# Шлях до XML файлу
xml_file_path = os.path.join(os.path.dirname(__file__), "statistics.xml")
html_file_path = os.path.join(os.path.dirname(__file__), "analize.html")
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
        full_time = initial_block.find('full_time').text if initial_block.find('full_time') is not None else "N/A"

        for pair in initial_block.findall('pair'):
            target_count = int(pair.find('target_vertex_count').text)
            average_time = float(pair.find('average_time').text)
            time_gain = pair.find('time_gain').text
            average_success_rate = float(pair.find('average_success_rate').text)

            data.append({
                'Initial Point Count': initial_count,
                'Target Vertex Count': target_count,
                'Average Time (s)': average_time,
                'Full Time (s)': full_time,
                'Time Gain': time_gain,
                'Average Success Rate (%)': average_success_rate
            })

    df = pd.DataFrame(data)
    return df

def save_to_html(df, file_path):
    html_string = df.to_html(index=False)
    html_string = """
    <html>
        <head>
            <title>Analysis Results</title>
        </head>
        <body>
            <h1>Analysis Results</h1>
            <p>
                <strong>Initial Point Count:</strong> The number of points before any vertices are removed.<br>
                <strong>Target Vertex Count:</strong> The target number of vertices after the point removal process.<br>
                <strong>Average Time (s):</strong> The average time taken to execute one run of the program.<br>
                <strong>Full Time (s):</strong> The time taken to execute the brute-force approach.<br>
                <strong>Time Gain:</strong> The ratio of Full Time to Average Time, indicating the speed-up.<br>
                <strong>Average Success Rate (%):</strong> The average success rate of the heuristic method as a percentage.
            </p>
            {}
        </body>
    </html>
    """.format(html_string)

    with open(file_path, "w") as f:
        f.write(html_string)

def plot_graphs(df):
    # Виграш у часі в залежності від початкової кількості точок
    plt.figure(figsize=(10, 6))
    for target_count in df['Target Vertex Count'].unique():
        subset = df[df['Target Vertex Count'] == target_count]
        plt.plot(subset['Initial Point Count'], subset['Time Gain'], marker='o', label=f'Target Vertex Count {target_count}')
    plt.xlabel('Initial Point Count')
    plt.ylabel('Time Gain')
    plt.title('Time Gain vs. Initial Point Count')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(graphs_dir, 'time_gain_vs_initial_point_count.png'))

    # Середній відсоток успішності в залежності від цільової кількості точок
    plt.figure(figsize=(10, 6))
    for initial_count in df['Initial Point Count'].unique():
        subset = df[df['Initial Point Count'] == initial_count]
        plt.plot(subset['Target Vertex Count'], subset['Average Success Rate (%)'], marker='o', label=f'Initial Point Count {initial_count}')
    plt.xlabel('Target Vertex Count')
    plt.ylabel('Average Success Rate (%)')
    plt.title('Average Success Rate vs. Target Vertex Count')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(graphs_dir, 'average_success_rate_vs_target_vertex_count.png'))

def main():
    # Парсинг XML файлу в DataFrame
    df = parse_xml_to_dataframe(xml_file_path)

    # Виведення таблиці
    print(tabulate(df, headers='keys', tablefmt='pretty', showindex=False))

    # Збереження таблиці у HTML
    save_to_html(df, html_file_path)
    print(f"Таблиця збережена у файл {html_file_path}")

    # Побудова графіків
    plot_graphs(df)
    print("Графіки збережені як зображення в папці 'graphs'")

if __name__ == "__main__":
    main()
