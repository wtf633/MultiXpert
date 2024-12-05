# from radgraph import RadGraph
# radgraph = RadGraph()
# annotations = radgraph(["no acute cardiopulmonary abnormality",
#             "endotracheal tube terminates 2 5 cm above the carina bibasilar opacities likely represent atelectasis or aspiration"])
# print(annotations)

import pandas as pd
from radgraph import RadGraph
import json
from tqdm import tqdm  # 引入 tqdm 库

# 读取用户上传的 CSV 文件
file_path = './sample_data.csv'
data = pd.read_csv(file_path)

# 提取第二列需要处理的文本内容
texts_to_process = data['Findings/Impression'].tolist()

# 初始化 RadGraph
radgraph = RadGraph()


# 定义一个函数来处理每一行文本并转换为指定的格式
def process_texts_to_json_format(texts):
    results = {}
    for idx, text in tqdm(enumerate(texts), total=len(texts), desc="Processing texts"):  # 使用 tqdm 包装 enumerate
        # 使用 RadGraph 处理文本
        annotations = radgraph([text])

        # 确保 annotations 结构正确
        if annotations and isinstance(annotations, dict) and '0' in annotations:
            processed = annotations['0']  # 取出处理结果
            results[f"report_{idx}"] = {
                "text": text,
                "entities": processed["entities"],
                "data_source": None,
                "data_split": "inference"
            }
        else:
            print(f"Error processing report {idx}: Unexpected annotations format")
    return results


# 批量处理所有文本内容
processed_data = process_texts_to_json_format(texts_to_process)

# 将结果保存为 JSON 文件
output_json_path = './sample_output.json'
with open(output_json_path, 'w') as json_file:
    json.dump(processed_data, json_file)




