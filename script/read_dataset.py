def read_dataset_amp(file_path: str):
    """
    读入AMP数据集
    :param file_path: AMP数据集文件路径
    :return: 蛋白质序列与标签（阴阳性）
    """
    labels = []
    sequences = []
    with open(file_path, 'r') as file:
        while True:
            label_line = file.readline().strip()
            if not label_line:
                break  # 如果读到文件末尾，则退出循环
            label = int(label_line.split('|')[-1])  # 获取标签

            sequence_line = file.readline().strip()

            labels.append(label)
            sequences.append(sequence_line)
        return sequences, labels

def read_dataset_zero(file_path: str):
    """
    读入AMP数据集
    :param file_path: AMP数据集文件路径
    :return: 蛋白质序列与标签（阴阳性）
    """
    labels = []
    sequences = []
    with open(file_path, 'r') as file:
        positive_count = 0
        while True:
            label_line = file.readline().strip()
            if not label_line:
                break  # 如果读到文件末尾，则退出循环
            if 'AMP' in label_line:
                label = int(1)  # 获取标签
                positive_count += 1
            elif 'NEGATIVE' in label_line:
                label = int(0)
            else:
                print("Error!")
            sequence_line = file.readline().strip()
            labels.append(label)
            sequences.append(sequence_line)
        return sequences, labels

def read_dataset_aip(file_path: str):
    """
    读入AIP数据集
    :param file_path: AMP数据集文件路径
    :return: 蛋白质序列与标签（阴阳性）
    """
    labels = []
    sequences = []
    with open(file_path, 'r') as file:
        positive_count = 0
        while True:
            label_line = file.readline().strip()
            if not label_line:
                break  # 如果读到文件末尾，则退出循环
            if 'Positive' in label_line:
                label = int(1)  # 获取标签
                positive_count += 1
            elif 'Negative' in label_line:
                label = int(0)
            else:
                print("Error!")
            sequence_line = file.readline().strip()
            labels.append(label)
            sequences.append(sequence_line)
        return sequences, labels

def read_dataset_from_aipstack_work(file_path: str):
    """
    从AIP stack的开源仓库中读入AIP数据集
    :param file_path: AIP数据集文件路径
    :return: 蛋白质序列与标签（阴阳性）
    """
    labels = []
    sequences = []
    with open(file_path, 'r') as file:
        positive_count = 0
        while True:
            label_line = file.readline().strip()
            if not label_line:
                break  # 如果读到文件末尾，则退出循环
            if '1' in label_line:
                label = int(1)  # 获取标签
                positive_count += 1
            elif '0' in label_line:
                label = int(0)
            else:
                print("Error!")
            sequence_line = file.readline().strip()
            labels.append(label)
            sequences.append(sequence_line)
        return sequences, labels

def read_dataset_bd(file_path: str):
    """
    读入AIP数据集
    :param file_path: AMP数据集文件路径
    :return: 蛋白质序列与标签（阴阳性）
    """
    labels = []
    sequences = []
    with open(file_path, 'r') as file:
        positive_count = 0
        while True:
            label_line = file.readline().strip()
            if not label_line:
                break  # 如果读到文件末尾，则退出循环
            if 'pos' in label_line:
                label = int(1)  # 获取标签
                positive_count += 1
            elif 'neg' in label_line:
                label = int(0)
            else:
                print("Error!")
            sequence_line = file.readline().strip()
            labels.append(label)
            sequences.append(sequence_line)
        return sequences, labels

# read_dataset_aip('../datasets/AMP_1.txt')
