
import math
import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.modules
import torch.nn.functional as F
import torch.optim
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
import matplotlib.pyplot as plt


class TorpedoDataset(Dataset):
    def __init__(self, filepath):
        raw_data = pd.read_csv(filepath, header=None)
        df = pd.DataFrame(raw_data)
        df = df.iloc[0:, 1:]
        col_nums = df.shape[1]
        iter_nums = int((col_nums - 1) / 20)
        raw_data_len = int(0.5 * len(df))
        # 先对所有raw序列（全长序列）产生可信度label，皆为1.0
        confidence_list = [1.0] * raw_data_len
        # 转出来个numpy,主要是为了算isnan
        raw_data_array = np.array(df)
        # 给个空list，存序列长度
        seq_lengths = []
        for j in range(raw_data_len):
            seq_length = df.shape[1] - np.isnan(raw_data_array[2 * j]).sum()
            seq_lengths.append(seq_length)
        for i in range(iter_nums):
            df_1 = pd.DataFrame(raw_data)
            df_1 = df_1.iloc[0:, 1:i * 20 + 21]
            # 如果以20分段的长度已经超过raw_length，仍赋予1.0
            for j in range(raw_data_len):
                if (i+1)*20 < seq_lengths[j]:
                    confidence = (i + 1) * 20/seq_lengths[j]
                else:
                    confidence = 1.0
                confidence_list.append(confidence)
            df = pd.concat([df, df_1])
        df = df.fillna(0.0)
        point_data = []
        data_len = int(0.5 * len(df))

        for row in range(data_len):
            raw_data_x = np.array(df.iloc[2 * row, 0:], dtype=np.float32)
            raw_data_y = np.array(df.iloc[2 * row + 1, 0:], dtype=np.float32)
            cord_seq = np.dstack((raw_data_x, raw_data_y))
            cord_seq2 = cord_seq.squeeze()
            point_data.append(cord_seq2)
        # label_data要原样扩倍
        df = pd.DataFrame(raw_data)
        df_1 = pd.DataFrame(raw_data)
        df = df.iloc[df.index % 2 == 0, 0]
        df_1 = df_1.iloc[df_1.index % 2 == 0, 0]
        for i in range(iter_nums):
            df = pd.concat([df, df_1])
        label_data = list(df)
        self.len = data_len
        self.x_data = point_data
        self.y_data = label_data
        self.data_confidence = confidence_list

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], self.data_confidence[index]

    def __len__(self):
        return self.len


class TorpedoValidateDataset(Dataset):
    def __init__(self, filepath):
        raw_data = pd.read_csv(filepath, header=None)
        df = pd.DataFrame(raw_data)
        col_nums = df.shape[1]
        iter_nums = int((col_nums-1)/20)
        df = df.iloc[0:, 1:]
        for i in range(iter_nums):
            df_1 = pd.DataFrame(raw_data)
            df_1 = df_1.iloc[0:, 1:i * 20 + 21]
            df = pd.concat([df, df_1])
        df = df.fillna(0.0)
        point_data = []
        data_len = int(0.5 * len(df))
        for row in range(data_len):
            raw_data_x = np.array(df.iloc[2 * row, 0:], dtype=np.float32)
            raw_data_y = np.array(df.iloc[2 * row + 1, 0:], dtype=np.float32)
            cord_seq = np.dstack((raw_data_x, raw_data_y))
            cord_seq2 = cord_seq.squeeze()
            point_data.append(cord_seq2)
        # label_data要原样扩倍
        df = pd.DataFrame(raw_data)
        df_1 = pd.DataFrame(raw_data)
        df = df.iloc[df.index % 2 == 0, 0]
        df_1 = df_1.iloc[df_1.index % 2 == 0, 0]
        for i in range(iter_nums):
            df = pd.concat([df, df_1])
        label_data = list(df)
        self.len = data_len
        self.x_data = point_data
        self.y_data = label_data

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


# 以轨迹序列点数增加划分测试集
class TorpedoDivideDataset(Dataset):
    def __init__(self, filepath, length):
        raw_data = pd.read_csv(filepath, header=None)
        df = pd.DataFrame(raw_data)
        df = df.fillna(0.0)
        point_data = []
        data_len = int(0.5 * len(df))
        for row in range(data_len):
            raw_data_x = np.array(df.iloc[2 * row, 1:length], dtype=np.float32)
            raw_data_y = np.array(df.iloc[2 * row + 1, 1:length], dtype=np.float32)
            cord_seq = np.dstack((raw_data_x, raw_data_y))
            cord_seq2 = cord_seq.squeeze()
            point_data.append(cord_seq2)

        label_data = list(df.iloc[df.index % 2 == 0, 0])
        self.len = data_len
        self.x_data = point_data
        self.y_data = label_data

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class TorpedoFullDataset(Dataset):
    def __init__(self, filepath):
        raw_data = pd.read_csv(filepath, header=None)
        df = pd.DataFrame(raw_data)
        df = df.fillna(0.0)
        point_data = []
        data_len = int(0.5 * len(df))
        for row in range(data_len):
            raw_data_x = np.array(df.iloc[2 * row, 1:], dtype=np.float32)
            raw_data_y = np.array(df.iloc[2 * row + 1, 1:], dtype=np.float32)
            cord_seq = np.dstack((raw_data_x, raw_data_y))
            cord_seq2 = cord_seq.squeeze()
            point_data.append(cord_seq2)

        label_data = list(df.iloc[df.index % 2 == 0, 0])
        self.len = data_len
        self.x_data = point_data
        self.y_data = label_data

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


train_set = TorpedoDataset('train2.csv')
train_loader = DataLoader(dataset=train_set, batch_size=1500, shuffle=True)
validate_set = TorpedoValidateDataset('test2.csv')
validate_loader = DataLoader(dataset=validate_set, batch_size=600, shuffle=False)

# 不完全轨迹序列验证集
test_loader_list = []
for i in range(24):
    test_part_set = TorpedoDivideDataset('add.csv', length=21+i*20)
    test_part_loader = DataLoader(dataset=test_part_set, batch_size=100, shuffle=False)
    test_loader_list.append(test_part_loader)

test_set = TorpedoFullDataset('add.csv')
test_loader = DataLoader(dataset=test_set, batch_size=100, shuffle=False)


n_class = 3
N_EPOCHS = 800
USE_GPU = torch.cuda.is_available()
device = torch.device("cuda:1")


def create_tensor(tensor):
    if USE_GPU:
        tensor = tensor.to(device)
    return tensor


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


class RNNClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, bidirectional=True):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_directions = 2 if bidirectional else 1
        self.gru = torch.nn.GRU(input_size, hidden_size, num_layers, bidirectional=bidirectional)
        self.fc = torch.nn.Linear(hidden_size*self.n_directions, output_size)

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers*self.n_directions, batch_size, self.hidden_size)
        return create_tensor(hidden)

    def forward(self, input_data, seq_lengths):
        batch_size = input_data.size(0)
        hidden = self._init_hidden(batch_size)
        gru_input = pack_padded_sequence(input_data, seq_lengths, batch_first=True)
        output, hidden = self.gru(gru_input, hidden)
        if self.n_directions == 2:
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1]
        fc_output = self.fc(hidden_cat)
        return fc_output


def make_all_tensors(point_data, label_data, data_confidence):
    """
    先算出来每个序列的长度,再得到序列长度降序的索引
    :param point_data: list,坐标序列数据的列表
    :param label_data: list，标签的列表
    :param data_confidence: list，置信的列表
    :return: 序列张量、序列长度的列表张量（用于pack_padded_sequence）
    """
    point_np = np.array(point_data)
    label_data_np = np.array(label_data)
    data_confidence_np = np.array(data_confidence)
    seq_lengths = []
    for j in range(len(point_data)):
        seq_length = len(point_data[j]) - np.isnan(point_np[j, 1:]).sum()
        seq_lengths.append(seq_length)
    seq_lengths_np = np.array(seq_lengths)
    sequence_lengths = torch.LongTensor(seq_lengths_np)
    sequence_tensor = torch.FloatTensor(point_np)
    confidence_tensor = torch.FloatTensor(data_confidence_np)
    label_tensor = torch.LongTensor(label_data_np)
    sequence_lengths, perm_idx = sequence_lengths.sort(dim=0, descending=True)
    sequence_tensor = sequence_tensor[perm_idx]
    label_tensor = label_tensor[perm_idx]
    confidence_tensor = confidence_tensor[perm_idx]
    print(confidence_tensor)

    return create_tensor(sequence_tensor), sequence_lengths, create_tensor(label_tensor), create_tensor(confidence_tensor)


def make_tensors(point_data, label_data):
    """
    先算出来每个序列的长度,再得到序列长度降序的索引
    :param point_data: list,坐标序列数据的列表
    :param label_data: list，标签的列表
    :return: 序列张量、序列长度的列表张量（用于pack_padded_sequence）
    """
    point_np = np.array(point_data)
    label_data_np = np.array(label_data)
    seq_lengths = []
    for j in range(len(point_data)):
        seq_length = len(point_data[j]) - np.isnan(point_np[j, 1:]).sum()
        seq_lengths.append(seq_length)
    seq_lengths_np = np.array(seq_lengths)
    sequence_lengths = torch.LongTensor(seq_lengths_np)
    sequence_tensor = torch.FloatTensor(point_np)
    label_tensor = torch.LongTensor(label_data_np)
    sequence_lengths, perm_idx = sequence_lengths.sort(dim=0, descending=True)
    sequence_tensor = sequence_tensor[perm_idx]
    label_tensor = label_tensor[perm_idx]

    return create_tensor(sequence_tensor), sequence_lengths, create_tensor(label_tensor)


class SelfLoss(torch.nn.Module):
    """
    SelfLoss是自己定义的损失函数类
    CrossEntropy是关于softmax+log的处理
    在计算权重均值之前就刹停，把自己的权重赋予过去即可
    :return:
    """

    def __init__(self):
        super().__init__()

    def forward(self, input, target, instance_weight):
        unweightedloss = F.cross_entropy(input, target, reduction="none")
        weightedloss = unweightedloss * instance_weight
        loss = torch.mean(weightedloss)
        return loss


def train_model():
    total_loss = 0
    for i, (x_data, label, confidence) in enumerate(train_loader, 1):
        input_data, sequence_lengths, target, data_confidence = make_all_tensors(x_data, label, confidence)
        output = classifier(input_data, sequence_lengths)
        loss = criterion(output, target, data_confidence)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i % 10 == 0:
            print(f'[{time_since(start)}] Epoch{epoch}', end='')
            print(f'[{i * len(input_data)}/{len(train_set)}]', end='')
            print(f'loss={total_loss/(i * len(input_data))}')

    return total_loss


def validate_model():
    correct = 0
    total = len(validate_set)
    print("evaluating trained model...")
    with torch.no_grad():
        for i, (x_data, label) in enumerate(validate_loader):
            input_data, sequence_lengths, target = make_tensors(x_data, label)
            output = classifier(input_data, sequence_lengths)
            pred = output.max(dim=1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        percent = '%.2f' % (100 * correct / total)
        print(f'Test set :Accuracy {correct}/{total} {percent}%')

    return correct/total


def test_part_model():
    correct = 0
    acc_part_list = []
    total = len(test_part_set)
    print("evaluating trained model...")
    with torch.no_grad():
        for j in range(24):
            for i, (x_data, label) in enumerate(test_loader_list[j]):

                input_data, sequence_lengths, target = make_tensors(x_data, label)
                output = classifier(input_data, sequence_lengths)
                pred = output.max(dim=1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

            acc_part_list.append(100 * correct / total)
            correct = 0

    return acc_part_list


def test_model():
    correct = 0
    total = len(test_set)
    print("evaluating trained model...")
    with torch.no_grad():
        for i, (x_data, label) in enumerate(test_loader):
            input_data, sequence_lengths, target = make_tensors(x_data, label)
            output = classifier(input_data, sequence_lengths)
            pred = output.max(dim=1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        percent = '%.2f' % (100 * correct / total)
        print(f'Test set :Accuracy {correct}/{total} {percent}%')

    return 100*correct / total


if __name__ == '__main__':
    """训练主程序"""
    classifier = RNNClassifier(input_size=2, hidden_size=100, output_size=n_class).to(device)

    # criterion = torch.nn.CrossEntropyLoss()
    criterion = SelfLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    start = time.time()
    print("Training for %d epochs..." % N_EPOCHS)
    acc_list = []
    max_acc = 0.8
    for epoch in range(1, N_EPOCHS + 1):
        # Train cycle
        train_model()
        acc = validate_model()
        acc_list.append(acc)
        if acc > max_acc:
            max_acc = acc
            print("save model,the accuracy of model is %f" % max_acc)
            torch.save(classifier.state_dict(), '/data2/home/chaoqun/simulator/model/advance4_model')
    epoch = np.arange(1, len(acc_list) + 1, 1)
    acc_list = [i * 100 for i in acc_list]
    acc_list = np.array(acc_list)
    plt.plot(epoch, acc_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy%')
    plt.grid()
    plt.show()

    """
    将不同序列长度丢进用全长序列训练好的模型验证识别率规律，使用以下程序时先把上面的训练主程序注释掉
    """
    # classifier = RNNClassifier(input_size=2, hidden_size=100, output_size=n_class).to(device)
    # classifier.load_state_dict(torch.load('/data2/home/chaoqun/simulator/model/advance4_model'))
    # acc_list = test_part_model()
    # acc_max = test_model()
    # acc_list.append(acc_max)
    # print(acc_list)
    # epoch = np.arange(0, (len(acc_list))*20, 20)
    # acc_list = np.array(acc_list)
    # plt.plot(epoch, acc_list)
    # # ax = plt.gca()
    # # ax.locator_params('x', nbins=10)
    # plt.xlabel('From_1_To_number,1-20,1-40,...,1-Max')
    # plt.ylabel('Accuracy%')
    # plt.grid()
    # plt.show()








