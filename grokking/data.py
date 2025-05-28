from math import ceil # 向上取整函数
import torch

# 反向构造，保证所有x/y 都存在
DIVISION_MODULO_OPERATIONS = {
    "x/y": lambda x, y, p: (x*y % p, y, x),
}

ALL_MODULO_OPERATIONS = {
    "x+y": lambda x, y, _: (x, y, x + y),
    "x-y": lambda x, y, _: (x, y, x - y),
    **DIVISION_MODULO_OPERATIONS, # Python字典解包语法，合并字典
}

ALL_OPERATIONS = {
    **ALL_MODULO_OPERATIONS,
}

def operation_mod_p_data(operation: str, p: int, eq_token: int, op_token: int):
    """
    x◦y (mod p) for 0 <= x < p, 1 <= y < p if operation in DIVISION_MODULO_OPERATIONS
    x◦y (mod p) for 0 <= x, y < p otherwise
    """
    x = torch.arange(0, p) # 创建0到p-1的张量
    y = torch.arange(0 if not operation in DIVISION_MODULO_OPERATIONS else 1, p)
    # 创建两个集合的笛卡尔积，生成所有可能的(x,y)对
    # .T 进行转置，将形状从(n,2)变为(2,n)，得到两个长度为n的向量
    x, y = torch.cartesian_prod(x, y).T

    # torch.ones_like 创建与x形状相同的全1张量
    # 乘以token值生成表示特定符号的张量
    eq = torch.ones_like(x) * eq_token # 创建表示等号的张量
    op = torch.ones_like(x) * op_token # 创建表示操作符的张量

    # 根据操作类型调用相应的函数，计算结果
    # labels 是操作的结果， 意味着我们将这个任务视为一个监督学习问题
    x, y, labels = ALL_OPERATIONS[operation](x, y, p)

    # torch.stack 沿指定维度堆叠张量列表
    # dim=1 表示沿第二维堆叠，生成形状为(n,4)的张量，每行是[x,op,y,eq]序列
    inputs = torch.stack([x, op, y, eq], dim=1)

    return inputs, labels

def get_data(operation: str, prime: int, training_fraction: float, batch_size: int):
    
    inputs, labels = operation_mod_p_data(operation, prime, prime, prime+1)
    
    # 创建一个TensorDataset，将输入和标签组合在一起，允许通过索引访问
    # TensorDataset 是 PyTorch 中的一个数据集类，用于将多个张量组合在一起
    dataset = torch.utils.data.TensorDataset(inputs, labels)

    # 计算训练集和验证集的大小，然后将数据集随机划分为训练集和验证集
    train_size = int(training_fraction * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # 确保batch_size不超过验证集的大小，ceil 保证 batch_size 至少为1
    batch_size = min(batch_size, ceil(len(dataset) / 2))

    # 创建数据加载器，使用随机打乱的方式加载训练集和验证集
    # DataLoader 是 PyTorch 中的一个类，用于批量加载数据集
    # shuffle=True 表示在每个epoch开始时打乱数据
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader
