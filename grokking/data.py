from math import ceil # 向上取整函数
import torch

# 反向构造，保证所有x/y 都存在
# 这段定义了一个字典，键是字符串 "x/y"，值是一个 lambda 函数。
# Lambda 函数：是匿名函数，这里接收三个参数 x、y 和 p，返回一个三元组
DIVISION_MODULO_OPERATIONS = {
    "x/y": lambda x, y, p: (x*y % p, y, x),
}

ALL_MODULO_OPERATIONS = {
    "x+y": lambda x, y, p: (x, y, (x + y) % p),  # 加上模p运算
    "x-y": lambda x, y, p: (x, y, (x - y) % p),  # 加上模p运算
    **DIVISION_MODULO_OPERATIONS,
}

# 注： " ** " 操作符用于解包字典，在 Python 3.5+ 中引入。这允许将一个字典的内容合并到另一个字典中，避免使用 .update() 方法。
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

def get_data(operation: str, prime: int, training_fraction: float, batch_size: int, custom_split: bool = False):
    """
    获取数据的统一接口
    custom_split: 是否使用自定义划分方式
    """
    if custom_split and operation in ["x+y", "x-y", "x/y"]:
        return get_data_custom_split(operation, prime, batch_size)
    else:
        inputs, labels = operation_mod_p_data(operation, prime, prime, prime+1)
        
        # 创建一个TensorDataset，将输入和标签组合在一起，允许通过索引访问
        # TensorDataset 是 PyTorch 中的一个数据集类，用于将多个张量组合在一起
        dataset = torch.utils.data.TensorDataset(inputs, labels)

        # 计算Train Data和Val Data的大小，然后将数据集随机划分为Train Data和Val Data
        train_size = int(training_fraction * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        # 确保batch_size不超过Val Data的大小，ceil 保证 batch_size 至少为1
        batch_size = min(batch_size, ceil(len(dataset) / 2))

        # 创建数据加载器，使用随机打乱的方式加载Train Data和Val Data
        # DataLoader 是 PyTorch 中的一个类，用于批量加载数据集
        # shuffle=True 表示在每个epoch开始时打乱数据
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        return train_loader, val_loader

################################################################################
# Self-defined data split for addition operation
################################################################################
def get_data_custom_split(operation: str, prime: int, batch_size: int):
    """
    自定义数据划分：
    - 加法 "x+y": Train Data a≤b, Val Data a>b
    - 减法 "x-y": Train Data a≥b, Val Data a<b  
    - 除法 "x/y": Train Data y≤prime//2, Val Data y>prime//2
    """
    inputs, labels = operation_mod_p_data(operation, prime, prime, prime+1)
    
    if operation == "x+y":
        # 获取原始的x, y值
        x_orig = torch.arange(0, prime)
        y_orig = torch.arange(0, prime)
        x_orig, y_orig = torch.cartesian_prod(x_orig, y_orig).T
        
        # Train Data：a <= b，Val Data：a > b
        train_mask = x_orig <= y_orig
        val_mask = x_orig > y_orig
        split_info = "Train Data: a≤b, Val Data: a>b"
        
    elif operation == "x-y":
        # 获取原始的x, y值
        x_orig = torch.arange(0, prime)
        y_orig = torch.arange(0, prime)
        x_orig, y_orig = torch.cartesian_prod(x_orig, y_orig).T
        
        # Train Data：a >= b，Val Data：a < b
        train_mask = x_orig >= y_orig
        val_mask = x_orig < y_orig
        split_info = "Train Data: a≥b, Val Data: a<b"
        
    elif operation == "x/y":
        # 对于除法，y的范围是1到prime-1
        x_orig = torch.arange(0, prime)
        y_orig = torch.arange(1, prime)
        x_orig, y_orig = torch.cartesian_prod(x_orig, y_orig).T
        
        # 按除数大小划分：Train Data用小除数，Val Data用大除数
        threshold = prime // 2
        train_mask = y_orig <= threshold
        val_mask = y_orig > threshold
        split_info = f"Train Data: y≤{threshold}, Val Data: y>{threshold}"
        
    else:
        raise ValueError(f"Unsupported operation type: {operation}")
    
    # 划分数据
    train_inputs = inputs[train_mask]
    train_labels = labels[train_mask]
    val_inputs = inputs[val_mask]
    val_labels = labels[val_mask]
    
    # 创建数据集
    train_dataset = torch.utils.data.TensorDataset(train_inputs, train_labels)
    val_dataset = torch.utils.data.TensorDataset(val_inputs, val_labels)
    
    # 调整batch_size
    batch_size = min(batch_size, max(1, len(train_dataset) // 2))
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    print(f"{operation} operation - {split_info}")
    print(f"Train Size: {len(train_dataset)}")
    print(f"Val Size: {len(val_dataset)}")
    
    return train_loader, val_loader