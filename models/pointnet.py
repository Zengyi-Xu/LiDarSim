"""PointNet 分类网络实现 (PyTorch)"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    """T-Net: 学习点云的 kxk 仿射变换矩阵"""
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k

        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        # 初始化为单位矩阵
        self.fc3.bias.data.zero_()
        self.fc3.weight.data.zero_()
        self.identity = torch.eye(k).view(1, k * k)

    def forward(self, x):
        # x: (B, 3, N)
        B = x.size(0)

        x = F.relu(self.bn1(self.conv1(x)))      # (B, 64, N)
        x = F.relu(self.bn2(self.conv2(x)))      # (B, 128, N)
        x = F.relu(self.bn3(self.conv3(x)))      # (B, 1024, N)
        x = torch.max(x, 2, keepdim=False)[0]    # (B, 1024)

        x = F.relu(self.bn4(self.fc1(x)))        # (B, 512)
        x = F.relu(self.bn5(self.fc2(x)))        # (B, 256)
        x = self.fc3(x)                          # (B, k*k)

        # 加单位矩阵,确保初始状态为恒等变换
        identity = self.identity.repeat(B, 1).to(x.device)
        x = x + identity
        x = x.view(B, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    """PointNet 编码器: 提取全局点云特征"""
    def __init__(self, global_feat=True, feature_transform=False):
        super(PointNetEncoder, self).__init__()
        self.global_feat = global_feat
        self.feature_transform = feature_transform

        self.input_transform = TNet(k=3)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        if self.feature_transform:
            self.feature_transform_net = TNet(k=64)

    def forward(self, x):
        # x: (B, N, 3)
        B, N, _ = x.shape
        x = x.transpose(2, 1)  # (B, 3, N)

        # Input Transform
        trans = self.input_transform(x)           # (B, 3, 3)
        x = x.transpose(2, 1)                     # (B, N, 3)
        x = torch.bmm(x, trans)                   # (B, N, 3) @ (B, 3, 3)
        x = x.transpose(2, 1)                     # (B, 3, N)

        x = F.relu(self.bn1(self.conv1(x)))       # (B, 64, N)

        if self.feature_transform:
            f_trans = self.feature_transform_net(x)  # (B, 64, 64)
            x = x.transpose(2, 1)                    # (B, N, 64)
            x = torch.bmm(x, f_trans)                # (B, N, 64)
            x = x.transpose(2, 1)                    # (B, 64, N)
            local_feat = x.clone()                   # 保存局部特征
        else:
            local_feat = x.clone()

        x = F.relu(self.bn2(self.conv2(x)))       # (B, 128, N)
        x = self.bn3(self.conv3(x))               # (B, 1024, N)

        x = torch.max(x, 2, keepdim=False)[0]     # (B, 1024)  Max Pooling 实现置换不变

        if self.global_feat:
            return x, trans
        else:
            # 拼接局部和全局特征,用于分割任务
            x = x.view(B, 1024, 1).repeat(1, 1, N)  # (B, 1024, N)
            return torch.cat([local_feat, x], 1), trans  # (B, 1088, N)


class PointNetClassifier(nn.Module):
    """PointNet 分类器"""
    def __init__(self, num_classes=40, dropout=0.3, feature_transform=False):
        super(PointNetClassifier, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetEncoder(global_feat=True, feature_transform=feature_transform)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: (B, N, 3)
        x, trans = self.feat(x)   # x: (B, 1024)

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        return x, trans


def pointnet_loss(outputs, labels, trans, reg_weight=0.001):
    """分类损失 + 正交正则化(保证 T-Net 输出近似正交矩阵)"""
    criterion = nn.CrossEntropyLoss()
    classify_loss = criterion(outputs, labels)

    # 正交正则化: || I - A*A^T ||^2
    d = trans.size(1)
    identity = torch.eye(d).to(trans.device)
    orth_loss = torch.mean(
        torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - identity, dim=(1, 2))
    )

    return classify_loss + reg_weight * orth_loss


if __name__ == '__main__':
    model = PointNetClassifier(num_classes=40)
    x = torch.randn(4, 1024, 3)
    out, trans = model(x)
    print("Output shape:", out.shape)      # (4, 40)
    print("Transform shape:", trans.shape)  # (4, 3, 3)
