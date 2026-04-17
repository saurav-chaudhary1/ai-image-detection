import torch
import torch.nn as nn
import math

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_dim, 1),
            Swish(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.se(x)

class MBConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, 
                 expand_ratio, se_ratio=0.25, drop_rate=0.2):
        super().__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.drop_rate = drop_rate
        

        hidden_dim = in_channels * expand_ratio
        self.expand = expand_ratio != 1
        
        if self.expand:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                Swish()
            )
        

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride=stride,
                     padding=kernel_size // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            Swish()
        )
        

        se_channels = max(1, int(in_channels * se_ratio))
        self.se = SEBlock(hidden_dim, se_channels)
        
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        
        self.dropout = nn.Dropout2d(drop_rate) if drop_rate > 0 else None
    
    def forward(self, x):
        identity = x
        
        
        if self.expand:
            x = self.expand_conv(x)
        
        
        x = self.depthwise_conv(x)
        
        
        x = self.se(x)
        
        
        x = self.project(x)
        
        
        if self.use_residual:
            if self.dropout is not None and self.training:
                x = self.dropout(x)
            x = x + identity
        
        return x

class EfficientNet(nn.Module):
    
    def __init__(self, width_mult=1.0, depth_mult=1.0, dropout_rate=0.2, 
                 num_classes=1000, input_size=224):
        super().__init__()
        
        # [expand_ratio, channels, num_blocks, stride, kernel_size]
        base_config = [
            [1,  16,  1, 1, 3],  
            [6,  24,  2, 2, 3],  
            [6,  40,  2, 2, 5],  
            [6,  80,  3, 2, 3],  
            [6,  112, 3, 1, 5],  
            [6,  192, 4, 2, 5],  
            [6,  320, 1, 1, 3],  
        ]
        
        out_channels = self._round_filters(32, width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(3, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            Swish()
        )
        

        self.blocks = nn.ModuleList([])
        in_channels = out_channels
        
        total_blocks = sum([self._round_repeats(config[2], depth_mult) 
                           for config in base_config])
        block_idx = 0
        
        for expand_ratio, channels, num_blocks, stride, kernel_size in base_config:
            out_channels = self._round_filters(channels, width_mult)
            num_blocks = self._round_repeats(num_blocks, depth_mult)
            
            for i in range(num_blocks):

                drop_rate = dropout_rate * block_idx / total_blocks
                
                self.blocks.append(
                    MBConvBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride if i == 0 else 1,
                        expand_ratio=expand_ratio,
                        drop_rate=drop_rate
                    )
                )
                in_channels = out_channels
                block_idx += 1
        

        final_channels = self._round_filters(1280, width_mult)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, final_channels, 1, bias=False),
            nn.BatchNorm2d(final_channels),
            Swish(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(final_channels, num_classes)
        )
        

        self._initialize_weights()
    
    def _round_filters(self, filters, width_mult, divisor=8):

        filters *= width_mult
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)

        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)
    
    def _round_repeats(self, repeats, depth_mult):

        return int(math.ceil(depth_mult * repeats))
    
    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.stem(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.head(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x


def efficientnet_b0(num_classes=1000):

    return EfficientNet(
        width_mult=1.0, 
        depth_mult=1.0, 
        dropout_rate=0.2,
        num_classes=num_classes,
        input_size=224
    )

def efficientnet_b1(num_classes=1):
    
    return EfficientNet(
        width_mult=1.0,  
        depth_mult=1.1,
        dropout_rate=0.2,
        num_classes=num_classes,
        input_size=240 
    )

def efficientnet_b2(num_classes=1000):

    return EfficientNet(
        width_mult=1.1,
        depth_mult=1.2,
        dropout_rate=0.3,
        num_classes=num_classes,
        input_size=260
    )

def efficientnet_b3(num_classes=1000):

    return EfficientNet(
        width_mult=1.2,
        depth_mult=1.4,
        dropout_rate=0.3,
        num_classes=num_classes,
        input_size=300
    )