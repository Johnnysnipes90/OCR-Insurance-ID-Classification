import torch
import torch.nn as nn

class OCRModel(nn.Module):
    def __init__(self):
        super(OCRModel, self).__init__()
        
        # --- Image feature extractor ---
        self.image_layer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 32 * 32, 128),
            nn.ReLU()
        )
        
        # --- Type (insurance type) feature extractor ---
        self.type_layer = nn.Sequential(
            nn.Linear(5, 10),   # 5 types (home, life, auto, health, other)
            nn.ReLU()
        )
        
        # --- Final classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(128 + 10, 64),  # concatenate image + type features
            nn.ReLU(),
            nn.Linear(64, 2)          # Output: Primary or Secondary ID
        )

    def forward(self, x_image, x_type):
        x_image = self.image_layer(x_image)
        x_type = self.type_layer(x_type)
        x = torch.cat((x_image, x_type), dim=1)
        return self.classifier(x)