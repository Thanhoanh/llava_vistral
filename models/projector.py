class ProjectorMLP(nn.Module):
    def __init__(self, input_dim=512, output_dim=4096):  # CHUYỂN 1024 → 512
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)
