from model_class import LinReg
class LinReg(nn.Module):
    def __init__(self):
        super(LinReg, self).__init__()
        self.linear = nn.Linear(1,1)
    def forward(self, x):
        return self.linear
