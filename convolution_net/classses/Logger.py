from torch.utils.tensorboard import SummaryWriter


# ============================================================
# defining the logger
# ============================================================
class Logger:
    def __init__(self, logdir):
        # -------------------------------------
        # tensorboard logger
        # -------------------------------------
        self.logger = SummaryWriter(logdir)
        self.logger_tag = []

