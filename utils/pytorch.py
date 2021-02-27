from torch.backends import cudnn
from torch.autograd import set_detect_anomaly
from torch.autograd.profiler import profile, emit_nvtx


def init_torch():
    cudnn.benchmark = True
    set_detect_anomaly(False)
    profile(False)
    emit_nvtx(False)
