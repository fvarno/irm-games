import torch
from torch.utils.tensorboard import SummaryWriter
from fedsim.distributed.centralized.training import FedDyn
from datamanager.irm_dm import IRMDM
from fedsim.models.mcmahan_nets import cnn_mnist
from fedsim.scores import cross_entropy
from fedsim.scores import accuracy
from functools import partial


dm = IRMDM("./data", "colored_mnist")
sw = SummaryWriter()


alg = FedDyn(
    data_manager=dm,
    num_clients=2,
    sample_scheme="uniform",
    sample_rate=1,
    model_class=partial(cnn_mnist, num_classes=2, num_channels=2),
    epochs=5,
    loss_fn=cross_entropy,
    batch_size=32,
    metric_logger=sw,
    log_freq = 5,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
alg.hook_global_score_function("test", "accuracy", accuracy)
for key in dm.get_local_splits_names():
    alg.hook_local_score_function(key, "accuracy", accuracy)

alg.train(rounds=11)