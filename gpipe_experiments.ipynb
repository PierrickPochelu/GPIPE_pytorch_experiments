from torchgpipe import GPipe
from torchgpipe.balance import balance_by_size, balance_by_time
import torch
from torch import nn, empty, cuda, utils, optim, Tensor, long
from torch.utils.data import TensorDataset, DataLoader
from collections import OrderedDict
import torchvision.models as models
import time
import numpy as np
import os
import sys

TIMEOUT = 600
DEVICES_WANTED = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
]  # <- GPU and CPU to use by the below code (-1 is the CPU, other integer GPU ID)
for i in range(len(DEVICES_WANTED)):
    DEVICES_WANTED.append(-1)


def _display_neural_network_stats(model):
    table = []
    total_params = 0
    for name, parameter in model.named_parameters():
        # if not parameter.requires_grad: continue
        param = parameter.numel()
        table.append([name, param])
        total_params += param
    print(
        f"Total Trainable Params: {total_params} , nb trainable tensors: {len(table)}"
    )


def GPIPE_running_context(
    model_seq: torch.nn.Sequential,
    torch_samples: torch.Tensor = None,
    gpipe_nb_chunks: int = None,
    gpipe_nb_partitions: int = None,
    gpipe_is_enabled: bool = False,
    balance_strategy="time",
):
    devices = _get_torch_gpu()
    print("Devices:")
    print(devices)
    if gpipe_is_enabled:
        print("Smart allocation between tensors and devices ...")
        with torch.no_grad():
            if balance_strategy == "time":
                balance = balance_by_time(
                    partitions=gpipe_nb_partitions,
                    module=model_seq,
                    sample=torch_samples,
                    timeout=TIMEOUT,
                )
            elif balance_strategy == "size":
                balance = balance_by_size(
                    partitions=gpipe_nb_partitions,
                    input=torch_samples,
                    module=model_seq,
                    param_scale=4,
                )
            else:
                raise ValueError(
                    "balance_strategy not understood. It must be either 'time' or 'size'"
                )
            print("GPIPE balance:", balance)
        print("Build gpipe model...")
        model = GPipe(
            model_seq,
            balance=balance,
            chunks=gpipe_nb_chunks,
            devices=devices,
            checkpoint="never",
        )
        print("Gpipe model is build")
    else:
        print("Gpipe is disabled")
        model = model_seq
        if torch.cuda.is_available():
            model.cuda()
    return model


def _get_torch_gpu():
    cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
    if cuda_visible_devices in {"-1", ""}:
        devices = None
    else:
        devices = []
        for d in DEVICES_WANTED:
            if d == -1:
                devices.append(torch.device("cpu"))
            else:
                devices.append(torch.device("cuda:" + str(d)))
    return devices


def partition(model) -> torch.nn.Sequential:
    if isinstance(model, torch.nn.Sequential):
        return model  # Your model is already a Sequential Torch object

    def recursively_partition(torch_model) -> list:
        finest_grain_nodes = []
        nodes = list(torch_model.children())
        for node in nodes:
            if isinstance(node, torch.nn.modules.container.Sequential):
                finer_grain_nodes = recursively_partition(node)
                for finer_grain_node in finer_grain_nodes:
                    finest_grain_nodes.append(finer_grain_node)
            else:
                finest_grain_nodes.append(node)
        return finest_grain_nodes

    partitions_list = recursively_partition(model)
    model_seq = torch.nn.Sequential(*(partitions_list[:-1]))
    return model_seq


def my_usual_torch_code(usual_torch_code_config):
    multiplier = usual_torch_code_config["multiplier"]
    X = np.random.uniform(0, 1, (nb_samples, 3, 224, 224)).astype(float)
    Y = np.array([i for i in range(nb_samples)], dtype=float)

    def _add_last_layer(model, nb_in, nb_out=1000):
        class Flatten(nn.Module):
            def forward(self, x):
                x = x.view(x.size()[0], -1)
                return x

        flat1 = Flatten()
        flat2 = Flatten()
        classifier = nn.Linear(nb_in, nb_out)
        soft = torch.nn.Softmax()
        modules = []
        for i, m in enumerate(model):
            modules.append(("layer" + str(i), m))
        modules.append(("flat2", flat2))
        modules.append(("flat1", flat1))
        modules.append(("classifier", classifier))
        modules.append(("softmax", soft))
        model2 = torch.nn.Sequential(OrderedDict(modules))
        return model2

    model = models.resnet.ResNet(
        block=models.resnet.Bottleneck,
        layers=[3 * multiplier, 8 * multiplier, 36 * multiplier, 3 * multiplier],
        num_classes=1000,
    )
    nb_out_elems = 2048

    # uncomment for trying with EfficientNet DNNs
    # model = models.efficientnet_b7()
    # nb_out_elems=2560

    model_seq = partition(model)
    model_seq = _add_last_layer(model_seq, nb_out_elems)

    print(f"Your model contains {len(model_seq)} partitions")
    gpipe_calibration_samples = torch.empty(nb_samples, 3, 224, 224)
    optimizer = optim.Adam(model_seq.parameters(), lr=0.01)
    loss = nn.CrossEntropyLoss()
    return X, Y, gpipe_calibration_samples, model_seq, loss, optimizer


def GPIPE_EXPERIMENT(
    devices_wanted: list,
    usual_torch_code_config: dict,
    gpipe_nb_chunks: int,
    gpipe_nb_partitions: int,
    is_gpipe_enabled: bool,
    default_device_name: str = "cuda:0",
):
    # DEVICES to use
    gputxt = ""
    for d in devices_wanted:
        if d != -1:
            gputxt += str(d) + ","
    os.environ["CUDA_VISIBLE_DEVICES"] = gputxt[:-1]  # remove the last ','

    start_init_time = time.time()

    ########## Create an usual Torch deep learning project ############
    X, Y, gpipe_calibration_samples, model_seq, loss, optimizer = my_usual_torch_code(
        usual_torch_code_config
    )

    # Convert the torch model to a GPIPE model. This model is distributed across multiple GPUs.
    model = GPIPE_running_context(
        model_seq,
        torch_samples=gpipe_calibration_samples,
        gpipe_nb_chunks=gpipe_nb_chunks,
        gpipe_nb_partitions=gpipe_nb_partitions,
        gpipe_is_enabled=is_gpipe_enabled,
    )

    ############# USUAL TORCH TRAINING LOOP #################
    import torch

    tensor_x = torch.cuda.FloatTensor(X)
    tensor_y = torch.cuda.FloatTensor(Y).long()
    tensor_x.to(torch.device(default_device_name))
    tensor_y.to(torch.device(default_device_name))
    my_dataset = TensorDataset(tensor_x, tensor_y)
    trainloader = utils.data.DataLoader(my_dataset, batch_size=batch_size, shuffle=True)
    # The training loop
    enlapsed_init_time = time.time() - start_init_time
    training_time = 0
    for inputs, labels in trainloader:
        start_time = time.time()

        # 1 SGD step
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.to(torch.device(default_device_name))
        labels = labels.to(torch.device(default_device_name))
        l = loss(outputs, labels)
        l.backward()
        optimizer.step()

        enlapsed_time = time.time() - start_time
        training_time += enlapsed_time
    return training_time, enlapsed_init_time, model


if __name__ == "__main__":
    neural_network_multiplier = 1  # Vary it to simulate big neural network workload
    gpipe_nb_chunks = 1
    nb_samples = 16
    batch_size = 4
    is_gpipe_enabled = True
    default_device_name = "cuda:0"
    training_time, enlapsed_init_time, model = GPIPE_EXPERIMENT(
        devices_wanted=DEVICES_WANTED,
        usual_torch_code_config={"multiplier": neural_network_multiplier},
        gpipe_nb_chunks=gpipe_nb_chunks,
        gpipe_nb_partitions=len(DEVICES_WANTED),
        is_gpipe_enabled=is_gpipe_enabled,
        default_device_name=default_device_name,
    )
    ########## DISPLAY EXPERIMENTAL RESULTS ##############
    print("-----------------")
    print(
        f"multiplier={neural_network_multiplier}, gpipe_nb_chunks={gpipe_nb_chunks},"
        f"DEVICES_WANTED={DEVICES_WANTED}, gpipe_is_enabled={is_gpipe_enabled}"
    )
    print(f"Training time: {training_time}")
    print(f"Init time: {enlapsed_init_time}")
    _display_neural_network_stats(model)
    print("-----------------")
