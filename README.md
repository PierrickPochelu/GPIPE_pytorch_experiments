# GPIPE_pytorch_experiments
Original paper of GPIPE URL: https://arxiv.org/abs/1811.06965

The deep learning methods require multiple experiments to be well-tuned. And more, the most computing-intensive deep learning models (ensembles or big neural networks) produce generally better predictions. To overcome this computation requirement, optimizing the hardware utilization is strategic for machine learning applications.

In this repo, I evaluate them.

## Different methods compared
Hardware: Tesla GPUs in a DGX computing server.

* Sequential training one image at a time with accumulation of gradient to simulate bigger batch size.
* LMS (Large Model Support). It implement DNNs computing on the GPU and the DNN parameters stored on the CPU with asynchronous marshaling of parameters between the GPU memory and the main memory.
* Data-parallel training.
* Model-parallel with manual assignement of computing blocks on devices.

The initialization is done once before training, the training batch is done thousands (or millions) of times to fit the neural network.

![Different methods to accelerate training and reduce the memory](training_time_and_memory.png)

## GPIPE with different settings


Model-parallel with automatic (greedy algorithm) assignement of computing blocks on devices.

![Different hardware settings and GPIPE settings](gpipe.png)


## Conclusion

Using model parallelism efficiently requires generally that the neural network can be split into fine grains blocks to spread the memory consumption across target devices. Mixing model-parallelism and data-parallelism have been proposed, but data parallelism is challenging to tune in practice due to the large communication cost at each batch and convergence issues when using aynshronous SGD with stale gradients.
