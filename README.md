Original repository.   
https://github.com/FMInference/FlexGen    

Группа исследователей из Стендфордского университета, Калифорнийского университета в Беркли, Швейцарской высшей технической школы Цюриха, Высшей школы экономики, университета Карнеги — Меллона, а также компаний Yandex и Meta, опубликовала исходные тексты движка для выполнения крупных языковых моделей на системах с ограниченными ресурсами. Например, движок предоставляет возможность создания функциональности, напоминающей ChatGPT и Copilot, через выполнение готовой натренированной модели OPT-175B, охватывающей 175 миллиардов параметров, на обычном компьютере с игровой видеокартой NVIDIA RTX3090, оснащённой 24GB видеопамяти. Код написан на языке Python, использует фреймворк PyTorch и распространяется под лицензией Apache 2.0.    

В состав входит пример скрипта для создания ботов, позволяющего загрузить одну из публично доступных языковых моделей и сразу начать общение (например, выполнив команду "python apps/chatbot.py --model facebook/opt-30b - -percent 0 100 100 0 100 0"). В качестве базовой предлагается использовать опубликованную Facebook крупную языковую модель, обученную на коллекциях BookCorpus (10 тысяч книг), CC-Stories, Pile (OpenSubtitles, Wikipedia, DM Mathematics, HackerNews и т.п.), Pushshift.io (на основе данных Reddit) и CCNewsV2 (архив новостей). Модель охватывает около 180 миллиардов токенов (800 ГБ данных). На тренировку модели было потрачено 33 дня работы кластера с 992 GPU NVIDIA A100 80GB.   

При выполнении модели OPT-175B на системе с одним GPU NVIDIA T4 (16ГБ) движок FlexGen продемонстрировал производительность до 100 раз опережающую ранее предлагавшиеся решения, что делает использование крупных языковых моделей более доступными и позволяет запускать их на системах без специализированных ускорителей. При этом FlexGen может масштабироваться для распараллеливания вычислений при наличии нескольких GPU. Для сокращения размерам модели дополнительно применяется собственная схема сжатия параметров и механизм кэширования моделей.   

В настоящее время FlexGen поддерживает только языковые модели OPT, но в дальнейшем разработчики также обещают добавить поддержку моделей BLOOM (176 миллиардов параметров, поддерживает 46 языков и 13 языков программирования), CodeGen (может генерировать код на 22 языках программирования) и GLM. Пример диалога с ботом на базе FlexGen и модели OPT-30B:   

# FlexGen (Still Working in Progress!)

FlexGen is a high-throughput generation engine for running large language models with limited GPU memory (e.g., a 16GB T4 GPU or a 24GB RTX3090 gaming card!). FlexGen allows **high-throughput** generation by IO-efficient offloading, compression and **large effective batch sizes**.

## Recent Changes (It is getting better thanks to you🙏)
We are glad the public has been really excited about FlexGen. However, our work is still under preparation and not ready for public release / announcement yet.
Thanks to early feedback about this project, we realized that early versions of this README and our paper were a bit unclear about the purpose of FlexGen.
**This is a preliminary effort to lower the resource requirements of LLMs, but it also has a lot of limitations and does not aim to replace use cases when sufficient resources are available.**
Our primary contributions are increasing throughput on single GPU instances - by effectively increasing the batch size.
We're really excited about our techniques for offloading and automatically searching through the design space, as well as our results that suggest it's possible to go down to 4-bit quantization without hurting accuracy.
This naturally trades off latency, but we think it's a really interesting direction for future work.
We'd like to thank everyone for their feedback - keep it coming!

----------

FlexGen was made possible thanks to a collaboration with

<a href="https://cs.stanford.edu/"><img src="https://identity.stanford.edu/wp-content/uploads/sites/3/2020/06/wordmark-nospace-red.png" height="20"></a> &nbsp;&nbsp;&nbsp; <a href="https://sky.cs.berkeley.edu/"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/82/University_of_California%2C_Berkeley_logo.svg/1280px-University_of_California%2C_Berkeley_logo.svg.png" height="22"></a> &nbsp;&nbsp;&nbsp; <a href="https://www.together.xyz/"><img src="https://images.squarespace-cdn.com/content/v1/6358bea282189a0adf57fe16/eef09191-631f-40d9-9bfd-f875b25bcf0b/together-logo-black-transparent2.png" height="20"></a> &nbsp;&nbsp;&nbsp; <a href="https://ds3lab.inf.ethz.ch/"><img src="https://user-images.githubusercontent.com/1608867/220273382-c09669b3-42fd-47c2-b88c-7ed55cb43820.png" height="20"></a>

----------

The high computational and memory requirements of large language model (LLM) inference traditionally make it feasible only with multiple high-end accelerators.
FlexGen aims to lower the resource requirements of LLM inference down to a single commodity GPU (e.g., T4, 3090) and allow flexible deployment for various hardware setups. The key technique behind FlexGen is to trade off between **latency** and **throughput**.

The key features of FlexGen include:  

⚡ **High-Throughput Offloading**.  
Higher-throughput generation than other offloading-based systems (e.g., Hugging Face Accelerate, DeepSpeed Zero-Inference) - sometimes by orders of magnitude. The key innovation is a new offloading technique that can effectively increase the batch size. This can be useful for batch inference scenarios, such as benchmarking (e.g., [HELM](https://github.com/stanford-crfm/helm)) and [data wrangling](https://arxiv.org/abs/2205.09911).

📦 **Extreme Compression**.  
Compress both the parameters and attention cache of models, such as OPT-175B, down to 4 bits with negligible accuracy loss.

🚀 **Scalability**.  
Come with a distributed pipeline parallelism runtime to allow scaling if more GPUs are given.

❌ **Limitation**.  
As an offloading-based system running on weak GPUs, FlexGen also has its limitations.
FlexGen can be significantly slower than the case when you have enough powerful GPUs to hold the whole model, especially for small-batch cases.
FlexGen is mostly optimized for throughput-oriented batch processing settings (e.g., classifying or extracting information from many documents in batches), on single GPUs.

[**Join Discord**](https://discord.gg/JfphDTkBAh)

## Content
- [Benchmark Results](#benchmark-results)
- [Install](#install)
- [Get Started with a Single GPU](#get-started-with-a-single-gpu)
- [API Example](#api-example)
- [Scaling to Distributed GPUs](#scaling-to-distributed-gpus)
- [Roadmap](#roadmap)

## Benchmark Results
### Generation Throughput (token/s)
The corresponding effective batch size is in the bracket. Please see [here](benchmark/batch_size_table.md) for more details.
| System | OPT-6.7B | OPT-30B | OPT-175B |
| ------ | -------- | ------- | -------- |
| Hugging Face Accelerate   | 25.12 (2 on gpu) | 0.62 (8 on cpu	) | 0.01 (2 on disk) |
| DeepSpeed ZeRO-Inference | 9.28 (16 on cpu)  | 0.60 (4 on cpu) | 0.01 (1 on disk) |
| Petals\*                 | -     | -    | 0.05 |
| FlexGen                  | 25.26 (2 on gpu) | 7.32 (144 on cpu) | 0.69 (256 on disk) |
| FlexGen with Compression | **29.12** (72 on gpu) | **8.38** (512 on cpu) | **1.12** (144 on cpu) |

- Hardware: an NVIDIA T4 (16GB) instance on GCP with 208GB of DRAM and 1.5TB of SSD.  
- Workload: input sequence length = 512, output sequence length = 32. The batch size is tuned to **a large value** that maximizes the generation throughput for each system.
- Metric: generation throughput (token/s) = number of the generated tokens / (time for processing prompts + time for generation).  

How to [reproduce](benchmark/flexgen).

### Latency-Throughput Trade-Off
The figure below shows the latency and throughput trade-off of three offloading-based systems on OPT-175B (left) and OPT-30B (right).
FlexGen achieves a new Pareto-optimal frontier with significatnly higher maximum throughput for both models.
Other systems cannot further increase throughput due to out-of-memory.
"FlexGen(c)" is FlexGen with compression.

<img src="https://github.com/FMInference/FlexGen/blob/main/docs/throughput_vs_latency.jpg" alt="image" width="500"></img>

## How It Works
FlexGen can be flexibly configured under various hardware resource constraints by aggregating memory and computation from the GPU, CPU, and disk. Through a linear programming optimizer, it searches for the best pattern to store and access the tensors, including weights, activations, and attention key/value (KV) cache. FlexGen further compresses both weights and KV cache to 4 bits with negligible accuracy loss. 

One key idea of FlexGen is to play the latency-throughput trade-off. Achieving low latency is inherently challenging for offloading methods, 
but the I/O efficiency of offloading can be greatly boosted for throughput-oriented scenarios (see the figure above).
FlexGen utilizes a block schedule to reuse weight and overlap I/O with computation, as shown in figure (b) below, while other baseline systems use an inefficient row-by-row schedule, as shown in figure (a) below.

<img src="https://github.com/FMInference/FlexGen/raw/main/docs/block_schedule.jpg" alt="image" width="500"></img>

## Install
Requirements:  
 - PyTorch >= 1.12 [(Help)](https://pytorch.org/get-started/locally/)

### Method 1: With pip
```
pip install flexgen
```

### Method 2: From source
```
git clone https://github.com/FMInference/FlexGen.git
cd FlexGen
pip install -e .
```

### Optional
Install openmpi for multi-gpu execution.
```
sudo apt install openmpi-bin
```

## Get Started with a Single GPU

### OPT-1.3B
To get started, you can try a small model like OPT-1.3B first. It fits into a single GPU so no offloading is required.
FlexGen will automatically download weights from Hugging Face.
```
python3 -m flexgen.flex_opt --model facebook/opt-1.3b
```

You should see some text generated by OPT-1.3B and the benchmark results.

### OPT-30B
To run large models like OPT-30B, you will need to use CPU offloading. You can try commands below.
The `--percent` argument specifies the offloading strategy for parameters, attention cache and hidden states separately.
The exact meaning of this argument can be found [here](https://github.com/FMInference/FlexGen/blob/9d092d848f106cd9eaf305c12ef3590f7bcb0277/flexgen/flex_opt.py#L1271-L1279).
```
python3 -m flexgen.flex_opt --model facebook/opt-30b --percent 0 100 100 0 100 0
```

### OPT-175B
To run OPT-175B, you need to download the weights from [metaseq](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT) and convert the weights into Alpa [format](https://alpa.ai/tutorials/opt_serving.html#convert-opt-175b-weights-into-alpa-formats).
The numpy weights should be put under `~/opt_weights/opt-175b-np/` by default. You can check the downloaded weights of smaller models for the required format.
You can then try to offloading all weights to disk by
```
python3 -m flexgen.flex_opt --model facebook/opt-175b --percent 0 0 100 0 100 0 --offload-dir YOUR_SSD_FOLDER
```

### How to set the offloading strategy and `--percent`?
We will release an automatic policy optimizer later, but now you have to manually try a few strategies.
The idea of high-throughput generation is to offload parameters and attention cache as much as possible to the CPU and disk if necessary.
You can see the reference strategies in our benchmark [here](https://github.com/FMInference/FlexGen/blob/9d092d848f106cd9eaf305c12ef3590f7bcb0277/benchmark/flexgen/bench_suite.py#L39-L79).
To avoid out-of-memory, you can tune the `--percent` of offload more tensors to the CPU and disk.

## Scaling to Distributed GPUs
If you have more GPUs, FlexGen can combine offloading with pipeline parallelism to allow scaling.
For example, if you have 2 GPUs but the aggregated GPU memory is less than the model size, you still need offloading. FlexGen allow you to do pipeline parallelism with these 2 GPUs to accelerate the generation.
See examples [here](https://github.com/FMInference/FlexGen/tree/main/benchmark/flexgen#distributed-gpus).

## API Example
We demonstrate the usage of FlexGen API in [completion.py](flexgen/apps/completion.py).
This example shows how to run generation for two sentences.
To get the best throughput out of FlexGen, you typically need to batch more sentences.

### Generation API
FlexGen has a generation API following the style of Hugging Face's transformers.
```python
output_ids = model.generate(
	input_ids,
	do_sample=True,
	temperature=0.7,
	max_new_tokens=32,
	stop=stop)
```

### Example Commands
You can use the example commands below.
If you do not have enough GPU/CPU memory, see the [Handle Out-Of-Memory](#handle-out-of-memory) section.

```
# Complete with OPT-6.7B. You need at least 15GB of GPU memory.
python3 -m flexgen.apps.completion --model facebook/opt-6.7b
```

```
# Complete with OPT-30B. You need about 90GB of CPU memory.
python3 -m flexgen.apps.completion --model facebook/opt-30b --percent 0 100 100 0 100 0
```

```
# Complete with instruction-tuned OPT-IML-MAX-30B. You need about 90GB of CPU memory.
python3 -m flexgen.apps.completion --model facebook/opt-iml-max-30b --percent 0 100 100 0 100 0
```

### Handle Out-Of-Memory
If you do not have enough GPU/CPU memory, here are a few things you can try.
They save more memory but run slower.

- Do not pin weights by adding `--pin-weight 0`. This can reduce the weight memory usage on CPU by around 20% or more.
- Enable weight compression by adding `--compress-weight`. This can reduce the weight memory usage by around 70%.
- Offload all weights to disk by using `--percent 0 0 100 0 100 0`. This requires very little CPU and GPU memory.

### More Applications
See [flexgen/apps](flexgen/apps) for more example applications.

## Roadmap
We plan to work on the following features. Community contributions are welcome.

- [ ] Support Apple silicon M1/M2 deployment
- [ ] Support Colab deployment
- [ ] Support more models (BLOOM, CodeGen, GLM)
- [ ] Release the cost model and policy optimizer
