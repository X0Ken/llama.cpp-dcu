# llama.cpp - Hygon Z100L 双卡验证版

本项目基于 [CNB114514-wksk/llama.cpp-dcu](https://github.com/CNB114514-wksk/llama.cpp-dcu)，针对 Hygon Z100L (`gfx906`) 和 DTK 25.04 修复编译及 softmax 正确性问题。原移植版的 llama.cpp 基线为 `26394b4`，本 fork 的硬件适配提交为 `63071d4`。

本文配置与跑分均来自下述 Z100L 双卡环境。模型为 Qwen3.8-Flash-Next 的 `UD-Q3_K_XL` 量化，使用 CPU/GPU 混合放置、4096 token 上下文及单请求并发。

## 验证环境

| 项目 | 实测配置 |
|---|---|
| DCU | 2 x Hygon Z100L，`gfx906`，wave size 64 |
| 显存 | 每卡 32752 MiB，合计约 64 GiB；两卡显存分别管理 |
| CPU | Hygon C86 7380，虚拟机分配 16 vCPU |
| 内存 | 约 62.8 GiB，无 swap |
| 系统 | Ubuntu 22.04.1，Linux 5.15.0-43 |
| 软件 | DTK 25.04，HIP clang 15，驱动 6.2.31 |
| 互联 | PCIe，两卡之间通过 PCIe 通信 |
| 模型 | `unsloth/Qwen3.8-Flash-Next-GGUF`，`UD-Q3_K_XL`，3 个分片共约 83.81 GiB |
| 模型版本 | `766911a6b7369840a91dbcd95f9f997acaab6cd6`，各分片均核验 SHA256 |

Z100L 与原项目主要实测的 K100_AI (`gfx928`) 不同。原项目的自定义 MMAC GEMM 仅对 `gfx928/gfx936` 启用，本环境使用 gfx906 的 HIP/rocBLAS 等计算路径。原项目的 K100_AI 跑分及调优参数已从本文替换，不代表 Z100L 的性能。

## 从源码编译

本环境需要本地编译。仓库附带的旧预编译包要求更高版本的 GLIBC/libstdc++，无法直接在本次 Ubuntu 22.04 环境运行；该包也不包含本 fork 的修复。

```sh
source /usr/local/hygon/dtk-25.04/env.sh
export LIBRARY_PATH="$ROCM_PATH/llvm/lib/clang/15.0.0/lib/linux:${LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="$ROCM_PATH/.hyhal:$ROCM_PATH:${CMAKE_PREFIX_PATH:-}"

cmake -S . -B build-z100l -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DGPU_TARGETS=gfx906 \
  -DGGML_HIP_GRAPHS=ON -DGGML_HIP_NO_VMM=ON \
  -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON \
  -DGGML_FMA=ON -DGGML_F16C=ON \
  -DCMAKE_HIP_COMPILER="$ROCM_PATH/llvm/bin/clang"

cmake --build build-z100l -j8 \
  --target llama-server llama-cli llama-bench test-backend-ops
./build-z100l/bin/llama-server --list-devices
```

应识别到两张 Z100L，每卡 32752 MiB。DTK 安装在其他路径时，调整 `source` 的路径。

## 模型存放与启动

持久化模型位置：

```text
/share/maasfile/model/public/unsloth/Qwen3.8-Flash-Next-GGUF/UD-Q3_K_XL/
```

本环境共享存储存在访问延迟，因此实测先将已校验的三个分片复制到节点本地磁盘，再进行 SHA256 校验。下面的 `MODEL_DIR` 指向本地缓存；持久化原件保留在 `/share`。本地缓存需要约 84 GiB 额外磁盘空间，三个分片必须放在同一目录。

```sh
source /usr/local/hygon/dtk-25.04/env.sh
MODEL_DIR=/root/code/llama-dcu-validation/model-cache

./build-z100l/bin/llama-server \
  -m "$MODEL_DIR/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf" \
  -ngl all -ncmoe 4 -ot per_layer_token_embd=CPU \
  -sm layer -ts 1,1 \
  -c 4096 -b 512 -ub 128 -fa on -t 8 -np 1 --jinja \
  --host 127.0.0.1 --port 18091
```

| 参数 | 本次验证值 | 用途 |
|---|---|---|
| `-ncmoe` | 4 | 前 4 层的 MoE 权重放到 CPU |
| `-ot` | `per_layer_token_embd=CPU` | 将该大型嵌入张量放到 CPU |
| `-sm` / `-ts` | `layer` / `1,1` | 按层在两卡之间分配 GPU 权重 |
| `-c` / `-np` | 4096 / 1 | 短上下文，单请求并发 |
| `-b` / `-ub` | 512 / 128 | 控制计算缓冲占用 |
| `-fa` | `on` | 启用 Flash Attention |
| `-t` | 8 | CPU 计算线程数 |

这些参数是本次通过验证的起点，尚未做吞吐最优调参。83.81 GiB 模型不能全放入两卡约 64 GiB 显存，CPU 放置是本配置的一部分。

## 正确性与稳定性验证

- 两张卡分别运行现有 `test-backend-ops` 数值测试，共 3994 项通过：每卡 softmax 214 项、量化矩阵运算 115 项、ARGSORT/GATED_DELTA_NET 82 项、F16 Flash Attention 1586 项。
- Qwen3-0.6B 小模型进行 CPU、单卡、双卡对照，所测提示的输出一致且重复可复现。这是运行路径对照，不代表小模型回答的语义一定正确。
- 目标 Q3 模型两次独立启动，关闭提示缓存，合计 21 次请求全部通过指定答案检查；覆盖中英文、计算、商品 SKU 字段提取、长输入检索及代码说明。6 个短答案用例在重复请求及重启后输出一致。
- 资源采样：进程峰值 RSS **52.12 GiB**，系统最低 `MemAvailable` **54.56 GiB**，两卡显存峰值分别为 **25.96 / 28.3 GiB**。

RSS 包含 mmap 文件映射页，`MemAvailable` 包含可回收缓存，两者统计口径不同。

首次检查中，一个包含 `access code` 的提取提示在第二次启动后触发了模型安全拒答，该轮检查未通过。原始失败记录保留，随后改用无安全歧义的商品 SKU 提取提示重新执行上述完整检查。这个现象的根因尚未确定，不能据此声称任意提示的输出在重启后都一致。

以上是短程功能与重启验证，未进行长时间压力、多并发、4096 token 上限压力或更长上下文测试，也未进行量化精度基准评测。

## Z100L 双卡实测跑分

测试日期：2026-10-20。使用上述同一模型及启动参数，模型已加载，串行请求；每组固定生成 128 token，重复 3 次，表中为中位数。通过 `/completion` 流式接口发送固定长度 token 序列，`cache_prompt=false`、`ignore_eos=true`，避免前缀缓存和提前结束影响比较。

prefill/decode 为服务端 `timings` 数据；首字延迟为同机客户端从发出请求到收到首个非空文本片段的时间，包含请求处理开销，不包含模型加载。

| 输入 tokens | 输出 tokens | prefill (tok/s) | 首字延迟 (s) | decode (tok/s) |
|---:|---:|---:|---:|---:|
| 128 | 128 | 57.95 | 2.211 | 17.96 |
| 512 | 128 | 76.00 | 6.739 | 17.99 |
| 1024 | 128 | 79.72 | 12.848 | 17.82 |
| 2048 | 128 | 82.42 | 24.851 | 17.75 |

复现跑分，在服务启动后执行：

```sh
python3 scripts/bench-z100l.py --url http://127.0.0.1:18091 \
  --prompts 128,512,1024,2048 --repeats 3 --n-predict 128
```

该配置涉及 CPU/GPU 混合计算，速度受 CPU、内存带宽、PCIe 和张量放置共同影响。上述数据仅对应表中环境、量化及参数。

## 本 fork 的修复与来源

- 修复 gfx906/clang 15 下未使用 cp_async 路径的非依赖静态断言导致的编译错误。
- 调整 HIP cooperative groups 头文件顺序，避免 CUDA 兼容宏干扰 DTK 25.04 头文件。
- 修复 softmax 模板块大小与实际启动块大小不一致造成的错误；修复后所测 softmax 数值用例全部通过。

Hygon 移植、MoE 适配及 MMAC 等基础工作来自 [原移植项目](https://github.com/CNB114514-wksk/llama.cpp-dcu)。其他架构的代码保留，但本 fork 的上述验证仅覆盖 Z100L/gfx906。通用 llama.cpp 文档见 [README-upstream.md](README-upstream.md)。
