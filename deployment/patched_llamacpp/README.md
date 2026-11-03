# patched_llamacpp —— llama.cpp 适配位（当前不可用）

## 现状（2026-11）

**结论：GGUF / llama.cpp 路线目前走不通，不作为阶段一的依赖路径。**

原因：基座 `sensenova/SenseNova-Loop-9B-Preview` 的架构是 `sensenova_loop`，包含
普通 GGUF 转换器不支持的结构：

1. **混合注意力**：每 4 层中 3 层线性注意力 + 1 层全注意力。
   llama.cpp 的注意力实现是单一范式，没有线性注意力算子。
2. **mHC 双流残差**：残差流是 `[B, T, S, D]` 四条（双流 × 主副），
   靠 Sinkhorn 归一化的双随机矩阵混合。llama.cpp 的残差是单流 `[B, T, D]`。
3. **循环块**：中间 24 层权重共享并跑 2 遍。llama.cpp 的图是按「顺序层」展开的，
   没有循环边，也没有 BPTT/退出路由的概念。

## 这对原方案意味着什么

原方案把「GGUF Q4_K_M 5.2GB / Q8_0 8.9GB」当作 16G 显卡的落地手段。
这条路暂时封死。替代方案：

| 目标 | 替代手段 | 落地位置 |
| :--- | :--- | :--- |
| 16G 显卡跑推理 | FP8 权重存储（约 10GB）+ CPU offload 兜底 | `src/utils/quantize.py` 的 `quantize_fp8()` |
| 服务化部署 | LightLLM（tp4，官方给了示例）/ SGLang 指定 PR 分支 | `docs/hpc_deployment.md` |
| A100 40G 推理 | BF16 + tensor_parallel=2 | `configs/inference_16g.yaml` 的 `a100_40g` 档 |

## 什么时候可以回来

盯这几个信号，任一成立即可重启 GGUF 路线：

- [ ] llama.cpp 上游合入 `sensenova_loop` 架构支持
- [ ] 社区放出该架构的 GGUF 转换脚本
- [ ] 官方发布量化版权重（模型页的 Files and versions）

## 若将来要打补丁，工作量参考

按本目录存放补丁的计划，主要改动点：

```
llama.cpp/
├── src/llama-arch.cpp        # 新增 LLM_ARCH_SENSENOVA_LOOP
├── src/llama-model.cpp       # 循环块加载：24 层权重映射到 48 次执行
├── src/llama-graph.cpp       # 建图时插入循环边 + mHC 混合节点
├── ggml/src/ggml-cuda/       # 线性注意力 CUDA 算子（chunked linear attn）
└── convert_hf_to_gguf.py     # 权重名映射 + mHC 参数 FP32 保留
```

其中「线性注意力 CUDA 算子」是最大工作量，约需 1~2 人周。

## 备注

放这个目录不是为了现在做，是为了**将来不用重新调研一遍**。
真要做之前，先确认官方是否已经支持——大概率他们会先出。
