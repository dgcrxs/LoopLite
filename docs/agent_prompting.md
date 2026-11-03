# Agent 提示词工程规范

> 配套代码：`src/apps/agent/tools.py`（工具定义）、`src/apps/agent/executor.py`（调度循环）

## 一、核心原则

Agent 场景的成功指标只有一个：**工具调用准确率**。
不是回答的文采，不是推理的详尽程度，是「调对了没有」。

## 二、工具定义怎么写

### 要这样

```json
{
  "name": "read_excel",
  "description": "读取 Excel 指定工作表，返回前 N 行",
  "parameters": {
    "type": "object",
    "properties": {
      "path": {"type": "string", "description": "文件路径"},
      "sheet": {"type": "string", "description": "工作表名，默认第一个"},
      "n_rows": {"type": "integer", "description": "返回行数，默认 20"}
    },
    "required": ["path"]
  }
}
```

### 不要这样

```json
{
  "name": "read_excel",
  "description": "读取表格",           // ← 太模糊，模型不知道能用在哪
  "parameters": {
    "properties": {
      "opts": {"type": "object"}      // ← 嵌套对象，模型容易编造内部结构
    }
  }
}
```

三条规则：
1. **工具名直白**：`read_excel` 而不是 `re` 或 `tool_1`
2. **参数扁平**：一层 `properties`，别嵌套
3. **枚举优先**：取值有限时直接列 `enum`，别让模型猜

## 三、系统提示词模板

```
你可以调用以下工具。调用时必须且只能输出一个 JSON 对象：
{"tool": "<工具名>", "args": {<参数>}}
不要输出解释文字，不要输出 markdown 围栏。

可用工具：
- read_excel: 读取 Excel 指定工作表，返回前 N 行
  参数: {...}
  示例: {"tool": "read_excel", "args": {"path": "data/raw/sales.xlsx", "n_rows": 10}}

已完成步骤：
（暂无，这是第一步）

请输出下一步的 JSON 调用；若任务已完成，输出
{"tool": "final_answer", "args": {"answer": "..."}}。
```

关键点：
- **「必须且只能输出一个 JSON 对象」** 这句不能省，能显著提升格式合格率
- **给示例**。一个示例比十句描述管用
- **显式告知已完成步骤**，避免模型重复调用同一个工具

## 四、循环次数与延迟的权衡

Agent 是多轮调用场景，延迟会被放大。

| loop_pass | 准确率 | 延迟 |
| :-- | :-- | :-- |
| 1 | 基准 | 基准 |
| 2 | +? | 约 1.7x |

`configs/inference_16g.yaml` 里 agent 场景默认 `loop_pass=1`。

**跑通后必须做 A/B**：
1. 用同一批 100 个真实任务
2. 分别跑 loop_pass=1 和 2
3. 统计：工具调用准确率、JSON 解析率、P50/P95 延迟
4. 准确率提升 < 3% 就用 1，别为一点提升付双倍延迟

## 五、偏好数据（DPO）怎么造

`data/scripts/format_agent.py` 里有完整实现与质量统计。要点：

**chosen 必须**：
- ✅ 合法 JSON
- ✅ 工具名在注册表里存在
- ✅ 参数与 schema 完全对得上
- ✅ 这一步确实推进了任务

**rejected 优先取这三类**（按收益排序）：
1. 格式错误：输出自然语言而不是 JSON
2. 参数幻觉：编造不存在的参数值或工具名
3. 空转：没调工具，直接编了个答案

⚠️ **不要拿长度差异当偏好信号** —— 模型会学会啰嗦。

## 六、DPO 超参

| 参数 | 值 | 为什么 |
| :-- | :-- | :-- |
| `beta` | 0.1 | 太小偏离参考模型太远（能力崩），太大学不动 |
| `learning_rate` | 5e-6 | 比 SFT 低一个数量级 |
| `num_train_epochs` | 1 | DPO 极易过拟合 |
| `lora.r` | 32 | 比 SFT 小，避免冲掉已有能力 |

## 七、评估口径

训练后**必须**跑一批真实工具调用请求，统计两个数：

```
JSON 解析率   = 能解析成 JSON 的输出 / 总输出
工具调用准确率 = 调对工具且参数正确 / 应调用的次数
```

**低于 90% 不要上线。**

`ExecutionTrace` 类已经内置了这两个指标：
```python
trace = executor.run(task)
print(trace.summary())
# 步数: 3 | JSON 解析率: 100% | 工具成功率: 67%
```

## 八、已知短板（来自官方自测数据）

基座在 **Terminal Bench 2.1**（40.45 vs Qwen3.5-27B 的 47.19）和
**τ³-Banking**（18.56 vs 19.59）两项上**落后**。

所以：
- 终端操作类 Agent → 先做基准测试再决定
- 高精度金融规则场景 → 同上
- 办公协作、数据分析、自动化 → 基座强项，优先做

这些是商汤自测数据，第三方复现还没出来，**自己测才是最准的**。
