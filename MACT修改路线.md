可以。你现在对工作二最容易迷惑的点在于：**到底应该在MACT哪里动刀，才不是只改prompt，也不是大改到不可控。**

我建议你把工作二明确改成：

# 工作二名称

**基于问题类型路由与结果验证的多智能体表格推理方法**

它的核心不是“重新做一个MACT”，而是在MACT原始“规划Agent+代码Agent+工具”的框架上，加入两个实质模块：

1. **问题类型路由模块**：先判断问题属于查询、筛选、比较、聚合、计算、多跳哪一类，再决定推理流程；
    
2. **结果验证与错误恢复模块**：工具执行后不直接接受结果，而是检查执行结果是否满足问题约束，失败则重新规划或修正代码。
    

这样改动清楚、实验好做、论文里也好讲。

---

# 一、先理解MACT源码主流程

MACT论文中本身是两个Agent协作：planning agent负责在线规划，coding agent负责生成代码，工具包括Python解释器、计算器和Wikipedia搜索。论文明确说MACT包含planning agent、coding agent和工具集合，规划Agent逐步生成plan，代码Agent和工具负责产生中间结果。

源码层面，主要文件结构是：

|文件|作用|
|---|---|
|`tqa.py`|主运行脚本，读取数据集，创建`ReactAgent`并逐条运行|
|`agents.py`|核心Agent逻辑，包括`ReactAgent`、工具调用、代码执行、动作选择|
|`prompts_table.py`|ReAct提示词、表格操作prompt、数值计算prompt|
|`tot.py`|多个候选action/result的选择与打分|
|`utils.py`|表格格式化、答案抽取、评估辅助函数|

GitHub README也说明，`tqa.py`是运行实验的主脚本，`agent.py/agents.py`控制Agent行为，`tot.py`负责LLM选择最佳动作，`prompts_table.py`和`fewshots_table.py`保存实验prompt和few-shot示例。([GitHub](https://github.com/boschresearch/MACT "GitHub - boschresearch/MACT · GitHub"))

源码里的主流程大致是：

```text
tqa.py
  ↓
为每条样本创建 ReactAgent
  ↓
agent.run()
  ↓
循环调用 agent.step()
  ↓
planning model生成多个 Thought/Action
  ↓
as_reward_fn选择一个Action
  ↓
根据Action类型调用工具：
    Retrieve → retriever_tool
    Calculate → calculator_tool / numerical_tool
    Search → Wikipedia
    Finish → 输出答案
  ↓
Observation写入scratchpad
  ↓
继续下一步，直到Finish或达到max_step
```

MACT的prompt里定义的主要Action包括`Retrieve[cells]`、`Calculate[formula/instruction]`、`Search[entity]`和`Finish[answer]`。([GitHub](https://raw.githubusercontent.com/boschresearch/MACT/main/code/prompts_table.py "raw.githubusercontent.com"))

---

# 二、你的修改目标：不要大改框架，只改“决策前”和“执行后”

最清晰的修改路线是：

```text
原MACT：
问题 + 表格
  → ReAct式逐步规划
  → 工具执行
  → 继续规划或Finish

你的工作二：
问题 + 表格
  → 问题类型路由
  → 类型约束下的任务规划
  → 工具执行
  → 结果验证
  → 失败则修正/重规划
  → Finish
```

也就是说，你只需要在MACT原流程中插入两个关键机制：

```text
[新增1] Question Router：在step之前决定问题类型和工具链
[新增2] Result Verifier：在Observation之后检查结果是否可信
```

不用动数据集读取、不用动整体实验框架、不用训练模型。

---

# 三、最推荐的代码修改路线

## 修改点1：新增问题类型路由模块

在`agents.py`里新增一个函数，例如：

```python
def route_question(self):
    """
    根据问题和表头判断问题类型。
    输出：
    - question_type: lookup / filter / comparison / aggregation / arithmetic / multi_hop
    - required_ops: retrieve / filter / sort / groupby / calculate / search
    - target_columns: 可能相关的列名
    - constraints: 问题中的约束条件
    """
```

问题类型建议分成六类：

|类型|例子|推荐工具链|
|---|---|---|
|`lookup`事实查询|“Who is the winner?”|Retrieve→Finish|
|`filter`条件筛选|“Which teams are from China?”|Retrieve/Operate→Finish|
|`comparison`比较排序|“Which country has the most medals?”|Retrieve→Calculate(sort/max)→Finish|
|`aggregation`聚合统计|“How many players are older than30?”|Calculate(count/sum/avg)→Finish|
|`arithmetic`数值计算|“What is the difference between A and B?”|Calculate→Verify→Finish|
|`multi_hop`多步推理|“Among X, which Y has highest Z?”|Retrieve→Calculate→Verify→Finish|

这一块可以用LLM prompt实现，不需要训练。

论文里可以写成：

> 本文首先设计问题类型路由模块，根据问题中的操作意图将复杂表格问答划分为事实查询、条件筛选、比较排序、聚合统计、数值计算和多跳推理等类型，并为不同类型分配差异化的工具调用策略，从而减少无效规划和错误工具选择。

## 修改点3：新增结果验证模块

这是最重要的修改。

原MACT执行工具后，基本是把Observation写入scratchpad，然后进入下一步。它有候选结果投票机制，源码中`as_reward_fn`会在多个候选action里选择较优路径，`retriever_tool`和`numerical_tool`也会采样多个代码结果并用多数一致性选结果。([GitHub](https://raw.githubusercontent.com/boschresearch/MACT/main/code/agents.py "raw.githubusercontent.com"))

你的改法是在工具执行后加一个显式验证：

```python
def verify_observation(self, action_type, argument, observation):
    """
    检查当前工具返回结果是否可信。
    返回：
    - valid: True/False
    - reason: 错误原因
    - repair_suggestion: 修复建议
    """
```

验证规则可以分成两类。

### 规则验证

不需要LLM，直接检查：

```text
1. Observation是否为空；
2. DataFrame结果是否为空；
3. 是否包含目标列；
4. 是否满足筛选条件；
5. 数值计算结果是否可解析；
6. Finish答案是否出现在Observation或执行结果中；
7. 对比较/排序问题，是否真的进行了max/min/sort；
8. 对聚合问题，是否真的进行了sum/count/avg。
```

### LLM验证

当规则验证不够时，再让LLM判断，注意应该避免token大量消耗：

```text
Given the question, action, and observation, determine whether the observation is sufficient and faithful for answering the question.
Return:
- valid: yes/no
- missing_constraints
- suggested_next_action
```

论文里可以写成：

> 本文进一步设计结果一致性验证机制，从结果非空性、列约束满足性、计算操作匹配性和答案证据支撑性等方面对工具输出进行检查。若验证失败，系统将根据错误类型触发重新规划或代码修正。

## 修改点4：启用代码修正机制

MACT源码里已经有`code_revise`函数，用于根据错误信息修正代码，但是你可以把它真正纳入你的流程：当`numerical_tool`或`retriever_tool`执行失败时，不是简单丢弃，而是调用修正器。

你的逻辑可以是：

```text
代码生成
  ↓
执行代码
  ↓
如果报错：
    调用code_revise
    重新执行
  ↓
如果结果为空：
    生成更宽松的检索/计算指令
  ↓
如果验证失败：
    回到任务规划器重新生成Action
```

这会让你的方法有一个清楚的“错误恢复”机制。

论文里可以写成：

> 针对程序化工具执行中常见的语法错误、列名匹配错误和空结果问题，本文设计错误恢复机制，根据执行错误信息和验证反馈对代码或动作指令进行修正，从而提升工具调用成功率。

## 修改点5：新增`Verify`动作（应避免过多验证造成token浪费）

这是最能体现你方法差异的地方。

原动作空间：

```text
Retrieve / Calculate / Search / Finish
```

你的动作空间：

```text
Retrieve / Calculate / Search / Verify / Finish
```

`Verify`可以有两种用法：

### 用法一：中间验证

```text
Action: Verify[Check whether the retrieved rows contain all countries satisfying the condition.]
```

### 用法二：最终验证

```text
Action: Verify[Check whether the answer "China" is supported by the calculated maximum medal count.]
```

实现上，`Verify`不需要复杂工具。可以走一个函数：

```python
def verifier_tool(self, claim, recent_observation, scratchpad):
    prompt = VERIFY_PROMPT.format(
        question=self.question,
        table=self.table_string,
        scratchpad=self.scratchpad,
        claim=claim,
        observation=recent_observation
    )
    return llm_verify(prompt)
```

如果验证通过，就允许`Finish`；如果不通过，就返回：

```text
Observation: Verification failed. Missing constraint: ...
Suggested next action: Calculate[...]
```

论文里可以把这个叫做：

> 显式答案验证动作。

这比MACT原来的“多候选投票”更可解释。

---

# 四、最清晰的最终系统流程

你可以把工作二设计成下面这个流程：

```text
输入：问题Q，表格T，上下文C

Step 1：问题类型路由
  - 判断问题类型：lookup/filter/comparison/aggregation/arithmetic/multi_hop
  - 生成推荐工具链和推理模式

Step 2：类型约束任务规划
  - 规划Agent根据问题类型生成下一步Action
  - Action必须符合当前类型对应的工具链

Step 3：工具协同执行
  - Retrieve：生成代码筛选相关行列
  - Calculate：生成代码执行计算、排序、聚合
  - Search：补充外部实体知识
  - Verify：检查中间结果或最终答案

Step 4：结果验证
  - 检查Observation是否为空
  - 检查是否满足问题约束
  - 检查计算/排序/聚合是否执行正确
  - 检查答案是否被Observation支撑

Step 5：错误恢复
  - 代码错误：调用代码修正
  - 空结果：放宽或重写Action
  - 约束缺失：重新规划下一步
  - 答案无支撑：禁止Finish

Step 6：答案生成
  - 当验证通过后，输出最终答案
```

一句话概括：

> 原MACT是“规划—执行—观察”的在线协作流程；你的方法是“类型路由—约束规划—工具执行—结果验证—错误恢复”的可靠表格推理流程。

---

# 五、具体应该改哪些文件

## 1.`prompts_table.py`

新增三个prompt：

### A.问题类型路由prompt

```python
QUESTION_ROUTER_PROMPT = """
Given a table question and table headers, classify the question into one of:
lookup, filter, comparison, aggregation, arithmetic, multi_hop.

Return in JSON:
{
  "question_type": "...",
  "target_columns": [...],
  "constraints": [...],
  "required_operations": [...],
  "allowed_tools": [...],
  "reasoning_pattern": "..."
}

Question: {question}
Table headers: {headers}
Context: {context}
"""
```

### B.类型约束ReAct prompt

```python
ROUTED_REACT_INSTRUCTION = """
Solve the table question using Thought, Action, Observation steps.

Question type: {question_type}
Allowed tools: {allowed_tools}
Reasoning pattern: {reasoning_pattern}

You should not Finish before the result is verified.

Actions:
(1) Retrieve[instruction]
(2) Calculate[instruction]
(3) Search[entity]
(4) Verify[claim]
(5) Finish[answer]

Table: {table}
Context: {context}
Question: {question}
{scratchpad}
"""
```

### C.验证prompt

```python
VERIFY_PROMPT = """
Given a question, table, reasoning history, and a candidate observation or answer,
judge whether it is sufficient and faithful.

Check:
1. whether the result is empty;
2. whether all constraints in the question are satisfied;
3. whether the required operation is performed;
4. whether the final answer is supported by evidence.

Return JSON:
{
  "valid": true/false,
  "error_type": "none/empty_result/missing_constraint/wrong_operation/unsupported_answer",
  "reason": "...",
  "suggested_next_action": "..."
}

Question: {question}
Reasoning history: {scratchpad}
Candidate: {candidate}
"""
```

---

## 2.`agents.py`

新增或修改以下函数。

### A.新增`route_question`

```python
def route_question(self):
    headers = self.get_table_headers()
    prompt = QUESTION_ROUTER_PROMPT.format(
        question=self.question,
        headers=headers,
        context=self.context
    )
    output = self.call_plan_llm(prompt)
    self.question_profile = parse_json(output)
```

在`run()`开始时调用：

```python
def run(self, reset=True, given_plan=None):
    if reset:
        self.__reset_agent()
    self.route_question()
    while not self.is_halted() and not self.is_finished():
        self.step()
```

### B.修改`_build_agent_prompt`

让它把问题类型信息放进prompt：

```python
def _build_agent_prompt(self, mode="both"):
    return routed_agent_prompt.format(
        examples=self.react_examples,
        table=self.table_string,
        context=self.context,
        question=self.question,
        scratchpad=self.scratchpad,
        question_type=self.question_profile["question_type"],
        allowed_tools=self.question_profile["allowed_tools"],
        reasoning_pattern=self.question_profile["reasoning_pattern"]
    )
```

### C.在`step()`中新增`Verify`分支

原来只有：

```python
if action_type == "Calculate":
...
elif action_type == "Retrieve":
...
elif action_type == "Search":
...
```

你新增：

```python
elif action_type == "Verify":
    observation = self.verifier_tool(argument)
```

### D.在工具执行后调用验证

例如：

```python
valid, feedback = self.verify_observation(action_type, argument, observation)

if not valid:
    observation = f"Observation {self.step_n}: Verification failed. {feedback}"
else:
    observation = f"Observation {self.step_n}: {observation}"
```

### E.新增`verifier_tool`

```python
def verifier_tool(self, claim):
    prompt = VERIFY_PROMPT.format(
        question=self.question,
        scratchpad=self.scratchpad,
        candidate=claim
    )
    return self.call_plan_llm(prompt)
```

### F.新增错误恢复

```python
def repair_action(self, action_type, argument, feedback):
    prompt = REPAIR_PROMPT.format(
        question=self.question,
        action_type=action_type,
        argument=argument,
        feedback=feedback,
        scratchpad=self.scratchpad
    )
    return self.call_plan_llm(prompt)
```

如果验证失败，可以把反馈写入scratchpad，让下一步规划自动修正：

```text
Observation 2: Verification failed. The result does not include the required column "score". Please retrieve rows with both player and score columns.
```

---

## 3.`tqa.py`

新增几个参数，方便做消融实验：

```python
parser.add_argument('--use_router', action='store_true')
parser.add_argument('--use_verifier', action='store_true')
parser.add_argument('--use_repair', action='store_true')
parser.add_argument('--verify_before_finish', action='store_true')
```

这样你可以跑：

```bash
# 原始MACT式
python tqa.py --task wtq ...

# 加问题类型路由
python tqa.py --task wtq ... --use_router

# 加路由+验证
python tqa.py --task wtq ... --use_router --use_verifier

# 完整方法
python tqa.py --task wtq ... --use_router --use_verifier --use_repair --verify_before_finish
```

这对论文实验非常重要。

---

# 六、论文里怎么呈现这个工作

不要说：

> 本章基于MACT进行了修改。

建议写成：

> 针对复杂表格问答中工具调用路径不稳定、推理步骤缺乏约束以及中间结果难以验证的问题，本文提出一种基于问题类型路由与结果验证的多智能体表格推理方法。该方法首先通过问题类型路由模块识别问题中的操作意图，并为不同类型问题分配差异化的推理模式和工具链；随后由任务规划模块在类型约束下生成推理动作，并调用程序化表格工具完成筛选、排序、聚合和计算等操作；最后，设计结果一致性验证与错误恢复机制，对工具输出和最终答案进行约束检查，在发现空结果、约束缺失或计算不一致时触发重新规划或代码修正，从而提升复杂表格问答的准确性与可靠性。

---

# 七、实验应该怎么设计

## 主实验

对比：

| 方法         | 说明                 |
| ---------- | ------------------ |
| Direct LLM | 直接输入表格和问题          |
| CoT        | 让LLM直接逐步推理         |
| PoT        | 让LLM生成Python代码求解   |
| 你的方法       | 类型路由+工具协同+结果验证     |
| 其他方法       | 例如ReAcTable、Dater等 |

## 消融实验

(目前还没考虑好)

| 版本                       | 目的           |
| ------------------------ | ------------ |
| w/o Router               | 验证问题类型路由是否有效 |
| w/o Verifier             | 验证结果验证是否有效   |
| w/o Repair               | 验证错误恢复是否有效   |
| w/o Verify-before-Finish | 验证最终答案验证是否有效 |
| Router only              | 看单独路由是否提升    |
| Verifier only            | 看单独验证是否提升    |
| w/o search               |              |
| w/o calculator           |              |
| w/o Python interpreter   |              |
| w/o coding agent         |              |

## 指标

|指标|作用|
|---|---|
|Answer Accuracy / EM|最终问答准确率|
|Execution Success Rate|代码执行成功率|
|Invalid Observation Rate|空结果/错误结果比例|
|Verified Answer Rate|通过验证的答案比例|
|Error Recovery Rate|出错后修复成功比例|
|Average Steps|平均推理步数|
|Tool Call Count|工具调用次数|

## 分类型实验

这个非常适合你的方法，因为你有Router。

可以统计不同问题类型的效果：

|问题类型|原MACT式|你的方法|提升来源|
|---|---|---|---|
|lookup|可能提升小|稳定||
|filter|提升中等|路由约束有效||
|comparison|提升明显|sort/max验证有效||
|aggregation|提升明显|count/sum/avg工具有效||
|arithmetic|提升明显|计算验证有效||
|multi_hop|提升明显|分步规划+验证有效||

---

# 八、最终最清晰修改路线

你可以按这个顺序实现，千万不要一上来大改。

## 第一阶段：跑通原MACT

目标：确认原始代码能在WTQ/FeTaQA格式上跑通。

只做：

```text
数据格式转换 → tqa.py能读 → ReactAgent能跑 → 输出pred_answer/history
```

## 第二阶段：加问题类型路由

目标：每个样本先输出`question_type`和`allowed_tools`。

修改：

```text
prompts_table.py：新增QUESTION_ROUTER_PROMPT
agents.py：新增route_question()
agents.py：_build_agent_prompt加入question_type
```

实验：

```text
原MACT vs Router-MACT
```

## 第三阶段：加结果验证

目标：每步工具执行后判断结果是否可信。

修改：

```text
prompts_table.py：新增VERIFY_PROMPT
agents.py：新增verify_observation()
agents.py：step()执行工具后调用验证
```

实验：

```text
Router-MACT vs Router+Verifier
```

## 第四阶段：加错误恢复

目标：验证失败后让下一步知道失败原因，必要时修正代码。

修改：

```text
agents.py：新增repair_action()
agents.py：使用已有code_revise逻辑或新增修正prompt
```

实验：

```text
Router+Verifier vs Router+Verifier+Repair
```

## 第五阶段：加最终答案验证

目标：`Finish`之前必须检查答案是否被执行结果支撑。

修改：

```text
step()中遇到Finish时，先verify_final_answer()
通过才finished=True
不通过则写入反馈，继续下一步
```

实验：

```text
完整方法 vs w/o final verification
```

---

# 九、这项工作最后应该长什么样

你的工作二最终不要叫“MACT改进版”，而是叫：

**基于问题类型路由与结果验证的多智能体表格推理方法**

方法结构：

```text
问题类型路由模块
  ↓
类型约束任务规划模块
  ↓
程序化工具执行模块
  ↓
结果一致性验证模块
  ↓
错误恢复与答案生成模块
```

核心创新点：

1. **问题类型路由**：不同类型问题走不同工具链，减少盲目规划；
    
2. **类型约束动作生成**：规划不是自由生成，而是受问题类型和推理模式约束；
    
3. **结果一致性验证**：工具输出后检查是否满足问题约束；
    
4. **错误恢复机制**：针对代码错误、空结果、约束缺失进行修正；
    
5. **最终答案验证**：答案必须被中间结果支撑后才允许输出。
    

你按这条路线做，工作量是可控的，而且和原MACT有明确差异：**MACT强调多Agent协作和工具使用，你的方法强调问题类型驱动、执行结果验证和错误恢复。**



# MACT修改路线：问题类型路由 + 预算约束验证

## 0.总目标

在MACT原始框架上实现一个独立的多智能体表格推理方法，命名为：

**Routed-Verified MACT**

或论文中表述为：

**基于问题类型路由与预算约束验证的多智能体表格推理方法**

不要重写MACT整体结构，不接入工作一的单元格检索模块。输入仍然是原始问题、原始表格和上下文。

整体流程从：

```text
Question + Table
→ ReAct Planning
→ Tool Execution
→ Observation
→ Finish
```

修改为：

```text
Question + Table
→ Question Router
→ Routed ReAct Planning
→ Tool Execution
→ Rule-based Check
→ Selective LLM Verification
→ Error Feedback / Repair
→ Final Verification
→ Finish
```

核心要求：控制token开销，不允许每一步都调用LLM Verifier。

---

## 1.需要修改的文件

优先修改以下文件：

```text
code/tqa.py
code/agents.py
code/prompts_table.py
```

可选修改：

```text
code/utils.py
```

不要大规模重构项目结构，尽量以新增函数和新增参数为主。

---

## 2.在tqa.py中新增运行参数

新增以下命令行参数，用于做主实验和消融实验：

```python
parser.add_argument("--use_router", action="store_true")
parser.add_argument("--use_rule_check", action="store_true")
parser.add_argument("--use_llm_verifier", action="store_true")
parser.add_argument("--use_repair", action="store_true")
parser.add_argument("--verify_before_finish", action="store_true")

parser.add_argument("--max_verify_calls", type=int, default=1)
parser.add_argument("--max_final_verify_calls", type=int, default=1)
parser.add_argument("--max_repair_calls", type=int, default=1)
parser.add_argument("--max_observation_chars_for_verify", type=int, default=1200)
parser.add_argument("--max_scratchpad_steps_for_verify", type=int, default=3)
```

推荐默认实验配置：

```bash
--use_router \
--use_rule_check \
--use_llm_verifier \
--use_repair \
--verify_before_finish \
--max_verify_calls 1 \
--max_final_verify_calls 1 \
--max_repair_calls 1
```

注意：

- `max_verify_calls`只限制中间步骤验证；
    
- `max_final_verify_calls`单独控制最终答案验证；
    
- 中间Verifier最多1次即可，避免token消耗过大；
    
- 最终答案验证最多1次。
    

---

## 3.在prompts_table.py中新增三个Prompt

### 3.1 问题类型路由Prompt

Router只在每个问题开始时调用一次。

Router输入不要包含完整大表，尽量只输入：

```text
question
table title / context
table headers
前几行样例，最多3行
```

新增：

```python
QUESTION_ROUTER_PROMPT = """
You are a table question router.

Classify the question into one of the following types:
lookup, filter, comparison, aggregation, arithmetic, multi_hop.

Return compact JSON only:
{
  "question_type": "...",
  "required_operations": [...],
  "allowed_tools": [...],
  "reasoning_pattern": "...",
  "risk_level": "low/medium/high"
}

Definitions:
- lookup: directly find a value or entity.
- filter: select rows satisfying conditions.
- comparison: compare, rank, max, min, highest, lowest.
- aggregation: count, sum, average, total, number of.
- arithmetic: difference, ratio, percentage, numerical calculation.
- multi_hop: needs multiple conditions or intermediate reasoning.

Question: {question}
Table headers: {headers}
Context: {context}
Table preview: {table_preview}
"""
```

输出必须是JSON，方便解析。

---

### 3.2 类型约束ReAct Prompt

在原MACT ReAct prompt基础上加入路由信息。

不要每一步重复很长的router输出，只保留压缩后的profile：

```python
ROUTED_REACT_INSTRUCTION = """
Solve the table question using Thought, Action, Observation steps.

Question profile:
- Type: {question_type}
- Required operations: {required_operations}
- Allowed tools: {allowed_tools}
- Reasoning pattern: {reasoning_pattern}

Rules:
1. Follow the reasoning pattern when choosing actions.
2. For comparison, aggregation, or arithmetic questions, do not directly Finish before a calculation or verification step.
3. Use tools only when necessary.
4. Keep actions concise.

Available actions:
(1) Retrieve[instruction]
(2) Calculate[instruction]
(3) Search[entity]
(4) Finish[answer]

Table: {table}
Context: {context}
Question: {question}

{scratchpad}
"""
```

注意：不建议一开始就新增显式`Verify[action]`，这样会增加规划复杂度。先把验证作为系统内部机制，而不是让Agent自由决定是否Verify。

---

### 3.3 LLM验证Prompt

Verifier输入必须压缩，不要传完整表格。

只传：

```text
question
question_type
required_operations
recent action
recent observation summary
candidate answer if any
last few scratchpad steps
```

新增：

```python
VERIFY_PROMPT = """
You are a lightweight verifier for table question answering.

Check whether the candidate result is sufficient and faithful for the question.

Return compact JSON only:
{
  "valid": true/false,
  "error_type": "none/empty_result/missing_constraint/wrong_operation/unsupported_answer/uncertain",
  "reason": "...",
  "suggested_next_action": "..."
}

Question: {question}
Question type: {question_type}
Required operations: {required_operations}

Recent action: {action}
Recent observation: {observation}

Recent reasoning history:
{recent_scratchpad}

Candidate answer if any:
{candidate_answer}
"""
```

重要要求：

- 不传完整表格；
    
- 不传全部scratchpad；
    
- observation超过长度时截断；
    
- Verifier只返回JSON，不要长篇解释。
    

---

## 4.在agents.py中新增Question Router

在`ReactAgent`初始化或`run()`开始处新增：

```python
def route_question(self):
    """
    每个样本只调用一次。
    输出question_profile。
    """
```

实现逻辑：

```python
if not self.use_router:
    self.question_profile = {
        "question_type": "unknown",
        "required_operations": [],
        "allowed_tools": ["Retrieve", "Calculate", "Search", "Finish"],
        "reasoning_pattern": "standard react reasoning",
        "risk_level": "medium"
    }
    return
```

若启用router：

```python
headers = self.get_table_headers()
table_preview = self.get_table_preview(max_rows=3)
prompt = QUESTION_ROUTER_PROMPT.format(...)
output = call_llm(prompt)
self.question_profile = parse_json_with_fallback(output)
```

解析失败时不要中断，直接fallback为unknown。

Router结果建议缓存到样本输出中，方便后续实验分析。

---

## 5.修改Agent Prompt构造逻辑

在`_build_agent_prompt()`或对应构造prompt的位置加入：

```python
if self.use_router:
    use ROUTED_REACT_INSTRUCTION
else:
    use original MACT prompt
```

传入：

```python
question_type=self.question_profile["question_type"]
required_operations=", ".join(...)
allowed_tools=", ".join(...)
reasoning_pattern=self.question_profile["reasoning_pattern"]
```

注意控制长度：

- `required_operations`最多保留5个；
    
- `reasoning_pattern`控制在一句话；
    
- 不要把完整JSON塞进每轮prompt。
    

---

## 6.新增Rule-based Check

新增函数：

```python
def rule_check_observation(self, action_type, action_arg, observation):
    """
    不调用LLM，低成本检查工具结果。
    返回：
    {
      "status": "pass/fail/uncertain",
      "error_type": "...",
      "reason": "...",
      "need_llm_verify": True/False
    }
    """
```

规则如下：

### 必须判为fail的情况

```text
1. observation为空；
2. observation包含明显报错，如 Traceback, Error, Exception, KeyError, ValueError；
3. Calculate结果无法解析；
4. comparison问题中没有出现max/min/sort/highest/lowest等操作痕迹；
5. aggregation问题中没有出现count/sum/avg/total等操作痕迹；
6. 当前准备Finish，但answer为空。
```

### 判为uncertain的情况

```text
1. 返回多行，但问题似乎要求唯一答案；
2. 多跳问题已经执行了关键中间步骤；
3. observation很长，无法规则判断；
4. action和question_type不匹配但不一定错误。
```

### 判为pass的情况

```text
1. Retrieve返回非空结果；
2. Calculate成功返回数值或明确实体；
3. lookup/filter类问题的中间结果格式正常；
4. Search返回非空背景信息。
```

---

## 7.新增Verifier触发函数，控制token开销

新增：

```python
def need_llm_verify(self, action_type, action_arg, observation, rule_result, is_final=False):
    """
    决定是否调用LLM Verifier。
    """
```

触发条件：

```text
1. is_final=True，并且开启verify_before_finish；
2. rule_result.status == "fail"，且还有验证预算；
3. rule_result.status == "uncertain"，且问题类型为comparison/aggregation/arithmetic/multi_hop；
4. Calculate动作执行后，问题类型是comparison/aggregation/arithmetic，并且结果不明显；
5. 连续两步没有有效进展。
```

不触发条件：

```text
1. 普通Retrieve成功；
2. lookup类问题中间步骤；
3. Search结果；
4. 已达到max_verify_calls；
5. observation和action已经验证过。
```

新增计数器：

```python
self.verify_calls = 0
self.final_verify_calls = 0
self.repair_calls = 0
self.verification_cache = {}
```

缓存key：

```python
hash(question + action_type + action_arg + observation[:500])
```

避免同一结果重复验证。

---

## 8.新增LLM Verifier函数

新增：

```python
def llm_verify(self, action_type, action_arg, observation, candidate_answer=None, is_final=False):
    """
    低成本LLM验证。
    """
```

输入压缩：

```python
observation = truncate(observation, self.max_observation_chars_for_verify)
recent_scratchpad = get_last_k_steps(self.scratchpad, k=self.max_scratchpad_steps_for_verify)
```

调用VERIFY_PROMPT。

返回：

```python
{
  "valid": bool,
  "error_type": str,
  "reason": str,
  "suggested_next_action": str
}
```

解析失败时：

```python
return {
  "valid": True,
  "error_type": "parse_failed",
  "reason": "Verifier parse failed, skip verification.",
  "suggested_next_action": ""
}
```

不要因为Verifier输出格式错误导致主流程中断。

---

## 9.修改step()工具执行后的逻辑

原MACT大致是：

```text
生成Action
执行Tool
把Observation写入scratchpad
进入下一步
```

修改为：

```text
生成Action
执行Tool
Rule-based Check
判断是否需要LLM Verify
如果验证通过：
    写入正常Observation
如果验证失败：
    写入Verification Feedback
    如开启repair，则触发修正或让下一步重新规划
进入下一步
```

伪代码：

```python
observation = execute_tool(action_type, action_arg)

if self.use_rule_check:
    rule_result = self.rule_check_observation(action_type, action_arg, observation)
else:
    rule_result = {"status": "pass", "need_llm_verify": False}

verify_result = None
if self.use_llm_verifier and self.need_llm_verify(...):
    verify_result = self.llm_verify(...)

if verify_result and not verify_result["valid"]:
    feedback = format_verification_feedback(verify_result)
    observation = observation + "\nVerification feedback: " + feedback

    if self.use_repair and self.repair_calls < self.max_repair_calls:
        # 不要立刻多次调用LLM修复，优先把反馈写入scratchpad
        self.repair_calls += 1
        observation += "\nPlease revise the next action according to the feedback."

self.scratchpad += format_observation(observation)
```

注意：

- 不建议验证失败后立刻额外调用一个Repair LLM；
    
- 更省token的方式是把失败原因写入scratchpad，让下一轮planner自然修正；
    
- 只有代码报错且已有MACT的code_revise函数时，才调用一次repair。
    

---

## 10.Finish前强制最终验证

当Action为`Finish[answer]`时，不要立即结束。

新增逻辑：

```python
if action_type == "Finish":
    if self.verify_before_finish:
        rule_result = rule_check_final_answer(answer)
        if need final llm verify and budget remains:
            verify_result = llm_verify(..., candidate_answer=answer, is_final=True)

        if verification passed:
            self.finished = True
            self.answer = answer
        else:
            self.scratchpad += "Final verification failed: ... Please continue reasoning."
            self.finished = False
    else:
        self.finished = True
```

最终验证输入也不要传完整表格，只传：

```text
question
question_type
answer
last 3 steps scratchpad
last observation
```

如果最终验证失败，只允许继续1轮或2轮，避免无限循环。

可以设置：

```python
self.max_extra_steps_after_final_fail = 2
```

---

## 11.降低token消耗的关键策略

必须实现以下策略：

```text
1. Router每个问题只调用一次；
2. Router不输入完整表格，只输入表头、上下文、前3行；
3. 中间LLM Verifier默认最多调用1次；
4. Final Verifier默认最多调用1次；
5. 普通Retrieve/Search不调用Verifier；
6. Rule-based Check优先，只有fail/uncertain才考虑LLM Verifier；
7. Verifier不输入完整表格，只输入最近Observation和最近3步scratchpad；
8. Observation超过1200字符截断；
9. scratchpad只截取最近3步给Verifier；
10. 验证失败后优先写反馈，不额外调用Repair LLM；
11. 使用verification_cache避免重复验证；
12. 保留原MACT最大步数限制，避免验证失败导致循环。
```

---

## 12.实验版本设计

实现后至少支持以下版本：

```bash
# 1.原始MACT
python tqa.py ...

# 2.只加Router
python tqa.py ... --use_router

# 3.Router + Rule Check
python tqa.py ... --use_router --use_rule_check

# 4.Router + Rule Check + Selective Verifier
python tqa.py ... --use_router --use_rule_check --use_llm_verifier --max_verify_calls 1

# 5.完整方法
python tqa.py ... \
  --use_router \
  --use_rule_check \
  --use_llm_verifier \
  --verify_before_finish \
  --use_repair \
  --max_verify_calls 1 \
  --max_final_verify_calls 1 \
  --max_repair_calls 1
```

论文消融实验对应：

```text
Full Method
w/o Router
w/o Rule Check
w/o LLM Verifier
w/o Final Verification
w/o Repair
```

---

## 13.输出日志字段

每个样本输出中增加以下字段，便于论文分析：

```json
{
  "question_id": "...",
  "question": "...",
  "gold_answer": "...",
  "pred_answer": "...",
  "question_profile": {...},
  "num_steps": 0,
  "num_tool_calls": 0,
  "num_verify_calls": 0,
  "num_final_verify_calls": 0,
  "num_repair_calls": 0,
  "verification_failures": [...],
  "execution_errors": [...],
  "history": "..."
}
```

这些字段后续可以统计：

```text
Answer Accuracy
Execution Success Rate
Verification Pass Rate
Invalid Observation Rate
Average Tool Calls
Average Verify Calls
Average Steps
Token开销变化
```

---

## 14.最终代码修改顺序

按下面顺序让Codex实现，不要一次性全改。

### 第一阶段：Router

```text
1. 在prompts_table.py加入QUESTION_ROUTER_PROMPT；
2. 在agents.py加入route_question()；
3. 在run()开始时调用route_question()；
4. 输出question_profile到日志；
5. 保证不用router时原MACT结果不变。
```

### 第二阶段：Routed Prompt

```text
1. 加ROUTED_REACT_INSTRUCTION；
2. 修改prompt构造函数；
3. 当use_router=True时使用routed prompt；
4. 跑通Router-only实验。
```

### 第三阶段：Rule Check

```text
1. 新增rule_check_observation()；
2. 工具执行后调用；
3. 不调用LLM；
4. 把rule_result写入日志；
5. 跑通Router + Rule Check。
```

### 第四阶段：Selective Verifier

```text
1. 加VERIFY_PROMPT；
2. 新增need_llm_verify()；
3. 新增llm_verify()；
4. 加verify_calls预算；
5. observation和scratchpad截断；
6. 加verification_cache。
```

### 第五阶段：Final Verification

```text
1. 修改Finish逻辑；
2. Finish前调用final verification；
3. 验证失败后最多继续1-2步；
4. 输出final verification日志。
```

### 第六阶段：Repair

```text
1. 优先使用scratchpad反馈修正；
2. 只有代码报错时才调用一次已有code_revise或repair_action；
3. 加max_repair_calls限制。
```

---

## 15.论文中的方法表述

论文里不要写“本文基于MACT进行了简单修改”。

可以写成：

```text
针对复杂表格问答中问题类型差异大、工具调用路径不稳定以及执行结果缺乏可靠性检查的问题，本文提出一种基于问题类型路由与预算约束验证的多智能体表格推理方法。该方法首先通过问题类型路由模块识别问题中的操作意图，并为不同类型问题分配差异化的推理模式和工具链；随后在类型约束下进行任务规划和程序化工具执行；最后设计规则优先、按需触发的大模型验证机制，在控制额外token开销的同时，对关键中间结果和最终答案进行一致性检查，从而提升复杂表格问答的准确性、稳定性与可解释性。
```

---

## 16.最终方法亮点

最终方法相对于原MACT的差异点应体现为：

```text
1. 从自由规划改为问题类型路由下的约束规划；
2. 从执行后直接进入下一步改为规则优先检查；
3. 从无显式验证改为预算约束的选择性验证；
4. 从直接Finish改为最终答案验证后Finish；
5. 从盲目重试改为基于错误反馈的轻量恢复。
```

核心原则：

```text
不要每一步Verifier；
不要每次传完整表格；
不要无限修复；
不要让验证模块比原推理更贵。
```