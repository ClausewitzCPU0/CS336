---
workflow:
  name: ChatGPT-Web-Course-Note-Workflow
  version: v4.9.4
course:
  name: "CS336: Language Modeling from Scratch"
  term: "Spring 2026"
  instructors: "Tatsunori Hashimoto, Percy Liang"
chapter:
  index: 12
  local_label: P12
  local_title: "斯坦福 CS336：评估"
  official_unit_label: Lecture
  official_unit_number: 12
  official_title: Evaluation
  date: "2026-05-06"
  instructor: "Percy Liang"
provenance:
  lecture_video_term: "Spring 2026"
  official_materials_term: "Spring 2026"
---

# Lecture 12 — Evaluation（评估）

> **章节映射**：本地 `P12` → Spring 2026 official **Lecture 12 — Evaluation**。本地视频时长约 1:18:34，字幕与视频从开场、中段到结尾均能一致配对。

## 配套资料

- [Spring 2026 课程网站与 Schedule](https://cs336.stanford.edu/)
- [Spring 2026 官方 lectures repository](https://github.com/stanford-cs336/lectures)
- [Lecture 12 executable lecture：`lecture_12.py`](https://github.com/stanford-cs336/lectures/blob/main/lecture_12.py)
- 本讲没有发现被 Spring 2026 官方页面明确指定为 **Lecture 12 required/assigned reading** 的 PDF；`lecture_12.py` 中出现的 benchmark paper 链接按参考资料处理，因此不自动打包进 `papers/`。

## 本讲主线

此前课程已经讨论了模型架构、optimizer、training loop、kernels、parallelism、scaling laws 和 inference。接下来要讨论 training data，但数据会直接塑造模型行为：训练代码、自然语言或 DNA 序列，会得到截然不同的能力分布。因此在讨论“拿什么数据训练”之前，先要回答更基础的问题：**我们究竟希望模型表现成什么样，以及如何把这种希望变成可测量的指标？**

这就是 evaluation 的核心：

> **abstract construct → concrete metric**

例如“会对话”“会推理”“安全”“适合真实工作”都是抽象概念；评估的工作，是把这些概念落实为 prompts、environments、judges 与 metrics。

![Evaluation 的核心难题：从抽象目标到具体指标](assets/coverage/0130-01b-core-challenge-text.jpg)

本讲依次讨论六类评估：**perplexity、exam benchmarks、chat benchmarks、agentic benchmarks、pure reasoning benchmarks、safety benchmarks**，最后再从 **realism、validity、评估目的与评估对象** 三个层面反思这些数字到底意味着什么。

---

## 1. “好模型”并没有唯一含义（约 01:13–05:16）

最机械的评估流程看起来很简单：定义 prompts，把 prompts 发给模型得到 responses，再计算 accuracy。但真正困难的是：**先决定“什么叫好”**。

课堂用几个不同视角说明这个问题：

1. **Benchmark capability**：模型在一组标准 benchmark 上分数高不高，例如综合能力榜单。
2. **Capability × cost**：如果两个模型能力相近，推理成本可能决定哪个更实用。
3. **Human preference**：用户更喜欢哪个模型的回答，例如 Arena 式的成对比较。
4. **Actual usage / willingness to pay**：真实用户最终选择并付费使用哪些模型，也是一种经济意义上的“好”。

![能力与推理成本并不完全一致](assets/coverage/0205-03-cost-vs-intelligence.jpg)

![OpenRouter 使用量提供了另一种“真实选择”视角](assets/coverage/0274-05-openrouter-usage.jpg)

这些指标没有哪个天然是“真相”。它们分别测量不同 construct。更重要的是，开发者会围绕评估指标持续优化，因此 **evaluation 本身会直接塑造模型的发展方向**。

---

## 2. Perplexity：从语言模型本身出发的评估（约 05:16–18:10）

### 2.1 定义

语言模型定义了 token sequence 上的概率分布 $p(x)$。给定测试序列 $D=(x_1,\ldots,x_N)$，perplexity 可以写成：

$$
\mathrm{PPL}(D)=p(D)^{-1/N}
=\exp\left(-\frac{1}{N}\sum_{i=1}^{N}\log p(x_i\mid x_{<i})\right).
$$

直观上，它衡量模型给测试数据分配了多少 probability mass。训练阶段最小化 token-level negative log-likelihood；最自然的传统评估，就是在 held-out test set 上计算同一量。

### 2.2 传统范式：in-distribution evaluation

经典 language modeling 常见数据集包括：

- Penn Treebank（PTB）；
- WikiText-103；
- One Billion Word Benchmark（1BW）。

传统做法是：在某个数据集的 train split 上训练，再在同分布 test split 上报告 perplexity。课堂提到 2016 年的纯 CNN/LSTM 系统在 1BW 上把 perplexity 从约 51.3 推到 30.0，代表了神经语言模型相对 n-gram / hybrid 方法的一次明显跃迁。

### 2.3 GPT-2：从 in-distribution 转向 zero-shot OOD

GPT-2 的评估方式改变了范式。它训练在 WebText（课堂给出的规模约 40 GB）上，却 zero-shot 地跑传统 LM benchmark，因此属于 **out-of-distribution evaluation**。课堂展示的论文表格中，1.5B 参数模型在小型 PTB 上能做到约 35 perplexity，而此前 SOTA 约为 46；在数据更大的 benchmark 上，则未必超过专门的 in-distribution 模型。

![GPT-2 在传统语言模型数据集上的 zero-shot perplexity](assets/coverage/0505-07-gpt2-perplexity.jpg)

这里已经出现一个后来会反复出现的问题：**训练集到底包含了什么？** 即使没有直接训练在 benchmark test set 上，Web-scale 数据也可能与 benchmark 的来源发生重叠。

### 2.4 “Perplexity is all you need”的直觉

课堂把一种在 scaling research 中很有影响力的信念概括成：只要持续把 perplexity 压低，最终就会得到越来越通用的能力。

设真实数据分布为 $t$，模型为 $p$。从 cross-entropy 的角度，$p=t$ 时达到最优；若用自然对数定义 entropy，则对应的最小 perplexity 为：

$$
\mathrm{PPL}_{\min}=\exp(H(t)).
$$

因此当 $p$ 越来越接近 $t$ 时，条件分布 $p(\text{solution}\mid\text{problem})$、$p(\text{answer}\mid\text{question})$ 也应越来越接近真实分布。这就是“继续 scale、继续降低 loss/perplexity，能力自然会出现”的思想基础之一。

> 技术上应区分 **entropy / cross-entropy** 与 **perplexity**：perplexity 是平均 cross-entropy 的指数，而不是 entropy 本身。

### 2.5 Perplexity 也可能“测得太多”

例子：`Stanford was founded in 1885`。

如果关心的是事实知识，`1885` 很重要；但标准 perplexity 同样会惩罚模型对 `Stanford`、`was`、`founded` 等所有 token 的预测误差。很多 token 可能与想测的能力无关。

一种修正是 **conditional perplexity**：给定 prompt，只计算 response 的概率。

$$
\mathrm{PPL}(r\mid q)=p(r\mid q)^{-1/|r|}.
$$

这样可以把测量焦点放到真正关心的 response token 上。

### 2.6 有些 benchmark 是“perplexity in disguise”

**LAMBADA** 是 cloze / fill-in-the-blank 任务：上下文经过挑选，使最后一个词需要较长距离的信息才能确定。虽然最后报告 accuracy，本质上仍是对特定位置 next-token probability 的评估。

![LAMBADA：把 perplexity 聚焦到需要长程上下文的位置](assets/coverage/0815-10-lambada.jpg)

**HellaSwag** 是 multiple-choice sentence completion。表面上是选择题，实质仍是在比较不同 continuation 的合理性。

![HellaSwag 的句子补全例子](assets/coverage/0900-11-hellaswag.jpg)

### 2.7 一个容易忽略的 leaderboard 问题

如果 benchmark 接口要求参赛者提交 `LM(test_data)` 的 log probability，那么 evaluator 必须相信提交者返回的概率真的是合法分布——例如总概率确实归一化。否则一个“永远返回 logprob=0”的伪 LM 就能得到荒谬的高分。

而 downstream task 通常只需要黑盒接口：

```text
response = LM(prompt)
metric = score(response, reference)
```

这也是为什么 perplexity evaluation 与普通 task accuracy 在评估契约上不同。

### 2.8 小结

Perplexity 仍然重要，尤其适合：

- 跟踪 pretraining 进度；
- 拟合 scaling laws，因为该指标随规模变化较平滑；
- 在私有文本上做 evaluation，因为只需要 held-out text 与 log probabilities。

但如果目标是证明模型在真实任务上有用，仅有 perplexity 通常不够。

---

## 3. Exam benchmarks：难度可控，但离真实使用有距离（约 18:14–30:33）

考试型 benchmark 的优势与人类考试相似：可以控制 subject、difficulty，并尽量设计成有明确答案、容易自动评分。

### 3.1 MMLU

**MMLU（Massive Multitask Language Understanding）**覆盖 57 个 subject，包括数学、美国历史、法律、道德等。课堂特别提醒：尽管名字里有 “Language Understanding”，它实际上更接近 **knowledge + reasoning** 测试。

原始工作用 GPT-3 做 few-shot prompting：prompt 中先给若干 question/answer demonstrations，再让模型回答最后一道题。

![MMLU：不同模型规模在多类任务上的表现](assets/coverage/1216-14-mmlu.jpg)

随着 frontier models 变强，MMLU 很快接近 saturation。课堂现场打开动态榜单，就是为了强调 benchmark 会很快失去区分度：一旦大量模型接近上限，就难以继续区分更强的系统。

### 3.2 MMLU-Pro：benchmark 变简单以后，就把它做难

MMLU-Pro 的主要变化：

- 删除 noisy / trivial questions；
- 从 4 个选项扩展为 10 个；
- 用 chain-of-thought 给模型更充分的推理机会。

同一批模型在 MMLU-Pro 上的 accuracy 相比 MMLU 可下降约 16–33 个百分点。这个例子体现了贯穿本讲的一条规律：**模型不断把旧 benchmark 做饱和，benchmark 也不断被重新设计得更难。**

![MMLU-Pro：更难的选择题设置](assets/coverage/1348-16-mmlu-pro.jpg)

### 3.3 GPQA：Google-Proof 的研究生级问题

GPQA 的目标是让“非专家拿 Google 搜一搜”也很难解决。题目来自 61 位 PhD contractors，并经历 expert validation、question revision、second expert review 与 non-expert testing。

![GPQA 的人工题目生产与验证流程](assets/coverage/1468-17-gpqa-process.jpg)

课堂给出的原论文数据：

- PhD experts：约 **65%**；
- non-experts 在有 Google、30 分钟条件下：约 **34%**；
- 当时 GPT-4：约 **39%**。

而到本讲的 2026-05 课堂快照，GPQA 榜单已经进入约 94% 区间。这里的重点不是某个具体 leaderboard 数字，而是：**即使最初非常困难的 benchmark，也可能很快被 frontier models 饱和。**

### 3.4 Contamination 并不只是“直接把 test set 喂进训练”

课堂中途有学生追问：这些 benchmark 会不会已经进入训练集？Percy 的回答是：我们往往不知道，因为训练数据没有完全公开；而且 contamination 很 subtle。

即使 test questions 本身没有直接出现在训练集中，题目也可能由某些网页、教材、论坛内容派生，而这些 source material 已经进入训练数据。后面讨论 validity 时会系统回到这个问题。

### 3.5 Humanity's Last Exam（HLE）

HLE 试图继续把难度推高：约 2500 道题，覆盖多学科、multimodal、multiple-choice 与 short-answer。题目采用 crowdsourcing，并以约 **$500K prize pool + co-authorship** 激励出题者，再通过 frontier LLM filtering 与多阶段人工 review 过滤。

![HLE 的出题、过滤与审核流程](assets/coverage/1715-19-hle-pipeline.jpg)

![HLE 在课堂材料中展示的 benchmark 结果](assets/coverage/1730-20b-hle-results-chart.jpg)

本讲还指出 HLE 保留 private holdout，以降低训练污染风险。课堂展示的 2026-05 动态榜单快照中，Mythos 约为 **64.7**；当时仍留有明显 headroom，说明它尚未像 MMLU / GPQA 那样完全饱和。

### 3.6 多选题真正的局限不是“太容易”

Multiple choice 可以做得任意困难，所以“选择题天然简单”并不成立。真正的问题是 **format restricts the questions you can ask**：

- 真实用户通常不向 assistant 提四选一题；
- 真实请求经常是 open-ended；
- 很多请求没有唯一正确答案；
- prompt 甚至可能是不完整、含糊或需要互动澄清的。

因此 exam benchmarks 很适合控制 difficulty，却可能牺牲 ecological validity。

---

## 4. Chat benchmarks：如何评价没有标准答案的回答？（约 32:23–44:58）

课堂用一个很普通的真实式问题切入：

> “I want to make a beet salad with cheese. What herbs would work well and what wouldn't work well?”

![开放式 beet salad 请求：没有唯一 ground truth](assets/coverage/1955-22-open-ended-beet-example.jpg)

这类任务不能用 exact match。核心问题变成：**谁来判断，以及判断标准是什么？**

### 4.1 Chatbot Arena：让人做 pairwise preference

Arena 的机制：

1. 真实用户输入 prompt；
2. 系统随机给出两个匿名模型的 responses；
3. 用户判断 A better / tie / both bad / B better；
4. 聚合大量 pairwise comparisons 形成模型排序。

![Arena 的双模型匿名比较界面](assets/coverage/2015-23-arena-interface.jpg)

课堂用 Elo 形式解释 pairwise ranking：

$$
P(A\text{ beats }B)
=\frac{1}{1+10^{(\mathrm{ELO}_B-\mathrm{ELO}_A)/400}}.
$$

然后拟合各模型的 Elo，使观测到的 pairwise outcomes 具有较高 likelihood。

![从 pairwise comparisons 拟合 Elo ranking](assets/coverage/2065-24-elo-pairwise-model.jpg)

这个机制有几个明显优点：

- prompts 来自真实用户，有一定 ecological realism；
- 人类只需比较两个回答，而不是给几十个模型打绝对分；
- 不要求所有模型都跑同一批 prompt，只要 comparison graph 足够连通即可；
- 新 prompt、新模型可以不断加入，评估天然是动态的。

但也有重要偏差来源：

- “Arena user” 不是一个明确、可控的人群分布；
- 可能有 spam / gaming；
- 一个二元 preference 会混合 **style 与 correctness**；
- 用户之所以提问，往往正是因为不知道答案，因此不一定能判断 factual correctness；
- sycophantic、迎合式回答可能比正确但不讨喜的回答更容易获胜。

### 4.2 AlpacaEval：把 human judge 换成 LLM judge

AlpacaEval 使用固定的一组 instructions，把待测模型输出与 baseline model 的输出进行比较，再让 LLM judge 判断哪个更好，指标是相对 baseline 的 win rate。

优势是便宜、快、可重复；问题是 judge 本身带偏差。课堂重点讲了 **length bias**：早期 LLM judges 偏爱更长回答，导致一些模型只要学会“写得更长”就能 gaming leaderboard。AlpacaEval 2.0 后来通过 regression debiasing 修正这一问题。

如何评价一个 evaluation metric 本身？一种 sanity check 是看它与另一种评价是否相关。课堂展示的模型集合上，AlpacaEval 与 Chatbot Arena 的相关性约 **0.98**。

![AlpacaEval 与 Chatbot Arena 的高相关性](assets/coverage/2442-27-alpacaeval-correlation.jpg)

不过这种 correlation 只对当时那组模型成立；当模型分布、能力区间或 judge 发生变化时，关系可能失效。

### 4.3 WildBench：给 judge 一份 task-specific checklist

WildBench 从约 100 万条 human-chatbot conversations 中选取 1024 个 examples，并用 LLM judge 评分。关键改进是：**先为具体 prompt 生成 checklist / rubric，再让 judge 按清单判断。**

![WildBench：用 checklist 约束开放式评价](assets/coverage/2508-28-wildbench.jpg)

这解决的是 open-ended evaluation 的一个根本问题：直接问“这个回答好不好？”本身就定义不清。给出 rubric 后，judge 才知道应该分别检查 factuality、instruction following、style、completeness 等哪些维度。

### 4.4 本节结论

对于相近的两个回答，**pairwise comparison 往往比 7/10、8/10 这种 absolute score 更有信号**。但无论 judge 是人还是 LLM，都要考虑 bias。可靠做法通常包括：

- pairwise comparison；
- multiple judges / ensemble；
- 清晰的 rubric / checklist；
- 对 judge 本身做校验，而不是只相信一个 leaderboard 数字。

---

## 5. Agentic benchmarks：评估模型“做了什么”（约 44:58–53:56）

Chat benchmark 主要测 **what LMs say**；agentic benchmark 要测 **what LMs do**。

课堂给出的定义可以写成：

$$
\text{Agent} = \text{Language Model} + \text{Agent Scaffold}.
$$

Scaffold 是决定何时调用模型、可以调用哪些工具、如何迭代、如何管理上下文与状态的外部逻辑。因此 agent benchmark 测到的从来不只是 base model。

### 5.1 SWE-Bench：issue → patch → unit tests

SWE-Bench 包含 12 个 Python repositories 中的 2294 个任务。输入是 codebase + GitHub issue description，agent 要提交 patch，最终通过 unit tests 判断任务是否完成。

![SWE-Bench：从 issue 与代码上下文生成 patch](assets/coverage/2778-31-swebench.jpg)

这个评价目标很清晰：不必判断回答“写得好不好”，直接运行测试即可。但后来又出现 **SWE-Bench Verified**，原因正是原任务/测试中也存在质量问题。课堂的 2026-05 leaderboard 快照展示了这类 coding benchmark 从早期约 16% 快速上升到约 93% 的过程。

### 5.2 Terminal-Bench：把环境统一成 terminal

Terminal-Bench 试图覆盖更通用的任务：只要任务能在 computer terminal 中完成，就可以定义成 agent task。官方 lecture material 给出的数据是 229 个 crowdsourced tasks、93 位 contributors，其中 89 个构成 Terminal-Bench 2.0。

![Terminal-Bench 的任务/agent/evaluator 结构](assets/coverage/2895-33-terminalbench.jpg)

有些任务对人类专家需要约 1 小时，对 junior 甚至可能需要一周以上。更关键的是：**同一个 language model，换 agent scaffold 后成绩也可能显著不同。**

### 5.3 CyBench：40 个 CTF cybersecurity tasks

CyBench 使用 40 个 Capture The Flag tasks。Agent 可以查看 source code、访问 web server、运行命令，目标是找到证明成功入侵的 flag。

![CyBench 的 CTF task 结构](assets/coverage/2978-36-cybench.jpg)

早期 scaffold 很简单：模型基于当前 memory 生成 action，environment 返回 feedback，再把结果持续拼回 context。

![早期 CyBench agent 的 action–environment feedback loop](assets/coverage/3012-37-cybench-agent.jpg)

随着任务变复杂，这种“把全部历史一直塞进 context”的方式会迅速失效。

### 5.4 MLE-Bench：把 Kaggle competition 变成 agent benchmark

MLE-Bench 包含 75 个 Kaggle competitions，需要 agent 读取数据、写代码、训练模型、生成 submission，再由 competition metric 评分。

![MLE-Bench：competition、agent、grader 形成闭环](assets/coverage/3085-39-mlebench.jpg)

这种任务把问题从“模型会不会回答 ML 问题”推进到“模型能不能完整执行一套 ML 工程流程”。

### 5.5 Scaffold 本身已经成为能力的一部分

要完成长任务，课堂总结出几类越来越重要的 scaffold 技术：

- **Explicit planning**：维护可勾选的 todo list，而不是纯 stream-of-consciousness；
- **Hierarchical delegation**：主 agent 调用带干净 context 的 sub-agent，只拿回结果，避免把所有细节堆回主上下文；
- **Persistent memory**：把长期状态读写到 files，而不是完全依赖 context window；
- **Context engineering**：明确规定何时 delegate、何时切换策略、哪些信息写入 persistent memory。

![更复杂的 agent scaffold：planning、delegation、memory](assets/coverage/3148-40-agent-scaffolds.jpg)

所以一个关键结论是：

> **Evaluating agents = evaluating the language model + evaluating the scaffold.**

如果榜单没有把 scaffold、tool access、budget、context policy 等规则说清楚，就很难把分数解释成“模型本身的能力”。

---

## 6. Pure reasoning：能否把推理从知识中分离？（约 53:56–60:12）

前面的 benchmark 都不同程度依赖 language / world knowledge。ARC-AGI 这条路线试图测更接近 **fluid intelligence / reasoning** 的东西：

- 任务目标是人类可以解决，但 AI 很难；
- 每个 task 尽量独特，降低直接 memorization 的帮助；
- 输入主要是抽象 grid / spatial patterns，而不是知识问答。

### ARC-AGI-1 → ARC-AGI-2 → ARC-AGI-3

- **ARC-AGI-1（2019）**：经典 grid transformation；
- **ARC-AGI-2（2025）**：更强调 multi-step reasoning；
- **ARC-AGI-3（2026-03）**：进一步变成 interactive environments，模型要通过交互发现规则。

![ARC-AGI-2：更复杂的多步空间推理](assets/coverage/3350-43-arc-agi-2.jpg)

课堂的历史图强调：纯 pretrained LMs 早期在 ARC 上几乎没有推动进展，reasoning models 出现后曲线才明显上升。

![ARC-AGI 历代模型结果变化](assets/coverage/3388-44-arc-results.jpg)

ARC-AGI-3 进一步加入类似小游戏的交互环境：

![ARC-AGI-3 的交互式环境](assets/coverage/3475-45-arc-agi-3.jpg)

这里仍有两个限制：

1. “reasoning 与 knowledge 完全解耦”本身很难做到；任何 task family 都可能隐含 prior。
2. ARC 明确追求人类可解，因此主要测试 **human-level reasoning**，并不直接覆盖 superhuman theorem proving、开放数学研究等能力。

对于图形输入，语言模型可以接收 image，也可以使用 ASCII / textual representation；无论哪种方式，本质上都要求处理非自然语言的 spatial structure。

---

## 7. Safety benchmarks：安全不是一个单一标量（约 60:12–65:09）

课堂先类比汽车 crash test：汽车行业经过长期标准化后，“安全”至少有相对明确的测试流程。AI 则还远没有这么统一。

### 7.1 HarmBench

HarmBench 基于约 **510 个 harmful behaviors**，核心问题是：当用户请求违法或有害行为时，模型能否适当拒绝。

### 7.2 AIR-Bench：从政策与法规构造风险 taxonomy

AIR-Bench 试图覆盖更广的安全定义：从不同地区监管框架与公司 policy 中构建 taxonomy。官方 lecture material 给出 **314 个 risk categories、5694 个 prompts**。

![AIR-Bench 的分层风险 taxonomy](assets/coverage/3720-49-air-bench.jpg)

### 7.3 Jailbreaking 与 GCG

即使模型训练成会拒绝 harmful instructions，也可能被 adversarial prompt 绕过。GCG（Greedy Coordinate Gradient）通过自动优化 token suffix 找到 jailbreak prompt；早期研究还发现，在 open-weight models 上优化得到的攻击可以 transfer 到 closed models。

![GCG jailbreak 的 transfer 示例](assets/coverage/3785-50-gcg-jailbreak.jpg)

### 7.4 “安全”高度依赖 context

AI safety 之所以难统一，是因为它同时涉及：

- politics、law、social norms，而且不同国家不同；
- hallucination，在医疗、法律、金融环境下尤其危险；
- sycophancy；
- abetting crime；
- inequality；
- 长期依赖 AI 后可能造成的 critical-thinking loss。

这些风险甚至可能与“capability”方向不同：更强的事实能力可能降低 hallucination，却也可能提升某些 dual-use 能力。

Cybersecurity 是最直观的双刃剑：同一个高能力 agent 可以被用于攻击，也可以做 penetration testing 与防御。

---

## 8. Realism / ecological validity：benchmark 像不像真实世界？（约 65:09–68:43）

**Ecological validity** 关注：evaluation 在多大程度上代表实际使用。

- Exam benchmark：控制严格，但通常远离真实用户任务；
- Chatbot Arena：prompt 来自真人，但用户与 use-case 分布不可控；
- 更现实的 benchmark：尝试直接从职业、临床或真实产品使用场景定义任务。

### 8.1 GDPVal

GDPVal 从美国 GDP 占比最高的 9 个 sectors 中选取 44 个 occupations，让实际 professionals 构造任务；课堂材料给出的平均工作经验约 14 年。

![GDPVal：从真实职业任务中构造 evaluation](assets/coverage/3985-53-gdpval.jpg)

### 8.2 MedHELM

传统 medical benchmark 往往是标准化考试，但“通过医学院考试”显然不等于“可以直接临床工作”。MedHELM 因此收集更接近真实临床工作的任务：官方 lecture material 给出 **121 个 clinical tasks、29 位 clinicians**，并混合 private 与 public datasets。

![MedHELM：从临床工作流而非考试题定义任务](assets/coverage/4038-54-medhelm.jpg)

### 8.3 Clio：真实 user data 与 privacy 的张力

最清楚用户实际如何使用模型的是 model provider。Clio 的思路是让 language models 分析真实 user data，然后只输出 aggregate patterns，而不是直接暴露个人对话。

![Clio：对真实使用数据做聚合分析](assets/coverage/4078-55-clio.jpg)

这里出现一个结构性 trade-off：

> **越接近真实 query stream，evaluation 越有现实性；但越接近真实用户数据，privacy 风险越高。**

---

## 9. Validity：分数真的测到了我们想测的东西吗？（约 68:43–75:41）

### 9.1 Train–test overlap / contamination

传统 supervised learning 的规则很清楚：train split 与 test split 分开。Foundation models 改变了这一点：模型训练在大规模 Internet data 上，而外部 evaluator 往往不知道训练语料具体包含什么。

课堂给出四条应对路线。

#### Route 1：从模型行为推断 overlap

一种方法利用 benchmark items 的 **exchangeability**：如果题目顺序本应随机，而模型却对某个“原始 benchmark 顺序”表现出异常偏好，这可以作为见过原数据的统计信号。

![利用 exchangeability 检测可能的 benchmark contamination](assets/coverage/4205-57-contamination-exchangeability.jpg)

#### Route 2：建立 reporting norms

像统计论文默认报告 confidence intervals 一样，模型提供者如果声称某个 GPQA / MMLU 分数，也应报告或解释 train–test overlap 风险。这里评价的不只是模型，也包括 provider 的透明度与 evidence quality。

#### Route 3：使用 fresh evals

LiveCodeBench、UncheatableEval 一类方法从模型 training cutoff 之后的新网页、GitHub、论文等来源构造任务。

但 timestamp 不是绝对保证：一个刚发布的 repo 也可能复制自更早、已进入训练集的代码。

#### Route 4：使用 private evals

公司可以用内部 codebase；个人也可以用从未公开的 private writings。只要能确信数据没进入训练，这类 evaluation 的 contamination 风险很低。

Perplexity 特别适合 private eval：只需要一批可靠 held-out text，就可以直接计算 log probabilities。

### 9.2 Dataset quality：benchmark 本身可能坏掉

SWE-Bench 后来需要 SWE-Bench Verified，就是因为原任务、测试与环境存在不够严谨的部分。类似问题也出现在许多 benchmark 中：

- 问题缺少必要信息；
- 题目本身有歧义；
- ground truth 错；
- agentic task 的 tests 不完整；
- 一个“错误实现”可能意外通过所有 tests。

课堂展示的审计例子包括“题目问曲线但根本没有 curve”“问 baby 是否穿 socks 但图片看不出来”等。

![Benchmark audit：题目缺信息、歧义与错误答案都会污染分数](assets/coverage/4445-61-benchmark-audit-examples.jpg)

Agentic benchmarks 更难，因为不仅有 question/answer，还包含整个 environment。课堂举出一个极端例子：某个 TorchBench setting 下，agent **输出空响应也能得到 38%**。这说明通过 test cases 并不自动等于完成真实任务。

### 9.3 Docent：用 qualitative trace inspection 补足分数

Docent 使用 LLM 去检查 agent traces，帮助发现 benchmark task、environment 或 scoring 的异常。

![Docent：把 agent trace inspection 加回 evaluation loop](assets/coverage/4515-62-docent.jpg)

这引出 Percy 很实用的一条建议：

> 不管你在设计 benchmark，还是只是在 benchmark 上跑模型，都要实际看 outputs / traces，确认“高分”真的是你以为的那种成功。

Evaluation 不能只剩下一列数字。

---

## 10. Evaluation 到底为谁服务？我们到底在评估什么？（约 75:41–77:37）

### 10.1 先说明目的

没有 “one evaluation to rule them all”。至少可以区分四种目标：

1. **购买/部署决策**：用户或公司要在 model A / B 中选一个，针对具体 use case；
2. **能力研究**：研究者希望测量某种 raw capability，例如 reasoning / intelligence；
3. **Benefits + harms**：为业务或 policy 理解模型带来的收益与风险；
4. **开发反馈**：模型开发者需要定位弱点、指导下一轮训练。

目标不同，就应该选不同的 dataset、judge、metric 与 environment。

### 10.2 Methods vs. models/systems

Foundation model 时代之前，很多研究其实在评估 **method**：所有人使用相同 train/test split，只改变 algorithm，然后比较结果。

今天的排行榜更多是在评估 **model/system**：训练数据、compute、post-training、tools、scaffold 等都可以不同，最终只看成品系统有多好。这对下游用户很有价值，但不利于隔离“某个算法改进到底贡献了多少”。

课堂给出的例外是 **NanoGPT speedrun**：固定 data 与目标 validation loss，比的是达到目标需要多长 compute time，因此更接近 method / training-efficiency evaluation。

![NanoGPT speedrun：固定目标后比较训练方法的速度](assets/coverage/4632-65-nanogpt-speedrun.jpg)

因此必须先说清楚 **rules of the game**：

- 是在比较 methods，还是 finished models？
- 如果是 agents，tool access、scaffold、budget 是否相同？
- 是否允许 test-time compute、retrieval、browser、code execution？
- judge 是 human 还是 LLM？
- 数据是否可能 contaminated？

如果规则不清楚，单个分数几乎无法解释。

---

## 11. 总结：没有唯一正确的 evaluation（约 77:37–78:27）

![Lecture 12 最终 takeaways](assets/coverage/4665-66-takeaways.jpg)

本讲最终可以压缩成三句话：

1. **There is no one true evaluation.** 先说清楚你要测什么、为什么测。
2. **Define the rules of the game.** Methods、models、systems、agents 混在一个榜单里时，必须明确可用资源与评估契约。
3. **同时检查 difficulty、realism、validity。** 三者通常存在 trade-off：越困难未必越真实，越真实越可能碰到 privacy / reproducibility 问题，越公开越容易 contamination。

可以把本讲的 benchmark families 看成一条从“模型内部概率”逐步走向“真实世界行为”的轴：

| 评估族 | 主要测量对象 | 优点 | 典型风险 |
|---|---|---|---|
| Perplexity | token probability distribution | 连续、平滑、适合训练与 scaling | 可能测到大量无关 token；提交概率需要可信 |
| Exam benchmarks | knowledge / reasoning on controlled questions | 难度可控、易评分 | saturation、contamination、realism 较低 |
| Chat benchmarks | open-ended response quality | 更接近实际 assistant 使用 | judge bias、style/correctness 混合、rubric 不清 |
| Agentic benchmarks | task completion in environments | 能测 tool use 与长期执行 | scaffold 强烈影响结果；tests/environment 可有漏洞 |
| Pure reasoning | knowledge-light abstract reasoning | 暴露 memorization 之外的弱点 | 很难真正与 prior knowledge 解耦 |
| Safety benchmarks | harmful behavior / policy risk | 直接面向风险 | 安全定义高度 contextual，且存在 dual-use |

最终，evaluation 不是训练完成后的附属步骤。它决定团队优化什么、相信什么，也决定“进步”这个词在一个研究项目里究竟指什么。

## 复习检查

读完本讲，应该能回答：

- 为什么 perplexity 对 pretraining 很自然，但不能取代 downstream evaluation？
- 为什么 MMLU → MMLU-Pro → GPQA → HLE 会形成不断加难的序列？
- 为什么 pairwise preference 往往比绝对评分更稳定？
- 为什么 LLM-as-a-judge 需要关注 length bias、judge bias 与 rubric？
- 为什么 agent benchmark 实际评估的是 **LM + scaffold**？
- ARC-AGI 想从 knowledge 中剥离什么，它又为什么不可能完全“纯净”？
- Safety benchmark 为什么很难压成一个统一指标？
- Ecological validity 与 privacy 为什么会发生冲突？
- Foundation-model 时代的 train–test contamination 为什么比传统 ML 更难处理？
- 为什么 benchmark 的 dataset / environment 本身也必须被 audit？
- 在比较两个模型之前，如何先定义清楚 evaluation 的目的与 rules of the game？
