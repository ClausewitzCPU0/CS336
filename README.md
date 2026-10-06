# CS336: Language Modeling from Scratch — Spring 2026

本仓库用于自学 Stanford CS336: **Language Modeling from Scratch (Spring 2026)**，整理课程视频、Lecture materials、作业与实验，并为每个 Lecture 生成一篇可长期阅读和检索的图文 Markdown 笔记。

课程重点是从较低层级理解和实现现代 language model 的核心组件，而不是只调用现成训练框架。学习过程以课程视频和同学期官方资料为主，并结合代码实验验证实现细节。

## 课程信息

- 课程：CS336: Language Modeling from Scratch
- 学期：Spring 2026
- 主要授课教师：Tatsunori Hashimoto、Percy Liang
- 官方课程网站：https://cs336.stanford.edu/
- Spring 2026 Lecture materials：https://github.com/stanford-cs336/lectures
- 官方 GitHub organization：https://github.com/stanford-cs336
- 官方课程视频列表：https://www.youtube.com/watch?v=JuoVZkPBiKk&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV

本地课程视频来自：

- Bilibili 合集：https://www.bilibili.com/video/BV1msTD6CE6j

Bilibili 仅作为本地视频来源。Lecture 编号、标题、授课教师或 guest speaker、课件和关联作业等信息，以 Spring 2026 官方资料与实际视频内容的交叉核验结果为准。

## 仓库内容

```
.
├── README.md
├── notes/          # 每个 Lecture 的图文 Markdown 笔记
├── assignments/    # 作业、实现记录和解题笔记
├── experiments/    # 课程相关代码实验
├── docs/           # 环境、工具和补充说明
└── resources/      # 课程资料索引或其他辅助文件
```

目录可随实际学习进度扩展；课程原始视频等大文件不建议直接提交到 Git 仓库。

## 笔记组织

本仓库采用：

```
一个教学视频 / Lecture
→ 一篇最终图文 Markdown 笔记
```

每篇笔记尽量同时覆盖：

- 视频中的主要讲解内容；
- 对应 Spring 2026 Lecture materials；
- 重要公式、算法、代码和实现细节；
- 对理解有帮助的板书、图示或关键视频截图；
- 与该 Lecture 明确关联的 assignment、reading 或实验资料。

笔记以实际 Lecture 内容为主，不为了覆盖课程网站上的所有材料而强行加入与本讲关系不明确的内容。

## 图文笔记生成

图文笔记使用 `ChatGPT-Web-Course-Note-Workflow-v4.9.4` 生成。

Google Drive 中的课程文件目录：

```
cs_courses\AI\CS336\
```

workflow 目录：

```
note-gen-workflow/
```

每次 ChatGPT Web 会话只处理一个本地章节，prompt 顶部只需要修改：

```
TARGET_LOCAL_CHAPTER: P01
```

下一讲依次改为：

```
TARGET_LOCAL_CHAPTER: P02
```

然后继续 `P03`、`P04` 等。

`Pxx` 是本地视频编号，只用于输入发现和初始 mapping hint。不要预先假定：

```
Pxx == official Lecture xx
```

每次生成前都通过 workflow 的 Chapter Identity Preflight，结合本地视频、字幕、Spring 2026 官方 Schedule、Lecture materials 和官方录像实际建立：

```
local chapter -> official Lecture
```

因此即使官方公开视频列表存在缺失、编号 gap 或与本地合集顺序不完全一致，也不需要修改整套笔记组织方式。

## 资料使用原则

资料优先级：

1. 本地实际视频和字幕，用于确认本讲真正讲授的内容；
2. Spring 2026 官方课程网站和 Schedule；
3. Spring 2026 `stanford-cs336/lectures` Lecture materials；
4. Spring 2026 Stanford Online 等官方录像；
5. `stanford-cs336` 下的其他同版本官方资料；
6. 其他学期的官方 CS336 资料，仅在 Spring 2026 资料不足时作为补充。

课程网站中的链接不保证始终可访问，也不能仅凭链接所在页面判断资料一定与 Spring 2026 视频完全对应。发现链接失效、资料缺失或版本不一致时，应优先寻找同一学期的 Stanford first-party source。

如果只能使用 Spring 2025、Spring 2024 等其他学期资料，需要明确区分版本，不把其他学期新增、删除或修改的内容写成 Spring 2026 视频中明确讲授的内容。

## 学习方式

推荐每个 Lecture 按以下顺序学习：

1. 观看课程视频，先建立整体理解；
2. 阅读生成后的图文笔记并核对关键推导；
3. 回看官方 Lecture materials；
4. 完成对应 assignment 或自己实现关键组件；
5. 将实验结果、错误分析和补充理解记录到仓库中。

对于实现型内容，优先自己写出可运行的最小版本，再与课程参考实现或成熟库进行比较。笔记用于组织和复习知识，代码实验用于验证自己是否真正理解了实现细节。

## 目标

完成课程后，本仓库应形成一套围绕 CS336 Spring 2026 的学习记录，包括：

- 按 Lecture 组织的完整图文笔记；
- 课程作业和关键实现；
- 可复现实验及结果；
- 对 language modeling 训练流程、模型实现和相关系统问题的理解记录。

最终目标是能够从实现角度理解一套 language model 训练系统，而不仅停留在 API 使用层面。
