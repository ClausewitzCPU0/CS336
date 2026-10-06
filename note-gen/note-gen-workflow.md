# ChatGPT Web Chat 图文笔记生成流程记录

### ChatGPT web + Google Drive 插件版本 prompt ，v4.9.4

将课程文件以及 workflow 文件先上传至 Google Drive, 然后通过 ChatGPT Web 的 Google Drive 插件访问.

需要使用 ChatGPT Web 中的 Google Drive 插件来访问谷歌云盘中的文件
注意ChatGPT Google Drive 插件的连接器单文件原始下载上限是 **256 MiB**, 因此要**分卷打包**：
"这里出现了一个明确的接口限制：Drive 连接器能定位到 667,666,512 字节的视频，但单文件原始下载上限是 256 MiB，因此这次直接获取 MP4 被拒绝。为了不在缺少视频核验的情况下违反 workflow，我先检查是否存在可用的流式/分段 Drive 下载能力；

本次使用的网页端模型为 GPT 5.6 Sol Extra High

ChatGPT web 端输入以下 prompt 即可:

> @google-drive
>
> 本次唯一需要修改的参数：
> TARGET_LOCAL_CHAPTER: P01
>
> 已经批准访问 Google Drive 中的文件，本次会话不需要让用户手动确认。
> 使用已上传的 ChatGPT-Web-Course-Note-Workflow-v4.9.4.zip 图文笔记生成 workflow，完成以下图文笔记生成任务。
> workflow 文件可以在 Google Drive 中的 `note-gen-workflow/` 路径下找到。
>
> # 第一部分：课程 Profile
>
> 课程名称：CS336: Language Modeling from Scratch
> 本次固定使用的课程版本：Spring 2026
> 主要授课教师：Tatsunori Hashimoto、Percy Liang
>
> 官方课程网站（Schedule、课程资料、实验作业等入口）：
> https://cs336.stanford.edu/
>
> Spring 2026 官方 Lecture materials repository：
> https://github.com/stanford-cs336/lectures
>
> 官方课程 GitHub organization（作业、课程网站和其他官方资料的 fallback 入口）：
> https://github.com/stanford-cs336
>
> Spring 2026 官方课程视频列表：
> https://www.youtube.com/watch?v=JuoVZkPBiKk&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV
>
> 本次使用的本地课程视频来自以下 Bilibili 合集：
> https://www.bilibili.com/video/BV1msTD6CE6j
>
> Bilibili 合集仅作为本地视频来源说明。本次目标视频版本固定为 Spring 2026。official Lecture number、official title、授课教师或 guest speaker、课件以及与本讲明确关联的实验/作业资料，应通过本地实际视频与 Spring 2026 官方 Schedule、官方 Lecture materials、Stanford Online 官方录像等同版本 first-party sources 交叉核验。
>
> 官方课程网站中的链接不保证始终可访问，也不得仅因为链接来自当前课程页面就假定目标资料一定与本次 Spring 2026 视频版本一致。使用资料前先核验其课程版本和 Lecture identity。若官网链接失效、资料缺失或发现版本不一致，可以联网搜索对应 Spring 2026 Lecture 的 Stanford 官方网站、`stanford-cs336` 官方 GitHub repositories、Stanford Online 官方录像等 first-party source。
>
> 优先使用 Spring 2026 同版本资料。只有无法获得足够的 Spring 2026 官方资料时，才使用 Spring 2025、Spring 2024 等其他学期官方资料作为补充；不得把其他学期新增、删除或修改的内容描述为本次 Spring 2026 视频中明确讲授的内容，并按照 workflow 保留 provenance。
>
> 本地连续 P 编号不得直接假定与 official Lecture number 一一相等。若官方 Schedule、官方公开视频和本地合集之间存在缺失 Lecture、编号 gap 或顺序差异，以 Chapter Identity Preflight 的实际核验结果建立 `local chapter -> official Lecture` 映射。
>
> 本课程文件可以在 Google Drive 中的以下路径找到：
> `cs_courses\AI\CS336\`
>
> # 第二部分：本次 Lecture 任务
>
> 按照 ChatGPT-Web-Course-Note-Workflow-v4.9.4，完成开头 `TARGET_LOCAL_CHAPTER` 所对应的单个 Lecture 图文 Markdown 笔记，并输出最终打包产物。
>
> 在第一部分给出的 Google Drive 课程目录中，根据 `TARGET_LOCAL_CHAPTER` 自动发现本讲所需输入。输入可能以直接文件、chapter ZIP 或 multipart ZIP 等形式存在；按照 workflow 的 required input discovery 规则处理视频、字幕、XML 以及其他本讲输入。
>
> 执行 workflow 完整的 Chapter Identity Preflight，核对本地视频、字幕及其他输入之间的章节一致性，并结合 Spring 2026 官方 Schedule、官方 Lecture materials 和官方录像建立 `local chapter -> official Lecture` 映射。
>
> official Lecture number、official title、授课教师或 guest speaker、对应日期以及本讲相关官方资料，均以本次实际核验结果为准。本地 P 编号只能作为 discovery / mapping hint，不得替代实际核验。
>
> 完成 identity 确认后，按照 workflow 执行 official material discovery。优先寻找与该 Lecture 明确对应的 Spring 2026 slides / executable lecture、assignment handout、code、reading 或其他官方资料；不存在明确 Lecture 关联关系的课程资料不要为了增加内容而强行并入正文。
>
> 如果官方链接无法访问，应主动搜索同一 Spring 2026 版本的 first-party alternative；如果只能找到其他学期资料，可作为补充来源，但必须保持正确 provenance。如果章节身份、输入配对、official mapping 或必要来源无法可靠确认，按照 workflow 的 fail-fast 规则停止，不要猜测并继续生成。
>
> 完成 required input discovery、Chapter Identity Preflight、official material discovery、source provenance、正文生成、视觉覆盖检查、reader-facing sanitation、final package checks 以及最终打包后，再向用户提供最终产物。