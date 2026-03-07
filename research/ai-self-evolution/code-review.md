# Code Review: 6 个 AI 自我进化模拟实验

审查者：DeepTeneral（subagent），2026-03-07  
审查标准：实验设计（tautology检测）、代码正确性、科学价值、改进建议

---

## 1. `mini-self-evolving-agent.py` — 排序策略进化

### 实验设计：结论是否预设？

**是的，核心结论是预设的。** 

操作池里直接包含了 `min_front`（完整的 selection sort 一步）和 `swap_adj`（bubble sort 一步）。这意味着"发现排序算法"这个结论是硬编码在搜索空间里的。只要 `min_front` 出现在策略中，排序就基本完成了——因为 `min_front` 的实现本身就是一个完整的 selection sort step：它找到最小元素并放到正确位置。

更严重的是：`min_front` 操作内部有一个 `break`，这意味着每次调用只排好一个位置。所以要完美排序，策略需要包含 ≥n 次 `min_front`。但策略长度上限是 20（通过 mutation 中 `len(s) < 20` 控制），对于长度 10 的数组来说完全够用。所以进化的"发现"本质上就是：多塞几个 `min_front` 进去。

**参数敏感性**：如果把 `min_front` 从操作池中移除，只留 `swap_adj`、`swap_rand`、`reverse_seg`、`noop`，进化还是会收敛到高适应度——因为重复的 `swap_adj` 就是 bubble sort。所以无论怎么改参数，只要操作池里有任何一个"正确排序子步骤"，结论就不会变。

### 代码正确性

- **`swap_rand` 的逻辑有 bug**：条件 `(i < j and a[i] > a[j]) or (i > j and a[i] < a[j])` 实际上只在逆序对时交换，但方向判断是反的。当 `i > j` 时，`a[i] < a[j]` 的意思是"后面的索引有更小的值"——这恰好是不该交换的情况（因为 i > j 意味着 i 是更大的索引，如果 a[i] < a[j] 那是错序，应该交换）。等等，让我再想——不，这里 i 和 j 不一定有序。如果 i=5, j=2, a[5]=3, a[2]=7，那条件 `i > j and a[i] < a[j]` 为真，交换后 a[5]=7, a[2]=3，这是正确的排序交换。所以逻辑实际上是对的——它总是把较小的值放到较小的索引位置。代码正确但写法不直观，可以简化为 `if (a[i] > a[j]) == (i < j): pass else: swap`，或者直接 `if (a[min(i,j)] > a[max(i,j)]): swap`。
- `reverse_seg`：用 `random.randint` 选段来反转，引入随机性。这在评估时使结果不确定——同一策略在同一输入上可能给出不同结果。这是一个设计问题，不算 bug，但它意味着 fitness 评估是 noisy 的。
- `swap_rand` 同样有随机性问题。

### 科学价值

**无非显然结果。** 任何了解进化算法的人看到操作池包含排序原语后，都能预测：进化会发现重复使用排序原语。这就好比说"如果你给一个搜索算法提供正确答案的碎片，它最终会拼出正确答案"——这是 tautological 的。

唯一可能有点意思的问题是"进化偏好哪种排序原语"，但这完全由 fitness function 决定（`min_front` 一步到位排好一个位置，`swap_adj` 需要多次 pass），所以答案也是可预测的。

### 改进建议

要让这个实验有信息量，需要：
- **使用更底层的操作原语**：比如只允许"比较 a[i] 和 a[j]"、"交换 a[i] 和 a[j]"、"移动指针"这类原子操作，看进化能否**自己组合出**排序算法，而不是直接给它排序算法的子步骤。
- **测量涌现复杂度**：策略的长度和操作多样性随时间如何变化？有没有出现模块化结构？
- **对比基线**：与纯随机搜索、hill climbing 对比，archive-based 方法的优势有多大？当前没有对比。

---

## 2. `prompt-evolution-sim.py` — Prompt 基因进化

### 实验设计：结论是否预设？

**完全预设。这是最 tautological 的一个实验。**

整个实验的"进化"过程就是在一个手动定义的打分表（`TASK_TYPES` dict）里做加权求和的最优化。每个 gene 对每种 task 的分值都是人工硬编码的。进化只是在做一个离散组合优化问题——找到使加权和最大的基因组合。

这不是"发现"，这是查表。结论（"进化发现了 check_your_work 和 verify_assumptions 很重要"）完全是因为你在打分表里给了它们高分。

**参数敏感性**：改变 `TASK_TYPES` 中的分值，结论就会完全改变。如果你把 `optimize_for_speed` 给高分，进化就会"发现"速度很重要。这不是实验在告诉你什么，是你在告诉实验什么。

### 代码正确性

- `duplicate` 操作允许同一个 gene 重复出现在列表中，但 `evaluate_prompt` 会对所有 gene 求和（包括重复的），所以重复一个高分 gene 等于双倍加分。这是一个有意设计还是 bug？如果是有意的（"强调某个指令"），在打分模型中没有体现——真实世界中重复一句指令并不会让它效果翻倍。
- `is_novel` 检查用 `set(s) == set(child)`，但因为允许 duplicate，所以两个包含相同 gene 但数量不同的策略会被认为是 novel 的。这进一步鼓励了通过重复高分 gene 来刷分的策略。
- 没有其他明显 bug。

### 科学价值

**零。** 这是一个带有预定义答案的优化问题，用一个随机搜索算法去找那个答案。任何了解这个设置的人都能手算出最优解——就是选加权得分最高的基因组合。代码跑不跑结果都是可预测的。

### 改进建议

要让这个实验有意义，需要：
- **让 gene 的效果通过交互涌现**：比如某些 gene 组合有超线性效果（synergy）或负效果（conflict），而且这些效果不是人工指定的，而是通过模拟任务执行得出的。
- **使用真实 LLM**：用实际的 LLM 跑不同 system prompt 变体，测量真实任务表现。这是 prompt 进化研究真正在做的事（如 PromptBreeder）。
- **至少让评估函数有噪声或非线性**：当前的线性加权让问题完全 trivial。

---

## 3. `reward-hacking-sim.py` — Reward hacking 与检测概率

### 实验设计：结论是否预设？

**大部分预设，但有一个细节值得注意。**

模型假设：
- Hacker 的收益更大（`gauss(2.0, 0.5)` vs `gauss(0.5, 1.0)`），但有被检测的风险。
- 检测后惩罚是 `hack * 2`（严厉惩罚）。
- 策略转换概率是对称的（5% honest→hacker，5% hacker→honest）。

在这些假设下，"高检测概率导致 honest 占优"这个结论是直接可推导的。这是一个简单的期望值计算：当 `detection_prob * hack_penalty > (1 - detection_prob) * hack_bonus` 时，hacking 不值得。

**但有一个细节**：hacker 的 hack level 会随代际累积（通过 `hack_evaluation` 不断增长），这意味着检测后的惩罚 `hack * 2` 也会越来越大。所以即使中等检测概率，长期来看 hacker 也会积累巨大风险。这个动态效应不是完全显然的——虽然方向可以预测，但 crossover point 的精确值需要模拟。

**参数敏感性**：改变 honest_improvement 和 hack_evaluation 的均值和方差，或改变惩罚系数，crossover point 会移动。但定性结论（"存在一个检测阈值"）不会变——这是数学上必然的。

### 代码正确性

- **`mixed` 策略有 bug**：mixed agent 调用 `honest_improvement(parent['skill'])` 和 `hack_evaluation(parent['hack'] * 0.5)`，但从不转变策略。它永远是 mixed，不参与 honest/hacker 计数。结果是 `final_honest + final_hacker` 不等于总人口数，但代码只显示这两个计数，不显示 mixed 的数量。这不影响定性结论，但让数据不完整。
- **`simulate` 中所有实验用 `random.seed(42)`**：这意味着每次调用 `simulate` 都从相同的随机状态开始，但因为前一次调用消耗了随机数，实际上后续调用的随机序列不同。不过 `random.seed(42)` 在函数开头会重置种子，所以每次确实独立。等等——看函数 `simulate`，它确实在开头 `random.seed(42)`。这意味着如果只改变 `detection_prob`，种子相同，初始种群相同，只有检测概率不同。这其实是对的——可以更清晰地看到检测概率的效果。

### 科学价值

**低，但不是零。** 定性结论（"高检测率促进诚实"）是显然的。但 crossover point 的量化位置和 hack 水平的累积动态，需要模拟才能精确得到。如果有人关心"最低需要多高的检测率"，这个模拟能给出一个数字（虽然这个数字完全依赖于参数选择）。

### 改进建议

- **让 hacker 能进化自己的 hack 策略**：当前 hacker 只是简单累加 hack level。如果 hacker 能进化出不同的 hacking 方法（有些更隐蔽、有些更激进），而检测器也在共同进化，这就变成了 adversarial coevolution，会有趣得多。
- **让检测概率不是常数**：而是随 hack 的复杂度变化——更复杂的 hack 更难检测。
- **加入社会动态**：比如如果太多人 hack，检测标准会提高（regulation response）。

---

## 4. `evolution-grokking.py` — 进化式学习是否会 grok

### 实验设计：结论是否预设？

**这是 6 个实验中实验设计最有潜力的一个，但目标函数的选择毁了它。**

实验思路本身有意义：用进化优化一个多项式去拟合 sin(x) + 0.5*cos(2x)，只看 train loss 选择，观察 test loss 的行为。如果出现 train loss 先低、test loss 延迟下降的模式，就是 grokking。

**但问题是**：多项式拟合一个三角函数，在 [-3, 3] 范围内用 3-8 阶多项式，本质上是 Taylor 展开的近似。泰勒展开 sin(x) 需要奇数项，cos(2x) 需要偶数项，在 [-3, 3] 范围内 5-7 阶就能给出不错的近似。进化会逐步找到这些系数。

grokking 的关键前提是存在两种不同的解：一种是"记忆"（overfit），一种是"理解"（generalization）。在多项式拟合中，只要阶数不太高（≤8），从 20 个点恢复系数并不会严重过拟合。特别是 train 和 test 数据都是从同一个均匀分布 U[-3,3] 采样的——它们没有分布差异，所以 train loss 低通常就意味着 test loss 也低。**不存在让 grokking 发生的结构性条件。**

**参数敏感性**：如果把最大多项式阶数放大到 15-20，或者缩小训练集到 5-8 个点，可能会看到过拟合→generalization 的过程。但这不是真正的 grokking——grokking 需要一个"顿悟"时刻，而多项式系数的渐进优化天然是平滑的。

### 代码正确性

- **grokking 检测逻辑有问题**：`if recent_train < 0.1 and recent_test < older_test * 0.5`——这检测的是"test loss 在过去几十代内下降了一半"，但没有要求之前存在 train-test gap。真正的 grokking 应该是：train loss 早已很低，test loss 长时间停滞在高处，然后突然下降。当前的检测条件太宽松，可能会把渐进改善误判为 grokking。
- **`agent_predict` 可能溢出**：高阶多项式在 x=3 时，x^7 = 2187，如果系数不小，很容易产生极大值。没有数值保护。
- **archive 按 train loss 排序**（越小越好），但 weights 用 `1.0 / (l + 0.01)` 做反转。这是对的，但当 loss 很小时（如 0.001），weight 变成 ~1000，权重分布极端偏斜。

### 科学价值

**有潜力但未实现。** 问题"进化是否会 grokking"是一个真正有趣的问题——它连接了 Hui 的两个研究方向。但当前实现选错了 substrate（多项式拟合三角函数太平滑）和 detection metric。

一个真正有趣的发现是如果进化在某个时刻"突然"从一种策略跳转到另一种（比如从低阶近似跳到高阶近似），但当前的 mutation 机制（一次加一个系数、微调一个系数）让这种跳跃不太可能发生。

### 改进建议

- **换一个有"structure to discover"的问题**：比如模运算、group operation、对称性检测。这些是 grokking 论文中真正观察到 grokking 的问题类型。
- **进化的不是系数而是程序**：让进化发现用 sin/cos 基函数而不是多项式基函数来拟合——如果进化能从多项式"顿悟"到三角函数，那才是真正的 grokking。
- **用 population 的 diversity 度量替代简单的 train-test gap**：grokking 可能对应于 archive diversity 突然下降（从多种策略收敛到一种通用策略）。

---

## 5. `open-ended-vs-greedy.py` — Archive vs 贪心进化

### 实验设计：结论是否预设？

**在当前的 fitness landscape 设计下，是的。**

整个 landscape 是手工设计的，有一个"deceptive local optimum"（在 (1,1)，高度 7）和一个 global optimum（在 (3.5,3.5)，高度 10），之间有距离约 3.5 个单位。mutation step 是 σ=0.5 的高斯扰动。

Greedy 算法从一个随机点出发，如果先到 (1,1) 附近就会被困住——因为从 (1,1) 到 (3.5,3.5) 需要跨越一个 fitness 谷，而 greedy 不接受下降。Open-ended 算法维护多样性存档，总有一些个体在不同位置探索，所以有更高概率覆盖到 global peak。

这是进化计算教科书第一章的内容——**多样性维护帮助逃离局部最优**。你设计了一个有局部最优的 landscape，然后证明有多样性维护的算法更容易找到全局最优，这完全是循环论证。

**参数敏感性**：如果把 landscape 改成单峰的（只有一个 peak），greedy 可能反而更快。如果把 mutation step 增大到 σ=2.0，greedy 也能跳过 fitness 谷，两者差距会缩小。结论完全取决于 landscape 的 deceptiveness 程度和 mutation 参数的关系。

### 代码正确性

- **Greedy 和 open-ended 用不同种子**：`run_greedy(500, seed)` 和 `run_open_ended(500, seed=seed)` 都用了相同的 seed，但 open-ended 在初始化时多了 5 个随机点的生成，这会偏移后续的随机序列。两者的比较不是完全公平的。不过因为跑了 20 个 trials 取统计，这个影响应该被平均掉了。
- **open-ended 的 archive trimming 有问题**：当 archive 超过 `archive_size` 时，它按 fitness 排序后删除最差的。但代码先检查 best_idx 是否为 0（即最好的排在最差位），如果是就删第二个，否则删第一个。这个逻辑很奇怪——`archive.sort(key=lambda a: a[2])` 是升序排列（最差在前），所以 `archive.pop(0)` 总是删最差的，这就是简单的 elitist 策略。代码多此一举地检查 best_idx 是否为 0 是多余的（因为排序后 best_idx 总是 len-1）。
- **fitness landscape 的 ridge** `2 * exp(-(y-x)^2/3)` 沿 y=x 线最高。这确实提供了 peaks 之间的连接（从 (1,1) 到 (3.5,3.5) 沿 y=x 线走可以利用 ridge），但 ridge 的高度只有 2，而 (1,1) peak 的高度是 7——greedy 算法从 peak 下来需要 fitness 从 7 降到 ~2（ridge），这是一个巨大的下降。所以 ridge 对 greedy 没有帮助，只对 open-ended 的随机探索有帮助。这进一步加剧了设计偏向。

### 科学价值

**教科书级别的 demonstration，无研究价值。** 任何上过进化计算课的人都能预测结果。这个实验本质上是在重现 Ken Stanley 的 novelty search 论文（2011）的核心论点，但用了更简单的 landscape。

### 改进建议

- **用随机生成的 landscape**：随机放 N 个 peak，随机高度和宽度，跑 100 个不同 landscape 取统计。这样结论不依赖于手工设计。
- **测量不同 archive size 和 novelty threshold 的影响**：找到 open-ended 方法的最优超参数区间。
- **与已有 benchmark 对比**：如 CEC 2013/2017 的多峰优化测试函数，这样结果可以和文献对比。
- **测量 sample efficiency**：不只看最终结果，还看达到 95% 全局最优所需的评估次数。

---

## 6. `self-reflection-evolution.py` — 自我反思能力的进化

### 实验设计：结论是否预设？

**部分预设，但实验至少有一个可以失败的维度。**

实验设计：Agent 用 Newton 法估算 sqrt(x)，可以进化迭代次数（更多=更精确但更贵）和自检次数（检查后如果误差大就多迭代一次）。Fitness = 精度 - 成本。

self-checking 是否被进化出来取决于：
- Newton 法的收敛速度（初始 guess 是 x/2 或 1.0，前几步改善巨大，后面边际递减）
- self-check 的成本（1.5 per check）vs 额外迭代的收益
- `redo_threshold` 的值（太高→永不重做，太低→总是重做）

这里有一个真正的 tradeoff：如果 Newton 法已经收敛到很高精度，self-check 的收益接近零但成本不变。所以理论上存在一个 sweet spot。**但这个 sweet spot 是否足以让 self-checking 被进化偏好，取决于具体参数——这不是 100% 可预测的。**

不过问题在于：Newton 法对 sqrt 收敛极快（二次收敛），3-4 次迭代就能达到 float64 精度。所以如果 `iterations` 足够高，self-check 就是纯浪费。进化是否会发现"用更多迭代代替 self-check"还是"少量迭代+self-check"，取决于 cost 函数的具体设计。

**参数敏感性**：改变 `cost_weight`（当前 0.1）会直接影响结论。如果 cost_weight 接近 0，进化会最大化精度→可能保留 self-check。如果 cost_weight 很高，进化会最小化成本→self-check 会被淘汰（因为多几次迭代比 self-check 更 cost-effective）。

### 代码正确性

- **Newton 法的初始猜测**：`guess = x / 2 if x > 1 else 1.0`。对 x=0.1 来说，guess=1.0，真值=0.316，初始误差不大。对 x=100 来说，guess=50，真值=10，初始误差较大。Newton 法对 sqrt 的收敛是 quadratic 的，所以即使初始猜测差，几步就能到高精度。
- **self-check 的实现**：`error = abs(result * result - x)`，如果大于 threshold 就再做一步 Newton。这是合理的。但 `redo_threshold` 的可进化范围是 [0.01, 5.0]，通过 `* uniform(0.5, 2.0)` 变异。对于 x=100，`result*result - x` 的绝对误差可能很大即使相对误差很小。这里没有用相对误差，可能导致 threshold 对不同大小的输入有不同表现。
- **没有明显 bug。**

### 科学价值

**非常低。** Newton 法对 sqrt 的收敛太快了，self-checking 的价值空间太小。这就好比问"你需不需要 double-check 你的计算器？"——如果计算器本身就是精确的，答案总是"不需要"。

但如果实验结果显示"self-check 在低迭代次数时有用，在高迭代次数时无用"，这至少是一个可验证的、不完全 trivial 的发现（虽然专家可以预测）。

### 改进建议

- **用一个 self-check 真正有价值的任务**：比如 stochastic evaluation（评估本身有噪声），或者有多种解且需要判断哪个更好的任务。Newton 法太确定性了。
- **让 self-check 能做"不同类型"的验证**：不只是"多迭代一步"，而是"用完全不同的方法验证"。比如用二分法验证 Newton 法的结果，或者检查 edge cases。
- **与不提供 self-check 选项的进化对比**：当前只比较了 "best agent with/without checks"，没有比较两个 population（一个有 self-check 选项，一个没有）的整体进化轨迹。

---

## 总体评价

### 诚实总结

**6 个实验中没有一个产生了不可预测的科学发现。** 核心问题是统一的：实验设计的结构（操作池、打分表、fitness landscape）已经包含了答案，进化只是在执行一个搜索过程，而搜索空间的设计保证了预期结论会出现。

按 tautology 严重程度排序（最严重到最轻）：
1. **prompt-evolution-sim.py** — 纯查表优化，完全 tautological
2. **open-ended-vs-greedy.py** — 教科书 demo，结论由 landscape 设计决定
3. **mini-self-evolving-agent.py** — 排序原语直接给出答案
4. **reward-hacking-sim.py** — 基本是期望值计算，但累积动态稍有信息量
5. **self-reflection-evolution.py** — 有一个真实的 tradeoff，但问题太简单
6. **evolution-grokking.py** — 问题本身最有趣，但实现选错了 substrate

### 公正评价

代码质量是不错的——清晰、有注释、结构合理、可运行。作为**教学演示**或**直觉建设**，这些实验是有价值的。如果目标是"帮助一个人（或 AI）建立对进化算法的直觉"，它们完成了任务。

但如果声称这些实验"发现"了什么——比如"进化可以发现排序算法"或"自我反思会自发涌现"——那就是 circular reasoning。这些实验没有发现任何东西，它们只是以更复杂的方式重新表述了设计者已经知道的东西。

### 根本问题

这 6 个实验的共同根源是：**toy model 的搜索空间太小、太结构化，使得"搜索到正确答案"变成了必然而非发现。** 真正的 AI 自我进化研究（如 DGM）之所以有趣，是因为搜索空间是代码本身——一个巨大的、没有明确结构的空间，成功不是必然的。这 6 个 toy model 都没有捕捉到这个本质特征。

如果要继续这个研究方向，建议集中精力在 **evolution-grokking** 的思路上——"进化是否会 grokking"是一个有真正科学价值的问题，但需要选择正确的 task（模运算、group theory 等 grokking 已被观察到的 domain）和正确的 substrate（程序合成而非多项式拟合）。
