# chap02 §2.3.5 紧束缚模型扩展 Patch

**日期:** 2026-03-06
**目标:** 在 §2.3.5 "最小模型示例与高阶费米弧" 中补充紧束缚模型的显式 Hamiltonian、参数表和相图讨论
**源材料:** SM.tex §III (L600–L709)

---

## Patch 概述

在 chap02.tex 的 §2.3.5 中，当前仅有定性描述。需要在首段（模型介绍）和 figure 环境之间，插入两个模型的定量内容：
1. **相位双配分模型** (phase-bipartite model) — 用于说明 $\theta^{(4)}$ 的拓扑相变
2. **P4/nmm 模型** — 正文已提及的"最小紧束缚模型"的显式 Hamiltonian

---

## 插入位置

**文件:** `projects/Thesis/data/chap02.tex`

**插入锚点:** 在 L518（首段末尾）之后、L520（`\begin{figure}` 环境）之前

**上下文验证:**

```
L517: % 本小节用P4/nmm最小模型说明gNBP如何指示高阶费米弧
L518: 
L519: 下面以非共形空间群$P4/nmm$的最小紧束缚模型来具体展示广义嵌套Berry相如何指示高阶费米弧的存在。该模型描述了一个在$k_z$轴上具有一对狄拉克点的三维狄拉克半金属，如图\ref{fig:chap2_model_illustrate}所示。
L520:
L521: \begin{figure}[htbp]
```

**操作:** 将 L519 的单段文字**替换**为扩展内容（包含原始描述 + 两个模型的定量内容），新内容在 `\begin{figure}` 之前结束。

---

## 替换内容

**被替换的原文（L519，单行）:**
```
下面以非共形空间群$P4/nmm$的最小紧束缚模型来具体展示广义嵌套Berry相如何指示高阶费米弧的存在。该模型描述了一个在$k_z$轴上具有一对狄拉克点的三维狄拉克半金属，如图\ref{fig:chap2_model_illustrate}所示。
```

**替换为以下 LaTeX 内容:**

```latex
下面以具体的紧束缚模型来展示广义嵌套Berry相如何指示高阶费米弧的存在。我们首先构建一个具有相位双配分 (phase-bipartite) 结构的二维紧束缚模型\cite{bernevig2020HOFA}，用以说明$\theta^{(4)}$所捕获的拓扑相变；然后给出非共形空间群$P4/nmm$的三维紧束缚模型\cite{zhang2022magnetictb}，作为高阶费米弧的最小模型示例。

\subsubsection{相位双配分紧束缚模型}

相位双配分模型的基本思路是：在具有非平凡传统嵌套Berry相$\theta^{(2)}$的模型基础上引入双配分相位因子，使体系的镜面对称性$\mathcal{M}_z$推广为滑移对称性$\mathcal{G}_z$。当相位因子在$\mathcal{M}_z\mathcal{T}$操作下为偶函数时，该二维模型可视为三维$\mathcal{G}_z$对称体系的$k_z$切片。

具体地，我们考虑方阵格子上$d_{x^2-y^2}$-$p_z$轨道杂化模型，其$k_z$切片的哈密顿量为
\begin{equation}\label{eq:chap2_dp_model}
\begin{aligned}
    H_{k_z}(k_x,k_y) = &\; t_{PH}\cos(k_z) + t_{z1}\cos(k_z)\,\tau^z
    + t_{xy}\left[\cos(k_x)+\cos(k_y)\right]\tau^z \\
    &+ v_s\left[\sin(k_x)\,\tau^x\sigma^y + \sin(k_y)\,\tau^x\sigma^x\right]
    + t_{z2}\cos(k_z)\,e^{i\phi\lambda_z}\left[\cos(k_x)+\cos(k_y)\right]\tau^z \\
    &+ v_{Q1}\sin(k_z)\,e^{i\phi\lambda_z}\left[\cos(k_x)-\cos(k_y)\right]\tau^y
    + v_{Q2}\sin(k_z)\sin(k_x)\sin(k_y)\,\tau^x\sigma^z,
\end{aligned}
\end{equation}
其中$\lambda$、$\tau$、$\sigma$分别为子格、轨道和自旋自由度的 Pauli 矩阵。各参数的物理含义如下：$t_{PH}$为粒子—空穴对称性破缺项，$t_{z1}$和$t_{xy}$为最近邻跳跃积分，$v_s$为最近邻自旋—轨道耦合(SOC)项，$t_{z2}$和$v_{Q1}$为次近邻跳跃和SOC项，$v_{Q2}$为第三近邻SOC项。相位因子$\phi$度量了体系从$\mathcal{M}_z$对称模型到$\mathcal{G}_z$对称模型的偏离程度。本文计算中所用的参数列于表\ref{tab:chap2_bipartite_params}。

\begin{table}[htbp]
    \centering
    \begin{tabular}{ccccccc}
    \toprule
         $t_{z1}$ & $t_{z2}$ & $t_{xy}$ & $t_{PH}$ & $v_s$ & $v_{Q1}$ & $v_{Q2}$ \\
    \midrule
         0.9 & 0.9 & 1.0 & 0.1 & 0.8 & 0.6 & 0.25 \\
    \bottomrule
    \end{tabular}
    \caption{相位双配分$d_{x^2-y^2}$-$p_z$杂化紧束缚模型~\eqref{eq:chap2_dp_model}的参数取值。}
    \label{tab:chap2_bipartite_params}
\end{table}

利用上述参数，我们计算了$\phi$-$k_z$相图。当$\phi=0$时，模型退化为具有$\mathcal{M}_z$对称性的$d$-$p$杂化模型，此时相边界与$\phi=0$轴的交点恰好对应高阶拓扑相变点(HOTPT)，可同时被$x\pm y$方向的$\theta^{(2)}$和$x$、$y$方向的$\theta^{(4)}$所捕获——这验证了广义嵌套Berry相与传统嵌套Berry相在正交格子超胞情形下的一致性。相图中呈现两类不同的相边界：一类伴随体能隙关闭和Wannier能带的剧烈变化；另一类仅伴随Wannier能隙的关闭（出现在$\nu_x=0$和$1/2$处），对应于边界阻碍拓扑相(BOTP)\cite{khalaf2021BOTP}。后者意味着广义嵌套Berry相$\theta^{(4)}$的跳变也能捕获仅在边界上体现的拓扑相变。

\subsubsection{\texorpdfstring{$P4/nmm$}{P4/nmm}紧束缚模型}

在此基础上，我们进一步构建空间群$P4/nmm$的三维紧束缚模型\cite{zhang2022magnetictb}，以$d_{x^2-y^2}$-$p_z$轨道置于$2c$ Wyckoff 位置。模型$k_z$切片的哈密顿量为
\begin{equation}\label{eq:chap2_P4nmm_model}
\begin{aligned}
    H_{k_z} = e_1\,\tau^z &+ t_1\left[\cos\frac{k_x+k_y}{2}-\cos\frac{k_x-k_y}{2}\right]\lambda_y\tau_x\sigma_0
    + t_2\left[\sin\frac{k_x+k_y}{2}-\sin\frac{k_x-k_y}{2}\right]\lambda_x\frac{1+\tau_z}{2}\sigma_0 \\
    &+ t_3\left[\sin\frac{k_x+k_y}{2}-\sin\frac{k_x-k_y}{2}\right]\lambda_x\tau_x\sigma_x
    + t_3\left[\sin\frac{k_x+k_y}{2}+\sin\frac{k_x-k_y}{2}\right]\lambda_x\tau_x\sigma_y \\
    &+ t_4\left[\sin\frac{k_x+k_y}{2}-\sin\frac{k_x-k_y}{2}\right]\lambda_x\frac{1-\tau_z}{2}\sigma_0
    + (s_5 - 2p_5\cos k_z)\left[\cos k_x - \cos k_y\right]\lambda_z\tau_x\sigma_0,
\end{aligned}
\end{equation}
其中$t_j = f_j + g_j\,e^{i\phi}$（$j=1,2,3,4$），$f_j$和$g_j$分别为最近邻和次近邻跳跃及SOC参数，相位因子$\phi = k_z$。$s_5$为面内SOC项，$p_5$为高阶SOC项。模型参数列于表\ref{tab:chap2_P4nmm_params}。

\begin{table}[htbp]
    \centering
    \begin{tabular}{cccccccccc}
    \toprule
         $e$ & $f_1$ & $g_1$ & $f_2$ & $g_2$ & $f_3$ & $g_3$ & $f_4$ & $g_4$ & $s_5$ \\
    \midrule
         1.8 & 0.3 & 0.4 & 0.1 & 0.2 & 0.1 & 0.3 & $-0.3$ & $-0.5$ & 0.4 \\
    \bottomrule
    \end{tabular}
    \caption{$P4/nmm$空间群$d_{x^2-y^2}$-$p_z$紧束缚模型~\eqref{eq:chap2_P4nmm_model}的参数取值。}
    \label{tab:chap2_P4nmm_params}
\end{table}

该模型在$k_z$轴上具有一对狄拉克点，描述了一个三维狄拉克半金属，如图\ref{fig:chap2_model_illustrate}所示。$p_5$-$k_z$相图同样呈现两类相边界：体能隙关闭型的HOTPT（对应Wannier能带的剧烈重构）和Wannier能隙关闭型的BOTP边界（出现在$\nu_x = \pm 1/4$处）。
```

---

## 后续上下文验证

替换后，紧接着应为原来的 `\begin{figure}` 环境（L521 起），其内容保持不变：

```
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.9\linewidth]{figures/chap02_Fig1.eps}
  \caption{(a) 空间群$P4/nmm$中具有一对狄拉克点的...}
  ...
```

---

## 注意事项

1. **引用 cite keys:** `bernevig2020HOFA`, `zhang2022magnetictb`, `khalaf2021BOTP` 需要确认已在 `refs.bib` 中定义。这些 cite keys 来自原始论文 SM，需要检查论文的 bib 文件是否已合并到学位论文的参考文献中。

2. **`\subsubsection` 层级:** 当前 §2.3.5 标题为 `\subsection{最小模型示例与高阶费米弧}`。新增的两个 `\subsubsection` 将 §2.3.5 细分为两个子小节。如果 Hui 不想要子小节标题，可将 `\subsubsection{...}` 替换为加粗段落标题 `\paragraph{...}`。

3. **equation label 命名:** 遵循 chap02 中已有的 `eq:chap2_*` 命名规范。

4. **表格样式:** 使用 `booktabs` 宏包的 `\toprule`/`\midrule`/`\bottomrule`（清华论文模板通常已加载此宏包），风格与学位论文一致。如未加载可回退为 `\hline`。

5. **原始内容保留:** 从 L527 起的对称性分析、Wannier 扇区讨论、填充异常讨论等段落完全保留不变。

6. **图片:** SM 中的 `bipartite.png` 和 `P4nmm.png` 相图如需引用，需复制到论文 `figures/` 目录。当前 patch 未新增图片引用（相图讨论用文字描述），但如果 Hui 希望将相图也纳入正文，可以在后续 patch 中添加。
