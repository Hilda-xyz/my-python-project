## 一、 什么是Transformer？

Transformer 是一种基于自注意力机制（Self-Attention）的深度学习模型架构，最初在 "Attention Is All You Need" 论文中被提出，用于神经机器翻译。

与依赖循环（RNN）或卷积（CNN）的传统模型不同，Transformer 完全依赖注意力机制来捕捉输入和输出之间的全局依赖关系。这使其具有高度并行化的能力，极大地提升了训练效率，并使其成为当今大型语言模型（LLM）的基础。

![[transformer知识点总结-20.png]]

### 1. NLP 与 LLM

* **NLP (自然语言处理)**：核心任务是让计算机理解和处理人类语言。
    * **传统NLP任务**：对整个句子分类、对句子中每一个单词分类、生成文本、从文本中提取答案、机器翻译等。
    * **传统NLP方法**：需要为特定任务（如情感分析、实体识别）构造专用模型。
* **LLM (大型语言模型)**：
    * **定义**：基于海量文本数据训练的、包含海量参数（数十亿甚至上万亿）的 AI 模型。
    * **特点**：无需为特定任务进行专门训练，可通过提示（Prompting）或微调（Fine-tuning）解决各种任务。
    * **局限性**：可能产生幻觉（Hallucinations）、缺乏深层理解、存在偏见、上下文窗口有限、计算资源消耗大。
    * **代表架构**：GPT（生成式预训练 Transformer）、Llama、BERT 等。

### 2. 核心思想：迁移学习

* **中心思想**：利用在**大规模数据集**上预训练好的模型（学到的“通用知识”），将其应用到**小数据集的新任务**上。

![[transformer知识点总结-13.png]]
* **预训练 (Pre-training)**：
    * 在海量无标签数据上以**自监督学习**的方式进行训练。
    * **自监督学习**：目标（标签）由模型输入自动计算得出，无需人工标注。
    * 例如：预测句子中的下一个词（CLM）或预测被遮盖的词（MLM）。
* **微调 (Fine-tuning)**：
    * 在模型预训练**之后**进行的训练。
    * 使用特定任务的、带标签的小规模数据集进行额外训练（监督学习）。
    * 例如：在一个预训练好的英语模型基础上，使用一个医疗领域的问答数据集进行微调，使其成为一个医疗问答模型。

## 二、 Transformer 原始架构 (Encoder-Decoder)

Transformer 原始架构主要由 **Encoder（编码器）** 和 **Decoder（解码器）** 两大部分组成。

### 1. 整体工作流程

**第一步：准备输入 (Input)**
获取输入句子中每个单词的表示向量 **X**。这个 **X** 由==单词的 Embedding==（词嵌入）和==位置的 Embedding==（位置编码）相加得到。

![[transformer知识点总结-21.png]]

**第二步：编码 (Encoder)**
将单词表示向量矩阵 **X**（$X_{n×d}$，n是单词数，d是维度）传入 Encoder。原始论文中 Encoder 由 6 个相同的 Encoder Block 堆叠而成。
最终，Encoder 输出句子所有单词的编码信息矩阵 **C**（维度与输入 **X** 相同）。

![[transformer知识点总结-22.png]]


**第三步：解码 (Decoder)**
将 Encoder 输出的编码信息矩阵 **C** 传递到 Decoder。Decoder 同样由 6 个 Decoder Block 堆叠而成。
Decoder 在生成（翻译）时是**自回归**的，即依次根据当前已翻译过的单词 1~i 来翻译下一个单词 i+1。在训练时，为防止模型“作弊”看到未来的信息，会使用 **Mask (掩码)** 操作遮盖住 i+1 及其之后的单词。

![[transformer知识点总结-23.png]]

### 2. Transformer 的输入

* **词嵌入 (Token Embedding)**：将单词（或子词）转换为高维向量。可以采用 Word2Vec、Glove 等预训练得到，也可以在 Transformer 训练过程中学习得到。
* **位置编码 (Positional Embedding)**：
    * **原因**：Transformer 抛弃了 RNN 的循环结构，完全基于 Self-Attention，这使其无法捕捉单词的顺序信息。
    * **作用**：为模型提供单词在序列中的相对或绝对位置信息。原始论文中使用 `sin` 和 `cos` 函数的组合来生成。

![[transformer知识点总结-24.png]]

### 3. 核心机制：Self-Attention (自注意力)

Self-Attention 允许模型在处理一个单词时，同时关注到句子中的所有其他单词，并计算它们对当前单词的重要性（权重）。

* **Q, K, V 矩阵**：
    * 在计算时，Self-Attention 接收的是输入矩阵 **X**（或上一层的输出）。
    * **X** 会通过三个不同的线性变换（乘以三个权重矩阵 $W^Q, W^K, W^V$）得到三个矩阵：**Q (Query, 查询)**、**K (Key, 键值)** 和 **V (Value, 值)**。
    * **Q**：代表当前单词，用于去“查询”其他单词。
    * **K**：代表句子中的所有单词（包括自己），用于被“查询”。
    * **V**：代表句子中所有单词的实际信息。

![[transformer知识点总结-25.png]]

* **计算过程**（Scaled Dot-Product Attention）：
    1.  **打分**：$Score = Q \cdot K^T$ （计算Q和K的相似度）
    2.  **缩放**：$\frac{Score}{\sqrt{d_k}}$ （$d_k$是K的维度，用于稳定梯度）
    3.  **归一化**：$Softmax(\frac{Q K^T}{\sqrt{d_k}})$ （得到0到1之间的权重）
    4.  **加权求和**：$Attention(Q, K, V) = Softmax(\frac{Q K^T}{\sqrt{d_k}}) V$

* **Multi-Head Attention (多头注意力)**：
    * Transformer 并不只使用一组 Q, K, V，而是将 Q, K, V 在维度上切分为多份（例如 8 "头"），每 "头" 单独计算一次 Self-Attention。
    * **好处**：允许模型在不同的表示子空间中共同关注来自不同位置的信息。
    * 最后，将所有 "头" 的输出矩阵拼接（Concat）起来，再通过一个线性变换得到最终输出。

![[transformer知识点总结-27.png]]

### 4. Encoder 结构 (编码器)
![[transformer知识点总结-28.png]]
每个 Encoder Block 由两个子层组成：

1.  **多头自注意力层 (Multi-Head Self-Attention)**
2.  **前馈神经网络 (Feed Forward Network)**

* **Add & Norm (残差连接与层归一化)**：
    * 每个子层（注意力层和前馈网络）的输出都使用了一个 **Add & Norm** 操作。
    * **Add (残差连接)**：$Sublayer(X) + X$。将子层的输入 **X** 直接加到子层的输出 $Sublayer(X)$ 上。这有助于防止梯度消失，使模型更容易训练深层网络。
    * **Norm (层归一化, LayerNorm)**：对残差连接后的结果进行归一化。
    * **公式**：$LayerNorm(X + Sublayer(X))$
* **Feed Forward Network (FFN)**：
    * 一个简单的两层全连接网络，第一层使用 ReLU 激活函数，第二层不使用激活函数。
    * **公式**：$FFN(x) = max(0, xW_1 + b_1)W_2 + b_2$
    * 这一层在每个单词的位置上是独立进行的。

### 5. Decoder 结构 (解码器)

![[transformer知识点总结-31.png]]
每个 Decoder Block 由三个子层组成：

1.  **带掩码的多头自注意力层 (Masked Multi-Head Self-Attention)**
    * 这是 Decoder 的第一个 Multi-Head Attention 层。
    * **Masked (掩码)**：在计算 Self-Attention 时，会遮盖住当前位置之后的所有单词（即未来信息），确保在预测第 `i` 个词时，只能依赖第 `1` 到 `i-1` 个词的信息。

![[transformer知识点总结-32.png]]

1.  **编码器-解码器注意力层 (Encoder-Decoder Attention / Cross-Attention)**
    * 这是 Decoder 的第二个 Multi-Head Attention 层。
    * **关键**：它的 **Q (Query)** 矩阵来自上一个 Decoder Block 的输出。
    * 它的 **K (Key)** 和 **V (Value)** 矩阵则**使用 Encoder 最终输出的编码信息矩阵 C**。
    * **作用**：允许 Decoder 在生成每个词时，都能“关注”到输入句子（源语言）的各个部分。
2.  **前馈神经网络 (Feed Forward Network)**
    * 与 Encoder 中的 FFN 结构相同。

* **Add & Norm**：Decoder 中的每个子层同样都使用了 Add & Norm 操作。
* **最终输出**：Decoder 栈的最终输出会经过一个线性层和 Softmax 层，用于计算词汇表中每个单词的概率，从而预测下一个翻译的单词。

## 三、 Transformer 模型家族分类

根据原始架构的使用方式，Transformer 模型主要分为三类：

1.  **仅编码器模型 (Encoder-only)**
    * **代表**：**BERT**, RoBERTa, ALBERT。
    * **特点**：能同时“看到”句子的双向上下文（通过MLM）。
    * **预训练目标**：**掩码语言模型 (MLM)**。随机屏蔽输入中的一些词条（如 `[MASK]`），训练模型根据周围的上下文预测原始词条。
    * **适用任务**：需要深度理解输入的任务，如文本分类、命名实体识别（NER）、问答（QA）。
2.  **仅解码器模型 (Decoder-only)**
    * **代表**：**GPT** 系列, Llama, Mistral。
    * **特点**：只能“看到”左侧的上下文（单向）。
    * **预训练目标**：**因果语言模型 (CLM)**。根据序列中所有先前的标记来预测下一个标记。
    * **适用任务**：生成式任务，如文本生成、文章续写。
3.  **编码器-解码器模型 (Encoder-Decoder / Seq2Seq)**
    * **代表**：**BART**, **T5**, mBART。
    * **特点**：结合了前两者的特点，适用于需要根据一个输入序列生成另一个序列的任务。
    * **预训练目标**：通常更复杂，如 T5 的“文本填充 (Text Infilling)”（用一个 `[MASK]` 标记替换一整段连续文字，让模型“填空”）。
    * **适用任务**：摘要、翻译。

| 模型 | 结构类型 | 注意力方向 | 预训练目标 | 擅长任务 |
| :--- | :--- | :--- | :--- | :--- |
| **BERT** | 仅编码器 | 双向（同时看两边） | 掩码语言模型 (MLM) | 理解文本（分类、问答、NER） |
| **GPT-2** | 仅解码器 | 单向（只能看左边） | 因果语言模型 (CLM) | 生成文本（写作、续写） |
| **BART/T5** | 编码器-解码器 | 双向(Encoder) + 单向(Decoder) | 序列到序列（如文本填充） | 摘要、翻译 |

## 四、 Hugging Face `pipeline` 与应用

Hugging Face `pipeline` (管道) 是 `transformers` 库中最高级的API，它将模型加载、预处理（Tokenization）和后处理步骤组合在一起，让我们能用几行代码完成复杂的NLP任务。

### 1. 文本 (Text) 任务

* **`text-classification` (文本分类)**
![[图片/transformer知识点总结/transformer知识点总结.png]]
    * **任务**：对整个句子进行分类（如情感分析）。
    * **示例模型**：**BERT**。
    * [BERT](https://huggingface.co/docs/transformers/model_doc/bert) 是一个仅编码器模型，也是第一个有效实现==深度双向性==的模型，通过关注两侧的单词来学习更丰富的文本表示。BERT 使用==[WordPiece]==标记化来生成文本的标记嵌入。`[SEP]`：放在句子末尾，用来**区分句子或句子对**。`[CLS]`：放在句首，整个句子的“代表”，  用来做**分类任务**（比如情感分析：正面/负面）。带有 `[CLS]` 标记的最终输出将用作分类任务的分类头的输入。BERT 还添加了一个==段嵌入==来指示一个标记属于一对句子中的第一个句子还是第二个句子。
    * BERT 预训练有两个目标：掩码语言建模和下一句预测。（1）在掩码语言建模中，一定比例的输入 token 会被==随机==掩码，模型需要预测这些 token。这解决了双向性问题，即模型可以“作弊”地查看所有单词并“预测”下一个单词。预测的掩码 token 的最终隐藏状态会被传递到一个前馈网络中，该网络使用一个基于词汇表的 softmax 函数来预测被掩码的单词。（2）第二个预训练对象是下一句预测。模型必须预测句子 B 是否紧随句子 A 之后。一半情况下，句子 B 是下一句；另一半情况下，句子 B 是一个随机句子。无论是否是下一句，预测结果都会被传递到一个前馈网络中，该网络使用 softmax 函数对两个类别（ `IsNext` 和 `NotNext` ）进行处理。输入嵌入经过多个编码器层来输出一些最终的隐藏状态。要使用预训练模型进行文本分类，请在基础 BERT 模型上添加一个==序列分类头==。序列分类头是一个线性层，它接受最终隐藏状态并执行线性变换将其转换为对数函数 (logits)。计算对数函数和目标函数之间的交叉熵损失，以找到最可能的标签。
    * **工作流程**：
        1.  输入文本前会添加一个特殊的 `[CLS]` 标记。
        2.  `[CLS]` 标记的最终输出向量被认为是整个句子的“代表”。
        3.  在 BERT 基础上添加一个**序列分类头**（一个线性层），用 `[CLS]` 向量来预测类别（如正面/负面）。
* **`token-classification` (标记分类)**

![[transformer知识点总结-5.png]]
    * **任务**：对句子中的每一个单词（Token）进行分类（如命名实体识别 NER）。
    * **示例模型**：**BERT**。
    * 要使用 BERT 执行命名实体识别 (NER) 等标记分类任务，请在基础 BERT 模型上添加一个==标记分类头==。标记分类头是一个线性层。输入：每个词在 BERT 最后一层的隐藏状态（即每个词的语义表示）执行线性变换输出：每个词属于哪个类别的概率（通过 softmax 得到）损失函数：计算对数函数和每个标记之间的交叉熵损失，以找到最可能的标签。
* **`question-answering` (问答)**
![[transformer知识点总结-6.png]]
    * **任务**：给定一个问题 (Question) 和一段上下文 (Context)，从上下文中提取答案。
    * **示例模型**：**BERT**。
    * **工作流程**：
        1.  在 BERT 基础上添加一个**跨度分类头 (Span Classification Head)**。
        2.  该头包含两组线性层，分别用于预测**答案的开始位置 (start logit)** 和**答案的结束位置 (end logit)**。
        3.  模型会计算哪个 (start, end) 组合得分最高，该片段即为答案。
* **`text-generation` (文本生成)**

![[transformer知识点总结-2.png]]
![[transformer知识点总结-3.png]]

![[transformer知识点总结-18.png]]
    * **任务**：根据提示 (Prompt) 生成后续文本。
    * **示例模型**：**GPT-2**。
    * [GPT-2](https://huggingface.co/docs/transformers/model_doc/gpt2) 是一个基于大量文本进行预训练的纯解码器模型。它可以根据提示生成令人信服的文本（尽管并非总是正确！），并且即使没有经过明确的训练，也能完成问答等其他 NLP 任务。GPT-2 的预训练目标完全基于[因果语言模型](https://huggingface.co/docs/transformers/glossary#causal-language-modeling) ，预测序列中的下一个单词。这使得 GPT-2 尤其擅长文本生成任务。GPT-2 使用==字节对编码 (BPE)==对单词进行标记并生成标记嵌入。==位置编码==被添加到标记嵌入中，以指示每个标记在序列中的位置。输入嵌入经过多个解码器块，以输出最终的隐藏状态。在每个解码器块中，GPT-2 使用一个==带掩码的自注意力层==，这意味着 GPT-2 无法关注未来的标记。它==只能关注左侧的标记==。这与 BERT 的 [ `mask` ] 标记不同，因为**在带掩码的自注意力中，注意力掩码用于将未来标记的分数设置为 `0`** 。解码器的输出被传递到语言建模头，该头执行线性变换，将隐藏状态转换为==逻辑向量 ==(logits)。标签是序列中的下一个标记，它是通过将逻辑向量向右移动一位而生成的。计算移位后的逻辑向量与标签之间的==交叉熵损失==，以输出下一个最可能的标记。
    * **工作流程**：
        1.  GPT-2 是一个纯解码器模型，预训练目标是 CLM（预测下一个词）。
        2.  它使用**带掩码的自注意力层**，确保在预测时只能关注左侧的词。
        3.  解码器的输出被传递到语言建模头，该头计算词汇表中每个词的概率（Logits），然后（通常通过采样）选择下一个词。
* **`summarization` (摘要)**

![[transformer知识点总结-7.png]]
    * **任务**：将长文本压缩为短文本。
    * **示例模型**：**BART**, **T5** (Encoder-Decoder 模型)。
    * BART 的编码器架构与 BERT 非常相似，并且接受文本的标记和位置嵌入。BART 通过==破坏==输入然后用解码器重建它来进行预训练。BART 可以用很多种“破坏方式”：- 随机打乱顺序- 删除一部分句子- 用 `[MASK]` 替换部分片段其中最有效的是叫 **“文本填充 (text infilling)”**：把**一整段连续文字**用**一个 `[MASK]` 标记**替换。 这迫使模型学会“填补”出正确长度、正确内容的文字。
    * **工作流程**：
        1.  BART 的预训练任务之一是“文本填充”，这使其擅长“修复”和“重构”文本。
        2.  **编码器**负责理解输入的长文本。
        3.  **解码器**负责根据编码器的理解，生成浓缩后的摘要文本。
* **`translation` (翻译)**
![[transformer知识点总结-8.png]]
    * **任务**：将文本从一种语言翻译成另一种语言。
    * **示例模型**：**T5**, **mBART** (多语言BART)。
    * BART 通过添加一个单独的==随机初始化编码器==来适应翻译，该编码器将源语言映射到可解码为目标语言的输入。这个新编码器的嵌入向量（而非原始词嵌入向量）被传递给预训练编码器。源编码器的训练方法是使用模型输出的交叉熵损失来更新源编码器、位置嵌入向量和输入嵌入向量。在第一步中，模型参数被冻结；在第二步中，所有模型参数一起训练。BART 随后推出了多语言版本 mBART，旨在用于翻译，并已针对多种不同语言进行了预训练。
    * **工作流程**：与摘要类似，使用 Encoder-Decoder 结构，将源语言编码，然后解码为目标语言。
* **`zero-shot-classification` (零样本分类)**

![[transformer知识点总结-1.png]]
    * **任务**：无需在特定标签上微调，即可对文本进行分类。
    * **工作流程**：模型（通常是 NLI 模型）会判断输入文本与你提供的候选标签之间的“蕴含”关系，并给出概率分数。
    
* 文本补全
![[transformer知识点总结-34.png]]
### 2. 语音 (Audio) 任务

* **`automatic-speech-recognition` (ASR - 自动语音识别)**
![[transformer知识点总结-33.png]]
    * **任务**：将语音转换为文本。
    * **示例模型**：**Whisper** (Encoder-Decoder)。
    *  [Whisper](https://huggingface.co/docs/transformers/main/en/model_doc/whisper) 是一个编码器-解码器（序列到序列）Transformer，已基于 68 万小时的带标签音频数据进行预训练（大规模，弱监督的预训练）。如此海量的预训练数据使其能够在英语和许多其他语言的音频任务上实现零样本性能。解码器允许 Whisper 将编码器学习到的语音表示映射到有用的输出（例如文本），而无需进行额外的微调。Whisper 开箱即用。
    * **工作流程**：
        1.  **编码器**处理音频（将其转换为对数梅尔声谱图），并构建音频表征。
        2.  **解码器**接收编码后的音频表征，并自回归地预测出文本标记。
        3.  Whisper 在海量带标签音频数据上预训练，使其具备强大的零样本转录和翻译能力。

```python
from transformers import pipeline

transcriber = pipeline(
    task="automatic-speech-recognition", model="openai/whisper-base.en"
)
transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
# Output: {'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}
```
    
* **`audio-classification` (音频分类)**
* **`text-to-speech` (TTS - 文本转语音)**

### 3. 视觉 (Vision) 任务

* **`image-classification` (图像分类)**
```python
from transformers import pipeline

image_classifier = pipeline(
    task="image-classification", model="google/vit-base-patch16-224"
)
result = image_classifier(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
)
print(result)


[{'label': 'lynx, catamount', 'score': 0.43350091576576233},
 {'label': 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor',
  'score': 0.034796204417943954},
 {'label': 'snow leopard, ounce, Panthera uncia',
  'score': 0.03240183740854263},
 {'label': 'Egyptian cat', 'score': 0.02394474856555462},
 {'label': 'tiger cat', 'score': 0.02288915030658245}]
```
* **任务**：识别图像中的主要物体。
    * **示例模型**：**Vision Transformer (ViT)**, ConvNeXT。
    * **ViT 工作流程**：
        1.  **图像分块 (Patching)**：将图像分割成一系列固定大小的方块（如 16x16）。
        2.  **块嵌入 (Patch Embedding)**：将每个块通过线性变换转换为一个向量（类似于词嵌入）。
        3.  **[CLS] 标记**：像 BERT 一样，在块嵌入序列的开头添加一个特殊的 `[CLS]` 标记。
        4.  **位置嵌入**：添加可学习的位置嵌入，告知模型每个块的原始位置。
        5.  **Transformer 编码器**：将所有嵌入传递给一个标准的 Transformer 编码器。
        6.  **分类头**：使用 `[CLS]` 标记的最终输出，通过一个 MLP 头进行分类。
* **`object-detection` (物体检测)**：定位并识别图像中的物体 (如 DETR)。
* **`image-segmentation` (图像分割)**：对图像中的每个像素进行分类 (如 Mask2Former)。

### 4. 多模态 (Multimodal) 任务

* **`image-to-text` (图像描述)**：生成图像的文本描述。
* **`visual-question-answering` (VQA)**：根据图像内容回答问题。
