# NeurIPS 2024 

This list will be refined after all paper PDFs are released.

1. **A Careful Examination of Large Language Model Performance on Grade School Arithmetic** [[pdf]](https://arxiv.org/abs/2405.00332v3) `NeurIPS 2024` (46 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          GSM1k is designed to mirror the style and complexity of the established GSM8k benchmark, the gold standard for measuring elementary mathematical reasoning, and ensures that the two benchmarks are comparable across important metrics such as human solve rates, number of steps in solution, answer magnitude, and more.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have achieved impressive success on many benchmarks for mathematical reasoning.However, there is growing concern that some of this performance actually reflects dataset contamination, where data closely resembling benchmark questions leaks into the training data, instead of true reasoning ability.To investigate this claim rigorously, we commission Grade School Math 1000 (GSM1k). GSM1k is designed to mirror the style and complexity of the established GSM8k benchmark,the gold standard for measuring elementary mathematical reasoning. We ensure that the two benchmarks are comparable across important metrics such as human solve rates, number of steps in solution, answer magnitude, and more.When evaluating leading open- and closed-source LLMs on GSM1k, we observe accuracy drops of up to 8%, with several families of models showing evidence of systematic overfitting across almost all model sizes.Further analysis suggests a positive relationship (Spearman's r^2=0.36) between a model's probability of generating an example from GSM8k and its performance gap between GSM8k and GSM1k, suggesting that some models may have partially memorized GSM8k.Nevertheless, many models, especially those on the frontier, show minimal signs of overfitting, and all models broadly demonstrate generalization to novel math problems guaranteed to not be in their training data.
     </details>

2. **OpenMathInstruct-1: A 1.8 Million Math Instruction Tuning Dataset** [[pdf]](http://arxiv.org/abs/2402.10176) `NeurIPS 2024` (42 cite) (4 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The dataset is constructed by synthesizing code-interpreter solutions for GSM8K and MATH, two popular math reasoning benchmarks, using the recently released and permissively licensed Mixtral model and achieves a score competitive with the best gpt-distilled models.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent work has shown the immense potential of synthetically generated datasets for training large language models (LLMs), especially for acquiring targeted skills. Current large-scale math instruction tuning datasets such as MetaMathQA (Yu et al., 2024) and MAmmoTH (Yue et al., 2024) are constructed using outputs from closed-source LLMs with commercially restrictive licenses. A key reason limiting the use of open-source LLMs in these data generation pipelines has been the wide gap between the mathematical skills of the best closed-source LLMs, such as GPT-4, and the best open-source LLMs. Building on the recent progress in open-source LLMs, our proposed prompting novelty, and some brute-force scaling, we construct OpenMathInstruct-1, a math instruction tuning dataset with 1.8M problem-solution pairs. The dataset is constructed by synthesizing code-interpreter solutions for GSM8K and MATH, two popular math reasoning benchmarks, using the recently released and permissively licensed Mixtral model. Our best model, OpenMath-CodeLlama-70B, trained on a subset of OpenMathInstruct-1, achieves a score of 84.6% on GSM8K and 50.7% on MATH, which is competitive with the best gpt-distilled models. We will release our code, models, and the OpenMathInstruct-1 dataset under a commercially permissive license.
     </details>

3. **Chain-of-Thought Reasoning Without Prompting** [[pdf]](http://arxiv.org/abs/2402.10200) `NeurIPS 2024` (38 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          These findings reveal that, intriguingly, CoT reasoning paths can be elicited from pre-trained LLMs by simply altering the decoding process, and that the presence of a CoT in the decoding path correlates with a higher confidence in the model's decoded answer.
     </details>


     <details>
          <summary>Abstract</summary>
          In enhancing the reasoning capabilities of large language models (LLMs), prior research primarily focuses on specific prompting techniques such as few-shot or zero-shot chain-of-thought (CoT) prompting. These methods, while effective, often involve manually intensive prompt engineering. Our study takes a novel approach by asking: Can LLMs reason effectively without any prompting? Our findings reveal that, intriguingly, CoT reasoning paths can be elicited from pre-trained LLMs by simply altering the \textit{decoding} process. Rather than conventional greedy decoding, we investigate the top-$k$ alternative tokens, uncovering that CoT paths are frequently inherent in these sequences. This approach not only bypasses the confounders of prompting but also allows us to assess the LLMs' \textit{intrinsic} reasoning abilities. Moreover, we observe that the presence of a CoT in the decoding path correlates with a higher confidence in the model's decoded answer. This confidence metric effectively differentiates between CoT and non-CoT paths. Extensive empirical studies on various reasoning benchmarks show that the proposed CoT-decoding effectively elicits reasoning capabilities from language models, which were previously obscured by standard greedy decoding.
     </details>

4. **MAmmoTH2: Scaling Instructions from the Web** [[pdf]](http://arxiv.org/abs/2405.03548) `NeurIPS 2024` (34 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work demonstrates how to harvest large-scale, high-quality instruction data without costly human annotation or GPT-4 distillation, providing a new paradigm for building better instruction tuning data.
     </details>


     <details>
          <summary>Abstract</summary>
          Instruction tuning improves the reasoning abilities of large language models (LLMs), with data quality and scalability being the crucial factors. Most instruction tuning data come from human crowd-sourcing or GPT-4 distillation. We propose a paradigm to efficiently harvest 10 million naturally existing instruction data from the pre-training web corpus to enhance LLM reasoning. Our approach involves (1) recalling relevant documents, (2) extracting instruction-response pairs, and (3) refining the extracted pairs using open-source LLMs. Fine-tuning base LLMs on this dataset, we build MAmmoTH2 models, which significantly boost performance on reasoning benchmarks. Notably, MAmmoTH2-7B’s (Mistral) performance increases from 11% to 36.7% on MATH and from 36% to 68.4% on GSM8K without training on any in-domain data. Further training MAmmoTH2 on public instruction tuning datasets yields MAmmoTH2-Plus, achieving state-of-the-art performance on several reasoning and chatbot benchmarks. Our work demonstrates how to harvest large-scale, high-quality instruction data without costly human annotation or GPT-4 distillation, providing a new paradigm for building better instruction tuning data.
     </details>

5. **SELF-DISCOVER: Large Language Models Self-Compose Reasoning Structures** [[pdf]](http://arxiv.org/abs/2402.03620) `NeurIPS 2024` (27 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          SELF-DISCOVER substantially improves GPT-4 and PaLM 2's performance on challenging reasoning benchmarks such as BigBench-Hard, grounded agent reasoning, and MATH, and outperforms inference-intensive methods such as CoT-Self-Consistency by more than 20%, while requiring 10-40x fewer inference compute.
     </details>


     <details>
          <summary>Abstract</summary>
          We introduce SELF-DISCOVER, a general framework for LLMs to self-discover the task-intrinsic reasoning structures to tackle complex reasoning problems that are challenging for typical prompting methods. Core to the framework is a self-discovery process where LLMs select multiple atomic reasoning modules such as critical thinking and step-by-step thinking, and compose them into an explicit reasoning structure for LLMs to follow during decoding. SELF-DISCOVER substantially improves GPT-4 and PaLM 2’s performance on challenging reasoning benchmarks such as BigBench-Hard, grounded agent reasoning, and MATH, by as much as 32% compared to Chain of Thought (CoT). Furthermore, SELF-DISCOVER outperforms inference-intensive methods such as CoT-Self-Consistency by more than 20%, while requiring 10-40x fewer inference compute. Finally, we show that the self-discovered reasoning structures are universally applicable across model families: from PaLM 2-L to GPT-4, and from GPT-4 to Llama2, and share commonalities with human reasoning patterns.
     </details>

6. **Measuring Multimodal Mathematical Reasoning with MATH-Vision Dataset** [[pdf]](http://arxiv.org/abs/2402.14804) `NeurIPS 2024` (24 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Recent advancements in Large Multimodal Models (LMMs) have shown promising results in mathematical reasoning within visual contexts, with models exceeding human-level performance on existing benchmarks such as MathVista. However, we observe significant limitations in the diversity of questions and breadth of subjects covered by these benchmarks. To address this issue, we present the MATH-Vision (MATH-V) dataset, a meticulously curated collection of 3,040 high-quality mathematical problems with visual contexts sourced from real math competitions. Spanning 16 distinct mathematical disciplines and graded across 5 levels of difficulty, our dataset provides a comprehensive and diverse set of challenges for evaluating the mathematical reasoning abilities of LMMs. Through extensive experimentation, we unveil a notable performance gap between current LMMs and human performance on \datasetname, underscoring the imperative for further advancements in LMMs. Moreover, our detailed categorization allows for a thorough error analysis of LMMs, offering valuable insights to guide future research and development. The dataset is released at MathLLMs/MathVision
     </details>

7. **Iterative Reasoning Preference Optimization** [[pdf]](http://arxiv.org/abs/2404.19733) `NeurIPS 2024` (21 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work develops an iterative approach that optimizes the preference between competing generated Chain-of-Thought (CoT) candidates by optimizing for winning vs. losing reasoning steps that lead to the correct answer.
     </details>


     <details>
          <summary>Abstract</summary>
          Iterative preference optimization methods have recently been shown to perform well for general instruction tuning tasks, but typically make little improvement on reasoning tasks. In this work we develop an iterative approach that optimizes the preference between competing generated Chain-of-Thought (CoT) candidates by optimizing for winning vs. losing reasoning steps that lead to the correct answer. We train using a modified DPO loss with an additional negative log-likelihood term, which we find to be crucial. We show reasoning improves across repeated iterations of this scheme. While only relying on examples in the training set, our approach results in increasing accuracy on GSM8K, MATH, and ARC-Challenge for Llama-2-70B-Chat, outperforming other Llama-2-based models not relying on additionally sourced datasets. For example, we see a large improvement from 55.6% to 81.6% on GSM8K and an accuracy of 88.7% with majority voting out of 32 samples.
     </details>

8. **Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing** [[pdf]](http://arxiv.org/abs/2404.12253) `NeurIPS 2024` (20 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          AlphaLLM is introduced, which integrates Monte Carlo Tree Search with LLMs to establish a self-improving loop, thereby enhancing the capabilities of LLMs without additional annotations, showing the potential for self-improvement in LLMs.
     </details>


     <details>
          <summary>Abstract</summary>
          Despite the impressive capabilities of Large Language Models (LLMs) on various tasks, they still struggle with scenarios that involves complex reasoning and planning. Self-correction and self-learning emerge as viable solutions, employing strategies that allow LLMs to refine their outputs and learn from self-assessed rewards. Yet, the efficacy of LLMs in self-refining its response, particularly in complex reasoning and planning task, remains dubious. In this paper, we introduce AlphaLLM for the self-improvements of LLMs, which integrates Monte Carlo Tree Search (MCTS) with LLMs to establish a self-improving loop, thereby enhancing the capabilities of LLMs without additional annotations. Drawing inspiration from the success of AlphaGo, AlphaLLM addresses the unique challenges of combining MCTS with LLM for self-improvement, including data scarcity, the vastness search spaces of language tasks, and the subjective nature of feedback in language tasks. AlphaLLM is comprised of prompt synthesis component, an efficient MCTS approach tailored for language tasks, and a trio of critic models for precise feedback. Our experimental results in mathematical reasoning tasks demonstrate that AlphaLLM significantly enhances the performance of LLMs without additional annotations, showing the potential for self-improvement in LLMs.
     </details>

9. **AlphaMath Almost Zero: Process Supervision Without Process** [[pdf]](http://arxiv.org/abs/2405.03553) `NeurIPS 2024` (16 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This study proposes an innovative framework, AlphaMath, that bypasses the need for process annotations (from humans or GPTs) by leveraging Monte Carlo Tree Search (MCTS), and focuses on unleashing the potential of a well-pretrained LLM to autonomously enhance its mathematical reasoning.
     </details>


     <details>
          <summary>Abstract</summary>
          Although recent advancements in large language models (LLMs) have significantly improved their performance on various tasks, they still face challenges with complex and symbolic multi-step reasoning, particularly in mathematical reasoning. To bolster the mathematical reasoning capabilities of LLMs, most existing efforts concentrate on seeking assistance from either domain experts or GPT-4 for high-quality process-supervised data, which is not only expensive but also labor-intensive. In our study, we propose an innovative framework, AlphaMath, that bypasses the need for process annotations (from humans or GPTs) by leveraging Monte Carlo Tree Search (MCTS). This framework focuses on unleashing the potential of a well-pretrained LLM to autonomously enhance its mathematical reasoning. Specifically, we integrate a value model with the LLM, automatically generating both process supervision and step-level evaluation signals in MCTS. Furthermore, we propose an efficient inference strategy—step-level beam search, where the value model is crafted to assist the policy model (i.e., LLM) in navigating more effective reasoning paths, rather than solely relying on prior probabilities. The experimental results on both in-domain and out-of-domain datasets demonstrate that even without GPT-4 or human-annotated process supervision, our AlphaMath framework achieves comparable or superior results to previous state-of-the-art methods.
     </details>

10. **Transformers Can Do Arithmetic with the Right Embeddings** [[pdf]](http://arxiv.org/abs/2405.17399) `NeurIPS 2024` (12 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work fixes the problem of poor performance of transformers on arithmetic tasks by adding an embedding to each digit that encodes its position relative to the start of the number, and shows that this fix enables architectural modifications such as input injection and recurrent layers to improve performance.
     </details>


     <details>
          <summary>Abstract</summary>
          The poor performance of transformers on arithmetic tasks seems to stem in large part from their inability to keep track of the exact position of each digit inside of a large span of digits. We mend this problem by adding an embedding to each digit that encodes its position relative to the start of the number. In addition to the boost these embeddings provide on their own, we show that this fix enables architectural modifications such as input injection and recurrent layers to improve performance even further.With positions resolved, we can study the logical extrapolation ability of transformers. Can they solve arithmetic problems that are larger and more complex than those in their training data? We find that training on only 20 digit numbers with a single GPU for one day, we can reach state-of-the-art performance, achieving up to 99% accuracy on 100 digit addition problems. Finally, we show that these gains in numeracy also unlock improvements on other multi-step reasoning tasks including sorting and multiplication.
     </details>

11. **JiuZhang3.0: Efficiently Improving Mathematical Reasoning by Training Small Data Synthesis Models** [[pdf]](http://arxiv.org/abs/2405.14365) `NeurIPS 2024` (10 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes an efficient way that trains a small LLM for math problem synthesis, to efficiently generate sufficient high-quality pre-training data, and craft a set of prompts based on human education stages to guide GPT-4, to synthesize problems covering diverse math knowledge and difficulty levels.
     </details>


     <details>
          <summary>Abstract</summary>
          Mathematical reasoning is an important capability of large language models~(LLMs) for real-world applications.To enhance this capability, existing work either collects large-scale math-related texts for pre-training, or relies on stronger LLMs (\eg GPT-4) to synthesize massive math problems. Both types of work generally lead to large costs in training or synthesis.To reduce the cost, based on open-source available texts, we propose an efficient way that trains a small LLM for math problem synthesis, to efficiently generate sufficient high-quality pre-training data.To achieve it, we create a dataset using GPT-4 to distill its data synthesis capability into the small LLM.Concretely, we craft a set of prompts based on human education stages to guide GPT-4, to synthesize problems covering diverse math knowledge and difficulty levels.Besides, we adopt the gradient-based influence estimation method to select the most valuable math-related texts.The both are fed into GPT-4 for creating the knowledge distillation dataset to train the small LLM.We leverage it to synthesize 6 million math problems for pre-training our JiuZhang3.0 model, which only needs to invoke GPT-4 API 9.3k times and pre-train on 4.6B data.Experimental results have shown that JiuZhang3.0 achieves state-of-the-art performance on several mathematical reasoning datasets, under both natural language reasoning and tool manipulation settings.
     </details>

12. **DART-Math: Difficulty-Aware Rejection Tuning for Mathematical Problem-Solving** [[pdf]](http://arxiv.org/abs/2407.13690) `NeurIPS 2024` (5 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Hypothesizing that difficult queries are crucial to learn complex reasoning, this work proposes Difficulty-Aware Rejection Tuning (DART), a method that allocates difficult queries more trials during the synthesis phase, enabling more extensive training on difficult samples.
     </details>


     <details>
          <summary>Abstract</summary>
          Solving mathematical problems requires advanced reasoning abilities and presents notable challenges for large language models. Previous works usually synthesize data from proprietary models to augment existing datasets, followed by instruction tuning to achieve top-tier results. However, our analysis of these datasets reveals severe biases towards easy queries, with frequent failures to generate any correct response for the most challenging queries.Hypothesizing that difficult queries are crucial to learn complex reasoning, we propose Difficulty-Aware Rejection Tuning (DART), a method that allocates difficult queries more trials during the synthesis phase, enabling more extensive training on difficult samples.Utilizing DART, we have created new datasets for mathematical problem-solving that focus more on difficult queries and are substantially smaller than previous ones. Remarkably, our synthesis process solely relies on a 7B-sized open-weight model, without reliance on the commonly used proprietary GPT-4.We fine-tune various base models on our datasets ranging from 7B to 70B in size, resulting in a series of strong models called DART-Math.In comprehensive in-domain and out-of-domain evaluation on 6 mathematical benchmarks, DART-Math outperforms vanilla rejection tuning significantly, being superior or comparable to previous arts, despite using much smaller datasets and no proprietary models. Furthermore, our results position our synthetic datasets as the most effective and cost-efficient publicly available resources for advancing mathematical problem-solving. Our datasets and models will be made publicly available following the review period.
     </details>

13. **Metacognitive Capabilities of LLMs: An Exploration in Mathematical Problem Solving** [[pdf]](https://openreview.net/forum?id=0MsI3bSmmD&name=pdf) `NeurIPS 2024` (5 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          \emph{Metacognitive knowledge} refers to humans' intuitive knowledge of their own thinking and reasoning processes. Today's best LLMs clearly possess some reasoning processes. The paper gives evidence that they also  have metacognitive knowledge, including ability to name skills and procedures to apply given a task. We explore this primarily in context of math reasoning, developing a prompt-guided interaction procedure  to get a powerful  LLM to assign sensible skill labels to math questions, followed by having it perform semantic clustering to obtain coarser families of skill labels. These coarse skill labels look interpretable to humans.To validate that these skill labels are meaningful and relevant to the LLM's reasoning processes we perform the following experiments. (a) We ask GPT-4 to assign skill labels to training questions in math datasets GSM8K and MATH.  (b) When using an LLM to solve the test questions, we present it with the full list of skill labels and ask it to identify the skill needed. Then it is presented with randomly selected exemplar solved questions associated with that skill label.  This improves accuracy on GSM8k and MATH for several strong LLMs, including code-assisted models. The methodology presented is domain-agnostic,  even though this article applies it to math problems.
     </details>

14. **Lean Workbook: A large-scale Lean problem set formalized from natural language math problems** [[pdf]](http://arxiv.org/abs/2406.03847) `NeurIPS 2024` `Lean` (4 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a novel pipeline that iteratively generates and filters synthetic data to translate natural language mathematical problems into Lean 4 statements, and vice versa, and indicates that the synthetic data pipeline can provide useful training data and improve the performance of LLMs in translating and understanding complex mathematical problems and proofs.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models have demonstrated impressive capabilities across various natural language processing tasks, especially in solving mathematical problems. However, large language models are not good at math theorem proving using formal languages like Lean. A significant challenge in this area is the scarcity of training data available in these formal languages. To address this issue, we propose a novel pipeline that iteratively generates and filters synthetic data to translate natural language mathematical problems into Lean 4 statements, and vice versa. Our results indicate that the synthetic data pipeline can provide useful training data and improve the performance of LLMs in translating and understanding complex mathematical problems and proofs. Our final dataset contains about 57K formal-informal question pairs along with searched proof from the math contest forum and 21 new IMO questions. We open-source our code at \url{https://github.com/InternLM/InternLM-Math} and our data at \url{https://huggingface.co/datasets/InternLM/Lean-Workbook}.
     </details>

15. **OlympicArena: Benchmarking Multi-discipline Cognitive Reasoning for Superintelligent AI** [[pdf]](https://arxiv.org/abs/2406.12753) `NeurIPS 2024` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work argues that the challenges in Olympic competition problems are ideal for evaluating AI's cognitive reasoning due to their complexity and interdisciplinary nature, which are essential for tackling complex scientific challenges and facilitating discoveries.
     </details>


     <details>
          <summary>Abstract</summary>
          The evolution of Artificial Intelligence (AI) has been significantly accelerated by advancements in Large Language Models (LLMs) and Large Multimodal Models (LMMs), gradually showcasing potential cognitive reasoning abilities in problem-solving and scientific discovery (i.e., AI4Science) once exclusive to human intellect. To comprehensively evaluate current models' performance in cognitive reasoning abilities, we introduce OlympicArena, which includes 11,163 bilingual problems across both text-only and interleaved text-image modalities. These challenges encompass a wide range of disciplines spanning seven fields and 62 international Olympic competitions, rigorously examined for data leakage. We argue that the challenges in Olympic competition problems are ideal for evaluating AI's cognitive reasoning due to their complexity and interdisciplinary nature, which are essential for tackling complex scientific challenges and facilitating discoveries. Beyond evaluating performance across various disciplines using answer-only criteria, we conduct detailed experiments and analyses from multiple perspectives. We delve into the models' cognitive reasoning abilities, their performance across different modalities, and their outcomes in process-level evaluations, which are vital for tasks requiring complex reasoning with lengthy solutions. Our extensive evaluations reveal that even advanced models like GPT-4o only achieve a 39.97\%  overall accuracy (28.67\%  for mathematics and 29.71\%  for physics), illustrating current AI limitations in complex reasoning and multimodal integration. Through the OlympicArena, we aim to advance AI towards superintelligence, equipping it to address more complex challenges in science and beyond. We also provide a comprehensive set of resources to support AI research, including a benchmark dataset, an open-source annotation platform, a detailed evaluation tool, and a leaderboard with automatic submission features.
     </details>

16. **StrategyLLM: Large Language Models as Strategy Generators, Executors, Optimizers, and Evaluators for Problem Solving** [[pdf]](https://arxiv.org/abs/2311.08803) `NeurIPS 2024` (3 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A comprehensive framework, StrategyLLM, allowing LLMs to perform inductive reasoning, deriving general strategies from specific task instances, and deductive reasoning, applying these general strategies to particular task examples, for constructing generalizable and consistent few-shot prompts.
     </details>


     <details>
          <summary>Abstract</summary>
          Most existing prompting methods suffer from the issues of generalizability and consistency, as they often rely on instance-specific solutions that may not be applicable to other instances and lack task-level consistency across the selected few-shot examples. To address these limitations, we propose a comprehensive framework, StrategyLLM, allowing LLMs to perform inductive reasoning, deriving general strategies from specific task instances, and deductive reasoning, applying these general strategies to particular task examples, for constructing generalizable and consistent few-shot prompts. It employs four LLM-based agents: strategy generator, executor, optimizer, and evaluator, working together to generate, evaluate, and select promising strategies for a given task. Experimental results demonstrate that StrategyLLM outperforms the competitive baseline CoT-SC that requires human-annotated solutions on 13 datasets across 4 challenging tasks without human involvement, including math reasoning (34.2\% $\rightarrow$ 38.8\%), commonsense reasoning (70.3\% $\rightarrow$ 72.5\%), algorithmic reasoning (73.7\% $\rightarrow$ 85.0\%), and symbolic reasoning (30.0\% $\rightarrow$ 79.2\%). Further analysis reveals that StrategyLLM is applicable to various LLMs and demonstrates advantages across numerous scenarios.
     </details>

17. **Visual Sketchpad: Sketching as a Visual Chain of Thought for Multimodal Language Models** [[pdf]](http://arxiv.org/abs/2406.09403) `NeurIPS 2024` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Sketchpad enables LMs to draw with lines, boxes, marks, etc., which is closer to human sketching and better facilitates reasoning, and sets a new state of the art on all tasks, including V*Bench and BLINK spatial reasoning.
     </details>


     <details>
          <summary>Abstract</summary>
          Humans draw to facilitate reasoning: we draw auxiliary lines when solving geometry problems; we mark and circle when reasoning on maps; we use sketches to amplify our ideas and relieve our limited-capacity working memory. However, such actions are missing in current multimodal language models (LMs). Current chain-of-thought and tool-use paradigms only use text as intermediate reasoning steps. In this work, we introduce Sketchpad, a framework that gives multimodal LMs a visual sketchpad and tools to draw on the sketchpad. The LM conducts planning and reasoning according to the visual artifacts it has drawn. Different from prior work, which uses text-to-image models to enable LMs to draw, Sketchpad enables LMs to draw with lines, boxes, marks, etc., which is closer to human sketching and better facilitates reasoning. \name can also use specialist vision models during the sketching process (e.g., draw bounding boxes with object detection models, draw masks with segmentation models), to further enhance visual perception and reasoning. We experiment on a wide range of math tasks (including geometry, functions, graph, chess) and complex visual reasoning tasks. Sketchpad substantially improves performance on all tasks over strong base models with no sketching, yielding an average gain of 12.7% on math tasks, and 8.6% on vision tasks. GPT-4o with Sketchpad sets a new state of the art on all tasks, including V*Bench (80.3%), BLINK spatial reasoning (83.9%), and visual correspondence (80.8%). We will release all code and data.
     </details>

18. **Calibrating Reasoning in Language Models with Internal Consistency** [[pdf]](https://arxiv.org/abs/2405.18711) `NeurIPS 2024` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The results demonstrate the potential of using internal representations for self-evaluation of LLMs by up-weighting reasoning paths with high internal consistency, resulting in a significant boost in reasoning performance.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have demonstrated impressive capabilities in various reasoning tasks, aided by techniques like chain-of-thought (CoT) prompting that elicits verbalized reasoning. However, LLMs often generate text with obvious mistakes and contradictions, raising doubts about their ability to robustly process and utilize generated rationales. In this work, we investigate CoT reasoning in LLMs through the lens of internal representations, focusing on how these representations are influenced by generated rationales. Our preliminary analysis reveals that while generated rationales improve answer accuracy, inconsistencies emerge between the model's internal representations in middle layers and those in final layers, potentially undermining the reliability of their reasoning processes. To address this, we propose internal consistency as a measure of the model's confidence by examining the agreement of latent predictions decoded from intermediate layers. Extensive empirical studies across different models and datasets demonstrate that internal consistency effectively distinguishes between correct and incorrect reasoning paths. Motivated by this, we propose a new approach to calibrate CoT reasoning by up-weighting reasoning paths with high internal consistency, resulting in a significant boost in reasoning performance. Further analysis uncovers distinct patterns in attention and feed-forward modules across layers, providing insights into the emergence of internal inconsistency. In summary, our results demonstrate the potential of using internal representations for self-evaluation of LLMs.
     </details>

19. **How Far Can Transformers Reason? The Locality Barrier and Inductive Scratchpad** [[pdf]](http://arxiv.org/abs/2406.06467) `NeurIPS 2024` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The notion of 'distribution locality' is put forward to capture when weak learning is efficiently achievable by regular Transformers, where the locality measures the least number of tokens required in addition to the tokens histogram to correlate nontrivially with the target.
     </details>


     <details>
          <summary>Abstract</summary>
          Can Transformers predict new syllogisms by composing established ones? More generally, what type of targets can be learned by such models from scratch? Recent works show that Transformers can be Turing-complete in terms of expressivity, but this does not address the learnability objective. This paper puts forward the notion of 'distribution locality' to capture when weak learning is efficiently achievable by regular Transformers, where the locality measures the least number of tokens required in addition to the tokens histogram to correlate nontrivially with the target. As shown experimentally and theoretically under additional assumptions, distributions with high locality cannot be learned efficiently. In particular, syllogisms cannot be composed on long chains. Furthermore, we argue that (i) an agnostic scratchpad cannot help to break the locality, (ii) an educated scratchpad can help if it breaks the locality at each step, (iii) a notion of 'inductive scratchpad' can both break the locality and help with out-of-distribution generalization.
     </details>

20. **DACO: Towards Application-Driven and Comprehensive Data Analysis via Code Generation** [[pdf]](https://openreview.net/forum?id=Y5iTZ52yFs&name=pdf) `NeurIPS 2024` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes to automatically generate high-quality answer annotations leveraging the code-generation capabilities of LLMs with a multi-turn prompting technique, and trains a 6B supervised fine-tuning model on DACO dataset, and finds that the SFT model learns reasonable data analysis capabilities.
     </details>


     <details>
          <summary>Abstract</summary>
          Data analysis is a crucial analytical process essential for deriving insights from real-world databases. As shown in Figure 1, the need for data analysis typically arises from specific application scenarios, and requires diverse reasoning skills including mathematical reasoning, logical reasoning, and strategic reasoning. Existing work often focus on simple factual retrieval or arithmetic resolutions and thus are insufficient for addressing complex real-world queries. This work aims to propose new resources and benchmarks on this crucial yet challenging and under-explored task. Due to the prohibitively high cost of collecting expert annotations, we use large language models (LLMs) enhanced by code generation to automatically generate high-quality data analysis, which will later be refined by human annotators. We construct the DACO dataset, containing (1) 440 databases (of tabular data) collected from real-world scenarios, (2) ~2k automatically generated query-answer pairs that can serve as weak supervision for model training, and (3) a concentrated but high-quality test set with human refined annotations that serves as our main evaluation benchmark. Experiments show that while LLMs like GPT-4 exhibit promising data analysis capabilities, they are still evaluated as less helpful than human-written analysis on 58.1% cases. Leveraging our weak supervision data, we experiment with various fine-tuning methods, including supervised fine-tuning (SFT) and reinforcement learning from human feedback (RLHF). Our trained model outperforms existing baselines for table question answering, and RLHF further boosts the helpfulness of generated analysis on 58.5% cases.Data and code are released at https://github.com/shirley-wu/daco.
     </details>

21. **Easy2Hard-Bench: Standardized Difficulty Labels for Profiling LLM Performance and Generalization** [[pdf]](http://arxiv.org/abs/2409.18433) `NeurIPS 2024` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Through extensive experiments with six state-of-the-art LLMs, this work provides a comprehensive analysis of their performance and generalization capabilities across varying levels of difficulty, with the aim of inspiring future research in LLM generalization.
     </details>


     <details>
          <summary>Abstract</summary>
          Despite the abundance of datasets available for assessing large language models (LLMs), the scarcity of continuous and reliable difficulty labels for individual data points, in most cases, curtails their capacity to benchmark model generalization performance across different levels of complexity. Addressing this limitation, we present Easy2Hard, an innovative collection of 6 benchmark datasets featuring standardized difficulty labels spanning a wide range of domains, such as mathematics and programming problems, chess puzzles, and reasoning questions, providing a much-needed tool for those in demand of a dataset with varying degrees of difficulty for LLM assessment. We estimate the difficulty of individual problems by leveraging the performance data of many human subjects and LLMs on prominent leaderboards. Harnessing the rich human performance data, we employ widely recognized difficulty ranking systems, including the Item Response Theory (IRT) and Glicko-2 models, to uniformly assign difficulty scores to problems. The Easy2Hard datasets distinguish themselves from previous collections by incorporating a significantly higher proportion of challenging problems, presenting a novel and demanding test for state-of-the-art LLMs. Through extensive experiments conducted with six state-of-the-art LLMs on the Easy2Hard datasets, we offer valuable insights into their performance and generalization capabilities across varying degrees of difficulty, setting the stage for future research in LLM generalization.
     </details>

22. **Learning Formal Mathematics From Intrinsic Motivation** [[pdf]](http://arxiv.org/abs/2407.00695) `NeurIPS 2024` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          How did humanity coax mathematics from the aether? We explore the Platonic view that mathematics can be discovered from its axioms---a game of conjecture and proof. We describe an agent that jointly learns to pose challenging problems for itself (conjecturing) and solve them (theorem proving). Given a mathematical domain axiomatized in dependent type theory, we first combine methods for constrained decoding and type-directed synthesis to sample valid conjectures from a language model. Our method guarantees well-formed conjectures by construction, even as we start with a randomly initialized model. We use the same model to represent a policy and value function for guiding proof search. Our agent targets generating hard but provable conjectures --- a moving target, since its own theorem proving ability also improves as it trains. We propose novel methods for hindsight relabeling on proof search trees to significantly improve the agent's sample efficiency in both tasks. Experiments on 3 axiomatic domains (propositional logic, arithmetic and group theory) demonstrate that our agent can bootstrap from only the axioms, self-improving in generating true and challenging conjectures and in finding proofs.
     </details>

23. **MACM: Utilizing a Multi-Agent System for Condition Mining in Solving Complex Mathematical Problems** [[pdf]](http://arxiv.org/abs/2404.04735) `NeurIPS 2024` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduces the MACM prompting method, which not only resolves intricate mathematical problems but also demonstrates strong generalization capabilities across various mathematical contexts.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent advancements in large language models, such as GPT-4, have demonstrated remarkable capabilities in processing standard queries. Despite these advancements, their performance substantially declines in advanced mathematical problems requiring complex, multi-step logical reasoning. To enhance their inferential capabilities, current research has delved into prompting engineering, exemplified by methodologies such as the Tree of Thought and Graph of Thought.Nonetheless, these existing approaches encounter two significant limitations. Firstly, their effectiveness in tackling complex mathematical problems is somewhat constrained. Secondly, the necessity to design distinct prompts for individual problems hampers their generalizability.In response to these limitations, this paper introduces the Multi-Agent System for conditional Mining (MACM) prompting method. It not only resolves intricate mathematical problems but also demonstrates strong generalization capabilities across various mathematical contexts.With the assistance of MACM, the accuracy of GPT-4 Turbo on the most challenging level five mathematical problems in the MATH dataset increase from $\mathbf{54.68\\%}  \text{ to } \mathbf{76.73\\%}$.
     </details>

24. **Proving Olympiad Algebraic Inequalities without Human Demonstrations** [[pdf]](http://arxiv.org/abs/2406.14219) `NeurIPS 2024` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes AIPS, an Algebraic Inequality Proving System capable of autonomously generating complex inequality theorems and effectively solving Olympiad-level inequality problems without requiring human demonstrations.
     </details>


     <details>
          <summary>Abstract</summary>
          Solving Olympiad-level mathematical problems represents a significant advancement in machine intelligence and automated reasoning. Current machine learning methods, however, struggle to solve Olympiad-level problems beyond Euclidean plane geometry due to a lack of large-scale, high-quality datasets. The challenge is even greater in algebraic systems, which involves infinite reasoning spaces within finite conditions. To address these issues, we propose \textit{AIPS}, an \textit{Algebraic Inequality Proving System} capable of autonomously generating complex inequality theorems and effectively solving Olympiad-level inequality problems without requiring human demonstrations. During proof search in a mixed reasoning manner, a value curriculum learning strategy on generated datasets is implemented to improve proving performance, demonstrating strong mathematical intuitions. On a test set of 20 International Mathematical Olympiad-level inequality problems, AIPS successfully solved 10, outperforming state-of-the-art methods. Furthermore, AIPS automatically generated a vast array of non-trivial theorems without human intervention, some of which have been evaluated by professional contestants and deemed to reach the level of the International Mathematical Olympiad. Notably, one theorem was selected as a competition problem in a major city 2024 Mathematical Olympiad.All the materials are available at {\it \href{https://sites.google.com/view/aips}{sites.google.com/view/aips}}.
     </details>

25. **Proving Theorems Recursively** [[pdf]](http://arxiv.org/abs/2405.14414) `NeurIPS 2024` `Isabelle` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          POETRY (PrOvE Theorems RecursivelY), which proves theorems in a recursive, level-by-level manner in the Isabelle theorem prover, is proposed, which allows the theorem to be tackled incrementally by outlining the overall theorem at the first level and then solving the intermediate conjectures at deeper levels.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent advances in automated theorem proving leverages language models to explore expanded search spaces by step-by-step proof generation. However, such approaches are usually based on short-sighted heuristics (e.g., log probability or value function scores) that potentially lead to suboptimal or even distracting subgoals, preventing us from finding longer proofs. To address this challenge, we propose POETRY (PrOvE Theorems RecursivelY), which proves theorems in a recursive, level-by-level manner in the Isabelle theorem prover. Unlike previous step-by-step methods, POETRY searches for a verifiable sketch of the proof at each level and focuses on solving the current level's theorem or conjecture. Detailed proofs of intermediate conjectures within the sketch are temporarily replaced by a placeholder tactic called sorry, deferring their proofs to subsequent levels. This approach allows the theorem to be tackled incrementally by outlining the overall theorem at the first level and then solving the intermediate conjectures at deeper levels. Experiments are conducted on the miniF2F and PISA datasets and significant performance gains are observed in our POETRY approach over state-of-the-art methods. POETRY on miniF2F achieves an average proving success rate improvement of 5.1%. Moreover, we observe a substantial increase in the maximum proof length found by POETRY, from 10 to 26.
     </details>

26. **PutnamBench: Evaluating Neural Theorem-Provers on the Putnam Mathematical Competition** [[pdf]](http://arxiv.org/abs/2407.11214) `NeurIPS 2024` `Lean, Isabelle, Coq` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          PutnamBench consists of 1697 hand-constructed formalizations of 640 theorems sourced from the William Lowell Putnam Mathematical Competition, the premier undergraduate-level mathematics competition in North America.
     </details>


     <details>
          <summary>Abstract</summary>
          We present PutnamBench, a new multilingual benchmark for evaluating the ability of neural theorem-provers to solve competition mathematics problems. PutnamBench consists of 1337 hand-constructed formalizations of 514 theorems sourced from the William Lowell Putnam Mathematical Competition, the premier undergraduate-level mathematics competition in North America.  All the theorems have formalizations in Lean 4 and Isabelle; a substantial subset also has Coq formalizations. Proving the theorems requires significant problem-solving ability and proficiency in a broad range of topics taught in undergraduate mathematics courses. We use PutnamBench to evaluate several established neural and symbolic theorem-provers.  These approaches can only solve a handful of the PutnamBench problems, establishing the benchmark as a difficult open challenge for research on neural theorem-proving. PutnamBench is available at https://github.com/trishullab/PUTNAM.
     </details>

27. **Trajectory Volatility for Out-of-Distribution Detection in Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2405.14039) `NeurIPS 2024` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A trajectory-based method TV score is proposed, which uses trajectory volatility for OOD detection in mathematical reasoning and outperforms all traditional algorithms on GLMs under mathematical reasoning scenarios and can be extended to more applications with high-density features in output spaces, such as multiple-choice questions.
     </details>


     <details>
          <summary>Abstract</summary>
          Real-world data deviating from the independent and identically distributed (\textit{i.i.d.}) assumption of in-distribution training data poses security threats to deep networks, thus advancing out-of-distribution (OOD) detection algorithms. Detection methods in generative language models (GLMs) mainly focus on uncertainty estimation and embedding distance measurement, with the latter proven to be most effective in traditional linguistic tasks like summarization and translation. However, another complex generative scenario mathematical reasoning poses significant challenges to embedding-based methods due to its high-density feature of output spaces, but this feature causes larger discrepancies in the embedding shift trajectory between different samples in latent spaces. Hence, we propose a trajectory-based method TV score, which uses trajectory volatility for OOD detection in mathematical reasoning. Experiments show that our method outperforms all traditional algorithms on GLMs under mathematical reasoning scenarios and can be extended to more applications with high-density features in output spaces, such as multiple-choice questions.
     </details>

28. **Autoformalize Mathematical Statements by Symbolic Equivalence and Semantic Consistency** [[pdf]](https://neurips.cc/virtual/2024/poster/96359) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Autoformalization, the task of automatically translating natural language descriptions into a formal language, poses a significant challenge across various domains, especially in mathematics. Recent advancements in large language models (LLMs) have unveiled their promising capabilities to formalize even competition-level math problems. However, we observe a considerable discrepancy between pass@1 and pass@k accuracies in LLM-generated formalizations. To address this gap, we introduce a novel framework that scores and selects the best result from k autoformalization candidates based on two complementary self-consistency methods: symbolic equivalence and semantic consistency. Elaborately, symbolic equivalence identifies the logical homogeneity among autoformalization candidates using automated theorem provers, and semantic consistency evaluates the preservation of the original meaning by informalizing the candidates and computing the similarity between the embeddings of the original and informalized texts. Our extensive experiments on the MATH and miniF2F datasets demonstrate that our approach significantly enhances autoformalization accuracy, achieving up to 0.22-1.35x relative improvements across various LLMs and baseline methods.
     </details>

29. **Benchmarking the Reasoning Robustness against Noisy Rationales in Chain-of-thought Prompting** [[pdf]](https://neurips.cc/virtual/2024/poster/95956) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          This paper investigates an under-explored challenge in large language models (LLMs): chain-of-thought prompting with noisy rationales—irrelevant or inaccurate reasoning steps—despite advancements in in-context learning. We construct the NoRa dataset, specifically designed to evaluate LLMs’ robustness to noisy rationales, based on which we reveal a widespread vulnerability among LLMs to such noise, with limited efficacy from existing reasoning methods. To combat this, we propose the contrastive denoising with noisy chain-of-thought (CD-CoT) method to enhance denoising-reasoning capabilities by contrasting noisy rationales with only one clean rationale, thereby advancing the robustness of LLMs in reasoning.
     </details>

30. **Counterfactual PPO Enhanced Shared Reflector for LLM-based Multi-agent Collaboration** [[pdf]](https://neurips.cc/virtual/2024/poster/93147) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Benefiting from the powerful language expression and planning capabilities of Large Language Models (LLMs), LLM-based autonomous agents achieve promising performance in various downstream tasks. Recently, based on the development of single-agent systems, researchers propose to construct LLM-based multi-agent systems to tackle more complicated tasks. In this paper, we propose a novel framework, named COPPER, to enhance the collaboration ability of multi-agent systems through learnable self-reflection mechanism. To improve the quality of reflections, we propose to fine-tune a shared reflector, which automatically tunes the prompts of actor models using our counterfactual PPO mechanism. On the one hand, we propose counterfactual rewards to assess the contribution of a single agent’s reflection within the system, alleviating the credit assignment problem. On the other hand, we propose to train a shared reflector, which enables the reflector to personalize generated reflections according to agent roles, while reducing the computational resource requirements and improving training stability. We conduct experiments on three datasets to evaluate the performance of multi-agent systems in multi-hop question answering, mathematics, and chess scenarios. Experimental results show that COPPER possesses stronger reflection capabilities and exhibits excellent generalization performance across different actor models.
     </details>

31. **Diffusion of Thought: Chain-of-Thought Reasoning in Diffusion Language Models** [[pdf]](https://neurips.cc/virtual/2024/poster/95935) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Recently, diffusion models have garnered significant interest in the field of text processing due to their many potential advantages compared to conventional autoregressive models.In this work, we propose Diffusion-of-Thought (DoT),  a novel approach that integrates diffusion models with Chain-of-Thought, a well-established technique for improving the reasoning ability of autoregressive language models. In contrast to autoregressive language models that make decisions in a left-to-right, token-by-token manner, DoT allows reasoning steps to diffuse over time through a diffusion language model and offers greater flexibility in trading-off computation for reasoning performance. Our experimental results demonstrate the effectiveness of DoT in multi-digit multiplication, boolean logic, and grade school math problems. In addition to that, DoT showcases promising self-correction abilities and benefits from existing reasoning-enhancing techniques like self-consistency decoding. Our findings contribute to the understanding and development of reasoning with diffusion language models.
     </details>

32. **Evaluating Large Vision-and-Language Models on Children's Mathematical Olympiads** [[pdf]](http://arxiv.org/abs/2406.15736) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Evaluating state-of-the-art LVLMs on their mathematical and algorithmic reasoning abilities using visuo-linguistic problems from children's Olympiads shows that modern LVLMs do demonstrate increasingly powerful reasoning skills in solving problems for higher grades, but lack the foundations to correctly answer problems designed for younger children.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent years have seen a significant progress in the general-purpose problem solving abilities of large vision and language models (LVLMs), such as ChatGPT, Gemini, etc.; some of these breakthroughs even seem to enable AI models to outperform human abilities in varied tasks that demand higher-order cognitive skills. Are the current large AI models indeed capable of generalized problem solving as humans do?  A systematic analysis of AI capabilities for joint vision and text reasoning, however, is missing in the current scientific literature. In this paper, we make an effort towards filling this gap, by evaluating state-of-the-art LVLMs on their mathematical and algorithmic reasoning abilities using visuo-linguistic problems from children's Olympiads. Specifically, we consider problems from the Mathematical Kangaroo (MK) Olympiad, which is a popular international competition targeted at children from grades 1-12, that tests children's deeper mathematical abilities using puzzles that are appropriately gauged to their age and skills. Using the puzzles from MK, we created a dataset, dubbed SMART-840, consisting of 840 problems from years 2020-2024. With our dataset, we analyze LVLMs power on mathematical reasoning; their responses on our puzzles offer a direct way to compare against that of children. Our results show that modern LVLMs do demonstrate increasingly powerful reasoning skills in solving problems for higher grades, but lack the foundations to correctly answer problems designed for younger children. Further analysis shows that there is no significant correlation between the reasoning capabilities of AI models and that of young children, and their capabilities appear to be based on a different type of reasoning than the cumulative knowledge that underlies children's mathematical skills.
     </details>

33. **Instance-adaptive Zero-shot Chain-of-Thought Prompting** [[pdf]](http://arxiv.org/abs/2409.20441) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces an instance-adaptive prompting algorithm as an alternative zero-shot CoT reasoning scheme by adaptively differentiating good and bad prompts, and proposes an instance-adaptive prompting strategy (IAP) for zero-shot CoT reasoning.
     </details>


     <details>
          <summary>Abstract</summary>
          Zero-shot Chain-of-Thought (CoT) prompting emerges as a simple and effective strategy for enhancing the performance of large language models (LLMs) in real-world reasoning tasks. Nonetheless, the efficacy of a singular, task-level prompt uniformly applied across the whole of instances is inherently limited since one prompt cannot be a good partner for all, a more appropriate approach should consider the interaction between the prompt and each instance meticulously. This work introduces an instance-adaptive prompting algorithm as an alternative zero-shot CoT reasoning scheme by adaptively differentiating good and bad prompts. Concretely, we first employ analysis on LLMs through the lens of information flow to detect the mechanism under zero-shot CoT reasoning, in which we discover that information flows from question to prompt and question to rationale jointly influence the reasoning results most. We notice that a better zero-shot CoT reasoning needs the prompt to obtain semantic information from the question then the rationale aggregates sufficient information from the question directly and via the prompt indirectly. On the contrary, lacking any of those would probably lead to a bad one. Stem from that, we further propose an instance-adaptive prompting strategy (IAP) for zero-shot CoT reasoning. Experiments conducted with LLaMA-2, LLaMA-3, and Qwen on math, logic, and commonsense reasoning tasks (e.g., GSM8K, MMLU, Causal Judgement) obtain consistent improvement, demonstrating that the instance-adaptive zero-shot CoT prompting performs better than other task-level methods with some curated prompts or sophisticated procedures, showing the significance of our findings in the zero-shot CoT reasoning mechanism.
     </details>

34. **Learning Goal-Conditioned Representations in Reward Models for Aligning Language Models** [[pdf]](https://neurips.cc/virtual/2024/poster/95067) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Representation learning is important for the success of Reinforcement Learning (RL) algorithms, but has been less explored for Language Model (LM) alignment with Reinforcement learning from Human Feedback (RLHF).In this work, we present a simple yet effective approach to improve the representations learned by reward models for aligning LMs.Our method uses a contrastive loss that encourages reward models to learn goal-conditioned representations which encode the expected reward at intermediate steps of the input sequence.By enforcing this loss on representations from intermediate steps, we can capture which trajectories are likely to reach a desired goal (e.g., correct solution or helpful response) at different points in the sequence.This method is flexible enough to support different kinds of alignment data and does not require extra annotations.We demonstrate the effectiveness of this approach in 2 domains: mathematical reasoning and natural language alignment.On math benchmarks, such as GSM8k, we show that our approach improves the reward model's ability to discern between correct/incorrect solutions, increasing AUROC score by up to 0.11 points, and that the learned representations can help prune undesirable generations.Using this reward model to improve a policy model via RLHF yields accuracy gains of 1.7\% across several math benchmarks when compared to a standard preference-ranking trained reward model.Additionally, we show the that learned representations can be used to steer LMs toward generations that are more aligned with human preferences via guided decoding.Overall, our study underscores the potential of incorporating feedback signals in RLHF frameworks via learned representations, which we believe is a promising avenue for improving the alignment of LLMs.
     </details>

35. **MathPile: A Billion-Token-Scale Pretraining Corpus for Math** [[pdf]](https://neurips.cc/virtual/2024/poster/97685) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          High-quality, large-scale corpora are the cornerstone of building foundation models. In this work, we introduce MathPile, a diverse and high-quality math-centric corpus comprising about 9.5 billion tokens. Throughout its creation, we adhered to the principle of “less is more”, firmly believing in the supremacy of data quality over quantity, even in the pre-training phase. Our meticulous data collection and processing efforts included a complex suite of preprocessing, prefiltering, language identification, cleaning, filtering, and deduplication, ensuring the high quality of our corpus. Furthermore, we performed data contamination detection on downstream benchmark test sets to eliminate duplicates and conducted continual pre-training experiments, booting the performance on common mathematical reasoning benchmarks. We aim for our MathPile to boost language models’ mathematical reasoning and plan to open-source its different versions and processing scripts to advance the field.
     </details>

36. **Multi-language Diversity Benefits Autoformalization** [[pdf]](https://neurips.cc/virtual/2024/poster/96799) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Autoformalization is the task of translating natural language materials into machine-verifiable formalisations. Progress in autoformalization research is hindered by the lack of a sizeable dataset consisting of informal-formal pairs expressing the same essence. Existing methods tend to circumvent this challenge by manually curating small corpora or using few-shot learning with large language models. But these methods suffer from data scarcity and formal language acquisition difficulty. In this work, we create mma, a large, flexible, multi-language, and multi-domain dataset of informal-formal pairs, by using a language model to translate in the reverse direction, that is, from formal mathematical statements into corresponding informal ones. Experiments show that language models fine-tuned on mma can produce up to $29-31$\% of statements acceptable with minimal corrections on the miniF2F and ProofNet benchmarks, up from $0$\% with the base model. We demonstrate that fine-tuning on multi-language formal data results in more capable autoformalization models even on single-language tasks.
     </details>

37. **Neuro-Symbolic Data Generation for Math Reasoning** [[pdf]](https://neurips.cc/virtual/2024/poster/96151) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          A critical question about Large Language Models (LLMs) is whether their apparent deficiency in mathematical reasoning is inherent, or merely a result of insufficient exposure to high-quality mathematical data. To explore this, we developed an automated method for generating high-quality, supervised mathematical datasets. The method carefully mutates existing math problems, ensuring both diversity and validity of the newly generated problems. This is achieved by a neuro-symbolic data generation framework combining the intuitive informalization strengths of LLMs, and the precise symbolic reasoning of math solvers along with projected Markov chain Monte Carlo sampling in the highly-irregular symbolic space.Empirical experiments demonstrate the high quality of data generated by the proposed method, and that the LLMs, specifically LLaMA-2 and Mistral, when realigned with the generated data, surpass their state-of-the-art counterparts.
     </details>

38. **Not All Tokens Are What You Need for Pretraining** [[pdf]](https://neurips.cc/virtual/2024/poster/96931) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Previous language model pre-training methods have uniformly applied a next-token prediction loss to all training tokens. Challenging this norm, we posit that ''Not all tokens in a corpus are equally important for language model training''. Our initial analysis examines token-level training dynamics of language model, revealing distinct loss patterns for different tokens. Leveraging these insights, we introduce a new language model called Rho-1. Unlike traditional LMs that learn to predict every next token in a corpus, Rho-1 employs Selective Language Modeling (SLM), which selectively trains on useful tokens that aligned with the desired distribution. This approach involves scoring pretraining tokens using a reference model, and then training the language model with a focused loss on tokens with higher scores. When continual pretraining on 15B OpenWebMath corpus, Rho-1 yields an absolute improvement in few-shot accuracy of up to 30% in 9 math tasks. After fine-tuning, Rho-1-1B and 7B achieved state-of-the-art results of 40.6% and 51.8% on MATH dataset, respectively - matching DeepSeekMath with only 3% of the pretraining tokens. Furthermore, when pretraining on 80B general tokens, Rho-1 achieves 6.8% average enhancement across 15 diverse tasks, increasing both efficiency and performance of the language model pre-training.
     </details>

39. **OccamLLM: Fast and Exact Language Model Arithmetic in a Single Step** [[pdf]](http://arxiv.org/abs/2406.06576) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a framework that enables exact arithmetic in a single autoregressive step, providing faster, more secure, and more interpretable LLM systems with arithmetic capabilities.
     </details>


     <details>
          <summary>Abstract</summary>
          Despite significant advancements in text generation and reasoning, Large Language Models (LLMs) still face challenges in accurately performing complex arithmetic operations. To achieve accurate calculations, language model systems often enable LLMs to generate code for arithmetic operations. However, this approach compromises speed and security and, if finetuning is involved, risks the language model losing prior capabilities. We propose a framework that enables exact arithmetic in a single autoregressive step, providing faster, more secure, and more interpretable LLM systems with arithmetic capabilities. We use the hidden states of an LLM to control a symbolic architecture which performs arithmetic. Our implementation using Llama 3 8B Instruct with OccamNet as a symbolic model (OccamLlama) achieves 100% accuracy on single arithmetic operations (+, -, *, /, sin, cos, log, exp, sqrt), outperforming GPT 4o and on par with GPT 4o using a code interpreter. OccamLlama also outperforms both Llama 3 8B Instruct and GPT 3.5 Turbo on multistep reasoning problems involving challenging arithmetic, thus enabling small LLMs to match the arithmetic performance of even much larger models. Our code is available at https://anonymous.4open.science/r/OccamLlama.
     </details>

40. **On the Inductive Bias of Stacking Towards Improving Reasoning** [[pdf]](http://arxiv.org/abs/2409.19044) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          An intriguing phenomenon is discovered: MIDAS is not only training-efficient but surprisingly also has an inductive bias towards improving downstream tasks, especially tasks that require reasoning abilities like reading comprehension and math problems, despite having similar or slightly worse perplexity compared to baseline training.
     </details>


     <details>
          <summary>Abstract</summary>
          Given the increasing scale of model sizes, novel training strategies like gradual stacking have garnered interest. Stacking enables efficient training by gradually growing the depth of a model in stages and using layers from a smaller model in an earlier stage to initialize the next stage. Although efficient for training, the model biases induced by such growing approaches is largely unexplored. In this work, we examine this fundamental aspect of gradual stacking, going beyond its efficiency benefits. We propose a variant of gradual stacking called MIDAS and discover an intriguing phenomenon for this approach: MIDAS is not only training efficient, but surprisingly also has an inductive bias towards improving downstream tasks, especially tasks that require reasoning abilities, despite having similar or slightly worse perplexity compared to baseline training. To further analyze this inductive bias, we construct {\em reasoning primitives} – simple synthetic tasks that are building blocks for reasoning – and find that a model pretrained with stacking is significantly better than standard pretraining on these primitives, with and without fine-tuning. This provides stronger and more robust evidence for this inductive bias towards reasoning. Furthermore, we conjecture the underlying reason for this inductive bias by exploring the connection of stacking to looped models and provide strong supporting empirical analysis.
     </details>

41. **Pretrained Large Language Models Use Fourier Features to Compute Addition** [[pdf]](https://neurips.cc/virtual/2024/poster/94033) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Pre-trained large language models (LLMs) exhibit impressive mathematical reasoning capabilities, yet how they compute basic arithmetic, such as addition, remains unclear. This paper shows that pre-trained LLMs add numbers using Fourier features---dimensions in the hidden state that represent numbers via a set of features sparse in the frequency domain. Within the model, MLP and attention layers use Fourier features in complementary ways: MLP layers primarily approximate the magnitude of the answer using low-frequency features, while attention layers primarily perform modular addition (e.g., computing whether the answer is even or odd) using high-frequency features.Pre-training is crucial for this mechanism: models trained from scratch to add numbers only exploit low-frequency features, leading to lower accuracy.Introducing pre-trained token embeddings to a randomly initialized model rescues its performance.Overall, our analysis demonstrates that appropriate pre-trained representations (e.g., Fourier features) can unlock the ability of Transformers to learn precise mechanisms for algorithmic tasks.
     </details>

42. **Recursive Introspection: Teaching Foundation Model Agents How to Self-Improve** [[pdf]](https://neurips.cc/virtual/2024/poster/96089) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          A central piece in enabling intelligent agentic behavior in foundation models is to make them capable of introspecting upon their behavior, to reason and correct their mistakes. Even strong proprietary large language models (LLMs) do not exhibit the ability of continually improving their responses sequentially, even in scenarios where they are explicitly told that they are making a mistake. In this paper, we develop $\textbf{RISE}$: $\textbf{R}$ecursive $\textbf{I}$ntro$\textbf{s}$p$\textbf{e}$ction, an approach for fine-tuning LLMs to introduce this ability. Our approach prescribes an iterative fine-tuning procedure, which attempts to teach the model how to alter its response after having seen previously unsuccessful attempts to solve a problem with additional environment feedback. RISE poses fine-tuning for a single-turn problem as solving a multi-turn Markov decision process (MDP), where the initial state is the prompt. Inspired by principles in online imitation learning, we derive effective strategies to dictate multi-turn data collection and training so as to imbue in an LLM the capability to recursively detect and correct its previous mistakes in subsequent iterations. Our experiments show that $\textbf{RISE}$ enables 7B Llama2 and Mistral models to improve themselves with more turns on math reasoning tasks, outperforming several single-turn strategies given an equal amount of inference-time computation. Our analysis shows that RISE makes meaningful improvements to responses to arrive at the correct solution for challenging prompts, without disrupting one-turn abilities.
     </details>

43. **SciInstruct: a Self-Reflective Instruction Annotated Dataset for Training Scientific Language Models** [[pdf]](https://neurips.cc/virtual/2024/poster/97744) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have shown promise in assisting scientific discovery. However, such applications are currently limited by LLMs' deficiencies in understanding intricate scientific concepts, deriving symbolic equations, and solving advanced numerical calculations. To bridge these gaps, we introduce SciInstruct, a suite of scientific instructions for training scientific language models capable of college-level scientific reasoning. Central to our approach is a novel self-reflective instruction annotation framework to address the data scarcity challenge in the science domain. This framework leverages existing LLMs to generate step-by-step reasoning for unlabelled scientific questions, followed by a process of self-reflective critic-and-revise. Applying this framework, we curated a diverse and high-quality dataset encompassing physics, chemistry, math, and formal proofs. We analyze the curated SciInstruct from multiple interesting perspectives (e.g., domain, scale, source, question type, answer length, etc.). To verify the effectiveness of SciInstruct, we fine-tuned different language models with SciInstruct, i.e., ChatGLM3 (6B and 32B), Llama3-8b-Instruct, and Mistral-7B, enhancing their scientific and mathematical reasoning capabilities, without sacrificing the language understanding capabilities of the base model. We release code and SciInstruct at https://github.com/THUDM/SciGLM.
     </details>

44. **Solving Intricate Problems with Human-like Decomposition and Rethinking** [[pdf]](https://neurips.cc/virtual/2024/poster/95441) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          In this paper, we introduce a novel reasoning framework DeAR (Decompose-Analyze-Rethink) for large language models (LLMs) to conduct intricate reasoning. Our key idea is inspired by human cognitive reasoning, which refines complex problem-solving by breaking it down into sub-questions within a Reasoning Tree and then updating prior answers based on the responses to these sub-questions. In our framework, we propose a Decompose-Analyze-Rethink cycle, which gradually forms a reasoning tree guiding the reasoning process. Specifically, given the problem, the Decompose stage introduces a prompt-based method to break it into simpler sub-ones at subsequent tree nodes. Then, the Analyze stage generates and self-checks the rationales at the node level. Last, the Rethink stage updates the rationales of parent nodes based on its children's feedback. Our reasoning paradigm is more flexible than state-of-the-art methods including Tree-of-Thoughts (ToT), and Graph-of-Thoughts (GoT), as each branch is autonomously generated without fixed settings, and moreover, allows for timely and globally rationale correction throughout the entire process. We conduct extensive experiments on three reasoning benchmarks including ScienceQA, StrategyQA, and GSM8K. Experimental results show that our approach can significantly reduce logical errors and enhance the performance with different LLMs. Our codes are available at: https://anonymous.4open.science/r/Coarse-to-Fine-F216/.
     </details>

45. **Unsupervised Discovery of Formulas for Mathematical Constants** [[pdf]](https://neurips.cc/virtual/2024/poster/95491) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          In recent years, we are witnessing a rise of AI and machine learning methods for scientific discovery and hypothesis creation. Despite the strides in other fields of science, a persistent challenge lies in the creation of formulas for mathematical constants.In the landscape of formula creation, there is no straightforward ‘’distance metric'' between two samples that can guide progress. Formulas are either true or false, with no continuous adjustments that can enhance their correctness.The absence of a systematic method left the realm of formula discovery elusive for automated methods. In this work, we propose a systematic methodology for categorization, characterization, and pattern identification of such formulas. We demonstrate this methodology on Polynomial Continued Fraction formulas, which are ubiquitous in their intrinsic connections to mathematical constants, and generalize many mathematical functions and structures.We discover organizing metrics for the space of polynomial continued fractions. We test our methodology on a set of 1,768,900 such formulas, identifying many known formulas for mathematical constants, and discover previously unknown formulas for $\pi$, $\ln(2)$, Gauss, and Lemniscate constants. The uncovered patterns enable a direct generalization of individual formulas to infinite families, unveiling rich mathematical structures. This success paves the way towards a generative model that creates continued fractions fulfilling requested mathematical properties, potentially accelerating by orders of magnitude the rate of discovery of useful formulas.
     </details>

46. **When and How Does Synthetic Data Improve Reasoning Capabilities of Language Models?** [[pdf]](https://neurips.cc/virtual/2024/poster/96295) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Training on model-generated synthetic data is a promising approach for finetuning LLMs, but it remains unclear when it helps or hurts. In this paper, we investigate this for reasoning problems via an empirical study, followed by a theoretical formalization of our observations. First, we find that while the typical approach of finetuning a model on synthetic correct or positive problem-solution pairs generated by capable models offers modest performance gains, sampling more correct solutions from the finetuned learner doubles the sample efficiency of synthetic data. At the same time, training on model-generated positives can amplify various spurious  correlations, resulting in flat or even inverse scaling trends as the amount of data increases. Surprisingly, we find that several of these issues can be addressed if we also utilize negative responses, i.e. model-generated responses that are deemed incorrect via final answer checking. Crucially, these negatives must be constructed such that the training can appropriately recover the utility or credit of each intermediate step in the negative response. With this per-step scheme, we are able to attain consistent gains over only positive data, attaining performance similar to amplifying the amount of synthetic data by 8x. We show that training on per-step negatives can help to unlearn spurious correlations in the positive data, and is equivalent to advantage-weighted reinforcement learning (RL), implying that it inherits benefits of RL over imitating positive data alone.
     </details>

