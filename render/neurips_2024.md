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

2. **OpenMathInstruct-1: A 1.8 Million Math Instruction Tuning Dataset** [[pdf]](http://arxiv.org/abs/2402.10176) `NeurIPS 2024` (42 cite) (5 AI4Math cite) 


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

10. **ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search** [[pdf]](http://arxiv.org/abs/2406.03816) `NeurIPS 2024` (12 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A reinforced self-training approach, called ReST-MCTS*, based on integrating process reward guidance with tree search MCTS* for collecting higher-quality reasoning traces as well as per-step value to train policy and reward models, which achieves higher accuracy and can continuously enhance the three language models for multiple iterations.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent methodologies in LLM self-training mostly rely on LLM generating responses and filtering those with correct output answers as training data. This approach often yields a low-quality fine-tuning training set (e.g., incorrect plans or intermediate reasoning). In this paper, we develop a reinforced self-training approach, called ReST-MCTS*, based on integrating process reward guidance with tree search MCTS* for collecting higher-quality reasoning traces as well as per-step value to train policy and reward models. ReST-MCTS* circumvents the per-step manual annotation typically used to train process rewards by tree-search-based reinforcement learning: Given oracle final correct answers, ReST-MCTS* is able to infer the correct process rewards by estimating the probability this step can help lead to the correct answer. These inferred rewards serve dual purposes: they act as value targets for further refining the process reward model and also facilitate the selection of high-quality traces for policy model self-training. We first show that the tree-search policy in ReST-MCTS* achieves higher accuracy compared with prior LLM reasoning baselines such as Best-of-N and Tree-of-Thought, within the same search budget. We then show that by using traces searched by this tree-search policy as training data, we can continuously enhance the three language models for multiple iterations, and outperform other self-training algorithms such as ReST$^\text{EM}$ and Self-Rewarding LM.
     </details>

11. **Transformers Can Do Arithmetic with the Right Embeddings** [[pdf]](http://arxiv.org/abs/2405.17399) `NeurIPS 2024` (12 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work fixes the problem of poor performance of transformers on arithmetic tasks by adding an embedding to each digit that encodes its position relative to the start of the number, and shows that this fix enables architectural modifications such as input injection and recurrent layers to improve performance.
     </details>


     <details>
          <summary>Abstract</summary>
          The poor performance of transformers on arithmetic tasks seems to stem in large part from their inability to keep track of the exact position of each digit inside of a large span of digits. We mend this problem by adding an embedding to each digit that encodes its position relative to the start of the number. In addition to the boost these embeddings provide on their own, we show that this fix enables architectural modifications such as input injection and recurrent layers to improve performance even further.With positions resolved, we can study the logical extrapolation ability of transformers. Can they solve arithmetic problems that are larger and more complex than those in their training data? We find that training on only 20 digit numbers with a single GPU for one day, we can reach state-of-the-art performance, achieving up to 99% accuracy on 100 digit addition problems. Finally, we show that these gains in numeracy also unlock improvements on other multi-step reasoning tasks including sorting and multiplication.
     </details>

12. **JiuZhang3.0: Efficiently Improving Mathematical Reasoning by Training Small Data Synthesis Models** [[pdf]](http://arxiv.org/abs/2405.14365) `NeurIPS 2024` (10 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes an efficient way that trains a small LLM for math problem synthesis, to efficiently generate sufficient high-quality pre-training data, and craft a set of prompts based on human education stages to guide GPT-4, to synthesize problems covering diverse math knowledge and difficulty levels.
     </details>


     <details>
          <summary>Abstract</summary>
          Mathematical reasoning is an important capability of large language models~(LLMs) for real-world applications.To enhance this capability, existing work either collects large-scale math-related texts for pre-training, or relies on stronger LLMs (\eg GPT-4) to synthesize massive math problems. Both types of work generally lead to large costs in training or synthesis.To reduce the cost, based on open-source available texts, we propose an efficient way that trains a small LLM for math problem synthesis, to efficiently generate sufficient high-quality pre-training data.To achieve it, we create a dataset using GPT-4 to distill its data synthesis capability into the small LLM.Concretely, we craft a set of prompts based on human education stages to guide GPT-4, to synthesize problems covering diverse math knowledge and difficulty levels.Besides, we adopt the gradient-based influence estimation method to select the most valuable math-related texts.The both are fed into GPT-4 for creating the knowledge distillation dataset to train the small LLM.We leverage it to synthesize 6 million math problems for pre-training our JiuZhang3.0 model, which only needs to invoke GPT-4 API 9.3k times and pre-train on 4.6B data.Experimental results have shown that JiuZhang3.0 achieves state-of-the-art performance on several mathematical reasoning datasets, under both natural language reasoning and tool manipulation settings.
     </details>

13. **Generative Verifiers: Reward Modeling as Next-Token Prediction** [[pdf]](http://arxiv.org/abs/2408.15240) `NeurIPS 2024 Workshop MATH-AI` (8 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is demonstrated that when using Gemma-based verifiers on algorithmic and grade-school math reasoning tasks, GenRM outperforms discriminative verifiers and LLM-as-a-Judge, and it is shown that GenRM scales favorably across dataset size, model capacity, and inference-time compute.
     </details>


     <details>
          <summary>Abstract</summary>
          Verifiers or reward models are often used to enhance the reasoning performance of large language models (LLMs). A common approach is the Best-of-N method, where N candidate solutions generated by the LLM are ranked by a verifier, and the best one is selected. While LLM-based verifiers are typically trained as discriminative classifiers to score solutions, they do not utilize the text generation capabilities of pretrained LLMs. To overcome this limitation, we instead propose training verifiers using the ubiquitous next-token prediction objective, jointly on verification and solution generation. Compared to standard verifiers, such generative verifiers (GenRM) can benefit from several advantages of LLMs: they integrate seamlessly with instruction tuning, enable chain-of-thought reasoning, and can utilize additional test-time compute via majority voting for better verification. We demonstrate that GenRM outperforms discriminative, DPO verifiers, and LLM-as-a-Judge, resulting in a 16-40% improvement in the number of problems solved with Best-of-N on algorithmic and math reasoning tasks. Furthermore, we find that training GenRM with synthetic verification rationales is sufficient to pick out subtle errors on math problems. Finally, we demonstrate that generative verifiers scale favorably with model size and inference-time compute.
     </details>

14. **DART-Math: Difficulty-Aware Rejection Tuning for Mathematical Problem-Solving** [[pdf]](http://arxiv.org/abs/2407.13690) `NeurIPS 2024` (5 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Hypothesizing that difficult queries are crucial to learn complex reasoning, this work proposes Difficulty-Aware Rejection Tuning (DART), a method that allocates difficult queries more trials during the synthesis phase, enabling more extensive training on difficult samples.
     </details>


     <details>
          <summary>Abstract</summary>
          Solving mathematical problems requires advanced reasoning abilities and presents notable challenges for large language models. Previous works usually synthesize data from proprietary models to augment existing datasets, followed by instruction tuning to achieve top-tier results. However, our analysis of these datasets reveals severe biases towards easy queries, with frequent failures to generate any correct response for the most challenging queries.Hypothesizing that difficult queries are crucial to learn complex reasoning, we propose Difficulty-Aware Rejection Tuning (DART), a method that allocates difficult queries more trials during the synthesis phase, enabling more extensive training on difficult samples.Utilizing DART, we have created new datasets for mathematical problem-solving that focus more on difficult queries and are substantially smaller than previous ones. Remarkably, our synthesis process solely relies on a 7B-sized open-weight model, without reliance on the commonly used proprietary GPT-4.We fine-tune various base models on our datasets ranging from 7B to 70B in size, resulting in a series of strong models called DART-Math.In comprehensive in-domain and out-of-domain evaluation on 6 mathematical benchmarks, DART-Math outperforms vanilla rejection tuning significantly, being superior or comparable to previous arts, despite using much smaller datasets and no proprietary models. Furthermore, our results position our synthetic datasets as the most effective and cost-efficient publicly available resources for advancing mathematical problem-solving. Our datasets and models will be made publicly available following the review period.
     </details>

15. **Metacognitive Capabilities of LLMs: An Exploration in Mathematical Problem Solving** [[pdf]](https://openreview.net/forum?id=0MsI3bSmmD&name=pdf) `NeurIPS 2024` (5 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          \emph{Metacognitive knowledge} refers to humans' intuitive knowledge of their own thinking and reasoning processes. Today's best LLMs clearly possess some reasoning processes. The paper gives evidence that they also  have metacognitive knowledge, including ability to name skills and procedures to apply given a task. We explore this primarily in context of math reasoning, developing a prompt-guided interaction procedure  to get a powerful  LLM to assign sensible skill labels to math questions, followed by having it perform semantic clustering to obtain coarser families of skill labels. These coarse skill labels look interpretable to humans.To validate that these skill labels are meaningful and relevant to the LLM's reasoning processes we perform the following experiments. (a) We ask GPT-4 to assign skill labels to training questions in math datasets GSM8K and MATH.  (b) When using an LLM to solve the test questions, we present it with the full list of skill labels and ask it to identify the skill needed. Then it is presented with randomly selected exemplar solved questions associated with that skill label.  This improves accuracy on GSM8k and MATH for several strong LLMs, including code-assisted models. The methodology presented is domain-agnostic,  even though this article applies it to math problems.
     </details>

16. **Lean Workbook: A large-scale Lean problem set formalized from natural language math problems** [[pdf]](http://arxiv.org/abs/2406.03847) `NeurIPS 2024` `Lean` (4 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a novel pipeline that iteratively generates and filters synthetic data to translate natural language mathematical problems into Lean 4 statements, and vice versa, and indicates that the synthetic data pipeline can provide useful training data and improve the performance of LLMs in translating and understanding complex mathematical problems and proofs.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models have demonstrated impressive capabilities across various natural language processing tasks, especially in solving mathematical problems. However, large language models are not good at math theorem proving using formal languages like Lean. A significant challenge in this area is the scarcity of training data available in these formal languages. To address this issue, we propose a novel pipeline that iteratively generates and filters synthetic data to translate natural language mathematical problems into Lean 4 statements, and vice versa. Our results indicate that the synthetic data pipeline can provide useful training data and improve the performance of LLMs in translating and understanding complex mathematical problems and proofs. Our final dataset contains about 57K formal-informal question pairs along with searched proof from the math contest forum and 21 new IMO questions. We open-source our code at \url{https://github.com/InternLM/InternLM-Math} and our data at \url{https://huggingface.co/datasets/InternLM/Lean-Workbook}.
     </details>

17. **AI-Assisted Generation of Difficult Math Questions** [[pdf]](http://arxiv.org/abs/2407.21009) `NeurIPS 2024 Workshop MATH-AI` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A design framework that combines the strengths of LLMs with a human-in-the-loop approach to generate a diverse array of challenging math questions and a striking relationship observed between models' performance on the new dataset, suggesting that successfully solving the question in MATH$^2$ requires a nontrivial combination of two distinct math skills.
     </details>


     <details>
          <summary>Abstract</summary>
          Current LLM training positions mathematical reasoning as a core capability. With publicly available sources fully tapped, there is unmet demand for diverse and challenging math questions. Relying solely on human experts is both time-consuming and costly, while LLM-generated questions often lack the requisite diversity and difficulty. We present a design framework that combines the strengths of LLMs with a human-in-the-loop approach to generate a diverse array of challenging math questions. We leverage LLM metacognition skills [Didolkar et al., 2024] of a strong LLM to extract core "skills" from existing math datasets. These skills serve as the basis for generating novel and difficult questions by prompting the LLM with random pairs of core skills. The use of two different skills within each question makes finding such questions an "out of distribution" task for both LLMs and humans. Our pipeline employs LLMs to iteratively generate and refine questions and solutions through multiturn prompting. Human annotators then verify and further refine the questions, with their efficiency enhanced via further LLM interactions. Applying this pipeline on skills extracted from the MATH dataset [Hendrycks et al., 2021] resulted in MATH$^2$ - a dataset of higher-quality math questions, as evidenced by: (a) Lower performance of all models on MATH$^2$ than on MATH (b) Higher performance on MATH when using MATH$^2$ questions as in-context examples. Although focused on mathematics, our methodology seems applicable to other domains requiring structured reasoning, and potentially as a component of scalable oversight. Also of interest is a striking relationship observed between models' performance on the new dataset: the success rate on MATH$^2$ is the square on MATH, suggesting that successfully solving the question in MATH$^2$ requires a nontrivial combination of two distinct math skills.
     </details>

18. **Lean-STaR: Learning to Interleave Thinking and Proving** [[pdf]](http://arxiv.org/abs/2407.10040) `NeurIPS 2024 Workshop MATH-AI` `Lean` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work presents Lean-STaR, a framework for training language models to produce informal thoughts prior to each step of a proof, thereby boosting the model's theorem-proving capabilities.
     </details>


     <details>
          <summary>Abstract</summary>
          Traditional language model-based theorem proving assumes that by training on a sufficient amount of formal proof data, a model will learn to prove theorems. Our key observation is that a wealth of informal information that is not present in formal proofs can be useful for learning to prove theorems. For instance, humans think through steps of a proof, but this thought process is not visible in the resulting code. We present Lean-STaR, a framework for training language models to produce informal thoughts prior to each step of a proof, thereby boosting the model's theorem-proving capabilities. Lean-STaR uses retrospective ground-truth tactics to generate synthetic thoughts for training the language model. At inference time, the trained model directly generates the thoughts prior to the prediction of the tactics in each proof step. Building on the self-taught reasoner framework, we then apply expert iteration to further fine-tune the model on the correct proofs it samples and verifies using the Lean solver. Lean-STaR achieves state-of-the-art results on the miniF2F-test benchmark within the Lean theorem proving environment, significantly outperforming base models ($\boldsymbol{43.4\% \rightarrow 46.3\%,}$ Pass@64). We also analyze the impact of the augmented thoughts on various aspects of the theorem proving process, providing insights into their effectiveness.
     </details>

19. **OlympicArena: Benchmarking Multi-discipline Cognitive Reasoning for Superintelligent AI** [[pdf]](https://arxiv.org/abs/2406.12753) `NeurIPS 2024` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work argues that the challenges in Olympic competition problems are ideal for evaluating AI's cognitive reasoning due to their complexity and interdisciplinary nature, which are essential for tackling complex scientific challenges and facilitating discoveries.
     </details>


     <details>
          <summary>Abstract</summary>
          The evolution of Artificial Intelligence (AI) has been significantly accelerated by advancements in Large Language Models (LLMs) and Large Multimodal Models (LMMs), gradually showcasing potential cognitive reasoning abilities in problem-solving and scientific discovery (i.e., AI4Science) once exclusive to human intellect. To comprehensively evaluate current models' performance in cognitive reasoning abilities, we introduce OlympicArena, which includes 11,163 bilingual problems across both text-only and interleaved text-image modalities. These challenges encompass a wide range of disciplines spanning seven fields and 62 international Olympic competitions, rigorously examined for data leakage. We argue that the challenges in Olympic competition problems are ideal for evaluating AI's cognitive reasoning due to their complexity and interdisciplinary nature, which are essential for tackling complex scientific challenges and facilitating discoveries. Beyond evaluating performance across various disciplines using answer-only criteria, we conduct detailed experiments and analyses from multiple perspectives. We delve into the models' cognitive reasoning abilities, their performance across different modalities, and their outcomes in process-level evaluations, which are vital for tasks requiring complex reasoning with lengthy solutions. Our extensive evaluations reveal that even advanced models like GPT-4o only achieve a 39.97\%  overall accuracy (28.67\%  for mathematics and 29.71\%  for physics), illustrating current AI limitations in complex reasoning and multimodal integration. Through the OlympicArena, we aim to advance AI towards superintelligence, equipping it to address more complex challenges in science and beyond. We also provide a comprehensive set of resources to support AI research, including a benchmark dataset, an open-source annotation platform, a detailed evaluation tool, and a leaderboard with automatic submission features.
     </details>

20. **StrategyLLM: Large Language Models as Strategy Generators, Executors, Optimizers, and Evaluators for Problem Solving** [[pdf]](https://arxiv.org/abs/2311.08803) `NeurIPS 2024` (3 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A comprehensive framework, StrategyLLM, allowing LLMs to perform inductive reasoning, deriving general strategies from specific task instances, and deductive reasoning, applying these general strategies to particular task examples, for constructing generalizable and consistent few-shot prompts.
     </details>


     <details>
          <summary>Abstract</summary>
          Most existing prompting methods suffer from the issues of generalizability and consistency, as they often rely on instance-specific solutions that may not be applicable to other instances and lack task-level consistency across the selected few-shot examples. To address these limitations, we propose a comprehensive framework, StrategyLLM, allowing LLMs to perform inductive reasoning, deriving general strategies from specific task instances, and deductive reasoning, applying these general strategies to particular task examples, for constructing generalizable and consistent few-shot prompts. It employs four LLM-based agents: strategy generator, executor, optimizer, and evaluator, working together to generate, evaluate, and select promising strategies for a given task. Experimental results demonstrate that StrategyLLM outperforms the competitive baseline CoT-SC that requires human-annotated solutions on 13 datasets across 4 challenging tasks without human involvement, including math reasoning (34.2\% $\rightarrow$ 38.8\%), commonsense reasoning (70.3\% $\rightarrow$ 72.5\%), algorithmic reasoning (73.7\% $\rightarrow$ 85.0\%), and symbolic reasoning (30.0\% $\rightarrow$ 79.2\%). Further analysis reveals that StrategyLLM is applicable to various LLMs and demonstrates advantages across numerous scenarios.
     </details>

21. **Visual Sketchpad: Sketching as a Visual Chain of Thought for Multimodal Language Models** [[pdf]](http://arxiv.org/abs/2406.09403) `NeurIPS 2024` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Sketchpad enables LMs to draw with lines, boxes, marks, etc., which is closer to human sketching and better facilitates reasoning, and sets a new state of the art on all tasks, including V*Bench and BLINK spatial reasoning.
     </details>


     <details>
          <summary>Abstract</summary>
          Humans draw to facilitate reasoning: we draw auxiliary lines when solving geometry problems; we mark and circle when reasoning on maps; we use sketches to amplify our ideas and relieve our limited-capacity working memory. However, such actions are missing in current multimodal language models (LMs). Current chain-of-thought and tool-use paradigms only use text as intermediate reasoning steps. In this work, we introduce Sketchpad, a framework that gives multimodal LMs a visual sketchpad and tools to draw on the sketchpad. The LM conducts planning and reasoning according to the visual artifacts it has drawn. Different from prior work, which uses text-to-image models to enable LMs to draw, Sketchpad enables LMs to draw with lines, boxes, marks, etc., which is closer to human sketching and better facilitates reasoning. \name can also use specialist vision models during the sketching process (e.g., draw bounding boxes with object detection models, draw masks with segmentation models), to further enhance visual perception and reasoning. We experiment on a wide range of math tasks (including geometry, functions, graph, chess) and complex visual reasoning tasks. Sketchpad substantially improves performance on all tasks over strong base models with no sketching, yielding an average gain of 12.7% on math tasks, and 8.6% on vision tasks. GPT-4o with Sketchpad sets a new state of the art on all tasks, including V*Bench (80.3%), BLINK spatial reasoning (83.9%), and visual correspondence (80.8%). We will release all code and data.
     </details>

22. **Calibrating Reasoning in Language Models with Internal Consistency** [[pdf]](https://arxiv.org/abs/2405.18711) `NeurIPS 2024` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The results demonstrate the potential of using internal representations for self-evaluation of LLMs by up-weighting reasoning paths with high internal consistency, resulting in a significant boost in reasoning performance.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have demonstrated impressive capabilities in various reasoning tasks, aided by techniques like chain-of-thought (CoT) prompting that elicits verbalized reasoning. However, LLMs often generate text with obvious mistakes and contradictions, raising doubts about their ability to robustly process and utilize generated rationales. In this work, we investigate CoT reasoning in LLMs through the lens of internal representations, focusing on how these representations are influenced by generated rationales. Our preliminary analysis reveals that while generated rationales improve answer accuracy, inconsistencies emerge between the model's internal representations in middle layers and those in final layers, potentially undermining the reliability of their reasoning processes. To address this, we propose internal consistency as a measure of the model's confidence by examining the agreement of latent predictions decoded from intermediate layers. Extensive empirical studies across different models and datasets demonstrate that internal consistency effectively distinguishes between correct and incorrect reasoning paths. Motivated by this, we propose a new approach to calibrate CoT reasoning by up-weighting reasoning paths with high internal consistency, resulting in a significant boost in reasoning performance. Further analysis uncovers distinct patterns in attention and feed-forward modules across layers, providing insights into the emergence of internal inconsistency. In summary, our results demonstrate the potential of using internal representations for self-evaluation of LLMs.
     </details>

23. **How Far Can Transformers Reason? The Locality Barrier and Inductive Scratchpad** [[pdf]](http://arxiv.org/abs/2406.06467) `NeurIPS 2024` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The notion of 'distribution locality' is put forward to capture when weak learning is efficiently achievable by regular Transformers, where the locality measures the least number of tokens required in addition to the tokens histogram to correlate nontrivially with the target.
     </details>


     <details>
          <summary>Abstract</summary>
          Can Transformers predict new syllogisms by composing established ones? More generally, what type of targets can be learned by such models from scratch? Recent works show that Transformers can be Turing-complete in terms of expressivity, but this does not address the learnability objective. This paper puts forward the notion of 'distribution locality' to capture when weak learning is efficiently achievable by regular Transformers, where the locality measures the least number of tokens required in addition to the tokens histogram to correlate nontrivially with the target. As shown experimentally and theoretically under additional assumptions, distributions with high locality cannot be learned efficiently. In particular, syllogisms cannot be composed on long chains. Furthermore, we argue that (i) an agnostic scratchpad cannot help to break the locality, (ii) an educated scratchpad can help if it breaks the locality at each step, (iii) a notion of 'inductive scratchpad' can both break the locality and help with out-of-distribution generalization.
     </details>

24. **miniCTX: Neural Theorem Proving with (Long-)Contexts** [[pdf]](https://arxiv.org/abs/2408.03350v1) `NeurIPS 2024 Workshop MATH-AI` `Lean` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces miniCTX, which tests a model's ability to prove formal mathematical theorems that depend on new context that is not seen during training, and offers ntp-toolkit for automatically extracting and annotating theorem proving data.
     </details>


     <details>
          <summary>Abstract</summary>
          We introduce miniCTX, which tests a model's ability to prove formal mathematical theorems that depend on new definitions, lemmas, or other contextual information that was not observed during training. miniCTX contains theorems sourced from real Lean projects and textbooks, each associated with a context that can span tens of thousands of tokens. Models are tasked with proving a theorem given access to code from the theorem's repository, which contains context that is helpful or needed for the proof. As a baseline for miniCTX, we introduce file-tuning, a simple recipe that trains a model to generate a proof step conditioned on the preceding file contents. File-tuning substantially outperforms the traditional neural theorem proving approach that fine-tunes on states alone. Additionally, our file-tuned model improves performance on the standard miniF2F benchmark, achieving a pass rate of 33.61%, which is a new state-of-the-art for 1.3B parameter models. Alongside miniCTX, we offer ntp-toolkit for automatically extracting and annotating theorem proving data, making it easy to add new projects into miniCTX to ensure that contexts are not seen during training. miniCTX offers a challenging and realistic perspective on evaluating neural theorem provers.
     </details>

25. **DACO: Towards Application-Driven and Comprehensive Data Analysis via Code Generation** [[pdf]](https://openreview.net/forum?id=Y5iTZ52yFs&name=pdf) `NeurIPS 2024` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes to automatically generate high-quality answer annotations leveraging the code-generation capabilities of LLMs with a multi-turn prompting technique, and trains a 6B supervised fine-tuning model on DACO dataset, and finds that the SFT model learns reasonable data analysis capabilities.
     </details>


     <details>
          <summary>Abstract</summary>
          Data analysis is a crucial analytical process essential for deriving insights from real-world databases. As shown in Figure 1, the need for data analysis typically arises from specific application scenarios, and requires diverse reasoning skills including mathematical reasoning, logical reasoning, and strategic reasoning. Existing work often focus on simple factual retrieval or arithmetic resolutions and thus are insufficient for addressing complex real-world queries. This work aims to propose new resources and benchmarks on this crucial yet challenging and under-explored task. Due to the prohibitively high cost of collecting expert annotations, we use large language models (LLMs) enhanced by code generation to automatically generate high-quality data analysis, which will later be refined by human annotators. We construct the DACO dataset, containing (1) 440 databases (of tabular data) collected from real-world scenarios, (2) ~2k automatically generated query-answer pairs that can serve as weak supervision for model training, and (3) a concentrated but high-quality test set with human refined annotations that serves as our main evaluation benchmark. Experiments show that while LLMs like GPT-4 exhibit promising data analysis capabilities, they are still evaluated as less helpful than human-written analysis on 58.1% cases. Leveraging our weak supervision data, we experiment with various fine-tuning methods, including supervised fine-tuning (SFT) and reinforcement learning from human feedback (RLHF). Our trained model outperforms existing baselines for table question answering, and RLHF further boosts the helpfulness of generated analysis on 58.5% cases.Data and code are released at https://github.com/shirley-wu/daco.
     </details>

26. **Easy2Hard-Bench: Standardized Difficulty Labels for Profiling LLM Performance and Generalization** [[pdf]](http://arxiv.org/abs/2409.18433) `NeurIPS 2024` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Through extensive experiments with six state-of-the-art LLMs, this work provides a comprehensive analysis of their performance and generalization capabilities across varying levels of difficulty, with the aim of inspiring future research in LLM generalization.
     </details>


     <details>
          <summary>Abstract</summary>
          Despite the abundance of datasets available for assessing large language models (LLMs), the scarcity of continuous and reliable difficulty labels for individual data points, in most cases, curtails their capacity to benchmark model generalization performance across different levels of complexity. Addressing this limitation, we present Easy2Hard, an innovative collection of 6 benchmark datasets featuring standardized difficulty labels spanning a wide range of domains, such as mathematics and programming problems, chess puzzles, and reasoning questions, providing a much-needed tool for those in demand of a dataset with varying degrees of difficulty for LLM assessment. We estimate the difficulty of individual problems by leveraging the performance data of many human subjects and LLMs on prominent leaderboards. Harnessing the rich human performance data, we employ widely recognized difficulty ranking systems, including the Item Response Theory (IRT) and Glicko-2 models, to uniformly assign difficulty scores to problems. The Easy2Hard datasets distinguish themselves from previous collections by incorporating a significantly higher proportion of challenging problems, presenting a novel and demanding test for state-of-the-art LLMs. Through extensive experiments conducted with six state-of-the-art LLMs on the Easy2Hard datasets, we offer valuable insights into their performance and generalization capabilities across varying degrees of difficulty, setting the stage for future research in LLM generalization.
     </details>

27. **Learning Formal Mathematics From Intrinsic Motivation** [[pdf]](http://arxiv.org/abs/2407.00695) `NeurIPS 2024` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          How did humanity coax mathematics from the aether? We explore the Platonic view that mathematics can be discovered from its axioms---a game of conjecture and proof. We describe an agent that jointly learns to pose challenging problems for itself (conjecturing) and solve them (theorem proving). Given a mathematical domain axiomatized in dependent type theory, we first combine methods for constrained decoding and type-directed synthesis to sample valid conjectures from a language model. Our method guarantees well-formed conjectures by construction, even as we start with a randomly initialized model. We use the same model to represent a policy and value function for guiding proof search. Our agent targets generating hard but provable conjectures --- a moving target, since its own theorem proving ability also improves as it trains. We propose novel methods for hindsight relabeling on proof search trees to significantly improve the agent's sample efficiency in both tasks. Experiments on 3 axiomatic domains (propositional logic, arithmetic and group theory) demonstrate that our agent can bootstrap from only the axioms, self-improving in generating true and challenging conjectures and in finding proofs.
     </details>

28. **Looped Transformers for Length Generalization** [[pdf]](http://arxiv.org/abs/2409.15647) `NeurIPS 2024 Workshop MATH-AI` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work demonstrates that looped Transformers with an adaptive number of steps significantly improve length generalization, and trains looped Transformers using the proposed learning algorithm and observes that they learn highly length-generalizable solutions for various tasks.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent work has shown that Transformers trained from scratch can successfully solve various arithmetic and algorithmic tasks, such as adding numbers and computing parity. While these Transformers generalize well on unseen inputs of the same length, they struggle with length generalization, i.e., handling inputs of unseen lengths. In this work, we demonstrate that looped Transformers with an adaptive number of steps significantly improve length generalization. We focus on tasks with a known iterative solution, involving multiple iterations of a RASP-L operation - a length-generalizable operation that can be expressed by a finite-sized Transformer. We train looped Transformers using our proposed learning algorithm and observe that they learn highly length-generalizable solutions for various tasks.
     </details>

29. **MACM: Utilizing a Multi-Agent System for Condition Mining in Solving Complex Mathematical Problems** [[pdf]](http://arxiv.org/abs/2404.04735) `NeurIPS 2024` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduces the MACM prompting method, which not only resolves intricate mathematical problems but also demonstrates strong generalization capabilities across various mathematical contexts.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent advancements in large language models, such as GPT-4, have demonstrated remarkable capabilities in processing standard queries. Despite these advancements, their performance substantially declines in advanced mathematical problems requiring complex, multi-step logical reasoning. To enhance their inferential capabilities, current research has delved into prompting engineering, exemplified by methodologies such as the Tree of Thought and Graph of Thought.Nonetheless, these existing approaches encounter two significant limitations. Firstly, their effectiveness in tackling complex mathematical problems is somewhat constrained. Secondly, the necessity to design distinct prompts for individual problems hampers their generalizability.In response to these limitations, this paper introduces the Multi-Agent System for conditional Mining (MACM) prompting method. It not only resolves intricate mathematical problems but also demonstrates strong generalization capabilities across various mathematical contexts.With the assistance of MACM, the accuracy of GPT-4 Turbo on the most challenging level five mathematical problems in the MATH dataset increase from $\mathbf{54.68\\%}  \text{ to } \mathbf{76.73\\%}$.
     </details>

30. **MathCAMPS: Fine-grained Synthesis of Mathematical Problems From Human Curricula** [[pdf]](http://arxiv.org/abs/2407.00900) `NeurIPS 2024 Workshop MATH-AI` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes MathCAMPS: a method to synthesize high-quality mathematical problems at scale, grounded on 44 fine-grained standards from the Mathematics Common Core (CC) Standard for K-8 grades, and proposes a cycle-consistency method for validating problem faithfulness.
     </details>


     <details>
          <summary>Abstract</summary>
          Mathematical problem solving is an important skill for Large Language Models (LLMs), both as an important capability and a proxy for a range of reasoning abilities. Existing benchmarks probe a diverse set of skills, but they yield aggregate accuracy metrics, obscuring specific abilities or weaknesses. Furthermore, they are difficult to extend with new problems, risking data contamination over time. To address these challenges, we propose MathCAMPS: a method to synthesize high-quality mathematical problems at scale, grounded on 44 fine-grained "standards" from the Mathematics Common Core (CC) Standard for K-8 grades. We encode each standard in a formal grammar, allowing us to sample diverse symbolic problems and their answers. We then use LLMs to realize the symbolic problems into word problems. We propose a cycle-consistency method for validating problem faithfulness. Finally, we derive follow-up questions from symbolic structures and convert them into follow-up word problems - a novel task of mathematical dialogue that probes for robustness in understanding. Experiments on 23 LLMs show surprising failures even in the strongest models (in particular when asked simple follow-up questions). Moreover, we evaluate training checkpoints of Pythia 12B on MathCAMPS, allowing us to analyze when particular mathematical skills develop during its training. Our framework enables the community to reproduce and extend our pipeline for a fraction of the typical cost of building new high-quality datasets.
     </details>

31. **Proving Olympiad Algebraic Inequalities without Human Demonstrations** [[pdf]](http://arxiv.org/abs/2406.14219) `NeurIPS 2024` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes AIPS, an Algebraic Inequality Proving System capable of autonomously generating complex inequality theorems and effectively solving Olympiad-level inequality problems without requiring human demonstrations.
     </details>


     <details>
          <summary>Abstract</summary>
          Solving Olympiad-level mathematical problems represents a significant advancement in machine intelligence and automated reasoning. Current machine learning methods, however, struggle to solve Olympiad-level problems beyond Euclidean plane geometry due to a lack of large-scale, high-quality datasets. The challenge is even greater in algebraic systems, which involves infinite reasoning spaces within finite conditions. To address these issues, we propose \textit{AIPS}, an \textit{Algebraic Inequality Proving System} capable of autonomously generating complex inequality theorems and effectively solving Olympiad-level inequality problems without requiring human demonstrations. During proof search in a mixed reasoning manner, a value curriculum learning strategy on generated datasets is implemented to improve proving performance, demonstrating strong mathematical intuitions. On a test set of 20 International Mathematical Olympiad-level inequality problems, AIPS successfully solved 10, outperforming state-of-the-art methods. Furthermore, AIPS automatically generated a vast array of non-trivial theorems without human intervention, some of which have been evaluated by professional contestants and deemed to reach the level of the International Mathematical Olympiad. Notably, one theorem was selected as a competition problem in a major city 2024 Mathematical Olympiad.All the materials are available at {\it \href{https://sites.google.com/view/aips}{sites.google.com/view/aips}}.
     </details>

32. **Proving Theorems Recursively** [[pdf]](http://arxiv.org/abs/2405.14414) `NeurIPS 2024` `Isabelle` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          POETRY (PrOvE Theorems RecursivelY), which proves theorems in a recursive, level-by-level manner in the Isabelle theorem prover, is proposed, which allows the theorem to be tackled incrementally by outlining the overall theorem at the first level and then solving the intermediate conjectures at deeper levels.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent advances in automated theorem proving leverages language models to explore expanded search spaces by step-by-step proof generation. However, such approaches are usually based on short-sighted heuristics (e.g., log probability or value function scores) that potentially lead to suboptimal or even distracting subgoals, preventing us from finding longer proofs. To address this challenge, we propose POETRY (PrOvE Theorems RecursivelY), which proves theorems in a recursive, level-by-level manner in the Isabelle theorem prover. Unlike previous step-by-step methods, POETRY searches for a verifiable sketch of the proof at each level and focuses on solving the current level's theorem or conjecture. Detailed proofs of intermediate conjectures within the sketch are temporarily replaced by a placeholder tactic called sorry, deferring their proofs to subsequent levels. This approach allows the theorem to be tackled incrementally by outlining the overall theorem at the first level and then solving the intermediate conjectures at deeper levels. Experiments are conducted on the miniF2F and PISA datasets and significant performance gains are observed in our POETRY approach over state-of-the-art methods. POETRY on miniF2F achieves an average proving success rate improvement of 5.1%. Moreover, we observe a substantial increase in the maximum proof length found by POETRY, from 10 to 26.
     </details>

33. **PutnamBench: Evaluating Neural Theorem-Provers on the Putnam Mathematical Competition** [[pdf]](http://arxiv.org/abs/2407.11214) `NeurIPS 2024` `Lean, Isabelle, Coq` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          PutnamBench consists of 1697 hand-constructed formalizations of 640 theorems sourced from the William Lowell Putnam Mathematical Competition, the premier undergraduate-level mathematics competition in North America.
     </details>


     <details>
          <summary>Abstract</summary>
          We present PutnamBench, a new multilingual benchmark for evaluating the ability of neural theorem-provers to solve competition mathematics problems. PutnamBench consists of 1337 hand-constructed formalizations of 514 theorems sourced from the William Lowell Putnam Mathematical Competition, the premier undergraduate-level mathematics competition in North America.  All the theorems have formalizations in Lean 4 and Isabelle; a substantial subset also has Coq formalizations. Proving the theorems requires significant problem-solving ability and proficiency in a broad range of topics taught in undergraduate mathematics courses. We use PutnamBench to evaluate several established neural and symbolic theorem-provers.  These approaches can only solve a handful of the PutnamBench problems, establishing the benchmark as a difficult open challenge for research on neural theorem-proving. PutnamBench is available at https://github.com/trishullab/PUTNAM.
     </details>

34. **Trajectory Volatility for Out-of-Distribution Detection in Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2405.14039) `NeurIPS 2024` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A trajectory-based method TV score is proposed, which uses trajectory volatility for OOD detection in mathematical reasoning and outperforms all traditional algorithms on GLMs under mathematical reasoning scenarios and can be extended to more applications with high-density features in output spaces, such as multiple-choice questions.
     </details>


     <details>
          <summary>Abstract</summary>
          Real-world data deviating from the independent and identically distributed (\textit{i.i.d.}) assumption of in-distribution training data poses security threats to deep networks, thus advancing out-of-distribution (OOD) detection algorithms. Detection methods in generative language models (GLMs) mainly focus on uncertainty estimation and embedding distance measurement, with the latter proven to be most effective in traditional linguistic tasks like summarization and translation. However, another complex generative scenario mathematical reasoning poses significant challenges to embedding-based methods due to its high-density feature of output spaces, but this feature causes larger discrepancies in the embedding shift trajectory between different samples in latent spaces. Hence, we propose a trajectory-based method TV score, which uses trajectory volatility for OOD detection in mathematical reasoning. Experiments show that our method outperforms all traditional algorithms on GLMs under mathematical reasoning scenarios and can be extended to more applications with high-density features in output spaces, such as multiple-choice questions.
     </details>

35. **Unlocking the Boundaries of Thought: A Reasoning Granularity Framework to Quantify and Optimize Chain-of-Thought** [[pdf]](http://arxiv.org/abs/2410.05695) `NeurIPS 2024` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work defines a reasoning granularity (RG) to quantify the upper bound of CoT and establishes a combination law for RG, enabling a practical quantitative approach applicable to various real-world CoT tasks.
     </details>


     <details>
          <summary>Abstract</summary>
          Chain-of-Thought (CoT) reasoning has emerged as a promising approach for enhancing the performance of large language models (LLMs) on complex reasoning tasks. Recently, a series of studies attempt to explain the mechanisms underlying CoT, aiming to deepen the understanding and enhance its efficacy. Nevertheless, the existing research faces two major challenges: (1) a lack of quantitative metrics to assess CoT capabilities and (2) a dearth of guidance on optimizing CoT performance. Motivated by this, in this work, we introduce a novel reasoning granularities (RG) methodological framework to address these challenges. To solve the lack of quantification, we first define an RG to quantify the upper bound of CoT and establish a combination law for RG, enabling a practical quantitative approach applicable to various real-world CoT tasks. To address the lack of optimization, we propose three categories of RGs. We further optimize these categories with combination laws focused on RG promotion and reasoning path optimization for CoT improvement. Through extensive experiments on 25 models and 4 tasks, the study validates the existence and rationality of the proposed framework. Furthermore, it explains the effectiveness of 10 CoT strategies and guides optimization from two perspectives. We hope this work can provide a comprehensive understanding of the boundaries and optimization strategies for reasoning in LLMs.
     </details>

36. **A Theoretical Understanding of Self-Correction through In-context Alignment** [[pdf]](https://openreview.net/forum?id=OtvNLTWYww) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Going beyond mimicking limited human experiences, recent studies show initial evidence that, like humans, large language models (LLMs) are capable of improving their abilities purely by self-correction, \textit{i.e.}, correcting previous responses through self-examination, in certain circumstances. Nevertheless, little is known about how such capabilities arise. In this work, based on a simplified setup akin to an alignment task, we theoretically analyze self-correction from an in-context learning perspective, showing that when LLMs give relatively accurate self-examinations as rewards, they are capable of refining responses in an in-context way. Notably, going beyond previous theories on over-simplified linear transformers, our theoretical construction underpins the roles of several key designs of realistic transformers for self-correction: softmax attention, multi-head attention, and the MLP block. We validate these findings extensively on synthetic datasets. Inspired by these findings, we also illustrate novel applications of self-correction, such as defending against LLM jailbreaks, where a simple self-correction step does make a large difference. We believe that these findings will inspire further research on understanding, exploiting, and enhancing self-correction for building better foundation models.
     </details>

37. **ABEL: Sample Efficient Online Reinforcement Learning for Neural Theorem Proving** [[pdf]](https://openreview.net/forum?id=kk3mSjVCUO) `NeurIPS 2024 Workshop MATH-AI` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          We propose a scalable and efficient reinforcement learning framework as a strong baseline for theorem proving with limited data. This baseline reaches performances comparable to the current state-of-the-art in theorem proving, while only training on a few hundred examples. This a first step toward an efficient and easily reproducible combination of autoformalization, synthetic data generation and reinforcement learning, which could unlock significant advancements in neural theorem proving.
     </details>

38. **AI for Math or Math for AI? On the Generalization of Learning Mathematical Problem Solving** [[pdf]](https://openreview.net/forum?id=xlnvZ85CSo) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          There has been a growing interest in enhancing the mathematical problem-solving (MPS) capabilities of LLMs. While some researchers focus on developing specialized math models to advance AI for math, another intriguing perspective is math for AI, positing that integrating mathematical reasoning data could enable LLMs to perform complex reasoning more broadly. This hypothesis draws from neuroscience studies which show that solving mathematical problems aids in the development of general reasoning skills in humans. The concept of "math for AI" has gained particular relevance as the research community increasingly focuses on complex reasoning -- Given the scarcity of complex and lengthy chain-of-thought data, MPS emerges as a prime candidate for collecting or synthesizing substantial volumes of intricate thought processes, thus serving as a potential key resource for enhancing general complex reasoning. However, it remains unclear whether skills acquired through learning MPS can extend to other reasoning tasks or merely improve MPS-specific benchmark scores.  In this paper, we present a comprehensive empirical analysis to address this question. Specifically, we explore three prevalent methods for improving MPS: (1) continual pretraining on mathematical text; (2) instruction pretraining on large-scale QA pairs synthesized from raw text; and (3) instruction tuning on benchmark-oriented datasets.  Through controlled experiments and evaluations across five distinct reasoning domains, we find that extensive continual pretraining on mathematical texts can improve performance on certain non-MPS reasoning tasks. Conversely, other training approaches either match or fall short of baseline performances. These findings indicate that most readily available data sources do not support the "math for AI" objective in enhancing non-MPS tasks. Identifying which data sources best contribute to the acquisition of complex reasoning skills remains a crucial question for future research.
     </details>

39. **Advancing Theorem Proving in LLMs through Large-Scale Synthetic Data** [[pdf]](https://openreview.net/forum?id=TPtXLihkny) `NeurIPS 2024 Workshop MATH-AI` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Proof assistants like Lean have revolutionized mathematical proof verification by providing high levels of accuracy and reliability. Although large language models (LLMs) have demonstrated potential in mathematical reasoning, their advancement in formal theorem proving is hindered by the scarcity of large, high-quality training datasets. To address this challenge, we present a novel approach to generate extensive Lean 4 proof data from natural language mathematical problems at the high school and undergraduate levels. Specifically, we synthesize 8 million formal statements with corresponding proofs, leveraging this dataset to fine-tune the DeepSeekMath 7B model. The resulting model, DS-Prover, achieves a pass rate of 50\% on the Lean 4 miniF2F benchmark, surpassing the previous state-of-the-art result of 41.0\%. These findings underscore the potential of large-scale synthetic data in significantly enhancing the theorem-proving capabilities of LLMs.
     </details>

40. **Attention Bias as an Inductive Bias: How to Teach Transformers Simple Arithmetic** [[pdf]](https://openreview.net/forum?id=Ei4bzOt8NG) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          In this paper, we study the transformer model's capability in learning arithmetic from an inductive learning perspective and draw attention to the importance of inductive biases. We first introduce a definition of length generalization, requiring the model to maintain near perfect accuracy on samples with length at least 10 times the training length, as an indicator of successful learning. Through experiments and attention analysis, we show that the failure of the vanilla Transformer on learning arithmetic is due to inadequate inductive biasing. We then present Attention Bias Scaffolding (ABS) which uses attention masking to enforce the necessary inductive bias, making it the first Transformer-based architecture to achieve complete generalization on several arithmetic tasks such as addition and parity. Additionally, we introduce Attention Bias Calibration (ABC), a calibration stage that allows the model to learn the proper attention biases, and obtain complete length generalization automatically on tasks that could interpolate. Finally, we show that ABC bears remarkable similarities to RPE and LoRA, which may indicate the potential for applications to more complex tasks.
     </details>

41. **Augmenting Language Models with Regression Loss on Number Tokens for Arithmetic Reasoning** [[pdf]](https://openreview.net/forum?id=Cb8RP9KLyh) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          While language models (LMs) have exceptional capabilities at text generation, they lack a natural inductive bias for emitting numbers and thus struggle in tasks involving reasoning over quantities, especially arithmetics.   This has particular relevance in scientific datasets where combinations of text and numerical data are abundant.    One fundamental limitation is the nature of the cross entropy loss which assumes a nominal (categorical) scale and thus cannot convey proximity between generated number tokens.   Here we present a number token loss (NTL) that relies on a dot product between the values of the number tokens and their predicted output probabilities.   The resulting regression loss can easily be added to any language model and augment the cross entropy objective during training.   We compare the proposed scheme on the DeepMind Mathematics Dataset against existing tokenization, encoding and decoding schemes for improving number representation in LMs. Our results reveal a significant improvement in numerical accuracy when equipping a standard T5 model with the proposed loss scheme.
     </details>

42. **Autoformalize Mathematical Statements by Symbolic Equivalence and Semantic Consistency** [[pdf]](http://arxiv.org/abs/2410.20936) `NeurIPS 2024` `Isabelle` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Autoformalization, the task of automatically translating natural language descriptions into a formal language, poses a significant challenge across various domains, especially in mathematics. Recent advancements in large language models (LLMs) have unveiled their promising capabilities to formalize even competition-level math problems. However, we observe a considerable discrepancy between pass@1 and pass@k accuracies in LLM-generated formalizations. To address this gap, we introduce a novel framework that scores and selects the best result from k autoformalization candidates based on two complementary self-consistency methods: symbolic equivalence and semantic consistency. Elaborately, symbolic equivalence identifies the logical homogeneity among autoformalization candidates using automated theorem provers, and semantic consistency evaluates the preservation of the original meaning by informalizing the candidates and computing the similarity between the embeddings of the original and informalized texts. Our extensive experiments on the MATH and miniF2F datasets demonstrate that our approach significantly enhances autoformalization accuracy, achieving up to 0.22-1.35x relative improvements across various LLMs and baseline methods.
     </details>

43. **Benchmarking the Reasoning Robustness against Noisy Rationales in Chain-of-thought Prompting** [[pdf]](https://neurips.cc/virtual/2024/poster/95956) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          This paper investigates an under-explored challenge in large language models (LLMs): chain-of-thought prompting with noisy rationales—irrelevant or inaccurate reasoning steps—despite advancements in in-context learning. We construct the NoRa dataset, specifically designed to evaluate LLMs’ robustness to noisy rationales, based on which we reveal a widespread vulnerability among LLMs to such noise, with limited efficacy from existing reasoning methods. To combat this, we propose the contrastive denoising with noisy chain-of-thought (CD-CoT) method to enhance denoising-reasoning capabilities by contrasting noisy rationales with only one clean rationale, thereby advancing the robustness of LLMs in reasoning.
     </details>

44. **Counterfactual PPO Enhanced Shared Reflector for LLM-based Multi-agent Collaboration** [[pdf]](https://neurips.cc/virtual/2024/poster/93147) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Benefiting from the powerful language expression and planning capabilities of Large Language Models (LLMs), LLM-based autonomous agents achieve promising performance in various downstream tasks. Recently, based on the development of single-agent systems, researchers propose to construct LLM-based multi-agent systems to tackle more complicated tasks. In this paper, we propose a novel framework, named COPPER, to enhance the collaboration ability of multi-agent systems through learnable self-reflection mechanism. To improve the quality of reflections, we propose to fine-tune a shared reflector, which automatically tunes the prompts of actor models using our counterfactual PPO mechanism. On the one hand, we propose counterfactual rewards to assess the contribution of a single agent’s reflection within the system, alleviating the credit assignment problem. On the other hand, we propose to train a shared reflector, which enables the reflector to personalize generated reflections according to agent roles, while reducing the computational resource requirements and improving training stability. We conduct experiments on three datasets to evaluate the performance of multi-agent systems in multi-hop question answering, mathematics, and chess scenarios. Experimental results show that COPPER possesses stronger reflection capabilities and exhibits excellent generalization performance across different actor models.
     </details>

45. **Decomposing Complex Visual Comprehension into Atomic Visual Skills for Vision Language Models** [[pdf]](https://openreview.net/forum?id=nFU4xCyoe0) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Recent Vision Language Models (VLMs) have demonstrated impressive multimodal comprehension and reasoning capabilities, but they often struggle with trivially simple visual tasks. In this work, we introduce the Atomic Visual Skills Benchmark (AVSBench) to evaluate whether VLMs possess capabilities to understand basic geometric features, which we refer to as atomic visual skills. Specifically, we systematically categorize the atomic visual skills and handcraft a set of 5,073 diverse questions designed to assess each individual atomic visual skill. Using AVSBench, we evaluate the current leading VLMs and find that they struggle with most of these atomic visual skills that are obvious to humans.
     </details>

46. **Diffusion of Thought: Chain-of-Thought Reasoning in Diffusion Language Models** [[pdf]](https://neurips.cc/virtual/2024/poster/95935) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Recently, diffusion models have garnered significant interest in the field of text processing due to their many potential advantages compared to conventional autoregressive models.In this work, we propose Diffusion-of-Thought (DoT),  a novel approach that integrates diffusion models with Chain-of-Thought, a well-established technique for improving the reasoning ability of autoregressive language models. In contrast to autoregressive language models that make decisions in a left-to-right, token-by-token manner, DoT allows reasoning steps to diffuse over time through a diffusion language model and offers greater flexibility in trading-off computation for reasoning performance. Our experimental results demonstrate the effectiveness of DoT in multi-digit multiplication, boolean logic, and grade school math problems. In addition to that, DoT showcases promising self-correction abilities and benefits from existing reasoning-enhancing techniques like self-consistency decoding. Our findings contribute to the understanding and development of reasoning with diffusion language models.
     </details>

47. **DrawEduMath: An Expert-Annotated Dataset of Students’ Math Images** [[pdf]](https://openreview.net/forum?id=0vQYvcinij) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          We introduce DrawEduMath, a dataset of 2,030 images drawn from an online learning platform containing students' handwritten responses to 188 K-12 math problems. Each image is accompanied by a teacher's detailed description of the student's response, and these annotations describe visual content ranging from students' problem-solving strategies to the positioning and specifications of students' drawings, diagrams, and labels. We then synthetically transform teachers' descriptions into question-answer pairs to evaluate vision language models (VLMs). We show that two state-of-the-art models, Gemini-1.5 Pro and GPT-4o, demonstrate room for improvement on DrawEduMath questions. Overall, this new benchmark assesses models' abilities to reason mathematically over images gathered with real-world educational contexts in mind.
     </details>

48. **Enhancing Language Model Reasoning via Weighted Reasoning in Self-Consistency** [[pdf]](http://arxiv.org/abs/2410.07839) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work enhances Wang et al's self-consistency framework by incorporating and analyzing both the reasoning paths of these rationales in addition to their final decisions before taking a majority vote, which not only improves the reliability of reasoning paths but also cause more robust performance on complex reasoning tasks.
     </details>


     <details>
          <summary>Abstract</summary>
          While large language models (LLMs) have rapidly improved their performance on a broad number of tasks, they still often fall short on reasoning tasks. As LLMs become more integrated in diverse real-world tasks, advancing their reasoning capabilities is crucial to their effectiveness in nuanced, complex problems. Wang et al's self-consistency framework reveals that sampling multiple rationales before taking a majority vote reliably improves model performance across various closed-answer reasoning tasks. Standard methods based on this framework aggregate the final decisions of these rationales but fail to utilize the detailed step-by-step reasoning paths applied by these paths. Our work enhances this approach by incorporating and analyzing both the reasoning paths of these rationales in addition to their final decisions before taking a majority vote. These methods not only improve the reliability of reasoning paths but also cause more robust performance on complex reasoning tasks.
     </details>

49. **Evaluating Large Vision-and-Language Models on Children's Mathematical Olympiads** [[pdf]](http://arxiv.org/abs/2406.15736) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Evaluating state-of-the-art LVLMs on their mathematical and algorithmic reasoning abilities using visuo-linguistic problems from children's Olympiads shows that modern LVLMs do demonstrate increasingly powerful reasoning skills in solving problems for higher grades, but lack the foundations to correctly answer problems designed for younger children.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent years have seen a significant progress in the general-purpose problem solving abilities of large vision and language models (LVLMs), such as ChatGPT, Gemini, etc.; some of these breakthroughs even seem to enable AI models to outperform human abilities in varied tasks that demand higher-order cognitive skills. Are the current large AI models indeed capable of generalized problem solving as humans do?  A systematic analysis of AI capabilities for joint vision and text reasoning, however, is missing in the current scientific literature. In this paper, we make an effort towards filling this gap, by evaluating state-of-the-art LVLMs on their mathematical and algorithmic reasoning abilities using visuo-linguistic problems from children's Olympiads. Specifically, we consider problems from the Mathematical Kangaroo (MK) Olympiad, which is a popular international competition targeted at children from grades 1-12, that tests children's deeper mathematical abilities using puzzles that are appropriately gauged to their age and skills. Using the puzzles from MK, we created a dataset, dubbed SMART-840, consisting of 840 problems from years 2020-2024. With our dataset, we analyze LVLMs power on mathematical reasoning; their responses on our puzzles offer a direct way to compare against that of children. Our results show that modern LVLMs do demonstrate increasingly powerful reasoning skills in solving problems for higher grades, but lack the foundations to correctly answer problems designed for younger children. Further analysis shows that there is no significant correlation between the reasoning capabilities of AI models and that of young children, and their capabilities appear to be based on a different type of reasoning than the cumulative knowledge that underlies children's mathematical skills.
     </details>

50. **Flow-DPO: Improving LLM Mathematical Reasoning through Online Multi-Agent Learning** [[pdf]](https://openreview.net/forum?id=uwagVHmyNA) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Mathematical reasoning is a crucial capability for Large Language Models (LLMs), yet generating detailed and accurate reasoning traces remains a significant challenge. This paper introduces a novel approach to produce high-quality reasoning traces for LLM fine-tuning using online learning \textbf{Flows}. Our method employs an incremental output production Flow, where component LLMs collaboratively construct solutions through iterative communication. We train the Flow using online Direct Preference Optimization (DPO) learning with rollouts, generating DPO pairs for each training example and updating models in real-time. We directly compare the quality of reasoning traces generated by our method with those produced through direct model inference, demonstrating the effectiveness of our approach in improving LLM performance in mathematical reasoning tasks.
     </details>

51. **Formal Representation and Solution of Plane Geometric Problems** [[pdf]](https://openreview.net/forum?id=8wDSfs1W3w) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Geometric problem solving has always been a long-standing challenge in the fields of mathematical reasoning and artificial intelligence. In this paper, we present formalgeo7k, a geometric problem dataset based on rigorous geometry formalization theory and consistent geometry formal system, serving as a benchmark for various tasks such as geometric diagram parsing and geometric problem solving. All problems are annotated with problem text, problem diagram, formal descriptions, and solution. Combining symbolic solver and deep learning techniques, we can achieve human-like, traceable and explainable solutions, which are stored in a hypergraph for graph-related tasks. We experiment with various methods and the best method achieves only 86.40% on formalgeo7k. This shows that formalgeo7k presents a challenge for future research.
     </details>

52. **Formal Theorem Proving by Rewarding LLMs to Decompose Proofs Hierarchically** [[pdf]](http://arxiv.org/abs/2411.01829) `NeurIPS 2024 Workshop MATH-AI` `Isabelle` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Mathematical theorem proving is an important testbed for large language models' deep and abstract reasoning capability. This paper focuses on improving LLMs' ability to write proofs in formal languages that permit automated proof verification/evaluation. Most previous results provide human-written lemmas to the theorem prover, which is an arguably oversimplified setting that does not sufficiently test the provers' planning and decomposition capabilities. Instead, we work in a more natural setup where the lemmas that are directly relevant to the theorem are not given to the theorem prover at test time. We design an RL-based training algorithm that encourages the model to decompose a theorem into lemmas, prove the lemmas, and then prove the theorem by using the lemmas. Our reward mechanism is inspired by how mathematicians train themselves: even if a theorem is too challenging to be proved by the current model, a positive reward is still given to the model for any correct and novel lemmas that are proposed and proved in this process. During training, our model proposes and proves lemmas that are not in the training dataset. In fact, these newly-proposed correct lemmas consist of 37.7% of the training replay buffer when we train on the dataset extracted from Archive of Formal Proofs (AFP). The model trained by our RL algorithm outperforms that trained by supervised finetuning, improving the pass rate from 40.8% to 45.5% on AFP test set, and from 36.5% to 39.5% on an out-of-distribution test set.
     </details>

53. **Give me a hint: Can LLMs take a hint to solve math problems?** [[pdf]](http://arxiv.org/abs/2410.05915) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes giving "hints" to improve the language model's performance on advanced mathematical problems, taking inspiration from how humans approach math pedagogically, and demonstrates the effectiveness of the approach by evaluating various LLMs.
     </details>


     <details>
          <summary>Abstract</summary>
          While state-of-the-art LLMs have shown poor logical and basic mathematical reasoning, recent works try to improve their problem-solving abilities using prompting techniques. We propose giving "hints" to improve the language model's performance on advanced mathematical problems, taking inspiration from how humans approach math pedagogically. We also test robustness to adversarial hints and demonstrate their sensitivity to them. We demonstrate the effectiveness of our approach by evaluating various diverse LLMs, presenting them with a broad set of problems of different difficulties and topics from the MATH dataset and comparing against techniques such as one-shot, few-shot, and chain of thought prompting.
     </details>

54. **Global Lyapunov functions: a long-standing open problem in mathematics, with symbolic transformers** [[pdf]](http://arxiv.org/abs/2410.08304) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Despite their spectacular progress, language models still struggle on complex reasoning tasks, such as advanced mathematics. We consider a long-standing open problem in mathematics: discovering a Lyapunov function that ensures the global stability of a dynamical system. This problem has no known general solution, and algorithmic solvers only exist for some small polynomial systems. We propose a new method for generating synthetic training samples from random solutions, and show that sequence-to-sequence transformers trained on such datasets perform better than algorithmic solvers and humans on polynomial systems, and can discover new Lyapunov functions for non-polynomial systems.
     </details>

55. **HARDMath: A Benchmark Dataset for Challenging Problems in Applied Mathematics** [[pdf]](http://arxiv.org/abs/2410.09988) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          HARDMath is introduced, a dataset inspired by a graduate course on asymptotic methods, featuring challenging applied mathematics problems that require analytical approximation techniques, and auto-generates a large number of problems with solutions validated against numerical ground truths.
     </details>


     <details>
          <summary>Abstract</summary>
          Advanced applied mathematics problems are underrepresented in existing Large Language Model (LLM) benchmark datasets. To address this, we introduce HARDMath, a dataset inspired by a graduate course on asymptotic methods, featuring challenging applied mathematics problems that require analytical approximation techniques. These problems demand a combination of mathematical reasoning, computational tools, and subjective judgment, making them difficult for LLMs. Our framework auto-generates a large number of problems with solutions validated against numerical ground truths. We evaluate both open- and closed-source LLMs on HARDMath-mini, a sub-sampled test set of 366 problems, as well as on 40 word problems formulated in applied science contexts. Even leading closed-source models like GPT-4 achieve only 43.8% overall accuracy with few-shot Chain-of-Thought prompting, and all models demonstrate significantly lower performance compared to results on existing mathematics benchmark datasets. We additionally conduct a detailed error analysis to gain insights into the failure cases of LLMs. These results demonstrate limitations of current LLM performance on advanced graduate-level applied math problems and underscore the importance of datasets like HARDMath to advance mathematical abilities of LLMs.
     </details>

56. **InfiMM-WebMath-40B: Advancing Multimodal Pre-Training for Enhanced Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2409.12568) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces InfiMM-WebMath-40B, a high-quality dataset of interleaved image-text documents that significantly enhances the performance of the 1.3B model, and sets a new state-of-the-art among open-source models on multi-modal math benchmarks such as MathVerse and The authors-Math.
     </details>


     <details>
          <summary>Abstract</summary>
          Pre-training on large-scale, high-quality datasets is crucial for enhancing the reasoning capabilities of Large Language Models (LLMs), especially in specialized domains such as mathematics. Despite the recognized importance, the Multimodal LLMs (MLLMs) field currently lacks a comprehensive open-source pre-training dataset specifically designed for mathematical reasoning. To address this gap, we introduce InfiMM-WebMath-40B, a high-quality dataset of interleaved image-text documents. It comprises 24 million web pages, 85 million associated image URLs, and 40 billion text tokens, all meticulously extracted and filtered from CommonCrawl. We provide a detailed overview of our data collection and processing pipeline. To demonstrate the robustness of InfiMM-WebMath-40B, we conducted evaluations in both text-only and multimodal settings. Our evaluations on text-only benchmarks show that, despite utilizing only 40 billion tokens, our dataset significantly enhances the performance of our 1.3B model, delivering results comparable to DeepSeekMath-1.3B, which uses 120 billion tokens for the same model size. Nevertheless, with the introduction of our multi-modal math pre-training dataset, our models set a new state-of-the-art among open-source models on multi-modal math benchmarks such as MathVerse and We-Math. We release our data at https://huggingface.co/datasets/Infi-MM/InfiMM-WebMath-40B.
     </details>

57. **Instance-adaptive Zero-shot Chain-of-Thought Prompting** [[pdf]](http://arxiv.org/abs/2409.20441) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces an instance-adaptive prompting algorithm as an alternative zero-shot CoT reasoning scheme by adaptively differentiating good and bad prompts, and proposes an instance-adaptive prompting strategy (IAP) for zero-shot CoT reasoning.
     </details>


     <details>
          <summary>Abstract</summary>
          Zero-shot Chain-of-Thought (CoT) prompting emerges as a simple and effective strategy for enhancing the performance of large language models (LLMs) in real-world reasoning tasks. Nonetheless, the efficacy of a singular, task-level prompt uniformly applied across the whole of instances is inherently limited since one prompt cannot be a good partner for all, a more appropriate approach should consider the interaction between the prompt and each instance meticulously. This work introduces an instance-adaptive prompting algorithm as an alternative zero-shot CoT reasoning scheme by adaptively differentiating good and bad prompts. Concretely, we first employ analysis on LLMs through the lens of information flow to detect the mechanism under zero-shot CoT reasoning, in which we discover that information flows from question to prompt and question to rationale jointly influence the reasoning results most. We notice that a better zero-shot CoT reasoning needs the prompt to obtain semantic information from the question then the rationale aggregates sufficient information from the question directly and via the prompt indirectly. On the contrary, lacking any of those would probably lead to a bad one. Stem from that, we further propose an instance-adaptive prompting strategy (IAP) for zero-shot CoT reasoning. Experiments conducted with LLaMA-2, LLaMA-3, and Qwen on math, logic, and commonsense reasoning tasks (e.g., GSM8K, MMLU, Causal Judgement) obtain consistent improvement, demonstrating that the instance-adaptive zero-shot CoT prompting performs better than other task-level methods with some curated prompts or sophisticated procedures, showing the significance of our findings in the zero-shot CoT reasoning mechanism.
     </details>

58. **Interleaving Text and Number Embeddings to Solve Mathemathics Problems** [[pdf]](https://openreview.net/forum?id=8cNJyqs45T) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Integrating text and numbers effectively is a crucial step towards enhancing Large Language Models (LLMs) capabilities in assisting in scientific tasks. While most current approaches rely on discrete tokenization of numbers, for instance, conversion to scientific notation or base 10-decomposition, a recent approach proposed a continuous numerical encoding as an inductive bias. In this paper, we build upon this approach by introducing more expressive numerical embeddings. Our method addresses key shortcomings, including the elimination of numerical artefacts and the ability to handle a wide range of magnitudes without clipping.   Our work presents two key contributions. First, we employ an MLP to assign distinct directions in the embedding space to different numbers. Our second contribution is the introduction of a routing layer that differentiates between numerical and text embeddings. We hypothesise that this combined approach enables the model to distinguish between text and number distributions while maintaining its capacity for arithmetic operations.   Using only a 45 M parameter encoder-decoder architecture our method achieves a $R^2$=0.9988 over a wide range of magnitude ($10^{-3},10^{8}$). In addition, we empirically observe a reduction of the numerical artefacts and biases observed compared to the baselines.
     </details>

59. **Intermediate Fine-Tuning Improves Mathematical Reasoning in Smaller Models** [[pdf]](https://openreview.net/forum?id=wzaMGXiOEy) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          While large models pre-trained on high-quality data exhibit excellent performance across various reasoning tasks, including mathematical reasoning (e.g. GSM8k, MultiArith), specializing smaller models in mathematical reasoning remains a challenging problem. A common research approach to address this challenge involves distilling knowledge from large pre-trained teacher models into smaller student models. Other techniques include augmenting datasets by rephrasing questions or using multiple views of solutions to improve reasoning performance. In this work, we explore intermediate fine-tuning and show that fine-tuning a model on an arithmetic dataset before fine-tuning it on a reasoning dataset helps improve the model's performance on the reasoning tasks. The arithmetic dataset can be generated programmatically, eliminating the resource-intensive task of dataset creation. We evaluate the impact of intermediate fine-tuning using the original GSM8k training set and an expanded GSM8k training set created through distillation. Our experiments on multiple datasets demonstrate that intermediate fine-tuning leads to average improvements of 6.3% and 14.2% in reasoning tasks using the original and distilled training sets, respectively, with greedy decoding compared to the models fine-tuned directly on these sets.
     </details>

60. **LLM Training Data Synthesis for More Effective Problem Solving using Satisfiability Modulo Theories** [[pdf]](https://openreview.net/forum?id=hR4Hskr4GX) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Mathematical reasoning with large language models (LLMs) is an emerging research area. A recent breakthrough is the use of off-the-shelf tools LLMs are trained to utilize to offload complex tasks they cannot perform independently. Unfortunately, this approach is limited to popular tools, as many specialized tools lack the data to train these models on. Motivated by our observation that the current tools used with LLMs are insufficient for solving counting problems, in this work, we explore the problem of using Satisfiability Modulo Theories (SMT) solvers with LLMs. Namely, we introduce a novel algorithm for generating synthetic data consisting of problem statements and their solutions represented as Python code interacting with the Z3 API. Our experiments show that fine-tuning LLMs on this dataset substantially enhances their ability to generate accurate Z3 constraint encodings and improves their overall mathematical problem-solving capabilities.
     </details>

61. **Learning Goal-Conditioned Representations in Reward Models for Aligning Language Models** [[pdf]](https://neurips.cc/virtual/2024/poster/95067) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Representation learning is important for the success of Reinforcement Learning (RL) algorithms, but has been less explored for Language Model (LM) alignment with Reinforcement learning from Human Feedback (RLHF).In this work, we present a simple yet effective approach to improve the representations learned by reward models for aligning LMs.Our method uses a contrastive loss that encourages reward models to learn goal-conditioned representations which encode the expected reward at intermediate steps of the input sequence.By enforcing this loss on representations from intermediate steps, we can capture which trajectories are likely to reach a desired goal (e.g., correct solution or helpful response) at different points in the sequence.This method is flexible enough to support different kinds of alignment data and does not require extra annotations.We demonstrate the effectiveness of this approach in 2 domains: mathematical reasoning and natural language alignment.On math benchmarks, such as GSM8k, we show that our approach improves the reward model's ability to discern between correct/incorrect solutions, increasing AUROC score by up to 0.11 points, and that the learned representations can help prune undesirable generations.Using this reward model to improve a policy model via RLHF yields accuracy gains of 1.7\% across several math benchmarks when compared to a standard preference-ranking trained reward model.Additionally, we show the that learned representations can be used to steer LMs toward generations that are more aligned with human preferences via guided decoding.Overall, our study underscores the potential of incorporating feedback signals in RLHF frameworks via learned representations, which we believe is a promising avenue for improving the alignment of LLMs.
     </details>

62. **Learning Mathematical Rules with Large Language Models** [[pdf]](https://openreview.net/forum?id=tIlDF5B6T4) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          In this paper, we study the ability of large language models to learn specific mathematical rules such as distributivity or simplifying equations. We present an empirical analysis of their ability to generalize these rules, as well as to reuse them in the context of word problems. For this purpose, we provide a rigorous methodology to build synthetic data incorporating such rules, and perform fine-tuning of large language models on such data. Our experiments show that our model can learn and generalize these rules to some extent, as well as suitably reuse them in the context of word problems.
     </details>

63. **Library Learning Doesn’t: The Curious Case of the Single-Use “Library”** [[pdf]](https://openreview.net/forum?id=et2T8SKF1O) `NeurIPS 2024 Workshop MATH-AI` `Isabelle` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Advances in Large Language Models (LLMs) have spurred a wave of LLM library learning systems for mathematical reasoning.  These systems aim to learn a reusable library of *tools*, such as formal Isabelle lemmas or Python programs that are tailored to a family of tasks. Many of these systems are inspired by the human structuring of knowledge into reusable and extendable concepts, but do current methods actually learn reusable libraries of tools?    We study two library learning systems for mathematics which both reported increased accuracy: LEGO-Prover and TroVE. We find that function reuse is extremely infrequent on miniF2F and MATH. Our followup ablation experiments suggest that, rather than reuse, self-correction and self-consistency are the primary drivers of the observed performance gains.
     </details>

64. **Machine Learning meets Algebraic Combinatorics: A Suite of Datasets to Accelerate AI for Mathematics Research** [[pdf]](https://openreview.net/forum?id=KQ1gI5qzAf) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          The use of benchmark datasets has become an important engine of progress in machine learning (ML) over the past 15 years. Recently there has been growing interest in utilizing machine learning to drive advances in research-level mathematics. However, off-the-shelf solutions often fail to deliver the types of insights required by mathematicians. This suggests the need for new ML methods specifically designed with mathematics in mind. The question then is: what benchmarks should the community use to evaluate these? On the one hand, toy problems such as learning the multiplicative structure of small finite groups have become popular in the mechanistic interpretability community whose perspective on explainability aligns well with the needs of mathematicians. While toy datasets are a useful to guide initial work, they lack the scale, complexity, and sophistication of many of the principal objects of study in modern mathematics. To address this, we introduce a new collection of datasets, the Algebraic Combinatorics Dataset Repository (ACD Repo), representing either classic or open problems in algebraic combinatorics, a subfield of mathematics that studies discrete structures arising from abstract algebra. After describing the datasets, we discuss the challenges involved in constructing``good'' mathematics dataset for ML and describe baseline model performance.
     </details>

65. **Machines and Mathematical Mutations: Using GNNs to Characterize Quiver Mutation Classes** [[pdf]](https://openreview.net/forum?id=SxeiWOp73E) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Machine learning is becoming an increasingly valuable tool in mathematics, enabling one to identify subtle patterns across collections of examples so vast that they would be impossible for a single researcher to feasibly review and analyze. In this work, we use graph neural networks to investigate quiver mutation -- an operation that transforms one quiver (or directed multigraph) into another -- which is central to the theory of cluster algebras with deep connections to geometry, topology, and physics. In the study of cluster algebras, the question of mutation equivalence is of fundamental concern: given two quivers, can one efficiently determine if one quiver can be transformed into the other through a sequence of mutations? Currently, this question has only been resolved in specific cases. In this paper, we use graph neural networks and AI explainability techniques to discover mutation equivalence criteria for the previously unknown case of quivers of type $\tilde{D}_n$. Along the way, we also show that even without explicit training to do so, our model captures structure within its hidden representation that allows us to reconstruct known criteria from type $D_n$, adding to the growing evidence that modern machine learning models are capable of learning abstract and general rules from mathematical data.
     </details>

66. **Math2Sym: A System for Solving Elementary Problems via Large Language Models and Symbolic Solvers** [[pdf]](https://openreview.net/forum?id=eQrkAPcGRF) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Traditional models for solving math word problems (MWPs) often struggle to capture both linguistic context and arithmetic reasoning. We propose Math2Sym, a novel approach integrating large language models (LLMs) with symbolic solvers. This method leverages LLMs' language comprehension and symbolic computation's precision to efficiently convert MWPs into solvable symbolic form. We introduce the EMSF dataset for training models to formalize math problems across various complexities. On our defined test set benchmark, fine-tuned models outperform GPT-3.5 by 17% in few-shot tasks and perform comparably to GPT-4-mini on elementary math problems.
     </details>

67. **MathDSL: A Domain-Specific Language for Concise Mathematical Solutions Via Program Synthesis** [[pdf]](http://arxiv.org/abs/2409.17490) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is demonstrated that a program synthesis system (DreamCoder) using MathDSL can generate programs that solve linear equations with greater accuracy and conciseness than using reinforcement learning systems.
     </details>


     <details>
          <summary>Abstract</summary>
          We present MathDSL, a Domain-Specific Language (DSL) for mathematical equation solving, which, when deployed in program synthesis models, outperforms state-of-the-art reinforcement-learning-based methods. We also introduce a quantitative metric for measuring the conciseness of a mathematical solution and demonstrate the improvement in the quality of generated solutions compared to other methods. Our system demonstrates that a program synthesis system (DreamCoder) using MathDSL can generate programs that solve linear equations with greater accuracy and conciseness than using reinforcement learning systems. Additionally, we demonstrate that if we use the action spaces of previous reinforcement learning systems as DSLs, MathDSL outperforms the action-space-DSLs. We use DreamCoder to store equation-solving strategies as learned abstractions in its program library and demonstrate that by using MathDSL, these can be converted into human-interpretable solution strategies that could have applications in mathematical education.
     </details>

68. **MathPile: A Billion-Token-Scale Pretraining Corpus for Math** [[pdf]](https://arxiv.org/abs/2312.17120) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          High-quality, large-scale corpora are the cornerstone of building foundation models. In this work, we introduce MathPile, a diverse and high-quality math-centric corpus comprising about 9.5 billion tokens. Throughout its creation, we adhered to the principle of “less is more”, firmly believing in the supremacy of data quality over quantity, even in the pre-training phase. Our meticulous data collection and processing efforts included a complex suite of preprocessing, prefiltering, language identification, cleaning, filtering, and deduplication, ensuring the high quality of our corpus. Furthermore, we performed data contamination detection on downstream benchmark test sets to eliminate duplicates and conducted continual pre-training experiments, booting the performance on common mathematical reasoning benchmarks. We aim for our MathPile to boost language models’ mathematical reasoning and plan to open-source its different versions and processing scripts to advance the field.
     </details>

69. **Mining Math Conjectures from LLMs: A Pruning Approach** [[pdf]](https://openreview.net/forum?id=aYlKvzY6ob) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          We present a novel approach to generating mathematical conjectures using Large Language Models (LLMs). Focusing on the solubilizer, a relatively recent construct in group theory, we demonstrate how LLMs such as ChatGPT, Gemini, and Claude can be leveraged to generate conjectures. These conjectures are pruned by allowing the LLMs to generate counterexamples. Our results indicate that LLMs are capable of producing original conjectures that, while not groundbreaking, are either plausible or falsifiable via counterexamples, though they exhibit limitations in code execution.
     </details>

70. **Models Can and Should Embrace the Communicative Nature of Human-Generated Math** [[pdf]](http://arxiv.org/abs/2409.17005) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Math is constructed by people for people: just as natural language corpora reflect not just propositions but the communicative goals of language users, the math data that models are trained on reflects not just idealized mathematical entities but rich communicative intentions. While there are important advantages to treating math in a purely symbolic manner, we here hypothesize that there are benefits to treating math as situated linguistic communication and that language models are well suited for this goal, in ways that are not fully appreciated. We illustrate these points with two case studies. First, we ran an experiment in which we found that language models interpret the equals sign in a humanlike way -- generating systematically different word problems for the same underlying equation arranged in different ways. Second, we found that language models prefer proofs to be ordered in naturalistic ways, even though other orders would be logically equivalent. We advocate for AI systems that learn from and represent the communicative intentions latent in human-generated math.
     </details>

71. **Multi-language Diversity Benefits Autoformalization** [[pdf]](https://neurips.cc/virtual/2024/poster/96799) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Autoformalization is the task of translating natural language materials into machine-verifiable formalisations. Progress in autoformalization research is hindered by the lack of a sizeable dataset consisting of informal-formal pairs expressing the same essence. Existing methods tend to circumvent this challenge by manually curating small corpora or using few-shot learning with large language models. But these methods suffer from data scarcity and formal language acquisition difficulty. In this work, we create mma, a large, flexible, multi-language, and multi-domain dataset of informal-formal pairs, by using a language model to translate in the reverse direction, that is, from formal mathematical statements into corresponding informal ones. Experiments show that language models fine-tuned on mma can produce up to $29-31$\% of statements acceptable with minimal corrections on the miniF2F and ProofNet benchmarks, up from $0$\% with the base model. We demonstrate that fine-tuning on multi-language formal data results in more capable autoformalization models even on single-language tasks.
     </details>

72. **NLIR: Natural Language Intermediate Representation for Mechanized Theorem Proving** [[pdf]](https://openreview.net/forum?id=QzOc0tpdef) `NeurIPS 2024 Workshop MATH-AI` `Coq` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Formal theorem proving is challenging for humans as well as for machines. Thanks to recent advances in LLM capabilities, we believe natural language can serve as a universal interface for reasoning about formal proofs. In this paper, 1) we introduce Pétanque, a new lightweight environment to interact with the Coq theorem prover; 2) we present two interactive proof protocols leveraging natural language as an intermediate representation for designing proof steps; 3) we implement beam search over these interaction protocols, using natural language to rerank proof candidates;  and 4) we use Pétanque to benchmark our search algorithms. Using our method with GPT-4o we can successfully synthesize proofs for 50% of the first 100/260 lemmas from the newly published Busy Beaver proofs.
     </details>

73. **Neuro-Symbolic Data Generation for Math Reasoning** [[pdf]](http://arxiv.org/abs/2412.04857) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          A critical question about Large Language Models (LLMs) is whether their apparent deficiency in mathematical reasoning is inherent, or merely a result of insufficient exposure to high-quality mathematical data. To explore this, we developed an automated method for generating high-quality, supervised mathematical datasets. The method carefully mutates existing math problems, ensuring both diversity and validity of the newly generated problems. This is achieved by a neuro-symbolic data generation framework combining the intuitive informalization strengths of LLMs, and the precise symbolic reasoning of math solvers along with projected Markov chain Monte Carlo sampling in the highly-irregular symbolic space.Empirical experiments demonstrate the high quality of data generated by the proposed method, and that the LLMs, specifically LLaMA-2 and Mistral, when realigned with the generated data, surpass their state-of-the-art counterparts.
     </details>

74. **Not All LLM Reasoners Are Created Equal** [[pdf]](http://arxiv.org/abs/2410.01748) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Overall, LLMs exhibit systematic differences in their reasoning abilities, despite what their performance on standard benchmarks indicates, with a significant reasoning gap in smaller, more cost-efficient, and math-specialized models.
     </details>


     <details>
          <summary>Abstract</summary>
          We study the depth of grade-school math (GSM) problem-solving capabilities of LLMs. To this end, we evaluate their performance on pairs of existing math word problems together so that the answer to the second problem depends on correctly answering the first problem. Our findings reveal a significant reasoning gap in most LLMs, that is performance difference between solving the compositional pairs and solving each question independently. This gap is more pronounced in smaller, more cost-efficient, and math-specialized models. Moreover, instruction-tuning recipes and code generation have varying effects across LLM sizes, while finetuning on GSM can lead to task overfitting. Our analysis indicates that large reasoning gaps are not because of test-set leakage, but due to distraction from additional context and poor second-hop reasoning. Overall, LLMs exhibit systematic differences in their reasoning abilities, despite what their performance on standard benchmarks indicates.
     </details>

75. **OccamLLM: Fast and Exact Language Model Arithmetic in a Single Step** [[pdf]](http://arxiv.org/abs/2406.06576) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a framework that enables exact arithmetic in a single autoregressive step, providing faster, more secure, and more interpretable LLM systems with arithmetic capabilities.
     </details>


     <details>
          <summary>Abstract</summary>
          Despite significant advancements in text generation and reasoning, Large Language Models (LLMs) still face challenges in accurately performing complex arithmetic operations. To achieve accurate calculations, language model systems often enable LLMs to generate code for arithmetic operations. However, this approach compromises speed and security and, if finetuning is involved, risks the language model losing prior capabilities. We propose a framework that enables exact arithmetic in a single autoregressive step, providing faster, more secure, and more interpretable LLM systems with arithmetic capabilities. We use the hidden states of an LLM to control a symbolic architecture which performs arithmetic. Our implementation using Llama 3 8B Instruct with OccamNet as a symbolic model (OccamLlama) achieves 100% accuracy on single arithmetic operations (+, -, *, /, sin, cos, log, exp, sqrt), outperforming GPT 4o and on par with GPT 4o using a code interpreter. OccamLlama also outperforms both Llama 3 8B Instruct and GPT 3.5 Turbo on multistep reasoning problems involving challenging arithmetic, thus enabling small LLMs to match the arithmetic performance of even much larger models. Our code is available at https://anonymous.4open.science/r/OccamLlama.
     </details>

76. **On the Inductive Bias of Stacking Towards Improving Reasoning** [[pdf]](http://arxiv.org/abs/2409.19044) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          An intriguing phenomenon is discovered: MIDAS is not only training-efficient but surprisingly also has an inductive bias towards improving downstream tasks, especially tasks that require reasoning abilities like reading comprehension and math problems, despite having similar or slightly worse perplexity compared to baseline training.
     </details>


     <details>
          <summary>Abstract</summary>
          Given the increasing scale of model sizes, novel training strategies like gradual stacking have garnered interest. Stacking enables efficient training by gradually growing the depth of a model in stages and using layers from a smaller model in an earlier stage to initialize the next stage. Although efficient for training, the model biases induced by such growing approaches is largely unexplored. In this work, we examine this fundamental aspect of gradual stacking, going beyond its efficiency benefits. We propose a variant of gradual stacking called MIDAS and discover an intriguing phenomenon for this approach: MIDAS is not only training efficient, but surprisingly also has an inductive bias towards improving downstream tasks, especially tasks that require reasoning abilities, despite having similar or slightly worse perplexity compared to baseline training. To further analyze this inductive bias, we construct {\em reasoning primitives} – simple synthetic tasks that are building blocks for reasoning – and find that a model pretrained with stacking is significantly better than standard pretraining on these primitives, with and without fine-tuning. This provides stronger and more robust evidence for this inductive bias towards reasoning. Furthermore, we conjecture the underlying reason for this inductive bias by exploring the connection of stacking to looped models and provide strong supporting empirical analysis.
     </details>

77. **OpenMathInstruct-2: Accelerating AI for Math with Massive Open-Source Instruction Data** [[pdf]](http://arxiv.org/abs/2410.01560) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The OpenMathInstruct-2 dataset is created, which consists of 14M question-solution pairs, making it nearly eight times larger than the previous largest open-source math reasoning dataset, and is released under a commercially permissive license.
     </details>


     <details>
          <summary>Abstract</summary>
          Mathematical reasoning continues to be a critical challenge in large language model (LLM) development with significant interest. However, most of the cutting-edge progress in mathematical reasoning with LLMs has become \emph{closed-source} due to lack of access to training data. This lack of data access limits researchers from understanding the impact of different choices for synthesizing and utilizing the data. With the goal of creating a high-quality finetuning (SFT) dataset for math reasoning, we conduct careful ablation experiments on data synthesis using the recently released \texttt{Llama3.1} family of models. Our experiments show that: (a) solution format matters, with excessively verbose solutions proving detrimental to SFT performance, (b) data generated by a strong teacher outperforms equally-sized data generated by a weak student model, (c) SFT is robust to low-quality solutions, allowing for imprecise data filtering, and (d) question diversity is crucial for achieving data scaling gains. Based on these insights, we create the OpenMathInstruct-2 dataset, which consists of 14M question-solution pairs ($\approx$ 600K unique questions), making it nearly eight times larger than the previous largest open-source math reasoning dataset. Finetuning the \texttt{Llama-3.1-8B-Base} using OpenMathInstruct-2 outperforms \texttt{Llama3.1-8B-Instruct} on MATH by an absolute 15.9\% (51.9\% $\rightarrow$ 67.8\%). Finally, to accelerate the open-source efforts, we release the code, the finetuned models, and the OpenMathInstruct-2 dataset under a commercially permissive license.
     </details>

78. **Pretrained Large Language Models Use Fourier Features to Compute Addition** [[pdf]](https://neurips.cc/virtual/2024/poster/94033) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Pre-trained large language models (LLMs) exhibit impressive mathematical reasoning capabilities, yet how they compute basic arithmetic, such as addition, remains unclear. This paper shows that pre-trained LLMs add numbers using Fourier features---dimensions in the hidden state that represent numbers via a set of features sparse in the frequency domain. Within the model, MLP and attention layers use Fourier features in complementary ways: MLP layers primarily approximate the magnitude of the answer using low-frequency features, while attention layers primarily perform modular addition (e.g., computing whether the answer is even or odd) using high-frequency features.Pre-training is crucial for this mechanism: models trained from scratch to add numbers only exploit low-frequency features, leading to lower accuracy.Introducing pre-trained token embeddings to a randomly initialized model rescues its performance.Overall, our analysis demonstrates that appropriate pre-trained representations (e.g., Fourier features) can unlock the ability of Transformers to learn precise mechanisms for algorithmic tasks.
     </details>

79. **Probabilistic Proof State Compression: Optimizing LLM-Guided Formal Verification** [[pdf]](https://openreview.net/forum?id=x2yiUEH0f9) `NeurIPS 2024 Workshop MATH-AI` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          While approaches to Large Language Model guided formal proof search have recently seen immense success, the scalability of these techniques is often hindered by the explosive growth of the proof search space. This paper introduces a novel approach that synergistically combines LLMs with conformal prediction techniques to guide and optimize formal proof search. We present a method for compressing the proof state space using adaptive, probability-based binning, informed by conformal prediction intervals. This compression technique significantly reduces the computational resources required for proof search while maintaining statistical guarantees on proof discovery. In addition, we present preliminary empirical results on a subset of the Lean 4 miniF2F test demonstrating the effectiveness of this method leading to a 23\% average reduction in proof search time compared to baseline open models, while maintaining comparable proof success rates.
     </details>

80. **Putnam-AXIOM: A Functional and Static Benchmark for Measuring Higher Level Mathematical Reasoning** [[pdf]](https://openreview.net/forum?id=YXnwlZe0yf) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          As large language models (LLMs) continue to advance, many existing benchmarks designed to evaluate their reasoning capabilities are becoming less challenging. These benchmarks, though foundational, no longer offer the complexity necessary to evaluate the cutting edge of artificial reasoning. In this paper, we present the Putnam-AXIOM Original benchmark, a dataset of 236 challenging problems from the William Lowell Putnam Mathematical Competition, along with detailed step-by-step solutions. To address the potential data contamination of Putnam problems, we create functional variations for 53 problems in Putnam-AXIOM. We see that most models get a significantly lower accuracy on the variations than the original problems. Even so, our results reveal that Claude-3.5 Sonnet, the best-performing model, achieves 15.96\% accuracy on the Putnam-AXIOM original but experiences more than a 50\% reduction in accuracy on the variations dataset when compared to its performance on corresponding original problems.
     </details>

81. **Reasoning in Reasoning: A Hierarchical Framework for (Better and Faster) Neural Theorem Proving** [[pdf]](https://openreview.net/forum?id=H5hePMXKht) `NeurIPS 2024 Workshop MATH-AI` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Learning to do complex reasoning is the central objective of artificial intelligence. Autoregressive language models have shown promise in generating intermediate steps for problem solving; however, complex reasoning tasks such as theorem proving still present challenges due to the vast search spaces. Classical works have considered reasoning by searching, e.g., expanding the reasoning space with tree search to explore intermediate steps, and reasoning by decomposing, i.e., breaking down the problem into higher-level thoughts that prompt lower-level steps. We develop Reasoning in Reasoning (RiR), a hierarchical framework that unifies strategic problem decomposing with goal-driven reasoning step generation and search, via a planner-actor game. Using neural theorem proving as a representative task, our approach breaks down complex theorem problems into achievable subgoals, giving models: (i) improved generalizability for reasoning step generation, (ii) a more compact and informative search space for reasoning paths, and (iii) an efficient mechanism for learning to plan. We support RiR via an information-theoretic analysis, and show it achieves state-of-the-art performance and efficiency on popular theorem proving benchmarks including LeanDojo and miniF2F.
     </details>

82. **Recursive Introspection: Teaching Foundation Model Agents How to Self-Improve** [[pdf]](https://neurips.cc/virtual/2024/poster/96089) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          A central piece in enabling intelligent agentic behavior in foundation models is to make them capable of introspecting upon their behavior, to reason and correct their mistakes. Even strong proprietary large language models (LLMs) do not exhibit the ability of continually improving their responses sequentially, even in scenarios where they are explicitly told that they are making a mistake. In this paper, we develop $\textbf{RISE}$: $\textbf{R}$ecursive $\textbf{I}$ntro$\textbf{s}$p$\textbf{e}$ction, an approach for fine-tuning LLMs to introduce this ability. Our approach prescribes an iterative fine-tuning procedure, which attempts to teach the model how to alter its response after having seen previously unsuccessful attempts to solve a problem with additional environment feedback. RISE poses fine-tuning for a single-turn problem as solving a multi-turn Markov decision process (MDP), where the initial state is the prompt. Inspired by principles in online imitation learning, we derive effective strategies to dictate multi-turn data collection and training so as to imbue in an LLM the capability to recursively detect and correct its previous mistakes in subsequent iterations. Our experiments show that $\textbf{RISE}$ enables 7B Llama2 and Mistral models to improve themselves with more turns on math reasoning tasks, outperforming several single-turn strategies given an equal amount of inference-time computation. Our analysis shows that RISE makes meaningful improvements to responses to arrive at the correct solution for challenging prompts, without disrupting one-turn abilities.
     </details>

83. **Repeated examples help learn arithmetic** [[pdf]](https://openreview.net/forum?id=qoUHqnE6A0) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          We study small transformers trained on two problems of arithmetic: the greatest common divisor (GCD) and modular multiplication, and show that models trained on a limited set of repeated examples achieve better performance than models trained from unlimited data. In fact, modular multiplication is only learned on small training sets. We also demonstrate that two-set training - repeated use of a small random subset of examples, along normal sampling on the rest of the training set - provides for faster learning and better performance. These experiments highlight that the benefits of repetition can outweigh those of data diversity; and shed light on the still poorly understood interplay between generalization and memorization in deep learning.
     </details>

84. **SBI-RAG: Enhancing Math Word Problem Solving for Students through Schema-Based Instruction and Retrieval-Augmented Generation.** [[pdf]](https://openreview.net/forum?id=fn0pQWkFsP) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Many students struggle with math word problems (MWPs), often finding it difficult to identify key information and select the appropriate mathematical operations. Schema-based instruction (SBI) is an evidence-based strategy that helps students categorize problems based on their structure, improving problem-solving accuracy. Building on this, we propose a Schema-Based Instruction Retrieval-Augmented Generation (SBI-RAG) framework that incorporates a large language model (LLM). Our approach emphasizes step-by-step reasoning by leveraging schemas to guide solution generation. We evaluate its performance on the GSM8K dataset, comparing it with GPT-4 and GPT-3.5 Turbo, and introduce a "reasoning score" metric to assess solution quality. Our findings suggest that SBI-RAG enhances reasoning clarity and problem-solving accuracy, potentially providing educational benefits for students.
     </details>

85. **SBSC: Step-by-Step Coding for Improving Mathematical Olympiad Performance** [[pdf]](https://openreview.net/forum?id=wSkvf2WyYz) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          We propose Step-by-Step Coding (SBSC): a multi-turn math reasoning framework that enables Large Language Models (LLMs) to generate sequence of programs for solving Olympiad level math problems. After each turn/step, by leveraging the code execution outputs and programs of previous steps, the model generates the next sub-task and the corresponding program to complete it. SBSC allows more granular, flexible and precise approach to problem-solving compared to existing methods. Extensive experiments highlight the effectiveness of SBSC in tackling competition and Olympiad-level math problems. For Claude-3.5-Sonnet, we observe SBSC (greedy decoding) surpasses existing state-of-the-art (SOTA) program generation based reasoning strategies by absolute 10.7% on AMC12, 8% on AIME and 12.6% on MathOdyssey. Given SBSC is multi-turn in nature, we also benchmark SBSC’s greedy decoding against self-consistency decoding results of existing SOTA math reasoning strategies and observe performance gain by absolute 6.2% on AMC, 6.7% on AIME and 7.4% on MathOdyssey.
     </details>

86. **STEM-POM: Evaluating Language Models Math-Symbol Reasoning in Document Parsing** [[pdf]](https://openreview.net/forum?id=jErJ8kansp) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduces STEM-PoM, a comprehensive benchmark dataset designed to evaluate LLMs' reasoning abilities on math symbols within contextual scientific text and shows that state-of-the-art LLMs achieve an average of 20-60% accuracy under in-context learning and 50-60% accuracy with fine-tuning, revealing a significant gap in their mathematical reasoning capabilities.
     </details>


     <details>
          <summary>Abstract</summary>
          Advances in large language models (LLMs) have spurred research into enhancing their reasoning capabilities, particularly in math-rich STEM documents. While LLMs can generate equations or solve math-related queries, their ability to fully understand and interpret abstract mathematical symbols in long, math-rich documents remains limited. In this paper, we introduce STEM-PoM, a comprehensive benchmark dataset designed to evaluate LLMs' reasoning abilities on math symbols within contextual scientific text. The dataset, sourced from real-world ArXiv documents, contains over 2K math symbols classified as main attributes of variables, constants, operators, and unit descriptors, with additional sub-attributes including scalar/vector/matrix for variables and local/global/discipline-specific labels for both constants and operators. Our extensive experiments show that state-of-the-art LLMs achieve an average of 20-60% accuracy under in-context learning and 50-60% accuracy with fine-tuning, revealing a significant gap in their mathematical reasoning capabilities. STEM-PoM fuels future research of developing advanced Math-AI models that can robustly handle math symbols.
     </details>

87. **Scaling Inference Computation: Compute-Optimal Inference for Problem-Solving with Language Models** [[pdf]](https://openreview.net/forum?id=j7DZWSc8qu) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Increased computation generally leads to improved intelligence, but the question of how to configure LLMs during inference to optimally utilize the available computation budget remains underexplored. This paper investigates inference scaling laws: increasing inference computation leads to higher task performance and explores the compute-optimal inference: designing models and inference strategies that optimally trade off additional inference-time compute for improved performance. As the first step toward understanding these scaling laws and designing compute-efficient inference methods, we analyze the trade-offs between model parameters and the number of decoding tokens, measured in FLOPs, while comparing tree search variants to sampling-based methods. We found that a smaller language model with a novel tree search algorithm typically achieves a Pareto-optimal trade-off. These results highlight the potential benefits of deploying smaller models equipped with more sophisticated decoding algorithms in budget-constrained scenarios, e.g., on end-devices, to enhance problem-solving accuracy.
     </details>

88. **SciInstruct: a Self-Reflective Instruction Annotated Dataset for Training Scientific Language Models** [[pdf]](https://arxiv.org/abs/2401.07950) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have shown promise in assisting scientific discovery. However, such applications are currently limited by LLMs' deficiencies in understanding intricate scientific concepts, deriving symbolic equations, and solving advanced numerical calculations. To bridge these gaps, we introduce SciInstruct, a suite of scientific instructions for training scientific language models capable of college-level scientific reasoning. Central to our approach is a novel self-reflective instruction annotation framework to address the data scarcity challenge in the science domain. This framework leverages existing LLMs to generate step-by-step reasoning for unlabelled scientific questions, followed by a process of self-reflective critic-and-revise. Applying this framework, we curated a diverse and high-quality dataset encompassing physics, chemistry, math, and formal proofs. We analyze the curated SciInstruct from multiple interesting perspectives (e.g., domain, scale, source, question type, answer length, etc.). To verify the effectiveness of SciInstruct, we fine-tuned different language models with SciInstruct, i.e., ChatGLM3 (6B and 32B), Llama3-8b-Instruct, and Mistral-7B, enhancing their scientific and mathematical reasoning capabilities, without sacrificing the language understanding capabilities of the base model. We release code and SciInstruct at https://github.com/THUDM/SciGLM.
     </details>

89. **Solving Intricate Problems with Human-like Decomposition and Rethinking** [[pdf]](https://neurips.cc/virtual/2024/poster/95441) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          In this paper, we introduce a novel reasoning framework DeAR (Decompose-Analyze-Rethink) for large language models (LLMs) to conduct intricate reasoning. Our key idea is inspired by human cognitive reasoning, which refines complex problem-solving by breaking it down into sub-questions within a Reasoning Tree and then updating prior answers based on the responses to these sub-questions. In our framework, we propose a Decompose-Analyze-Rethink cycle, which gradually forms a reasoning tree guiding the reasoning process. Specifically, given the problem, the Decompose stage introduces a prompt-based method to break it into simpler sub-ones at subsequent tree nodes. Then, the Analyze stage generates and self-checks the rationales at the node level. Last, the Rethink stage updates the rationales of parent nodes based on its children's feedback. Our reasoning paradigm is more flexible than state-of-the-art methods including Tree-of-Thoughts (ToT), and Graph-of-Thoughts (GoT), as each branch is autonomously generated without fixed settings, and moreover, allows for timely and globally rationale correction throughout the entire process. We conduct extensive experiments on three reasoning benchmarks including ScienceQA, StrategyQA, and GSM8K. Experimental results show that our approach can significantly reduce logical errors and enhance the performance with different LLMs. Our codes are available at: https://anonymous.4open.science/r/Coarse-to-Fine-F216/.
     </details>

90. **Synchronizing Verbal Responses and Board Writing for Cross-Modal Math Teaching with LLMs** [[pdf]](https://openreview.net/forum?id=esbIrV8N12) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          The advancement of Large Language Models (LLMs) has significantly contributed to the enhancement of mathematics learning, with the generated text serving as verbal responses to address students' questions. However, in educational practice, teachers often synchronize verbal responses with Board Writing (BW) to better facilitate students' knowledge construction. To this end, we propose MathBoard, a system designed for elementary mathematics instruction, which employs an LLM-driven multi-agent system to progressively generate BW. We developed a learning strategy for MathBoard grounded in Schema Theory, thereby improving the system's scalability and adaptability.   Our research focuses on providing learners with BW to effectively reduce cognitive load. This approach is complementary to existing methods aimed at enhancing mathematical reasoning skills and can be used in conjunction with them.   An empirical study involving 34 pre-service teachers demonstrates that the cross-modal interaction provided by MathBoard was perceived to be more effective and engaging across various dimensions compared to interactions guided solely by verbal responses. This cross-modal approach significantly fosters learners' social construction of knowledge. The link to the project will be publicly accessible.
     </details>

91. **Synthesizing Verified Mathematical Problems** [[pdf]](https://openreview.net/forum?id=L5US093OwO) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Mathematical data synthesis offers a potential effective solution for enhancing the mathematical capabilities of large language models. However, existing approaches either restrict data diversity or compromise accuracy. This paper introduces a novel method, Mathematical Data Synthesis through Algorithmic \textbf{A}bstraction, \textbf{I}mplementation, and \textbf{C}ontextualization (AIC), to generate new and accurate mathematical problem. AIC abstracts mathematical problems into algorithms, implements these algorithms as code functions, and contextualizes them to create new problems, which are then verified using code. This approach addresses the key limitations of current LLM-based mathematical data synthesis methods, and also have a significant performance improvement compare to other baselines. Our code is available at https://anonymous.4open.science/r/AIC-9684.
     </details>

92. **The Karp Dataset** [[pdf]](https://openreview.net/forum?id=RtTNbJthjV) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Understanding the mathematical reasoning capabilities of Large Language Models (LLMs) is a central topic in the study of artificial intelligence. This new domain necessitates the creation of datasets of reasoning tasks for both training and benchmarking the performance of LLMs. To this end, we introduce the Karp dataset: The first dataset composed of detailed proofs of NP-completeness reductions. The reductions vary in difficulty, ranging from simple exercises of undergraduate courses to more challenging reductions from academic papers. We compare the performance of state-of-the-art models on this task and demonstrate the effect of fine-tuning with the Karp dataset on reasoning capacity.
     </details>

93. **Unsupervised Discovery of Formulas for Mathematical Constants** [[pdf]](http://arxiv.org/abs/2412.16818) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          In recent years, we are witnessing a rise of AI and machine learning methods for scientific discovery and hypothesis creation. Despite the strides in other fields of science, a persistent challenge lies in the creation of formulas for mathematical constants.In the landscape of formula creation, there is no straightforward ‘’distance metric'' between two samples that can guide progress. Formulas are either true or false, with no continuous adjustments that can enhance their correctness.The absence of a systematic method left the realm of formula discovery elusive for automated methods. In this work, we propose a systematic methodology for categorization, characterization, and pattern identification of such formulas. We demonstrate this methodology on Polynomial Continued Fraction formulas, which are ubiquitous in their intrinsic connections to mathematical constants, and generalize many mathematical functions and structures.We discover organizing metrics for the space of polynomial continued fractions. We test our methodology on a set of 1,768,900 such formulas, identifying many known formulas for mathematical constants, and discover previously unknown formulas for $\pi$, $\ln(2)$, Gauss, and Lemniscate constants. The uncovered patterns enable a direct generalization of individual formulas to infinite families, unveiling rich mathematical structures. This success paves the way towards a generative model that creates continued fractions fulfilling requested mathematical properties, potentially accelerating by orders of magnitude the rate of discovery of useful formulas.
     </details>

94. **VinePPO: Accurate Credit Assignment in RL for LLM Mathematical Reasoning** [[pdf]](https://openreview.net/forum?id=KqALqWJSbF) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) are increasingly required to solve complex reasoning tasks, like mathematical problems, that involve multiple reasoning steps before feedback is received. Effectively identifying and prioritizing key steps by accurately assigning credit to these intermediate steps is essential for enhancing model performance. Proximal Policy Optimization (PPO), a state-of-the-art reinforcement learning algorithm for finetuning LLMs, addresses the credit assignment problem by employing value networks to predict the expected cumulative rewards of intermediate states. In this work, we identify significant limitations with this value estimation method. To address this, we propose \methodname that leverages the flexibility of language environments to compute unbiased Monte Carlo-based estimates of the intermediate values. VinePPO consistently outperforms standard PPO, doing so more efficiently and with lower divergence from the reference model. Our findings underscore the critical importance of accurate credit assignment in LLM post-training and present a simple, yet effective solution.
     </details>

95. **When and How Does Synthetic Data Improve Reasoning Capabilities of Language Models?** [[pdf]](https://neurips.cc/virtual/2024/poster/96295) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Training on model-generated synthetic data is a promising approach for finetuning LLMs, but it remains unclear when it helps or hurts. In this paper, we investigate this for reasoning problems via an empirical study, followed by a theoretical formalization of our observations. First, we find that while the typical approach of finetuning a model on synthetic correct or positive problem-solution pairs generated by capable models offers modest performance gains, sampling more correct solutions from the finetuned learner doubles the sample efficiency of synthetic data. At the same time, training on model-generated positives can amplify various spurious  correlations, resulting in flat or even inverse scaling trends as the amount of data increases. Surprisingly, we find that several of these issues can be addressed if we also utilize negative responses, i.e. model-generated responses that are deemed incorrect via final answer checking. Crucially, these negatives must be constructed such that the training can appropriately recover the utility or credit of each intermediate step in the negative response. With this per-step scheme, we are able to attain consistent gains over only positive data, attaining performance similar to amplifying the amount of synthetic data by 8x. We show that training on per-step negatives can help to unlearn spurious correlations in the positive data, and is equivalent to advantage-weighted reinforcement learning (RL), implying that it inherits benefits of RL over imitating positive data alone.
     </details>

96. **Wu’s Method Boosts Symbolic AI to Rival Silver Medalists and AlphaGeometry to Outperform Gold Medalists at IMO Geometry** [[pdf]](https://openreview.net/forum?id=aKRtC45gle) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Proving geometric theorems constitutes a hallmark of reasoning combining intuitive, visual, and logical skills, making automated theorem proving of Olympiad-level geometry problems a milestone for human-level automated reasoning. AlphaGeometry, a neuro-symbolic model trained with 100M synthetic samples, solved 25 of 30 International Mathematical Olympiad (IMO) problems. It marked a major breakthrough compared to the reported baseline using Wu's method which solved only 10. Revisiting the IMO-AG-30 benchmark, we find that Wu's method is surprisingly strong and solves 15 problems, including some unsolved by other methods. This leads to two key findings: (i) Combining Wu's method with the classic synthetic methods of deductive databases and angle, ratio & distance chasing solves 21 out of 30 problems on a CPU-only laptop limited to 5 minutes per problem. Essentially, this classic method solves just 4 fewer problems than AlphaGeometry and establishes the first *fully symbolic* baseline that rivals the performance of IMO silver medalists. (ii) Wu's method even solves 2 of the 5 problems that AlphaGeometry failed on. Combining both, we set a new state-of-the-art for automated theorem proving on IMO-AG-30 solving 27 out of 30 problems - the first AI method which outperforms an IMO gold medalist.
     </details>

