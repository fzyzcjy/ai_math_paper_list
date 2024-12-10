# Recent ArXiv 

ArXiv papers in recent months, sorted by time.

1. **Enhancing Mathematical Reasoning in LLMs with Background Operators** [[pdf]](http://arxiv.org/abs/2412.04110) `2024-12-05` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes utilizing background operators for mathematical reasoning in large language models by developing a Prolog solution that includes problem-specific predicates and intermediate predicates derived from these background operators, ensuring that each solution adheres to the defined operator set.
     </details>


     <details>
          <summary>Abstract</summary>
          We propose utilizing background operators for mathematical reasoning in large language models (LLMs). To achieve this, we define a set of fundamental mathematical predicates as the basic building blocks. For each mathematical problem, we develop a Prolog solution that includes problem-specific predicates and intermediate predicates derived from these background operators, ensuring that each solution adheres to the defined operator set. We introduce the MATH-Prolog corpus, which is derived from the counting and probability categories of the MATH corpus. For efficient data augmentation, we apply K-fold cross-validated self-training. This method incrementally generates new Prolog solutions for each fold, incorporating those verified as correct into the training set throughout the model training process. Our experimental results demonstrate that 5-fold crossvalidated self-training effectively identifies new, accurate Prolog solutions, achieving an accuracy of 84.6% on the cross-validated set, and 84.8% on the test set during fine-tuning the Meta-Llama-3.1-8B-Instruct model. This approach successfully uncovers new solutions with fully computable inference steps for previously unseen problems. Additionally, incorporating the background mathematical predicates into the prompt enhances solution coverage.
     </details>

2. **Evolutionary Pre-Prompt Optimization for Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2412.04291) `2024-12-05` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper explores the optimization of example selection for designing effective CoT pre-prompts and shows that the choice of the optimization algorithm, typically in favor of comparison-based methods such as evolutionary computation, significantly enhances efficacy and feasibility.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent advancements have highlighted that large language models (LLMs), when given a small set of task-specific examples, demonstrate remarkable proficiency, a capability that extends to complex reasoning tasks. In particular, the combination of few-shot learning with the chain-of-thought (CoT) approach has been pivotal in steering models towards more logically consistent conclusions. This paper explores the optimization of example selection for designing effective CoT pre-prompts and shows that the choice of the optimization algorithm, typically in favor of comparison-based methods such as evolutionary computation, significantly enhances efficacy and feasibility. Specifically, thanks to a limited exploitative and overfitted optimization, Evolutionary Pre-Prompt Optimization (EPPO) brings an improvement over the naive few-shot approach exceeding 10 absolute points in exact match scores on benchmark datasets such as GSM8k and MathQA. These gains are consistent across various contexts and are further amplified when integrated with self-consistency (SC)
     </details>

3. **MTMT: Consolidating Multiple Thinking Modes to Form a Thought Tree for Strengthening LLM** [[pdf]](http://arxiv.org/abs/2412.03987) `2024-12-05` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          MT (Multi-thinking Modes Tree) is introduced, a novel method that interacts with LLMs to construct a thought tree, simulating various advanced cognitive processes, including but not limited to association, counterfactual thinking, task decomposition, and comparison.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have shown limitations in tasks requiring complex logical reasoning and multi-step problem-solving. To address these challenges, researchers have employed carefully designed prompts and flowcharts, simulating human cognitive processes to enhance LLM performance, such as the Chain of Thought approach. In this paper, we introduce MTMT (Multi-thinking Modes Tree), a novel method that interacts with LLMs to construct a thought tree, simulating various advanced cognitive processes, including but not limited to association, counterfactual thinking, task decomposition, and comparison. By breaking down the original complex task into simpler sub-questions, MTMT facilitates easier problem-solving for LLMs, enabling more effective utilization of the latent knowledge within LLMs. We evaluate the performance of MTMT under different parameter configurations, using GPT-4o mini as the base model. Our results demonstrate that integrating multiple modes of thinking significantly enhances the ability of LLMs to handle complex tasks.
     </details>

4. **Artificial Expert Intelligence through PAC-reasoning** [[pdf]](http://arxiv.org/abs/2412.02441) `2024-12-03` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          AEI introduces a framework for ``Probably Approximately Correct (PAC) Reasoning", which provides robust theoretical guarantees for reliably decomposing complex problems, with a practical mechanism for controlling reasoning precision.
     </details>


     <details>
          <summary>Abstract</summary>
          Artificial Expert Intelligence (AEI) seeks to transcend the limitations of both Artificial General Intelligence (AGI) and narrow AI by integrating domain-specific expertise with critical, precise reasoning capabilities akin to those of top human experts. Existing AI systems often excel at predefined tasks but struggle with adaptability and precision in novel problem-solving. To overcome this, AEI introduces a framework for ``Probably Approximately Correct (PAC) Reasoning". This paradigm provides robust theoretical guarantees for reliably decomposing complex problems, with a practical mechanism for controlling reasoning precision. In reference to the division of human thought into System 1 for intuitive thinking and System 2 for reflective reasoning~\citep{tversky1974judgment}, we refer to this new type of reasoning as System 3 for precise reasoning, inspired by the rigor of the scientific method. AEI thus establishes a foundation for error-bounded, inference-time learning.
     </details>

5. **Mind the Gap: Examining the Self-Improvement Capabilities of Large Language Models** [[pdf]](http://arxiv.org/abs/2412.02674) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Self-improvement is a mechanism in Large Language Model (LLM) pre-training, post-training and test-time inference. We explore a framework where the model verifies its own outputs, filters or reweights data based on this verification, and distills the filtered data. Despite several empirical successes, a fundamental understanding is still lacking. In this work, we initiate a comprehensive, modular and controlled study on LLM self-improvement. We provide a mathematical formulation for self-improvement, which is largely governed by a quantity which we formalize as the generation-verification gap. Through experiments with various model families and tasks, we discover a scaling phenomenon of self-improvement -- a variant of the generation-verification gap scales monotonically with the model pre-training flops. We also examine when self-improvement is possible, an iterative self-improvement procedure, and ways to improve its performance. Our findings not only advance understanding of LLM self-improvement with practical implications, but also open numerous avenues for future research into its capabilities and boundaries.
     </details>

6. **Free Process Rewards without Process Labels** [[pdf]](http://arxiv.org/abs/2412.01981) `2024-12-02` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The implicit PRM, when instantiated with the cross-entropy (CE) loss, is more data-efficient and can keep improving generation models even when trained with only one response per instruction, the setup that suffers from extreme data scarcity and imbalance.
     </details>


     <details>
          <summary>Abstract</summary>
          Different from its counterpart outcome reward models (ORMs), which evaluate the entire responses, a process reward model (PRM) scores a reasoning trajectory step by step, providing denser and more fine grained rewards. However, training a PRM requires labels annotated at every intermediate step, presenting significant challenges for both manual and automatic data collection. This paper aims to address this challenge. Both theoretically and empirically, we show that an \textit{implicit PRM} can be obtained at no additional cost, by simply training an ORM on the cheaper response-level labels. The only assumption is to parameterize the outcome reward as the log-likelihood ratios of the policy and reference models, which can be optimized regardless of the specific choice of loss objectives. In experiments, we instantiate our implicit PRMs with various objectives and evaluate their performance on MATH. We show that our implicit PRM outperforms a strong MCTS-based baseline \textit{\'a la} Math-Shepherd using less than $1/38$ of the training data. Its performance can be further improved with majority voting. We further find that scaling up instructions and responses benefits our implicit PRM, and the latter brings a larger gain. Particularly, we find that our implicit PRM, when instantiated with the cross-entropy (CE) loss, is more data-efficient and can keep improving generation models even when trained with only one response per instruction, the setup that suffers from extreme data scarcity and imbalance. Further, instructions should be relevant to downstream tasks while the diversity of responses does not bring gains. Surprisingly, training on extra Math-Shepherd step labels brings no further improvements to our implicit PRM trained on only outcome data. We hope that our work will encourage a rethinking of PRM training approaches and contribute to making training PRMs more accessible.
     </details>

7. **MALT: Improving Reasoning with Multi-Agent LLM Training** [[pdf]](http://arxiv.org/abs/2412.01928) `2024-12-02` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a sequential multi-agent setup with heterogeneous LLMs assigned specialized roles: a generator, verifier, and refinement model iteratively solving problems and proposes a trajectory-expansion-based synthetic data generation process and a credit assignment strategy driven by joint outcome based rewards.
     </details>


     <details>
          <summary>Abstract</summary>
          Enabling effective collaboration among LLMs is a crucial step toward developing autonomous systems capable of solving complex problems. While LLMs are typically used as single-model generators, where humans critique and refine their outputs, the potential for jointly-trained collaborative models remains largely unexplored. Despite promising results in multi-agent communication and debate settings, little progress has been made in training models to work together on tasks. In this paper, we present a first step toward "Multi-agent LLM training" (MALT) on reasoning problems. Our approach employs a sequential multi-agent setup with heterogeneous LLMs assigned specialized roles: a generator, verifier, and refinement model iteratively solving problems. We propose a trajectory-expansion-based synthetic data generation process and a credit assignment strategy driven by joint outcome based rewards. This enables our post-training setup to utilize both positive and negative trajectories to autonomously improve each model's specialized capabilities as part of a joint sequential system. We evaluate our approach across MATH, GSM8k, and CQA, where MALT on Llama 3.1 8B models achieves relative improvements of 14.14%, 7.12%, and 9.40% respectively over the same baseline model. This demonstrates an early advance in multi-agent cooperative capabilities for performance on mathematical and common sense reasoning questions. More generally, our work provides a concrete direction for research around multi-agent LLM training approaches.
     </details>

8. **Think-to-Talk or Talk-to-Think? When LLMs Come Up with an Answer in Multi-Step Reasoning** [[pdf]](http://arxiv.org/abs/2412.01113) `2024-12-02` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Through causal probing experiments in controlled arithmetic reasoning tasks, this study finds systematic internal reasoning patterns across models; for example, simple subproblems are solved before CoT begins, and more complicated multi-hop calculations are performed during CoT.
     </details>


     <details>
          <summary>Abstract</summary>
          This study investigates the internal reasoning mechanism of language models during symbolic multi-step reasoning, motivated by the question of whether chain-of-thought (CoT) outputs are faithful to the model's internals. Specifically, we inspect when they internally determine their answers, particularly before or after CoT begins, to determine whether models follow a post-hoc "think-to-talk" mode or a step-by-step "talk-to-think" mode of explanation. Through causal probing experiments in controlled arithmetic reasoning tasks, we found systematic internal reasoning patterns across models; for example, simple subproblems are solved before CoT begins, and more complicated multi-hop calculations are performed during CoT.
     </details>

9. **Improving Multimodal LLMs Ability In Geometry Problem Solving, Reasoning, And Multistep Scoring** [[pdf]](http://arxiv.org/abs/2412.00846) `2024-12-01` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          GPSM4K encompasses 2157 multimodal question-answer pairs manually extracted from mathematics textbooks spanning grades 7-12 and is further augmented to 5340 problems, consisting of both numerical and theorem-proving questions, facilitating a comprehensive evaluation of problem-solving approaches.
     </details>


     <details>
          <summary>Abstract</summary>
          This paper presents GPSM4K, a comprehensive geometry multimodal dataset tailored to augment the problem-solving capabilities of Large Vision Language Models (LVLMs). GPSM4K encompasses 2157 multimodal question-answer pairs manually extracted from mathematics textbooks spanning grades 7-12 and is further augmented to 5340 problems, consisting of both numerical and theorem-proving questions. In contrast to PGPS9k, Geometry3K, and Geo170K which feature only objective-type questions, GPSM4K offers detailed step-by-step solutions in a consistent format, facilitating a comprehensive evaluation of problem-solving approaches. This dataset serves as an excellent benchmark for assessing the geometric reasoning capabilities of LVLMs. Evaluation of our test set shows that there is scope for improvement needed in open-source language models in geometry problem-solving. Finetuning on our training set increases the geometry problem-solving capabilities of models. Further, We also evaluate the effectiveness of techniques such as image captioning and Retrieval Augmentation generation (RAG) on model performance. We leveraged LLM to automate the task of final answer evaluation by providing ground truth and predicted solutions. This research will help to assess and improve the geometric reasoning capabilities of LVLMs.
     </details>

10. **Reverse Thinking Makes LLMs Stronger Reasoners** [[pdf]](http://arxiv.org/abs/2411.19865) `2024-11-29` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Reverse thinking plays a crucial role in human reasoning and to enable Large Language Models (LLMs) to perform reverse thinking, RevThink is introduced, a framework composed of data augmentation and learning objectives that outperforms a standard fine-tuning method trained on 10x more forward reasoning.
     </details>


     <details>
          <summary>Abstract</summary>
          Reverse thinking plays a crucial role in human reasoning. Humans can reason not only from a problem to a solution but also in reverse, i.e., start from the solution and reason towards the problem. This often enhances overall reasoning performance as it enables consistency checks between their forward and backward thinking. To enable Large Language Models (LLMs) to perform reverse thinking, we introduce Reverse-Enhanced Thinking (RevThink), a framework composed of data augmentation and learning objectives. In RevThink, we augment the dataset by collecting structured forward-backward reasoning from a teacher model, consisting of: (1) the original question, (2) forward reasoning, (3) backward question, and (4) backward reasoning. We then employ three objectives to train a smaller student model in a multi-task learning fashion: (a) generate forward reasoning from a question, (b) generate a backward question from a question, and (c) generate backward reasoning from the backward question. Experiments across 12 datasets covering commonsense, math, and logical reasoning show an average 13.53% improvement over the student model's zero-shot performance and a 6.84% improvement over the strongest knowledge distillation baselines. Moreover, our method demonstrates sample efficiency -- using only 10% of the correct forward reasoning from the training data, it outperforms a standard fine-tuning method trained on 10x more forward reasoning. RevThink also exhibits strong generalization to out-of-distribution held-out datasets.
     </details>

11. **A Lean Dataset for International Math Olympiad: Small Steps towards Writing Math Proofs for Hard Problems** [[pdf]](http://arxiv.org/abs/2411.18872) `2024-11-28` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work writes complete, original formal proofs for the remaining 13 IMO problems in Lean along with 3 extra problems from IMO 2022 and 2023, and expands the availability of proof currently in the public domain by creating 5,150 lines of Lean proof.
     </details>


     <details>
          <summary>Abstract</summary>
          Using AI to write formal proofs for mathematical problems is a challenging task that has seen some advancements in recent years. Automated systems such as Lean can verify the correctness of proofs written in formal language, yet writing the proofs in formal language can be challenging for humans and machines. The miniF2F benchmark has 20 IMO problems in its testing set, yet formal proofs are available only for 7 of these problems (3 of which are written only by mathematicians). The model with best accuracy can only prove 4 of these 20 IMO problems, from 1950s and 60s, while its training set is a secret. In this work, we write complete, original formal proofs for the remaining 13 IMO problems in Lean along with 3 extra problems from IMO 2022 and 2023. This effort expands the availability of proof currently in the public domain by creating 5,150 lines of Lean proof. The goal of the paper is to pave the way for developing AI models that can automatically write the formal proofs for all the IMO problems in miniF2F and beyond. In this pursuit, we devise a method to decompose the proof of these problems into their building blocks, constructing a dataset of about 900 lemmas with 25,500 lines of Lean code. These lemmas are not trivial, yet they are approachable, providing the opportunity to evaluate and diagnose the failures and successes of AI models. We then evaluate the ability of GPT-4 in writing formal proofs for these lemmas with zero shot prompting, CoT reasoning and lemma retrieval. In evaluating the responses, we also analyze the confounding factor of LLM's ability to write the proofs in natural language vs Lean language.
     </details>

12. **MATATA: a weak-supervised MAthematical Tool-Assisted reasoning for Tabular Applications** [[pdf]](http://arxiv.org/abs/2411.18915) `2024-11-28` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces MATATA, a novel cost-effective method to train LLM agents for tabular data problems through reasoning, planning, and tool use that reaches state-of-the-art performances on FinQA and TAT-QA among reasoning frameworks based on open-source models.
     </details>


     <details>
          <summary>Abstract</summary>
          Mathematical reasoning capabilities are increasing with tool-augmented language agents, but methods often rely either on closed-source or large models, external data, or extensive prompt engineering. This work introduces MATATA, a novel cost-effective method to train LLM agents for tabular data problems through reasoning, planning, and tool use. With a progressive self-improvement paradigm and an iterative weak supervision, it empowers 3.8B/8B Small Language Models (SLMs), particularly suited for local hosting and sensitive business contexts where data privacy is crucial. By employing a flexible and reusable tools across different datasets, it achieves robust performance with effective scalability across shared tasks. Experiments show that MATATA reaches state-of-the-art performances on FinQA and TAT-QA among reasoning frameworks based on open-source models. Moreover, MATATA models compete with GPT-4 based frameworks on TabMWP, while being SLMs.
     </details>

13. **Mars-PO: Multi-Agent Reasoning System Preference Optimization** [[pdf]](http://arxiv.org/abs/2411.19039) `2024-11-28` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Mars-PO is proposed, a novel framework to improve the mathematical reasoning capabilities of LLMs through a multi-agent system that combines high-quality outputs from multiple agents into a hybrid positive sample set and pairs them with agent-specific negative samples to construct robust preference pairs for training.
     </details>


     <details>
          <summary>Abstract</summary>
          Mathematical reasoning is a fundamental capability for large language models (LLMs), yet achieving high performance in this domain remains a significant challenge. The auto-regressive generation process often makes LLMs susceptible to errors, hallucinations, and inconsistencies, particularly during multi-step reasoning. In this paper, we propose Mars-PO, a novel framework to improve the mathematical reasoning capabilities of LLMs through a multi-agent system. It combines high-quality outputs from multiple agents into a hybrid positive sample set and pairs them with agent-specific negative samples to construct robust preference pairs for training. By aligning agents with shared positive samples while addressing individual weaknesses, Mars-PO achieves substantial performance improvements on mathematical reasoning benchmarks. For example, it increases the accuracy on the MATH benchmark of the state-of-the-art instruction-tuned LLM, Llama3.1-8B-Instruct, from 50.38% to 57.82%. Experimental results further demonstrate that our method consistently outperforms other baselines, such as supervised fine-tuning, vanilla DPO, and its enhanced versions, highlighting the effectiveness of our approach.
     </details>

14. **Beyond Examples: High-level Automated Reasoning Paradigm in In-Context Learning via MCTS** [[pdf]](http://arxiv.org/abs/2411.18478) `2024-11-27` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          HiAR-ICL introduces five atomic reasoning actions as fundamental components for constructing chain-structured patterns, and develops a cognitive complexity framework that dynamically matches problems with appropriate thought cards.
     </details>


     <details>
          <summary>Abstract</summary>
          In-context Learning (ICL) enables large language models (LLMs) to tackle downstream tasks through sophisticated prompting and high-quality demonstrations. However, this traditional ICL paradigm shows limitations when facing complex mathematical reasoning tasks, primarily due to its heavy dependence on example quality and the necessity for human intervention in challenging scenarios. To address these limitations, this paper presents HiAR-ICL, a \textbf{Hi}gh-level \textbf{A}utomated \textbf{R}easoning paradigm in \textbf{ICL} that shifts focus from specific examples to abstract thinking patterns, extending the conventional concept of context in ICL. HiAR-ICL introduces five atomic reasoning actions as fundamental components for constructing chain-structured patterns. Using Monte Carlo Tree Search, we explore reasoning paths and construct thought cards to guide subsequent inference. We then develop a cognitive complexity framework that dynamically matches problems with appropriate thought cards. Experimental results demonstrate HiAR-ICL's effectiveness, achieving state-of-the-art accuracy (79.6$\%$) on the MATH benchmark with Qwen2.5-7B-Instruct, surpassing GPT-4o (76.6$\%$) and Claude 3.5 (71.1$\%$).
     </details>

15. **Training and Evaluating Language Models with Template-based Data Generation** [[pdf]](http://arxiv.org/abs/2411.18104) `2024-11-27` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces Template-based Data Generation (TDG), a novel approach that leverages LLMs (GPT-4) to automatically generate parameterized meta-templates, which are then used to synthesize a vast array of high-quality problems and solutions.
     </details>


     <details>
          <summary>Abstract</summary>
          The rapid advancement of large language models (LLMs) such as GPT-3, PaLM, and Llama has significantly transformed natural language processing, showcasing remarkable capabilities in understanding and generating language. However, these models often struggle with tasks requiring complex reasoning, particularly in mathematical problem-solving, due in part to the scarcity of large-scale, high-quality, domain-specific datasets necessary for training sophisticated reasoning abilities. To address this limitation, we introduce Template-based Data Generation (TDG), a novel approach that leverages LLMs (GPT-4) to automatically generate parameterized meta-templates, which are then used to synthesize a vast array of high-quality problems and solutions. Leveraging TDG, we create TemplateMath Part I: TemplateGSM, a dataset comprising over 7 million synthetically generated grade school math problems--each accompanied by code-based and natural language solutions--with the potential to generate an effectively unlimited number more. This dataset alleviates the scarcity of large-scale mathematical datasets and serves as a valuable resource for pre-training, fine-tuning, and evaluating LLMs in mathematical reasoning. Our method not only enables the generation of virtually infinite data but also elevates data augmentation to a new level by using GPT-4 for meta-template generation, ensuring diverse and high-quality problem structures. The TemplateMath Part I: TemplateGSM dataset is publicly available at https://huggingface.co/datasets/math-ai/TemplateGSM. The code is available at https://github.com/iiis-ai/TemplateMath.
     </details>

16. **Enhancing LLM Reasoning via Critique Models with Test-Time and Training-Time Supervision** [[pdf]](http://arxiv.org/abs/2411.16579) `2024-11-25` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper delves into a two-player paradigm that separates the roles of reasoning and critique models, where the critique model provides step-level feedback to supervise the reasoning (actor) model during both test-time and train-time, and proposes a critique-in-the-loop self-improvement method.
     </details>


     <details>
          <summary>Abstract</summary>
          Training large language models (LLMs) to spend more time thinking and reflection before responding is crucial for effectively solving complex reasoning tasks in fields such as science, coding, and mathematics. However, the effectiveness of mechanisms like self-reflection and self-correction depends on the model's capacity to accurately assess its own performance, which can be limited by factors such as initial accuracy, question difficulty, and the lack of external feedback. In this paper, we delve into a two-player paradigm that separates the roles of reasoning and critique models, where the critique model provides step-level feedback to supervise the reasoning (actor) model during both test-time and train-time. We first propose AutoMathCritique, an automated and scalable framework for collecting critique data, resulting in a dataset of $76,321$ responses paired with step-level feedback. Fine-tuning language models with this dataset enables them to generate natural language feedback for mathematical reasoning. We demonstrate that the critique models consistently improve the actor's performance on difficult queries at test-time, especially when scaling up inference-time computation. Motivated by these findings, we introduce the critique-based supervision to the actor's self-training process, and propose a critique-in-the-loop self-improvement method. Experiments show that the method improves the actor's exploration efficiency and solution diversity, especially on challenging queries, leading to a stronger reasoning model. Lastly, we take the preliminary step to explore training self-talk reasoning models via critique supervision and showcase its potential. Our code and datasets are at \href{https://mathcritique.github.io/}{https://mathcritique.github.io/}.
     </details>

17. **Learning by Analogy: Enhancing Few-Shot Prompting for Math Word Problem Solving with Computational Graph-Based Retrieval** [[pdf]](http://arxiv.org/abs/2411.16454) `2024-11-25` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper presents how analogy from similarly structured questions can improve LLMs' problem-solving capabilities for MWPs, and relies on the retrieval of problems with similar computational graphs to the given question to serve as exemplars in the prompt, providing the correct reasoning path for the generation model to refer to.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) are known to struggle with complicated reasoning tasks such as math word problems (MWPs). In this paper, we present how analogy from similarly structured questions can improve LLMs' problem-solving capabilities for MWPs. Specifically, we rely on the retrieval of problems with similar computational graphs to the given question to serve as exemplars in the prompt, providing the correct reasoning path for the generation model to refer to. Empirical results across six math word problem datasets demonstrate the effectiveness of our proposed method, which achieves a significant improvement of up to 6.7 percent on average in absolute value, compared to baseline methods. These results highlight our method's potential in addressing the reasoning challenges in current LLMs.
     </details>

18. **O1 Replication Journey -- Part 2: Surpassing O1-preview through Simple Distillation, Big Progress or Bitter Lesson?** [[pdf]](http://arxiv.org/abs/2411.16489) `2024-11-25` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is shown that a base model fine-tuned on simply tens of thousands of samples O1-distilled long-thought chains outperforms O1-preview on the AIME with minimal technical complexity.
     </details>


     <details>
          <summary>Abstract</summary>
          This paper presents a critical examination of current approaches to replicating OpenAI's O1 model capabilities, with particular focus on the widespread but often undisclosed use of knowledge distillation techniques. While our previous work explored the fundamental technical path to O1 replication, this study reveals how simple distillation from O1's API, combined with supervised fine-tuning, can achieve superior performance on complex mathematical reasoning tasks. Through extensive experiments, we show that a base model fine-tuned on simply tens of thousands of samples O1-distilled long-thought chains outperforms O1-preview on the American Invitational Mathematics Examination (AIME) with minimal technical complexity. Moreover, our investigation extends beyond mathematical reasoning to explore the generalization capabilities of O1-distilled models across diverse tasks: hallucination, safety and open-domain QA. Notably, despite training only on mathematical problem-solving data, our models demonstrated strong generalization to open-ended QA tasks and became significantly less susceptible to sycophancy after fine-tuning. We deliberately make this finding public to promote transparency in AI research and to challenge the current trend of obscured technical claims in the field. Our work includes: (1) A detailed technical exposition of the distillation process and its effectiveness, (2) A comprehensive benchmark framework for evaluating and categorizing O1 replication attempts based on their technical transparency and reproducibility, (3) A critical discussion of the limitations and potential risks of over-relying on distillation approaches, our analysis culminates in a crucial bitter lesson: while the pursuit of more capable AI systems is important, the development of researchers grounded in first-principles thinking is paramount.
     </details>

19. **Unraveling Arithmetic in Large Language Models: The Role of Algebraic Structures** [[pdf]](http://arxiv.org/abs/2411.16260) `2024-11-25` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is proposed that LLMs learn arithmetic by capturing algebraic structures, such as \emph{Commutativity} and \emph{Identity} properties, since these structures are observable through input-output relationships, they can generalize to unseen data.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have demonstrated remarkable mathematical capabilities, largely driven by chain-of-thought (CoT) prompting, which decomposes complex reasoning into step-by-step solutions. This approach has enabled significant advancements, as evidenced by performance on benchmarks like GSM8K and MATH. However, the mechanisms underlying LLMs' ability to perform arithmetic in a single step of CoT remain poorly understood. Existing studies debate whether LLMs encode numerical values or rely on symbolic reasoning, while others explore attention and multi-layered processing in arithmetic tasks. In this work, we propose that LLMs learn arithmetic by capturing algebraic structures, such as \emph{Commutativity} and \emph{Identity} properties. Since these structures are observable through input-output relationships, they can generalize to unseen data. We empirically demonstrate that LLMs can learn algebraic structures using a custom dataset of arithmetic problems. Our findings indicate that leveraging algebraic structures can enhance the LLMs' arithmetic capabilities, offering insights into improving their arithmetic performance.
     </details>

20. **MC-NEST -- Enhancing Mathematical Reasoning in Large Language Models with a Monte Carlo Nash Equilibrium Self-Refine Tree** [[pdf]](http://arxiv.org/abs/2411.15645) `2024-11-23` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The Monte Carlo Nash Equilibrium Self-Refine Tree (MC-NEST) algorithm, an enhancement of the Monte Carlo Tree Self-Refine (MCTSr) approach, is introduced, demonstrating its potential to significantly boost complex mathematical reasoning performance in LLMs.
     </details>


     <details>
          <summary>Abstract</summary>
          Mathematical reasoning has proven to be a critical yet challenging task for large language models (LLMs), as they often struggle with complex multi-step problems. To address these limitations, we introduce the Monte Carlo Nash Equilibrium Self-Refine Tree (MC-NEST) algorithm, an enhancement of the Monte Carlo Tree Self-Refine (MCTSr) approach. By integrating Nash Equilibrium strategies with LLM-based self-refinement and self-evaluation processes, MC-NEST aims to improve decision-making for complex mathematical reasoning tasks. This method ensures balanced exploration and exploitation of potential solutions, leveraging Upper Confidence Bound (UCT) scores and various selection policies. Through iterative critique and refinement, MC-NEST enhances the reasoning capabilities of LLMs, particularly for problems requiring strategic decision-making. Comparative analysis reveals that GPT-4o, equipped with MC-NEST using an Importance Sampling Policy, achieved superior accuracy in domains such as Number Theory and Geometry. These results suggest that both LLMs GPT-4o and Phi-3-mini can benefit from MC-NEST, with iterative self-refinement proving especially effective in expanding the reasoning capacity and problem-solving performance of LLMs. We evaluate the effectiveness of MC-NEST on challenging Olympiad-level benchmarks, demonstrating its potential to significantly boost complex mathematical reasoning performance in LLMs.
     </details>

21. **Improving Mathematical Reasoning Capabilities of Small Language Models via Feedback-Driven Distillation** [[pdf]](http://arxiv.org/abs/2411.14698) `2024-11-22` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A Feedback-Driven Distillation (FDD) framework to enhance SLMs' mathematical reasoning capabilities and a multi-round distillation paradigm to iteratively enrich the distillation datasets, thereby progressively improving the mathematical reasoning abilities of SLMs.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) demonstrate exceptional reasoning capabilities, often achieving state-of-the-art performance in various tasks. However, their substantial computational and memory demands, due to billions of parameters, hinder deployment in resource-constrained environments. A promising solution is knowledge distillation, where LLMs transfer reasoning capabilities to Small Language Models (SLMs, $\le$ 1B parameters), enabling wider deployment on low-resource devices. Existing methods primarily focus on generating high-quality reasoning rationales for distillation datasets but often neglect the critical role of data quantity and quality. To address these challenges, we propose a Feedback-Driven Distillation (FDD) framework to enhance SLMs' mathematical reasoning capabilities. In the initialization stage, a distillation dataset is constructed by prompting LLMs to pair mathematical problems with corresponding reasoning rationales. We classify problems into easy and hard categories based on SLM performance. For easy problems, LLMs generate more complex variations, while for hard problems, new questions of similar complexity are synthesized. In addition, we propose a multi-round distillation paradigm to iteratively enrich the distillation datasets, thereby progressively improving the mathematical reasoning abilities of SLMs. Experimental results demonstrate that our method can make SLMs achieve SOTA mathematical reasoning performance.
     </details>

22. **Marco-o1: Towards Open Reasoning Models for Open-Ended Solutions** [[pdf]](http://arxiv.org/abs/2411.14405) `2024-11-21` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Marco-o1 aims to address the question: ''Can the o1 model effectively generalize to broader domains where clear standards are absent and rewards are challenging to quantify?'''
     </details>


     <details>
          <summary>Abstract</summary>
          Currently OpenAI o1 sparks a surge of interest in the study of large reasoning models (LRM). Building on this momentum, Marco-o1 not only focuses on disciplines with standard answers, such as mathematics, physics, and coding -- which are well-suited for reinforcement learning (RL) -- but also places greater emphasis on open-ended resolutions. We aim to address the question: ''Can the o1 model effectively generalize to broader domains where clear standards are absent and rewards are challenging to quantify?'' Marco-o1 is powered by Chain-of-Thought (CoT) fine-tuning, Monte Carlo Tree Search (MCTS), reflection mechanisms, and innovative reasoning strategies -- optimized for complex real-world problem-solving tasks.
     </details>

23. **Patience Is The Key to Large Language Model Reasoning** [[pdf]](http://arxiv.org/abs/2411.13082) `2024-11-20` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a simple method by encouraging models to adopt a more patient reasoning style without the need of introducing new knowledge or skills, and demonstrates a performance increase of up to 6.7% on GSM8k with training just on a lightweight dataset.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent advancements in the field of large language models, particularly through the Chain of Thought (CoT) approach, have demonstrated significant improvements in solving complex problems. However, existing models either tend to sacrifice detailed reasoning for brevity due to user preferences, or require extensive and expensive training data to learn complicated reasoning ability, limiting their potential in solving complex tasks. To bridge this gap, following the concept of scaling test-time, we propose a simple method by encouraging models to adopt a more patient reasoning style without the need of introducing new knowledge or skills. To employ a preference optimization approach, we generate detailed reasoning processes as positive examples and simple answers as negative examples, thereby training the model to favor thoroughness in its responses. Our results demonstrate a performance increase of up to 6.7% on GSM8k with training just on a lightweight dataset.
     </details>

24. **Enhancing Reasoning Capabilities of LLMs via Principled Synthetic Logic Corpus** [[pdf]](http://arxiv.org/abs/2411.12498) `2024-11-19` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work establishes principles for designing high-quality samples for designing high-quality samples by integrating symbolic logic theory and previous empirical insights, and constructs a synthetic corpus, comprising numerous samples of multi-step deduction with unknown facts, diverse reasoning rules, diverse linguistic expressions, and challenging distractors.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) are capable of solving a wide range of tasks, yet they have struggled with reasoning. To address this, we propose $\textbf{Additional Logic Training (ALT)}$, which aims to enhance LLMs' reasoning capabilities by program-generated logical reasoning samples. We first establish principles for designing high-quality samples by integrating symbolic logic theory and previous empirical insights. Then, based on these principles, we construct a synthetic corpus named $\textbf{Formal Logic Deduction Diverse}$ ($\textbf{FLD}$$^{\times 2}$), comprising numerous samples of multi-step deduction with unknown facts, diverse reasoning rules, diverse linguistic expressions, and challenging distractors. Finally, we empirically show that ALT on FLD$^{\times2}$ substantially enhances the reasoning capabilities of state-of-the-art LLMs, including LLaMA-3.1-70B. Improvements include gains of up to 30 points on logical reasoning benchmarks, up to 10 points on math and coding benchmarks, and 5 points on the benchmark suite BBH.
     </details>

25. **AtomThink: A Slow Thinking Framework for Multimodal Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2411.11930) `2024-11-18` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The proposed AtomThink significantly improves the performance of baseline MLLMs, achieving approximately 50\% relative accuracy gains on MathVista and 120\% on MathVerse, and will make the code and dataset publicly available on https://github.com/Quinn777/AtomThink.
     </details>


     <details>
          <summary>Abstract</summary>
          In this paper, we address the challenging task of multimodal mathematical reasoning by incorporating the ability of ``slow thinking" into multimodal large language models (MLLMs). Contrary to existing methods that rely on direct or fast thinking, our key idea is to construct long chains of thought (CoT) consisting of atomic actions in a step-by-step manner, guiding MLLMs to perform complex reasoning. To this end, we design a novel AtomThink framework composed of three key modules: (i) a CoT annotation engine that automatically generates high-quality CoT annotations to address the lack of high-quality visual mathematical data; (ii) an atomic step fine-tuning strategy that jointly optimizes an MLLM and a policy reward model (PRM) for step-wise reasoning; and (iii) four different search strategies that can be applied with the PRM to complete reasoning. Additionally, we propose AtomMATH, a large-scale multimodal dataset of long CoTs, and an atomic capability evaluation metric for mathematical tasks. Extensive experimental results show that the proposed AtomThink significantly improves the performance of baseline MLLMs, achieving approximately 50\% relative accuracy gains on MathVista and 120\% on MathVerse. To support the advancement of multimodal slow-thinking models, we will make our code and dataset publicly available on https://github.com/Quinn777/AtomThink.
     </details>

26. **Newclid: A User-Friendly Replacement for AlphaGeometry** [[pdf]](http://arxiv.org/abs/2411.11938) `2024-11-18` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Newclid contains a symbolic solver called DDARN (derived from DDAR-Newclid), which is a significant refactoring and upgrade of AlphaGeometry's DDAR symbolic solver by being more user-friendly - both for the end user as well as for a programmer wishing to extend the codebase.
     </details>


     <details>
          <summary>Abstract</summary>
          We introduce a new symbolic solver for geometry, called Newclid, which is based on AlphaGeometry. Newclid contains a symbolic solver called DDARN (derived from DDAR-Newclid), which is a significant refactoring and upgrade of AlphaGeometry's DDAR symbolic solver by being more user-friendly - both for the end user as well as for a programmer wishing to extend the codebase. For the programmer, improvements include a modularized codebase and new debugging and visualization tools. For the user, Newclid contains a new command line interface (CLI) that provides interfaces for agents to guide DDARN. DDARN is flexible with respect to its internal reasoning, which can be steered by agents. Further, we support input from GeoGebra to make Newclid accessible for educational contexts. Further, the scope of problems that Newclid can solve has been expanded to include the ability to have an improved understanding of metric geometry concepts (length, angle) and to use theorems such as the Pythagorean theorem in proofs. Bugs have been fixed, and reproducibility has been improved. Lastly, we re-evaluated the five remaining problems from the original AG-30 dataset that AlphaGeometry was not able to solve and contrasted them with the abilities of DDARN, running in breadth-first-search agentic mode (which corresponds to how DDARN runs by default), finding that DDARN solves an additional problem. We have open-sourced our code under: https://github.com/LMCRC/Newclid
     </details>

27. **PSPO*: An Effective Process-supervised Policy Optimization for Reasoning Alignment** [[pdf]](http://arxiv.org/abs/2411.11681) `2024-11-18` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The PSPO-WRS is developed, which considers the number of reasoning steps in determining reward scores and utilizes an adjusted Weibull distribution for nonlinear reward shaping and consistently outperforms current mainstream models.
     </details>


     <details>
          <summary>Abstract</summary>
          Process supervision enhances the performance of large language models in reasoning tasks by providing feedback at each step of chain-of-thought reasoning. However, due to the lack of effective process supervision methods, even advanced large language models are prone to logical errors and redundant reasoning. We claim that the effectiveness of process supervision significantly depends on both the accuracy and the length of reasoning chains. Moreover, we identify that these factors exhibit a nonlinear relationship with the overall reward score of the reasoning process. Inspired by these insights, we propose a novel process supervision paradigm, PSPO*, which systematically outlines the workflow from reward model training to policy optimization, and highlights the importance of nonlinear rewards in process supervision. Based on PSPO*, we develop the PSPO-WRS, which considers the number of reasoning steps in determining reward scores and utilizes an adjusted Weibull distribution for nonlinear reward shaping. Experimental results on six mathematical reasoning datasets demonstrate that PSPO-WRS consistently outperforms current mainstream models.
     </details>

28. **Rethinking Thinking Tokens: Understanding Why They Underperform in Practice** [[pdf]](http://arxiv.org/abs/2411.11371) `2024-11-18` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A comprehensive empirical analysis is provided to validate the reliance on a single embedding for TTs, which results in inconsistent learning signals and introduces noisy gradients and the implications for future research on unsupervised reasoning in LLMs are discussed.
     </details>


     <details>
          <summary>Abstract</summary>
          Thinking Tokens (TT) have been proposed as an unsupervised method to facilitate reasoning in language models. However, despite their conceptual appeal, our findings show that TTs marginally improves performance and consistently underperforms compared to Chain-of-Thought (CoT) reasoning across multiple benchmarks. We hypothesize that this underperformance stems from the reliance on a single embedding for TTs, which results in inconsistent learning signals and introduces noisy gradients. This paper provides a comprehensive empirical analysis to validate this hypothesis and discusses the implications for future research on unsupervised reasoning in LLMs.
     </details>

29. **Technical Report: Enhancing LLM Reasoning with Reward-guided Tree Search** [[pdf]](http://arxiv.org/abs/2411.11694) `2024-11-18` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Recently, test-time scaling has garnered significant attention from the research community, largely due to the substantial advancements of the o1 model released by OpenAI. By allocating more computational resources during the inference phase, large language models~(LLMs) can extensively explore the solution space by generating more thought tokens or diverse solutions, thereby producing more accurate responses. However, developing an o1-like reasoning approach is challenging, and researchers have been making various attempts to advance this open area of research. In this paper, we present a preliminary exploration into enhancing the reasoning abilities of LLMs through reward-guided tree search algorithms. This framework is implemented by integrating the policy model, reward model, and search algorithm. It is primarily constructed around a tree search algorithm, where the policy model navigates a dynamically expanding tree guided by a specially trained reward model. We thoroughly explore various design considerations necessary for implementing this framework and provide a detailed report of the technical aspects. To assess the effectiveness of our approach, we focus on mathematical reasoning tasks and conduct extensive evaluations on four challenging datasets, significantly enhancing the reasoning abilities of LLMs.
     </details>

30. **Understanding Chain-of-Thought in LLMs through Information Theory** [[pdf]](http://arxiv.org/abs/2411.11984) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper formalizes CoT reasoning in LLMs through an information-theoretic lens, and quantifies the `information gain' at each reasoning step, enabling the identification of failure modes in LLMs without the need for expensive annotated datasets.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have shown impressive performance in complex reasoning tasks through Chain-of-Thought (CoT) reasoning, allowing models to break down problems into manageable sub-tasks. However, existing CoT evaluation techniques either require annotated CoT data or fall short in accurately assessing intermediate reasoning steps, leading to high rates of false positives. In this paper, we formalize CoT reasoning in LLMs through an information-theoretic lens. Specifically, our framework quantifies the `information gain' at each reasoning step, enabling the identification of failure modes in LLMs without the need for expensive annotated datasets. We demonstrate the efficacy of our approach through extensive experiments on toy and GSM-8K data, where it significantly outperforms existing outcome-based methods by providing more accurate insights into model performance on individual tasks.
     </details>

31. **Enhancing the Reasoning Ability of Multimodal Large Language Models via Mixed Preference Optimization** [[pdf]](http://arxiv.org/abs/2411.10442) `2024-11-15` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work designs an automated preference data construction pipeline to create MMPR, a high-quality, large-scale multimodal reasoning preference dataset, and explores integrating PO with MLLMs, developing a simple yet effective method, termed Mixed Preference Optimization (MPO), which boosts multimodal CoT performance.
     </details>


     <details>
          <summary>Abstract</summary>
          Existing open-source multimodal large language models (MLLMs) generally follow a training process involving pre-training and supervised fine-tuning. However, these models suffer from distribution shifts, which limit their multimodal reasoning, particularly in the Chain-of-Thought (CoT) performance. To address this, we introduce a preference optimization (PO) process to enhance the multimodal reasoning capabilities of MLLMs. Specifically, (1) on the data side, we design an automated preference data construction pipeline to create MMPR, a high-quality, large-scale multimodal reasoning preference dataset. and (2) on the model side, we explore integrating PO with MLLMs, developing a simple yet effective method, termed Mixed Preference Optimization (MPO), which boosts multimodal CoT performance. Our approach demonstrates improved performance across multiple benchmarks, particularly in multimodal reasoning tasks. Notably, our model, InternVL2-8B-MPO, achieves an accuracy of 67.0 on MathVista, outperforming InternVL2-8B by 8.7 points and achieving performance comparable to the 10x larger InternVL2-76B. We hope this study could inspire further advancements in MLLMs. Code, data, and model shall be publicly released.
     </details>

32. **UTMath: Math Evaluation with Unit Test via Reasoning-to-Coding Thoughts** [[pdf]](http://arxiv.org/abs/2411.07240) `2024-11-11` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The UTMath Benchmark is introduced, which robustly evaluates the models through extensive unit tests, and the Reasoning-to-Coding of Thoughts (RCoT) approach is introduced, which encourages LLMs to perform explicit reasoning before generating code, leading to generating more advanced solution and improved performance.
     </details>


     <details>
          <summary>Abstract</summary>
          The evaluation of mathematical reasoning capabilities is essential for advancing Artificial General Intelligence (AGI). While Large Language Models (LLMs) have shown impressive performance in solving mathematical problems, existing benchmarks such as GSM8K and MATH present limitations, including narrow problem definitions with specific numbers and reliance on predetermined rules that hinder accurate assessments of reasoning and adaptability. This paper introduces the UTMath Benchmark, which robustly evaluates the models through extensive unit tests. It consists of 1,053 problems across 9 mathematical domains, with over 68 test cases per problem. We propose an innovative evaluation framework inspired by unit testing in software development, focusing on both accuracy and reliability of results. Furthermore, we introduce the Reasoning-to-Coding of Thoughts (RCoT) approach, which encourages LLMs to perform explicit reasoning before generating code, leading to generating more advanced solution and improved performance. Furthermore, we are releasing not only the UTMath benchmark but also the UTMath-Train training dataset (more than 70k samples), to support the community in further exploring mathematical reasoning.
     </details>

33. **OpenAI-o1 AB Testing: Does the o1 model really do good reasoning in math problem solving?** [[pdf]](http://arxiv.org/abs/2411.06198) `2024-11-09` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          The Orion-1 model by OpenAI is claimed to have more robust logical reasoning capabilities than previous large language models. However, some suggest the excellence might be partially due to the model "memorizing" solutions, resulting in less satisfactory performance when prompted with problems not in the training data. We conduct a comparison experiment using two datasets: one consisting of International Mathematics Olympiad (IMO) problems, which is easily accessible; the other one consisting of Chinese National Team Training camp (CNT) problems, which have similar difficulty but not as publically accessible. We label the response for each problem and compare the performance between the two datasets. We conclude that there is no significant evidence to show that the model relies on memorizing problems and solutions. Also, we perform case studies to analyze some features of the model's response.
     </details>

34. **Gap-Filling Prompting Enhances Code-Assisted Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2411.05407) `2024-11-08` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          GFP is introduced, a novel two-step prompting strategy designed to enhance the problem-solving process for SLMs that identifies gaps and provides hints for filling them, and adds the hints to the question to generate a final code solution.
     </details>


     <details>
          <summary>Abstract</summary>
          Despite the strong performance of large language models (LLMs) in tasks like mathematical reasoning, their practical use is limited by high computational demands and proprietary restrictions. Chain-of-thought (CoT) and program-of-thought (PoT) fine-tuning are common methods to transfer LLM knowledge to small language models (SLMs). However, CoT often leads to calculation errors in SLMs, while PoT has shown more promise. While most PoT-based approaches focus on direct problem-to-code conversion or extracting only the key information from questions and then providing code solution for it, this work emphasizes filling the gaps in the question to clearly illustrate the solution path, which can be challenging for an SLM to understand when such information is not explicitly provided. Therefore, this paper introduces Gap-Filling Prompting (GFP), a novel two-step prompting strategy designed to enhance the problem-solving process for SLMs. The first step identifies these gaps and provides hints for filling them, while the second step adds the hints to the question to generate a final code solution. Experimental results on two benchmark datasets demonstrate that GFP significantly improves the mathematical reasoning abilities of SLMs.
     </details>

35. **Mathematical Formalized Problem Solving and Theorem Proving in Different Fields in Lean 4** [[pdf]](http://arxiv.org/abs/2409.05977) `2024-11-08` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The use of Large Language Models to generate formal proof steps and complete formalized proofs and the basic structure and tactics of the Lean 4 language are introduced to determine how AI can be leveraged to assist the mathematical formalization process and improve its performance.
     </details>


     <details>
          <summary>Abstract</summary>
          Formalizing mathematical proofs using computerized verification languages like Lean 4 has the potential to significantly impact the field of mathematics, it offers prominent capabilities for advancing mathematical reasoning. However, existing efforts are largely limited to creating formalized versions of proofs from extensive online mathematical corpora, struggling to keep pace with the rapidly evolving nature of mathematics. To bridge the gap between traditional and computerized proof techniques, this paper explores the use of Large Language Models (LLMs) to generate formal proof steps and complete formalized proofs. By converting natural language (NL) mathematical proofs into formalized versions, this work introduces the basic structure and tactics of the Lean 4 language. The goal is to determine how AI can be leveraged to assist the mathematical formalization process and improve its performance. Several examples are provided that demonstrate solving problems using both traditional and Lean 4-based approaches. Ultimately, this paper presents an explanation of the foundations of Lean 4 and comparative analyses of the mathematical formalization process using traditional and AI-augmented techniques. The findings indicate that AI- powered tools have significant potential to accelerate and enhance the formalization of mathematical proofs, paving the way for more efficient and reliable theorem-proving for AI for Math in the future.
     </details>

36. **Quantifying artificial intelligence through algebraic generalization** [[pdf]](http://arxiv.org/abs/2411.05943) `2024-11-08` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          The rapid development of modern artificial intelligence (AI) systems has created an urgent need for their scientific quantification. While their fluency across a variety of domains is impressive, modern AI systems fall short on tests requiring symbolic processing and abstraction - a glaring limitation given the necessity for interpretable and reliable technology. Despite a surge of reasoning benchmarks emerging from the academic community, no comprehensive and theoretically-motivated framework exists to quantify reasoning (and more generally, symbolic ability) in AI systems. Here, we adopt a framework from computational complexity theory to explicitly quantify symbolic generalization: algebraic circuit complexity. Many symbolic reasoning problems can be recast as algebraic expressions. Thus, algebraic circuit complexity theory - the study of algebraic expressions as circuit models (i.e., directed acyclic graphs) - is a natural framework to study the complexity of symbolic computation. The tools of algebraic circuit complexity enable the study of generalization by defining benchmarks in terms of their complexity-theoretic properties (i.e., the difficulty of a problem). Moreover, algebraic circuits are generic mathematical objects; for a given algebraic circuit, an arbitrarily large number of samples can be generated for a specific circuit, making it an optimal testbed for the data-hungry machine learning algorithms that are used today. Here, we adopt tools from algebraic circuit complexity theory, apply it to formalize a science of symbolic generalization, and address key theoretical and empirical challenges for its successful application to AI science and its impact on the broader community.
     </details>

37. **Qwen2.5-32B: Leveraging Self-Consistent Tool-Integrated Reasoning for Bengali Mathematical Olympiad Problem Solving** [[pdf]](http://arxiv.org/abs/2411.05934) `2024-11-08` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work presents an innovative approach for solving mathematical problems in Bengali, developed for the DL Sprint 3.0 BUET CSE Fest 2024 Competition, which highlights the potential of advanced NLP techniques in solving Bengali mathematical problems.
     </details>


     <details>
          <summary>Abstract</summary>
          We present an innovative approach for solving mathematical problems in Bengali, developed for the DL Sprint 3.0 BUET CSE Fest 2024 Competition. Our method uses advanced deep learning models, notably the Qwen 2.5 series, with improvements made through prompt engineering, model quantization, and Tool Integrated Reasoning (TIR) to handle complex calculations. Initially, we explored various model architectures, including fine-tuned Mistral and quantized Qwen models, refining them with translation techniques, Retrieval-Augmented Generation (RAG), and custom dataset curation. Manual hyperparameter tuning optimized parameters like temperature and top-p to enhance model adaptability and accuracy. Removal of RAG and parameter adjustments further improved robustness. Our approach highlights the potential of advanced NLP techniques in solving Bengali mathematical problems.
     </details>

38. **VISTA: Visual Integrated System for Tailored Automation in Math Problem Generation Using LLM** [[pdf]](http://arxiv.org/abs/2411.05423) `2024-11-08` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel multi-agent framework that leverages Large Language Models (LLMs) to automate the creation of complex mathematical visualizations alongside coherent problem text, demonstrating the immense potential of LLMs in transforming the way educators generate and utilize visual aids in math education.
     </details>


     <details>
          <summary>Abstract</summary>
          Generating accurate and consistent visual aids is a critical challenge in mathematics education, where visual representations like geometric shapes and functions play a pivotal role in enhancing student comprehension. This paper introduces a novel multi-agent framework that leverages Large Language Models (LLMs) to automate the creation of complex mathematical visualizations alongside coherent problem text. Our approach not only simplifies the generation of precise visual aids but also aligns these aids with the problem's core mathematical concepts, improving both problem creation and assessment. By integrating multiple agents, each responsible for distinct tasks such as numeric calculation, geometry validation, and visualization, our system delivers mathematically accurate and contextually relevant problems with visual aids. Evaluation across Geometry and Function problem types shows that our method significantly outperforms basic LLMs in terms of text coherence, consistency, relevance and similarity, while maintaining the essential geometrical and functional integrity of the original problems. Although some challenges remain in ensuring consistent visual outputs, our framework demonstrates the immense potential of LLMs in transforming the way educators generate and utilize visual aids in math education.
     </details>

39. **Benchmarking Large Language Models with Integer Sequence Generation Tasks** [[pdf]](http://arxiv.org/abs/2411.04372) `2024-11-07` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduces an automated cheating detection mechanism that flags the use of lookup tables and validated this automation against human cheating evaluations, revealing that the o1 series of models outperform other frontier models from OpenAI, Anthropic, Meta, and Google in accuracy and cheating rates across both easy and hard integer sequences.
     </details>


     <details>
          <summary>Abstract</summary>
          This paper presents a novel benchmark where the large language model (LLM) must write code that computes integer sequences from the Online Encyclopedia of Integer Sequences (OEIS), a widely-used resource for mathematical sequences. The benchmark is designed to evaluate both the correctness of the generated code and its computational efficiency. Our benchmark reveals that the o1 series of models outperform other frontier models from OpenAI, Anthropic, Meta, and Google in accuracy and cheating rates across both easy and hard integer sequences. In order to ensure models do not exploit memorized sequence values, we introduce an automated cheating detection mechanism that flags the use of lookup tables and validated this automation against human cheating evaluations. This benchmark provides a meaningful challenge for current LLMs, offering insights into their mathematical reasoning and code writing capabilities, which can guide future research directions and model development in mathematical reasoning and code synthesis.
     </details>

40. **FrontierMath: A Benchmark for Evaluating Advanced Mathematical Reasoning in AI** [[pdf]](http://arxiv.org/abs/2411.04872) `2024-11-07` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          We introduce FrontierMath, a benchmark of hundreds of original, exceptionally challenging mathematics problems crafted and vetted by expert mathematicians. The questions cover most major branches of modern mathematics -- from computationally intensive problems in number theory and real analysis to abstract questions in algebraic geometry and category theory. Solving a typical problem requires multiple hours of effort from a researcher in the relevant branch of mathematics, and for the upper end questions, multiple days. FrontierMath uses new, unpublished problems and automated verification to reliably evaluate models while minimizing risk of data contamination. Current state-of-the-art AI models solve under 2% of problems, revealing a vast gap between AI capabilities and the prowess of the mathematical community. As AI systems advance toward expert-level mathematical abilities, FrontierMath offers a rigorous testbed that quantifies their progress.
     </details>

41. **Kwai-STaR: Transform LLMs into State-Transition Reasoners** [[pdf]](http://arxiv.org/abs/2411.04799) `2024-11-07` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work defines mathematical problem-solving as a process of transiting from an initial unsolved state to the final resolved state, and proposes Kwai-STaR framework, which transforms LLMs into State-Transition Reasoners to improve their intuitive reasoning capabilities.
     </details>


     <details>
          <summary>Abstract</summary>
          Mathematical reasoning presents a significant challenge to the cognitive capabilities of LLMs. Various methods have been proposed to enhance the mathematical ability of LLMs. However, few recognize the value of state transition for LLM reasoning. In this work, we define mathematical problem-solving as a process of transiting from an initial unsolved state to the final resolved state, and propose Kwai-STaR framework, which transforms LLMs into State-Transition Reasoners to improve their intuitive reasoning capabilities. Our approach comprises three main steps: (1) Define the state space tailored to the mathematical reasoning. (2) Generate state-transition data based on the state space. (3) Convert original LLMs into State-Transition Reasoners via a curricular training strategy. Our experiments validate the effectiveness of Kwai-STaR in enhancing mathematical reasoning: After training on the small-scale Kwai-STaR dataset, general LLMs, including Mistral-7B and LLaMA-3, achieve considerable performance gain on the GSM8K and GSM-Hard dataset. Additionally, the state transition-based design endows Kwai-STaR with remarkable training and inference efficiency. Further experiments are underway to establish the generality of Kwai-STaR.
     </details>

42. **EXPLORA: Efficient Exemplar Subset Selection for Complex Reasoning** [[pdf]](http://arxiv.org/abs/2411.03877) `2024-11-06` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          ExPLORA is introduced, a novel exploration method designed to estimate the parameters of the scoring function, which evaluates exemplar subsets without incorporating confidence information, and significantly reduces the number of LLM calls to ~11% of those required by state-of-the-art methods and achieves a substantial performance improvement of 12.24%.
     </details>


     <details>
          <summary>Abstract</summary>
          Answering reasoning-based complex questions over text and hybrid sources, including tables, is a challenging task. Recent advances in large language models (LLMs) have enabled in-context learning (ICL), allowing LLMs to acquire proficiency in a specific task using only a few demonstration samples (exemplars). A critical challenge in ICL is the selection of optimal exemplars, which can be either task-specific (static) or test-example-specific (dynamic). Static exemplars provide faster inference times and increased robustness across a distribution of test examples. In this paper, we propose an algorithm for static exemplar subset selection for complex reasoning tasks. We introduce EXPLORA, a novel exploration method designed to estimate the parameters of the scoring function, which evaluates exemplar subsets without incorporating confidence information. EXPLORA significantly reduces the number of LLM calls to ~11% of those required by state-of-the-art methods and achieves a substantial performance improvement of 12.24%. We open-source our code and data (https://github.com/kiranpurohit/EXPLORA).
     </details>

43. **Language Models are Hidden Reasoners: Unlocking Latent Reasoning Capabilities via Self-Rewarding** [[pdf]](http://arxiv.org/abs/2411.04282) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have shown impressive capabilities, but still struggle with complex reasoning tasks requiring multiple steps. While prompt-based methods like Chain-of-Thought (CoT) can improve LLM reasoning at inference time, optimizing reasoning capabilities during training remains challenging. We introduce LaTent Reasoning Optimization (LaTRO), a principled framework that formulates reasoning as sampling from a latent distribution and optimizes it via variational approaches. LaTRO enables LLMs to concurrently improve both their reasoning process and ability to evaluate reasoning quality, without requiring external feedback or reward models. We validate LaTRO through experiments on GSM8K and ARC-Challenge datasets using multiple model architectures. On GSM8K, LaTRO improves zero-shot accuracy by an average of 12.5% over base models and 9.6% over supervised fine-tuning across Phi-3.5-mini, Mistral-7B, and Llama-3.1-8B. Our findings suggest that pre-trained LLMs possess latent reasoning capabilities that can be unlocked and enhanced through our proposed optimization approach in a self-improvement manner. The code of LaTRO is available at \url{https://github.com/SalesforceAIResearch/LaTRO}.
     </details>

44. **Self-Consistency Preference Optimization** [[pdf]](http://arxiv.org/abs/2411.04109) `2024-11-06` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces self-consistency preference optimization (ScPO), which iteratively trains consistent answers to be preferred over inconsistent ones on unsupervised new problems, and shows ScPO leads to large improvements over conventional reward model training on reasoning tasks such as GSM8K and MATH.
     </details>


     <details>
          <summary>Abstract</summary>
          Self-alignment, whereby models learn to improve themselves without human annotation, is a rapidly growing research area. However, existing techniques often fail to improve complex reasoning tasks due to the difficulty of assigning correct rewards. An orthogonal approach that is known to improve correctness is self-consistency, a method applied at inference time based on multiple sampling in order to find the most consistent answer. In this work, we extend the self-consistency concept to help train models. We thus introduce self-consistency preference optimization (ScPO), which iteratively trains consistent answers to be preferred over inconsistent ones on unsupervised new problems. We show ScPO leads to large improvements over conventional reward model training on reasoning tasks such as GSM8K and MATH, closing the gap with supervised training with gold answers or preferences, and that combining ScPO with standard supervised learning improves results even further. On ZebraLogic, ScPO finetunes Llama-3 8B to be superior to Llama-3 70B, Gemma-2 27B, and Claude-3 Haiku.
     </details>

45. **Towards Interpreting Language Models: A Case Study in Multi-Hop Reasoning** [[pdf]](http://arxiv.org/abs/2411.05037) `2024-11-06` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Attention Lens is an open source tool that translates the outputs of attention heads into vocabulary tokens via learned transformations called lenses and demonstrates the use of lenses to reveal how a model arrives at its answer and uses them to localize sources of model failures such as in the case of biased and malicious language generation.
     </details>


     <details>
          <summary>Abstract</summary>
          Answering multi-hop reasoning questions requires retrieving and synthesizing information from diverse sources. Language models (LMs) struggle to perform such reasoning consistently. We propose an approach to pinpoint and rectify multi-hop reasoning failures through targeted memory injections on LM attention heads. First, we analyze the per-layer activations of GPT-2 models in response to single- and multi-hop prompts. We then propose a mechanism that allows users to inject relevant prompt-specific information, which we refer to as "memories," at critical LM locations during inference. By thus enabling the LM to incorporate additional relevant information during inference, we enhance the quality of multi-hop prompt completions. We empirically show that a simple, efficient, and targeted memory injection into a key attention layer often increases the probability of the desired next token in multi-hop tasks, by up to 424%. We observe that small subsets of attention heads can significantly impact the model prediction during multi-hop reasoning. To more faithfully interpret these heads, we develop Attention Lens: an open source tool that translates the outputs of attention heads into vocabulary tokens via learned transformations called lenses. We demonstrate the use of lenses to reveal how a model arrives at its answer and use them to localize sources of model failures such as in the case of biased and malicious language generation.
     </details>

46. **Automatic Generation of Question Hints for Mathematics Problems using Large Language Models in Educational Technology** [[pdf]](http://arxiv.org/abs/2411.03495) `2024-11-05` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Using LLMs (GPT-4o and Llama-3-8B-instruct) as teachers to generate effective hints for students simulated through LLMs (GPT-3.5-turbo, Llama-3-8B-Instruct, or Mistral-7B-instruct-v0.3) tackling math exercises designed for human high-school students, and designed using cognitive science principles.
     </details>


     <details>
          <summary>Abstract</summary>
          The automatic generation of hints by Large Language Models (LLMs) within Intelligent Tutoring Systems (ITSs) has shown potential to enhance student learning. However, generating pedagogically sound hints that address student misconceptions and adhere to specific educational objectives remains challenging. This work explores using LLMs (GPT-4o and Llama-3-8B-instruct) as teachers to generate effective hints for students simulated through LLMs (GPT-3.5-turbo, Llama-3-8B-Instruct, or Mistral-7B-instruct-v0.3) tackling math exercises designed for human high-school students, and designed using cognitive science principles. We present here the study of several dimensions: 1) identifying error patterns made by simulated students on secondary-level math exercises; 2) developing various prompts for GPT-4o as a teacher and evaluating their effectiveness in generating hints that enable simulated students to self-correct; and 3) testing the best-performing prompts, based on their ability to produce relevant hints and facilitate error correction, with Llama-3-8B-Instruct as the teacher, allowing for a performance comparison with GPT-4o. The results show that model errors increase with higher temperature settings. Notably, when hints are generated by GPT-4o, the most effective prompts include prompts tailored to specific errors as well as prompts providing general hints based on common mathematical errors. Interestingly, Llama-3-8B-Instruct as a teacher showed better overall performance than GPT-4o. Also the problem-solving and response revision capabilities of the LLMs as students, particularly GPT-3.5-turbo, improved significantly after receiving hints, especially at lower temperature settings. However, models like Mistral-7B-Instruct demonstrated a decline in performance as the temperature increased.
     </details>

47. **Formal Theorem Proving by Rewarding LLMs to Decompose Proofs Hierarchically** [[pdf]](http://arxiv.org/abs/2411.01829) `NeurIPS 2024 Workshop MATH-AI` `Isabelle` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Mathematical theorem proving is an important testbed for large language models' deep and abstract reasoning capability. This paper focuses on improving LLMs' ability to write proofs in formal languages that permit automated proof verification/evaluation. Most previous results provide human-written lemmas to the theorem prover, which is an arguably oversimplified setting that does not sufficiently test the provers' planning and decomposition capabilities. Instead, we work in a more natural setup where the lemmas that are directly relevant to the theorem are not given to the theorem prover at test time. We design an RL-based training algorithm that encourages the model to decompose a theorem into lemmas, prove the lemmas, and then prove the theorem by using the lemmas. Our reward mechanism is inspired by how mathematicians train themselves: even if a theorem is too challenging to be proved by the current model, a positive reward is still given to the model for any correct and novel lemmas that are proposed and proved in this process. During training, our model proposes and proves lemmas that are not in the training dataset. In fact, these newly-proposed correct lemmas consist of 37.7% of the training replay buffer when we train on the dataset extracted from Archive of Formal Proofs (AFP). The model trained by our RL algorithm outperforms that trained by supervised finetuning, improving the pass rate from 40.8% to 45.5% on AFP test set, and from 36.5% to 39.5% on an out-of-distribution test set.
     </details>

48. **Regress, Don't Guess -- A Regression-like Loss on Number Tokens for Language Models** [[pdf]](http://arxiv.org/abs/2411.02083) `2024-11-04` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work presents two versions of a number token loss that minimizes the Wasserstein-1 distance between the distribution of the predicted output probabilities and the ground truth distribution and compares the proposed schemes on a mathematics dataset against existing tokenization, encoding, and decoding schemes.
     </details>


     <details>
          <summary>Abstract</summary>
          While language models have exceptional capabilities at text generation, they lack a natural inductive bias for emitting numbers and thus struggle in tasks involving reasoning over quantities, especially arithmetics. This has particular relevance in scientific datasets where combinations of text and numerical data are abundant. One fundamental limitation is the nature of the CE loss, which assumes a nominal (categorical) scale and thus cannot convey proximity between generated number tokens. As a remedy, we here present two versions of a number token loss. The first is based on an $L_p$ loss between the ground truth token value and the weighted sum of the predicted class probabilities. The second loss minimizes the Wasserstein-1 distance between the distribution of the predicted output probabilities and the ground truth distribution. These regression-like losses can easily be added to any language model and extend the CE objective during training. We compare the proposed schemes on a mathematics dataset against existing tokenization, encoding, and decoding schemes for improving number representation in language models. Our results reveal a significant improvement in numerical accuracy when equipping a standard T5 model with the proposed loss schemes.
     </details>

49. **Learning Rules Explaining Interactive Theorem Proving Tactic Prediction** [[pdf]](http://arxiv.org/abs/2411.01188) `2024-11-02` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A machine learning based guidance for tactic selection that uses the learned rules to filter the output of an existing tactic selection approach and empirically show improvement over the non-filtering approaches.
     </details>


     <details>
          <summary>Abstract</summary>
          Formally verifying the correctness of mathematical proofs is more accessible than ever, however, the learning curve remains steep for many of the state-of-the-art interactive theorem provers (ITP). Deriving the most appropriate subsequent proof step, and reasoning about it, given the multitude of possibilities, remains a daunting task for novice users. To improve the situation, several investigations have developed machine learning based guidance for tactic selection. Such approaches struggle to learn non-trivial relationships between the chosen tactic and the structure of the proof state and represent them as symbolic expressions. To address these issues we (i) We represent the problem as an Inductive Logic Programming (ILP) task, (ii) Using the ILP representation we enriched the feature space by encoding additional, computationally expensive properties as background knowledge predicates, (iii) We use this enriched feature space to learn rules explaining when a tactic is applicable to a given proof state, (iv) we use the learned rules to filter the output of an existing tactic selection approach and empirically show improvement over the non-filtering approaches.
     </details>

50. **PatternBoost: Constructions in Mathematics with a Little Help from AI** [[pdf]](http://arxiv.org/abs/2411.00566) `2024-11-01` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Using this technique, the best known solutions to several long-standing problems are found, including the construction of a counterexample to a conjecture that had remained open for 30 years.
     </details>


     <details>
          <summary>Abstract</summary>
          We introduce PatternBoost, a flexible method for finding interesting constructions in mathematics. Our algorithm alternates between two phases. In the first ``local'' phase, a classical search algorithm is used to produce many desirable constructions. In the second ``global'' phase, a transformer neural network is trained on the best such constructions. Samples from the trained transformer are then used as seeds for the first phase, and the process is repeated. We give a detailed introduction to this technique, and discuss the results of its application to several problems in extremal combinatorics. The performance of PatternBoost varies across different problems, but there are many situations where its performance is quite impressive. Using our technique, we find the best known solutions to several long-standing problems, including the construction of a counterexample to a conjecture that had remained open for 30 years.
     </details>

51. **Can Language Models Perform Robust Reasoning in Chain-of-thought Prompting with Noisy Rationales?** [[pdf]](http://arxiv.org/abs/2410.23856) `2024-10-31` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The method of contrastive denoising with noisy chain-of-thought (CD-CoT) is proposed, which enhances LLMs' denoising-reasoning capabilities by contrasting noisy rationales with only one clean rationale, which can be the minimal requirement for denoising-purpose prompting.
     </details>


     <details>
          <summary>Abstract</summary>
          This paper investigates an under-explored challenge in large language models (LLMs): chain-of-thought prompting with noisy rationales, which include irrelevant or inaccurate reasoning thoughts within examples used for in-context learning. We construct NoRa dataset that is tailored to evaluate the robustness of reasoning in the presence of noisy rationales. Our findings on NoRa dataset reveal a prevalent vulnerability to such noise among current LLMs, with existing robust methods like self-correction and self-consistency showing limited efficacy. Notably, compared to prompting with clean rationales, base LLM drops by 1.4%-19.8% in accuracy with irrelevant thoughts and more drastically by 2.2%-40.4% with inaccurate thoughts.   Addressing this challenge necessitates external supervision that should be accessible in practice. Here, we propose the method of contrastive denoising with noisy chain-of-thought (CD-CoT). It enhances LLMs' denoising-reasoning capabilities by contrasting noisy rationales with only one clean rationale, which can be the minimal requirement for denoising-purpose prompting. This method follows a principle of exploration and exploitation: (1) rephrasing and selecting rationales in the input space to achieve explicit denoising and (2) exploring diverse reasoning paths and voting on answers in the output space. Empirically, CD-CoT demonstrates an average improvement of 17.8% in accuracy over the base model and shows significantly stronger denoising capabilities than baseline methods. The source code is publicly available at: https://github.com/tmlr-group/NoisyRationales.
     </details>

52. **What Happened in LLMs Layers when Trained for Fast vs. Slow Thinking: A Gradient Perspective** [[pdf]](http://arxiv.org/abs/2410.23743) `2024-10-31` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          In this study, fast thinking without CoT leads to larger gradients and larger differences of gradients across layers than slow thinking (Detailed CoT), indicating the learning stability brought by the latter.
     </details>


     <details>
          <summary>Abstract</summary>
          What makes a difference in the post-training of LLMs? We investigate the training patterns of different layers in large language models (LLMs), through the lens of gradient, when training with different responses and initial models. We are specifically interested in how fast vs. slow thinking affects the layer-wise gradients, given the recent popularity of training LLMs on reasoning paths such as chain-of-thoughts (CoT) and process rewards. In our study, fast thinking without CoT leads to larger gradients and larger differences of gradients across layers than slow thinking (Detailed CoT), indicating the learning stability brought by the latter. Moreover, pre-trained LLMs are less affected by the instability of fast thinking than instruction-tuned LLMs. Additionally, we study whether the gradient patterns can reflect the correctness of responses when training different LLMs using slow vs. fast thinking paths. The results show that the gradients of slow thinking can distinguish correct and irrelevant reasoning paths. As a comparison, we conduct similar gradient analyses on non-reasoning knowledge learning tasks, on which, however, trivially increasing the response length does not lead to similar behaviors of slow thinking. Our study strengthens fundamental understandings of LLM training and sheds novel insights on its efficiency and stability, which pave the way towards building a generalizable System-2 agent. Our code, data, and gradient statistics can be found in: https://github.com/MingLiiii/Layer_Gradient.
     </details>

53. **VisAidMath: Benchmarking Visual-Aided Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2410.22995) `2024-10-30` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          In-depth analysis reveals that the main cause of deficiencies lies in hallucination regarding the implicit visual reasoning process, shedding light on future research directions in the visual-aided MPS process.
     </details>


     <details>
          <summary>Abstract</summary>
          Although previous research on large language models (LLMs) and large multi-modal models (LMMs) has systematically explored mathematical problem-solving (MPS) within visual contexts, the analysis of how these models process visual information during problem-solving remains insufficient. To address this gap, we present VisAidMath, a benchmark for evaluating the MPS process related to visual information. We follow a rigorous data curation pipeline involving both automated processes and manual annotations to ensure data quality and reliability. Consequently, this benchmark includes 1,200 challenging problems from various mathematical branches, vision-aid formulations, and difficulty levels, collected from diverse sources such as textbooks, examination papers, and Olympiad problems. Based on the proposed benchmark, we conduct comprehensive evaluations on ten mainstream LLMs and LMMs, highlighting deficiencies in the visual-aided reasoning process. For example, GPT-4V only achieves 45.33% accuracy in the visual-aided reasoning task, even with a drop of 2 points when provided with golden visual aids. In-depth analysis reveals that the main cause of deficiencies lies in hallucination regarding the implicit visual reasoning process, shedding light on future research directions in the visual-aided MPS process.
     </details>

54. **DynaMath: A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Language Models** [[pdf]](http://arxiv.org/abs/2411.00836) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          The rapid advancements in Vision-Language Models (VLMs) have shown great potential in tackling mathematical reasoning tasks that involve visual context. Unlike humans who can reliably apply solution steps to similar problems with minor modifications, we found that SOTA VLMs like GPT-4o can consistently fail in these scenarios, revealing limitations in their mathematical reasoning capabilities. In this paper, we investigate the mathematical reasoning robustness in VLMs and evaluate how well these models perform under different variants of the same question, such as changes in visual numerical values or function graphs. While several vision-based math benchmarks have been developed to assess VLMs' problem-solving capabilities, these benchmarks contain only static sets of problems and cannot easily evaluate mathematical reasoning robustness. To fill this gap, we introduce DynaMath, a dynamic visual math benchmark designed for in-depth assessment of VLMs. DynaMath includes 501 high-quality, multi-topic seed questions, each represented as a Python program. Those programs are carefully designed and annotated to enable the automatic generation of a much larger set of concrete questions, including many different types of visual and textual variations. DynaMath allows us to evaluate the generalization ability of VLMs, by assessing their performance under varying input conditions of a seed question. We evaluated 14 SOTA VLMs with 5,010 generated concrete questions. Our results show that the worst-case model accuracy, defined as the percentage of correctly answered seed questions in all 10 variants, is significantly lower than the average-case accuracy. Our analysis emphasizes the need to study the robustness of VLMs' reasoning abilities, and DynaMath provides valuable insights to guide the development of more reliable models for mathematical reasoning.
     </details>

55. **Let's Be Self-generated via Step by Step: A Curriculum Learning Approach to Automated Reasoning with Large Language Models** [[pdf]](http://arxiv.org/abs/2410.21728) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel prompt approach for automatic reasoning named LBS3, inspired by curriculum learning which better reflects human learning habits is proposed, and it achieves strongly competitive performance compared to the SOTA baselines.
     </details>


     <details>
          <summary>Abstract</summary>
          While Chain of Thought (CoT) prompting approaches have significantly consolidated the reasoning capabilities of large language models (LLMs), they still face limitations that require extensive human effort or have performance needs to be improved. Existing endeavors have focused on bridging these gaps; however, these approaches either hinge on external data and cannot completely eliminate manual effort, or they fall short in effectively directing LLMs to generate high-quality exemplary prompts. To address the said pitfalls, we propose a novel prompt approach for automatic reasoning named \textbf{LBS3}, inspired by curriculum learning which better reflects human learning habits. Specifically, LBS3 initially steers LLMs to recall easy-to-hard proxy queries that are pertinent to the target query. Following this, it invokes a progressive strategy that utilizes exemplary prompts stemmed from easy-proxy queries to direct LLMs in solving hard-proxy queries, enabling the high-quality of the proxy solutions. Finally, our extensive experiments in various reasoning-intensive tasks with varying open- and closed-source LLMs show that LBS3 achieves strongly competitive performance compared to the SOTA baselines.
     </details>

56. **Arithmetic Without Algorithms: Language Models Solve Math With a Bag of Heuristics** [[pdf]](http://arxiv.org/abs/2410.21272) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Experimental results across several LLMs show that LLMs perform arithmetic using neither robust algorithms nor memorization; rather, they rely on a"bag of heuristics".
     </details>


     <details>
          <summary>Abstract</summary>
          Do large language models (LLMs) solve reasoning tasks by learning robust generalizable algorithms, or do they memorize training data? To investigate this question, we use arithmetic reasoning as a representative task. Using causal analysis, we identify a subset of the model (a circuit) that explains most of the model's behavior for basic arithmetic logic and examine its functionality. By zooming in on the level of individual circuit neurons, we discover a sparse set of important neurons that implement simple heuristics. Each heuristic identifies a numerical input pattern and outputs corresponding answers. We hypothesize that the combination of these heuristic neurons is the mechanism used to produce correct arithmetic answers. To test this, we categorize each neuron into several heuristic types-such as neurons that activate when an operand falls within a certain range-and find that the unordered combination of these heuristic types is the mechanism that explains most of the model's accuracy on arithmetic prompts. Finally, we demonstrate that this mechanism appears as the main source of arithmetic accuracy early in training. Overall, our experimental results across several LLMs show that LLMs perform arithmetic using neither robust algorithms nor memorization; rather, they rely on a "bag of heuristics".
     </details>

57. **Guiding Through Complexity: What Makes Good Supervision for Hard Reasoning Tasks?** [[pdf]](http://arxiv.org/abs/2410.20533) `2024-10-27` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is found that even when the outcome error rate for hard task supervision is high, training on such data can outperform perfectly correct supervision on easier subtasks on multiple hard math benchmarks, suggesting new avenues for data augmentation.
     </details>


     <details>
          <summary>Abstract</summary>
          How can "weak teacher models" such as average human annotators or existing AI systems, effectively supervise LLMs to improve performance on hard reasoning tasks, especially those that challenge and requires expertise or daily practice from the teacher models? In this paper, we seek for empirical answers to this question by investigating various data-driven strategies that offer supervision data at different quality levels upon tasks of varying complexity. Two intuitive strategies emerge for teacher models to provide supervision during alignment training: 1) using lower-quality supervision from complete tasks that match the difficulty of the target reasoning tasks, and 2) leveraging higher-quality supervision from easier subtasks that are less challenging. Interestingly, we find that even when the outcome error rate for hard task supervision is high (e.g., 90\%), training on such data can outperform perfectly correct supervision on easier subtasks on multiple hard math benchmarks. We further identify a more critical factor influencing training performance: step-wise error rates, which indicate the severity of errors in solutions. Specifically, training on hard task supervision with the same outcome error rates but disparate step-wise error rates can lead to a 30\% accuracy gap on MATH benchmark. Our results also reveal that supplementing hard task supervision with the corresponding subtask supervision can yield notable performance improvements than simply combining rephrased hard full task supervision, suggesting new avenues for data augmentation. Data and code are released at \url{https://github.com/hexuan21/Weak-to-Strong}.
     </details>

58. **GFlowNet Fine-tuning for Diverse Correct Solutions in Mathematical Reasoning Tasks** [[pdf]](http://arxiv.org/abs/2410.20147) `2024-10-26` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The results show that GFlowNet fine-tuning derives correct final answers from diverse intermediate reasoning steps, indicating the improvement of the capability of alternative solution generation.
     </details>


     <details>
          <summary>Abstract</summary>
          Mathematical reasoning problems are among the most challenging, as they typically require an understanding of fundamental laws to solve. The laws are universal, but the derivation of the final answer changes depending on how a problem is approached. When training large language models (LLMs), learning the capability of generating such multiple solutions is essential to accelerate their use in mathematical education. To this end, we train LLMs using generative flow network (GFlowNet). Different from reward-maximizing reinforcement learning (RL), GFlowNet fine-tuning seeks to find diverse solutions by training the LLM whose distribution is proportional to a reward function. In numerical experiments, we evaluate GFlowNet fine-tuning and reward-maximizing RL in terms of accuracy and diversity. The results show that GFlowNet fine-tuning derives correct final answers from diverse intermediate reasoning steps, indicating the improvement of the capability of alternative solution generation.
     </details>

59. **Reasoning or a Semblance of it? A Diagnostic Study of Transitive Reasoning in LLMs** [[pdf]](http://arxiv.org/abs/2410.20200) `2024-10-26` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Investigating the transitive reasoning capabilities of two distinct LLM architectures, LLaMA 2 and Flan-T5, by manipulating facts within two compositional datasets: QASC and Bamboogle reveals that Flan-T5 shows more resilience to experiments, suggesting that models may develop an understanding of transitivity through fine-tuning on knowingly relevant datasets.
     </details>


     <details>
          <summary>Abstract</summary>
          Evaluating Large Language Models (LLMs) on reasoning benchmarks demonstrates their ability to solve compositional questions. However, little is known of whether these models engage in genuine logical reasoning or simply rely on implicit cues to generate answers. In this paper, we investigate the transitive reasoning capabilities of two distinct LLM architectures, LLaMA 2 and Flan-T5, by manipulating facts within two compositional datasets: QASC and Bamboogle. We controlled for potential cues that might influence the models' performance, including (a) word/phrase overlaps across sections of test input; (b) models' inherent knowledge during pre-training or fine-tuning; and (c) Named Entities. Our findings reveal that while both models leverage (a), Flan-T5 shows more resilience to experiments (b and c), having less variance than LLaMA 2. This suggests that models may develop an understanding of transitivity through fine-tuning on knowingly relevant datasets, a hypothesis we leave to future work.
     </details>

60. **As Simple as Fine-tuning: LLM Alignment via Bidirectional Negative Feedback Loss** [[pdf]](http://arxiv.org/abs/2410.04834) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a novel LLM alignment loss that establishes a stable Bidirectional Negative Feedback (BNF) during optimization, and eliminates the need for pairwise contrastive losses and does not require any extra tunable hyper-parameters or pairwise preference data, streamlining the alignment pipeline to be as simple as supervised fine-tuning.
     </details>


     <details>
          <summary>Abstract</summary>
          Direct Preference Optimization (DPO) has emerged as a more computationally efficient alternative to Reinforcement Learning from Human Feedback (RLHF) with Proximal Policy Optimization (PPO), eliminating the need for reward models and online sampling. Despite these benefits, DPO and its variants remain sensitive to hyper-parameters and prone to instability, particularly on mathematical datasets. We argue that these issues arise from the unidirectional likelihood-derivative negative feedback inherent in the log-likelihood loss function. To address this, we propose a novel LLM alignment loss that establishes a stable Bidirectional Negative Feedback (BNF) during optimization. Our proposed BNF loss eliminates the need for pairwise contrastive losses and does not require any extra tunable hyper-parameters or pairwise preference data, streamlining the alignment pipeline to be as simple as supervised fine-tuning. We conduct extensive experiments across two challenging QA benchmarks and four reasoning benchmarks. The experimental results show that BNF achieves comparable performance to the best methods on QA benchmarks, while its performance decrease on the four reasoning benchmarks is significantly lower compared to the best methods, thus striking a better balance between value alignment and reasoning ability. In addition, we further validate the performance of BNF on non-pairwise datasets, and conduct in-depth analysis of log-likelihood and logit shifts across different preference optimization methods.
     </details>

61. **Can Stories Help LLMs Reason? Curating Information Space Through Narrative** [[pdf]](http://arxiv.org/abs/2410.19221) `2024-10-25` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Investigation of whether incorporating narrative elements can assist Large Language Models (LLMs) in solving complex problems more effectively finds that using various LLMs with SoT consistently surpasses using them with other techniques on physics, chemistry, math, and biology questions in both the GPQA and JEEBench datasets.
     </details>


     <details>
          <summary>Abstract</summary>
          Narratives are widely recognized as a powerful tool for structuring information and facilitating comprehension of complex ideas in various domains such as science communication. This paper investigates whether incorporating narrative elements can assist Large Language Models (LLMs) in solving complex problems more effectively. We propose a novel approach, Story of Thought (SoT), integrating narrative structures into prompting techniques for problem-solving. This approach involves constructing narratives around problem statements and creating a framework to identify and organize relevant information. Our experiments show that using various LLMs with SoT consistently surpasses using them with other techniques on physics, chemistry, math, and biology questions in both the GPQA and JEEBench datasets. The narrative-based information curation process in SoT enhances problem comprehension by contextualizing critical in-domain information and highlighting causal relationships within the problem space.
     </details>

62. **Cooperative Strategic Planning Enhances Reasoning Capabilities in Large Language Models** [[pdf]](http://arxiv.org/abs/2410.20007) `2024-10-25` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper proposes a novel cooperative multi-agent reasoning framework (CoPlanner) by separating reasoning steps and assigning distinct duties to different agents and demonstrates that the guidance from the planning agent and the effective cooperation between the agents contribute to the superior performance of CoPlanner in tackling multi-step reasoning problems.
     </details>


     <details>
          <summary>Abstract</summary>
          Enhancing the reasoning capabilities of large language models (LLMs) is crucial for enabling them to tackle complex, multi-step problems. Multi-agent frameworks have shown great potential in enhancing LLMs' reasoning capabilities. However, the lack of effective cooperation between LLM agents hinders their performance, especially for multi-step reasoning tasks. This paper proposes a novel cooperative multi-agent reasoning framework (CoPlanner) by separating reasoning steps and assigning distinct duties to different agents. CoPlanner consists of two LLM agents: a planning agent and a reasoning agent. The planning agent provides high-level strategic hints, while the reasoning agent follows these hints and infers answers. By training the planning agent's policy through the interactive reasoning process via Proximal Policy Optimization (PPO), the LLaMA-3-8B-based CoPlanner outperforms the previous best method by 9.94\% on LogiQA and 3.09\% on BBH. Our results demonstrate that the guidance from the planning agent and the effective cooperation between the agents contribute to the superior performance of CoPlanner in tackling multi-step reasoning problems.
     </details>

63. **Assessing the Creativity of LLMs in Proposing Novel Solutions to Mathematical Problems** [[pdf]](http://arxiv.org/abs/2410.18336) `2024-10-24` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This study explores the creative potential of Large Language Models in mathematical reasoning, an aspect that has received limited attention in prior research, and introduces a novel framework and benchmark, CreativeMath, designed to assess LLMs' ability to propose innovative solutions after some known solutions have been provided.
     </details>


     <details>
          <summary>Abstract</summary>
          The mathematical capabilities of AI systems are complex and multifaceted. Most existing research has predominantly focused on the correctness of AI-generated solutions to mathematical problems. In this work, we argue that beyond producing correct answers, AI systems should also be capable of, or assist humans in, developing novel solutions to mathematical challenges. This study explores the creative potential of Large Language Models (LLMs) in mathematical reasoning, an aspect that has received limited attention in prior research. We introduce a novel framework and benchmark, CreativeMath, which encompasses problems ranging from middle school curricula to Olympic-level competitions, designed to assess LLMs' ability to propose innovative solutions after some known solutions have been provided. Our experiments demonstrate that, while LLMs perform well on standard mathematical tasks, their capacity for creative problem-solving varies considerably. Notably, the Gemini-1.5-Pro model outperformed other LLMs in generating novel solutions. This research opens a new frontier in evaluating AI creativity, shedding light on both the strengths and limitations of LLMs in fostering mathematical innovation, and setting the stage for future developments in AI-assisted mathematical discovery.
     </details>

64. **From Blind Solvers to Logical Thinkers: Benchmarking LLMs' Logical Integrity on Faulty Mathematical Problems** [[pdf]](http://arxiv.org/abs/2410.18921) `2024-10-24` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A broad spectrum of LLMs are evaluated, including open-source, closed-source, and math-specialized models, using FaultyMath, to demonstrate that existing LLMs largely function as Blind Solver and fall short of the reasoning capabilities required to perform as Logical Thinker.
     </details>


     <details>
          <summary>Abstract</summary>
          Consider the math problem: "Lily received 3 cookies from her best friend yesterday and ate 5 for breakfast. Today, her friend gave her 3 more cookies. How many cookies does Lily have now?" Many large language models (LLMs) in previous research approach this problem by calculating the answer "1" using the equation "3 - 5 + 3." However, from a human perspective, we recognize the inherent flaw in this problem: Lily cannot eat 5 cookies if she initially only had 3. This discrepancy prompts a key question: Are current LLMs merely Blind Solver that apply mathematical operations without deeper reasoning, or can they function as Logical Thinker capable of identifying logical inconsistencies?   To explore this question, we propose a benchmark dataset, FaultyMath, which includes faulty math problems of rich diversity: i) multiple mathematical categories, e.g., algebra, geometry, number theory, etc., ii) varying levels of difficulty, and iii) different origins of faultiness -- ranging from violations of common sense and ambiguous statements to mathematical contradictions and more. We evaluate a broad spectrum of LLMs, including open-source, closed-source, and math-specialized models, using FaultyMath across three dimensions: (i) How accurately can the models detect faulty math problems without being explicitly prompted to do so? (ii) When provided with hints -- either correct or misleading -- about the validity of the problems, to what extent do LLMs adapt to become reliable Logical Thinker? (iii) How trustworthy are the explanations generated by LLMs when they recognize a math problem as flawed? Through extensive experimentation and detailed analysis, our results demonstrate that existing LLMs largely function as Blind Solver and fall short of the reasoning capabilities required to perform as Logical Thinker.
     </details>

65. **ReasonAgain: Using Extractable Symbolic Programs to Evaluate Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2410.19056) `2024-10-24` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Significant accuracy drops are observed using the proposed evaluation compared with original static examples, suggesting the fragility of math reasoning in state-of-the-art LLMs.
     </details>


     <details>
          <summary>Abstract</summary>
          Existing math datasets evaluate the reasoning abilities of large language models (LLMs) by either using the final answer or the intermediate reasoning steps derived from static examples. However, the former approach fails to surface model's uses of shortcuts and wrong reasoning while the later poses challenges in accommodating alternative solutions. In this work, we seek to use symbolic programs as a means for automated evaluation if a model can consistently produce correct final answers across various inputs to the program. We begin by extracting programs for popular math datasets (GSM8K and MATH) using GPT4-o. For those executable programs verified using the original input-output pairs, they are found to encapsulate the proper reasoning required to solve the original text questions. We then prompt GPT4-o to generate new questions using alternative input-output pairs based the extracted program. We apply the resulting datasets to evaluate a collection of LLMs. In our experiments, we observe significant accuracy drops using our proposed evaluation compared with original static examples, suggesting the fragility of math reasoning in state-of-the-art LLMs.
     </details>

66. **SIKeD: Self-guided Iterative Knowledge Distillation for mathematical reasoning** [[pdf]](http://arxiv.org/abs/2410.18574) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a distillation method SIKeD (Self-guided Iterative Knowledge Distillation for mathematical reasoning), where the large language model teaches the smaller model to approach a task using different strategies and the smaller model uses its self-generated on-policy outputs to choose the most suitable strategy for the given task.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) can transfer their reasoning skills to smaller models by teaching them to generate the intermediate reasoning process required to solve multistep reasoning tasks. While LLMs can accurately solve reasoning tasks through a variety of strategies, even without fine-tuning, smaller models are not expressive enough to fit the LLMs distribution on all strategies when distilled and tend to prioritize one strategy over the others. This reliance on one strategy poses a challenge for smaller models when attempting to solve reasoning tasks that may be difficult with their preferred strategy. To address this, we propose a distillation method SIKeD (Self-guided Iterative Knowledge Distillation for mathematical reasoning), where the LLM teaches the smaller model to approach a task using different strategies and the smaller model uses its self-generated on-policy outputs to choose the most suitable strategy for the given task. The training continues in a self-guided iterative manner, where for each training iteration, a decision is made on how to combine the LLM data with the self-generated outputs. Unlike traditional distillation methods, SIKeD allows the smaller model to learn which strategy is suitable for a given task while continuously learning to solve a task using different strategies. Our experiments on various mathematical reasoning datasets show that SIKeD significantly outperforms traditional distillation techniques across smaller models of different sizes. Our code is available at: https://github.com/kumar-shridhar/SIKeD
     </details>

67. **Unleashing Reasoning Capability of LLMs via Scalable Question Synthesis from Scratch** [[pdf]](http://arxiv.org/abs/2410.18693) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          The availability of high-quality data is one of the most important factors in improving the reasoning capability of LLMs. Existing works have demonstrated the effectiveness of creating more instruction data from seed questions or knowledge bases. Recent research indicates that continually scaling up data synthesis from strong models (e.g., GPT-4) can further elicit reasoning performance. Though promising, the open-sourced community still lacks high-quality data at scale and scalable data synthesis methods with affordable costs. To address this, we introduce ScaleQuest, a scalable and novel data synthesis method that utilizes "small-size" (e.g., 7B) open-source models to generate questions from scratch without the need for seed data with complex augmentation constraints. With the efficient ScaleQuest, we automatically constructed a mathematical reasoning dataset consisting of 1 million problem-solution pairs, which are more effective than existing open-sourced datasets. It can universally increase the performance of mainstream open-source models (i.e., Mistral, Llama3, DeepSeekMath, and Qwen2-Math) by achieving 29.2% to 46.4% gains on MATH. Notably, simply fine-tuning the Qwen2-Math-7B-Base model with our dataset can even surpass Qwen2-Math-7B-Instruct, a strong and well-aligned model on closed-source data, and proprietary models such as GPT-4-Turbo and Claude-3.5 Sonnet.
     </details>

68. **CLR-Bench: Evaluating Large Language Models in College-level Reasoning** [[pdf]](http://arxiv.org/abs/2410.17558) `2024-10-23` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The results demonstrate the key insights that LLMs, even the best closed-source LLM, i.e., GPT-4 turbo, tend to `guess' the college-level answers.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have demonstrated their remarkable performance across various language understanding tasks. While emerging benchmarks have been proposed to evaluate LLMs in various domains such as mathematics and computer science, they merely measure the accuracy in terms of the final prediction on multi-choice questions. However, it remains insufficient to verify the essential understanding of LLMs given a chosen choice. To fill this gap, we present CLR-Bench to comprehensively evaluate the LLMs in complex college-level reasoning. Specifically, (i) we prioritize 16 challenging college disciplines in computer science and artificial intelligence. The dataset contains 5 types of questions, while each question is associated with detailed explanations from experts. (ii) To quantify a fair evaluation of LLMs' reasoning ability, we formalize the criteria with two novel metrics. Q$\rightarrow$A is utilized to measure the performance of direct answer prediction, and Q$\rightarrow$AR effectively considers the joint ability to answer the question and provide rationale simultaneously. Extensive experiments are conducted with 40 LLMs over 1,018 discipline-specific questions. The results demonstrate the key insights that LLMs, even the best closed-source LLM, i.e., GPT-4 turbo, tend to `guess' the college-level answers. It shows a dramatic decrease in accuracy from 63.31% Q$\rightarrow$A to 39.00% Q$\rightarrow$AR, indicating an unsatisfactory reasoning ability.
     </details>

69. **Markov Chain of Thought for Efficient Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2410.17635) `2024-10-23` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work conceptualizes the standard multi-step CoT as a novel Markov Chain of Thought (MCoT), defining each reasoning step as text accompanied by a Python code snippet, and the empirical results indicate that MCoT not only significantly enhances efficiency but also maintains comparable accuracy.
     </details>


     <details>
          <summary>Abstract</summary>
          Chain of Thought (CoT) of multi-step benefits from the logical structure of the reasoning steps and task-specific actions, significantly enhancing the mathematical reasoning capabilities of large language models. As the prevalence of long CoT, the number of reasoning steps exceeds manageable token limits and leads to higher computational demands. Inspired by the fundamental logic of human cognition, ``derive, then reduce'', we conceptualize the standard multi-step CoT as a novel Markov Chain of Thought (MCoT). In this study, we consider the mathematical reasoning task, defining each reasoning step as text accompanied by a Python code snippet. To facilitate a longer reasoning path, self-correction is enabled through interactions with the code interpreter. Our MCoT aims to compress previous reasoning steps into a simplified question, enabling efficient next-step inference without relying on a lengthy KV cache. In our experiments, we curate the \texttt{MCoTInstruct} dataset, and the empirical results indicate that MCoT not only significantly enhances efficiency but also maintains comparable accuracy. While much remains to be explored, this work paves the way for exploring the long CoT reasoning abilities of LLMs.
     </details>

70. **R-CoT: Reverse Chain-of-Thought Problem Generation for Geometric Reasoning in Large Multimodal Models** [[pdf]](http://arxiv.org/abs/2410.17885) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Existing Large Multimodal Models (LMMs) struggle with mathematical geometric reasoning due to a lack of high-quality image-text paired data. Current geometric data generation approaches, which apply preset templates to generate geometric data or use Large Language Models (LLMs) to rephrase questions and answers (Q&A), unavoidably limit data accuracy and diversity. To synthesize higher-quality data, we propose a two-stage Reverse Chain-of-Thought (R-CoT) geometry problem generation pipeline. First, we introduce GeoChain to produce high-fidelity geometric images and corresponding descriptions highlighting relations among geometric elements. We then design a Reverse A&Q method that reasons step-by-step based on the descriptions and generates questions in reverse from the reasoning results. Experiments demonstrate that the proposed method brings significant and consistent improvements on multiple LMM baselines, achieving new performance records in the 2B, 7B, and 8B settings. Notably, R-CoT-8B significantly outperforms previous state-of-the-art open-source mathematical models by 16.6% on MathVista and 9.2% on GeoQA, while also surpassing the closed-source model GPT-4o by an average of 13% across both datasets. The code is available at https://github.com/dle666/R-CoT.
     </details>

71. **Math Neurosurgery: Isolating Language Models' Math Reasoning Abilities Using Only Forward Passes** [[pdf]](http://arxiv.org/abs/2410.16930) `2024-10-22` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces Math Neurosurgery (MathNeuro), a method for isolating math-specific parameters in LLMs using only forward passes, and highlights the potential for future work to intervene on math-specific parameters.
     </details>


     <details>
          <summary>Abstract</summary>
          Math reasoning is a highly active area of Large Language Model (LLM) research because it is a hallmark of artificial intelligence. However, few works have explored how math reasoning is encoded within LLM parameters and if it is a skill that can be isolated within a model. Doing so could allow targeted intervention to improve math performance without altering non-math behavior and foster understanding of how models encode math reasoning. We introduce Math Neurosurgery (MathNeuro), a method for isolating math-specific parameters in LLMs using only forward passes. MathNeuro builds on existing work by using weights and activations to calculate parameter importance, but isolates math-specific parameters by removing those important for general language tasks. Pruning parameters MathNeuro identifies deletes a LLM's math reasoning ability without destroying its general language ability. Scaling these parameters by a small constant improves a pretrained or instruction-tuned LLM's performance by 4-17% on GSM8K while leaving non-math behavior unaltered. MathNeuro is also data efficient: most of its effectiveness holds when identifying math-specific parameters using a single sample. MathNeuro highlights the potential for future work to intervene on math-specific parameters.
     </details>

72. **Optimizing Chain-of-Thought Reasoning: Tackling Arranging Bottleneck via Plan Augmentation** [[pdf]](http://arxiv.org/abs/2410.16812) `2024-10-22` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This study subdivide CoT reasoning into two parts: arranging and executing, and identifies that the bottleneck of models mainly lies in arranging rather than executing, and proposes a plan-based training and reasoning method that guides models to generate arranging steps through abstract plans.
     </details>


     <details>
          <summary>Abstract</summary>
          Multi-step reasoning ability of large language models is crucial in tasks such as math and tool utilization. Current researches predominantly focus on enhancing model performance in these multi-step reasoning tasks through fine-tuning with Chain-of-Thought (CoT) steps, yet these methods tend to be heuristic, without exploring nor resolving the bottleneck. In this study, we subdivide CoT reasoning into two parts: arranging and executing, and identify that the bottleneck of models mainly lies in arranging rather than executing. Based on this finding, we propose a plan-based training and reasoning method that guides models to generate arranging steps through abstract plans. We experiment on both math (GSM8k) and tool utilization (ToolBench) benchmarks. Results show that compared to fine-tuning directly with CoT data, our approach achieves a better performance on alleviating arranging bottleneck, particularly excelling in long-distance reasoning generalization.
     </details>

73. **SMART: Self-learning Meta-strategy Agent for Reasoning Tasks** [[pdf]](http://arxiv.org/abs/2410.16128) `ICLR 2025 Submission` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          SMART (Self-learning Meta-strategy Agent for Reasoning Tasks), a novel framework that enables LMs to autonomously learn and select the most effective strategies for various reasoning tasks, and reduces computational costs for refinement-based strategies, paving the way for more efficient and intelligent reasoning in LMs.
     </details>


     <details>
          <summary>Abstract</summary>
          Tasks requiring deductive reasoning, especially those involving multiple steps, often demand adaptive strategies such as intermediate generation of rationales or programs, as no single approach is universally optimal. While Language Models (LMs) can enhance their outputs through iterative self-refinement and strategy adjustments, they frequently fail to apply the most effective strategy in their first attempt. This inefficiency raises the question: Can LMs learn to select the optimal strategy in the first attempt, without a need for refinement? To address this challenge, we introduce SMART (Self-learning Meta-strategy Agent for Reasoning Tasks), a novel framework that enables LMs to autonomously learn and select the most effective strategies for various reasoning tasks. We model the strategy selection process as a Markov Decision Process and leverage reinforcement learning-driven continuous self-improvement to allow the model to find the suitable strategy to solve a given task. Unlike traditional self-refinement methods that rely on multiple inference passes or external feedback, SMART allows an LM to internalize the outcomes of its own reasoning processes and adjust its strategy accordingly, aiming for correct solutions on the first attempt. Our experiments across various reasoning datasets and with different model architectures demonstrate that SMART significantly enhances the ability of models to choose optimal strategies without external guidance (+15 points on the GSM8K dataset). By achieving higher accuracy with a single inference pass, SMART not only improves performance but also reduces computational costs for refinement-based strategies, paving the way for more efficient and intelligent reasoning in LMs.
     </details>

74. **A Simple Model of Inference Scaling Laws** [[pdf]](http://arxiv.org/abs/2410.16377) `2024-10-21` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A simple statistical ansatz based on memorization is proposed to study scaling laws in the context of inference, specifically how performance improves with multiple inference attempts, and an"inference loss", which exhibits a power law decay as the number of trials increases.
     </details>


     <details>
          <summary>Abstract</summary>
          Neural scaling laws have garnered significant interest due to their ability to predict model performance as a function of increasing parameters, data, and compute. In this work, we propose a simple statistical ansatz based on memorization to study scaling laws in the context of inference, specifically how performance improves with multiple inference attempts. We explore the coverage, or pass@k metric, which measures the chance of success over repeated attempts and provide a motivation for the observed functional form of the inference scaling behavior of the coverage in large language models (LLMs) on reasoning tasks. We then define an "inference loss", which exhibits a power law decay as the number of trials increases, and connect this result with prompting costs. We further test our construction by conducting experiments on a simple generative model, and find that our predictions are in agreement with the empirical coverage curves in a controlled setting. Our simple framework sets the ground for incorporating inference scaling with other known scaling laws.
     </details>

75. **A Theoretical Understanding of Chain-of-Thought: Coherent Reasoning and Error-Aware Demonstration** [[pdf]](http://arxiv.org/abs/2410.16540) `2024-10-21` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work theoretically shows that, compared to Stepwise ICL, the transformer gains better error correction ability and more accurate predictions if the reasoning from earlier steps (Coherent CoT) is integrated, and proposes an improvement on CoT by incorporating both correct and incorrect reasoning paths in the demonstration.
     </details>


     <details>
          <summary>Abstract</summary>
          Few-shot Chain-of-Thought (CoT) prompting has demonstrated strong performance in improving the reasoning capabilities of large language models (LLMs). While theoretical investigations have been conducted to understand CoT, the underlying transformer used in these studies isolates the CoT reasoning process into separated in-context learning steps (Stepwise ICL). In this work, we theoretically show that, compared to Stepwise ICL, the transformer gains better error correction ability and more accurate predictions if the reasoning from earlier steps (Coherent CoT) is integrated. Given that this coherent reasoning changes the behavior of the transformer, we further investigate the sensitivity of the transformer with Coherent CoT when the demonstration examples are corrupted at the inference stage. Our theoretical results indicate that the transformer is more sensitive to errors in intermediate reasoning steps than the final outcome. Building upon this observation, we propose an improvement on CoT by incorporating both correct and incorrect reasoning paths in the demonstration. Our experiments validate the effectiveness of the proposed approach.
     </details>

76. **Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count** [[pdf]](http://arxiv.org/abs/2410.15787) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Transformers often struggle with length generalization, meaning they fail to generalize to sequences longer than those encountered during training. While arithmetic tasks are commonly used to study length generalization, certain tasks are considered notoriously difficult, e.g., multi-operand addition (requiring generalization over both the number of operands and their lengths) and multiplication (requiring generalization over both operand lengths). In this work, we achieve approximately 2-3x length generalization on both tasks, which is the first such achievement in arithmetic Transformers. We design task-specific scratchpads enabling the model to focus on a fixed number of tokens per each next-token prediction step, and apply multi-level versions of Position Coupling (Cho et al., 2024; McLeish et al., 2024) to let Transformers know the right position to attend to. On the theory side, we prove that a 1-layer Transformer using our method can solve multi-operand addition, up to operand length and operand count that are exponential in embedding dimension.
     </details>

77. **InternLM2.5-StepProver: Advancing Automated Theorem Proving via Expert Iteration on Large-Scale LEAN Problems** [[pdf]](http://arxiv.org/abs/2410.15700) `2024-10-21` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes to use large scale LEAN problem datasets Lean-workbook for expert iteration with more than 20,000 CPU days and finds log-linear trends between solved problem amount with proof length and CPU usage.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have emerged as powerful tools in mathematical theorem proving, particularly when utilizing formal languages such as LEAN. The major learning paradigm is expert iteration, which necessitates a pre-defined dataset comprising numerous mathematical problems. In this process, LLMs attempt to prove problems within the dataset and iteratively refine their capabilities through self-training on the proofs they discover. We propose to use large scale LEAN problem datasets Lean-workbook for expert iteration with more than 20,000 CPU days. During expert iteration, we found log-linear trends between solved problem amount with proof length and CPU usage. We train a critic model to select relatively easy problems for policy models to make trials and guide the model to search for deeper proofs. InternLM2.5-StepProver achieves open-source state-of-the-art on MiniF2F, Lean-Workbook-Plus, ProofNet, and Putnam benchmarks. Specifically, it achieves a pass of 65.9% on the MiniF2F-test and proves (or disproves) 17.0% of problems in Lean-Workbook-Plus which shows a significant improvement compared to only 9.5% of problems proved when Lean-Workbook-Plus was released. We open-source our models and searched proofs at https://github.com/InternLM/InternLM-Math and https://huggingface.co/datasets/internlm/Lean-Workbook.
     </details>

78. **Language Models are Symbolic Learners in Arithmetic** [[pdf]](http://arxiv.org/abs/2410.15580) `2024-10-21` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work confirms that LLMs are pure symbolic learners in arithmetic tasks and underscores the importance of understanding them deeply through subgroup-level quantification, as well as hypothesizing that difficulties arise from subgroup complexity and selection.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) are thought to struggle with arithmetic learning due to the inherent differences between language modeling and numerical computation, but concrete evidence has been lacking. This work responds to this claim through a two-side experiment. We first investigate whether LLMs leverage partial products during arithmetic learning. We find that although LLMs can identify some partial products after learning, they fail to leverage them for arithmetic tasks, conversely. We then explore how LLMs approach arithmetic symbolically by breaking tasks into subgroups, hypothesizing that difficulties arise from subgroup complexity and selection. Our results show that when subgroup complexity is fixed, LLMs treat a collection of different arithmetic operations similarly. By analyzing position-level accuracy across different training sizes, we further observe that it follows a U-shaped pattern: LLMs quickly learn the easiest patterns at the first and last positions, while progressively learning the more difficult patterns in the middle positions. This suggests that LLMs select subgroup following an easy-to-hard paradigm during learning. Our work confirms that LLMs are pure symbolic learners in arithmetic tasks and underscores the importance of understanding them deeply through subgroup-level quantification.
     </details>

79. **Pantograph: A Machine-to-Machine Interaction Interface for Advanced Theorem Proving, High Level Reasoning, and Data Extraction in Lean 4** [[pdf]](http://arxiv.org/abs/2410.16429) `2024-10-21` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Pantograph is introduced, a tool that provides a versatile interface to the Lean 4 proof assistant and enables efficient proof search via powerful search algorithms such as Monte Carlo Tree Search and enables high-level reasoning by enabling a more robust handling of Lean 4's inference steps.
     </details>


     <details>
          <summary>Abstract</summary>
          Machine-assisted theorem proving refers to the process of conducting structured reasoning to automatically generate proofs for mathematical theorems. Recently, there has been a surge of interest in using machine learning models in conjunction with proof assistants to perform this task. In this paper, we introduce Pantograph, a tool that provides a versatile interface to the Lean 4 proof assistant and enables efficient proof search via powerful search algorithms such as Monte Carlo Tree Search. In addition, Pantograph enables high-level reasoning by enabling a more robust handling of Lean 4's inference steps. We provide an overview of Pantograph's architecture and features. We also report on an illustrative use case: using machine learning models and proof sketches to prove Lean 4 theorems. Pantograph's innovative features pave the way for more advanced machine learning models to perform complex proof searches and high-level reasoning, equipping future researchers to design more versatile and powerful theorem provers.
     </details>

80. **ToW: Thoughts of Words Improve Reasoning in Large Language Models** [[pdf]](http://arxiv.org/abs/2410.16235) `2024-10-21` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces thoughts of words (ToW), a novel training-time data-augmentation method for next-word prediction that effectively improves models' reasoning performances by 7% to 9% on average and reduces model hallucination by up to 10%.
     </details>


     <details>
          <summary>Abstract</summary>
          We introduce thoughts of words (ToW), a novel training-time data-augmentation method for next-word prediction. ToW views next-word prediction as a core reasoning task and injects fine-grained thoughts explaining what the next word should be and how it is related to the previous contexts in pre-training texts. Our formulation addresses two fundamental drawbacks of existing next-word prediction learning schemes: they induce factual hallucination and are inefficient for models to learn the implicit reasoning processes in raw texts. While there are many ways to acquire such thoughts of words, we explore the first step of acquiring ToW annotations through distilling from larger models. After continual pre-training with only 70K ToW annotations, we effectively improve models' reasoning performances by 7% to 9% on average and reduce model hallucination by up to 10%. At the same time, ToW is entirely agnostic to tasks and applications, introducing no additional biases on labels or semantics.
     </details>

81. **Keep Guessing? When Considering Inference Scaling, Mind the Baselines** [[pdf]](http://arxiv.org/abs/2410.15466) `2024-10-20` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A baseline that enumerates answers according to their prevalence in the training set outperforms repeated model sampling for some LLMs, while the coverage for others is on par with that of a mixture strategy that obtains answers by using only model samples and similarly guessing the remaining attempts via enumeration.
     </details>


     <details>
          <summary>Abstract</summary>
          Scaling inference compute in large language models (LLMs) through repeated sampling consistently increases the coverage (fraction of problems solved) as the number of samples increases. We conjecture that this observed improvement is partially due to the answer distribution of standard evaluation benchmarks, which is skewed towards a relatively small set of common answers. To test this conjecture, we define a baseline that enumerates answers according to their prevalence in the training set. Experiments spanning two domains -- mathematical reasoning and factual knowledge -- reveal that this baseline outperforms repeated model sampling for some LLMs, while the coverage for others is on par with that of a mixture strategy that obtains $k$ answers by using only $10$ model samples and similarly guessing the remaining $k-10$ attempts via enumeration. Our baseline enables a more accurate measurement of how much repeated sampling improves coverage in such settings beyond prompt-agnostic guessing.
     </details>

82. **On Designing Effective RL Reward at Training Time for LLM Reasoning** [[pdf]](http://arxiv.org/abs/2410.15115) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Reward models have been increasingly critical for improving the reasoning capability of LLMs. Existing research has shown that a well-trained reward model can substantially improve model performances at inference time via search. However, the potential of reward models during RL training time still remains largely under-explored. It is currently unclear whether these reward models can provide additional training signals to enhance the reasoning capabilities of LLMs in RL training that uses sparse success rewards, which verify the correctness of solutions. In this work, we evaluate popular reward models for RL training, including the Outcome-supervised Reward Model (ORM) and the Process-supervised Reward Model (PRM), and train a collection of LLMs for math problems using RL by combining these learned rewards with success rewards. Surprisingly, even though these learned reward models have strong inference-time performances, they may NOT help or even hurt RL training, producing worse performances than LLMs trained with the success reward only. Our analysis reveals that an LLM can receive high rewards from some of these reward models by repeating correct but unnecessary reasoning steps, leading to a severe reward hacking issue. Therefore, we introduce two novel reward refinement techniques, including Clipping and Delta. The key idea is to ensure the accumulative reward of any reasoning trajectory is upper-bounded to keep a learned reward model effective without being exploited. We evaluate our techniques with multiple reward models over a set of 1.5B and 7B LLMs on MATH and GSM8K benchmarks and demonstrate that with a carefully designed reward function, RL training without any additional supervised tuning can improve all the evaluated LLMs, including the state-of-the-art 7B LLM Qwen2.5-Math-7B-Instruct on MATH and GSM8K benchmarks.
     </details>

83. **Step Guided Reasoning: Improving Mathematical Reasoning using Guidance Generation and Step Reasoning** [[pdf]](http://arxiv.org/abs/2410.19817) `2024-10-18` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a novel method called Step Guidance Reasoning, in which LLMs reflect on small reasoning steps -- similar to how humans deliberate on and focus attention on what to do next and can effectively guide their reasoning from one step to the next.
     </details>


     <details>
          <summary>Abstract</summary>
          Mathematical reasoning has been a challenging aspect of large language models (LLMs). However, the introduction of step-by-step Chain-of-Thought (CoT) inference has significantly advanced the mathematical capabilities of LLMs. Despite this progress, current approaches either require massive inference datasets as training datasets or rely on few-shot methods that often sacrifice accuracy. To address this bottleneck in mathematical reasoning, we propose a novel method called Step Guidance Reasoning without involving further model fine-tuning. In this approach, LLMs reflect on small reasoning steps -- similar to how humans deliberate on and focus attention on what to do next. By incorporating this reflective process into the inference stage, LLMs can effectively guide their reasoning from one step to the next. Our method significantly improved the math performance, raising the accuracy on the AMC23 dataset from 30% to 57.5%, a relative improvement of 91.7%, and on the sampled level 5 problem of the MATH dataset, we achieved a relative accuracy improvement of 55.8%, increasing from 43% to 67%.
     </details>

84. **A Comparative Study on Reasoning Patterns of OpenAI's o1 Model** [[pdf]](http://arxiv.org/abs/2410.13639) `2024-10-17` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work compares o1 with existing Test-time Compute methods by using OpenAI's GPT-4o as a backbone on general reasoning benchmarks in three domains, and shows that the o1 model has achieved the best performance on most datasets.
     </details>


     <details>
          <summary>Abstract</summary>
          Enabling Large Language Models (LLMs) to handle a wider range of complex tasks (e.g., coding, math) has drawn great attention from many researchers. As LLMs continue to evolve, merely increasing the number of model parameters yields diminishing performance improvements and heavy computational costs. Recently, OpenAI's o1 model has shown that inference strategies (i.e., Test-time Compute methods) can also significantly enhance the reasoning capabilities of LLMs. However, the mechanisms behind these methods are still unexplored. In our work, to investigate the reasoning patterns of o1, we compare o1 with existing Test-time Compute methods (BoN, Step-wise BoN, Agent Workflow, and Self-Refine) by using OpenAI's GPT-4o as a backbone on general reasoning benchmarks in three domains (i.e., math, coding, commonsense reasoning). Specifically, first, our experiments show that the o1 model has achieved the best performance on most datasets. Second, as for the methods of searching diverse responses (e.g., BoN), we find the reward models' capability and the search space both limit the upper boundary of these methods. Third, as for the methods that break the problem into many sub-problems, the Agent Workflow has achieved better performance than Step-wise BoN due to the domain-specific system prompt for planning better reasoning processes. Fourth, it is worth mentioning that we have summarized six reasoning patterns of o1, and provided a detailed analysis on several reasoning benchmarks.
     </details>

85. **GeoCoder: Solving Geometry Problems by Generating Modular Code through Vision-Language Models** [[pdf]](http://arxiv.org/abs/2410.13510) `2024-10-17` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          GeoCoder is presented, which leverages modular code-finetuning to generate and execute code using a predefined geometry function library, and a multimodal retrieval-augmented variant of GeoCoder, named RAG-GeoCoder, which incorporates a non-parametric memory module for retrieving functions from the geometry library, thereby reducing reliance on parametric memory.
     </details>


     <details>
          <summary>Abstract</summary>
          Geometry problem-solving demands advanced reasoning abilities to process multimodal inputs and employ mathematical knowledge effectively. Vision-language models (VLMs) have made significant progress in various multimodal tasks. Yet, they still struggle with geometry problems and are significantly limited by their inability to perform mathematical operations not seen during pre-training, such as calculating the cosine of an arbitrary angle, and by difficulties in correctly applying relevant geometry formulas. To overcome these challenges, we present GeoCoder, which leverages modular code-finetuning to generate and execute code using a predefined geometry function library. By executing the code, we achieve accurate and deterministic calculations, contrasting the stochastic nature of autoregressive token prediction, while the function library minimizes errors in formula usage. We also propose a multimodal retrieval-augmented variant of GeoCoder, named RAG-GeoCoder, which incorporates a non-parametric memory module for retrieving functions from the geometry library, thereby reducing reliance on parametric memory. Our modular code-finetuning approach enhances the geometric reasoning capabilities of VLMs, yielding an average improvement of over 16% across various question complexities on the GeomVerse dataset compared to other finetuning methods.
     </details>

86. **How Numerical Precision Affects Mathematical Reasoning Capabilities of LLMs** [[pdf]](http://arxiv.org/abs/2410.13857) `2024-10-17` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A rigorous theoretical analysis of LLMs' mathematical abilities, with a specific focus on their arithmetic performances, shows that Transformers operating with low numerical precision fail to address arithmetic tasks, unless the model size grows super-polynomially with respect to the input length.
     </details>


     <details>
          <summary>Abstract</summary>
          Despite the remarkable success of Transformer-based Large Language Models (LLMs) across various domains, understanding and enhancing their mathematical capabilities remains a significant challenge. In this paper, we conduct a rigorous theoretical analysis of LLMs' mathematical abilities, with a specific focus on their arithmetic performances. We identify numerical precision as a key factor that influences their effectiveness in mathematical tasks. Our results show that Transformers operating with low numerical precision fail to address arithmetic tasks, such as iterated addition and integer multiplication, unless the model size grows super-polynomially with respect to the input length. In contrast, Transformers with standard numerical precision can efficiently handle these tasks with significantly smaller model sizes. We further support our theoretical findings through empirical experiments that explore the impact of varying numerical precision on arithmetic tasks, providing valuable insights for improving the mathematical reasoning capabilities of LLMs.
     </details>

87. **Proof Flow: Preliminary Study on Generative Flow Network Language Model Tuning for Formal Reasoning** [[pdf]](http://arxiv.org/abs/2410.13224) `2024-10-17` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper presents a proof of concept in the domain of formal reasoning, specifically in the Neural Theorem Proving (NTP) setting, where proofs specified in a formal language such as Lean can be deterministically and objectively verified.
     </details>


     <details>
          <summary>Abstract</summary>
          Reasoning is a fundamental substrate for solving novel and complex problems. Deliberate efforts in learning and developing frameworks around System 2 reasoning have made great strides, yet problems of sufficient complexity remain largely out of reach for open models. To address this gap, we examine the potential of Generative Flow Networks as a fine-tuning method for LLMs to unlock advanced reasoning capabilities. In this paper, we present a proof of concept in the domain of formal reasoning, specifically in the Neural Theorem Proving (NTP) setting, where proofs specified in a formal language such as Lean can be deterministically and objectively verified. Unlike classical reward-maximization reinforcement learning, which frequently over-exploits high-reward actions and fails to effectively explore the state space, GFlowNets have emerged as a promising approach for sampling compositional objects, improving generalization, and enabling models to maintain diverse hypotheses. Our early results demonstrate GFlowNet fine-tuning's potential for enhancing model performance in a search setting, which is especially relevant given the paradigm shift towards inference time compute scaling and "thinking slowly."
     </details>

88. **The Geometry of Numerical Reasoning: Language Models Compare Numeric Properties in Linear Subspaces** [[pdf]](http://arxiv.org/abs/2410.13194) `2024-10-17` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Experimental results show that LLMs utilize the linearly encoded information for numerical reasoning for numerical reasoning, indicating that LLMs utilize the linearly encoded information for numerical reasoning.
     </details>


     <details>
          <summary>Abstract</summary>
          This paper investigates whether large language models (LLMs) utilize numerical attributes encoded in a low-dimensional subspace of the embedding space when answering logical comparison questions (e.g., Was Cristiano born before Messi?). We first identified these subspaces using partial least squares regression, which effectively encodes the numerical attributes associated with the entities in comparison prompts. Further, we demonstrate causality by intervening in these subspaces to manipulate hidden states, thereby altering the LLM's comparison outcomes. Experimental results show that our findings hold for different numerical attributes, indicating that LLMs utilize the linearly encoded information for numerical reasoning.
     </details>

89. **Enhancing Mathematical Reasoning in LLMs by Stepwise Correction** [[pdf]](http://arxiv.org/abs/2410.12934) `2024-10-16` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel prompting method named Stepwise Correction (StepCo) is proposed that helps LLMs identify and revise incorrect steps in their generated reasoning paths that iterates verification and revision phases that employ a process-supervised verifier.
     </details>


     <details>
          <summary>Abstract</summary>
          Best-of-N decoding methods instruct large language models (LLMs) to generate multiple solutions, score each using a scoring function, and select the highest scored as the final answer to mathematical reasoning problems. However, this repeated independent process often leads to the same mistakes, making the selected solution still incorrect. We propose a novel prompting method named Stepwise Correction (StepCo) that helps LLMs identify and revise incorrect steps in their generated reasoning paths. It iterates verification and revision phases that employ a process-supervised verifier. The verify-then-revise process not only improves answer correctness but also reduces token consumption with fewer paths needed to generate. With StepCo, a series of LLMs demonstrate exceptional performance. Notably, using GPT-4o as the backend LLM, StepCo achieves an average accuracy of 94.1 across eight datasets, significantly outperforming the state-of-the-art Best-of-N method by +2.4, while reducing token consumption by 77.8%.
     </details>

90. **Not All Votes Count! Programs as Verifiers Improve Self-Consistency of Language Models for Math Reasoning** [[pdf]](http://arxiv.org/abs/2410.12608) `2024-10-16` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes PROVE, a simple yet effective framework that uses program-based verification as a heuristic to filter out potentially incorrect reasoning paths before aggregating the final answers, instead of relying on vanilla majority voting.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have shown increasing proficiency in solving mathematical reasoning problems. However, many current open-source LLMs often still make calculation and semantic understanding errors in their intermediate reasoning steps. In this work, we propose PROVE, a simple yet effective framework that uses program-based verification as a heuristic to filter out potentially incorrect reasoning paths before aggregating the final answers. Instead of relying on vanilla majority voting, our approach rejects solutions whose corresponding program outputs are inconsistent with the generated solution, aggregating only those validated by Python programs. We conducted extensive experiments on 13 open-source LLMs from various model families and sizes, ranging from 0.5B to 13B parameters, across seven math benchmarks. We demonstrate that PROVE consistently outperforms vanilla majority voting as a heuristic for solving mathematical reasoning tasks across all datasets and model sizes. Notably, PROVE increases accuracy on the GSM8K benchmark from 48.85% to 53.83% for Qwen2-0.5B-Instruct, from 65.66% to 73.01% for Llama-3.2-1B-Instruct, from 73.39% to 79.61% for Gemma-2-2b-it, and from 41.32% to 59.51% for Llama-2-7B-chat. Our codes are available at https://github.com/declare-lab/prove.
     </details>

91. **Reversal of Thought: Enhancing Large Language Models with Preference-Guided Reverse Reasoning Warm-up** [[pdf]](http://arxiv.org/abs/2410.12323) `2024-10-16` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Through reverse reasoning, the proposed Reversal of Thought (RoT) ultilize a Cognitive Preference Manager to assess knowledge boundaries and further expand LLMs' reasoning capabilities by aggregating solution logic for known tasks and stylistic templates for unknown tasks.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have shown remarkable performance in reasoning tasks but face limitations in mathematical and complex logical reasoning. Existing methods to improve LLMs' logical capabilities either involve traceable or verifiable logical sequences that generate more reliable responses by constructing logical structures yet increase computational costs, or introduces rigid logic template rules, reducing flexibility. In this paper, we propose Reversal of Thought (RoT), a novel framework aimed at enhancing the logical reasoning abilities of LLMs. RoT utilizes a Preference-Guided Reverse Reasoning warm-up strategy, which integrates logical symbols for pseudocode planning through meta-cognitive mechanisms and pairwise preference self-evaluation to generate task-specific prompts solely through demonstrations, aligning with LLMs' cognitive preferences shaped by Reinforcement Learning with Human Feedback (RLHF). Through reverse reasoning, we ultilize a Cognitive Preference Manager to assess knowledge boundaries and further expand LLMs' reasoning capabilities by aggregating solution logic for known tasks and stylistic templates for unknown tasks. Experiments across various tasks demonstrate that RoT surpasses existing baselines in both reasoning accuracy and efficiency.
     </details>

92. **When Not to Answer: Evaluating Prompts on GPT Models for Effective Abstention in Unanswerable Math Word Problems** [[pdf]](http://arxiv.org/abs/2410.13029) `2024-10-16` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper investigates whether GPTs can appropriately respond to unanswerable math word problems by applying prompts typically used in solvable mathematical scenarios, and reveals critical gaps in GPT models and the hallucination it suffers from for unsolvable problems.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) are increasingly relied upon to solve complex mathematical word problems. However, being susceptible to hallucination, they may generate inaccurate results when presented with unanswerable questions, raising concerns about their potential harm. While GPT models are now widely used and trusted, the exploration of how they can effectively abstain from answering unanswerable math problems and the enhancement of their abstention capabilities has not been rigorously investigated. In this paper, we investigate whether GPTs can appropriately respond to unanswerable math word problems by applying prompts typically used in solvable mathematical scenarios. Our experiments utilize the Unanswerable Word Math Problem (UWMP) dataset, directly leveraging GPT model APIs. Evaluation metrics are introduced, which integrate three key factors: abstention, correctness and confidence. Our findings reveal critical gaps in GPT models and the hallucination it suffers from for unsolvable problems, highlighting the need for improved models capable of better managing uncertainty and complex reasoning in math word problem-solving contexts.
     </details>

93. **Language Models Encode Numbers Using Digit Representations in Base 10** [[pdf]](http://arxiv.org/abs/2410.11781) `2024-10-15` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) frequently make errors when handling even simple numerical problems, such as comparing two small numbers. A natural hypothesis is that these errors stem from how LLMs represent numbers, and specifically, whether their representations of numbers capture their numeric values. We tackle this question from the observation that LLM errors on numerical tasks are often distributed across \textit{the digits} of the answer rather than normally around \textit{its numeric value}. Through a series of probing experiments and causal interventions, we show that LLMs internally represent numbers with individual circular representations per-digit in base 10. This digit-wise representation, as opposed to a value representation, sheds light on the error patterns of models on tasks involving numerical reasoning and could serve as a basis for future studies on analyzing numerical mechanisms in LLMs.
     </details>

94. **MIND: Math Informed syNthetic Dialogues for Pretraining LLMs** [[pdf]](http://arxiv.org/abs/2410.12881) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          The utility of synthetic data to enhance pretraining data quality and hence to improve downstream task accuracy has been widely explored in recent large language models (LLMs). Yet, these approaches fall inadequate in complex, multi-hop and mathematical reasoning tasks as the synthetic data typically fails to add complementary knowledge to the existing raw corpus. In this work, we propose a novel large-scale and diverse Math Informed syNthetic Dialogue (MIND) generation method that improves the mathematical reasoning ability of LLMs. Specifically, using MIND, we generate synthetic conversations based on OpenWebMath (OWM), resulting in a new math corpus, MIND-OWM. Our experiments with different conversational settings reveal that incorporating knowledge gaps between dialog participants is essential for generating high-quality math data. We further identify an effective way to format and integrate synthetic and raw data during pretraining to maximize the gain in mathematical reasoning, emphasizing the need to restructure raw data rather than use it as-is. Compared to pretraining just on raw data, a model pretrained on MIND-OWM shows significant boost in mathematical reasoning (GSM8K: +13.42%, MATH: +2.30%), including superior performance in specialized knowledge (MMLU: +4.55%, MMLU-STEM: +4.28%) and general purpose reasoning tasks (GENERAL REASONING: +2.51%).
     </details>

95. **How to Leverage Demonstration Data in Alignment for Large Language Model? A Self-Imitation Learning Perspective** [[pdf]](http://arxiv.org/abs/2410.10093) `2024-10-14` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel generalized self-imitation learning framework, which effectively and efficiently aligns large language models with offline demonstration data, and eliminates the need for complex adversarial training in standard imitation learning, achieving lightweight and efficient fine-tuning for large language models.
     </details>


     <details>
          <summary>Abstract</summary>
          This paper introduces a novel generalized self-imitation learning ($\textbf{GSIL}$) framework, which effectively and efficiently aligns large language models with offline demonstration data. We develop $\textbf{GSIL}$ by deriving a surrogate objective of imitation learning with density ratio estimates, facilitating the use of self-generated data and optimizing the imitation learning objective with simple classification losses. $\textbf{GSIL}$ eliminates the need for complex adversarial training in standard imitation learning, achieving lightweight and efficient fine-tuning for large language models. In addition, $\textbf{GSIL}$ encompasses a family of offline losses parameterized by a general class of convex functions for density ratio estimation and enables a unified view for alignment with demonstration data. Extensive experiments show that $\textbf{GSIL}$ consistently and significantly outperforms baselines in many challenging benchmarks, such as coding (HuamnEval), mathematical reasoning (GSM8K) and instruction-following benchmark (MT-Bench).
     </details>

96. **3D-Prover: Diversity Driven Theorem Proving With Determinantal Point Processes** [[pdf]](http://arxiv.org/abs/2410.11133) `ICLR 2025 Submission` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          A key challenge in automated formal reasoning is the intractable search space, which grows exponentially with the depth of the proof. This branching is caused by the large number of candidate proof tactics which can be applied to a given goal. Nonetheless, many of these tactics are semantically similar or lead to an execution error, wasting valuable resources in both cases. We address the problem of effectively pruning this search, using only synthetic data generated from previous proof attempts. We first demonstrate that it is possible to generate semantically aware tactic representations which capture the effect on the proving environment, likelihood of success and execution time. We then propose a novel filtering mechanism which leverages these representations to select semantically diverse and high quality tactics, using Determinantal Point Processes. Our approach, 3D-Prover, is designed to be general, and to augment any underlying tactic generator. We demonstrate the effectiveness of 3D-Prover on the miniF2F-valid and miniF2F-test benchmarks by augmenting the ReProver LLM. We show that our approach leads to an increase in the overall proof rate, as well as a significant improvement in the tactic success rate, execution time and diversity.
     </details>

97. **CoMAT: Chain of Mathematically Annotated Thought Improves Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2410.10336) `2024-10-14` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Chain of Mathematically Annotated Thought (CoMAT) is presented, which enhances reasoning through two stages: Symbolic Conversion and Reasoning Execution, offering a transparent reasoning process for complex mathematical tasks.
     </details>


     <details>
          <summary>Abstract</summary>
          Mathematical reasoning remains a significant challenge for large language models (LLMs), despite progress in prompting techniques such as Chain-of-Thought (CoT). We present Chain of Mathematically Annotated Thought (CoMAT), which enhances reasoning through two stages: Symbolic Conversion (converting natural language queries into symbolic form) and Reasoning Execution (deriving answers from symbolic representations). CoMAT operates entirely with a single LLM and without external solvers. Across four LLMs, CoMAT outperforms traditional CoT on six out of seven benchmarks, achieving gains of 4.48% on MMLU-Redux (MATH) and 4.58% on GaoKao MCQ. In addition to improved performance, CoMAT ensures faithfulness and verifiability, offering a transparent reasoning process for complex mathematical tasks
     </details>

98. **Embedding Self-Correction as an Inherent Ability in Large Language Models for Enhanced Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2410.10735) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces a novel mechanism, the Chain of Self-Correction (CoSC), specifically designed to embed self-correction as an inherent ability in LLMs, enabling them to validate and rectify their own results.
     </details>


     <details>
          <summary>Abstract</summary>
          Accurate mathematical reasoning with Large Language Models (LLMs) is crucial in revolutionizing domains that heavily rely on such reasoning. However, LLMs often encounter difficulties in certain aspects of mathematical reasoning, leading to flawed reasoning and erroneous results. To mitigate these issues, we introduce a novel mechanism, the Chain of Self-Correction (CoSC), specifically designed to embed self-correction as an inherent ability in LLMs, enabling them to validate and rectify their own results. The CoSC mechanism operates through a sequence of self-correction stages. In each stage, the LLMs generate a program to address a given problem, execute this program using program-based tools to obtain an output, subsequently verify this output. Based on the verification, the LLMs either proceed to the next correction stage or finalize the answer. This iterative self-correction process allows the LLMs to refine their reasoning steps and improve the accuracy of their mathematical reasoning. To enable the CoSC mechanism at a low cost, we employ a two-phase finetuning approach. In the first phase, the LLMs are trained with a relatively small volume of seeding data generated from GPT-4, establishing an initial CoSC capability. In the second phase, the CoSC capability is further enhanced by training with a larger volume of self-generated data using the trained model in the first phase, without relying on the paid GPT-4. Our comprehensive experiments demonstrate that CoSC significantly improves performance on traditional mathematical datasets among existing open-source LLMs. Notably, our CoSC-Code-34B model achieved a 53.5% score on MATH, the most challenging mathematical reasoning dataset in the public domain, surpassing the performance of well-established models such as ChatGPT, GPT-4, and even multi-modal LLMs like GPT-4V, Gemini-1.0 Pro, and Gemini-1.0 Ultra.
     </details>

99. **FLARE: Faithful Logic-Aided Reasoning and Exploration** [[pdf]](http://arxiv.org/abs/2410.11900) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Faithful CoT (F-CoT) is introduced, a novel interpretable approach for traversing the problem space using task decompositions and it is demonstrated that model faithfulness positively correlates with overall performance and allows pinpointing the decisive factors sufficient for and leading to the correct answer with optimal reasoning during the multi-hop search.
     </details>


     <details>
          <summary>Abstract</summary>
          Modern Question Answering (QA) and Reasoning approaches based on Large Language Models (LLMs) commonly use prompting techniques, such as Chain-of-Thought (CoT), assuming the resulting generation will have a more granular exploration and reasoning over the question space and scope. However, such methods struggle with generating outputs that are faithful to the intermediate chain of reasoning produced by the model. On the other end of the spectrum, neuro-symbolic methods such as Faithful CoT (F-CoT) propose to combine LLMs with external symbolic solvers. While such approaches boast a high degree of faithfulness, they usually require a model trained for code generation and struggle with tasks that are ambiguous or hard to formalise strictly. We introduce $\textbf{F}$aithful $\textbf{L}$ogic-$\textbf{A}$ided $\textbf{R}$easoning and $\textbf{E}$xploration ($\textbf{FLARE}$), a novel interpretable approach for traversing the problem space using task decompositions. We use the LLM to plan a solution, soft-formalise the query into facts and predicates using a logic programming code and simulate that code execution using an exhaustive multi-hop search over the defined space. Our method allows us to compute the faithfulness of the reasoning process w.r.t. the generated code and analyse the steps of the multi-hop search without relying on external solvers. Our methods achieve SOTA results on $\mathbf{7}$ out of $\mathbf{9}$ diverse reasoning benchmarks. We also show that model faithfulness positively correlates with overall performance and further demonstrate that $\textbf{FLARE}$ allows pinpointing the decisive factors sufficient for and leading to the correct answer with optimal reasoning during the multi-hop search.
     </details>

100. **Dualformer: Controllable Fast and Slow Thinking by Learning with Randomized Reasoning Traces** [[pdf]](http://arxiv.org/abs/2410.09918) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Dualformer is a single Transformer model that seamlessly integrates both the fast and slow reasoning modes, and achieves improved performance with LLM fine-tuning, showing its generalization beyond task-specific models.
     </details>


     <details>
          <summary>Abstract</summary>
          In human cognition theory, human thinking is governed by two systems: the fast and intuitive System 1 and the slower but more deliberative System 2. Recent studies have shown that incorporating System 2 process into Transformers including large language models (LLMs), significantly enhances their reasoning capabilities. Nevertheless, models that purely resemble System 2 thinking require substantially higher computational costs and are much slower to respond. To address this challenge, we present Dualformer, a single Transformer model that seamlessly integrates both the fast and slow reasoning modes. Dualformer is obtained by training on data with randomized reasoning traces, where different parts of the traces are dropped during training. The dropping strategies are specifically tailored according to the trace structure, analogous to analyzing our thinking process and creating shortcuts with patterns. At inference time, our model can be configured to output only the solutions (fast mode) or both the reasoning chain and the final solution (slow mode), or automatically decide which mode to engage (auto mode). In all cases, Dualformer outperforms the corresponding baseline models in both performance and computational efficiency: (1) in slow mode, Dualformer optimally solves unseen 30 x 30 maze navigation tasks 97.6% of the time, surpassing the Searchformer (trained on data with complete reasoning traces) baseline performance of 93.3%, while only using 45.5% fewer reasoning steps; (2) in fast mode, Dualformer completes those tasks with an 80% optimal rate, significantly outperforming the Solution-Only model (trained on solution-only data), which has an optimal rate of only 30%. For math problems, our techniques have also achieved improved performance with LLM fine-tuning, showing its generalization beyond task-specific models.
     </details>

101. **Expanding Search Space with Diverse Prompting Agents: An Efficient Sampling Approach for LLM Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2410.09780) `2024-10-13` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This study performs an experimental analysis of distinct prompting methods within the domain of mathematical reasoning to demonstrate that each method explores a distinct search space, and this differentiation becomes more evident with increasing problem complexity.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have exhibited remarkable capabilities in many complex tasks including mathematical reasoning. However, traditional approaches heavily rely on ensuring self-consistency within single prompting method, which limits the exploration of diverse problem-solving strategies. This study addresses these limitations by performing an experimental analysis of distinct prompting methods within the domain of mathematical reasoning. Our findings demonstrate that each method explores a distinct search space, and this differentiation becomes more evident with increasing problem complexity. To leverage this phenomenon, we applied efficient sampling process that uniformly combines samples from these diverse methods, which not only expands the maximum search space but achieves higher performance with fewer runs compared to single methods. Especially, within the subset of difficult questions of MATH dataset named MATH-hard, The maximum search space was achieved while utilizing approximately 43% fewer runs than single methods on average. These findings highlight the importance of integrating diverse problem-solving strategies to enhance the reasoning abilities of LLMs.
     </details>

102. **HARDMath: A Benchmark Dataset for Challenging Problems in Applied Mathematics** [[pdf]](http://arxiv.org/abs/2410.09988) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          HARDMath is introduced, a dataset inspired by a graduate course on asymptotic methods, featuring challenging applied mathematics problems that require analytical approximation techniques, and auto-generates a large number of problems with solutions validated against numerical ground truths.
     </details>


     <details>
          <summary>Abstract</summary>
          Advanced applied mathematics problems are underrepresented in existing Large Language Model (LLM) benchmark datasets. To address this, we introduce HARDMath, a dataset inspired by a graduate course on asymptotic methods, featuring challenging applied mathematics problems that require analytical approximation techniques. These problems demand a combination of mathematical reasoning, computational tools, and subjective judgment, making them difficult for LLMs. Our framework auto-generates a large number of problems with solutions validated against numerical ground truths. We evaluate both open- and closed-source LLMs on HARDMath-mini, a sub-sampled test set of 366 problems, as well as on 40 word problems formulated in applied science contexts. Even leading closed-source models like GPT-4 achieve only 43.8% overall accuracy with few-shot Chain-of-Thought prompting, and all models demonstrate significantly lower performance compared to results on existing mathematics benchmark datasets. We additionally conduct a detailed error analysis to gain insights into the failure cases of LLMs. These results demonstrate limitations of current LLM performance on advanced graduate-level applied math problems and underscore the importance of datasets like HARDMath to advance mathematical abilities of LLMs.
     </details>

103. **MIRAGE: Evaluating and Explaining Inductive Reasoning Process in Language Models** [[pdf]](http://arxiv.org/abs/2410.09542) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work evaluates LLMs' capabilities in both the inductive and deductive stages, allowing for flexible variation in input distribution, task scenario, and task difficulty to analyze the factors influencing LLMs' inductive reasoning.
     </details>


     <details>
          <summary>Abstract</summary>
          Inductive reasoning is an essential capability for large language models (LLMs) to achieve higher intelligence, which requires the model to generalize rules from observed facts and then apply them to unseen examples. We present {\scshape Mirage}, a synthetic dataset that addresses the limitations of previous work, specifically the lack of comprehensive evaluation and flexible test data. In it, we evaluate LLMs' capabilities in both the inductive and deductive stages, allowing for flexible variation in input distribution, task scenario, and task difficulty to analyze the factors influencing LLMs' inductive reasoning. Based on these multi-faceted evaluations, we demonstrate that the LLM is a poor rule-based reasoner. In many cases, when conducting inductive reasoning, they do not rely on a correct rule to answer the unseen case. From the perspectives of different prompting methods, observation numbers, and task forms, models tend to consistently conduct correct deduction without correct inductive rules. Besides, we find that LLMs are good neighbor-based reasoners. In the inductive reasoning process, the model tends to focus on observed facts that are close to the current test example in feature space. By leveraging these similar examples, the model maintains strong inductive capabilities within a localized region, significantly improving its deductive performance.
     </details>

104. **OpenR: An Open Source Framework for Advanced Reasoning with Large Language Models** [[pdf]](http://arxiv.org/abs/2410.09671) `2024-10-12` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work is the first to provide an open-source framework that explores the core techniques of OpenAI's o1 model with reinforcement learning, achieving advanced reasoning capabilities beyond traditional autoregressive methods.
     </details>


     <details>
          <summary>Abstract</summary>
          In this technical report, we introduce OpenR, an open-source framework designed to integrate key components for enhancing the reasoning capabilities of large language models (LLMs). OpenR unifies data acquisition, reinforcement learning training (both online and offline), and non-autoregressive decoding into a cohesive software platform. Our goal is to establish an open-source platform and community to accelerate the development of LLM reasoning. Inspired by the success of OpenAI's o1 model, which demonstrated improved reasoning abilities through step-by-step reasoning and reinforcement learning, OpenR integrates test-time compute, reinforcement learning, and process supervision to improve reasoning in LLMs. Our work is the first to provide an open-source framework that explores the core techniques of OpenAI's o1 model with reinforcement learning, achieving advanced reasoning capabilities beyond traditional autoregressive methods. We demonstrate the efficacy of OpenR by evaluating it on the MATH dataset, utilising publicly available data and search methods. Our initial experiments confirm substantial gains, with relative improvements in reasoning and performance driven by test-time computation and reinforcement learning through process reward models. The OpenR framework, including code, models, and datasets, is accessible at https://openreasoner.github.io.
     </details>

105. **Enhancing Multi-Step Reasoning Abilities of Language Models through Direct Q-Function Optimization** [[pdf]](http://arxiv.org/abs/2410.09302) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Direct Q-function Optimization (DQO), which formulates the response generation process as a Markov Decision Process (MDP) and utilizes the soft actor-critic (SAC) framework to optimize a Q-function directly parameterized by the language model.
     </details>


     <details>
          <summary>Abstract</summary>
          Reinforcement Learning (RL) plays a crucial role in aligning large language models (LLMs) with human preferences and improving their ability to perform complex tasks. However, current approaches either require significant computational resources due to the use of multiple models and extensive online sampling for training (e.g., PPO) or are framed as bandit problems (e.g., DPO, DRO), which often struggle with multi-step reasoning tasks, such as math problem-solving and complex reasoning that involve long chains of thought. To overcome these limitations, we introduce Direct Q-function Optimization (DQO), which formulates the response generation process as a Markov Decision Process (MDP) and utilizes the soft actor-critic (SAC) framework to optimize a Q-function directly parameterized by the language model. The MDP formulation of DQO offers structural advantages over bandit-based methods, enabling more effective process supervision. Experimental results on two math problem-solving datasets, GSM8K and MATH, demonstrate that DQO outperforms previous methods, establishing it as a promising offline reinforcement learning approach for aligning language models.
     </details>

106. **Exploring the Role of Reasoning Structures for Constructing Proofs in Multi-Step Natural Language Reasoning with Large Language Models** [[pdf]](http://arxiv.org/abs/2410.08436) `2024-10-11` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This study focuses on structure-aware demonstration and structure-aware pruning in generalist Large Language Models, and demonstrates that they both help improve performance.
     </details>


     <details>
          <summary>Abstract</summary>
          When performing complex multi-step reasoning tasks, the ability of Large Language Models (LLMs) to derive structured intermediate proof steps is important for ensuring that the models truly perform the desired reasoning and for improving models' explainability. This paper is centred around a focused study: whether the current state-of-the-art generalist LLMs can leverage the structures in a few examples to better construct the proof structures with \textit{in-context learning}. Our study specifically focuses on structure-aware demonstration and structure-aware pruning. We demonstrate that they both help improve performance. A detailed analysis is provided to help understand the results.
     </details>

107. **SuperCorrect: Supervising and Correcting Language Models with Error-Driven Insights** [[pdf]](http://arxiv.org/abs/2410.09008) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes SuperCorrect, a novel two-stage framework that uses a large teacher model to supervise and correct both the reasoning and reflection processes of a smaller student model and introduces cross-model collaborative direct preference optimization (DPO) to enhance the self-correction abilities of the student model by following the teacher's correction traces during training.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) like GPT-4, PaLM, and LLaMA have shown significant improvements in various reasoning tasks. However, smaller models such as Llama-3-8B and DeepSeekMath-Base still struggle with complex mathematical reasoning because they fail to effectively identify and correct reasoning errors. Recent reflection-based methods aim to address these issues by enabling self-reflection and self-correction, but they still face challenges in independently detecting errors in their reasoning steps. To overcome these limitations, we propose SuperCorrect, a novel two-stage framework that uses a large teacher model to supervise and correct both the reasoning and reflection processes of a smaller student model. In the first stage, we extract hierarchical high-level and detailed thought templates from the teacher model to guide the student model in eliciting more fine-grained reasoning thoughts. In the second stage, we introduce cross-model collaborative direct preference optimization (DPO) to enhance the self-correction abilities of the student model by following the teacher's correction traces during training. This cross-model DPO approach teaches the student model to effectively locate and resolve erroneous thoughts with error-driven insights from the teacher model, breaking the bottleneck of its thoughts and acquiring new skills and knowledge to tackle challenging problems. Extensive experiments consistently demonstrate our superiority over previous methods. Notably, our SuperCorrect-7B model significantly surpasses powerful DeepSeekMath-7B by 7.8%/5.3% and Qwen2.5-Math-7B by 15.1%/6.3% on MATH/GSM8K benchmarks, achieving new SOTA performance among all 7B models. Code: https://github.com/YangLing0818/SuperCorrect-llm
     </details>

108. **Testing GPT-4-o1-preview on math and science problems: A follow-up study** [[pdf]](http://arxiv.org/abs/2410.22340) `2024-10-11` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Overall I found that performance had significantly improved, but was still considerably short of perfect, in particular, problems that involve spatial reasoning are often stumbling blocks.
     </details>


     <details>
          <summary>Abstract</summary>
          In August 2023, Scott Aaronson and I reported the results of testing GPT4 with the Wolfram Alpha and Code Interpreter plug-ins over a collection of 105 original high-school level and college-level science and math problems (Davis and Aaronson, 2023). In September 2024, I tested the recently released model GPT-4o1-preview on the same collection. Overall I found that performance had significantly improved, but was still considerably short of perfect. In particular, problems that involve spatial reasoning are often stumbling blocks.
     </details>

109. **Automatic Curriculum Expert Iteration for Reliable LLM Reasoning** [[pdf]](http://arxiv.org/abs/2410.07627) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Automatic Curriculum Expert Iteration is proposed to enhance LLM reasoning and align responses to the model's capabilities--assertively answering within its limits and declining when tasks exceed them--and SOTA baselines are compared, where Auto-CEI achieves superior alignment by effectively balancing assertiveness and conservativeness.
     </details>


     <details>
          <summary>Abstract</summary>
          Hallucinations (i.e., generating plausible but inaccurate content) and laziness (i.e. excessive refusals or defaulting to "I don't know") persist as major challenges in LLM reasoning. Current efforts to reduce hallucinations primarily focus on factual errors in knowledge-grounded tasks, often neglecting hallucinations related to faulty reasoning. Meanwhile, some approaches render LLMs overly conservative, limiting their problem-solving capabilities. To mitigate hallucination and laziness in reasoning tasks, we propose Automatic Curriculum Expert Iteration (Auto-CEI) to enhance LLM reasoning and align responses to the model's capabilities--assertively answering within its limits and declining when tasks exceed them. In our method, Expert Iteration explores the reasoning trajectories near the LLM policy, guiding incorrect paths back on track to reduce compounding errors and improve robustness; it also promotes appropriate "I don't know" responses after sufficient reasoning attempts. The curriculum automatically adjusts rewards, incentivizing extended reasoning before acknowledging incapability, thereby pushing the limits of LLM reasoning and aligning its behaviour with these limits. We compare Auto-CEI with various SOTA baselines across logical reasoning, mathematics, and planning tasks, where Auto-CEI achieves superior alignment by effectively balancing assertiveness and conservativeness.
     </details>

110. **Diversity of Thought Elicits Stronger Reasoning Capabilities in Multi-Agent Debate Frameworks** [[pdf]](http://arxiv.org/abs/2410.12853) `2024-10-10` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is found that multi-agent debate helps at any model scale, and that diversity of thought elicits stronger reasoning in debating LLMs, underscoring the idea that the future of AI is agentic, with diverse cooperating agents yielding emergent capabilities beyond even the most powerful individual models.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) excel in natural language generation but often confidently produce incorrect responses, especially in tasks like mathematical reasoning. Chain-of-thought prompting, self-verification, and multi-agent debate are among the strategies proposed to improve the reasoning and factual accuracy of LLMs. Building on Du et al.'s multi-agent debate framework, we find that multi-agent debate helps at any model scale, and that diversity of thought elicits stronger reasoning in debating LLMs. Across various model sizes, performance on mathematical reasoning tasks benefits most when diverse trained models are used. Remarkably, after 4 rounds of debate, a diverse set of medium-capacity models (Gemini-Pro, Mixtral 7BX8, and PaLM 2-M) outperforms GPT-4 on the GSM-8K benchmark, scoring 91% accuracy. By comparison, when 3 instances of Gemini-Pro are used, performance only reaches 82%. Finally, this diverse set of medium-capacity models sets a new state-of-the-art performance on the ASDiv benchmark (94%). These results underscore the idea that the future of AI is agentic, with diverse cooperating agents yielding emergent capabilities beyond even the most powerful individual models.
     </details>

111. **Enhancing Language Model Reasoning via Weighted Reasoning in Self-Consistency** [[pdf]](http://arxiv.org/abs/2410.07839) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work enhances Wang et al's self-consistency framework by incorporating and analyzing both the reasoning paths of these rationales in addition to their final decisions before taking a majority vote, which not only improves the reliability of reasoning paths but also cause more robust performance on complex reasoning tasks.
     </details>


     <details>
          <summary>Abstract</summary>
          While large language models (LLMs) have rapidly improved their performance on a broad number of tasks, they still often fall short on reasoning tasks. As LLMs become more integrated in diverse real-world tasks, advancing their reasoning capabilities is crucial to their effectiveness in nuanced, complex problems. Wang et al's self-consistency framework reveals that sampling multiple rationales before taking a majority vote reliably improves model performance across various closed-answer reasoning tasks. Standard methods based on this framework aggregate the final decisions of these rationales but fail to utilize the detailed step-by-step reasoning paths applied by these paths. Our work enhances this approach by incorporating and analyzing both the reasoning paths of these rationales in addition to their final decisions before taking a majority vote. These methods not only improve the reliability of reasoning paths but also cause more robust performance on complex reasoning tasks.
     </details>

112. **Executing Arithmetic: Fine-Tuning Large Language Models as Turing Machines** [[pdf]](http://arxiv.org/abs/2410.07896) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A Composable Arithmetic Execution Framework (CAEF) is proposed that enables LLMs to learn to execute step-by-step computations by emulating Turing Machines, thereby gaining a genuine understanding of computational logic.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have demonstrated remarkable capabilities across a wide range of natural language processing and reasoning tasks. However, their performance in the foundational domain of arithmetic remains unsatisfactory. When dealing with arithmetic tasks, LLMs often memorize specific examples rather than learning the underlying computational logic, limiting their ability to generalize to new problems. In this paper, we propose a Composable Arithmetic Execution Framework (CAEF) that enables LLMs to learn to execute step-by-step computations by emulating Turing Machines, thereby gaining a genuine understanding of computational logic. Moreover, the proposed framework is highly scalable, allowing composing learned operators to significantly reduce the difficulty of learning complex operators. In our evaluation, CAEF achieves nearly 100% accuracy across seven common mathematical operations on the LLaMA 3.1-8B model, effectively supporting computations involving operands with up to 100 digits, a level where GPT-4o falls short noticeably in some settings.
     </details>

113. **Flow-DPO: Improving LLM Mathematical Reasoning through Online Multi-Agent Learning** [[pdf]](https://openreview.net/forum?id=uwagVHmyNA) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Mathematical reasoning is a crucial capability for Large Language Models (LLMs), yet generating detailed and accurate reasoning traces remains a significant challenge. This paper introduces a novel approach to produce high-quality reasoning traces for LLM fine-tuning using online learning \textbf{Flows}. Our method employs an incremental output production Flow, where component LLMs collaboratively construct solutions through iterative communication. We train the Flow using online Direct Preference Optimization (DPO) learning with rollouts, generating DPO pairs for each training example and updating models in real-time. We directly compare the quality of reasoning traces generated by our method with those produced through direct model inference, demonstrating the effectiveness of our approach in improving LLM performance in mathematical reasoning tasks.
     </details>

114. **Interleaving Text and Number Embeddings to Solve Mathemathics Problems** [[pdf]](https://openreview.net/forum?id=8cNJyqs45T) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Integrating text and numbers effectively is a crucial step towards enhancing Large Language Models (LLMs) capabilities in assisting in scientific tasks. While most current approaches rely on discrete tokenization of numbers, for instance, conversion to scientific notation or base 10-decomposition, a recent approach proposed a continuous numerical encoding as an inductive bias. In this paper, we build upon this approach by introducing more expressive numerical embeddings. Our method addresses key shortcomings, including the elimination of numerical artefacts and the ability to handle a wide range of magnitudes without clipping.   Our work presents two key contributions. First, we employ an MLP to assign distinct directions in the embedding space to different numbers. Our second contribution is the introduction of a routing layer that differentiates between numerical and text embeddings. We hypothesise that this combined approach enables the model to distinguish between text and number distributions while maintaining its capacity for arithmetic operations.   Using only a 45 M parameter encoder-decoder architecture our method achieves a $R^2$=0.9988 over a wide range of magnitude ($10^{-3},10^{8}$). In addition, we empirically observe a reduction of the numerical artefacts and biases observed compared to the baselines.
     </details>

115. **Learning Mathematical Rules with Large Language Models** [[pdf]](https://openreview.net/forum?id=tIlDF5B6T4) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          In this paper, we study the ability of large language models to learn specific mathematical rules such as distributivity or simplifying equations. We present an empirical analysis of their ability to generalize these rules, as well as to reuse them in the context of word problems. For this purpose, we provide a rigorous methodology to build synthetic data incorporating such rules, and perform fine-tuning of large language models on such data. Our experiments show that our model can learn and generalize these rules to some extent, as well as suitably reuse them in the context of word problems.
     </details>

116. **Machines and Mathematical Mutations: Using GNNs to Characterize Quiver Mutation Classes** [[pdf]](https://openreview.net/forum?id=SxeiWOp73E) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Machine learning is becoming an increasingly valuable tool in mathematics, enabling one to identify subtle patterns across collections of examples so vast that they would be impossible for a single researcher to feasibly review and analyze. In this work, we use graph neural networks to investigate quiver mutation -- an operation that transforms one quiver (or directed multigraph) into another -- which is central to the theory of cluster algebras with deep connections to geometry, topology, and physics. In the study of cluster algebras, the question of mutation equivalence is of fundamental concern: given two quivers, can one efficiently determine if one quiver can be transformed into the other through a sequence of mutations? Currently, this question has only been resolved in specific cases. In this paper, we use graph neural networks and AI explainability techniques to discover mutation equivalence criteria for the previously unknown case of quivers of type $\tilde{D}_n$. Along the way, we also show that even without explicit training to do so, our model captures structure within its hidden representation that allows us to reconstruct known criteria from type $D_n$, adding to the growing evidence that modern machine learning models are capable of learning abstract and general rules from mathematical data.
     </details>

117. **MathCoder2: Better Math Reasoning from Continued Pretraining on Model-translated Mathematical Code** [[pdf]](http://arxiv.org/abs/2410.08196) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel method for generating mathematical code accompanied with corresponding reasoning steps for continued pretraining, leading to a 19.2B-token high-performing mathematical pretraining corpus, which is training several popular base models with this corpus significantly improves their mathematical abilities, leading to the creation of the MathCoder2 family of models.
     </details>


     <details>
          <summary>Abstract</summary>
          Code has been shown to be effective in enhancing the mathematical reasoning abilities of large language models due to its precision and accuracy. Previous works involving continued mathematical pretraining often include code that utilizes math-related packages, which are primarily designed for fields such as engineering, machine learning, signal processing, or module testing, rather than being directly focused on mathematical reasoning. In this paper, we introduce a novel method for generating mathematical code accompanied with corresponding reasoning steps for continued pretraining. Our approach begins with the construction of a high-quality mathematical continued pretraining dataset by incorporating math-related web data, code using mathematical packages, math textbooks, and synthetic data. Next, we construct reasoning steps by extracting LaTeX expressions, the conditions needed for the expressions, and the results of the expressions from the previously collected dataset. Based on this extracted information, we generate corresponding code to accurately capture the mathematical reasoning process. Appending the generated code to each reasoning step results in data consisting of paired natural language reasoning steps and their corresponding code. Combining this data with the original dataset results in a 19.2B-token high-performing mathematical pretraining corpus, which we name MathCode-Pile. Training several popular base models with this corpus significantly improves their mathematical abilities, leading to the creation of the MathCoder2 family of models. All of our data processing and training code is open-sourced, ensuring full transparency and easy reproducibility of the entire data collection and training pipeline. The code is released at https://github.com/mathllm/MathCoder2 .
     </details>

118. **Omni-MATH: A Universal Olympiad Level Mathematic Benchmark For Large Language Models** [[pdf]](http://arxiv.org/abs/2410.07985) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a comprehensive and challenging benchmark specifically designed to assess LLMs' mathematical reasoning at the Olympiad level, and shows that even the most advanced models, OpenAI o1-mini and OpenAI o1-preview, struggle with highly challenging Olympiad-level problems.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent advancements in large language models (LLMs) have led to significant breakthroughs in mathematical reasoning capabilities. However, existing benchmarks like GSM8K or MATH are now being solved with high accuracy (e.g., OpenAI o1 achieves 94.8% on MATH dataset), indicating their inadequacy for truly challenging these models. To bridge this gap, we propose a comprehensive and challenging benchmark specifically designed to assess LLMs' mathematical reasoning at the Olympiad level. Unlike existing Olympiad-related benchmarks, our dataset focuses exclusively on mathematics and comprises a vast collection of 4428 competition-level problems with rigorous human annotation. These problems are meticulously categorized into over 33 sub-domains and span more than 10 distinct difficulty levels, enabling a holistic assessment of model performance in Olympiad-mathematical reasoning. Furthermore, we conducted an in-depth analysis based on this benchmark. Our experimental results show that even the most advanced models, OpenAI o1-mini and OpenAI o1-preview, struggle with highly challenging Olympiad-level problems, with 60.54% and 52.55% accuracy, highlighting significant challenges in Olympiad-level mathematical reasoning.
     </details>

119. **Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning** [[pdf]](http://arxiv.org/abs/2410.08146) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work describes the set of good provers and shows that optimizing process rewards from such provers improves exploration during test-time search and online RL and shows that weak prover policies can substantially improve a stronger base policy.
     </details>


     <details>
          <summary>Abstract</summary>
          A promising approach for improving reasoning in large language models is to use process reward models (PRMs). PRMs provide feedback at each step of a multi-step reasoning trace, potentially improving credit assignment over outcome reward models (ORMs) that only provide feedback at the final step. However, collecting dense, per-step human labels is not scalable, and training PRMs from automatically-labeled data has thus far led to limited gains. To improve a base policy by running search against a PRM or using it as dense rewards for reinforcement learning (RL), we ask: "How should we design process rewards?". Our key insight is that, to be effective, the process reward for a step should measure progress: a change in the likelihood of producing a correct response in the future, before and after taking the step, corresponding to the notion of step-level advantages in RL. Crucially, this progress should be measured under a prover policy distinct from the base policy. We theoretically characterize the set of good provers and our results show that optimizing process rewards from such provers improves exploration during test-time search and online RL. In fact, our characterization shows that weak prover policies can substantially improve a stronger base policy, which we also observe empirically. We validate our claims by training process advantage verifiers (PAVs) to predict progress under such provers, and show that compared to ORMs, test-time search against PAVs is $>8\%$ more accurate, and $1.5-5\times$ more compute-efficient. Online RL with dense rewards from PAVs enables one of the first results with $5-6\times$ gain in sample efficiency, and $>6\%$ gain in accuracy, over ORMs.
     </details>

120. **STEM-POM: Evaluating Language Models Math-Symbol Reasoning in Document Parsing** [[pdf]](https://openreview.net/forum?id=jErJ8kansp) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduces STEM-PoM, a comprehensive benchmark dataset designed to evaluate LLMs' reasoning abilities on math symbols within contextual scientific text and shows that state-of-the-art LLMs achieve an average of 20-60% accuracy under in-context learning and 50-60% accuracy with fine-tuning, revealing a significant gap in their mathematical reasoning capabilities.
     </details>


     <details>
          <summary>Abstract</summary>
          Advances in large language models (LLMs) have spurred research into enhancing their reasoning capabilities, particularly in math-rich STEM documents. While LLMs can generate equations or solve math-related queries, their ability to fully understand and interpret abstract mathematical symbols in long, math-rich documents remains limited. In this paper, we introduce STEM-PoM, a comprehensive benchmark dataset designed to evaluate LLMs' reasoning abilities on math symbols within contextual scientific text. The dataset, sourced from real-world ArXiv documents, contains over 2K math symbols classified as main attributes of variables, constants, operators, and unit descriptors, with additional sub-attributes including scalar/vector/matrix for variables and local/global/discipline-specific labels for both constants and operators. Our extensive experiments show that state-of-the-art LLMs achieve an average of 20-60% accuracy under in-context learning and 50-60% accuracy with fine-tuning, revealing a significant gap in their mathematical reasoning capabilities. STEM-PoM fuels future research of developing advanced Math-AI models that can robustly handle math symbols.
     </details>

121. **TPO: Aligning Large Language Models with Multi-branch & Multi-step Preference Trees** [[pdf]](http://arxiv.org/abs/2410.12854) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          In the domain of complex reasoning tasks, such as mathematical reasoning, recent advancements have proposed the use of Direct Preference Optimization (DPO) to suppress output of dispreferred responses, thereby enhancing the long-chain reasoning capabilities of large language models (LLMs). To this end, these studies employed LLMs to generate preference trees via Tree-of-thoughts (ToT) and sample the paired preference responses required by the DPO algorithm. However, the DPO algorithm based on binary preference optimization is unable to learn multiple responses with varying degrees of preference/dispreference that provided by the preference trees, resulting in incomplete preference learning. In this work, we introduce Tree Preference Optimization (TPO), that does not sample paired preference responses from the preference tree; instead, it directly learns from the entire preference tree during the fine-tuning. Specifically, TPO formulates the language model alignment as a Preference List Ranking problem, where the policy can potentially learn more effectively from a ranked preference list of responses given the prompt. In addition, to further assist LLMs in identifying discriminative steps within long-chain reasoning and increase the relative reward margin in the preference list, TPO utilizes Adaptive Step Reward to adjust the reward values of each step in trajectory for performing fine-grained preference optimization. We carry out extensive experiments on mathematical reasoning tasks to evaluate TPO. The experimental results indicate that TPO consistently outperforms DPO across three public large language models on four datasets.
     </details>

122. **Teaching-Inspired Integrated Prompting Framework: A Novel Approach for Enhancing Reasoning in Large Language Models** [[pdf]](http://arxiv.org/abs/2410.08068) `2024-10-10` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel and effective Teaching-Inspired Integrated Framework, which emulates the instructional process of a teacher guiding students, and equips LLMs with essential concepts, relevant theorems, and similar problems with analogous solution approaches, facilitating the enhancement of reasoning abilities.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) exhibit impressive performance across various domains but still struggle with arithmetic reasoning tasks. Recent work shows the effectiveness of prompt design methods in enhancing reasoning capabilities. However, these approaches overlook crucial requirements for prior knowledge of specific concepts, theorems, and tricks to tackle most arithmetic reasoning problems successfully. To address this issue, we propose a novel and effective Teaching-Inspired Integrated Framework, which emulates the instructional process of a teacher guiding students. This method equips LLMs with essential concepts, relevant theorems, and similar problems with analogous solution approaches, facilitating the enhancement of reasoning abilities. Additionally, we introduce two new Chinese datasets, MathMC and MathToF, both with detailed explanations and answers. Experiments are conducted on nine benchmarks which demonstrates that our approach improves the reasoning accuracy of LLMs. With GPT-4 and our framework, we achieve new state-of-the-art performance on four math benchmarks (AddSub, SVAMP, Math23K and AQuA) with accuracies of 98.2% (+3.3%), 93.9% (+0.2%), 94.3% (+7.2%) and 81.1% (+1.2%). Our data and code are available at https://github.com/SallyTan13/Teaching-Inspired-Prompting.
     </details>

123. **What Makes Large Language Models Reason in (Multi-Turn) Code Generation?** [[pdf]](http://arxiv.org/abs/2410.08105) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This study investigates the effects of a wide range of prompting strategies with a focus on automatic re-prompting over multiple turns and computational requirements, and reveals strategies that consistently improve performance across all models with small and large sampling budgets.
     </details>


     <details>
          <summary>Abstract</summary>
          Prompting techniques such as chain-of-thought have established themselves as a popular vehicle for improving the outputs of large language models (LLMs). For code generation, however, their exact mechanics and efficacy are under-explored. We thus investigate the effects of a wide range of prompting strategies with a focus on automatic re-prompting over multiple turns and computational requirements. After systematically decomposing reasoning, instruction, and execution feedback prompts, we conduct an extensive grid search on the competitive programming benchmarks CodeContests and TACO for multiple LLM families and sizes (Llama 3.0 and 3.1, 8B, 70B, 405B, and GPT-4o). Our study reveals strategies that consistently improve performance across all models with small and large sampling budgets. We then show how finetuning with such an optimal configuration allows models to internalize the induced reasoning process and obtain improvements in performance and scalability for multi-turn code generation.
     </details>

124. **Emergent properties with repeated examples** [[pdf]](http://arxiv.org/abs/2410.07041) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is demonstrated that for a fixed number of training steps, models trained on smaller sets of repeated examples outperform models trained on larger sets of single-use examples and that two-set training provides for faster learning and better performance.
     </details>


     <details>
          <summary>Abstract</summary>
          We study the performance of transformers as a function of the number of repetitions of training examples with algorithmically generated datasets. On three problems of mathematics: the greatest common divisor, modular multiplication, and matrix eigenvalues, we show that for a fixed number of training steps, models trained on smaller sets of repeated examples outperform models trained on larger sets of single-use examples. We also demonstrate that two-set training - repeated use of a small random subset of examples, along normal sampling on the rest of the training set - provides for faster learning and better performance. This highlights that the benefits of repetition can outweigh those of data diversity. These datasets and problems provide a controlled setting to shed light on the still poorly understood interplay between generalization and memorization in deep learning.
     </details>

125. **Subtle Errors Matter: Preference Learning via Error-injected Self-editing** [[pdf]](http://arxiv.org/abs/2410.06638) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel preference learning framework called RISE, which injects predefined subtle errors into partial tokens of correct solutions to construct hard pairs for error mitigation, and refines the training objective to focus on predefined errors and their tokens, without requiring fine-grained sampling or preference annotation.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have exhibited strong mathematical reasoning and computational prowess, tackling tasks ranging from basic arithmetic to advanced competition-level problems. However, frequently occurring subtle errors, such as miscalculations or incorrect substitutions, limit the models' full mathematical potential. Existing studies to improve mathematical ability typically involve distilling reasoning skills from stronger LLMs or applying preference learning to step-wise response pairs. Although these methods leverage samples of varying granularity to mitigate reasoning errors, they overlook the frequently occurring subtle errors. A major reason is that sampled preference pairs involve differences unrelated to the errors, which may distract the model from focusing on subtle errors. In this work, we propose a novel preference learning framework called eRror-Injected Self-Editing (RISE), which injects predefined subtle errors into partial tokens of correct solutions to construct hard pairs for error mitigation. In detail, RISE uses the model itself to edit a small number of tokens in the solution, injecting designed subtle errors. Then, pairs composed of self-edited solutions and their corresponding correct ones, along with pairs of correct and incorrect solutions obtained through sampling, are used together for subtle error-aware DPO training. Compared with other preference learning methods, RISE further refines the training objective to focus on predefined errors and their tokens, without requiring fine-grained sampling or preference annotation. Extensive experiments validate the effectiveness of RISE, with preference learning on Qwen2-7B-Instruct yielding notable improvements of 3.0% on GSM8K and 7.9% on MATH.
     </details>

126. **Towards Self-Improvement of LLMs via MCTS: Leveraging Stepwise Knowledge with Curriculum Preference Learning** [[pdf]](http://arxiv.org/abs/2410.06508) `2024-10-09` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Experimental results on mathematical reasoning tasks demonstrate that AlphaLLM-CPL significantly outperforms previous MCTS behavior distillation methods, substantially boosting the reasoning capabilities of LLMs.
     </details>


     <details>
          <summary>Abstract</summary>
          Monte Carlo Tree Search (MCTS) has recently emerged as a powerful technique for enhancing the reasoning capabilities of LLMs. Techniques such as SFT or DPO have enabled LLMs to distill high-quality behaviors from MCTS, improving their reasoning performance. However, existing distillation methods underutilize the rich trajectory information generated by MCTS, limiting the potential for improvements in LLM reasoning. In this paper, we propose AlphaLLM-CPL, a novel pairwise training framework that enables LLMs to self-improve through MCTS behavior distillation. AlphaLLM-CPL efficiently leverages MCTS trajectories via two key innovations: (1) AlphaLLM-CPL constructs stepwise trajectory pairs from child nodes sharing the same parent in the search tree, providing step-level information for more effective MCTS behavior distillation. (2) AlphaLLM-CPL introduces curriculum preference learning, dynamically adjusting the training sequence of trajectory pairs in each offline training epoch to prioritize critical learning steps and mitigate overfitting. Experimental results on mathematical reasoning tasks demonstrate that AlphaLLM-CPL significantly outperforms previous MCTS behavior distillation methods, substantially boosting the reasoning capabilities of LLMs.
     </details>

127. **Beyond Captioning: Task-Specific Prompting for Improved VLM Performance in Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2410.05928) `2024-10-08` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work finds that captioning results are not generalizable, and presents a promising alternative: task-based prompting, enriching the prompt with task-specific guidance, which shows promise and proves more effective than direct captioning methods for math-heavy problems.
     </details>


     <details>
          <summary>Abstract</summary>
          Vision-Language Models (VLMs) have transformed tasks requiring visual and reasoning abilities, such as image retrieval and Visual Question Answering (VQA). Despite their success, VLMs face significant challenges with tasks involving geometric reasoning, algebraic problem-solving, and counting. These limitations stem from difficulties effectively integrating multiple modalities and accurately interpreting geometry-related tasks. Various works claim that introducing a captioning pipeline before VQA tasks enhances performance. We incorporated this pipeline for tasks involving geometry, algebra, and counting. We found that captioning results are not generalizable, specifically with larger VLMs primarily trained on downstream QnA tasks showing random performance on math-related challenges. However, we present a promising alternative: task-based prompting, enriching the prompt with task-specific guidance. This approach shows promise and proves more effective than direct captioning methods for math-heavy problems.
     </details>

128. **Give me a hint: Can LLMs take a hint to solve math problems?** [[pdf]](http://arxiv.org/abs/2410.05915) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes giving "hints" to improve the language model's performance on advanced mathematical problems, taking inspiration from how humans approach math pedagogically, and demonstrates the effectiveness of the approach by evaluating various LLMs.
     </details>


     <details>
          <summary>Abstract</summary>
          While state-of-the-art LLMs have shown poor logical and basic mathematical reasoning, recent works try to improve their problem-solving abilities using prompting techniques. We propose giving "hints" to improve the language model's performance on advanced mathematical problems, taking inspiration from how humans approach math pedagogically. We also test robustness to adversarial hints and demonstrate their sensitivity to them. We demonstrate the effectiveness of our approach by evaluating various diverse LLMs, presenting them with a broad set of problems of different difficulties and topics from the MATH dataset and comparing against techniques such as one-shot, few-shot, and chain of thought prompting.
     </details>

129. **LeanAgent: Lifelong Learning for Formal Theorem Proving** [[pdf]](http://arxiv.org/abs/2410.06209) `ICLR 2025 Submission` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          LeanAgent is presented, a novel lifelong learning framework for theorem proving that continuously generalizes to and improves on ever-expanding mathematical knowledge without forgetting previously learned knowledge and achieves exceptional scores in stability and backward transfer.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have been successful in mathematical reasoning tasks such as formal theorem proving when integrated with interactive proof assistants like Lean. Existing approaches involve training or fine-tuning an LLM on a specific dataset to perform well on particular domains, such as undergraduate-level mathematics. These methods struggle with generalizability to advanced mathematics. A fundamental limitation is that these approaches operate on static domains, failing to capture how mathematicians often work across multiple domains and projects simultaneously or cyclically. We present LeanAgent, a novel lifelong learning framework for formal theorem proving that continuously generalizes to and improves on ever-expanding mathematical knowledge without forgetting previously learned knowledge. LeanAgent introduces several key innovations, including a curriculum learning strategy that optimizes the learning trajectory in terms of mathematical difficulty, a dynamic database for efficient management of evolving mathematical knowledge, and progressive training to balance stability and plasticity. LeanAgent successfully proves 155 theorems previously unproved formally by humans across 23 diverse Lean repositories, many from advanced mathematics. It performs significantly better than the static LLM baseline, proving challenging theorems in domains like abstract algebra and algebraic topology while showcasing a clear progression of learning from basic concepts to advanced topics. In addition, we analyze LeanAgent's superior performance on key lifelong learning metrics. LeanAgent achieves exceptional scores in stability and backward transfer, where learning new tasks improves performance on previously learned tasks. This emphasizes LeanAgent's continuous generalizability and improvement, explaining its superior theorem-proving performance.
     </details>

130. **GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models** [[pdf]](https://arxiv.org/abs/2410.05229v1) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          GSM-Symbolic is introduced, an improved benchmark created from symbolic templates that allow for the generation of a diverse set of questions that enables more controllable evaluations, providing key insights and more reliable metrics for measuring the reasoning capabilities of models.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent advancements in Large Language Models (LLMs) have sparked interest in their formal reasoning capabilities, particularly in mathematics. The GSM8K benchmark is widely used to assess the mathematical reasoning of models on grade-school-level questions. While the performance of LLMs on GSM8K has significantly improved in recent years, it remains unclear whether their mathematical reasoning capabilities have genuinely advanced, raising questions about the reliability of the reported metrics. To address these concerns, we conduct a large-scale study on several SOTA open and closed models. To overcome the limitations of existing evaluations, we introduce GSM-Symbolic, an improved benchmark created from symbolic templates that allow for the generation of a diverse set of questions. GSM-Symbolic enables more controllable evaluations, providing key insights and more reliable metrics for measuring the reasoning capabilities of models.Our findings reveal that LLMs exhibit noticeable variance when responding to different instantiations of the same question. Specifically, the performance of all models declines when only the numerical values in the question are altered in the GSM-Symbolic benchmark. Furthermore, we investigate the fragility of mathematical reasoning in these models and show that their performance significantly deteriorates as the number of clauses in a question increases. We hypothesize that this decline is because current LLMs cannot perform genuine logical reasoning; they replicate reasoning steps from their training data. Adding a single clause that seems relevant to the question causes significant performance drops (up to 65%) across all state-of-the-art models, even though the clause doesn't contribute to the reasoning chain needed for the final answer. Overall, our work offers a more nuanced understanding of LLMs' capabilities and limitations in mathematical reasoning.
     </details>

131. **ImProver: Agent-Based Automated Proof Optimization** [[pdf]](https://arxiv.org/abs/2410.04753v1) `ICLR 2025 Submission` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          ImProver is presented, a large-language-model agent that rewrites proofs to optimize arbitrary user-defined metrics in Lean and is capable of rewriting proofs so that they are substantially shorter, more modular, and more readable.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have been used to generate formal proofs of mathematical theorems in proofs assistants such as Lean. However, we often want to optimize a formal proof with respect to various criteria, depending on its downstream use. For example, we may want a proof to adhere to a certain style, or to be readable, concise, or modularly structured. Having suitably optimized proofs is also important for learning tasks, especially since human-written proofs may not optimal for that purpose. To this end, we study a new problem of automated proof optimization: rewriting a proof so that it is correct and optimizes for an arbitrary criterion, such as length or readability. As a first method for automated proof optimization, we present ImProver, a large-language-model agent that rewrites proofs to optimize arbitrary user-defined metrics in Lean. We find that naively applying LLMs to proof optimization falls short, and we incorporate various improvements into ImProver, such as the use of symbolic Lean context in a novel Chain-of-States technique, as well as error-correction and retrieval. We test ImProver on rewriting real-world undergraduate, competition, and research-level mathematics theorems, finding that ImProver is capable of rewriting proofs so that they are substantially shorter, more modular, and more readable.
     </details>

132. **Reasoning Paths Optimization: Learning to Reason and Explore From Diverse Paths** [[pdf]](http://arxiv.org/abs/2410.10858) `2024-10-07` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Advanced models such as OpenAI o1 exhibit impressive problem-solving capabilities through step-by-step reasoning. However, they may still falter on more complex problems, making errors that disrupt their reasoning paths. We attribute this to the expansive solution space, where each step has the risk of diverging into mistakes. To enhance language model reasoning, we introduce a specialized training framework called Reasoning Paths Optimization (RPO), which enables learning to reason and explore from diverse paths. Our approach encourages favorable branches at each reasoning step while penalizing unfavorable ones, enhancing the model's overall problem-solving performance. Reasoning Paths Optimization does not rely on large-scale human-annotated rationales or outputs from closed-source models, making it scalable and data-efficient. We focus on multi-step reasoning tasks, such as math word problems and science-based exam questions. The experiments demonstrate that our framework significantly enhances the reasoning performance of large language models, with up to 3.1% and 4.3% improvement on GSM8K and MMLU (STEM) respectively. Our data and code can be found at https://reasoning-paths.github.io.
     </details>

133. **ErrorRadar: Benchmarking Complex Mathematical Reasoning of Multimodal Large Language Models Via Error Detection** [[pdf]](https://arxiv.org/abs/2410.04509v1) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          ErrorRadar is introduced, the first benchmark designed to assess MLLMs' capabilities in such a task, and contains 2,500 high-quality multimodal K-12 mathematical problems, collected from real-world student interactions in an educational organization, with rigorous annotation and rich metadata such as problem type and error category.
     </details>


     <details>
          <summary>Abstract</summary>
          As the field of Multimodal Large Language Models (MLLMs) continues to evolve, their potential to revolutionize artificial intelligence is particularly promising, especially in addressing mathematical reasoning tasks. Current mathematical benchmarks predominantly focus on evaluating MLLMs' problem-solving ability, yet there is a crucial gap in addressing more complex scenarios such as error detection, for enhancing reasoning capability in complicated settings. To fill this gap, we formally formulate the new task: multimodal error detection, and introduce ErrorRadar, the first benchmark designed to assess MLLMs' capabilities in such a task. ErrorRadar evaluates two sub-tasks: error step identification and error categorization, providing a comprehensive framework for evaluating MLLMs' complex mathematical reasoning ability. It consists of 2,500 high-quality multimodal K-12 mathematical problems, collected from real-world student interactions in an educational organization, with rigorous annotation and rich metadata such as problem type and error category. Through extensive experiments, we evaluated both open-source and closed-source representative MLLMs, benchmarking their performance against educational expert evaluators. Results indicate significant challenges still remain, as GPT-4o with best performance is still around 10% behind human evaluation. The dataset will be available upon acceptance.
     </details>

134. **MathHay: An Automated Benchmark for Long-Context Mathematical Reasoning in LLMs** [[pdf]](http://arxiv.org/abs/2410.04698) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduces MathHay, an automated benchmark designed to assess the long-context mathematical reasoning capabilities of LLMs, and conducts extensive experiments on MathHay to assess the long-context mathematical reasoning abilities of eight top-performing LLMs.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent large language models (LLMs) have demonstrated versatile capabilities in long-context scenarios. Although some recent benchmarks have been developed to evaluate the long-context capabilities of LLMs, there is a lack of benchmarks evaluating the mathematical reasoning abilities of LLMs over long contexts, which is crucial for LLMs' application in real-world scenarios. In this paper, we introduce MathHay, an automated benchmark designed to assess the long-context mathematical reasoning capabilities of LLMs. Unlike previous benchmarks like Needle in a Haystack, which focus primarily on information retrieval within long texts, MathHay demands models with both information-seeking and complex mathematical reasoning abilities. We conduct extensive experiments on MathHay to assess the long-context mathematical reasoning abilities of eight top-performing LLMs. Even the best-performing model, Gemini-1.5-Pro-002, still struggles with mathematical reasoning over long contexts, achieving only 51.26% accuracy at 128K tokens. This highlights the significant room for improvement on the MathHay benchmark.
     </details>

135. **Polymath: A Challenging Multi-modal Mathematical Reasoning Benchmark** [[pdf]](http://arxiv.org/abs/2410.14702) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is discovered that models do not truly comprehend visual diagrams and the spatial information therein, and are thus prone to logical errors, and the OpenAI o1 models are evaluated and find that their performance only matches the human baseline, highlighting the difficulty of the benchmark.
     </details>


     <details>
          <summary>Abstract</summary>
          Multi-modal Large Language Models (MLLMs) exhibit impressive problem-solving abilities in various domains, but their visual comprehension and abstract reasoning skills remain under-evaluated. To this end, we present PolyMATH, a challenging benchmark aimed at evaluating the general cognitive reasoning abilities of MLLMs. PolyMATH comprises 5,000 manually collected high-quality images of cognitive textual and visual challenges across 10 distinct categories, including pattern recognition, spatial reasoning, and relative reasoning. We conducted a comprehensive, and quantitative evaluation of 15 MLLMs using four diverse prompting strategies, including Chain-of-Thought and Step-Back. The best scores achieved on PolyMATH are ~41%, ~36%, and ~27%, obtained by Claude-3.5 Sonnet, GPT-4o and Gemini-1.5 Pro respectively - highlighting the logical and visual complexity of these questions. A further fine-grained error analysis reveals that these models struggle to understand spatial relations and perform drawn-out, high-level reasoning. This is further strengthened by our ablation study estimating MLLM performance when given textual descriptions in place of diagrams. As evidenced by ~4% improvement over textual descriptions as opposed to actual images, we discover that models do not truly comprehend visual diagrams and the spatial information therein, and are thus prone to logical errors. Finally, we evaluate the OpenAI o1 models and find that their performance only matches the human baseline, highlighting the difficulty of the benchmark. The results on PolyMATH highlight the room for improvement in multi-modal reasoning and provide unique insights to guide the development of future MLLMs.
     </details>

136. **Wrong-of-Thought: An Integrated Reasoning Framework with Multi-Perspective Verification and Wrong Information** [[pdf]](https://arxiv.org/abs/2410.04463v1) `2024-10-06` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Wrong-of-Thought (WoT) is proposed, which includes two core modules: Multi-Perspective Verification: A multi-perspective verification method for accurately refining the reasoning process and result, and Wrong Information Utilization: Utilizing wrong information to alert LLMs and reduce the probability of LLMs making same mistakes.
     </details>


     <details>
          <summary>Abstract</summary>
          Chain-of-Thought (CoT) has become a vital technique for enhancing the performance of Large Language Models (LLMs), attracting increasing attention from researchers. One stream of approaches focuses on the iterative enhancement of LLMs by continuously verifying and refining their reasoning outputs for desired quality. Despite its impressive results, this paradigm faces two critical issues: (1) Simple verification methods: The current paradigm relies solely on a single verification method. (2) Wrong Information Ignorance: Traditional paradigms directly ignore wrong information during reasoning and refine the logic paths from scratch each time. To address these challenges, we propose Wrong-of-Thought (WoT), which includes two core modules: (1) Multi-Perspective Verification: A multi-perspective verification method for accurately refining the reasoning process and result, and (2) Wrong Information Utilization: Utilizing wrong information to alert LLMs and reduce the probability of LLMs making same mistakes. Experiments on 8 popular datasets and 5 LLMs demonstrate that WoT surpasses all previous baselines. In addition, WoT exhibits powerful capabilities in difficult computation tasks.
     </details>

137. **BloomWise: Enhancing Problem-Solving capabilities of Large Language Models using Bloom's-Taxonomy-Inspired Prompts** [[pdf]](https://arxiv.org/abs/2410.04094v1) `2024-10-05` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          BloomWise, a new prompting technique, inspired by Bloom's Taxonomy, aiming to improve LLMs' performance in solving such problems by encouraging them to approach the problem starting from simple, i.e., remembering, and progressing to higher cognitive skills, i.e., analyzing, until the correct solution is reached.
     </details>


     <details>
          <summary>Abstract</summary>
          Despite the continuous progress of Large Language Models (LLMs) across various tasks, their performance on mathematical problems and reasoning tasks remains limited. This limitation can be attributed, among other factors, to the inherent difficulty of these problems and the fact that solutions often consist of multiple steps, potentially of varying nature, making it challenging for a single prompting technique to execute all required steps. To address this, we introduce BloomWise, a new prompting technique, inspired by Bloom's Taxonomy, aiming to improve LLMs' performance in solving such problems by encouraging them to approach the problem starting from simple, i.e., remembering, and progressing to higher cognitive skills, i.e., analyzing, until the correct solution is reached. The decision regarding the need to employ more sophisticated cognitive skills is based on self-evaluation performed by the LLM. Thus, we encourage the LLM to deploy the appropriate cognitive processes. In extensive experiments across 4 popular math reasoning datasets, we have demonstrated the effectiveness of our proposed approach. We also present extensive ablations, analyzing the strengths of each module within our system.
     </details>

138. **Consistent Autoformalization for Constructing Mathematical Libraries** [[pdf]](http://arxiv.org/abs/2410.04194) `2024-10-05` `Isabelle` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper proposes the coordinated use of three mechanisms, most-similar retrieval augmented generation (MS-RAG), denoising steps, and auto-correction with syntax error feedback (Auto-SEF) to improve autoformalization quality.
     </details>


     <details>
          <summary>Abstract</summary>
          Autoformalization is the task of automatically translating mathematical content written in natural language to a formal language expression. The growing language interpretation capabilities of Large Language Models (LLMs), including in formal languages, are lowering the barriers for autoformalization. However, LLMs alone are not capable of consistently and reliably delivering autoformalization, in particular as the complexity and specialization of the target domain grows. As the field evolves into the direction of systematically applying autoformalization towards large mathematical libraries, the need to improve syntactic, terminological and semantic control increases. This paper proposes the coordinated use of three mechanisms, most-similar retrieval augmented generation (MS-RAG), denoising steps, and auto-correction with syntax error feedback (Auto-SEF) to improve autoformalization quality. The empirical analysis, across different models, demonstrates that these mechanisms can deliver autoformalizaton results which are syntactically, terminologically and semantically more consistent. These mechanisms can be applied across different LLMs and have shown to deliver improve results across different model types.
     </details>

139. **Improving LLM Reasoning through Scaling Inference Computation with Collaborative Verification** [[pdf]](http://arxiv.org/abs/2410.05318) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces a comprehensive dataset consisting of correct and incorrect solutions for math and code tasks, generated by multiple LLMs, and proposes a novel collaborative method integrating Chain-of-Thought (CoT) and Program-of-Thought (PoT) solutions for verification.
     </details>


     <details>
          <summary>Abstract</summary>
          Despite significant advancements in the general capability of large language models (LLMs), they continue to struggle with consistent and accurate reasoning, especially in complex tasks such as mathematical and code reasoning. One key limitation is that LLMs are trained primarily on correct solutions, reducing their ability to detect and learn from errors, which hampers their ability to reliably verify and rank outputs. To address this, we scale up the inference-time computation by generating multiple reasoning paths and employing verifiers to assess and rank the generated outputs by correctness. To facilitate this, we introduce a comprehensive dataset consisting of correct and incorrect solutions for math and code tasks, generated by multiple LLMs. This diverse set of solutions enables verifiers to more effectively distinguish and rank correct answers from erroneous outputs. The training methods for building verifiers were selected based on an extensive comparison of existing approaches. Moreover, to leverage the unique strengths of different reasoning strategies, we propose a novel collaborative method integrating Chain-of-Thought (CoT) and Program-of-Thought (PoT) solutions for verification. CoT provides a clear, step-by-step reasoning process that enhances interpretability, while PoT, being executable, offers a precise and error-sensitive validation mechanism. By taking both of their strengths, our approach significantly improves the accuracy and reliability of reasoning verification. Our verifiers, Math-Rev and Code-Rev, demonstrate substantial performance gains to existing LLMs, achieving state-of-the-art results on benchmarks such as GSM8k and MATH and even outperforming GPT-4o with Qwen-72B-Instruct as the reasoner.
     </details>

140. **Alchemy: Amplifying Theorem-Proving Capability through Symbolic Mutation** [[pdf]](https://openreview.net/forum?id=7NL74jUiMg) `ICLR 2025 Submission` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Alchemy, a general framework for data synthesis that constructs formal theorems through symbolic mutation, is proposed, where for each candidate theorem in Mathlib, all invocable theorems that can be used to rewrite or apply to it are identified.
     </details>


     <details>
          <summary>Abstract</summary>
          Formal proofs are challenging to write even for experienced experts. Recent progress in Neural Theorem Proving (NTP) shows promise in expediting this process. However, the formal corpora available on the Internet are limited compared to the general text, posing a significant data scarcity challenge for NTP. To address this issue, this work proposes Alchemy, a general framework for data synthesis that constructs formal theorems through symbolic mutation. Specifically, for each candidate theorem in Mathlib, we identify all invocable theorems that can be used to rewrite or apply to it. Subsequently, we mutate the candidate theorem by replacing the corresponding term in the statement with its equivalent form or antecedent. As a result, our method increases the number of theorems in Mathlib by an order of magnitude, from 110k to 6M. Furthermore, we perform continual pretraining and supervised finetuning on this augmented corpus for large language models. Experimental results demonstrate the effectiveness of our approach, achieving a 5% absolute performance improvement on Leandojo benchmark. Additionally, our synthetic data achieve a 2.5% absolute performance gain on the out-of-distribution miniF2F benchmark. To provide further insights, we conduct a comprehensive analysis of synthetic data composition and the training paradigm, offering valuable guidance for developing a strong theorem prover.
     </details>

141. **DOTS: Learning to Reason Dynamically in LLMs via Optimal Reasoning Trajectories Search** [[pdf]](http://arxiv.org/abs/2410.03864) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          DOTS is proposed, an approach enabling LLMs to reason dynamically via optimal reasoning trajectory search, tailored to the specific characteristics of each question and the inherent capability of the task-solving LLM.
     </details>


     <details>
          <summary>Abstract</summary>
          Enhancing the capability of large language models (LLMs) in reasoning has gained significant attention in recent years. Previous studies have demonstrated the effectiveness of various prompting strategies in aiding LLMs in reasoning (called "reasoning actions"), such as step-by-step thinking, reflecting before answering, solving with programs, and their combinations. However, these approaches often applied static, predefined reasoning actions uniformly to all questions, without considering the specific characteristics of each question or the capability of the task-solving LLM. In this paper, we propose DOTS, an approach enabling LLMs to reason dynamically via optimal reasoning trajectory search, tailored to the specific characteristics of each question and the inherent capability of the task-solving LLM. Our approach involves three key steps: i) defining atomic reasoning action modules that can be composed into various reasoning action trajectories; ii) searching for the optimal action trajectory for each training question through iterative exploration and evaluation for the specific task-solving LLM; and iii) using the collected optimal trajectories to train an LLM to plan for the reasoning trajectories of unseen questions. In particular, we propose two learning paradigms, i.e., fine-tuning an external LLM as a planner to guide the task-solving LLM, or directly fine-tuning the task-solving LLM with an internalized capability for reasoning actions planning. Our experiments across eight reasoning tasks show that our method consistently outperforms static reasoning techniques and the vanilla instruction tuning approach. Further analysis reveals that our method enables LLMs to adjust their computation based on problem complexity, allocating deeper thinking and reasoning to harder problems.
     </details>

142. **Deliberate Reasoning for LLMs as Structure-aware Planning with Accurate World Model** [[pdf]](https://arxiv.org/abs/2410.03136v1) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a novel multi-step reasoning framework for LLMs, referred to as Structure-aware Planning with Accurate World Model (SWAP), which incorporates structural information to guide the reasoning process via a world model and provides a soft verification mechanism over the steps.
     </details>


     <details>
          <summary>Abstract</summary>
          Enhancing the reasoning capabilities of large language models (LLMs) remains a key challenge, especially for tasks that require complex, multi-step decision-making. Humans excel at these tasks by leveraging deliberate planning with an internal world model to simulate the potential outcomes of various actions. Inspired by this, we propose a novel multi-step reasoning framework for LLMs, referred to as Structure-aware Planning with Accurate World Model (SWAP). Unlike previous approaches that rely solely on Chain-of-Thought (CoT) reasoning in natural language, SWAP incorporates structural information to guide the reasoning process via a world model and provides a soft verification mechanism over the steps. Moreover, SWAP overcomes the challenge of accurate world state predictions in complex reasoning tasks by introducing a Generator-Discriminator architecture, which enables more reliable world modeling. Specifically, the generator predicts the next state, and the discriminator ensures alignment with the logical consistency required by the problem context. SWAP also encourages the policy model to explore a broad range of potential actions to prevent premature convergence. By resolving the bottlenecks of generation diversity for both actions and states using diversity-based modeling (DBM) and improving discrimination accuracy through contrastive ranking (CR), SWAP significantly enhances the reasoning performance of LLMs. We evaluate SWAP across diverse reasoning-intensive benchmarks including math reasoning, logical reasoning, and coding tasks. Extensive experiments demonstrate that SWAP achieves substantial improvements over the baselines and consistently outperforms existing methods.
     </details>

143. **FormalAlign: Automated Alignment Evaluation for Autoformalization** [[pdf]](https://openreview.net/forum?id=B5RrIFMqbe) `ICLR 2025 Submission` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces \textsc{FormalAlign], the first automated framework designed for evaluating the alignment between natural and formal languages in autoformalization, and significantly reduces the need for manual verification.
     </details>


     <details>
          <summary>Abstract</summary>
          Autoformalization aims to convert informal mathematical proofs into machine-verifiable formats, bridging the gap between natural and formal languages. However, ensuring semantic alignment between the informal and formalized statements remains challenging. Existing approaches heavily rely on manual verification, hindering scalability. To address this, we introduce \textsc{FormalAlign}, the first automated framework designed for evaluating the alignment between natural and formal languages in autoformalization. \textsc{FormalAlign} trains on both the autoformalization sequence generation task and the representational alignment between input and output, employing a dual loss that combines a pair of mutually enhancing autoformalization and alignment tasks. Evaluated across four benchmarks augmented by our proposed misalignment strategies, \textsc{FormalAlign} demonstrates superior performance. In our experiments, \textsc{FormalAlign} outperforms GPT-4, achieving an Alignment-Selection Score 11.58\% higher on \forml-Basic (99.21\% vs. 88.91\%) and 3.19\% higher on MiniF2F-Valid (66.39\% vs. 64.34\%). This effective alignment evaluation significantly reduces the need for manual verification. Both the dataset and code can be accessed via~\url{https://github.com/rookie-joe/FormalAlign}.
     </details>

144. **MathGAP: Out-of-Distribution Evaluation on Problems with Arbitrarily Complex Proofs** [[pdf]](https://openreview.net/forum?id=5ck9PIrTpH) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) can solve arithmetic word problems with high accuracy, but little is known about how well they generalize to problems that are more complex than the ones on which they have been trained. Empirical investigations of such questions are impeded by two major flaws of current evaluations: (i) much of the evaluation data is contaminated, in the sense that it has already been seen during training, and (ii) benchmark datasets do not capture how problem proofs may be arbitrarily complex in various ways. As a step towards addressing these issues, we present a framework for evaluating LLMs on problems with arbitrarily complex arithmetic proofs, called MathGAP. MathGAP generates problems that follow fixed proof specifications -- along with chain-of-thought reasoning annotations -- enabling systematic studies on generalization with respect to arithmetic proof complexity. We apply MathGAP to analyze how in-context learning interacts with generalization to problems that have more complex proofs. We find that among the models tested, most show a significant decrease in performance as proofs get deeper and wider. This effect is more pronounced in complex, nonlinear proof structures, which are challenging even for GPT-4o. Surprisingly, providing in-context examples from the same distribution as the test set is not always beneficial for performance. In particular, zero-shot prompting as well as demonstrating a diverse range of examples that are less complex than the test data sometimes yield similar or higher accuracies.
     </details>

145. **Number Cookbook: Number Understanding of Language Models and How to Improve It** [[pdf]](https://openreview.net/forum?id=BWS5gVjgeY) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) can solve an increasing number of complex reasoning tasks while making surprising mistakes in basic numerical understanding and processing (such as 9.11 > 9.9). The latter ability is essential for tackling complex arithmetic and mathematical problems and serves as a foundation for most reasoning tasks, but previous work paid little attention to it or only discussed several restricted tasks (like integer addition). In this paper, we comprehensively investigate the numerical understanding and processing ability (NUPA) of LLMs. Firstly, we introduce a benchmark covering four common numerical representations and 17 distinct numerical tasks in four major categories, resulting in 41 meaningful combinations in total. These tasks are derived from primary and secondary education curricula, encompassing nearly all everyday numerical understanding and processing scenarios, and the rules of these tasks are very simple and clear. Through the benchmark, we find that current LLMs fail frequently in many of the tasks. To study the problem, we train small models with existing and potential techniques for enhancing NUPA (such as tokenizers, PEs, and number formats), comprehensively evaluating their effectiveness using our testbed. We also finetune practical-scale LLMs on our proposed NUPA tasks and find that 1) naive finetuning can improve NUPA a lot on many but not all tasks, and 2) surprisingly, techniques designed to enhance NUPA prove ineffective for finetuning pretrained models. We further explore the impact of chain-of-thought techniques on NUPA. Our work provides a more detailed and comprehensive understanding of NUPA in LLMs. Our benchmark and code are released at https://github.com/GraphPKU/number_cookbook.
     </details>

146. **Preference Optimization for Reasoning with Pseudo Feedback** [[pdf]](https://openreview.net/forum?id=jkUp3lybXf) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Preference optimization techniques, such as Direct Preference Optimization (DPO), are frequently employed to enhance the reasoning capabilities of large language models (LLMs) in domains like mathematical reasoning and coding, typically following supervised fine-tuning. These methods rely on high-quality labels for reasoning tasks to generate preference pairs; however, the availability of reasoning datasets with human-verified labels is limited. In this study, we introduce a novel approach to generate pseudo feedback for reasoning tasks by framing the labeling of solutions to reason problems as an evaluation against associated test cases. We explore two forms of pseudo feedback based on test cases: one generated by frontier LLMs and the other by extending self-consistency to multi-test-case. We conduct experiments on both mathematical reasoning and coding tasks using pseudo feedback for preference optimization, and observe improvements across both tasks. Specifically, using Mathstral-7B as our base model, we improve MATH results from 58.3 to 68.6, surpassing both NuminaMath-72B and GPT-4-Turbo-1106-preview. In GSM8K and College Math, our scores increase from 85.6 to 90.3 and from 34.3 to 42.3, respectively. Building on Deepseek-coder-7B-v1.5, we achieve a score of 24.6 on LiveCodeBench (from 21.1), surpassing Claude-3-Haiku.
     </details>

147. **Seq-VCR: Preventing Collapse in Intermediate Transformer Representations for Enhanced Reasoning** [[pdf]](https://openreview.net/forum?id=30oIfmrcFO) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes Sequential Variance-Covariance Regularization (Seq-VCR), which enhances the entropy of intermediate representations and prevents collapse, and combines with dummy pause tokens as substitutes for chain-of-thought (CoT) tokens to improve performance in arithmetic reasoning problems.
     </details>


     <details>
          <summary>Abstract</summary>
          Decoder-only Transformers often struggle with complex reasoning tasks, particularly arithmetic reasoning requiring multiple sequential operations. In this work, we identify representation collapse in the model's intermediate layers as a key factor limiting their reasoning capabilities. To address this, we propose Sequential Variance-Covariance Regularization (Seq-VCR), which enhances the entropy of intermediate representations and prevents collapse. Combined with dummy pause tokens as substitutes for chain-of-thought (CoT) tokens, our method significantly improves performance in arithmetic reasoning problems. In the challenging $5 \times 5$ integer multiplication task, our approach achieves $99.5\%$ exact match accuracy, outperforming models of the same size (which yield $0\%$ accuracy) and GPT-4 with five-shot CoT prompting ($44\%$). We also demonstrate superior results on arithmetic expression and longest increasing subsequence (LIS) datasets. Our findings highlight the importance of preventing intermediate layer representation collapse to enhance the reasoning capabilities of Transformers and show that Seq-VCR offers an effective solution without requiring explicit CoT supervision.
     </details>

148. **Steering Large Language Models between Code Execution and Textual Reasoning** [[pdf]](https://arxiv.org/abs/2410.03524v1) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is discovered that results from LLM written code are not always better than using textual reasoning, even if the task could be solved through code, so three methods to better steer LLM code/text generation are proposed and achieve a notable improvement.
     </details>


     <details>
          <summary>Abstract</summary>
          While a lot of recent research focuses on enhancing the textual reasoning capabilities of Large Language Models (LLMs) by optimizing the multi-agent framework or reasoning chains, several benchmark tasks can be solved with 100% success through direct coding, which is more scalable and avoids the computational overhead associated with textual iterating and searching. Textual reasoning has inherent limitations in solving tasks with challenges in math, logics, optimization, and searching, which is unlikely to be solved by simply scaling up the model and data size. The recently released OpenAI GPT Code Interpreter and multi-agent frameworks such as AutoGen have demonstrated remarkable proficiency of integrating code generation and execution to solve complex tasks using LLMs. However, based on our experiments on 7 existing popular methods for steering code/text generation in both single- and multi-turn settings with 14 tasks and 6 types of LLMs (including the new O1-preview), currently there is no optimal method to correctly steer LLMs to write code when needed. We discover some interesting patterns on when models use code vs. textual reasoning with the evolution to task complexity and model sizes, which even result in an astonishingly inverse scaling law. We also discover that results from LLM written code are not always better than using textual reasoning, even if the task could be solved through code. To mitigate the above issues, we propose three methods to better steer LLM code/text generation and achieve a notable improvement. The costs of token lengths and runtime are thoroughly discussed for all the methods. We believe the problem of steering LLM code/text generation is critical for future research and has much space for further improvement. Project Page, Datasets, and Codes are available at https://yongchao98.github.io/CodeSteer/.
     </details>

149. **System 2 reasoning capabilities are nigh** [[pdf]](https://arxiv.org/abs/2410.03662v1) `2024-10-04` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is argued that if current models are insufficient to be classed as performing reasoning, there remains very little additional progress needed to attain that goal.
     </details>


     <details>
          <summary>Abstract</summary>
          In recent years, machine learning models have made strides towards human-like reasoning capabilities from several directions. In this work, we review the current state of the literature and describe the remaining steps to achieve a neural model which can perform System 2 reasoning analogous to a human. We argue that if current models are insufficient to be classed as performing reasoning, there remains very little additional progress needed to attain that goal.
     </details>

150. **U-MATH: A University-Level Benchmark for Evaluating Mathematical Skills in LLMs** [[pdf]](https://openreview.net/forum?id=xlxGsX1pc7) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          The current evaluation of mathematical skills in LLMs is limited, as existing benchmarks are either relatively small, primarily focus on elementary and high-school problems, or lack diversity in topics. Additionally, the inclusion of visual elements in tasks remains largely under-explored.   To address these gaps, we introduce U-MATH, a novel benchmark of 1,100 unpublished open-ended university-level problems sourced from teaching materials. It is balanced across six core subjects, with 20% of multimodal problems. Given the open-ended nature of U-MATH problems, we employ an LLM to judge the correctness of generated solutions. To this end, we release $\mu$-MATH, a dataset to evaluate the LLMs' capabilities in judging solutions.   The evaluation of general domain, math-specific, and multimodal LLMs highlights the challenges presented by U-MATH. Our findings reveal that LLMs achieve a maximum accuracy of only 63% on text-based tasks, with even lower 45% on visual problems. The solution assessment proves challenging for LLMs, with the best LLM judge having an F1-score of 80% on $\mu$-MATH.
     </details>

151. **VerifierQ: Enhancing LLM Test Time Compute with Q-Learning-based Verifiers** [[pdf]](https://openreview.net/forum?id=OD9pwKQzXl) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          VerifierQ is among the first to investigate the verifier (critic) aspect in LLMs through Q-learning, and introduces a modified Bellman update for bounded Q-values, incorporates Implicit Q-learning (IQL) for efficient action space management, and integrates a novel Conservative Q-learning (CQL) formulation for balanced Q-value estimation.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent advancements in test time compute, particularly through the use of verifier models, have significantly enhanced the reasoning capabilities of Large Language Models (LLMs). This generator-verifier approach closely resembles the actor-critic framework in reinforcement learning (RL). However, current verifier models in LLMs often rely on supervised fine-tuning without temporal difference learning such as Q-learning. This paper introduces VerifierQ, a novel approach that integrates Offline Q-learning into LLM verifier models. We address three key challenges in applying Q-learning to LLMs: (1) handling utterance-level Markov Decision Processes (MDPs), (2) managing large action spaces, and (3) mitigating overestimation bias. VerifierQ introduces a modified Bellman update for bounded Q-values, incorporates Implicit Q-learning (IQL) for efficient action space management, and integrates a novel Conservative Q-learning (CQL) formulation for balanced Q-value estimation. Our method enables parallel Q-value computation and improving training efficiency. While recent work has explored RL techniques like MCTS for generators, VerifierQ is among the first to investigate the verifier (critic) aspect in LLMs through Q-learning. This integration of RL principles into verifier models complements existing advancements in generator techniques, potentially enabling more robust and adaptive reasoning in LLMs. Experimental results on mathematical reasoning tasks demonstrate VerifierQ's superior performance compared to traditional supervised fine-tuning approaches, with improvements in efficiency, accuracy and robustness. By enhancing the synergy between generation and evaluation capabilities, VerifierQ contributes to the ongoing evolution of AI systems in addressing complex cognitive tasks across various domains.
     </details>

152. **ZIP-FIT: Embedding-Free Data Selection via Compression-Based Alignment** [[pdf]](https://openreview.net/forum?id=4JBEpP6eRS) `ICLR 2025 Submission` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Data selection is crucial for optimizing language model (LM) performance on specific tasks, yet most existing methods fail to effectively consider the target task distribution.   Current approaches either ignore task-specific requirements entirely or rely on approximations that fail to capture the nuanced patterns needed for tasks like Autoformalization or code generation.   Methods that do consider the target distribution often rely on simplistic, sometimes noisy, representations, like hashed n-gram features, which can lead to collisions and introduce noise.   We introduce ZIP-FIT, a data selection framework that uses gzip compression to directly measure alignment between potential training data and the target task distribution.   In extensive evaluations on Autoformalization and Python code generation, ZIP-FIT significantly outperforms leading baselines like DSIR and D4.   Models trained on ZIP-FIT-selected data achieve their lowest cross-entropy loss up to 85.1\% faster than baselines, demonstrating that better task alignment leads to more efficient learning.   In addition, ZIP-FIT performs selection up to 65.8\% faster than DSIR and two orders of magnitude faster than D4.   Notably, ZIP-FIT shows that smaller, well-aligned datasets often outperform larger but less targeted ones, demonstrating that a small amount of higher quality data is superior to a large amount of lower quality data.   Our results imply that task-aware data selection is crucial for efficient domain adaptation, and that compression offers a principled way to measure task alignment.   By showing that targeted data selection can dramatically improve task-specific performance, our work provides new insights into the relationship between data quality, task alignment, and model learning efficiency.
     </details>

153. **Adaptive Inference-Time Compute: LLMs Can Predict if They Can Do Better, Even Mid-Generation** [[pdf]](https://arxiv.org/abs/2410.02725v1) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A new generative self-evaluation scheme designed to adaptively reduce the number of generated samples while maintaining or even improving performance and enabling more efficient and scalable compute utilization during inference for LLMs is introduced.
     </details>


     <details>
          <summary>Abstract</summary>
          Inference-time computation is a powerful paradigm to enhance the performance of large language models (LLMs), with Best-of-N sampling being a widely used technique. However, this method is computationally expensive, requiring both (1) an external reward model and (2) the generation of multiple samples. In this work, we introduce a new generative self-evaluation scheme designed to adaptively reduce the number of generated samples while maintaining or even improving performance. We use a generative reward model formulation, allowing the LLM to predict mid-generation the probability that restarting the generation will yield a better response. These predictions are obtained without an external reward model and can be used to decide whether or not to generate more samples, prune unpromising samples early on, or to pick the best sample. This capability is very inexpensive as it involves generating a single predefined token. Trained using a dataset constructed with real unfiltered LMSYS user prompts, Llama 3.1 8B's win rate against GPT-4 on AlpacaEval increases from 21% to 34% with 16 samples and math performance on GSM8K improves from 84% to 91%. By sampling only when the LLM determines that it is beneficial to do so and adaptively adjusting temperature annealing, we demonstrate that 74% of the improvement from using 16 samples can be achieved with only 1.2 samples on average. We further demonstrate that 50-75% of samples can be pruned early in generation with minimal degradation in performance. Overall, our methods enable more efficient and scalable compute utilization during inference for LLMs.
     </details>

154. **CodePMP: Scalable Preference Model Pretraining for Large Language Model Reasoning** [[pdf]](http://arxiv.org/abs/2410.02229) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          CodePMP improves RM finetuning efficiency by pretraining preference models on large-scale synthesized code-preference pairs from publicly available high-quality source code, and highlighting the importance of scalable preference model pretraining for efficient reward modeling.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have made significant progress in natural language understanding and generation, driven by scalable pretraining and advanced finetuning. However, enhancing reasoning abilities in LLMs, particularly via reinforcement learning from human feedback (RLHF), remains challenging due to the scarcity of high-quality preference data, which is labor-intensive to annotate and crucial for reward model (RM) finetuning. To alleviate this issue, we introduce CodePMP, a scalable preference model pretraining (PMP) pipeline that utilizes a large corpus of synthesized code-preference pairs from publicly available high-quality source code. CodePMP improves RM finetuning efficiency by pretraining preference models on large-scale synthesized code-preference pairs. We evaluate CodePMP on mathematical reasoning tasks (GSM8K, MATH) and logical reasoning tasks (ReClor, LogiQA2.0), consistently showing significant improvements in reasoning performance of LLMs and highlighting the importance of scalable preference model pretraining for efficient reward modeling.
     </details>

155. **GraphIC: A Graph-Based In-Context Example Retrieval Model for Multi-Step Reasoning** [[pdf]](http://arxiv.org/abs/2410.02203) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          GraphIC is presented, a novel approach that leverages graph-based representations of reasoning processes, coupled with Bayesian Networks (BNs) to select ICEs, and outperforms both training-free and training-based models in selecting ICEs, excelling in terms of both effectiveness and efficiency.
     </details>


     <details>
          <summary>Abstract</summary>
          In-context learning (ICL) enables large language models (LLMs) to generalize to new tasks by incorporating a few in-context examples (ICEs) directly in the input, without updating parameters. However, the effectiveness of ICL heavily relies on the selection of ICEs, and conventional text-based embedding methods are often inadequate for tasks that require multi-step reasoning, such as mathematical and logical problem solving. This is due to the bias introduced by shallow semantic similarities that fail to capture the deeper reasoning structures required for these tasks. We present GraphIC, a novel approach that leverages graph-based representations of reasoning processes, coupled with Bayesian Networks (BNs) to select ICEs. Graph structures inherently filter out shallow semantics while preserving the core reasoning structure. Importantly, BNs capture the dependency of a node's attributes on its parent nodes, closely mirroring the hierarchical nature of human cognition-where each thought is shaped by preceding ones. This makes BNs particularly well-suited for multi-step reasoning tasks, aligning the process more closely with human-like reasoning. Extensive experiments across three types of reasoning tasks (mathematical reasoning, code generation, and logical reasoning) demonstrate that GraphIC outperforms both training-free and training-based models in selecting ICEs, excelling in terms of both effectiveness and efficiency. We show that GraphIC enhances ICL's performance and interoperability, significantly advancing ICE selection for multi-step reasoning tasks.
     </details>

156. **LLaMA-Berry: Pairwise Optimization for O1-like Olympiad-Level Mathematical Reasoning** [[pdf]](https://arxiv.org/abs/2410.02884v1) `2024-10-03` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          An advanced mathematical problem-solving framework, LLaMA-Berry, for enhancing the mathematical reasoning ability of Large Language Models (LLMs), which combines Monte Carlo Tree Search (MCTS) with iterative Self-Refine to optimize the reasoning path and utilizes a pairwise reward model to evaluate different paths globally.
     </details>


     <details>
          <summary>Abstract</summary>
          This paper presents an advanced mathematical problem-solving framework, LLaMA-Berry, for enhancing the mathematical reasoning ability of Large Language Models (LLMs). The framework combines Monte Carlo Tree Search (MCTS) with iterative Self-Refine to optimize the reasoning path and utilizes a pairwise reward model to evaluate different paths globally. By leveraging the self-critic and rewriting capabilities of LLMs, Self-Refine applied to MCTS (SR-MCTS) overcomes the inefficiencies and limitations of conventional step-wise and greedy search algorithms by fostering a more efficient exploration of solution spaces. Pairwise Preference Reward Model~(PPRM), inspired by Reinforcement Learning from Human Feedback (RLHF), is then used to model pairwise preferences between solutions, utilizing an Enhanced Borda Count (EBC) method to synthesize these preferences into a global ranking score to find better answers. This approach addresses the challenges of scoring variability and non-independent distributions in mathematical reasoning tasks. The framework has been tested on general and advanced benchmarks, showing superior performance in terms of search efficiency and problem-solving capability compared to existing methods like ToT and rStar, particularly in complex Olympiad-level benchmarks, including GPQA, AIME24 and AMC23.
     </details>

157. **The Role of Deductive and Inductive Reasoning in Large Language Models** [[pdf]](https://arxiv.org/abs/2410.02892v1) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The Deductive and InDuctive method is proposed, which enhances LLM reasoning by dynamically integrating both deductive and inductive reasoning within the prompt construction process, and provides a more robust and cognitively aligned framework for reasoning in LLMs.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have achieved substantial progress in artificial intelligence, particularly in reasoning tasks. However, their reliance on static prompt structures, coupled with limited dynamic reasoning capabilities, often constrains their adaptability to complex and evolving problem spaces. In this paper, we propose the Deductive and InDuctive(DID) method, which enhances LLM reasoning by dynamically integrating both deductive and inductive reasoning within the prompt construction process. Drawing inspiration from cognitive science, the DID approach mirrors human adaptive reasoning mechanisms, offering a flexible framework that allows the model to adjust its reasoning pathways based on task context and performance. We empirically validate the efficacy of DID on established datasets such as AIW and MR-GSM8K, as well as on our custom dataset, Holiday Puzzle, which presents tasks about different holiday date calculating challenges. By leveraging DID's hybrid prompt strategy, we demonstrate significant improvements in both solution accuracy and reasoning quality, achieved without imposing substantial computational overhead. Our findings suggest that DID provides a more robust and cognitively aligned framework for reasoning in LLMs, contributing to the development of advanced LLM-driven problem-solving strategies informed by cognitive science models.
     </details>

158. **Unlocking Structured Thinking in Language Models with Cognitive Prompting** [[pdf]](https://arxiv.org/abs/2410.02953v1) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Cognitive prompting, particularly when dynamically adapted, significantly improves the performance of larger models, such as LLaMA3.1 70B, and enhances their ability to handle multi-step reasoning tasks, highlighting cognitive prompting as a promising strategy for general-purpose AI reasoning.
     </details>


     <details>
          <summary>Abstract</summary>
          We propose cognitive prompting as a novel approach to guide problem-solving in large language models (LLMs) through structured, human-like cognitive operations such as goal clarification, decomposition, filtering, abstraction, and pattern recognition. By employing systematic, step-by-step reasoning, cognitive prompting enables LLMs to efficiently tackle complex, multi-step tasks. We evaluate the effectiveness of cognitive prompting on Meta's LLaMA models, comparing performance on arithmetic reasoning tasks using the GSM8K dataset and on commonsense reasoning benchmarks. Our analysis includes comparisons between models without cognitive prompting, models with a static sequence of cognitive operations, and models using reflective cognitive prompting, where the LLM dynamically self-selects the sequence of cognitive operations. The results show that cognitive prompting, particularly when dynamically adapted, significantly improves the performance of larger models, such as LLaMA3.1 70B, and enhances their ability to handle multi-step reasoning tasks. This approach also improves interpretability and flexibility, highlighting cognitive prompting as a promising strategy for general-purpose AI reasoning.
     </details>

159. **Can We Further Elicit Reasoning in LLMs? Critic-Guided Planning with Retrieval-Augmentation for Solving Challenging Tasks** [[pdf]](http://arxiv.org/abs/2410.01428) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Critic-guided planning with Retrieval-augmentation, CR-Planner is introduced, a novel framework that leverages fine-tuned critic models to guide both reasoning and retrieval processes through planning that significantly outperforms baselines on challenging domain-knowledge-intensive and reasoning-heavy tasks.
     </details>


     <details>
          <summary>Abstract</summary>
          State-of-the-art large language models (LLMs) exhibit impressive problem-solving capabilities but may struggle with complex reasoning and factual correctness. Existing methods harness the strengths of chain-of-thought and retrieval-augmented generation (RAG) to decompose a complex problem into simpler steps and apply retrieval to improve factual correctness. These methods work well on straightforward reasoning tasks but often falter on challenging tasks such as competitive programming and mathematics, due to frequent reasoning errors and irrelevant knowledge retrieval. To address this, we introduce Critic-guided planning with Retrieval-augmentation, CR-Planner, a novel framework that leverages fine-tuned critic models to guide both reasoning and retrieval processes through planning. CR-Planner solves a problem by iteratively selecting and executing sub-goals. Initially, it identifies the most promising sub-goal from reasoning, query generation, and retrieval, guided by rewards given by a critic model named sub-goal critic. It then executes this sub-goal through sampling and selecting the optimal output based on evaluations from another critic model named execution critic. This iterative process, informed by retrieved information and critic models, enables CR-Planner to effectively navigate the solution space towards the final answer. We employ Monte Carlo Tree Search to collect the data for training the critic models, allowing for a systematic exploration of action sequences and their long-term impacts. We validate CR-Planner on challenging domain-knowledge-intensive and reasoning-heavy tasks, including competitive programming, theorem-driven math reasoning, and complex domain retrieval problems. Our experiments demonstrate that CR-Planner significantly outperforms baselines, highlighting its effectiveness in addressing challenging problems by improving both reasoning and retrieval.
     </details>

160. **CreDes: Causal Reasoning Enhancement and Dual-End Searching for Solving Long-Range Reasoning Problems using LLMs** [[pdf]](http://arxiv.org/abs/2410.01696) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          By integrating CRE and DES (CreDes), the model has realized simultaneous multi-step reasoning, circumventing the inefficiencies from cascading multiple one-step reasoning like the Chain-of-Thought (CoT).
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have demonstrated limitations in handling combinatorial optimization problems involving long-range reasoning, partially due to causal hallucinations and huge search space. As for causal hallucinations, i.e., the inconsistency between reasoning and corresponding state transition, this paper introduces the Causal Relationship Enhancement (CRE) mechanism combining cause-effect interventions and the Individual Treatment Effect (ITE) to guarantee the solid causal rightness between each step of reasoning and state transition. As for the long causal range and huge search space limiting the performances of existing models featuring single-direction search, a Dual-End Searching (DES) approach is proposed to seek solutions by simultaneously starting from both the initial and goal states on the causal probability tree. By integrating CRE and DES (CreDes), our model has realized simultaneous multi-step reasoning, circumventing the inefficiencies from cascading multiple one-step reasoning like the Chain-of-Thought (CoT). Experiments demonstrate that CreDes significantly outperforms existing State-Of-The-Art (SOTA) solutions in long-range reasoning tasks in terms of both accuracy and time efficiency.
     </details>

161. **Evaluating Robustness of Reward Models for Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2410.01729) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Reward models are key in reinforcement learning from human feedback (RLHF) systems, aligning the model behavior with human preferences. Particularly in the math domain, there have been plenty of studies using reward models to align policies for improving reasoning capabilities. Recently, as the importance of reward models has been emphasized, RewardBench is proposed to understand their behavior. However, we figure out that the math subset of RewardBench has different representations between chosen and rejected completions, and relies on a single comparison, which may lead to unreliable results as it only see an isolated case. Therefore, it fails to accurately present the robustness of reward models, leading to a misunderstanding of its performance and potentially resulting in reward hacking. In this work, we introduce a new design for reliable evaluation of reward models, and to validate this, we construct RewardMATH, a benchmark that effectively represents the robustness of reward models in mathematical reasoning tasks. We demonstrate that the scores on RewardMATH strongly correlate with the results of optimized policy and effectively estimate reward overoptimization, whereas the existing benchmark shows almost no correlation. The results underscore the potential of our design to enhance the reliability of evaluation, and represent the robustness of reward model. We make our code and data publicly available.
     </details>

162. **Interpretable Contrastive Monte Carlo Tree Search Reasoning** [[pdf]](http://arxiv.org/abs/2410.01707) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work conducted extensive ablation studies and quantitative analysis on components of MCTS, revealing the impact of each component on the MCTS reasoning performance of LLMs and designed a highly interpretable reward model based on the principle of contrastive decoding.
     </details>


     <details>
          <summary>Abstract</summary>
          We propose SC-MCTS*: a novel Monte Carlo Tree Search (MCTS) reasoning algorithm for Large Language Models (LLMs), significantly improves both reasoning accuracy and speed. Our motivation comes from: 1. Previous MCTS LLM reasoning works often overlooked its biggest drawback--slower speed compared to CoT; 2. Previous research mainly used MCTS as a tool for LLM reasoning on various tasks with limited quantitative analysis or ablation studies of its components from reasoning interpretability perspective. 3. The reward model is the most crucial component in MCTS, however previous work has rarely conducted in-depth study or improvement of MCTS's reward models. Thus, we conducted extensive ablation studies and quantitative analysis on components of MCTS, revealing the impact of each component on the MCTS reasoning performance of LLMs. Building on this, (i) we designed a highly interpretable reward model based on the principle of contrastive decoding and (ii) achieved an average speed improvement of 51.9% per node using speculative decoding. Additionally, (iii) we improved UCT node selection strategy and backpropagation used in previous works, resulting in significant performance improvement. We outperformed o1-mini by an average of 17.4% on the Blocksworld multi-step reasoning dataset using Llama-3.1-70B with SC-MCTS*.
     </details>

163. **Layer Swapping for Zero-Shot Cross-Lingual Transfer in Large Language Models** [[pdf]](http://arxiv.org/abs/2410.01335) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The ability to successfully re-compose LLMs for cross-lingual transfer in this manner opens up future possibilities to combine model expertise, create modular solutions, and transfer reasoning capabilities across languages all post hoc.
     </details>


     <details>
          <summary>Abstract</summary>
          Model merging, such as model souping, is the practice of combining different models with the same architecture together without further training. In this work, we present a model merging methodology that addresses the difficulty of fine-tuning Large Language Models (LLMs) for target tasks in non-English languages, where task-specific data is often unavailable. We focus on mathematical reasoning and without in-language math data, facilitate cross-lingual transfer by composing language and math capabilities. Starting from the same pretrained model, we fine-tune separate "experts" on math instruction data in English and on generic instruction data in the target language. We then replace the top and bottom transformer layers of the math expert directly with layers from the language expert, which consequently enhances math performance in the target language. The resulting merged models outperform the individual experts and other merging methods on the math benchmark, MGSM, by 10% across four major languages where math instruction data is scarce. In addition, this layer swapping is simple, inexpensive, and intuitive, as it is based on an interpretative analysis of the most important parameter changes during the fine-tuning of each expert. The ability to successfully re-compose LLMs for cross-lingual transfer in this manner opens up future possibilities to combine model expertise, create modular solutions, and transfer reasoning capabilities across languages all post hoc.
     </details>

164. **Not All LLM Reasoners Are Created Equal** [[pdf]](http://arxiv.org/abs/2410.01748) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Overall, LLMs exhibit systematic differences in their reasoning abilities, despite what their performance on standard benchmarks indicates, with a significant reasoning gap in smaller, more cost-efficient, and math-specialized models.
     </details>


     <details>
          <summary>Abstract</summary>
          We study the depth of grade-school math (GSM) problem-solving capabilities of LLMs. To this end, we evaluate their performance on pairs of existing math word problems together so that the answer to the second problem depends on correctly answering the first problem. Our findings reveal a significant reasoning gap in most LLMs, that is performance difference between solving the compositional pairs and solving each question independently. This gap is more pronounced in smaller, more cost-efficient, and math-specialized models. Moreover, instruction-tuning recipes and code generation have varying effects across LLM sizes, while finetuning on GSM can lead to task overfitting. Our analysis indicates that large reasoning gaps are not because of test-set leakage, but due to distraction from additional context and poor second-hop reasoning. Overall, LLMs exhibit systematic differences in their reasoning abilities, despite what their performance on standard benchmarks indicates.
     </details>

165. **OpenMathInstruct-2: Accelerating AI for Math with Massive Open-Source Instruction Data** [[pdf]](http://arxiv.org/abs/2410.01560) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The OpenMathInstruct-2 dataset is created, which consists of 14M question-solution pairs, making it nearly eight times larger than the previous largest open-source math reasoning dataset, and is released under a commercially permissive license.
     </details>


     <details>
          <summary>Abstract</summary>
          Mathematical reasoning continues to be a critical challenge in large language model (LLM) development with significant interest. However, most of the cutting-edge progress in mathematical reasoning with LLMs has become \emph{closed-source} due to lack of access to training data. This lack of data access limits researchers from understanding the impact of different choices for synthesizing and utilizing the data. With the goal of creating a high-quality finetuning (SFT) dataset for math reasoning, we conduct careful ablation experiments on data synthesis using the recently released \texttt{Llama3.1} family of models. Our experiments show that: (a) solution format matters, with excessively verbose solutions proving detrimental to SFT performance, (b) data generated by a strong teacher outperforms equally-sized data generated by a weak student model, (c) SFT is robust to low-quality solutions, allowing for imprecise data filtering, and (d) question diversity is crucial for achieving data scaling gains. Based on these insights, we create the OpenMathInstruct-2 dataset, which consists of 14M question-solution pairs ($\approx$ 600K unique questions), making it nearly eight times larger than the previous largest open-source math reasoning dataset. Finetuning the \texttt{Llama-3.1-8B-Base} using OpenMathInstruct-2 outperforms \texttt{Llama3.1-8B-Instruct} on MATH by an absolute 15.9\% (51.9\% $\rightarrow$ 67.8\%). Finally, to accelerate the open-source efforts, we release the code, the finetuned models, and the OpenMathInstruct-2 dataset under a commercially permissive license.
     </details>

166. **PersonaMath: Enhancing Math Reasoning through Persona-Driven Data Augmentation** [[pdf]](http://arxiv.org/abs/2410.01504) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a data augmentation approach and introduces PersonaMathQA, a dataset derived from MATH and GSM8K, on which the PersonaMath models are trained, and introduces a novel persona-driven data augmentation technique to enhance the dataset's quantity and diversity.
     </details>


     <details>
          <summary>Abstract</summary>
          While closed-source Large Language Models (LLMs) demonstrate strong mathematical problem-solving abilities, open-source models continue to struggle with such tasks. To bridge this gap, we propose a data augmentation approach and introduce PersonaMathQA, a dataset derived from MATH and GSM8K, on which we train the PersonaMath models. Our approach consists of two stages: the first stage is learning from Persona Diversification, and the second stage is learning from Reflection. In the first stage, we regenerate detailed chain-of-thought (CoT) solutions as instructions using a closed-source LLM and introduce a novel persona-driven data augmentation technique to enhance the dataset's quantity and diversity. In the second stage, we incorporate reflection to fully leverage more challenging and valuable questions. Evaluation of our PersonaMath models on MATH and GSM8K reveals that the PersonaMath-7B model (based on LLaMA-2-7B) achieves an accuracy of 24.2% on MATH and 68.7% on GSM8K, surpassing all baseline methods and achieving state-of-the-art performance. Notably, our dataset contains only 70.3K data points-merely 17.8% of MetaMathQA and 27% of MathInstruct-yet our model outperforms these baselines, demonstrating the high quality and diversity of our dataset, which enables more efficient model training. We open-source the PersonaMathQA dataset, PersonaMath models, and our code for public usage.
     </details>

167. **ReGenesis: LLMs can Grow into Reasoning Generalists via Self-Improvement** [[pdf]](http://arxiv.org/abs/2410.02108) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          ReGenesis self-synthesizes reasoning paths by converting general reasoning guidelines into task-specific ones, generating reasoning structures, and subsequently transforming these structures into reasoning paths, without the need for human-designed task-specific examples used in existing methods.
     </details>


     <details>
          <summary>Abstract</summary>
          Post-training Large Language Models (LLMs) with explicit reasoning trajectories can enhance their reasoning abilities. However, acquiring such high-quality trajectory data typically demands meticulous supervision from humans or superior models, which can be either expensive or license-constrained. In this paper, we explore how far an LLM can improve its reasoning by self-synthesizing reasoning paths as training data without any additional supervision. Existing self-synthesizing methods, such as STaR, suffer from poor generalization to out-of-domain (OOD) reasoning tasks. We hypothesize it is due to that their self-synthesized reasoning paths are too task-specific, lacking general task-agnostic reasoning guidance. To address this, we propose Reasoning Generalist via Self-Improvement (ReGenesis), a method to self-synthesize reasoning paths as post-training data by progressing from abstract to concrete. More specifically, ReGenesis self-synthesizes reasoning paths by converting general reasoning guidelines into task-specific ones, generating reasoning structures, and subsequently transforming these structures into reasoning paths, without the need for human-designed task-specific examples used in existing methods. We show that ReGenesis achieves superior performance on all in-domain and OOD settings tested compared to existing methods. For six OOD tasks specifically, while previous methods exhibited an average performance decrease of approximately 4.6% after post training, ReGenesis delivers around 6.1% performance improvement. We also conduct in-depth analysis of our framework and show ReGenesis is effective across various LLMs and design choices.
     </details>

168. **Reasoning Elicitation in Language Models via Counterfactual Feedback** [[pdf]](http://arxiv.org/abs/2410.03767) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work derives novel metrics that balance accuracy in factual and counterfactual questions, capturing a more complete view of the reasoning abilities of language models than traditional factual-only based metrics.
     </details>


     <details>
          <summary>Abstract</summary>
          Despite the increasing effectiveness of language models, their reasoning capabilities remain underdeveloped. In particular, causal reasoning through counterfactual question answering is lacking. This work aims to bridge this gap. We first derive novel metrics that balance accuracy in factual and counterfactual questions, capturing a more complete view of the reasoning abilities of language models than traditional factual-only based metrics. Second, we propose several fine-tuning approaches that aim to elicit better reasoning mechanisms, in the sense of the proposed metrics. Finally, we evaluate the performance of the fine-tuned language models in a variety of realistic scenarios. In particular, we investigate to what extent our fine-tuning approaches systemically achieve better generalization with respect to the base models in several problems that require, among others, inductive and deductive reasoning capabilities.
     </details>

169. **Step-by-Step Reasoning for Math Problems via Twisted Sequential Monte Carlo** [[pdf]](http://arxiv.org/abs/2410.01920) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduces a novel verification method based on Twisted Sequential Monte Carlo (TSMC), which sequentially refines its sampling effort to focus exploration on promising candidates, resulting in more efficient generation of high-quality solutions.
     </details>


     <details>
          <summary>Abstract</summary>
          Augmenting the multi-step reasoning abilities of Large Language Models (LLMs) has been a persistent challenge. Recently, verification has shown promise in improving solution consistency by evaluating generated outputs. However, current verification approaches suffer from sampling inefficiencies, requiring a large number of samples to achieve satisfactory performance. Additionally, training an effective verifier often depends on extensive process supervision, which is costly to acquire. In this paper, we address these limitations by introducing a novel verification method based on Twisted Sequential Monte Carlo (TSMC). TSMC sequentially refines its sampling effort to focus exploration on promising candidates, resulting in more efficient generation of high-quality solutions. We apply TSMC to LLMs by estimating the expected future rewards at partial solutions. This approach results in a more straightforward training target that eliminates the need for step-wise human annotations. We empirically demonstrate the advantages of our method across multiple math benchmarks, and also validate our theoretical analysis of both our approach and existing verification methods.
     </details>

170. **TypedThinker: Typed Thinking Improves Large Language Model Reasoning** [[pdf]](http://arxiv.org/abs/2410.01952) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          TypedThinker is a novel framework that enhances LLMs' problem-solving abilities by incorporating multiple reasoning types (deductive, inductive, abductive, and analogical) and shows effective generalization to new benchmarks and can further enhance the reasoning capability of powerful models like GPT-4o.
     </details>


     <details>
          <summary>Abstract</summary>
          Despite significant advancements in the reasoning capabilities of Large Language Models (LLMs), the lack of diverse reasoning solutions often makes them trapped in a limited solution search area. In this paper, we propose TypedThinker, a novel framework that enhances LLMs' problem-solving abilities by incorporating multiple reasoning types (deductive, inductive, abductive, and analogical). Our analysis across four benchmarks reveals that different reasoning types uniquely solve distinct sets of problems, highlighting the importance of diverse thinking approaches. TypedThinker addresses two key challenges: selecting appropriate reasoning types for given problems and effectively implementing specific reasoning types. Through self-training on successful experiences, TypedThinker learns an implicit policy for reasoning type selection and application. Experimental results demonstrate significant improvements over baseline models, with accuracy increases of 3.4% for Mistral 7B and 16.7% for LLaMA3 8B across four reasoning benchmarks. Notably, TypedThinker shows effective generalization to new benchmarks and can further enhance the reasoning capability of powerful models like GPT-4o. The code is released at https://github.com/dqwang122/ThinkHub.
     </details>

171. **VinePPO: Unlocking RL Potential For LLM Reasoning Through Refined Credit Assignment** [[pdf]](http://arxiv.org/abs/2410.01679) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          VinePPO is proposed, a straightforward approach that leverages the flexibility of language environments to compute unbiased Monte Carlo-based estimates, bypassing the need for large value networks, and consistently outperforms PPO and other RL-free baselines across MATH and GSM8K datasets.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) are increasingly applied to complex reasoning tasks that require executing several complex steps before receiving any reward. Properly assigning credit to these steps is essential for enhancing model performance. Proximal Policy Optimization (PPO), a state-of-the-art reinforcement learning (RL) algorithm used for LLM finetuning, employs value networks to tackle credit assignment. However, value networks face challenges in predicting the expected cumulative rewards accurately in complex reasoning tasks, often leading to high-variance updates and suboptimal performance. In this work, we systematically evaluate the efficacy of value networks and reveal their significant shortcomings in reasoning-heavy LLM tasks, showing that they barely outperform a random baseline when comparing alternative steps. To address this, we propose VinePPO, a straightforward approach that leverages the flexibility of language environments to compute unbiased Monte Carlo-based estimates, bypassing the need for large value networks. Our method consistently outperforms PPO and other RL-free baselines across MATH and GSM8K datasets with fewer gradient updates (up to 9x), less wall-clock time (up to 3.0x). These results emphasize the importance of accurate credit assignment in RL finetuning of LLM and demonstrate VinePPO's potential as a superior alternative.
     </details>

172. **When a language model is optimized for reasoning, does it still show embers of autoregression? An analysis of OpenAI o1** [[pdf]](http://arxiv.org/abs/2410.01792) `2024-10-02` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is found that o1 substantially outperforms previous LLMs in many cases, with particularly large improvements on rare variants of common tasks, and shows that optimizing a language model for reasoning can mitigate but might not fully overcome the language model's probability sensitivity.
     </details>


     <details>
          <summary>Abstract</summary>
          In "Embers of Autoregression" (McCoy et al., 2023), we showed that several large language models (LLMs) have some important limitations that are attributable to their origins in next-word prediction. Here we investigate whether these issues persist with o1, a new system from OpenAI that differs from previous LLMs in that it is optimized for reasoning. We find that o1 substantially outperforms previous LLMs in many cases, with particularly large improvements on rare variants of common tasks (e.g., forming acronyms from the second letter of each word in a list, rather than the first letter). Despite these quantitative improvements, however, o1 still displays the same qualitative trends that we observed in previous systems. Specifically, o1 -- like previous LLMs -- is sensitive to the probability of examples and tasks, performing better and requiring fewer "thinking tokens" in high-probability settings than in low-probability ones. These results show that optimizing a language model for reasoning can mitigate but might not fully overcome the language model's probability sensitivity.
     </details>

173. **RATIONALYST: Pre-training Process-Supervision for Improving Reasoning** [[pdf]](http://arxiv.org/abs/2410.01044) `2024-10-01` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          RATIONALYST, a model for process-supervision of reasoning based on pre-training on a vast collection of rationale annotations extracted from unlabeled data, is introduced, which demonstrates superior performance compared to significantly larger verifiers like GPT-4 and similarly sized models fine-tuned on matching training sets.
     </details>


     <details>
          <summary>Abstract</summary>
          The reasoning steps generated by LLMs might be incomplete, as they mimic logical leaps common in everyday communication found in their pre-training data: underlying rationales are frequently left implicit (unstated). To address this challenge, we introduce RATIONALYST, a model for process-supervision of reasoning based on pre-training on a vast collection of rationale annotations extracted from unlabeled data. We extract 79k rationales from web-scale unlabelled dataset (the Pile) and a combination of reasoning datasets with minimal human intervention. This web-scale pre-training for reasoning allows RATIONALYST to consistently generalize across diverse reasoning tasks, including mathematical, commonsense, scientific, and logical reasoning. Fine-tuned from LLaMa-3-8B, RATIONALYST improves the accuracy of reasoning by an average of 3.9% on 7 representative reasoning benchmarks. It also demonstrates superior performance compared to significantly larger verifiers like GPT-4 and similarly sized models fine-tuned on matching training sets.
     </details>

174. **Scheherazade: Evaluating Chain-of-Thought Math Reasoning in LLMs with Chain-of-Problems** [[pdf]](http://arxiv.org/abs/2410.00151) `2024-09-30` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Benchmarks are critical for measuring progress of math reasoning abilities of Large Language Models (LLMs). However, existing widely-used benchmarks such as GSM8K have been rendered less useful as multiple cutting-edge LLMs achieve over 94% accuracy. While harder benchmarks have been proposed, their creation is often manual and expensive. We present Scheherazade, an automated approach for producing challenging mathematical reasoning benchmarks by logically chaining mathematical reasoning problems. We propose two different chaining methods, forward chaining and backward chaining, which require reasoning forward and backward through the chain respectively. We apply Scheherazade on GSM8K to create GSM8K-Scheherazade and evaluate 3 frontier LLMs and OpenAI's o1-preview on it. We show that while frontier models' performance declines precipitously at only a few questions chained, a preliminary evaluation suggests o1-preview performance persists up to 5 questions chained backwards. In addition, while all other models perform worse when problems are chained backwards, o1-preview performs better on backward-chained benchmarks. We will release the dataset and code publicly.
     </details>

175. **The Perfect Blend: Redefining RLHF with Mixture of Judges** [[pdf]](http://arxiv.org/abs/2409.20370) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel post-training paradigm which can detect and mitigate reward hacking behaviors while reaching a pareto-optimal point across an extremely large number of objectives is introduced, called Constrained Generative Policy Optimization (CGPO).
     </details>


     <details>
          <summary>Abstract</summary>
          Reinforcement learning from human feedback (RLHF) has become the leading approach for fine-tuning large language models (LLM). However, RLHF has limitations in multi-task learning (MTL) due to challenges of reward hacking and extreme multi-objective optimization (i.e., trade-off of multiple and/or sometimes conflicting objectives). Applying RLHF for MTL currently requires careful tuning of the weights for reward model and data combinations. This is often done via human intuition and does not generalize. In this work, we introduce a novel post-training paradigm which we called Constrained Generative Policy Optimization (CGPO). The core of CGPO is Mixture of Judges (MoJ) with cost-efficient constrained policy optimization with stratification, which can identify the perfect blend in RLHF in a principled manner. It shows strong empirical results with theoretical guarantees, does not require extensive hyper-parameter tuning, and is plug-and-play in common post-training pipelines. Together, this can detect and mitigate reward hacking behaviors while reaching a pareto-optimal point across an extremely large number of objectives.   Our empirical evaluations demonstrate that CGPO significantly outperforms standard RLHF algorithms like PPO and DPO across various tasks including general chat, STEM questions, instruction following, and coding. Specifically, CGPO shows improvements of 7.4% in AlpacaEval-2 (general chat), 12.5% in Arena-Hard (STEM & reasoning), and consistent gains in other domains like math and coding. Notably, PPO, while commonly used, is prone to severe reward hacking in popular coding benchmarks, which CGPO successfully addresses. This breakthrough in RLHF not only tackles reward hacking and extreme multi-objective optimization challenges but also advances the state-of-the-art in aligning general-purpose LLMs for diverse applications.
     </details>

176. **BEATS: Optimizing LLM Mathematical Capabilities with BackVerify and Adaptive Disambiguate based Efficient Tree Search** [[pdf]](http://arxiv.org/abs/2409.17972) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a novel approach, BEATS, to enhance mathematical problem-solving abilities of large Language Models, which leverages newly designed prompts that guide the model to iteratively rewrite, advance by one step, and generate answers based on previous steps.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have exhibited exceptional performance across a broad range of tasks and domains. However, they still encounter difficulties in solving mathematical problems due to the rigorous and logical nature of mathematics. Previous studies have employed techniques such as supervised fine-tuning (SFT), prompt engineering, and search-based methods to improve the mathematical problem-solving abilities of LLMs. Despite these efforts, their performance remains suboptimal and demands substantial computational resources. To address this issue, we propose a novel approach, BEATS, to enhance mathematical problem-solving abilities. Our method leverages newly designed prompts that guide the model to iteratively rewrite, advance by one step, and generate answers based on previous steps. Additionally, we introduce a new back-verification technique that uses LLMs to validate the correctness of the generated answers. Furthermore, we employ a pruning tree search to optimize search time while achieving strong performance. Notably, our method improves Qwen2-7b-Instruct's score from 36.94 to 61.52, outperforming GPT4's 42.5 on the MATH benchmark.
     </details>

177. **Learning to Love Edge Cases in Formative Math Assessment: Using the AMMORE Dataset and Chain-of-Thought Prompting to Improve Grading Accuracy** [[pdf]](http://arxiv.org/abs/2409.17904) `2024-09-26` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          AMMORE, a new dataset of 53,000 math open-response question-answer pairs from Rori, is introduced and two experiments to evaluate the use of large language models (LLM) for grading particularly challenging student answers suggest that LLMs could be a valuable tool for grading open-response questions in K-12 mathematics education.
     </details>


     <details>
          <summary>Abstract</summary>
          This paper introduces AMMORE, a new dataset of 53,000 math open-response question-answer pairs from Rori, a learning platform used by students in several African countries and conducts two experiments to evaluate the use of large language models (LLM) for grading particularly challenging student answers. The AMMORE dataset enables various potential analyses and provides an important resource for researching student math acquisition in understudied, real-world, educational contexts. In experiment 1 we use a variety of LLM-driven approaches, including zero-shot, few-shot, and chain-of-thought prompting, to grade the 1% of student answers that a rule-based classifier fails to grade accurately. We find that the best-performing approach -- chain-of-thought prompting -- accurately scored 92% of these edge cases, effectively boosting the overall accuracy of the grading from 98.7% to 99.9%. In experiment 2, we aim to better understand the consequential validity of the improved grading accuracy, by passing grades generated by the best-performing LLM-based approach to a Bayesian Knowledge Tracing (BKT) model, which estimated student mastery of specific lessons. We find that relatively modest improvements in model accuracy at the individual question level can lead to significant changes in the estimation of student mastery. Where the rules-based classifier currently used to grade student, answers misclassified the mastery status of 6.9% of students across their completed lessons, using the LLM chain-of-thought approach this misclassification rate was reduced to 2.6% of students. Taken together, these findings suggest that LLMs could be a valuable tool for grading open-response questions in K-12 mathematics education, potentially enabling encouraging wider adoption of open-ended questions in formative assessment.
     </details>

178. **SciDFM: A Large Language Model with Mixture-of-Experts for Science** [[pdf]](http://arxiv.org/abs/2409.18412) `2024-09-26` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          SciDFM, a mixture-of-experts LLM, is introduced, which is trained from scratch and is able to conduct college-level scientific reasoning and understand molecules and amino acid sequences and reaches a SOTA performance on domain-specific benchmarks among models of similar size.
     </details>


     <details>
          <summary>Abstract</summary>
          Recently, there has been a significant upsurge of interest in leveraging large language models (LLMs) to assist scientific discovery. However, most LLMs only focus on general science, while they lack domain-specific knowledge, such as chemical molecules and amino acid sequences. To bridge these gaps, we introduce SciDFM, a mixture-of-experts LLM, which is trained from scratch and is able to conduct college-level scientific reasoning and understand molecules and amino acid sequences. We collect a large-scale training corpus containing numerous scientific papers and books from different disciplines as well as data from domain-specific databases. We further fine-tune the pre-trained model on lots of instruction data to improve performances on downstream benchmarks. From experiment results, we show that SciDFM achieves strong performance on general scientific benchmarks such as SciEval and SciQ, and it reaches a SOTA performance on domain-specific benchmarks among models of similar size. We further analyze the expert layers and show that the results of expert selection vary with data from different disciplines. To benefit the broader research community, we open-source SciDFM at https://huggingface.co/OpenDFM/SciDFM-MoE-A5.6B-v1.0.
     </details>

179. **Looped Transformers for Length Generalization** [[pdf]](http://arxiv.org/abs/2409.15647) `NeurIPS 2024 Workshop MATH-AI` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work demonstrates that looped Transformers with an adaptive number of steps significantly improve length generalization, and trains looped Transformers using the proposed learning algorithm and observes that they learn highly length-generalizable solutions for various tasks.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent work has shown that Transformers trained from scratch can successfully solve various arithmetic and algorithmic tasks, such as adding numbers and computing parity. While these Transformers generalize well on unseen inputs of the same length, they struggle with length generalization, i.e., handling inputs of unseen lengths. In this work, we demonstrate that looped Transformers with an adaptive number of steps significantly improve length generalization. We focus on tasks with a known iterative solution, involving multiple iterations of a RASP-L operation - a length-generalizable operation that can be expressed by a finite-sized Transformer. We train looped Transformers using our proposed learning algorithm and observe that they learn highly length-generalizable solutions for various tasks.
     </details>

180. **HDFlow: Enhancing LLM Complex Problem-Solving with Hybrid Thinking and Dynamic Workflows** [[pdf]](http://arxiv.org/abs/2409.17433) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel framework HDFlow for complex reasoning with LLMs that combines fast and slow thinking modes in an adaptive manner and hybrid thinking achieves the highest accuracy while providing an effective balance between computational efficiency and performance is proposed.
     </details>


     <details>
          <summary>Abstract</summary>
          Despite recent advancements in large language models (LLMs), their performance on complex reasoning problems requiring multi-step thinking and combining various skills is still limited. To address this, we propose a novel framework HDFlow for complex reasoning with LLMs that combines fast and slow thinking modes in an adaptive manner. Our approach consists of two key components: 1) a new approach for slow, deliberate reasoning called Dynamic Workflow, which automatically decomposes complex problems into more manageable sub-tasks and dynamically designs a workflow to assemble specialized LLM or symbolic reasoning tools to solve sub-tasks; 2) Hybrid Thinking, a general framework that dynamically combines fast and slow thinking based on problem complexity. Finally, we propose an easy-to-scale method for automatically synthesizing a large-scale dataset of 27K challenging reasoning problems for complex reasoning and a hybrid thinking tuning method that trains smaller LLMs on this dataset to internalize the fast/slow hybrid reasoning strategies. Experiments on four reasoning benchmark datasets demonstrate that our slow thinking with dynamic workflows significantly outperforms Chain-of-Thought, and hybrid thinking achieves the highest accuracy while providing an effective balance between computational efficiency and performance. Fine-tuning using our hybrid thinking approach also significantly boosts the complex reasoning capabilities of open-source language models. The results showcase the promise of slow thinking, dynamic workflows, and hybrid thinking in expanding the frontier of complex problem-solving with LLMs\footnote{Code and data will be released at \url{https://github.com/wenlinyao/HDFlow}.}.
     </details>

181. **LLaMa-SciQ: An Educational Chatbot for Answering Science MCQ** [[pdf]](http://arxiv.org/abs/2409.16779) `2024-09-25` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) often struggle with tasks requiring mathematical reasoning, particularly multiple-choice questions (MCQs). To address this issue, we developed LLaMa-SciQ, an educational chatbot designed to assist college students in solving and understanding MCQs in STEM fields. We begin by fine-tuning and aligning the models to human preferences. After comparing the performance of Mistral-7B and LLaMa-8B, we selected the latter as the base model due to its higher evaluation accuracy. To further enhance accuracy, we implement Retrieval-Augmented Generation (RAG) and apply quantization to compress the model, reducing inference time and increasing accessibility for students. For mathematical reasoning, LLaMa-SciQ achieved 74.5% accuracy on the GSM8k dataset and 30% on the MATH dataset. However, RAG does not improve performance and even reduces it, likely due to retriever issues or the model's unfamiliarity with context. Despite this, the quantized model shows only a 5% loss in performance, demonstrating significant efficiency improvements.
     </details>

182. **MathDSL: A Domain-Specific Language for Concise Mathematical Solutions Via Program Synthesis** [[pdf]](http://arxiv.org/abs/2409.17490) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is demonstrated that a program synthesis system (DreamCoder) using MathDSL can generate programs that solve linear equations with greater accuracy and conciseness than using reinforcement learning systems.
     </details>


     <details>
          <summary>Abstract</summary>
          We present MathDSL, a Domain-Specific Language (DSL) for mathematical equation solving, which, when deployed in program synthesis models, outperforms state-of-the-art reinforcement-learning-based methods. We also introduce a quantitative metric for measuring the conciseness of a mathematical solution and demonstrate the improvement in the quality of generated solutions compared to other methods. Our system demonstrates that a program synthesis system (DreamCoder) using MathDSL can generate programs that solve linear equations with greater accuracy and conciseness than using reinforcement learning systems. Additionally, we demonstrate that if we use the action spaces of previous reinforcement learning systems as DSLs, MathDSL outperforms the action-space-DSLs. We use DreamCoder to store equation-solving strategies as learned abstractions in its program library and demonstrate that by using MathDSL, these can be converted into human-interpretable solution strategies that could have applications in mathematical education.
     </details>

183. **Models Can and Should Embrace the Communicative Nature of Human-Generated Math** [[pdf]](http://arxiv.org/abs/2409.17005) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Math is constructed by people for people: just as natural language corpora reflect not just propositions but the communicative goals of language users, the math data that models are trained on reflects not just idealized mathematical entities but rich communicative intentions. While there are important advantages to treating math in a purely symbolic manner, we here hypothesize that there are benefits to treating math as situated linguistic communication and that language models are well suited for this goal, in ways that are not fully appreciated. We illustrate these points with two case studies. First, we ran an experiment in which we found that language models interpret the equals sign in a humanlike way -- generating systematically different word problems for the same underlying equation arranged in different ways. Second, we found that language models prefer proofs to be ordered in naturalistic ways, even though other orders would be logically equivalent. We advocate for AI systems that learn from and represent the communicative intentions latent in human-generated math.
     </details>

184. **Proof Automation with Large Language Models** [[pdf]](https://arxiv.org/abs/2409.14274v1) `2024-09-22` `Coq` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          PALM is a novel generate-then-repair approach that first prompts an LLM to generate an initial proof and then leverages targeted symbolic methods to iteratively repair low-level problems, and significantly outperforms other state-of-the-art approaches.
     </details>


     <details>
          <summary>Abstract</summary>
          Interactive theorem provers such as Coq are powerful tools to formally guarantee the correctness of software. However, using these tools requires significant manual effort and expertise. While Large Language Models (LLMs) have shown promise in automatically generating informal proofs in natural language, they are less effective at generating formal proofs in interactive theorem provers. In this paper, we conduct a formative study to identify common mistakes made by LLMs when asked to generate formal proofs. By analyzing 520 proof generation errors made by GPT-3.5, we found that GPT-3.5 often identified the correct high-level structure of a proof, but struggled to get the lower-level details correct. Based on this insight, we propose PALM, a novel generate-then-repair approach that first prompts an LLM to generate an initial proof and then leverages targeted symbolic methods to iteratively repair low-level problems. We evaluate PALM on a large dataset that includes more than 10K theorems. Our results show that PALM significantly outperforms other state-of-the-art approaches, successfully proving 76.6% to 180.4% more theorems. Moreover, PALM proves 1270 theorems beyond the reach of existing approaches. We also demonstrate the generalizability of PALM across different LLMs.
     </details>

185. **GroupDebate: Enhancing the Efficiency of Multi-Agent Debate Using Group Discussion** [[pdf]](http://arxiv.org/abs/2409.14051) `2024-09-21` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A method to significantly reduce token cost in multi-agent debates by dividing all agents into multiple debate groups, with agents engaging in debates within their respective groups and sharing interim debate results between groups.
     </details>


     <details>
          <summary>Abstract</summary>
          In recent years, Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse NLP tasks. Extensive research has explored how to enhance the logical reasoning abilities such as Chain-of-Thought, Chain-of-Thought with Self-Consistency, Tree-Of-Thoughts, and multi-agent debates. In the context of multi-agent debates, significant performance improvements can be achieved with an increasing number of agents and debate rounds. However, the escalation in the number of agents and debate rounds can drastically raise the tokens cost of debates, thereby limiting the scalability of the multi-agent debate technique. To better harness the advantages of multi-agent debates in logical reasoning tasks, this paper proposes a method to significantly reduce token cost in multi-agent debates. This approach involves dividing all agents into multiple debate groups, with agents engaging in debates within their respective groups and sharing interim debate results between groups. Comparative experiments across multiple datasets have demonstrated that this method can reduce the total tokens by up to 51.7% during debates and while potentially enhancing accuracy by as much as 25%. Our method significantly enhances the performance and efficiency of interactions in the multi-agent debate.
     </details>

186. **Uncovering Latent Chain of Thought Vectors in Language Models** [[pdf]](http://arxiv.org/abs/2409.14026) `2024-09-21` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work investigates the technique of steering vectors: biasing the forward pass of language models using a steering vector derived from a specific task to steer language models toward performing Chain of Thought (CoT) Reasoning without the need to prompt through natural language.
     </details>


     <details>
          <summary>Abstract</summary>
          As language models grow more influential and trusted in our society, our ability to reliably steer them toward favorable behaviors becomes increasingly paramount. For this, we investigate the technique of steering vectors: biasing the forward pass of language models using a "steering vector" derived from a specific task. We apply them to steer language models toward performing Chain of Thought (CoT) Reasoning without the need to prompt through natural language. We demonstrate this approach on Llama3 8b and Mistral 7b v0.2, and obtain competitive results compared to CoT-prompted performances on a series of reasoning benchmarks (GSM8k, MMLU, AGI Eval, ARC AI2) and qualitative examples. We find this approach yields consistent steering towards CoT responses and takes less compute than traditional methods of fine-tuning models towards CoT.
     </details>

187. **CSCE: Boosting LLM Reasoning by Simultaneous Enhancing of Casual Significance and Consistency** [[pdf]](http://arxiv.org/abs/2409.17174) `2024-09-20` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A non-chain-based reasoning framework for simultaneous consideration of causal significance and consistency, i.e., the Causal Significance and Consistency Enhancer (CSCE) is proposed and extensive experiments show that this method improves both the reasoning success rate and speed.
     </details>


     <details>
          <summary>Abstract</summary>
          Chain-based reasoning methods like chain of thought (CoT) play a rising role in solving reasoning tasks for large language models (LLMs). However, the causal illusions between \textit{a step of reasoning} and \textit{corresponding state transitions} are becoming a significant obstacle to advancing LLMs' reasoning capabilities, especially in long-range reasoning tasks. This paper proposes a non-chain-based reasoning framework for simultaneous consideration of causal significance and consistency, i.e., the Causal Significance and Consistency Enhancer (CSCE). We customize LLM's loss function utilizing treatment effect assessments to enhance its reasoning ability from two aspects: causal significance and consistency. This ensures that the model captures essential causal relationships and maintains robust and consistent performance across various scenarios. Additionally, we transform the reasoning process from the cascading multiple one-step reasoning commonly used in Chain-Based methods, like CoT, to a causal-enhanced method that outputs the entire reasoning process in one go, further improving the model's reasoning efficiency. Extensive experiments show that our method improves both the reasoning success rate and speed. These improvements further demonstrate that non-chain-based methods can also aid LLMs in completing reasoning tasks.
     </details>

188. **Neural-Symbolic Collaborative Distillation: Advancing Small Language Models for Complex Reasoning Tasks** [[pdf]](http://arxiv.org/abs/2409.13203) `2024-09-20` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The proposed NesyCD can achieve superior performance cost-effectively, utilizing smaller models and blending parameterized neural networks with symbolic KB, which generalizes well and is comprehended and manipulated by humans.
     </details>


     <details>
          <summary>Abstract</summary>
          In this paper, we propose $\textbf{Ne}$ural-$\textbf{Sy}$mbolic $\textbf{C}$ollaborative $\textbf{D}$istillation ($\textbf{NesyCD}$), a novel knowledge distillation method for learning the complex reasoning abilities of Large Language Models (LLMs, e.g., \textgreater 13B). We argue that complex reasoning tasks are difficult for Small Language Models (SLMs, e.g., $\leq$ 7B), as these tasks demand not only general cognitive abilities but also specialized knowledge, which is often sparse and difficult for these neural-based SLMs to effectively capture. Therefore, NesyCD distills the general capabilities and specialized knowledge in LLMs using different manners. On the one hand, we distill only general abilities from teacher LLMs into the student SLMs of parameterized neural networks. On the other hand, for the specialized abilities and uncommon knowledge of a complex reasoning task, we employ a symbolic knowledge distillation approach to obtain and store the specialized knowledge within a symbolic knowledge base (KB). By decoupling general and specialized capabilities, the proposed NesyCD can achieve superior performance cost-effectively, utilizing smaller models and blending parameterized neural networks with symbolic KB. Moreover, the specialized KB generalizes well and is comprehended and manipulated by humans. Our experiments show that NesyCD significantly boosts SLMs' complex reasoning performance on in-domain (BBH, GSM8K) and out-of-domain (AGIEval, ARC) datasets. Notably, our approach enabled the LLaMA3-8B and Qwen2-7B to surpass GPT-3.5-turbo in performance and come close to matching LLaMA3-70B, despite the latter having nine times more parameters. Our code will be available at https://github.com/Xnhyacinth/NesyCD.
     </details>

189. **Training Language Models to Self-Correct via Reinforcement Learning** [[pdf]](http://arxiv.org/abs/2409.12917) `ICLR 2025 Submission` (9 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A multi-turn online reinforcement learning (RL) approach that significantly improves an LLM's self-correction ability using entirely self-generated data, and uses appropriate regularization to steer the learning process into learning a self-correction behavior that is effective at test time as opposed to fitting high-reward responses for a given prompt.
     </details>


     <details>
          <summary>Abstract</summary>
          Self-correction is a highly desirable capability of large language models (LLMs), yet it has consistently been found to be largely ineffective in modern LLMs. Existing approaches for training self-correction either require multiple models or rely on a more capable model or other forms of supervision. To this end, we develop a multi-turn online reinforcement learning (RL) approach, SCoRe, that significantly improves an LLM's self-correction ability using entirely self-generated data. To build SCoRe, we first show that variants of supervised fine-tuning (SFT) on offline model-generated correction traces are insufficient for instilling self-correction behavior. In particular, we observe that training via SFT either suffers from a distribution mismatch between the training data and the model's own responses or implicitly prefers only a certain mode of correction behavior that is often not effective at test time. SCoRe addresses these challenges by training under the model's own distribution of self-generated correction traces and using appropriate regularization to steer the learning process into learning a self-correction strategy that is effective at test time as opposed to simply fitting high-reward responses for a given prompt. This regularization prescribes running a first phase of RL on a base model to generate a policy initialization that is less susceptible to collapse and then using a reward bonus to amplify self-correction during training. When applied to Gemini 1.0 Pro and 1.5 Flash models, we find that SCoRe achieves state-of-the-art self-correction performance, improving the base models' self-correction by 15.6% and 9.1% respectively on the MATH and HumanEval benchmarks.
     </details>

190. **CodePlan: Unlocking Reasoning Potential in Large Langauge Models by Scaling Code-form Planning** [[pdf]](http://arxiv.org/abs/2409.12452) `2024-09-19` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Despite the remarkable success of large language models (LLMs) on traditional natural language processing tasks, their planning ability remains a critical bottleneck in tackling complex multi-step reasoning tasks. Existing approaches mainly rely on prompting or task-specific fine-tuning, often suffering from weak robustness and cross-task generalization. To address the limitation, we introduce CODEPLAN, a scalable paradigm that empowers LLMs to generate and follow code-form plans pseudocode that outlines high-level, structured reasoning processes. By leveraging the structured and versatile nature of code, CODEPLAN effectively captures the rich semantics and control flows inherent to sophisticated reasoning. Importantly, CODEPLAN allows the automatic extraction of code-form plans from massive, wide-ranging text corpora without the need for curated, task-specific datasets. This enables it to scale up efficiently and improve reasoning capabilities across diverse scenarios. To train CODEPLAN, we construct a large-scale dataset of 2M examples that integrate code-form plans with standard prompt-response pairs from existing corpora. With minimal computation overhead during both training and inference, CODEPLAN achieves a 25.1% relative improvement compared with directly generating responses, averaged across 13 challenging multi-step reasoning benchmarks, spanning mathematical reasoning, symbolic reasoning, instruction-following, multi-hop QA, and decision-making tasks. Further analysis reveals CODEPLAN's increasing performance gains on more complex reasoning tasks, as well as significant data efficiency thanks to its generalization ability.
     </details>

191. **ControlMath: Controllable Data Generation Promotes Math Generalist Models** [[pdf]](http://arxiv.org/abs/2409.15376) `2024-09-19` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          ControlMath is proposed, an iterative method involving an equation-generator module and two LLM-based agents that enables the generation of diverse math problems, not limited to specific domains or distributions.
     </details>


     <details>
          <summary>Abstract</summary>
          Utilizing large language models (LLMs) for data augmentation has yielded encouraging results in mathematical reasoning. However, these approaches face constraints in problem diversity, potentially restricting them to in-domain/distribution data generation. To this end, we propose ControlMath, an iterative method involving an equation-generator module and two LLM-based agents. The module creates diverse equations, which the Problem-Crafter agent then transforms into math word problems. The Reverse-Agent filters and selects high-quality data, adhering to the "less is more" principle, achieving better results with fewer data points. This approach enables the generation of diverse math problems, not limited to specific domains or distributions. As a result, we collect ControlMathQA, which involves 190k math word problems. Extensive results prove that combining our dataset with in-domain datasets like GSM8K can help improve the model's mathematical ability to generalize, leading to improved performances both within and beyond specific domains.
     </details>

192. **InfiMM-WebMath-40B: Advancing Multimodal Pre-Training for Enhanced Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2409.12568) `NeurIPS 2024 Workshop MATH-AI` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces InfiMM-WebMath-40B, a high-quality dataset of interleaved image-text documents that significantly enhances the performance of the 1.3B model, and sets a new state-of-the-art among open-source models on multi-modal math benchmarks such as MathVerse and The authors-Math.
     </details>


     <details>
          <summary>Abstract</summary>
          Pre-training on large-scale, high-quality datasets is crucial for enhancing the reasoning capabilities of Large Language Models (LLMs), especially in specialized domains such as mathematics. Despite the recognized importance, the Multimodal LLMs (MLLMs) field currently lacks a comprehensive open-source pre-training dataset specifically designed for mathematical reasoning. To address this gap, we introduce InfiMM-WebMath-40B, a high-quality dataset of interleaved image-text documents. It comprises 24 million web pages, 85 million associated image URLs, and 40 billion text tokens, all meticulously extracted and filtered from CommonCrawl. We provide a detailed overview of our data collection and processing pipeline. To demonstrate the robustness of InfiMM-WebMath-40B, we conducted evaluations in both text-only and multimodal settings. Our evaluations on text-only benchmarks show that, despite utilizing only 40 billion tokens, our dataset significantly enhances the performance of our 1.3B model, delivering results comparable to DeepSeekMath-1.3B, which uses 120 billion tokens for the same model size. Nevertheless, with the introduction of our multi-modal math pre-training dataset, our models set a new state-of-the-art among open-source models on multi-modal math benchmarks such as MathVerse and We-Math. We release our data at https://huggingface.co/datasets/Infi-MM/InfiMM-WebMath-40B.
     </details>

193. **Iteration of Thought: Leveraging Inner Dialogue for Autonomous Large Language Model Reasoning** [[pdf]](http://arxiv.org/abs/2409.12618) `2024-09-19` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The Iteration of Thought (IoT) framework for enhancing LLM responses by generating "thought"-provoking prompts vis a vis an input query and the current iteration of an LLM's response is proposed.
     </details>


     <details>
          <summary>Abstract</summary>
          Iterative human engagement is a common and effective means of leveraging the advanced language processing power of large language models (LLMs). Using well-structured prompts in a conversational manner, human users can effectively influence an LLM to develop more thoughtful and accurate responses. Motivated by this insight, we propose the Iteration of Thought (IoT) framework for enhancing LLM responses by generating "thought"-provoking prompts vis a vis an input query and the current iteration of an LLM's response. Unlike static or semi-static approaches, e.g. Chain of Thought (CoT) or Tree of Thoughts (ToT), IoT adapts its reasoning path dynamically, based on evolving context, and without generating alternate explorative thoughts which are ultimately discarded. The three components of the IoT framework are (1) an Inner Dialogue Agent (IDA) responsible for generating instructive, context-specific prompts; (2) an LLM Agent (LLMA) that processes these prompts to refine its responses; and (3) an iterative prompting loop that implements a conversation between the former two components. We introduce two variants of our framework: Autonomous Iteration of Thought (AIoT), where an LLM decides when to stop iterating, and Guided Iteration of Thought (GIoT), which always forces a fixed number iterations. We investigate the performance of IoT across various datasets, spanning complex reasoning tasks from the GPQA dataset, explorative problem-solving in Game of 24, puzzle solving in Mini Crosswords, and multi-hop question answering from the HotpotQA dataset. Our results show that IoT represents a viable paradigm for autonomous response refinement in LLMs, showcasing significant improvements over CoT and thereby enabling more adaptive and efficient reasoning systems that minimize human intervention.
     </details>

194. **LogicPro: Improving Complex Logical Reasoning via Program-Guided Learning** [[pdf]](http://arxiv.org/abs/2409.12929) `2024-09-19` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This approach achieves significant improvements in multiple models for the BBH$^{27}$, GSM8K, HellSwag, Logicqa, Reclor, and RTE datasets, outperforming a wide range of existing reasoning datasets.
     </details>


     <details>
          <summary>Abstract</summary>
          In this paper, we present a novel approach, called LogicPro, to enhance Large Language Models (LLMs) complex Logical reasoning through Program Examples. We do this effectively by simply utilizing widely available algorithmic problems and their code solutions. First, we constructed diverse test samples input based on algorithmic questions and code solutions. Then, we designed different complex reasoning questions based on algorithmic problems and test samples. Finally, combining the intermediate variable outputs of the code solutions and the complex reasoning questions, we derived the reasoning process and the final answer. With this approach, we can construct a dataset that is sufficiently difficult (all models are ineffective), diverse (synthesized from 2,360 different algorithmic questions), and scalable (building different test samples and collecting more algorithmic questions). In addition, we obtain a high-quality reasoning process guided by the values of intermediate variables. As a result, our approach achieves significant improvements in multiple models for the BBH$^{27}$, GSM8K, HellSwag, Logicqa, Reclor, and RTE datasets, outperforming a wide range of existing reasoning datasets.
     </details>

195. **System 2 thinking in OpenAI's o1-preview model: Near-perfect performance on a mathematics exam** [[pdf]](http://arxiv.org/abs/2410.07114) `2024-09-19` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Although O1-preview had the ability to achieve a perfect score, its performance showed some variability, as it made occasional mistakes with repeated prompting, which suggests that the self-consistency method, where the consensus output is selected, could improve accuracy.
     </details>


     <details>
          <summary>Abstract</summary>
          The processes underlying human cognition are often divided into two systems: System 1, which involves fast, intuitive thinking, and System 2, which involves slow, deliberate reasoning. Previously, large language models were criticized for lacking the deeper, more analytical capabilities of System 2. In September 2024, OpenAI introduced the O1 model series, specifically designed to handle System 2-like reasoning. While OpenAI's benchmarks are promising, independent validation is still needed. In this study, we tested the O1-preview model twice on the Dutch 'Mathematics B' final exam. It scored a near-perfect 76 and 73 out of 76 points. For context, only 24 out of 16,414 students in the Netherlands achieved a perfect score. By comparison, the GPT-4o model scored 66 and 61 out of 76, well above the Dutch average of 40.63 points. The O1-preview model completed the exam in around 10 minutes, while GPT-4o took 3 minutes, and neither model had access to the exam figures. Although O1-preview had the ability to achieve a perfect score, its performance showed some variability, as it made occasional mistakes with repeated prompting. This suggests that the self-consistency method, where the consensus output is selected, could improve accuracy. We conclude that while OpenAI's new model series holds great potential, certain risks must be considered.
     </details>

196. **Textualized Agent-Style Reasoning for Complex Tasks by Multiple Round LLM Generation** [[pdf]](https://arxiv.org/abs/2409.12411v1) `2024-09-19` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work presents AgentCOT, a llm-based autonomous agent framework, which can solve complex problems in an agent-style manner by multiple round LLM generation by integrating the step's index into the reasoning process to form a graph structure for complex inference logic.
     </details>


     <details>
          <summary>Abstract</summary>
          Chain-of-thought prompting significantly boosts the reasoning ability of large language models but still faces three issues: hallucination problem, restricted interpretability, and uncontrollable generation. To address these challenges, we present AgentCOT, a llm-based autonomous agent framework, which can solve complex problems in an agent-style manner by multiple round LLM generation. At each step, AgentCOT selects an action and executes it to yield an intermediate result with supporting evidence. In addition, we integrate the step's index into the reasoning process to form a graph structure for complex inference logic. We introduce two new strategies to enhance the performance of AgentCOT.We conduct extensive experiments to verify the effectiveness of our method on six common benchmarks. Results exhibit that our method brings in substantial improvements over current competitive approaches.
     </details>

197. **To CoT or not to CoT? Chain-of-thought helps mainly on math and symbolic reasoning** [[pdf]](https://arxiv.org/abs/2409.12183v1) `ICLR 2025 Submission` (5 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The results show that CoT gives strong performance benefits primarily on tasks involving math or logic, with much smaller gains on other types of tasks, and suggest a need to move beyond prompt-based CoT to new paradigms that better leverage intermediate computation across the whole range of LLM applications.
     </details>


     <details>
          <summary>Abstract</summary>
          Chain-of-thought (CoT) via prompting is the de facto method for eliciting reasoning capabilities from large language models (LLMs). But for what kinds of tasks is this extra ``thinking'' really helpful? To analyze this, we conducted a quantitative meta-analysis covering over 100 papers using CoT and ran our own evaluations of 20 datasets across 14 models. Our results show that CoT gives strong performance benefits primarily on tasks involving math or logic, with much smaller gains on other types of tasks. On MMLU, directly generating the answer without CoT leads to almost identical accuracy as CoT unless the question or model's response contains an equals sign, indicating symbolic operations and reasoning. Following this finding, we analyze the behavior of CoT on these problems by separating planning and execution and comparing against tool-augmented LLMs. Much of CoT's gain comes from improving symbolic execution, but it underperforms relative to using a symbolic solver. Our results indicate that CoT can be applied selectively, maintaining performance while saving inference costs. Furthermore, they suggest a need to move beyond prompt-based CoT to new paradigms that better leverage intermediate computation across the whole range of LLM applications.
     </details>

198. **Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement** [[pdf]](https://arxiv.org/abs/2409.12122v1) `2024-09-18` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A series of math-specific large language models that possess advanced mathematical reasoning capabilities, including Chain-of-Thought (CoT) and Tool-Integrated Reasoning (TIR), and are evaluated on 10 mathematics datasets in both English and Chinese.
     </details>


     <details>
          <summary>Abstract</summary>
          In this report, we present a series of math-specific large language models: Qwen2.5-Math and Qwen2.5-Math-Instruct-1.5B/7B/72B. The core innovation of the Qwen2.5 series lies in integrating the philosophy of self-improvement throughout the entire pipeline, from pre-training and post-training to inference: (1) During the pre-training phase, Qwen2-Math-Instruct is utilized to generate large-scale, high-quality mathematical data. (2) In the post-training phase, we develop a reward model (RM) by conducting massive sampling from Qwen2-Math-Instruct. This RM is then applied to the iterative evolution of data in supervised fine-tuning (SFT). With a stronger SFT model, it's possible to iteratively train and update the RM, which in turn guides the next round of SFT data iteration. On the final SFT model, we employ the ultimate RM for reinforcement learning, resulting in the Qwen2.5-Math-Instruct. (3) Furthermore, during the inference stage, the RM is used to guide sampling, optimizing the model's performance.   Qwen2.5-Math-Instruct supports both Chinese and English, and possess advanced mathematical reasoning capabilities, including Chain-of-Thought (CoT) and Tool-Integrated Reasoning (TIR). We evaluate our models on 10 mathematics datasets in both English and Chinese, such as GSM8K, MATH, GaoKao, AMC23, and AIME24, covering a range of difficulties from grade school level to math competition problems.
     </details>

199. **Enhancing Logical Reasoning in Large Language Models through Graph-based Synthetic Data** [[pdf]](http://arxiv.org/abs/2409.12437) `2024-09-18` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is demonstrated that supervised fine-tuning (SFT) with synthetic graph-based reasoning data effectively enhances LLMs' reasoning performance without compromising their effectiveness on other standard evaluation benchmarks.
     </details>


     <details>
          <summary>Abstract</summary>
          Despite recent advances in training and prompting strategies for Large Language Models (LLMs), these models continue to face challenges with complex logical reasoning tasks that involve long reasoning chains. In this work, we explore the potential and limitations of using graph-based synthetic reasoning data as training signals to enhance LLMs' reasoning capabilities. Our extensive experiments, conducted on two established natural language reasoning tasks -- inductive reasoning and spatial reasoning -- demonstrate that supervised fine-tuning (SFT) with synthetic graph-based reasoning data effectively enhances LLMs' reasoning performance without compromising their effectiveness on other standard evaluation benchmarks.
     </details>

200. **MAgICoRe: Multi-Agent, Iterative, Coarse-to-Fine Refinement for Reasoning** [[pdf]](https://arxiv.org/abs/2409.12147v1) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes MAgICoRe, which avoids excessive refinement by categorizing problem difficulty as easy or hard, solving easy problems with coarse-grained aggregation and hard ones with fine-grained and iterative multi-agent refinement, and employs a multi-agent loop with three agents.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models' (LLM) reasoning can be improved using test-time aggregation strategies, i.e., generating multiple samples and voting among generated samples. While these improve performance, they often reach a saturation point. Refinement offers an alternative by using LLM-generated feedback to improve solution quality. However, refinement introduces 3 key challenges: (1) Excessive refinement: Uniformly refining all instances can over-correct and reduce the overall performance. (2) Inability to localize and address errors: LLMs have a limited ability to self-correct and struggle to identify and correct their own mistakes. (3) Insufficient refinement: Deciding how many iterations of refinement are needed is non-trivial, and stopping too soon could leave errors unaddressed. To tackle these issues, we propose MAgICoRe, which avoids excessive refinement by categorizing problem difficulty as easy or hard, solving easy problems with coarse-grained aggregation and hard ones with fine-grained and iterative multi-agent refinement. To improve error localization, we incorporate external step-wise reward model (RM) scores. Moreover, to ensure effective refinement, we employ a multi-agent loop with three agents: Solver, Reviewer (which generates targeted feedback based on step-wise RM scores), and the Refiner (which incorporates feedback). To ensure sufficient refinement, we re-evaluate updated solutions, iteratively initiating further rounds of refinement. We evaluate MAgICoRe on Llama-3-8B and GPT-3.5 and show its effectiveness across 5 math datasets. Even one iteration of MAgICoRe beats Self-Consistency by 3.4%, Best-of-k by 3.2%, and Self-Refine by 4.0% while using less than half the samples. Unlike iterative refinement with baselines, MAgICoRe continues to improve with more iterations. Finally, our ablations highlight the importance of MAgICoRe's RMs and multi-agent communication.
     </details>

201. **Small Language Models are Equation Reasoners** [[pdf]](http://arxiv.org/abs/2409.12393) `2024-09-18` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper investigates why sLM perform poorly on arithmetic reasoning tasks and hypothesizes that natural language format variability introduces high ambiguity for these smaller models, and conducts experiments with equation-only format, which is a reasoning format that unifies arithmetic reasoning previously expressed in natural language formats into mathematical equations.
     </details>


     <details>
          <summary>Abstract</summary>
          Chain-of-Thought (CoT) reasoning has enabled Large Language Model (LLM) to achieve remarkable performance in various NLP tasks, including arithmetic problem-solving. However, this success does not generalize to small language model (sLM) like T5, due to their limited capacity and absence of emergent abilities associated with larger models. Recent works to enhance sLM through knowledge distillation have yielded some improvements but still face significant limitations, particularly high ambiguity from the variability in natural language expressions and substantial computational costs. In this paper, we investigate why sLM perform poorly on arithmetic reasoning tasks and hypothesize that natural language format variability introduces high ambiguity for these smaller models. Based on this hypothesis, we conduct experiments with equation-only format, which is a reasoning format that unifies arithmetic reasoning previously expressed in natural language formats into mathematical equations. Experiment results demonstrate that equation-only format effectively boosts the arithmetic reasoning abilities of sLM, especially in very small models like T5-Tiny.
     </details>

202. **Improving LLM Reasoning with Multi-Agent Tree-of-Thought Validator Agent** [[pdf]](https://arxiv.org/abs/2409.11527v1) `2024-09-17` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces a novel approach combining ToT-based Reasoner agents with a Thought Validator agent, enabling a more robust voting strategy by discarding faulty reasoning paths, enhancing the system's ability to tackle tasks requiring systematic and trustworthy reasoning.
     </details>


     <details>
          <summary>Abstract</summary>
          Multi-agent strategies have emerged as a promising approach to enhance the reasoning abilities of Large Language Models (LLMs) by assigning specialized roles in the problem-solving process. Concurrently, Tree of Thoughts (ToT) methods have shown potential in improving reasoning for complex question-answering tasks by exploring diverse reasoning paths. A critical limitation in multi-agent reasoning is the 'Reasoner' agent's shallow exploration of reasoning paths. While ToT strategies could help mitigate this problem, they may generate flawed reasoning branches, which could harm the trustworthiness of the final answer. To leverage the strengths of both multi-agent reasoning and ToT strategies, we introduce a novel approach combining ToT-based Reasoner agents with a Thought Validator agent. Multiple Reasoner agents operate in parallel, employing ToT to explore diverse reasoning paths. The Thought Validator then scrutinizes these paths, considering a Reasoner's conclusion only if its reasoning is valid. This method enables a more robust voting strategy by discarding faulty reasoning paths, enhancing the system's ability to tackle tasks requiring systematic and trustworthy reasoning. Our method demonstrates superior performance compared to existing techniques when evaluated on the GSM8K dataset, outperforming the standard ToT strategy by an average 5.6% across four LLMs. The code and related content can be found in: https://github.com/SecureAIAutonomyLab/MA-ToT
     </details>

203. **Reasoning Graph Enhanced Exemplars Retrieval for In-Context Learning** [[pdf]](http://arxiv.org/abs/2409.11147) `2024-09-17` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel method named Reasoning Graph-enhanced Exemplar Retrieval (RGER), which first quires LLM to generate an initial response, then expresses intermediate problem-solving steps to a graph structure, and employs graph kernel to select exemplars with semantic and structural similarity.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models(LLMs) have exhibited remarkable few-shot learning capabilities and unified the paradigm of NLP tasks through the in-context learning(ICL) technique. Despite the success of ICL, the quality of the exemplar demonstrations can significantly influence the LLM's performance. Existing exemplar selection methods mainly focus on the semantic similarity between queries and candidate exemplars. On the other hand, the logical connections between reasoning steps can be beneficial to depict the problem-solving process as well. In this paper, we proposes a novel method named Reasoning Graph-enhanced Exemplar Retrieval(RGER). RGER first quires LLM to generate an initial response, then expresses intermediate problem-solving steps to a graph structure. After that, it employs graph kernel to select exemplars with semantic and structural similarity. Extensive experiments demonstrate the structural relationship is helpful to the alignment of queries and candidate exemplars. The efficacy of RGER on math and logit reasoning tasks showcases its superiority over state-of-the-art retrieval-based approaches. Our code is released at https://github.com/Yukang-Lin/RGER.
     </details>

204. **RoMath: A Mathematical Reasoning Benchmark in Romanian** [[pdf]](http://arxiv.org/abs/2409.11074) `2024-09-17` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Mathematics has long been conveyed through natural language, primarily for human understanding. With the rise of mechanized mathematics and proof assistants, there is a growing need to understand informal mathematical text, yet most existing benchmarks focus solely on English, overlooking other languages. This paper introduces RoMath, a Romanian mathematical reasoning benchmark suite comprising three datasets: RoMath-Baccalaureate, RoMath-Competitions and RoMath-Synthetic, which cover a range of mathematical domains and difficulty levels, aiming to improve non-English language models and promote multilingual AI development. By focusing on Romanian, a low-resource language with unique linguistic features, RoMath addresses the limitations of Anglo-centric models and emphasizes the need for dedicated resources beyond simple automatic translation. We benchmark several open-weight language models, highlighting the importance of creating resources for underrepresented languages. We make the code and dataset available.
     </details>

205. **On the Diagram of Thought** [[pdf]](http://arxiv.org/abs/2409.10038) `2024-09-16` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Diagram of Thought is introduced, a framework that models iterative reasoning in large language models (LLMs) as the construction of a directed acyclic graph (DAG) within a single model, eliminating the need for multiple models or external control mechanisms.
     </details>


     <details>
          <summary>Abstract</summary>
          We introduce Diagram of Thought (DoT), a framework that models iterative reasoning in large language models (LLMs) as the construction of a directed acyclic graph (DAG) within a single model. Unlike traditional approaches that represent reasoning as linear chains or trees, DoT organizes propositions, critiques, refinements, and verifications into a cohesive DAG structure, allowing the model to explore complex reasoning pathways while maintaining logical consistency. Each node in the diagram corresponds to a proposition that has been proposed, critiqued, refined, or verified, enabling the LLM to iteratively improve its reasoning through natural language feedback. By leveraging auto-regressive next-token prediction with role-specific tokens, DoT facilitates seamless transitions between proposing ideas and critically evaluating them, providing richer feedback than binary signals. Furthermore, we formalize the DoT framework using Topos Theory, providing a mathematical foundation that ensures logical consistency and soundness in the reasoning process. This approach enhances both the training and inference processes within a single LLM, eliminating the need for multiple models or external control mechanisms. DoT offers a conceptual framework for designing next-generation reasoning-specialized models, emphasizing training efficiency, robust reasoning capabilities, and theoretical grounding. The code is available at https://github.com/diagram-of-thought/diagram-of-thought.
     </details>

206. **CPL: Critical Planning Step Learning Boosts LLM Generalization in Reasoning Tasks** [[pdf]](http://arxiv.org/abs/2409.08642) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes searching within the action space on high-level abstract plans to enhance model generalization and introduces Critical Plan Step Learning (CPL), comprising: searching on plan, using Monte Carlo Tree Search (MCTS) to explore diverse plan steps in multi-step reasoning tasks, and learning critical plan steps through Step-level Advantage Preference Optimization (Step-APO), which integrates advantage estimates for step preference obtained via MCTS into Direct Preference Optimization (DPO).
     </details>


     <details>
          <summary>Abstract</summary>
          Post-training large language models (LLMs) to develop reasoning capabilities has proven effective across diverse domains, such as mathematical reasoning and code generation. However, existing methods primarily focus on improving task-specific reasoning but have not adequately addressed the model's generalization capabilities across a broader range of reasoning tasks. To tackle this challenge, we introduce Critical Planning Step Learning (CPL), which leverages Monte Carlo Tree Search (MCTS) to explore diverse planning steps in multi-step reasoning tasks. Based on long-term outcomes, CPL learns step-level planning preferences to improve the model's planning capabilities and, consequently, its general reasoning capabilities. Furthermore, while effective in many scenarios for aligning LLMs, existing preference learning approaches like Direct Preference Optimization (DPO) struggle with complex multi-step reasoning tasks due to their inability to capture fine-grained supervision at each step. We propose Step-level Advantage Preference Optimization (Step-APO), which integrates an advantage estimate for step-level preference pairs obtained via MCTS into the DPO. This enables the model to more effectively learn critical intermediate planning steps, thereby further improving its generalization in reasoning tasks. Experimental results demonstrate that our method, trained exclusively on GSM8K and MATH, not only significantly improves performance on GSM8K (+10.5%) and MATH (+6.5%), but also enhances out-of-domain reasoning benchmarks, such as ARC-C (+4.0%), BBH (+1.8%), MMLU-STEM (+2.2%), and MMLU (+0.9%).
     </details>

207. **Expediting and Elevating Large Language Model Reasoning via Hidden Chain-of-Thought Decoding** [[pdf]](http://arxiv.org/abs/2409.08561) `2024-09-13` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a novel approach to compress the CoT process through semantic alignment, enabling more efficient decoding while preserving the benefits of CoT reasoning and paves the way for more efficient exploitation of multi-step reasoning capabilities in LLMs across a wide range of applications.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have demonstrated remarkable capabilities in tasks requiring reasoning and multi-step problem-solving through the use of chain-of-thought (CoT) prompting. However, generating the full CoT process results in significantly longer output sequences, leading to increased computational costs and latency during inference. To address this challenge, we propose a novel approach to compress the CoT process through semantic alignment, enabling more efficient decoding while preserving the benefits of CoT reasoning. Our method introduces an auxiliary CoT model that learns to generate and compress the full thought process into a compact special token representation semantically aligned with the original CoT output. This compressed representation is then integrated into the input of the Hidden Chain-of-Thought (HCoT) model. The training process follows a two-stage procedure: First, the CoT model is optimized to generate the compressed token representations aligned with the ground-truth CoT outputs using a contrastive loss. Subsequently, with the CoT model parameters frozen, the HCoT model is fine-tuned to generate accurate subsequent predictions conditioned on the prefix instruction and the compressed CoT representations from the CoT model. Extensive experiments across three challenging domains - mathematical reasoning, agent invocation, and question answering - demonstrate that our semantic compression approach achieves competitive or improved performance compared to the full CoT baseline, while providing significant speedups of at least 1.5x in decoding time. Moreover, incorporating contrastive learning objectives further enhances the quality of the compressed representations, leading to better CoT prompting and improved task accuracy. Our work paves the way for more efficient exploitation of multi-step reasoning capabilities in LLMs across a wide range of applications.
     </details>

208. **Can We Count on LLMs? The Fixed-Effect Fallacy and Claims of GPT-4 Capabilities** [[pdf]](http://arxiv.org/abs/2409.07638) `2024-09-11` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Evaluation of LLM capabilities is explored, finding that efforts to quantify LLM capabilities easily succumb to the language-as-fixed-effect fallacy, where experimental observations are improperly generalized beyond what the data supports.
     </details>


     <details>
          <summary>Abstract</summary>
          In this paper we explore evaluation of LLM capabilities. We present measurements of GPT-4 performance on several deterministic tasks; each task involves a basic calculation and takes as input parameter some element drawn from a large well-defined population (e.g., count elements in a list, multiply two k-digit numbers, etc). We examine several conditions per-task and perform enough trials so that statistically significant differences can be detected. This allows us to investigate the sensitivity of task-accuracy both to query phrasing and input parameter population. We find that seemingly trivial modifications in the task-prompt or input population can yield differences far larger than can be explained by sampling effects. For example, performance on a simple list-counting task varies with query-phrasing and list-length, but also with list composition (i.e., the thing-to-be-counted) and object frequency (e.g., success when an element accounts for $\approx$ 50\% of a list is different from when it accounts for $\approx$ 70\% etc).   We conclude that efforts to quantify LLM capabilities easily succumb to the language-as-fixed-effect fallacy, where experimental observations are improperly generalized beyond what the data supports. A consequence appears to be that intuitions that have been formed based on interactions with humans form a very unreliable guide as to which input modifications should ``make no difference'' to LLM performance.
     </details>

209. **MathGLM-Vision: Solving Mathematical Problems with Multi-Modal Large Language Model** [[pdf]](http://arxiv.org/abs/2409.13729) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have demonstrated significant capabilities in mathematical reasoning, particularly with text-based mathematical problems. However, current multi-modal large language models (MLLMs), especially those specialized in mathematics, tend to focus predominantly on solving geometric problems but ignore the diversity of visual information available in other areas of mathematics. Moreover, the geometric information for these specialized mathematical MLLMs is derived from several public datasets, which are typically limited in diversity and complexity. To address these limitations, we aim to construct a fine-tuning dataset named MathVL, and develop a series of specialized mathematical MLLMs termed MathGLM-Vision by conducting Supervised Fine-Tuning (SFT) on MathVL with various parameter-scale backbones. To extensively evaluate the effectiveness of MathGLM-Vision, we conduct experiments on several public benchmarks and our curated MathVL-test consisting of 2,000 problems. Experimental results demonstrate that MathGLM-Vision achieves significant improvements compared with some existing models, including backbone models and open-source mathematical MLLMs. These findings indicate the importance of diversity dataset in enhancing the mathematical reasoning abilities of MLLMs.
     </details>

210. **VisScience: An Extensive Benchmark for Evaluating K12 Educational Multi-modal Scientific Reasoning** [[pdf]](http://arxiv.org/abs/2409.13730) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A comprehensive benchmark is constructed, named VisScience, which is utilized to assess the multi-modal scientific reasoning across the three disciplines of mathematics, physics, and chemistry and demonstrates that closed-source MLLMs generally outperform open-source models.
     </details>


     <details>
          <summary>Abstract</summary>
          Multi-modal large language models (MLLMs) have demonstrated promising capabilities across various tasks by integrating textual and visual information to achieve visual understanding in complex scenarios. Despite the availability of several benchmarks aims to evaluating MLLMs in tasks from visual question answering to complex problem-solving, most focus predominantly on mathematics or general visual understanding tasks. This reveals a critical gap in current benchmarks, which often overlook the inclusion of other key scientific disciplines such as physics and chemistry. To address this gap, we meticulously construct a comprehensive benchmark, named VisScience, which is utilized to assess the multi-modal scientific reasoning across the three disciplines of mathematics, physics, and chemistry. This benchmark comprises 3,000 questions drawn from K12 education - spanning elementary school through high school - equally distributed across three disciplines, with 1,000 questions per discipline. The questions within VisScience span 21 distinct subjects and are categorized into five difficulty levels, offering a broad spectrum of topics within each discipline. With VisScience, we present a detailed evaluation of the performance of 25 representative MLLMs in scientific reasoning. Experimental results demonstrate that closed-source MLLMs generally outperform open-source models. The best performance observed include a 53.4\% accuracy in mathematics by Claude3.5-Sonnet, 38.2\% in physics by GPT-4o, and 47.0\% in chemistry by Gemini-1.5-Pro. These results underscore the strengths and limitations of MLLMs, suggesting areas for future improvement and highlighting the importance of developing models that can effectively handle the diverse demands of multi-modal scientific reasoning.
     </details>

211. **Diagram Formalization Enhanced Multi-Modal Geometry Problem Solver** [[pdf]](http://arxiv.org/abs/2409.04214) `2024-09-06` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The Diagram Formalization Enhanced Geometry Problem Solver (DFE-GPS) is introduced, a new framework that integrates visual features, geometric formal language, and natural language representations that improves MLLMs' ability to process geometric diagrams and extends their application to open-ended tasks on the formalgeo7k dataset.
     </details>


     <details>
          <summary>Abstract</summary>
          Mathematical reasoning remains an ongoing challenge for AI models, especially for geometry problems that require both linguistic and visual signals. As the vision encoders of most MLLMs are trained on natural scenes, they often struggle to understand geometric diagrams, performing no better in geometry problem solving than LLMs that only process text. This limitation is amplified by the lack of effective methods for representing geometric relationships. To address these issues, we introduce the Diagram Formalization Enhanced Geometry Problem Solver (DFE-GPS), a new framework that integrates visual features, geometric formal language, and natural language representations. We propose a novel synthetic data approach and create a large-scale geometric dataset, SynthGeo228K, annotated with both formal and natural language captions, designed to enhance the vision encoder for a better understanding of geometric structures. Our framework improves MLLMs' ability to process geometric diagrams and extends their application to open-ended tasks on the formalgeo7k dataset.
     </details>

212. **From Calculation to Adjudication: Examining LLM judges on Mathematical Reasoning Tasks** [[pdf]](http://arxiv.org/abs/2409.04168) `2024-09-06` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The analysis uncovers a strong correlation between judgment performance and the candidate model task performance and shows that regularities in the judgments are quantifiable using statistical measures and provides various angles on exploiting them.
     </details>


     <details>
          <summary>Abstract</summary>
          To reduce the need for human annotations, large language models (LLMs) have been proposed as judges of the quality of other candidate models. LLM judges are typically evaluated by measuring the correlation with human judgments on generation tasks such as summarization or machine translation. In contrast, we study LLM judges on mathematical reasoning tasks. These tasks require multi-step reasoning, and the correctness of their solutions is verifiable, enabling a more objective evaluation. We perform a detailed performance analysis and find that the used judges are mostly unable to improve task performance but are able to pick the better model. Our analysis uncovers a strong correlation between judgment performance and the candidate model task performance. We observe that judges tend to choose the model of higher quality even if its answer is incorrect. Further, we show that it is possible to use statistics, such as the task performances of the individual models, to predict judgment performance. In an ablation, we either swap or mask the candidate answers and observe that judges often keep the original judgment, providing evidence that judges incorporate writing style in their judgments. In summary, we find that regularities in the judgments are quantifiable using statistical measures and provide various angles on exploiting them.
     </details>

213. **Self-Harmonized Chain of Thought** [[pdf]](http://arxiv.org/abs/2409.04057) `2024-09-06` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          ECHO is proposed, a self-harmonized chain-of-thought prompting method that consolidates diverse solution paths into a uniform and effective solution pattern and demonstrates the best overall performance across three reasoning domains.
     </details>


     <details>
          <summary>Abstract</summary>
          Chain-of-Thought (CoT) prompting reveals that large language models are capable of performing complex reasoning via intermediate steps. CoT prompting is primarily categorized into three approaches. The first approach utilizes straightforward prompts like ``Let's think step by step'' to generate a sequential thought process before yielding an answer. The second approach makes use of human-crafted, step-by-step demonstrations to guide the model's reasoning process. The third automates the generation of reasoned demonstrations with the 'Let's think step by step'.This approach sometimes leads to reasoning errors, highlighting the need to diversify demonstrations to mitigate its misleading effects. However, diverse demonstrations pose challenges for effective representations. In this work, we propose ECHO, a self-harmonized chain-of-thought prompting method. It consolidates diverse solution paths into a uniform and effective solution pattern.ECHO demonstrates the best overall performance across three reasoning domains.
     </details>

214. **Strategic Chain-of-Thought: Guiding Accurate Reasoning in LLMs through Strategy Elicitation** [[pdf]](http://arxiv.org/abs/2409.03271) `2024-09-05` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The SCoT framework is extended to develop a few-shot method with automatically matched demonstrations, yielding even stronger results, underscore the efficacy of SCoT, highlighting its potential to substantially enhance LLM performance in complex reasoning tasks.
     </details>


     <details>
          <summary>Abstract</summary>
          The Chain-of-Thought (CoT) paradigm has emerged as a critical approach for enhancing the reasoning capabilities of large language models (LLMs). However, despite their widespread adoption and success, CoT methods often exhibit instability due to their inability to consistently ensure the quality of generated reasoning paths, leading to sub-optimal reasoning performance. To address this challenge, we propose the \textbf{Strategic Chain-of-Thought} (SCoT), a novel methodology designed to refine LLM performance by integrating strategic knowledge prior to generating intermediate reasoning steps. SCoT employs a two-stage approach within a single prompt: first eliciting an effective problem-solving strategy, which is then used to guide the generation of high-quality CoT paths and final answers. Our experiments across eight challenging reasoning datasets demonstrate significant improvements, including a 21.05\% increase on the GSM8K dataset and 24.13\% on the Tracking\_Objects dataset, respectively, using the Llama3-8b model. Additionally, we extend the SCoT framework to develop a few-shot method with automatically matched demonstrations, yielding even stronger results. These findings underscore the efficacy of SCoT, highlighting its potential to substantially enhance LLM performance in complex reasoning tasks.
     </details>

215. **Building Math Agents with Multi-Turn Iterative Preference Learning** [[pdf]](https://arxiv.org/abs/2409.02392v1) `ICLR 2025 Submission` (5 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces a multi-turn direct preference learning framework, tailored for this context, that leverages feedback from code interpreters and optimizes trajectory-level preferences and includes multi-turn DPO and multi-turn KTO as specific implementations.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent studies have shown that large language models' (LLMs) mathematical problem-solving capabilities can be enhanced by integrating external tools, such as code interpreters, and employing multi-turn Chain-of-Thought (CoT) reasoning. While current methods focus on synthetic data generation and Supervised Fine-Tuning (SFT), this paper studies the complementary direct preference learning approach to further improve model performance. However, existing direct preference learning algorithms are originally designed for the single-turn chat task, and do not fully address the complexities of multi-turn reasoning and external tool integration required for tool-integrated mathematical reasoning tasks. To fill in this gap, we introduce a multi-turn direct preference learning framework, tailored for this context, that leverages feedback from code interpreters and optimizes trajectory-level preferences. This framework includes multi-turn DPO and multi-turn KTO as specific implementations. The effectiveness of our framework is validated through training of various language models using an augmented prompt set from the GSM8K and MATH datasets. Our results demonstrate substantial improvements: a supervised fine-tuned Gemma-1.1-it-7B model's performance increased from 77.5% to 83.9% on GSM8K and from 46.1% to 51.2% on MATH. Similarly, a Gemma-2-it-9B model improved from 84.1% to 86.3% on GSM8K and from 51.0% to 54.5% on MATH.
     </details>

216. **CMM-Math: A Chinese Multimodal Math Dataset To Evaluate and Enhance the Mathematics Reasoning of Large Multimodal Models** [[pdf]](https://arxiv.org/abs/2409.02834v1) `2024-09-04` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper releases a Chinese multimodal math dataset, including benchmark and training parts, to evaluate and enhance the mathematical reasoning of LMMs, and proposes a Multimodal Mathematical LMM (Math-LMM) to handle the problems with mixed input of multiple images and text segments.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have obtained promising results in mathematical reasoning, which is a foundational skill for human intelligence. Most previous studies focus on improving and measuring the performance of LLMs based on textual math reasoning datasets (e.g., MATH, GSM8K). Recently, a few researchers have released English multimodal math datasets (e.g., MATHVISTA and MATH-V) to evaluate the effectiveness of large multimodal models (LMMs). In this paper, we release a Chinese multimodal math (CMM-Math) dataset, including benchmark and training parts, to evaluate and enhance the mathematical reasoning of LMMs. CMM-Math contains over 28,000 high-quality samples, featuring a variety of problem types (e.g., multiple-choice, fill-in-the-blank, and so on) with detailed solutions across 12 grade levels from elementary to high school in China. Specifically, the visual context may be present in the questions or opinions, which makes this dataset more challenging. Through comprehensive analysis, we discover that state-of-the-art LMMs on the CMM-Math dataset face challenges, emphasizing the necessity for further improvements in LMM development. We also propose a Multimodal Mathematical LMM (Math-LMM) to handle the problems with mixed input of multiple images and text segments. We train our model using three stages, including foundational pre-training, foundational fine-tuning, and mathematical fine-tuning. The extensive experiments indicate that our model effectively improves math reasoning performance by comparing it with the SOTA LMMs over three multimodal mathematical datasets.
     </details>

217. **Deconfounded Causality-aware Parameter-Efficient Fine-Tuning for Problem-Solving Improvement of LLMs** [[pdf]](https://arxiv.org/abs/2409.02686v1) `2024-09-04` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Deconfounded Causal Adaptation (DCA), a novel parameter-efficient fine-tuning (PEFT) method to enhance the model's reasoning capabilities by encouraging the model to extract the general problem-solving skills and apply these skills to different questions.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have demonstrated remarkable efficiency in tackling various tasks based on human instructions, but studies reveal that they often struggle with tasks requiring reasoning, such as math or physics. This limitation raises questions about whether LLMs truly comprehend embedded knowledge or merely learn to replicate the token distribution without a true understanding of the content. In this paper, we delve into this problem and aim to enhance the reasoning capabilities of LLMs. First, we investigate if the model has genuine reasoning capabilities by visualizing the text generation process at the attention and representation level. Then, we formulate the reasoning process of LLMs into a causal framework, which provides a formal explanation of the problems observed in the visualization. Finally, building upon this causal framework, we propose Deconfounded Causal Adaptation (DCA), a novel parameter-efficient fine-tuning (PEFT) method to enhance the model's reasoning capabilities by encouraging the model to extract the general problem-solving skills and apply these skills to different questions. Experiments show that our method outperforms the baseline consistently across multiple benchmarks, and with only 1.2M tunable parameters, we achieve better or comparable results to other fine-tuning methods. This demonstrates the effectiveness and efficiency of our method in improving the overall accuracy and reliability of LLMs.
     </details>

218. **S$^3$c-Math: Spontaneous Step-level Self-correction Makes Large Language Models Better Mathematical Reasoners** [[pdf]](http://arxiv.org/abs/2409.01524) `2024-09-02` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Self-correction is a novel method that can stimulate the potential reasoning abilities of large language models (LLMs). It involves detecting and correcting errors during the inference process when LLMs solve reasoning problems. However, recent works do not regard self-correction as a spontaneous and intrinsic capability of LLMs. Instead, such correction is achieved through post-hoc generation, external knowledge introduction, multi-model collaboration, and similar techniques. In this paper, we propose a series of mathematical LLMs called S$^3$c-Math, which are able to perform Spontaneous Step-level Self-correction for Mathematical reasoning. This capability helps LLMs to recognize whether their ongoing inference tends to contain errors and simultaneously correct these errors to produce a more reliable response. We proposed a method, which employs a step-level sampling approach to construct step-wise self-correction data for achieving such ability. Additionally, we implement a training strategy that uses above constructed data to equip LLMs with spontaneous step-level self-correction capacities. Our data and methods have been demonstrated to be effective across various foundation LLMs, consistently showing significant progress in evaluations on GSM8K, MATH, and other mathematical benchmarks. To the best of our knowledge, we are the first to introduce the spontaneous step-level self-correction ability of LLMs in mathematical reasoning.
     </details>

219. **CoqPilot, a plugin for LLM-based generation of proofs** [[pdf]](https://arxiv.org/abs/2410.19605) `2024-09-01` `Coq` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          We present CoqPilot, a VS Code extension designed to help automate writing of Coq proofs. The plugin collects the parts of proofs marked with the admit tactic in a Coq file, i.e., proof holes, and combines LLMs along with non-machine-learning methods to generate proof candidates for the holes. Then, CoqPilot checks if each proof candidate solves the given subgoal and, if successful, replaces the hole with it. The focus of CoqPilot is twofold. Firstly, we want to allow users to seamlessly combine multiple Coq generation approaches and provide a zero-setup experience for our tool. Secondly, we want to deliver a platform for LLM-based experiments on Coq proof generation. We developed a benchmarking system for Coq generation methods, available in the plugin, and conducted an experiment using it, showcasing the framework's possibilities. Demo of CoqPilot is available at: https://youtu.be/oB1Lx-So9Lo. Code at: https://github.com/JetBrains-Research/coqpilot
     </details>

220. **Dynamic Self-Consistency: Leveraging Reasoning Paths for Efficient LLM Sampling** [[pdf]](http://arxiv.org/abs/2408.17017) `2024-08-30` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Reasoning-Aware Self-Consistency (RASC), an innovative early-stopping framework that dynamically adjusts the number of sample generations by considering both the output answer and the RPs from Chain of Thought (CoT) prompting, is proposed.
     </details>


     <details>
          <summary>Abstract</summary>
          Self-Consistency (SC) is a widely used method to mitigate hallucinations in Large Language Models (LLMs) by sampling the LLM multiple times and outputting the most frequent solution. Despite its benefits, SC results in significant computational costs proportional to the number of samples generated. Previous early-stopping approaches, such as Early Stopping Self Consistency and Adaptive Consistency, have aimed to reduce these costs by considering output consistency, but they do not analyze the quality of the reasoning paths (RPs) themselves. To address this issue, we propose Reasoning-Aware Self-Consistency (RASC), an innovative early-stopping framework that dynamically adjusts the number of sample generations by considering both the output answer and the RPs from Chain of Thought (CoT) prompting. RASC assigns confidence scores sequentially to the generated samples, stops when certain criteria are met, and then employs weighted majority voting to optimize sample usage and enhance answer reliability. We comprehensively test RASC with multiple LLMs across varied QA datasets. RASC outperformed existing methods and significantly reduces sample usage by an average of 80% while maintaining or improving accuracy up to 5% compared to the original SC
     </details>

221. **MultiMath: Bridging Visual and Mathematical Reasoning for Large Language Models** [[pdf]](http://arxiv.org/abs/2409.00147) `2024-08-30` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A multimodal large language model that bridges the gap between math and vision, MultiMath-7B achieves state-of-the-art (SOTA) performance among open-source models on existing multimodal mathematical benchmarks and also excels on text-only mathematical benchmarks.
     </details>


     <details>
          <summary>Abstract</summary>
          The rapid development of large language models (LLMs) has spurred extensive research into their domain-specific capabilities, particularly mathematical reasoning. However, most open-source LLMs focus solely on mathematical reasoning, neglecting the integration with visual injection, despite the fact that many mathematical tasks rely on visual inputs such as geometric diagrams, charts, and function plots. To fill this gap, we introduce \textbf{MultiMath-7B}, a multimodal large language model that bridges the gap between math and vision. \textbf{MultiMath-7B} is trained through a four-stage process, focusing on vision-language alignment, visual and math instruction-tuning, and process-supervised reinforcement learning. We also construct a novel, diverse and comprehensive multimodal mathematical dataset, \textbf{MultiMath-300K}, which spans K-12 levels with image captions and step-wise solutions. MultiMath-7B achieves state-of-the-art (SOTA) performance among open-source models on existing multimodal mathematical benchmarks and also excels on text-only mathematical benchmarks. Our model and dataset are available at {\textcolor{blue}{\url{https://github.com/pengshuai-rin/MultiMath}}}.
     </details>

222. **Physics of Language Models: Part 2.2, How to Learn From Mistakes on Grade-School Math Problems** [[pdf]](http://arxiv.org/abs/2408.16293) `2024-08-29` (5 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Language models have demonstrated remarkable performance in solving reasoning tasks; however, even the strongest models still occasionally make reasoning mistakes. Recently, there has been active research aimed at improving reasoning accuracy, particularly by using pretrained language models to "self-correct" their mistakes via multi-round prompting. In this paper, we follow this line of work but focus on understanding the usefulness of incorporating "error-correction" data directly into the pretraining stage. This data consists of erroneous solution steps immediately followed by their corrections. Using a synthetic math dataset, we show promising results: this type of pretrain data can help language models achieve higher reasoning accuracy directly (i.e., through simple auto-regression, without multi-round prompting) compared to pretraining on the same amount of error-free data. We also delve into many details, such as (1) how this approach differs from beam search, (2) how such data can be prepared, (3) whether masking is needed on the erroneous tokens, (4) the amount of error required, (5) whether such data can be deferred to the fine-tuning stage, and many others.
     </details>

223. **Critic-CoT: Boosting the reasoning abilities of large language model via Chain-of-thoughts Critic** [[pdf]](http://arxiv.org/abs/2408.16326) `2024-08-29` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Critic-CoT is proposed, a novel framework that pushes LLMs toward System-2-like critic capability through a step-wise CoT reasoning paradigm and the automatic construction of distant-supervision data without human annotation, which enables LLMs to engage in slow, analytic self-critique and refinement, thereby improving their reasoning abilities.
     </details>


     <details>
          <summary>Abstract</summary>
          Self-critic has become a crucial mechanism for enhancing the reasoning performance of LLMs. However, current approaches mainly involve basic prompts for intuitive instance-level feedback, which resembles System-1 processes and limits the reasoning capabilities. Moreover, there is a lack of in-depth investigations into the relationship between LLM's ability to criticize and its task-solving performance. To address these issues, we propose Critic-CoT, a novel framework that pushes LLMs toward System-2-like critic capability. Through a step-wise CoT reasoning paradigm and the automatic construction of distant-supervision data without human annotation, Critic-CoT enables LLMs to engage in slow, analytic self-critique and refinement, thereby improving their reasoning abilities. Experiments on GSM8K and MATH demonstrate that our enhanced model significantly boosts task-solving performance by filtering out invalid solutions or iterative refinement. Furthermore, we investigate the intrinsic correlation between critique and task-solving abilities within LLMs, discovering that these abilities can mutually reinforce each other rather than conflict.
     </details>

224. **Logic Contrastive Reasoning with Lightweight Large Language Model for Math Word Problems** [[pdf]](http://arxiv.org/abs/2409.00131) `2024-08-29` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel method for measuring mathematical logic similarity and an automatic screening mechanism to construct a set of reference problems that integrate both semantic and logical similarity are introduced, in the first attempt to utilize retrieval-enhanced generation for mathematical problem-solving.
     </details>


     <details>
          <summary>Abstract</summary>
          This study focuses on improving the performance of lightweight Large Language Models (LLMs) in mathematical reasoning tasks. We introduce a novel method for measuring mathematical logic similarity and design an automatic screening mechanism to construct a set of reference problems that integrate both semantic and logical similarity. By employing carefully crafted positive and negative example prompts, we guide the model towards adopting sound reasoning logic. To the best of our knowledge, this is the first attempt to utilize retrieval-enhanced generation for mathematical problem-solving. Experimental results demonstrate that our method achieves a 15.8% improvement over the Chain of Thought approach on the SVAMP dataset and a 21.5 % improvement on the GSM8K dataset. Further application of this method to a large-scale model with 175 billion parameters yields performance comparable to the best results on both aforementioned datasets. Finally, we conduct an analysis of errors during the reasoning process, providing valuable insights and directions for future research on reasoning tasks using large language models.
     </details>

225. **AutoGeo: Automating Geometric Image Dataset Creation for Enhanced Geometry Understanding** [[pdf]](http://arxiv.org/abs/2409.09039) `2024-08-28` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Experimental results indicate significant improvements in the model's ability in handling geometric images, as evidenced by enhanced accuracy in tasks such as geometric captioning and mathematical reasoning.
     </details>


     <details>
          <summary>Abstract</summary>
          With the rapid advancement of large language models, there has been a growing interest in their capabilities in mathematical reasoning. However, existing research has primarily focused on text-based algebra problems, neglecting the study of geometry due to the lack of high-quality geometric datasets. To address this gap, this paper introduces AutoGeo, a novel approach for automatically generating mathematical geometric images to fulfill the demand for large-scale and diverse geometric datasets. AutoGeo facilitates the creation of AutoGeo-100k, an extensive repository comprising 100k high-quality geometry image-text pairs. By leveraging precisely defined geometric clauses, AutoGeo-100k contains a wide variety of geometric shapes, including lines, polygons, circles, and complex spatial relationships, etc. Furthermore, this paper demonstrates the efficacy of AutoGeo-100k in enhancing the performance of multimodal large language models through fine-tuning. Experimental results indicate significant improvements in the model's ability in handling geometric images, as evidenced by enhanced accuracy in tasks such as geometric captioning and mathematical reasoning. This research not only fills a critical gap in the availability of geometric datasets but also paves the way for the advancement of sophisticated AI-driven tools in education and research. Project page: https://autogeo-official.github.io/.
     </details>

226. **SIaM: Self-Improving Code-Assisted Mathematical Reasoning of Large Language Models** [[pdf]](https://arxiv.org/abs/2408.15565v1) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a novel paradigm that uses a code-based critic model to guide steps including question-code data construction, quality control, and complementary evaluation and explores different alignment algorithms with self-generated instruction/preference data to foster continuous improvement.
     </details>


     <details>
          <summary>Abstract</summary>
          There is a growing trend of teaching large language models (LLMs) to solve mathematical problems through coding. Existing studies primarily focus on prompting powerful, closed-source models to generate seed training data followed by in-domain data augmentation, equipping LLMs with considerable capabilities for code-aided mathematical reasoning. However, continually training these models on augmented data derived from a few datasets such as GSM8K may impair their generalization abilities and restrict their effectiveness to a narrow range of question types. Conversely, the potential of improving such LLMs by leveraging large-scale, expert-written, diverse math question-answer pairs remains unexplored. To utilize these resources and tackle unique challenges such as code response assessment, we propose a novel paradigm that uses a code-based critic model to guide steps including question-code data construction, quality control, and complementary evaluation. We also explore different alignment algorithms with self-generated instruction/preference data to foster continuous improvement. Experiments across both in-domain (up to +5.7%) and out-of-domain (+4.4%) benchmarks in English and Chinese demonstrate the effectiveness of the proposed paradigm.
     </details>

227. **Generative Verifiers: Reward Modeling as Next-Token Prediction** [[pdf]](http://arxiv.org/abs/2408.15240) `NeurIPS 2024 Workshop MATH-AI` (8 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is demonstrated that when using Gemma-based verifiers on algorithmic and grade-school math reasoning tasks, GenRM outperforms discriminative verifiers and LLM-as-a-Judge, and it is shown that GenRM scales favorably across dataset size, model capacity, and inference-time compute.
     </details>


     <details>
          <summary>Abstract</summary>
          Verifiers or reward models are often used to enhance the reasoning performance of large language models (LLMs). A common approach is the Best-of-N method, where N candidate solutions generated by the LLM are ranked by a verifier, and the best one is selected. While LLM-based verifiers are typically trained as discriminative classifiers to score solutions, they do not utilize the text generation capabilities of pretrained LLMs. To overcome this limitation, we instead propose training verifiers using the ubiquitous next-token prediction objective, jointly on verification and solution generation. Compared to standard verifiers, such generative verifiers (GenRM) can benefit from several advantages of LLMs: they integrate seamlessly with instruction tuning, enable chain-of-thought reasoning, and can utilize additional test-time compute via majority voting for better verification. We demonstrate that GenRM outperforms discriminative, DPO verifiers, and LLM-as-a-Judge, resulting in a 16-40% improvement in the number of problems solved with Best-of-N on algorithmic and math reasoning tasks. Furthermore, we find that training GenRM with synthetic verification rationales is sufficient to pick out subtle errors on math problems. Finally, we demonstrate that generative verifiers scale favorably with model size and inference-time compute.
     </details>

228. **What makes math problems hard for reinforcement learning: a case study** [[pdf]](https://arxiv.org/abs/2408.15332v1) `2024-08-27` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Using a long-standing conjecture from combinatorial group theory, we explore, from multiple angles, the challenges of finding rare instances carrying disproportionately high rewards. Based on lessons learned in the mathematical context defined by the Andrews-Curtis conjecture, we propose algorithmic improvements that can be relevant in other domains with ultra-sparse reward problems. Although our case study can be formulated as a game, its shortest winning sequences are potentially $10^6$ or $10^9$ times longer than those encountered in chess. In the process of our study, we demonstrate that one of the potential counterexamples due to Akbulut and Kirby, whose status escaped direct mathematical methods for 39 years, is stably AC-trivial.
     </details>

229. **Multi-tool Integration Application for Math Reasoning Using Large Language Model** [[pdf]](https://arxiv.org/abs/2408.12148v1) `2024-08-22` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel multi tool application framework for mathematical reasoning is proposed, aiming to achieve more comprehensive and accurate mathematical reasoning by utilizing the collaborative effect of large language models (LLMs) and multiple external tools.
     </details>


     <details>
          <summary>Abstract</summary>
          Mathematical reasoning is an important research direction in the field of artificial intelligence. This article proposes a novel multi tool application framework for mathematical reasoning, aiming to achieve more comprehensive and accurate mathematical reasoning by utilizing the collaborative effect of large language models (LLMs) and multiple external tools. Firstly, use a Math Tool to perform basic mathematical calculations during the inference process through interaction with LLM. Secondly, Code Tool can generate code fragments that comply with syntax rules and execute them, providing support for complex mathematical problems. Then, through the iterative reasoning of the CoT Tool, the logical coherence and accuracy of mathematical reasoning are enhanced. Ultimately, by using self consistency tools to select the final answer based on different parameters, the consistency and reliability of reasoning are improved. Through the synergistic effect of these tools, the framework has achieved significant performance improvement in mathematical reasoning tasks. We conducted experiments on the NumGLUE Task 4 test set, which includes 220 mathematical reasoning fill in the blank questions. The experimental results showed that, based on Math Tool, Code Tool, and CoT Tool, in Task 4 task,our method achieved an accuracy of 89.09,compared with the GPT3+FewShot baseline, Few Shot+ERNIE-4.0+self consistency improved by 49.09%, and compared with fine-tuning the Fine tuning baseline, Few Shot+ERNIE-4.0+self consistency improved by 52.29%
     </details>

230. **Positional Description for Numerical Normalization** [[pdf]](https://arxiv.org/abs/2408.12430v1) `INTERSPEECH 2024 Speech Synthesis` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A Positional Description Scheme tailored for digit sequences, integrating placeholder value information for each digit, which effectively mitigates fatal numerical normalization errors in neural models, requiring only a modest amount of training data without rule-based Finite State Transducers.
     </details>


     <details>
          <summary>Abstract</summary>
          We present a Positional Description Scheme (PDS) tailored for digit sequences, integrating placeholder value information for each digit. Given the structural limitations of subword tokenization algorithms, language models encounter critical Text Normalization (TN) challenges [1] when handling numerical tasks. Our schema addresses this challenge through straightforward pre-processing, preserving the model architecture while significantly simplifying number normalization, rendering the problem tractable. This simplifies the task and facilitates more compact production-ready models capable of learning from smaller datasets. Furthermore, our investigations reveal that PDS enhances the arithmetic processing capabilities of language models, resulting in a relative accuracy improvement of 23% to 51% on complex arithmetic tasks. We demonstrate that PDS effectively mitigates fatal numerical normalization errors in neural models, requiring only a modest amount of training data without rule-based Finite State Transducers (FST). We demonstrate that PDS is essential for both the Text-To-Speech and Speech Recognition text processing, enabling effective TN under production constraints.
     </details>

231. **EAGLE: Elevating Geometric Reasoning through LLM-empowered Visual Instruction Tuning** [[pdf]](https://arxiv.org/abs/2408.11397v1) `2024-08-21` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper proposes EAGLE, a novel two-stage end-to-end visual enhancement MLLM framework designed to ElevAte Geometric reasoning through LLM-Empowered visual instruction tuning, and develops the geometry expert model EAGLE-7B.
     </details>


     <details>
          <summary>Abstract</summary>
          Multi-modal Large Language Models have recently experienced rapid developments and excel in various multi-modal tasks. However, they still struggle with mathematical geometric problem solving, which requires exceptional visual perception proficiency. Existing MLLMs mostly optimize the LLM backbone to acquire geometric reasoning capabilities, while rarely emphasizing improvements in visual comprehension. In this paper, we first investigate the visual perception performance of MLLMs when facing geometric diagrams. Our findings reveal that current MLLMs severely suffer from inaccurate geometric perception and hallucinations. To address these limitations, we propose EAGLE, a novel two-stage end-to-end visual enhancement MLLM framework designed to ElevAte Geometric reasoning through LLM-Empowered visual instruction tuning. Specifically, in the preliminary stage, we feed geometric image-caption pairs into our MLLM that contains a fully fine-tuning CLIP ViT and a frozen LLM, aiming to endow our model with basic geometric knowledge. In the subsequent advanced stage, we incorporate LoRA modules into the vision encoder and unfreeze the LLM backbone. This enables the model to leverage the inherent CoT rationales within question-answer pairs, guiding the MLLM to focus on nuanced visual cues and enhancing its overall perceptual capacity. Moreover, we optimize the cross-modal projector in both stages to foster adaptive visual-linguistic alignments. After the two-stage visual enhancement, we develop the geometry expert model EAGLE-7B. Extensive experiments on popular benchmarks demonstrate the effectiveness of our model. For example, on the GeoQA benchmark, EAGLE-7B not only surpasses the exemplary G-LLaVA 7B model by 2.9%, but also marginally outperforms the larger G-LLaVA 13B model. On the MathVista benchmark, EAGLE-7B achieves remarkable 3.8% improvements compared with the proprietary model GPT-4V.
     </details>

232. **Benchmarking Large Language Models for Math Reasoning Tasks** [[pdf]](https://arxiv.org/abs/2408.10839v1) `2024-08-20` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The results indicate that larger foundation models like GPT-4o and LLaMA 3-70B can solve mathematical reasoning independently from the concrete prompting strategy, while for smaller models the in-context learning approach significantly influences the performance.
     </details>


     <details>
          <summary>Abstract</summary>
          The use of Large Language Models (LLMs) in mathematical reasoning has become a cornerstone of related research, demonstrating the intelligence of these models and enabling potential practical applications through their advanced performance, such as in educational settings. Despite the variety of datasets and in-context learning algorithms designed to improve the ability of LLMs to automate mathematical problem solving, the lack of comprehensive benchmarking across different datasets makes it complicated to select an appropriate model for specific tasks. In this project, we present a benchmark that fairly compares seven state-of-the-art in-context learning algorithms for mathematical problem solving across five widely used mathematical datasets on four powerful foundation models. Furthermore, we explore the trade-off between efficiency and performance, highlighting the practical applications of LLMs for mathematical reasoning. Our results indicate that larger foundation models like GPT-4o and LLaMA 3-70B can solve mathematical reasoning independently from the concrete prompting strategy, while for smaller models the in-context learning approach significantly influences the performance. Moreover, the optimal prompt depends on the chosen foundation model. We open-source our benchmark code to support the integration of additional models in future research.
     </details>

233. **SubgoalXL: Subgoal-based Expert Learning for Theorem Proving** [[pdf]](https://arxiv.org/abs/2408.11172v1) `ICLR 2025 Submission` `Isabelle` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          SubgoalXL is introduced, a novel approach that synergizes subgoal-based proofs with expert learning to enhance LLMs' capabilities in formal theorem proving within the Isabelle environment, and achieves a new state-of-the-art performance.
     </details>


     <details>
          <summary>Abstract</summary>
          Formal theorem proving, a field at the intersection of mathematics and computer science, has seen renewed interest with advancements in large language models (LLMs). This paper introduces SubgoalXL, a novel approach that synergizes subgoal-based proofs with expert learning to enhance LLMs' capabilities in formal theorem proving within the Isabelle environment. SubgoalXL addresses two critical challenges: the scarcity of specialized mathematics and theorem-proving data, and the need for improved multi-step reasoning abilities in LLMs. By optimizing data efficiency and employing subgoal-level supervision, SubgoalXL extracts richer information from limited human-generated proofs. The framework integrates subgoal-oriented proof strategies with an expert learning system, iteratively refining formal statement, proof, and subgoal generators. Leveraging the Isabelle environment's advantages in subgoal-based proofs, SubgoalXL achieves a new state-of-the-art performance of 56.1\% in Isabelle on the standard miniF2F dataset, marking an absolute improvement of 4.9\%. Notably, SubgoalXL successfully solves 41 AMC12, 9 AIME, and 3 IMO problems from miniF2F. These results underscore the effectiveness of maximizing limited data utility and employing targeted guidance for complex reasoning in formal theorem proving, contributing to the ongoing advancement of AI reasoning capabilities. The implementation is available at \url{https://github.com/zhaoxlpku/SubgoalXL}.
     </details>

234. **Concept Distillation from Strong to Weak Models via Hypotheses-to-Theories Prompting** [[pdf]](https://arxiv.org/abs/2408.09365v1) `2024-08-18` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Hand-crafting high quality prompts to optimize the performance of language models is a complicated and labor-intensive process. Furthermore, when migrating to newer, smaller, or weaker models (possibly due to latency or cost gains), prompts need to be updated to re-optimize the task performance. We propose Concept Distillation (CD), an automatic prompt optimization technique for enhancing weaker models on complex tasks. CD involves: (1) collecting mistakes made by weak models with a base prompt (initialization), (2) using a strong model to generate reasons for these mistakes and create rules/concepts for weak models (induction), and (3) filtering these rules based on validation set performance and integrating them into the base prompt (deduction/verification). We evaluated CD on NL2Code and mathematical reasoning tasks, observing significant performance boosts for small and weaker language models. Notably, Mistral-7B's accuracy on Multi-Arith increased by 20%, and Phi-3-mini-3.8B's accuracy on HumanEval rose by 34%. Compared to other automated methods, CD offers an effective, cost-efficient strategy for improving weak models' performance on complex tasks and enables seamless workload migration across different language models without compromising performance.
     </details>

235. **Math-PUMA: Progressive Upward Multimodal Alignment to Enhance Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2408.08640) `2024-08-16` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This approach is designed to improve the mathematical reasoning skills of MLLMs through a three-stage training process, with the second stage being the critical alignment stage and effectively narrows the performance gap for problems presented in different modalities.
     </details>


     <details>
          <summary>Abstract</summary>
          Multimodal Large Language Models (MLLMs) excel in solving text-based mathematical problems, but they struggle with mathematical diagrams since they are primarily trained on natural scene images. For humans, visual aids generally enhance problem-solving, but MLLMs perform worse as information shifts from textual to visual modality. This decline is mainly due to their shortcomings in aligning images and text. To tackle aforementioned challenges, we propose Math-PUMA, a methodology focused on Progressive Upward Multimodal Alignment. This approach is designed to improve the mathematical reasoning skills of MLLMs through a three-stage training process, with the second stage being the critical alignment stage. We first enhance the language model's mathematical reasoning capabilities with extensive set of textual mathematical problems. We then construct a multimodal dataset with varying degrees of textual and visual information, creating data pairs by presenting each problem in at least two forms. By leveraging the Kullback-Leibler (KL) divergence of next-token prediction distributions to align visual and textual modalities, consistent problem-solving abilities are ensured. Finally, we utilize multimodal instruction tuning for MLLMs with high-quality multimodal data. Experimental results on multiple mathematical reasoning benchmarks demonstrate that the MLLMs trained with Math-PUMA surpass most open-source MLLMs. Our approach effectively narrows the performance gap for problems presented in different modalities. The code and data are available at: \url{https://github.com/wwzhuang01/Math-PUMA}.
     </details>

236. **DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for Reinforcement Learning and Monte-Carlo Tree Search** [[pdf]](http://arxiv.org/abs/2408.08152) `2024-08-15` `Lean` (8 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          An open-source language model designed for theorem proving in Lean 4, which enhances DeepSeek-Prover-V1 by optimizing both training and inference processes, and proposes RMaxTS, a variant of Monte-Carlo tree search that employs an intrinsic-reward-driven exploration strategy to generate diverse proof paths.
     </details>


     <details>
          <summary>Abstract</summary>
          We introduce DeepSeek-Prover-V1.5, an open-source language model designed for theorem proving in Lean 4, which enhances DeepSeek-Prover-V1 by optimizing both training and inference processes. Pre-trained on DeepSeekMath-Base with specialization in formal mathematical languages, the model undergoes supervised fine-tuning using an enhanced formal theorem proving dataset derived from DeepSeek-Prover-V1. Further refinement is achieved through reinforcement learning from proof assistant feedback (RLPAF). Beyond the single-pass whole-proof generation approach of DeepSeek-Prover-V1, we propose RMaxTS, a variant of Monte-Carlo tree search that employs an intrinsic-reward-driven exploration strategy to generate diverse proof paths. DeepSeek-Prover-V1.5 demonstrates significant improvements over DeepSeek-Prover-V1, achieving new state-of-the-art results on the test set of the high school level miniF2F benchmark ($63.5\%$) and the undergraduate level ProofNet benchmark ($25.3\%$).
     </details>

237. **Automated Design of Agentic Systems** [[pdf]](http://arxiv.org/abs/2408.08435) `ICLR 2025 Submission` (4 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work forms a new research area, Automated Design of Agentic Systems (ADAS), which aims to automatically create powerful agentic system designs, including inventing novel building blocks and/or combining them in new ways, and presents a simple yet effective algorithm named Meta Agent Search, which can progressively invent agents with novel designs that greatly outperform state-of-the-art hand-designed agents.
     </details>


     <details>
          <summary>Abstract</summary>
          Researchers are investing substantial effort in developing powerful general-purpose agents, wherein Foundation Models are used as modules within agentic systems (e.g. Chain-of-Thought, Self-Reflection, Toolformer). However, the history of machine learning teaches us that hand-designed solutions are eventually replaced by learned solutions. We formulate a new research area, Automated Design of Agentic Systems (ADAS), which aims to automatically create powerful agentic system designs, including inventing novel building blocks and/or combining them in new ways. We further demonstrate that there is an unexplored yet promising approach within ADAS where agents can be defined in code and new agents can be automatically discovered by a meta agent programming ever better ones in code. Given that programming languages are Turing Complete, this approach theoretically enables the learning of any possible agentic system: including novel prompts, tool use, control flows, and combinations thereof. We present a simple yet effective algorithm named Meta Agent Search to demonstrate this idea, where a meta agent iteratively programs interesting new agents based on an ever-growing archive of previous discoveries. Through extensive experiments across multiple domains including coding, science, and math, we show that our algorithm can progressively invent agents with novel designs that greatly outperform state-of-the-art hand-designed agents. Importantly, we consistently observe the surprising result that agents invented by Meta Agent Search maintain superior performance even when transferred across domains and models, demonstrating their robustness and generality. Provided we develop it safely, our work illustrates the potential of an exciting new research direction toward automatically designing ever-more powerful agentic systems to benefit humanity.
     </details>

238. **V-STaR: Training Verifiers for Self-Taught Reasoners** [[pdf]](http://arxiv.org/abs/2402.06457) `COLM 2024` (47 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          V-STaR is proposed that utilizes both the correct and incorrect solutions generated during the self-improvement process to train a verifier using DPO that judges correctness of model-generated solutions.
     </details>


     <details>
          <summary>Abstract</summary>
          Common self-improvement approaches for large language models (LLMs), such as STaR, iteratively fine-tune LLMs on self-generated solutions to improve their problem-solving ability. However, these approaches discard the large amounts of incorrect solutions generated during this process, potentially neglecting valuable information in such solutions. To address this shortcoming, we propose V-STaR that utilizes both the correct and incorrect solutions generated during the self-improvement process to train a verifier using DPO that judges correctness of model-generated solutions. This verifier is used at inference time to select one solution among many candidate solutions. Running V-STaR for multiple iterations results in progressively better reasoners and verifiers, delivering a 4% to 17% test accuracy improvement over existing self-improvement and verification approaches on common code generation and math reasoning benchmarks with LLaMA2 models.
     </details>

239. **MathScape: Evaluating MLLMs in multimodal Math Scenarios through a Hierarchical Benchmark** [[pdf]](https://arxiv.org/abs/2408.07543v3) `2024-08-14` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A new benchmark that emphasizes the understanding and application of combined visual and textual information is proposed, MathScape, designed to evaluate photo-based math problem scenarios, assessing the theoretical understanding and application ability of MLLMs through a categorical hierarchical approach.
     </details>


     <details>
          <summary>Abstract</summary>
          With the development of Multimodal Large Language Models (MLLMs), the evaluation of multimodal models in the context of mathematical problems has become a valuable research field. Multimodal visual-textual mathematical reasoning serves as a critical indicator for evaluating the comprehension and complex multi-step quantitative reasoning abilities of MLLMs. However, previous multimodal math benchmarks have not sufficiently integrated visual and textual information. To address this gap, we proposed MathScape, a new benchmark that emphasizes the understanding and application of combined visual and textual information. MathScape is designed to evaluate photo-based math problem scenarios, assessing the theoretical understanding and application ability of MLLMs through a categorical hierarchical approach. We conduct a multi-dimensional evaluation on 11 advanced MLLMs, revealing that our benchmark is challenging even for the most sophisticated models. By analyzing the evaluation results, we identify the limitations of MLLMs, offering valuable insights for enhancing model performance.
     </details>

240. **Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers** [[pdf]](https://arxiv.org/abs/2408.06195v1) `2024-08-12` (6 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          This paper introduces rStar, a self-play mutual reasoning approach that significantly improves reasoning capabilities of small language models (SLMs) without fine-tuning or superior models. rStar decouples reasoning into a self-play mutual generation-discrimination process. First, a target SLM augments the Monte Carlo Tree Search (MCTS) with a rich set of human-like reasoning actions to construct higher quality reasoning trajectories. Next, another SLM, with capabilities similar to the target SLM, acts as a discriminator to verify each trajectory generated by the target SLM. The mutually agreed reasoning trajectories are considered mutual consistent, thus are more likely to be correct. Extensive experiments across five SLMs demonstrate rStar can effectively solve diverse reasoning problems, including GSM8K, GSM-Hard, MATH, SVAMP, and StrategyQA. Remarkably, rStar boosts GSM8K accuracy from 12.51% to 63.91% for LLaMA2-7B, from 36.46% to 81.88% for Mistral-7B, from 74.53% to 91.13% for LLaMA3-8B-Instruct. Code will be available at https://github.com/zhentingqi/rStar.
     </details>

241. **InfinityMATH: A Scalable Instruction Tuning Dataset in Programmatic Mathematical Reasoning** [[pdf]](https://arxiv.org/abs/2408.07089v1) `2024-08-09` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          InfinityMATH is introduced, a scalable instruction tuning dataset for programmatic mathematical reasoning that ensures that models are more versatile and effective across a broader range of mathematical problems.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent advancements in Chain-of-Thoughts (CoT) and Program-of-Thoughts (PoT) methods have greatly enhanced language models' mathematical reasoning capabilities, facilitating their integration into instruction tuning datasets with LLMs. However, existing methods for large-scale dataset creation require substantial seed data and high computational costs for data synthesis, posing significant challenges for scalability. We introduce InfinityMATH, a scalable instruction tuning dataset for programmatic mathematical reasoning. The construction pipeline emphasizes decoupling numbers from mathematical problems to synthesize number-independent programs, enabling efficient and flexible scaling while minimizing dependency on specific numerical values. Fine-tuning experiments with open-source language and code models, such as Llama2 and CodeLlama, demonstrate the practical benefits of InfinityMATH. These fine-tuned models, showed significant relative improvements on both in-domain and out-of-domain benchmarks, ranging from 184.7% to 514.3% on average. Additionally, these models exhibited high robustness on the GSM8K+ and MATH+ benchmarks, which are enhanced version of test sets with simply the number variations. InfinityMATH ensures that models are more versatile and effective across a broader range of mathematical problems. The data is available at https://huggingface.co/datasets/flagopen/InfinityMATH.
     </details>

242. **Evaluating Language Model Math Reasoning via Grounding in Educational Curricula** [[pdf]](https://arxiv.org/abs/2408.04226v2) `2024-08-08` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Our work presents a novel angle for evaluating language models' (LMs) mathematical abilities, by investigating whether they can discern skills and concepts enabled by math content. We contribute two datasets: one consisting of 385 fine-grained descriptions of K-12 math skills and concepts, or standards, from Achieve the Core (ATC), and another of 9.9K problems labeled with these standards (MathFish). Working with experienced teachers, we find that LMs struggle to tag and verify standards linked to problems, and instead predict labels that are close to ground truth, but differ in subtle ways. We also show that LMs often generate problems that do not fully align with standards described in prompts. Finally, we categorize problems in GSM8k using math standards, allowing us to better understand why some problems are more difficult to solve for models than others.
     </details>

243. **miniCTX: Neural Theorem Proving with (Long-)Contexts** [[pdf]](https://arxiv.org/abs/2408.03350v1) `NeurIPS 2024 Workshop MATH-AI` `Lean` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces miniCTX, which tests a model's ability to prove formal mathematical theorems that depend on new context that is not seen during training, and offers ntp-toolkit for automatically extracting and annotating theorem proving data.
     </details>


     <details>
          <summary>Abstract</summary>
          We introduce miniCTX, which tests a model's ability to prove formal mathematical theorems that depend on new definitions, lemmas, or other contextual information that was not observed during training. miniCTX contains theorems sourced from real Lean projects and textbooks, each associated with a context that can span tens of thousands of tokens. Models are tasked with proving a theorem given access to code from the theorem's repository, which contains context that is helpful or needed for the proof. As a baseline for miniCTX, we introduce file-tuning, a simple recipe that trains a model to generate a proof step conditioned on the preceding file contents. File-tuning substantially outperforms the traditional neural theorem proving approach that fine-tunes on states alone. Additionally, our file-tuned model improves performance on the standard miniF2F benchmark, achieving a pass rate of 33.61%, which is a new state-of-the-art for 1.3B parameter models. Alongside miniCTX, we offer ntp-toolkit for automatically extracting and annotating theorem proving data, making it easy to add new projects into miniCTX to ensure that contexts are not seen during training. miniCTX offers a challenging and realistic perspective on evaluating neural theorem provers.
     </details>

244. **MathLearner: A Large Language Model Agent Framework for Learning to Solve Mathematical Problems** [[pdf]](http://arxiv.org/abs/2408.01779) `2024-08-03` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work has proposed an agent framework for learning to solve mathematical problems based on inductive reasoning that improves global accuracy over the baseline method (chain-of-thought) and solves 17.54% of the mathematical problems that the baseline cannot solve.
     </details>


     <details>
          <summary>Abstract</summary>
          With the development of artificial intelligence (AI), large language models (LLM) are widely used in many fields. However, the reasoning ability of LLM is still very limited when it comes to mathematical reasoning. Mathematics plays an important role in all aspects of human society and is a technical guarantee in the fields of healthcare, transport and aerospace, for this reason, the development of AI big language models in the field of mathematics has great potential significance. To improve the mathematical reasoning ability of large language models, we proposed an agent framework for learning to solve mathematical problems based on inductive reasoning. By emulating the human learning process of generalization of learned information and effective application of previous knowledge in new reasoning tasks, this framework has great performance in the mathematical reasoning process. It improves global accuracy over the baseline method (chain-of-thought) by 20.96% and solves 17.54% of the mathematical problems that the baseline cannot solve. Benefiting from the efficient RETRIEVAL method, our model improves the ability of large language models to efficiently use external knowledge, i.e., the mathematical computation of the model can be based on written procedures. In education, our model can be used as a personalised learning aid, thus reducing the inequality of educational resources.
     </details>

245. **GeoEval: Benchmark for Evaluating LLMs and Multi-Modal Models on Geometry Problem-Solving** [[pdf]](https://aclanthology.org/2024.findings-acl.73) `ACL 2024 Findings` (8 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The GeoEval benchmark is introduced, a comprehensive collection that includes a main subset of 2,000 problems, a 750 problems subset focusing on backward reasoning, an augmented subset of 2,000 problems, and a hard subset of 300 problems that facilitates a deeper investigation into the performance of LLMs and MMs in solving geometry math problems.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent advancements in large language models (LLMs) and multi-modal models (MMs) have demonstrated their remarkable capabilities in problem-solving. Yet, their proficiency in tackling geometry math problems, which necessitates an integrated understanding of both textual and visual information, has not been thoroughly evaluated. To address this gap, we introduce the GeoEval benchmark, a comprehensive collection that includes a main subset of 2,000 problems, a 750 problems subset focusing on backward reasoning, an augmented sub- set of 2,000 problems, and a hard subset of 300 problems. This benchmark facilitates a deeper investigation into the performance of LLMs and MMs in solving geometry math problems. Our evaluation of ten LLMs and MMs across these varied subsets reveals that the WizardMath model excels, achieving a 55.67% accuracy rate on the main subset but only a 6.00% accuracy on the hard subset. This highlights the critical need for testing models against datasets on which they have not been pre-trained. Additionally, our findings indicate that GPT-series models perform more effectively on problems they have rephrased, suggesting a promising method for enhancing model capabilities.
     </details>

246. **Small Language Models Need Strong Verifiers to Self-Correct Reasoning** [[pdf]](https://aclanthology.org/2024.findings-acl.924) `ACL 2024 Findings` (7 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work explores whether small (<= 13B) language models (LMs) have the ability of self-correction on reasoning tasks with minimal inputs from stronger LMs and proposes a novel pipeline that prompts smaller LMs to collect self-correction data that supports the training of self-refinement abilities.
     </details>


     <details>
          <summary>Abstract</summary>
          Self-correction has emerged as a promising solution to boost the reasoning performance of large language models (LLMs), where LLMs refine their solutions using self-generated critiques that pinpoint the errors. This work explores whether small (≤ 13B) language models (LMs) have the ability of self-correction on reasoning tasks with minimal inputs from stronger LMs. We propose a novel pipeline that prompts smaller LMs to collect self-correction data that supports the training of self-refinement abilities. First, we leverage correct solutions to guide the model in critiquing their incorrect responses. Second, the generated critiques, after filtering, are used for supervised fine-tuning of the self-correcting reasoner through solution refinement. Our experimental results show improved self-correction abilities of two models on five datasets spanning math and commonsense reasoning, with notable performance gains when paired with a strong GPT-4-based verifier, though limitations are identified when using a weak self-verifier for determining when to correct.
     </details>

247. **SceMQA: A Scientific College Entrance Level Multimodal Question Answering Benchmark** [[pdf]](https://aclanthology.org/2024.acl-short.11) `ACL 2024 Short Papers` (6 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          The paper introduces SceMQA, a novel benchmark for scientific multimodal question answering at the college entrance level. It addresses a critical educational phase often overlooked in existing benchmarks, spanning high school to pre-college levels. SceMQA focuses on core science subjects including Mathematics, Physics, Chemistry, and Biology. It features a blend of multiple-choice and free-response formats, ensuring a comprehensive evaluation of AI models’ abilities. Additionally, our benchmark provides specific knowledge points for each problem and detailed explanations for each answer. SceMQA also uniquely presents problems with identical contexts but varied questions to facilitate a more thorough and accurate assessment of reasoning capabilities. In the experiment, we evaluate both open-source and close-source state-of-the-art Multimodal Large Language Models (MLLMs), across various experimental settings. The results show that further research and development are needed in developing more capable MLLM, as highlighted by only 50% to 60% accuracy achieved by the strongest models.
     </details>

248. **LANS: A Layout-Aware Neural Solver for Plane Geometry Problem** [[pdf]](https://aclanthology.org/2024.findings-acl.153) `ACL 2024 Findings` (5 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A layout-aware neural solver named LANS is proposed, integrated with two new modules: multimodal layout-aware pre-trained language module (MLA-PLM) and layout-aware fusion attention (LA-FA), which employs a layout-aware attention mask to realize point-guided cross-modal fusion for further boosting layout awareness of LANS.
     </details>


     <details>
          <summary>Abstract</summary>
          Geometry problem solving (GPS) is a challenging mathematical reasoning task requiring multi-modal understanding, fusion, and reasoning. Existing neural solvers take GPS as a vision-language task but are short in the representation of geometry diagrams that carry rich and complex layout information. In this paper, we propose a layout-aware neural solver named LANS, integrated with two new modules: multimodal layout-aware pre-trained language module (MLA-PLM) and layout-aware fusion attention (LA-FA). MLA-PLM adopts structural-semantic pre-training (SSP) to implement global relationship modeling, and point-match pre-training (PMP) to achieve alignment between visual points and textual points. LA-FA employs a layout-aware attention mask to realize point-guided cross-modal fusion for further boosting layout awareness of LANS. Extensive experiments on datasets Geometry3K and PGPS9K validate the effectiveness of the layout-aware modules and superior problem-solving performance of our LANS solver, over existing symbolic and neural solvers. We have made our code and data publicly available.
     </details>

249. **Distilling Mathematical Reasoning Capabilities into Small Language Models** [[pdf]](http://arxiv.org/abs/2401.11864) `2024-08-01` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Equation-of-Thought Distillation (EoTD) is introduced, a novel technique that encapsulates the reasoning process into equation-based representations to construct an EoTD dataset for fine-tuning SLMs and the Ensemble Thoughts Distillation framework is proposed to enhance the reasoning performance of SLMs.
     </details>


     <details>
          <summary>Abstract</summary>
          This work addresses the challenge of democratizing advanced Large Language Models (LLMs) by compressing their mathematical reasoning capabilities into sub-billion parameter Small Language Models (SLMs) without compromising performance. We introduce Equation-of-Thought Distillation (EoTD), a novel technique that encapsulates the reasoning process into equation-based representations to construct an EoTD dataset for fine-tuning SLMs. Additionally, we propose the Ensemble Thoughts Distillation (ETD) framework to enhance the reasoning performance of SLMs. This involves creating a reasoning dataset with multiple thought processes, including Chain-of-Thought (CoT), Program-of-Thought (PoT), and Equation-of-Thought (EoT), and using it for fine-tuning. Our experimental performance demonstrates that EoTD significantly boosts the reasoning abilities of SLMs, while ETD enables these models to achieve state-of-the-art reasoning performance.
     </details>

250. **Meta-Reasoning: Semantics-Symbol Deconstruction for Large Language Models** [[pdf]](https://aclanthology.org/2024.findings-acl.34) `ACL 2024 Findings` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Experimental results demonstrate that Meta-Reasoning significantly enhances in-context reasoning accuracy, learning efficiency, out-of-domain generalization, and output stability compared to the Chain-of-Thought technique.
     </details>


     <details>
          <summary>Abstract</summary>
          Neural-symbolic methods have demonstrated efficiency in enhancing the reasoning abilities of large language models (LLMs). However, existing methods mainly rely on syntactically mapping natural languages to complete formal languages like Python and SQL. Those methods require that reasoning tasks be convertible into programs, which cater to the computer execution mindset and deviate from human reasoning habits. To broaden symbolic methods’ applicability and adaptability in the real world, we propose Meta-Reasoning from a linguistic perspective. This method empowers LLMs to deconstruct reasoning-independent semantic information into generic symbolic representations, thereby efficiently capturing more generalized reasoning knowledge. We conduct extensive experiments on more than ten datasets encompassing conventional reasoning tasks like arithmetic, symbolic, and logical reasoning, and the more complex interactive reasoning tasks like theory-of-mind reasoning. Experimental results demonstrate that Meta-Reasoning significantly enhances in-context reasoning accuracy, learning efficiency, out-of-domain generalization, and output stability compared to the Chain-of-Thought technique.
     </details>

251. **Prover-Verifier Games improve legibility of LLM outputs** [[pdf]](http://arxiv.org/abs/2407.13692) `ICLR 2025 Submission` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A training algorithm inspired by Prover-Verifier Game is proposed, which finds that the helpful prover's accuracy and the verifier's robustness to adversarial attacks increase over the course of training, and that legibility training transfers to time-constrained humans tasked with verifying solution correctness.
     </details>


     <details>
          <summary>Abstract</summary>
          One way to increase confidence in the outputs of Large Language Models (LLMs) is to support them with reasoning that is clear and easy to check -- a property we call legibility. We study legibility in the context of solving grade-school math problems and show that optimizing chain-of-thought solutions only for answer correctness can make them less legible. To mitigate the loss in legibility, we propose a training algorithm inspired by Prover-Verifier Game from Anil et al. (2021). Our algorithm iteratively trains small verifiers to predict solution correctness, "helpful" provers to produce correct solutions that the verifier accepts, and "sneaky" provers to produce incorrect solutions that fool the verifier. We find that the helpful prover's accuracy and the verifier's robustness to adversarial attacks increase over the course of training. Furthermore, we show that legibility training transfers to time-constrained humans tasked with verifying solution correctness. Over course of LLM training human accuracy increases when checking the helpful prover's solutions, and decreases when checking the sneaky prover's solutions. Hence, training for checkability by small verifiers is a plausible technique for increasing output legibility. Our results suggest legibility training against small verifiers as a practical avenue for increasing legibility of large LLMs to humans, and thus could help with alignment of superhuman models.
     </details>

252. **Evaluating LLMs’ Mathematical Reasoning in Financial Document Question Answering** [[pdf]](https://aclanthology.org/2024.findings-acl.231) `ACL 2024 Findings` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel prompting technique is introduced tailored to semi-structured documents, matching or outperforming other baselines in performance while providing a nuanced understanding of LLMs abilities for such a task.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs), excel in natural language understanding, but their capability for complex mathematical reasoning with a hybrid of structured tables and unstructured text remain uncertain. This study explores LLMs’ mathematical reasoning on four financial tabular question-answering datasets: TATQA, FinQA, ConvFinQA, and Multihiertt. Through extensive experiments with various models and prompting techniques, we assess how LLMs adapt to complex tables and mathematical tasks. We focus on sensitivity to table complexity and performance variations with an increasing number of arithmetic reasoning steps. The results provide insights into LLMs’ capabilities and limitations in handling complex mathematical scenarios for semi-structured tables. Ultimately, we introduce a novel prompting technique EEDP tailored to semi-structured documents, matching or outperforming baselines performance while providing a nuanced understanding of LLMs abilities.
     </details>

253. **BBA: Bi-Modal Behavioral Alignment for Reasoning with Large Vision-Language Models** [[pdf]](https://aclanthology.org/2024.findings-acl.433) `ACL 2024 Findings` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The BBA prompting method is introduced, designed to maximize the potential of DSL in augmenting complex multi-modal reasoning tasks, and substantially improves the performance of GPT-4V(ision) on geometry problem solving, chess positional advantage prediction and molecular property prediction.
     </details>


     <details>
          <summary>Abstract</summary>
          Multimodal reasoning stands as a pivotal capability for large vision-language models (LVLMs). The integration with Domain-Specific Languages (DSL), offering precise visual representations, equips these models with the opportunity to execute more accurate reasoning in complex and professional domains. However, the vanilla Chain-of-Thought (CoT) prompting method faces challenges in effectively leveraging the unique strengths of visual and DSL representations, primarily due to their differing reasoning mechanisms. Additionally, it often falls short in addressing critical steps in multi-step reasoning tasks. To mitigate these challenges, we introduce the Bi-Modal Behavioral Alignment (BBA) prompting method, designed to maximize the potential of DSL in augmenting complex multi-modal reasoning tasks. This method initiates by guiding LVLMs to create separate reasoning chains for visual and DSL representations. Subsequently, it aligns these chains by addressing any inconsistencies, thus achieving a cohesive integration of behaviors from different modalities. Our experiments demonstrate that BBA substantially improves the performance of GPT-4V(ision) on geometry problem solving (28.34% → 34.22%), chess positional advantage prediction (42.08% → 46.99%) and molecular property prediction (77.47% → 83.52%).
     </details>

254. **Bi-Chainer: Automated Large Language Models Reasoning with Bidirectional Chaining** [[pdf]](https://aclanthology.org/2024.findings-acl.507) `ACL 2024 Findings` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A bidirectional chaining method, Bi-Chainer, which dynamically switches to depth-first reasoning in the opposite reasoning direction when it encounters multiple branching options within the current direction, resulting in more efficient and accurate reasoning.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have shown human-like reasoning abilities but still face challenges in solving complex logical problems. Existing unidirectional chaining methods, such as forward chaining and backward chaining, suffer from issues like low prediction accuracy and efficiency. To address these, we propose a bidirectional chaining method, Bi-Chainer, which dynamically switches to depth-first reasoning in the opposite reasoning direction when it encounters multiple branching options within the current direction. Thus, the intermediate reasoning results can be utilized as guidance to facilitate the reasoning process. We show that Bi-Chainer achieves sizable accuracy boots over unidirectional chaining frameworks on four challenging logical reasoning datasets. Moreover, Bi-Chainer enhances the accuracy of intermediate proof steps and reduces the average number of inference calls, resulting in more efficient and accurate reasoning.
     </details>

255. **Language Models Do Hard Arithmetic Tasks Easily and Hardly Do Easy Arithmetic Tasks** [[pdf]](https://aclanthology.org/2024.acl-short.8) `ACL 2024 Short Papers` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is shown that LLMs are frequently able to correctly and confidently predict the first digit of n-digit by m-digit multiplication tasks without using chain of thought reasoning, despite these tasks require compounding operations to solve.
     </details>


     <details>
          <summary>Abstract</summary>
          The ability (and inability) of large language models (LLMs) to perform arithmetic tasks has been the subject of much theoretical and practical debate. We show that LLMs are frequently able to correctly and confidently predict the first digit of n-digit by m-digit multiplication tasks without using chain of thought reasoning, despite these tasks require compounding operations to solve. Simultaneously, LLMs in practice often fail to correctly or confidently predict the last digit of an n-digit by m-digit multiplication, a task equivalent to 1-digit by 1-digit multiplication which can be easily learned or memorized. We show that the latter task can be solved more robustly when the LLM is conditioned on all of the correct higher-order digits, which on average increases the confidence of the correct last digit on 5-digit by 5-digit multiplication tasks using Llama 2-13B by over 230% (0.13→0.43) and Mistral-7B by 150% (0.22→0.55).
     </details>

256. **NUMCoT: Numerals and Units of Measurement in Chain-of-Thought Reasoning using Large Language Models** [[pdf]](https://aclanthology.org/2024.findings-acl.848) `ACL 2024 Findings` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper anatomizes the reasoning of math word problems to different sub-procedures like numeral conversions from language to numbers and measurement conversions based on units and demonstrates that LLMs still encounter difficulties in handling numeral and measurement conversions.
     </details>


     <details>
          <summary>Abstract</summary>
          Numeral systems and units of measurement are two conjoined topics in activities of human beings and have mutual effects with the languages expressing them. Currently, the evaluation of Large Language Models (LLMs) often involves mathematical reasoning, yet little attention is given to how minor changes in numbers or units can drastically alter the complexity of problems and the performance of LLMs. In this paper, we scrutinize existing LLMs on processing of numerals and units of measurement by constructing datasets with perturbations. We first anatomize the reasoning of math word problems to different sub-procedures like numeral conversions from language to numbers and measurement conversions based on units. Then we further annotate math word problems from ancient Chinese arithmetic works which are challenging in numerals and units of measurement. Experiments on perturbed datasets demonstrate that LLMs still encounter difficulties in handling numeral and measurement conversions.
     </details>

257. **Question-Analysis Prompting Improves LLM Performance in Reasoning Tasks** [[pdf]](https://aclanthology.org/2024.acl-srw.45) `ACL 2024 Student Research Workshop` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This research proposes a novel prompting strategy called Question Analysis Prompting (QAP), in which the model is prompted to explain the question in words before solving, where detailed responses are beneficial when answering harder questions, but can negatively affect easy questions.
     </details>


     <details>
          <summary>Abstract</summary>
          Although LLMs have the potential to transform many fields, they still underperform humans in reasoning tasks. Existing methods induce the model to produce step-by-step calculations, but this research explores the question: Does making the LLM analyze the question improve its performance? We propose a novel prompting strategy called Question Analysis Prompting (QAP), in which the model is prompted to explain the question in ’n’ words before solving. The value of ’n’ influences the length of response generated by the model. QAP is evaluated on GPT-3.5 Turbo and GPT-4 Turbo on arithmetic datasets GSM8K, AQuA, and SAT and commonsense dataset StrategyQA. QAP is compared with other state-of-the-art prompts including chain-of-thought (CoT), Plan and Solve Prompting (PS+) and Take A Deep Breath (TADB). QAP outperforms all state-of-the-art prompts on AQuA and SAT datasets on both GPT-3.5 and GPT-4. QAP consistently ranks among the top-2 prompts on 75% of the tests. A key factor of QAP performance can be attributed to response length, where detailed responses are beneficial when answering harder questions, but can negatively affect easy questions.
     </details>

258. **Large Language Monkeys: Scaling Inference Compute with Repeated Sampling** [[pdf]](http://arxiv.org/abs/2407.21787) `ICLR 2025 Submission` (21 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Inference compute is explored as another axis for scaling by increasing the number of generated samples across multiple tasks and models, observing that coverage - the fraction of problems solved by any attempt - scales with the number of samples over four orders of magnitude.
     </details>


     <details>
          <summary>Abstract</summary>
          Scaling the amount of compute used to train language models has dramatically improved their capabilities. However, when it comes to inference, we often limit the amount of compute to only one attempt per problem. Here, we explore inference compute as another axis for scaling by increasing the number of generated samples. Across multiple tasks and models, we observe that coverage - the fraction of problems solved by any attempt - scales with the number of samples over four orders of magnitude. In domains like coding and formal proofs, where all answers can be automatically verified, these increases in coverage directly translate into improved performance. When we apply repeated sampling to SWE-bench Lite, the fraction of issues solved with DeepSeek-V2-Coder-Instruct increases from 15.9% with one sample to 56% with 250 samples, outperforming the single-attempt state-of-the-art of 43% which uses more capable frontier models. Moreover, using current API pricing, amplifying the cheaper DeepSeek model with five samples is more cost-effective and solves more issues than paying a premium for one sample from GPT-4o or Claude 3.5 Sonnet. Interestingly, the relationship between coverage and the number of samples is often log-linear and can be modelled with an exponentiated power law, suggesting the existence of inference-time scaling laws. Finally, we find that identifying correct samples out of many generations remains an important direction for future research in domains without automatic verifiers. When solving math word problems from GSM8K and MATH, coverage with Llama-3 models grows to over 95% with 10,000 samples. However, common methods to pick correct solutions from a sample collection, such as majority voting or reward models, plateau beyond several hundred samples and fail to fully scale with the sample budget.
     </details>

259. **Inductive or Deductive? Rethinking the Fundamental Reasoning Abilities of LLMs** [[pdf]](https://arxiv.org/abs/2408.00114v2) `ICLR 2025 Submission` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a novel framework, SolverLearner, that enables LLMs to learn the underlying function of inductive reasoning, and reveals that LLMs demonstrate remarkable inductive reasoning capabilities through SolverLearner, achieving near-perfect performance with ACC of 1 in most cases.
     </details>


     <details>
          <summary>Abstract</summary>
          Reasoning encompasses two typical types: deductive reasoning and inductive reasoning. Despite extensive research into the reasoning capabilities of Large Language Models (LLMs), most studies have failed to rigorously differentiate between inductive and deductive reasoning, leading to a blending of the two. This raises an essential question: In LLM reasoning, which poses a greater challenge - deductive or inductive reasoning? While the deductive reasoning capabilities of LLMs, (i.e. their capacity to follow instructions in reasoning tasks), have received considerable attention, their abilities in true inductive reasoning remain largely unexplored. To investigate into the true inductive reasoning capabilities of LLMs, we propose a novel framework, SolverLearner. This framework enables LLMs to learn the underlying function (i.e., $y = f_w(x)$), that maps input data points $(x)$ to their corresponding output values $(y)$, using only in-context examples. By focusing on inductive reasoning and separating it from LLM-based deductive reasoning, we can isolate and investigate inductive reasoning of LLMs in its pure form via SolverLearner. Our observations reveal that LLMs demonstrate remarkable inductive reasoning capabilities through SolverLearner, achieving near-perfect performance with ACC of 1 in most cases. Surprisingly, despite their strong inductive reasoning abilities, LLMs tend to relatively lack deductive reasoning capabilities, particularly in tasks involving ``counterfactual'' reasoning.
     </details>

260. **AI-Assisted Generation of Difficult Math Questions** [[pdf]](http://arxiv.org/abs/2407.21009) `NeurIPS 2024 Workshop MATH-AI` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A design framework that combines the strengths of LLMs with a human-in-the-loop approach to generate a diverse array of challenging math questions and a striking relationship observed between models' performance on the new dataset, suggesting that successfully solving the question in MATH$^2$ requires a nontrivial combination of two distinct math skills.
     </details>


     <details>
          <summary>Abstract</summary>
          Current LLM training positions mathematical reasoning as a core capability. With publicly available sources fully tapped, there is unmet demand for diverse and challenging math questions. Relying solely on human experts is both time-consuming and costly, while LLM-generated questions often lack the requisite diversity and difficulty. We present a design framework that combines the strengths of LLMs with a human-in-the-loop approach to generate a diverse array of challenging math questions. We leverage LLM metacognition skills [Didolkar et al., 2024] of a strong LLM to extract core "skills" from existing math datasets. These skills serve as the basis for generating novel and difficult questions by prompting the LLM with random pairs of core skills. The use of two different skills within each question makes finding such questions an "out of distribution" task for both LLMs and humans. Our pipeline employs LLMs to iteratively generate and refine questions and solutions through multiturn prompting. Human annotators then verify and further refine the questions, with their efficiency enhanced via further LLM interactions. Applying this pipeline on skills extracted from the MATH dataset [Hendrycks et al., 2021] resulted in MATH$^2$ - a dataset of higher-quality math questions, as evidenced by: (a) Lower performance of all models on MATH$^2$ than on MATH (b) Higher performance on MATH when using MATH$^2$ questions as in-context examples. Although focused on mathematics, our methodology seems applicable to other domains requiring structured reasoning, and potentially as a component of scalable oversight. Also of interest is a striking relationship observed between models' performance on the new dataset: the success rate on MATH$^2$ is the square on MATH, suggesting that successfully solving the question in MATH$^2$ requires a nontrivial combination of two distinct math skills.
     </details>

261. **Key-Point-Driven Mathematical Reasoning Distillation of Large Language Model** [[pdf]](http://arxiv.org/abs/2407.10167) `2024-07-30` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          KPDD enhances the reasoning performance of SLMs by breaking down the problem-solving process into three stages: Core Question Extraction, Problem-Solving Information Extraction, and Step-by-Step Solution, and KPDD-PoT achieves state-of-the-art performance in mathematical reasoning tasks.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have demonstrated exceptional proficiency in mathematical reasoning tasks due to their extensive parameter counts and training on vast datasets. Despite these capabilities, deploying LLMs is hindered by their computational demands. Distilling LLM mathematical reasoning into Smaller Language Models (SLMs) has emerged as a solution to this challenge, although these smaller models often suffer from errors in calculation and semantic understanding. Prior work has proposed Program-of-Thought Distillation (PoTD) to avoid calculation error. To further address semantic understanding errors, we propose Key-Point-Driven Mathematical Reasoning Distillation (KPDD). KPDD enhances the reasoning performance of SLMs by breaking down the problem-solving process into three stages: Core Question Extraction, Problem-Solving Information Extraction, and Step-by-Step Solution. This method is further divided into KPDD-CoT, which generates Chain-of-Thought rationales, and KPDD-PoT, which creates Program-of-Thought rationales. The experiment results show that KPDD-CoT significantly improves reasoning abilities, while KPDD-PoT achieves state-of-the-art performance in mathematical reasoning tasks. Our approach effectively mitigates misunderstanding errors, advancing the deployment of efficient and capable SLMs.
     </details>

262. **Physics of Language Models: Part 2.1, Grade-School Math and the Hidden Reasoning Process** [[pdf]](http://arxiv.org/abs/2407.20311) `2024-07-29` (8 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Recent advances in language models have demonstrated their capability to solve mathematical reasoning problems, achieving near-perfect accuracy on grade-school level math benchmarks like GSM8K. In this paper, we formally study how language models solve these problems. We design a series of controlled experiments to address several fundamental questions: (1) Can language models truly develop reasoning skills, or do they simply memorize templates? (2) What is the model's hidden (mental) reasoning process? (3) Do models solve math questions using skills similar to or different from humans? (4) Do models trained on GSM8K-like datasets develop reasoning skills beyond those necessary for solving GSM8K problems? (5) What mental process causes models to make reasoning mistakes? (6) How large or deep must a model be to effectively solve GSM8K-level math questions?   Our study uncovers many hidden mechanisms by which language models solve mathematical questions, providing insights that extend beyond current understandings of LLMs.
     </details>

263. **PATCH! Psychometrics-AssisTed benCHmarking of Large Language Models: A Case Study of Proficiency in 8th Grade Mathematics** [[pdf]](http://arxiv.org/abs/2404.01799) `2024-07-25` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Many existing benchmarks of large (multimodal) language models (LLMs) focus on measuring LLMs' academic proficiency, often with also an interest in comparing model performance with human test takers. While these benchmarks have proven key to the development of LLMs, they suffer from several limitations, including questionable measurement quality (e.g., Do they measure what they are supposed to in a reliable way?), lack of quality assessment on the item level (e.g., Are some items more important or difficult than others?) and unclear human population reference (e.g., To whom can the model be compared?). In response to these challenges, we propose leveraging knowledge from psychometrics - a field dedicated to the measurement of latent variables like academic proficiency - into LLM benchmarking. We make three primary contributions. First, we introduce PATCH: a novel framework for {P}sychometrics-{A}ssis{T}ed ben{CH}marking of LLMs. PATCH addresses the aforementioned limitations, presenting a new direction for LLM benchmark research. Second, we implement PATCH by measuring GPT-4 and Gemini-Pro-Vision's proficiency in 8th grade mathematics against 56 human populations. We show that adopting a psychometrics-based approach yields evaluation outcomes that diverge from those based on existing benchmarking practices. Third, we release 4 high-quality datasets to support measuring and comparing LLM proficiency in grade school mathematics and science against human populations.
     </details>

264. **Relating the Seemingly Unrelated: Principled Understanding of Generalization for Generative Models in Arithmetic Reasoning Tasks** [[pdf]](https://arxiv.org/abs/2407.17963v1) `2024-07-25` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Extensive experiments with GPT-like models validate the theoretical predictions and deepen the understanding of the generalization mechanisms, and facilitate more data-efficient model training and objective-oriented AI alignment.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have demonstrated impressive versatility across numerous tasks, yet their generalization capabilities remain poorly understood. To investigate these behaviors, arithmetic tasks serve as important venues. In previous studies, seemingly unrelated mysteries still exist -- (1) models with appropriate positional embeddings can correctly perform longer unseen arithmetic operations such as addition, but their effectiveness varies in more complex tasks like multiplication; (2) models perform well for longer unseen cases in modular addition under specific moduli (e.g., modulo 100) but struggle under very close moduli (e.g., modulo 101), regardless of the positional encoding used. We believe previous studies have been treating the symptoms rather than addressing the root cause -- they have paid excessive attention to improving model components, while overlooking the differences in task properties that may be the real drivers. This is confirmed by our unified theoretical framework for different arithmetic scenarios. For example, unlike multiplication, the digital addition task has the property of translation invariance which naturally aligns with the relative positional encoding, and this combination leads to successful generalization of addition to unseen longer domains. The discrepancy in operations modulo 100 and 101 arises from the base. Modulo 100, unlike 101, is compatible with the decimal system (base 10), such that unseen information in digits beyond the units digit and the tens digit is actually not needed for the task. Extensive experiments with GPT-like models validate our theoretical predictions. These findings deepen our understanding of the generalization mechanisms, and facilitate more data-efficient model training and objective-oriented AI alignment.
     </details>

265. **LEAN-GitHub: Compiling GitHub LEAN repositories for a versatile LEAN prover** [[pdf]](http://arxiv.org/abs/2407.17227) `2024-07-24` `Lean` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes LEAN-GitHub, a dataset consisting of large-scale formal data extracted from almost all Lean 4 repositories on GitHub, and achieves state-of-the-art on two other Lean 4 benchmarks targeting different fields/levels of math.
     </details>


     <details>
          <summary>Abstract</summary>
          Recently, large language models have presented promising results in aiding formal mathematical reasoning. However, their performance is restricted due to the scarcity of formal theorem-proving data, which requires additional effort to be extracted from raw formal language corpora. Meanwhile, a significant amount of human-written formal language corpora remains underutilized. To address this issue, we propose LEAN-GitHub, a dataset consisting of large-scale formal data extracted from almost all Lean 4 repositories on GitHub. After fine-tuning InternLM-math-plus on this dataset, our model achieved accuracies of 48.8% with a single pass and 54.5% with 64 passes on the Lean 4 miniF2F test, surpassing state-of-the-art method at 52%. And it also achieves state-of-the-art on two other Lean 4 benchmarks (ProofNet and Putnam) targeting different fields/levels of math. These results demonstrate that our proposed dataset is beneficial for formal reasoning on a wide range of math topics. We open-source our model at https://GitHub. com/InternLM/InternLM-Math and our data at https://huggingface.co/ datasets/InternLM/Lean-GitHub
     </details>

266. **Boosting Large Language Models with Socratic Method for Conversational Mathematics Teaching** [[pdf]](https://arxiv.org/abs/2407.17349v1) `2024-07-24` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper proposes a knowledge-enhanced LLM as a strong baseline to generate reliable responses with review, guidance/heuristic, rectification, and summarization, and shows the great advantages of \texttt{SocraticLLM} by comparing it with several strong generative models.
     </details>


     <details>
          <summary>Abstract</summary>
          With the introduction of large language models (LLMs), automatic math reasoning has seen tremendous success. However, current methods primarily focus on providing solutions or using techniques like Chain-of-Thought to enhance problem-solving accuracy. In this paper, we focus on improving the capability of mathematics teaching via a Socratic teaching-based LLM (\texttt{SocraticLLM}), which guides learners toward profound thinking with clarity and self-discovery via conversation. We collect and release a high-quality mathematical teaching dataset, named \texttt{SocraticMATH}, which provides Socratic-style conversations of problems with extra knowledge. Also, we propose a knowledge-enhanced LLM as a strong baseline to generate reliable responses with review, guidance/heuristic, rectification, and summarization. Experimental results show the great advantages of \texttt{SocraticLLM} by comparing it with several strong generative models. The codes and datasets are available on \url{https://github.com/ECNU-ICALK/SocraticMath}.
     </details>

267. **MathViz-E: A Case-study in Domain-Specialized Tool-Using Agents** [[pdf]](http://arxiv.org/abs/2407.17544) `2024-07-24` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          An automated math visualizer and solver system for mathematical pedagogy that orchestrates mathematical solvers and math graphing tools to produce accurate visualizations from simple natural language commands is presented.
     </details>


     <details>
          <summary>Abstract</summary>
          There has been significant recent interest in harnessing LLMs to control software systems through multi-step reasoning, planning and tool-usage. While some promising results have been obtained, application to specific domains raises several general issues including the control of specialized domain tools, the lack of existing datasets for training and evaluation, and the non-triviality of automated system evaluation and improvement. In this paper, we present a case-study where we examine these issues in the context of a specific domain. Specifically, we present an automated math visualizer and solver system for mathematical pedagogy. The system orchestrates mathematical solvers and math graphing tools to produce accurate visualizations from simple natural language commands. We describe the creation of specialized data-sets, and also develop an auto-evaluator to easily evaluate the outputs of our system by comparing them to ground-truth expressions. We have open sourced the data-sets and code for the proposed system.
     </details>

268. **Lean-STaR: Learning to Interleave Thinking and Proving** [[pdf]](http://arxiv.org/abs/2407.10040) `NeurIPS 2024 Workshop MATH-AI` `Lean` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work presents Lean-STaR, a framework for training language models to produce informal thoughts prior to each step of a proof, thereby boosting the model's theorem-proving capabilities.
     </details>


     <details>
          <summary>Abstract</summary>
          Traditional language model-based theorem proving assumes that by training on a sufficient amount of formal proof data, a model will learn to prove theorems. Our key observation is that a wealth of informal information that is not present in formal proofs can be useful for learning to prove theorems. For instance, humans think through steps of a proof, but this thought process is not visible in the resulting code. We present Lean-STaR, a framework for training language models to produce informal thoughts prior to each step of a proof, thereby boosting the model's theorem-proving capabilities. Lean-STaR uses retrospective ground-truth tactics to generate synthetic thoughts for training the language model. At inference time, the trained model directly generates the thoughts prior to the prediction of the tactics in each proof step. Building on the self-taught reasoner framework, we then apply expert iteration to further fine-tune the model on the correct proofs it samples and verifies using the Lean solver. Lean-STaR achieves state-of-the-art results on the miniF2F-test benchmark within the Lean theorem proving environment, significantly outperforming base models ($\boldsymbol{43.4\% \rightarrow 46.3\%,}$ Pass@64). We also analyze the impact of the augmented thoughts on various aspects of the theorem proving process, providing insights into their effectiveness.
     </details>

269. **Q*: Improving Multi-step Reasoning for LLMs with Deliberative Planning** [[pdf]](http://arxiv.org/abs/2406.14283) `ICLR 2025 Submission` (4 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          By casting multi-step reasoning of LLMs as a heuristic search problem, Q*, a general, versatile and agile framework for guiding LLMs decoding process with deliberative planning is introduced, which can effectively guide LLMs to select the most promising next reasoning step without fine-tuning LLMs for the current task.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have demonstrated impressive capability in many natural language tasks. However, the auto-regressive generation process makes LLMs prone to produce errors, hallucinations and inconsistent statements when performing multi-step reasoning. In this paper, by casting multi-step reasoning of LLMs as a heuristic search problem, we aim to alleviate the pathology by introducing Q*, a general, versatile and agile framework for guiding LLMs decoding process with deliberative planning. By learning a plug-and-play Q-value model as heuristic function for estimating expected future rewards, our Q* can effectively guide LLMs to select the most promising next reasoning step without fine-tuning LLMs for the current task, which avoids the significant computational overhead and potential risk of performance degeneration on other tasks. Extensive experiments on GSM8K, MATH and MBPP demonstrate the superiority of our method, contributing to improving the reasoning performance of existing open-source LLMs.
     </details>

270. **TaskGen: A Task-Based, Memory-Infused Agentic Framework using StrictJSON** [[pdf]](http://arxiv.org/abs/2407.15734) `2024-07-22` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work empirically evaluates TaskGen on various environments such as 40x40 dynamic maze navigation with changing obstacle locations, TextWorld escape room solving with dense rewards and detailed goals, web browsing, and Retrieval Augmented Generation on NaturalQuestions dataset.
     </details>


     <details>
          <summary>Abstract</summary>
          TaskGen is an open-sourced agentic framework which uses an Agent to solve an arbitrary task by breaking them down into subtasks. Each subtask is mapped to an Equipped Function or another Agent to execute. In order to reduce verbosity (and hence token usage), TaskGen uses StrictJSON that ensures JSON output from the Large Language Model (LLM), along with additional features such as type checking and iterative error correction. Key to the philosophy of TaskGen is the management of information/memory on a need-to-know basis. We empirically evaluate TaskGen on various environments such as 40x40 dynamic maze navigation with changing obstacle locations (100% solve rate), TextWorld escape room solving with dense rewards and detailed goals (96% solve rate), web browsing (69% of actions successful), solving the MATH dataset (71% solve rate over 100 Level-5 problems), Retrieval Augmented Generation on NaturalQuestions dataset (F1 score of 47.03%)
     </details>

271. **Reliable Reasoning Beyond Natural Language** [[pdf]](http://arxiv.org/abs/2407.11373) `2024-07-19` `Prolog` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a neurosymbolic approach that prompts LLMs to extract and encode all relevant information from a problem statement as logical code statements, and then uses a logic programming language (Prolog) to conduct the iterative computations of explicit deductive reasoning.
     </details>


     <details>
          <summary>Abstract</summary>
          Despite their linguistic competence, Large Language models (LLMs) often exhibit limitations in their ability to reason reliably and flexibly. To address this, we propose a neurosymbolic approach that prompts LLMs to extract and encode all relevant information from a problem statement as logical code statements, and then use a logic programming language (Prolog) to conduct the iterative computations of explicit deductive reasoning. Our approach significantly enhances the performance of LLMs on the standard mathematical reasoning benchmark, GSM8k, and the Navigate dataset from the BIG-bench dataset. Additionally, we introduce a novel dataset, the Non-Linear Reasoning (NLR) dataset, consisting of 55 unique word problems that target the shortcomings of the next token prediction paradigm of LLMs and require complex non-linear reasoning but only basic arithmetic skills to solve. Our findings demonstrate that the integration of Prolog enables LLMs to achieve high performance on the NLR dataset, which even the most advanced language models (including GPT4) fail to solve using text only.
     </details>

272. **Learning From Correctness Without Prompting Makes LLM Efficient Reasoner** [[pdf]](http://arxiv.org/abs/2403.19094) `2024-07-18` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduces an intrinsic self-correct reasoning framework for LLMs that eliminates the need for human feedback, external tools, and handcraft prompts, and improves reasoning performance without needing to learn from errors.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have demonstrated outstanding performance across various tasks, yet they still exhibit limitations such as hallucination, unfaithful reasoning, and toxic content. One potential approach to mitigate these issues is learning from human or external feedback (e.g. tools). In this paper, we introduce an intrinsic self-correct reasoning framework for LLMs that eliminates the need for human feedback, external tools, and handcraft prompts. The proposed framework, based on a multi-step reasoning paradigm \textbf{Le}arning from \textbf{Co}rrectness (\textsc{LeCo}), improves reasoning performance without needing to learn from errors. This paradigm prioritizes learning from correct reasoning steps, and a unique method to measure confidence for each reasoning step based on generation logits. Experimental results across various multi-step reasoning tasks demonstrate the effectiveness of the framework in improving reasoning performance with reduced token consumption.
     </details>

273. **Weak-to-Strong Reasoning** [[pdf]](http://arxiv.org/abs/2407.13647) `2024-07-18` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A progressive learning framework that enables the strong model to autonomously refine its training data, without requiring input from either a more advanced model or human-annotated data is introduced.
     </details>


     <details>
          <summary>Abstract</summary>
          When large language models (LLMs) exceed human-level capabilities, it becomes increasingly challenging to provide full-scale and accurate supervision for these models. Weak-to-strong learning, which leverages a less capable model to unlock the latent abilities of a stronger model, proves valuable in this context. Yet, the efficacy of this approach for complex reasoning tasks is still untested. Furthermore, tackling reasoning tasks under the weak-to-strong setting currently lacks efficient methods to avoid blindly imitating the weak supervisor including its errors. In this paper, we introduce a progressive learning framework that enables the strong model to autonomously refine its training data, without requiring input from either a more advanced model or human-annotated data. This framework begins with supervised fine-tuning on a selective small but high-quality dataset, followed by preference optimization on contrastive samples identified by the strong model itself. Extensive experiments on the GSM8K and MATH datasets demonstrate that our method significantly enhances the reasoning capabilities of Llama2-70b using three separate weak models. This method is further validated in a forward-looking experimental setup, where Llama3-8b-instruct effectively supervises Llama3-70b on the highly challenging OlympicArena dataset. This work paves the way for a more scalable and sophisticated strategy to enhance AI reasoning powers. All relevant code and resources are available in \url{https://github.com/GAIR-NLP/weak-to-strong-reasoning}.
     </details>

274. **Learning Goal-Conditioned Representations for Language Reward Models** [[pdf]](http://arxiv.org/abs/2407.13887) `2024-07-18` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes training reward models in a contrastive, goal-conditioned fashion by increasing the representation similarity of future states along sampled preferred trajectories and decreasing the similarity along randomly sampled dispreferred trajectories, and finds that these representations can perform fine-grained control by conditioning on desired future goal-states.
     </details>


     <details>
          <summary>Abstract</summary>
          Techniques that learn improved representations via offline data or self-supervised objectives have shown impressive results in traditional reinforcement learning (RL). Nevertheless, it is unclear how improved representation learning can benefit reinforcement learning from human feedback (RLHF) on language models (LMs). In this work, we propose training reward models (RMs) in a contrastive, $\textit{goal-conditioned}$ fashion by increasing the representation similarity of future states along sampled preferred trajectories and decreasing the similarity along randomly sampled dispreferred trajectories. This objective significantly improves RM performance by up to 0.09 AUROC across challenging benchmarks, such as MATH and GSM8k. These findings extend to general alignment as well -- on the Helpful-Harmless dataset, we observe $2.3\%$ increase in accuracy. Beyond improving reward model performance, we show this way of training RM representations enables improved $\textit{steerability}$ because it allows us to evaluate the likelihood of an action achieving a particular goal-state (e.g., whether a solution is correct or helpful). Leveraging this insight, we find that we can filter up to $55\%$ of generated tokens during majority voting by discarding trajectories likely to end up in an "incorrect" state, which leads to significant cost savings. We additionally find that these representations can perform fine-grained control by conditioning on desired future goal-states. For example, we show that steering a Llama 3 model towards helpful generations with our approach improves helpfulness by $9.6\%$ over a supervised-fine-tuning trained baseline. Similarly, steering the model towards complex generations improves complexity by $21.6\%$ over the baseline. Overall, we find that training RMs in this contrastive, goal-conditioned fashion significantly improves performance and enables model steerability.
     </details>

275. **Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate** [[pdf]](http://arxiv.org/abs/2305.19118) `2024-07-17` (204 cite) (8 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A Multi-Agent Debate (MAD) framework is proposed, in which multiple agents express their arguments in the state of"tit for tat"and a judge manages the debate process to obtain a final solution.
     </details>


     <details>
          <summary>Abstract</summary>
          Modern large language models (LLMs) like ChatGPT have shown remarkable performance on general language tasks but still struggle on complex reasoning tasks, which drives the research on cognitive behaviors of LLMs to explore human-like problem-solving strategies. Along this direction, one representative strategy is self-reflection, which asks an LLM to refine the solution with the feedback generated by itself iteratively. However, our study shows that such reflection-style methods suffer from the Degeneration-of-Thought (DoT) problem: once the LLM has established confidence in its solutions, it is unable to generate novel thoughts later through reflection even if its initial stance is incorrect. To address the DoT problem, we propose a Multi-Agent Debate (MAD) framework, in which multiple agents express their arguments in the state of "tit for tat" and a judge manages the debate process to obtain a final solution. Clearly, our MAD framework encourages divergent thinking in LLMs which would be helpful for tasks that require deep levels of contemplation. Experiment results on two challenging datasets, commonsense machine translation and counter-intuitive arithmetic reasoning, demonstrate the effectiveness of our MAD framework. Extensive analyses suggest that the adaptive break of debate and the modest level of "tit for tat" state are required for MAD to obtain good performance. Moreover, we find that LLMs might not be a fair judge if different LLMs are used for agents. Code is available at https://github.com/Skytliang/Multi-Agents-Debate.
     </details>

276. **DotaMath: Decomposition of Thought with Code Assistance and Self-correction for Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2407.04078) `2024-07-17` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduces a series of LLMs that employs the Decomposition of thought with code assistance and self-correction for mathematical reasoning, dubbed as DotaMath, which achieve remarkable performance compared to open-source LLMs across various in-domain and out-of-domain benchmarks.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have made impressive progress in handling simple math problems, yet they still struggle with more challenging and complex mathematical tasks. In this paper, we introduce a series of LLMs that employs the Decomposition of thought with code assistance and self-correction for mathematical reasoning, dubbed as DotaMath. DotaMath models tackle complex mathematical tasks by decomposing them into simpler logical subtasks, leveraging code to solve these subtasks, obtaining fine-grained feedback from the code interpreter, and engaging in self-reflection and correction. By annotating diverse interactive tool-use trajectories and employing query evolution on GSM8K and MATH datasets, we generate an instruction fine-tuning dataset called DotaMathQA with 574K query-response pairs. We train a series of base LLMs using imitation learning on DotaMathQA, resulting in DotaMath models that achieve remarkable performance compared to open-source LLMs across various in-domain and out-of-domain benchmarks. Notably, DotaMath-deepseek-7B showcases an outstanding performance of 64.8% on the competitive MATH dataset and 86.7% on GSM8K. Besides, DotaMath-deepseek-7B maintains strong competitiveness on a series of in-domain and out-of-domain benchmarks (Avg. 80.1%). Looking forward, we anticipate that the DotaMath paradigm will open new pathways for addressing intricate mathematical problems. Our code is publicly available at https://github.com/ChengpengLi1003/DotaMath.
     </details>

277. **Skywork-Math: Data Scaling Laws for Mathematical Reasoning in Large Language Models -- The Story Goes On** [[pdf]](http://arxiv.org/abs/2407.08348) `2024-07-17` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          In this paper, we investigate the underlying factors that potentially enhance the mathematical reasoning capabilities of large language models (LLMs). We argue that the data scaling law for math reasoning capabilities in modern LLMs is far from being saturated, highlighting how the model's quality improves with increases in data quantity. To support this claim, we introduce the Skywork-Math model series, supervised fine-tuned (SFT) on common 7B LLMs using our proposed 2.5M-instance Skywork-MathQA dataset. Skywork-Math 7B has achieved impressive accuracies of 51.2% on the competition-level MATH benchmark and 83.9% on the GSM8K benchmark using only SFT data, outperforming an early version of GPT-4 on MATH. The superior performance of Skywork-Math models contributes to our novel two-stage data synthesis and model SFT pipelines, which include three different augmentation methods and a diverse seed problem set, ensuring both the quantity and quality of Skywork-MathQA dataset across varying difficulty levels. Most importantly, we provide several practical takeaways to enhance math reasoning abilities in LLMs for both research and industry applications.
     </details>

278. **Steamroller Problems: An Evaluation of LLM Reasoning Capability with Automated Theorem Prover Strategies** [[pdf]](https://arxiv.org/abs/2407.20244v1) `2024-07-17` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is confirmed that LLMs have a preference for, and are best able to follow, bottom up reasoning processes, and the reasoning strategies can still be beneficial for deriving small and relevant sets of formulas for external processing by a trusted inference engine.
     </details>


     <details>
          <summary>Abstract</summary>
          This study presents the first examination of the ability of Large Language Models (LLMs) to follow reasoning strategies that are used to guide Automated Theorem Provers (ATPs). We evaluate the performance of GPT4, GPT3.5 Turbo and Google's recent Gemini model on problems from a steamroller domain. In addition to determining accuracy we make use of the Natural Language Processing library spaCy to explore new methods of investigating LLM's reasoning capabilities. This led to one alarming result, the low correlation between correct reasoning and correct answers for any of the tested models. We found that the models' performance when using the ATP reasoning strategies was comparable to one-shot chain of thought and observe that attention to uncertainty in the accuracy results is critical when drawing conclusions about model performance. Consistent with previous speculation we confirm that LLMs have a preference for, and are best able to follow, bottom up reasoning processes. However, the reasoning strategies can still be beneficial for deriving small and relevant sets of formulas for external processing by a trusted inference engine.
     </details>

279. **Reasoning with Large Language Models, a Survey** [[pdf]](http://arxiv.org/abs/2407.11511) `2024-07-16` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is found that self-improvement, self-reflection, and some metacognitive abilities of the reasoning processes are possible through the judicious use of prompts, suggesting that true self-improvement and self-reasoning, to go from reasoning with LLMs to reasoning by LLMs, remains future work.
     </details>


     <details>
          <summary>Abstract</summary>
          Scaling up language models to billions of parameters has opened up possibilities for in-context learning, allowing instruction tuning and few-shot learning on tasks that the model was not specifically trained for. This has achieved breakthrough performance on language tasks such as translation, summarization, and question-answering. Furthermore, in addition to these associative "System 1" tasks, recent advances in Chain-of-thought prompt learning have demonstrated strong "System 2" reasoning abilities, answering a question in the field of artificial general intelligence whether LLMs can reason. The field started with the question whether LLMs can solve grade school math word problems. This paper reviews the rapidly expanding field of prompt-based reasoning with LLMs. Our taxonomy identifies different ways to generate, evaluate, and control multi-step reasoning. We provide an in-depth coverage of core approaches and open problems, and we propose a research agenda for the near future. Finally, we highlight the relation between reasoning and prompt-based learning, and we discuss the relation between reasoning, sequential decision processes, and reinforcement learning. We find that self-improvement, self-reflection, and some metacognitive abilities of the reasoning processes are possible through the judicious use of prompts. True self-improvement and self-reasoning, to go from reasoning with LLMs to reasoning by LLMs, remains future work.
     </details>

280. **Diffusion of Thoughts: Chain-of-Thought Reasoning in Diffusion Language Models** [[pdf]](http://arxiv.org/abs/2402.07754) `2024-07-15` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes Diffusion-of-Thought (DoT), a novel approach that integrates diffusion models with Chain-of-Thought, a well-established technique for improving the reasoning ability of autoregressive language models.
     </details>


     <details>
          <summary>Abstract</summary>
          Recently, diffusion models have garnered significant interest in the field of text processing due to their many potential advantages compared to conventional autoregressive models. In this work, we propose Diffusion-of-Thought (DoT), a novel approach that integrates diffusion models with Chain-of-Thought, a well-established technique for improving the reasoning ability of autoregressive language models. In contrast to autoregressive language models that make decisions in a left-to-right, token-by-token manner, DoT allows reasoning steps to diffuse over time through a diffusion language model and offers greater flexibility in trading-off computation for reasoning performance. Our experimental results demonstrate the effectiveness of DoT in multi-digit multiplication, boolean logic, and grade school math problems, with a small diffusion model outperforming a much larger autoregressive model in both efficiency and accuracy. In addition to that, DoT showcases promising self-correction abilities and benefits from existing reasoning-enhancing techniques like self-consistency decoding. Our findings contribute to the understanding and development of reasoning with diffusion language models.
     </details>

281. **COMET: "Cone of experience" enhanced large multimodal model for mathematical problem generation** [[pdf]](http://arxiv.org/abs/2407.11315) `2024-07-15` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A three-stage fine-turning framework guided by the "Cone of Experience" is proposed, which divides the fine-tuning data into symbolic experience, iconic experience, and direct experience to draw parallels with experiences in the career growth of teachers.
     </details>


     <details>
          <summary>Abstract</summary>
          The automatic generation of high-quality mathematical problems is practically valuable in many educational scenarios. Large multimodal model provides a novel technical approach for the mathematical problem generation because of its wide success in cross-modal data scenarios. However, the traditional method of separating problem solving from problem generation and the mainstream fine-tuning framework of monotonous data structure with homogeneous training objectives limit the application of large multimodal model in mathematical problem generation. Addressing these challenges, this paper proposes COMET, a "Cone of Experience" enhanced large multimodal model for mathematical problem generation. Firstly, from the perspective of mutual ability promotion and application logic, we unify stem generation and problem solving into mathematical problem generation. Secondly, a three-stage fine-turning framework guided by the "Cone of Experience" is proposed. The framework divides the fine-tuning data into symbolic experience, iconic experience, and direct experience to draw parallels with experiences in the career growth of teachers. Several fine-grained data construction and injection methods are designed in this framework. Finally, we construct a Chinese multimodal mathematical problem dataset to fill the vacancy of Chinese multimodal data in this field. Combined with objective and subjective indicators, experiments on multiple datasets fully verify the effectiveness of the proposed framework and model.
     </details>

282. **Step-Controlled DPO: Leveraging Stepwise Error for Enhanced Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2407.00782) `2024-07-14` (5 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes Step-Controlled DPO (SCDPO), a method for automatically providing stepwise error supervision by creating negative samples of mathematical reasoning rationales that start making errors at a specified step, which can better align the model to understand reasoning errors and output accurate reasoning steps.
     </details>


     <details>
          <summary>Abstract</summary>
          Direct Preference Optimization (DPO) has proven effective at improving the performance of large language models (LLMs) on downstream tasks such as reasoning and alignment. In this work, we propose Step-Controlled DPO (SCDPO), a method for automatically providing stepwise error supervision by creating negative samples of mathematical reasoning rationales that start making errors at a specified step. By applying these samples in DPO training, SCDPO can better align the model to understand reasoning errors and output accurate reasoning steps. We apply SCDPO to both code-integrated and chain-of-thought solutions, empirically showing that it consistently improves the performance compared to naive DPO on three different SFT models, including one existing SFT model and two models we finetuned. Qualitative analysis of the credit assignment of SCDPO and DPO demonstrates the effectiveness of SCDPO at identifying errors in mathematical solutions. We then apply SCDPO to an InternLM2-20B model, resulting in a 20B model that achieves high scores of 88.5% on GSM8K and 58.1% on MATH, rivaling all other open-source LLMs, showing the great potential of our method.
     </details>

283. **Stepwise Verification and Remediation of Student Reasoning Errors with Large Language Model Tutors** [[pdf]](https://arxiv.org/abs/2407.09136v1) `2024-07-12` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work focuses on verifying student solutions and shows how grounding to such verification improves the overall quality of tutor response generation and proposes and evaluates several verifiers for detecting student's errors.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) present an opportunity to scale high-quality personalized education to all. A promising approach towards this means is to build dialog tutoring models that scaffold students' problem-solving. However, even though existing LLMs perform well in solving reasoning questions, they struggle to precisely detect student's errors and tailor their feedback to these errors. Inspired by real-world teaching practice where teachers identify student errors and customize their response based on them, we focus on verifying student solutions and show how grounding to such verification improves the overall quality of tutor response generation. We collect a dataset of 1K stepwise math reasoning chains with the first error step annotated by teachers. We show empirically that finding the mistake in a student solution is challenging for current models. We propose and evaluate several verifiers for detecting these errors. Using both automatic and human evaluation we show that the student solution verifiers steer the generation model towards highly targeted responses to student errors which are more often correct with less hallucinations compared to existing baselines.
     </details>

284. **Token-Supervised Value Models for Enhancing Mathematical Reasoning Capabilities of Large Language Models** [[pdf]](http://arxiv.org/abs/2407.12863) `2024-07-12` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Experimental results on mathematical reasoning benchmarks show that Token-Supervised Value Model (TVM) can outperform step-by-step verifiers on GSM8K and MATH with Mistral and Llama.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have demonstrated impressive problem-solving capabilities in mathematics through step-by-step reasoning chains. However, they are susceptible to reasoning errors that impact the quality of subsequent reasoning chains and the final answer due to language models' autoregressive token-by-token generating nature. Recent works have proposed adopting external verifiers to guide the generation of reasoning paths, but existing works utilize models that have been trained with step-by-step labels to assess the correctness of token-by-token reasoning chains. Consequently, they struggle to recognize discriminative details of tokens within a reasoning path and lack the ability to evaluate whether an intermediate reasoning path is on a promising track toward the correct final answer. To amend the lack of sound and token-grained math-verification signals, we devise a novel training scheme for verifiers that apply token-level supervision with the expected cumulative reward (i.e., value). Furthermore, we propose a practical formulation of the cumulative reward by reducing it to finding the probability of future correctness of the final answer and thereby enabling the empirical estimation of the value. Experimental results on mathematical reasoning benchmarks show that Token-Supervised Value Model (TVM) can outperform step-by-step verifiers on GSM8K and MATH with Mistral and Llama.
     </details>

285. **Is Your Model Really A Good Math Reasoner? Evaluating Mathematical Reasoning with Checklist** [[pdf]](http://arxiv.org/abs/2407.08733) `ICLR 2025 Submission` (4 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is argued that if a model really understands a problem, it should be robustly and readily applied across a diverse array of tasks, and introduced MATHCHECK, a well-designed checklist for testing task generalization and reasoning robustness, as well as an automatic tool to generate checklists efficiently.
     </details>


     <details>
          <summary>Abstract</summary>
          Exceptional mathematical reasoning ability is one of the key features that demonstrate the power of large language models (LLMs). How to comprehensively define and evaluate the mathematical abilities of LLMs, and even reflect the user experience in real-world scenarios, has emerged as a critical issue. Current benchmarks predominantly concentrate on problem-solving capabilities, presenting a substantial risk of model overfitting and fails to accurately measure the genuine mathematical reasoning abilities. In this paper, we argue that if a model really understands a problem, it should be robustly applied across a diverse array of tasks. To this end, we introduce MathCheck, a well-designed checklist for testing task generalization and reasoning robustness, as well as an automatic tool to generate checklists efficiently. MathCheck includes multiple mathematical reasoning tasks and robustness tests to facilitate a comprehensive evaluation of both mathematical reasoning ability and behavior testing. Utilizing MathCheck, we develop MathCheck-GSM and MathCheck-GEO to assess math textual reasoning and multi-modal reasoning abilities, respectively, serving as upgraded versions of benchmarks including GSM8k, GeoQA, UniGeo, and Geometry3K. We adopt MathCheck-GSM and MathCheck-GEO to evaluate 26 LLMs and 17 MLLMs. Our results demonstrate that while frontier LLMs like GPT-4o continue to excel in various abilities on the checklist, many other model families exhibit a significant decline. Further experiments indicate that, compared to traditional math benchmarks, MathCheck better reflects true mathematical abilities and represents mathematical intelligence more linearly, thereby supporting our design. Using MathCheck, we can efficiently conduct informative behavior analysis to deeply investigate models. Finally, we show that our checklist paradigm can easily extend to other reasoning tasks.
     </details>

286. **MAVIS: Mathematical Visual Instruction Tuning** [[pdf]](http://arxiv.org/abs/2407.08739) `2024-07-11` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper proposes MAVIS, the first MAthematical VISual instruction tuning paradigm for MLLMs, involving a series of mathematical visual datasets and specialized MLLMs, and introduces MAVIS-Instruct, which is adopted to instruct-tune the MLLM for robust mathematical reasoning skills.
     </details>


     <details>
          <summary>Abstract</summary>
          Multi-modal Large Language Models (MLLMs) have recently emerged as a significant focus in academia and industry. Despite their proficiency in general multi-modal scenarios, the mathematical problem-solving capabilities in visual contexts remain insufficiently explored. We identify three key areas within MLLMs that need to be improved: visual encoding of math diagrams, diagram-language alignment, and mathematical reasoning skills. This draws forth an urgent demand for large-scale, high-quality data and training pipelines in visual mathematics. In this paper, we propose MAVIS, the first MAthematical VISual instruction tuning paradigm for MLLMs, involving a series of mathematical visual datasets and specialized MLLMs. Targeting the three issues, MAVIS contains three progressive training stages from scratch. First, we curate MAVIS-Caption, consisting of 558K diagram-caption pairs, to fine-tune a math-specific vision encoder (CLIP-Math) through contrastive learning, tailored for improved diagram visual encoding. Second, we utilize MAVIS-Caption to align the CLIP-Math with a large language model (LLM) by a projection layer, enhancing vision-language alignment in mathematical domains. Third, we introduce MAVIS-Instruct, including 900K meticulously collected and annotated visual math problems, which is adopted to finally instruct-tune the MLLM for robust mathematical reasoning skills. In MAVIS-Instruct, we incorporate complete chain-of-thought (CoT) rationales for each problem, and minimize textual redundancy, thereby concentrating the model towards the visual elements. Data and Models are released at https://github.com/ZrrSkywalker/MAVIS
     </details>

287. **Exploring the Compositional Deficiency of Large Language Models in Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2405.06680) `2024-07-11` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work constructs a new dataset \textsc{MathTrap}\footnotemark[3] by introducing carefully designed logical traps into the problem descriptions of MATH and GSM8k and finds that LLMs' performance can be improved through the above external intervention.
     </details>


     <details>
          <summary>Abstract</summary>
          Human cognition exhibits systematic compositionality, the algebraic ability to generate infinite novel combinations from finite learned components, which is the key to understanding and reasoning about complex logic. In this work, we investigate the compositionality of large language models (LLMs) in mathematical reasoning. Specifically, we construct a new dataset \textsc{MathTrap} by introducing carefully designed logical traps into the problem descriptions of MATH and GSM8K. Since problems with logical flaws are quite rare in the real world, these represent "unseen" cases to LLMs. Solving these requires the models to systematically compose (1) the mathematical knowledge involved in the original problems with (2) knowledge related to the introduced traps. Our experiments show that while LLMs possess both components of requisite knowledge, they do not \textbf{spontaneously} combine them to handle these novel cases. We explore several methods to mitigate this deficiency, such as natural language prompts, few-shot demonstrations, and fine-tuning. Additionally, we test the recently released OpenAI o1 model and find that human-like `slow thinking' helps improve the compositionality of LLMs. Overall, systematic compositionality remains an open challenge for large language models.
     </details>

288. **Self-training Language Models for Arithmetic Reasoning** [[pdf]](http://arxiv.org/abs/2407.08400) `2024-07-11` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work explores the potential of improving models' reasoning capabilities without new data, merely using automated feedback to the validity of their predictions in arithmetic reasoning (self-training), and finds that models can substantially improve in both single-round (offline) and online self-training.
     </details>


     <details>
          <summary>Abstract</summary>
          Language models achieve impressive results in tasks involving complex multistep reasoning, but scaling these capabilities further traditionally requires expensive collection of more annotated data. In this work, we explore the potential of improving the capabilities of language models without new data, merely using automated feedback to the validity of their predictions in arithmetic reasoning (self-training). We find that models can substantially improve in both single-round (offline) and online self-training. In the offline setting, supervised methods are able to deliver gains comparable to preference optimization, but in online self-training, preference optimization shows to largely outperform supervised training thanks to superior stability and robustness on unseen types of problems.
     </details>

289. **Fuse, Reason and Verify: Geometry Problem Solving with Parsed Clauses from Diagram** [[pdf]](https://arxiv.org/abs/2407.07327v1) `2024-07-10` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper proposes a neural-symbolic model for plane geometry problem solving (PGPS), named PGPSNet-v2, with three key steps: modal fusion, reasoning process and knowledge verification, which outperforms existing symbolic and neural solvers in GPS performance, while maintaining good explainability and reliability.
     </details>


     <details>
          <summary>Abstract</summary>
          Geometry problem solving (GPS) requires capacities of multi-modal understanding, multi-hop reasoning and theorem knowledge application. In this paper, we propose a neural-symbolic model for plane geometry problem solving (PGPS), named PGPSNet-v2, with three key steps: modal fusion, reasoning process and knowledge verification. In modal fusion, we leverage textual clauses to express fine-grained structural and semantic content of geometry diagram, and fuse diagram with textual problem efficiently through structural-semantic pre-training. For reasoning, we design an explicable solution program to describe the geometric reasoning process, and employ a self-limited decoder to generate solution program autoregressively. To reduce solution errors, a multi-level theorem verifier is proposed to eliminate solutions that do not match geometric principles, alleviating the hallucination of the neural model. We also construct a large-scale geometry problem dataset called PGPS9K, containing fine-grained annotations of textual clauses, solution program and involved knowledge tuples. Extensive experiments on datasets Geometry3K and PGPS9K show that our PGPSNet solver outperforms existing symbolic and neural solvers in GPS performance, while maintaining good explainability and reliability, and the solver components (fusion, reasoning, verification) are all justified effective.
     </details>

290. **Regularization in Spider-Style Strategy Discovery and Schedule Construction** [[pdf]](http://arxiv.org/abs/2403.12869) `2024-07-09` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper reports on a large-scale experiment with discovering strategies for the Vampire prover, targeting the FOF fragment of the TPTP library and constructing a schedule for it, based on the ideas of Andrei Voronkov's system Spider.
     </details>


     <details>
          <summary>Abstract</summary>
          To achieve the best performance, automatic theorem provers often rely on schedules of diverse proving strategies to be tried out (either sequentially or in parallel) on a given problem. In this paper, we report on a large-scale experiment with discovering strategies for the Vampire prover, targeting the FOF fragment of the TPTP library and constructing a schedule for it, based on the ideas of Andrei Voronkov's system Spider. We examine the process from various angles, discuss the difficulty (or ease) of obtaining a strong Vampire schedule for the CASC competition, and establish how well a schedule can be expected to generalize to unseen problems and what factors influence this property.
     </details>

291. **LLM Critics Help Catch Bugs in Mathematics: Towards a Better Mathematical Verifier with Natural Language Feedback** [[pdf]](http://arxiv.org/abs/2406.14024) `2024-07-08` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper proposes Math-Minos, a natural language feedback enhanced verifier by constructing automatically-generated training data and a two-stage training paradigm for effective training and efficient inference.
     </details>


     <details>
          <summary>Abstract</summary>
          In recent progress, mathematical verifiers have achieved success in mathematical reasoning tasks by validating the correctness of solutions generated by policy models. However, existing verifiers are trained with binary classification labels, which are not informative enough for the model to accurately assess the solutions. To mitigate the aforementioned insufficiency of binary labels, we introduce step-wise natural language feedback as rationale labels, that is, the correctness of each step and the detailed explanations. In this paper, we propose Math-Minos, a natural language feedback-enhanced verifier by constructing automatically generated training data and a two-stage training paradigm for effective training and efficient inference. Our experiments reveal that a small set of natural language feedback can significantly boost the performance of the verifier in both verification and reinforcement learning. We have released the code and data for further exploration.
     </details>

292. **Can Large Language Models Replicate ITS Feedback on Open-Ended Math Questions?** [[pdf]](http://arxiv.org/abs/2405.06414) `2024-07-08` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The capabilities of large language models (LLMs) to generate feedback for open-ended math questions, similar to that of an established ITS that uses a template-based approach are examined, finding that open-source and proprietary models both show promise in replicating the feedback they see during training, but do not generalize well to previously unseen student errors.
     </details>


     <details>
          <summary>Abstract</summary>
          Intelligent Tutoring Systems (ITSs) often contain an automated feedback component, which provides a predefined feedback message to students when they detect a predefined error. To such a feedback component, we often resort to template-based approaches. These approaches require significant effort from human experts to detect a limited number of possible student errors and provide corresponding feedback. This limitation is exemplified in open-ended math questions, where there can be a large number of different incorrect errors. In our work, we examine the capabilities of large language models (LLMs) to generate feedback for open-ended math questions, similar to that of an established ITS that uses a template-based approach. We fine-tune both open-source and proprietary LLMs on real student responses and corresponding ITS-provided feedback. We measure the quality of the generated feedback using text similarity metrics. We find that open-source and proprietary models both show promise in replicating the feedback they see during training, but do not generalize well to previously unseen student errors. These results suggest that despite being able to learn the formatting of feedback, LLMs are not able to fully understand mathematical errors made by students.
     </details>

293. **Fill in the Blank: Exploring and Enhancing LLM Capabilities for Backward Reasoning in Math Word Problems** [[pdf]](http://arxiv.org/abs/2310.01991) `2024-07-07` (8 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel Bayesian formulation for creating an ensemble over the base methods to further boost the accuracy of LLMs on the backward reasoning task, with the ensemble-based method resulting in significant performance gains compared to the SOTA forward reasoning strategies the authors adapt.
     </details>


     <details>
          <summary>Abstract</summary>
          While forward reasoning (i.e., find the answer given the question) has been explored extensively in recent literature, backward reasoning is relatively unexplored. We examine the backward reasoning capabilities of LLMs on Math Word Problems (MWPs): given a mathematical question and its answer, with some details omitted from the question, can LLMs effectively retrieve the missing information? On modifying three benchmark datasets for this task, to evaluate this task: GSM8k, SVAMP, and MultiArith, we find a significant drop in the accuracy of models on this task compared to forward reasoning across SOTA LLMs (GPT4, GPT3.5, PaLM-2, and LLaMa). Motivated by the fact backward reasoning can be seen as the ''inverse'' of forward reasoning, we propose variations of three different forward reasoning strategies to improve performance. Rephrase reformulates the given problem into a forward reasoning problem, PAL-Tools combines the idea of Program-Aided LLMs to produce a set of equations that can be solved by an external solver, and Check your Work exploits the availability of natural verifier of high accuracy in the forward direction, interleaving solving and verification steps. Finally, realizing that each of our base methods correctly solves a different set of problems, we propose a novel Bayesian formulation for creating an ensemble over the base methods to further boost the accuracy. Extensive experimentation demonstrates successive improvement in the performance of LLMs on the backward reasoning task, using our strategies, with our ensemble-based method resulting in significant performance gains compared to the SOTA forward reasoning strategies we adapt.
     </details>

294. **LogicVista: Multimodal LLM Logical Reasoning Benchmark in Visual Contexts** [[pdf]](https://arxiv.org/abs/2407.04973v1) `ICLR 2025 Submission` (5 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          LogicVista is proposed, an evaluation benchmark that assesses the integrated logical reasoning capabilities of multimodal large language models (MLLMs) in Visual contexts using a sample of 448 multiple-choice questions.
     </details>


     <details>
          <summary>Abstract</summary>
          We propose LogicVista, an evaluation benchmark that assesses the integrated logical reasoning capabilities of multimodal large language models (MLLMs) in Visual contexts. Recent advancements in MLLMs have demonstrated various fascinating abilities, from crafting poetry based on an image to performing mathematical reasoning. However, there is still a lack of systematic evaluation of MLLMs' proficiency in logical reasoning tasks, which are essential for activities like navigation and puzzle-solving. Thus we evaluate general logical cognition abilities across 5 logical reasoning tasks encompassing 9 different capabilities, using a sample of 448 multiple-choice questions. Each question is annotated with the correct answer and the human-written reasoning behind the selection, enabling both open-ended and multiple-choice evaluation. A total of 8 MLLMs are comprehensively evaluated using LogicVista. Code and Data Available at https://github.com/Yijia-Xiao/LogicVista.
     </details>

295. **Solving for X and Beyond: Can Large Language Models Solve Complex Math Problems with More-Than-Two Unknowns?** [[pdf]](https://arxiv.org/abs/2407.05134v1) `2024-07-06` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The Formulate-and-Solve strategy, a generalized prompting approach that effectively handles problems with an arbitrary number of unknowns, is proposed, and is revealed to enhance LLM performance on the BeyondX benchmark but also provides deeper insights into the computational limits of LLMs when faced with more complex mathematical challenges.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have demonstrated remarkable performance in solving math problems, a hallmark of human intelligence. Despite high success rates on current benchmarks; however, these often feature simple problems with only one or two unknowns, which do not sufficiently challenge their reasoning capacities. This paper introduces a novel benchmark, BeyondX, designed to address these limitations by incorporating problems with multiple unknowns. Recognizing the challenges in proposing multi-unknown problems from scratch, we developed BeyondX using an innovative automated pipeline that progressively increases complexity by expanding the number of unknowns in simpler problems. Empirical study on BeyondX reveals that the performance of existing LLMs, even those fine-tuned specifically on math tasks, significantly decreases as the number of unknowns increases - with a performance drop of up to 70\% observed in GPT-4. To tackle these challenges, we propose the Formulate-and-Solve strategy, a generalized prompting approach that effectively handles problems with an arbitrary number of unknowns. Our findings reveal that this strategy not only enhances LLM performance on the BeyondX benchmark but also provides deeper insights into the computational limits of LLMs when faced with more complex mathematical challenges.
     </details>

296. **Towards Automated Functional Equation Proving: A Benchmark Dataset and A Domain-Specific In-Context Agent** [[pdf]](https://arxiv.org/abs/2407.14521v1) `2024-07-05` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          FEAS, an agent that enhances the COPRA in-context learning framework within Lean, is introduced, an agent that enhances the COPRA in-context learning framework within Lean and outperforms baselines on FunEq, particularly with the integration of domain-specific heuristics.
     </details>


     <details>
          <summary>Abstract</summary>
          Automated Theorem Proving (ATP) faces challenges due to its complexity and computational demands. Recent work has explored using Large Language Models (LLMs) for ATP action selection, but these methods can be resource-intensive. This study introduces FEAS, an agent that enhances the COPRA in-context learning framework within Lean. FEAS refines prompt generation, response parsing, and incorporates domain-specific heuristics for functional equations. It introduces FunEq, a curated dataset of functional equation problems with varying difficulty. FEAS outperforms baselines on FunEq, particularly with the integration of domain-specific heuristics. The results demonstrate FEAS's effectiveness in generating and formalizing high-level proof strategies into Lean proofs, showcasing the potential of tailored approaches for specific ATP challenges.
     </details>

297. **AgentInstruct: Toward Generative Teaching with Agentic Flows** [[pdf]](http://arxiv.org/abs/2407.03502) `2024-07-03` (4 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces AgentInstruct, an extensible agentic framework for automatically creating large amounts of diverse and high-quality synthetic data, and demonstrates the utility of AgentInstruct by creating a post training dataset of 25M pairs to teach language models different skills.
     </details>


     <details>
          <summary>Abstract</summary>
          Synthetic data is becoming increasingly important for accelerating the development of language models, both large and small. Despite several successful use cases, researchers also raised concerns around model collapse and drawbacks of imitating other models. This discrepancy can be attributed to the fact that synthetic data varies in quality and diversity. Effective use of synthetic data usually requires significant human effort in curating the data. We focus on using synthetic data for post-training, specifically creating data by powerful models to teach a new skill or behavior to another model, we refer to this setting as Generative Teaching. We introduce AgentInstruct, an extensible agentic framework for automatically creating large amounts of diverse and high-quality synthetic data. AgentInstruct can create both the prompts and responses, using only raw data sources like text documents and code files as seeds. We demonstrate the utility of AgentInstruct by creating a post training dataset of 25M pairs to teach language models different skills, such as text editing, creative writing, tool usage, coding, reading comprehension, etc. The dataset can be used for instruction tuning of any base model. We post-train Mistral-7b with the data. When comparing the resulting model Orca-3 to Mistral-7b-Instruct (which uses the same base model), we observe significant improvements across many benchmarks. For example, 40% improvement on AGIEval, 19% improvement on MMLU, 54% improvement on GSM8K, 38% improvement on BBH and 45% improvement on AlpacaEval. Additionally, it consistently outperforms other models such as LLAMA-8B-instruct and GPT-3.5-turbo.
     </details>

298. **TheoremLlama: Transforming General-Purpose LLMs into Lean4 Experts** [[pdf]](http://arxiv.org/abs/2407.03203) `2024-07-03` `Lean` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper proposes TheoremLlama, an end-to-end framework that trains a general-purpose LLM to be a Lean4 expert, and provides Open Bootstrapped Theorems (OBT), an NL-FL aligned and bootstrapped dataset.
     </details>


     <details>
          <summary>Abstract</summary>
          Proving mathematical theorems using computer-verifiable formal languages like Lean significantly impacts mathematical reasoning. One approach to formal theorem proving involves generating complete proofs using Large Language Models (LLMs) based on Natural Language (NL) proofs. Similar methods have shown promising results in code generation. However, most modern LLMs exhibit suboptimal performance due to the scarcity of aligned NL and Formal Language (FL) theorem-proving data. This scarcity results in a paucity of methodologies for training LLMs and techniques to fully utilize their capabilities in composing formal proofs. To address the challenges, this paper proposes **TheoremLlama**, an end-to-end framework to train a general-purpose LLM to become a Lean4 expert. This framework encompasses NL-FL aligned dataset generation methods, training approaches for the LLM formal theorem prover, and techniques for LLM Lean4 proof writing. Using the dataset generation method, we provide *Open Bootstrapped Theorems* (OBT), an NL-FL aligned and bootstrapped dataset. A key innovation in this framework is the NL-FL bootstrapping method, where NL proofs are integrated into Lean4 code for training datasets, leveraging the NL reasoning ability of LLMs for formal reasoning. The **TheoremLlama** framework achieves cumulative accuracies of 36.48% and 33.61% on MiniF2F-Valid and Test datasets respectively, surpassing the GPT-4 baseline of 22.95% and 25.41%. We have also open-sourced our model checkpoints and generated dataset, and will soon make all the code publicly available.
     </details>

299. **Artificial intelligence and machine learning generated conjectures with TxGraffiti** [[pdf]](https://arxiv.org/abs/2407.02731v1) `2024-07-03` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          \emph{TxGraffiti} is a machine learning and heuristic based artificial intelligence designed to automate the task of conjecturing in mathematics. Since its inception, TxGraffiti has generated many surprising conjectures leading to publication in respectable mathematical journals. In this paper we outline the machine learning and heuristic techniques implemented by TxGraffiti. We also recall its contributions to the mathematical literature and announce a new online version of the program available for anyone curious to explore conjectures in graph theory.
     </details>

300. **We-Math: Does Your Large Multimodal Model Achieve Human-like Mathematical Reasoning?** [[pdf]](http://arxiv.org/abs/2407.01284) `2024-07-01` (6 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The IK issue of LMMs can be effectively improved via knowledge augmentation strategies and the primary challenge of GPT-4o has significantly transitioned from IK to IG, establishing it as the first LMM advancing towards the knowledge generalization stage.
     </details>


     <details>
          <summary>Abstract</summary>
          Visual mathematical reasoning, as a fundamental visual reasoning ability, has received widespread attention from the Large Multimodal Models (LMMs) community. Existing benchmarks, such as MathVista and MathVerse, focus more on the result-oriented performance but neglect the underlying principles in knowledge acquisition and generalization. Inspired by human-like mathematical reasoning, we introduce WE-MATH, the first benchmark specifically designed to explore the problem-solving principles beyond end-to-end performance. We meticulously collect and categorize 6.5K visual math problems, spanning 67 hierarchical knowledge concepts and five layers of knowledge granularity. We decompose composite problems into sub-problems according to the required knowledge concepts and introduce a novel four-dimensional metric, namely Insufficient Knowledge (IK), Inadequate Generalization (IG), Complete Mastery (CM), and Rote Memorization (RM), to hierarchically assess inherent issues in LMMs' reasoning process. With WE-MATH, we conduct a thorough evaluation of existing LMMs in visual mathematical reasoning and reveal a negative correlation between solving steps and problem-specific performance. We confirm the IK issue of LMMs can be effectively improved via knowledge augmentation strategies. More notably, the primary challenge of GPT-4o has significantly transitioned from IK to IG, establishing it as the first LMM advancing towards the knowledge generalization stage. In contrast, other LMMs exhibit a marked inclination towards Rote Memorization - they correctly solve composite problems involving multiple knowledge concepts yet fail to answer sub-problems. We anticipate that WE-MATH will open new pathways for advancements in visual mathematical reasoning for LMMs. The WE-MATH data and evaluation code are available at https://github.com/We-Math/We-Math.
     </details>

301. **First-Step Advantage: Importance of Starting Right in Multi-Step Math Reasoning** [[pdf]](http://arxiv.org/abs/2311.07945) `ICLR 2025 Submission` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is shown that getting the right start can lead to significant performance gains across all models, and proposed QuestCoT, where a smaller model first asks itself how to start, before proceeding with a chain of reasoning.
     </details>


     <details>
          <summary>Abstract</summary>
          Language models can solve complex reasoning tasks better by learning to generate rationales for their predictions. Often these models know how to solve a task but their auto-regressive decoding nature leads to incorrect results if they start incorrectly. We observe that smaller models in particular when corrected, can solve a task that they would have otherwise struggled with. We demonstrate this phenomenon by using a larger model to guide smaller models, which leads to significantly improved performance (up to +24 points on the GSM8K dataset by 7B models). To assist smaller models in initiating the starting step, we propose QuestCoT, where a smaller model first asks itself how to start, before proceeding with a chain of reasoning. On various multistep mathematical reasoning datasets over multiple smaller models, we show that getting the right start can lead to significant performance gains across all models (gains of up to +6 points on GSM8K, +9 on SVAMP, +5 on ASDiv, and +7 on MultiArith).
     </details>

302. **MalAlgoQA: A Pedagogical Approach for Evaluating Counterfactual Reasoning Abilities** [[pdf]](https://arxiv.org/abs/2407.00938v1) `2024-07-01` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is found that the chain-of-thought prompting technique not only fails to consistently enhance MIA, but can also lead to underperformance compared to simple prompting, which holds significant implications for the development of more cognitively-inspired LLMs to improve their counterfactual reasoning abilities.
     </details>


     <details>
          <summary>Abstract</summary>
          This paper introduces MalAlgoQA, a novel dataset designed to evaluate the counterfactual reasoning capabilities of Large Language Models (LLMs) through a pedagogical approach. The dataset comprises mathematics and reading comprehension questions, each accompanied by four answer choices and their corresponding rationales. We focus on the incorrect answer rationales, termed "malgorithms", which highlights flawed reasoning steps leading to incorrect answers and offers valuable insights into erroneous thought processes. We also propose the Malgorithm Identification task, where LLMs are assessed based on their ability to identify corresponding malgorithm given an incorrect answer choice. To evaluate the model performance, we introduce two metrics: Algorithm Identification Accuracy (AIA) for correct answer rationale identification, and Malgorithm Identification Accuracy (MIA) for incorrect answer rationale identification. The task is challenging since state-of-the-art LLMs exhibit significant drops in MIA as compared to AIA. Moreover, we find that the chain-of-thought prompting technique not only fails to consistently enhance MIA, but can also lead to underperformance compared to simple prompting. These findings hold significant implications for the development of more cognitively-inspired LLMs to improve their counterfactual reasoning abilities, particularly through a pedagogical perspective where understanding and rectifying student misconceptions are crucial.
     </details>

303. **FRoG: Evaluating Fuzzy Reasoning of Generalized Quantifiers in Large Language Models** [[pdf]](https://arxiv.org/abs/2407.01046v2) `2024-07-01` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A new benchmark, FRoG, is introduced for fuzzy reasoning, featuring real-world mathematical word problems that incorporate generalized quantifiers, and it is found that existing methods designed to enhance reasoning do not consistently improve performance in tasks involving fuzzy logic.
     </details>


     <details>
          <summary>Abstract</summary>
          Fuzzy reasoning is vital due to the frequent use of imprecise information in daily contexts. However, the ability of current large language models (LLMs) to handle such reasoning remains largely uncharted. In this paper, we introduce a new benchmark, FRoG, for fuzzy reasoning, featuring real-world mathematical word problems that incorporate generalized quantifiers. Our experimental findings reveal that fuzzy reasoning continues to pose significant challenges for LLMs. Moreover, we find that existing methods designed to enhance reasoning do not consistently improve performance in tasks involving fuzzy logic. Additionally, our results show an inverse scaling effect in the performance of LLMs on FRoG. Interestingly, we also demonstrate that strong mathematical reasoning skills are not necessarily indicative of success on our benchmark.
     </details>

304. **Mamo: a Mathematical Modeling Benchmark with Solvers** [[pdf]](http://arxiv.org/abs/2405.13144) `2024-06-30` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces a new benchmark, Mamo, that transcends traditional result-oriented assessments of Large Language Models' mathematical modeling capabilities and sets a new standard for evaluating their performance in complex problem-solving scenarios.
     </details>


     <details>
          <summary>Abstract</summary>
          Mathematical modeling involves representing real-world phenomena, systems, or problems using mathematical expressions and equations to analyze, understand, and predict their behavior. Given that this process typically requires experienced experts, there is an interest in exploring whether Large Language Models (LLMs) can undertake mathematical modeling to potentially decrease human labor. To evaluate of LLMs in mathematical modeling, we introduce a new benchmark, Mamo, that transcends traditional result-oriented assessments. Unlike conventional methods that primarily assess LLMs based on the accuracy of solutions to mathematical problems, our approach offers deeper insight into the modeling process itself. By focusing on the processes LLMs undertake rather than the correctness of their final solutions, Mamo pioneers a novel evaluation paradigm. This shift underscores the importance of understanding the inherent modeling capabilities of LLMs, paving the way for a more nuanced and comprehensive analysis of their problem-solving strategies. Our work marks a significant advancement in the field, suggesting a new direction for future research by emphasizing the evaluation of LLMs' modeling processes over the mere correctness of answers. This benchmark not only facilitates a better understanding of LLMs' mathematical modeling capabilities but also sets a new standard for evaluating their performance in complex problem-solving scenarios.
     </details>

305. **MathCAMPS: Fine-grained Synthesis of Mathematical Problems From Human Curricula** [[pdf]](http://arxiv.org/abs/2407.00900) `NeurIPS 2024 Workshop MATH-AI` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes MathCAMPS: a method to synthesize high-quality mathematical problems at scale, grounded on 44 fine-grained standards from the Mathematics Common Core (CC) Standard for K-8 grades, and proposes a cycle-consistency method for validating problem faithfulness.
     </details>


     <details>
          <summary>Abstract</summary>
          Mathematical problem solving is an important skill for Large Language Models (LLMs), both as an important capability and a proxy for a range of reasoning abilities. Existing benchmarks probe a diverse set of skills, but they yield aggregate accuracy metrics, obscuring specific abilities or weaknesses. Furthermore, they are difficult to extend with new problems, risking data contamination over time. To address these challenges, we propose MathCAMPS: a method to synthesize high-quality mathematical problems at scale, grounded on 44 fine-grained "standards" from the Mathematics Common Core (CC) Standard for K-8 grades. We encode each standard in a formal grammar, allowing us to sample diverse symbolic problems and their answers. We then use LLMs to realize the symbolic problems into word problems. We propose a cycle-consistency method for validating problem faithfulness. Finally, we derive follow-up questions from symbolic structures and convert them into follow-up word problems - a novel task of mathematical dialogue that probes for robustness in understanding. Experiments on 23 LLMs show surprising failures even in the strongest models (in particular when asked simple follow-up questions). Moreover, we evaluate training checkpoints of Pythia 12B on MathCAMPS, allowing us to analyze when particular mathematical skills develop during its training. Our framework enables the community to reproduce and extend our pipeline for a fraction of the typical cost of building new high-quality datasets.
     </details>

306. **How to Leverage Digit Embeddings to Represent Numbers?** [[pdf]](http://arxiv.org/abs/2407.00894) `2024-06-30` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper explores the use of mathematical priors to compute aggregated digit embeddings and explicitly incorporate these aggregates into transformer models and evaluates the effectiveness of incorporating this explicit aggregation.
     </details>


     <details>
          <summary>Abstract</summary>
          Apart from performing arithmetic operations, understanding numbers themselves is still a challenge for existing language models. Simple generalisations, such as solving 100+200 instead of 1+2, can substantially affect model performance (Sivakumar and Moosavi, 2023). Among various techniques, character-level embeddings of numbers have emerged as a promising approach to improve number representation. However, this method has limitations as it leaves the task of aggregating digit representations to the model, which lacks direct supervision for this process. In this paper, we explore the use of mathematical priors to compute aggregated digit embeddings and explicitly incorporate these aggregates into transformer models. This can be achieved either by adding a special token to the input embeddings or by introducing an additional loss function to enhance correct predictions. We evaluate the effectiveness of incorporating this explicit aggregation, analysing its strengths and shortcomings, and discuss future directions to better benefit from this approach. Our methods, while simple, are compatible with any pretrained model and require only a few lines of code, which we have made publicly available.
     </details>

307. **LiteSearch: Efficacious Tree Search for LLM** [[pdf]](http://arxiv.org/abs/2407.00320) `2024-06-29` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This study introduces a novel guided tree search algorithm with dynamic node selection and node-level exploration budget (maximum number of children) calculation to tackle the issue of wasteful search strategies in LLM.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent research suggests that tree search algorithms (e.g. Monte Carlo Tree Search) can dramatically boost LLM performance on complex mathematical reasoning tasks. However, they often require more than 10 times the computational resources of greedy decoding due to wasteful search strategies, making them difficult to be deployed in practical applications. This study introduces a novel guided tree search algorithm with dynamic node selection and node-level exploration budget (maximum number of children) calculation to tackle this issue. By considering the search progress towards the final answer (history) and the guidance from a value network (future) trained without any step-wise annotations, our algorithm iteratively selects the most promising tree node before expanding it within the boundaries of the allocated computational budget. Experiments conducted on the GSM8K and TabMWP datasets demonstrate that our approach not only offers competitive performance but also enjoys significantly lower computational costs compared to baseline methods.
     </details>

308. **Advancing Process Verification for Large Language Models via Tree-Based Preference Learning** [[pdf]](https://arxiv.org/abs/2407.00390v1) `2024-06-29` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This study proposes Tree-based Preference Learning Verifier (Tree-PLV), a novel approach that constructs reasoning trees via a best-first search algorithm and collects step-level paired data for preference training and empirically evaluates it across a range of arithmetic and commonsense reasoning tasks, where it significantly outperforms existing benchmarks.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have demonstrated remarkable potential in handling complex reasoning tasks by generating step-by-step rationales.Some methods have proven effective in boosting accuracy by introducing extra verifiers to assess these paths. However, existing verifiers, typically trained on binary-labeled reasoning paths, fail to fully utilize the relative merits of intermediate steps, thereby limiting the effectiveness of the feedback provided. To overcome this limitation, we propose Tree-based Preference Learning Verifier (Tree-PLV), a novel approach that constructs reasoning trees via a best-first search algorithm and collects step-level paired data for preference training. Compared to traditional binary classification, step-level preferences more finely capture the nuances between reasoning steps, allowing for a more precise evaluation of the complete reasoning path. We empirically evaluate Tree-PLV across a range of arithmetic and commonsense reasoning tasks, where it significantly outperforms existing benchmarks. For instance, Tree-PLV achieved substantial performance gains over the Mistral-7B self-consistency baseline on GSM8K (67.55% to 82.79%), MATH (17.00% to 26.80%), CSQA (68.14% to 72.97%), and StrategyQA (82.86% to 83.25%).Additionally, our study explores the appropriate granularity for applying preference learning, revealing that step-level guidance provides feedback that better aligns with the evaluation of the reasoning process.
     </details>

309. **CMMaTH: A Chinese Multi-modal Math Skill Evaluation Benchmark for Foundation Models** [[pdf]](https://arxiv.org/abs/2407.12023v1) `2024-06-28` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Due to the rapid advancements in multimodal large language models, evaluating their multimodal mathematical capabilities continues to receive wide attention. Despite the datasets like MathVista proposed benchmarks for assessing mathematical capabilities in multimodal scenarios, there is still a lack of corresponding evaluation tools and datasets for fine-grained assessment in the context of K12 education in Chinese language. To systematically evaluate the capability of multimodal large models in solving Chinese multimodal mathematical problems, we propose a Chinese Multi-modal Math Skill Evaluation Benchmark, named CMMaTH, contraining 23k multimodal K12 math related questions, forming the largest Chinese multimodal mathematical problem benchmark to date. CMMaTH questions from elementary to high school levels, provide increased diversity in problem types, solution objectives, visual elements, detailed knowledge points, and standard solution annotations. We have constructed an open-source tool GradeGPT integrated with the CMMaTH dataset, facilitating stable, rapid, and cost-free model evaluation. Our data and code are available.
     </details>

310. **LiveBench: A Challenging, Contamination-Free LLM Benchmark** [[pdf]](http://arxiv.org/abs/2406.19314) `ICLR 2025 Submission` (12 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces a new benchmark for LLMs designed to be immune to both test set contamination and the pitfalls of LLM judging and human crowdsourcing, and releases LiveBench, the first benchmark that contains frequently-updated questions from recent information sources, and scores answers automatically according to objective ground-truth values.
     </details>


     <details>
          <summary>Abstract</summary>
          Test set contamination, wherein test data from a benchmark ends up in a newer model's training set, is a well-documented obstacle for fair LLM evaluation and can quickly render benchmarks obsolete. To mitigate this, many recent benchmarks crowdsource new prompts and evaluations from human or LLM judges; however, these can introduce significant biases, and break down when scoring hard questions. In this work, we introduce a new benchmark for LLMs designed to be immune to both test set contamination and the pitfalls of LLM judging and human crowdsourcing. We release LiveBench, the first benchmark that (1) contains frequently-updated questions from recent information sources, (2) scores answers automatically according to objective ground-truth values, and (3) contains a wide variety of challenging tasks, spanning math, coding, reasoning, language, instruction following, and data analysis. To achieve this, LiveBench contains questions that are based on recently-released math competitions, arXiv papers, news articles, and datasets, and it contains harder, contamination-free versions of tasks from previous benchmarks such as Big-Bench Hard, AMPS, and IFEval. We evaluate many prominent closed-source models, as well as dozens of open-source models ranging from 0.5B to 110B in size. LiveBench is difficult, with top models achieving below 65% accuracy. We release all questions, code, and model answers. Questions will be added and updated on a monthly basis, and we will release new tasks and harder versions of tasks over time so that LiveBench can distinguish between the capabilities of LLMs as they improve in the future. We welcome community engagement and collaboration for expanding the benchmark tasks and models.
     </details>

311. **DiVERT: Distractor Generation with Variational Errors Represented as Text for Math Multiple-choice Questions** [[pdf]](https://arxiv.org/abs/2406.19356v1) `2024-06-27` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduces DiVERT (Distractor Generation with Variational Errors Represented as Text), a novel variational approach that learns an interpretable representation of errors behind distractors in math MCQs and finds that DiVERT leads to error labels that are of comparable quality to human-authored ones.
     </details>


     <details>
          <summary>Abstract</summary>
          High-quality distractors are crucial to both the assessment and pedagogical value of multiple-choice questions (MCQs), where manually crafting ones that anticipate knowledge deficiencies or misconceptions among real students is difficult. Meanwhile, automated distractor generation, even with the help of large language models (LLMs), remains challenging for subjects like math. It is crucial to not only identify plausible distractors but also understand the error behind them. In this paper, we introduce DiVERT (Distractor Generation with Variational Errors Represented as Text), a novel variational approach that learns an interpretable representation of errors behind distractors in math MCQs. Through experiments on a real-world math MCQ dataset with 1,434 questions used by hundreds of thousands of students, we show that DiVERT, despite using a base open-source LLM with 7B parameters, outperforms state-of-the-art approaches using GPT-4o on downstream distractor generation. We also conduct a human evaluation with math educators and find that DiVERT leads to error labels that are of comparable quality to human-authored ones.
     </details>

312. **Evaluating LLMs' Mathematical and Coding Competency through Ontology-guided Interventions** [[pdf]](http://arxiv.org/abs/2401.09395) `2024-06-27` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Through comprehensive evaluations of both closed-source and open-source LLMs, a significant performance drop is shown across all the models against the perturbed questions, suggesting that the current LLMs lack robust problem solving skills and structured reasoning abilities in many areas, as defined by the ontology.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent advancements in Large Language Models (LLMs) have showcased striking results on existing logical reasoning benchmarks, with some models even surpassing human performance. However, the true depth of their competencies and robustness in reasoning tasks remains an open question. To this end, in this paper, we focus on two popular reasoning tasks: arithmetic reasoning and code generation. Particularly, we introduce (i) a general ontology of perturbations for math and coding questions, (ii) a semi-automatic method to apply these perturbations, and (iii) two datasets, GSMORE and HUMANEVAL-CORE, respectively, of perturbed math and coding problems to probe LLM capabilities in numeric reasoning and coding tasks. Through comprehensive evaluations of both closed-source and open-source LLMs, we show a significant performance drop across all the models against the perturbed questions, suggesting that the current LLMs lack robust problem solving skills and structured reasoning abilities in many areas, as defined by our ontology. We open-source the datasets and source codes at: https://github.com/declare-lab/LLM-ReasoningTest.
     </details>

313. **Step-DPO: Step-wise Preference Optimization for Long-chain Reasoning of LLMs** [[pdf]](https://arxiv.org/abs/2406.18629v1) `ICLR 2025 Submission` (7 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a simple, effective, and data-efficient method called Step-DPO, which treats individual reasoning steps as units for preference optimization rather than evaluating answers holistically, and observes that in DPO, self-generated data is more effective than data generated by humans or GPT-4, due to the latter's out-of-distribution nature.
     </details>


     <details>
          <summary>Abstract</summary>
          Mathematical reasoning presents a significant challenge for Large Language Models (LLMs) due to the extensive and precise chain of reasoning required for accuracy. Ensuring the correctness of each reasoning step is critical. To address this, we aim to enhance the robustness and factuality of LLMs by learning from human feedback. However, Direct Preference Optimization (DPO) has shown limited benefits for long-chain mathematical reasoning, as models employing DPO struggle to identify detailed errors in incorrect answers. This limitation stems from a lack of fine-grained process supervision. We propose a simple, effective, and data-efficient method called Step-DPO, which treats individual reasoning steps as units for preference optimization rather than evaluating answers holistically. Additionally, we have developed a data construction pipeline for Step-DPO, enabling the creation of a high-quality dataset containing 10K step-wise preference pairs. We also observe that in DPO, self-generated data is more effective than data generated by humans or GPT-4, due to the latter's out-of-distribution nature. Our findings demonstrate that as few as 10K preference data pairs and fewer than 500 Step-DPO training steps can yield a nearly 3% gain in accuracy on MATH for models with over 70B parameters. Notably, Step-DPO, when applied to Qwen2-72B-Instruct, achieves scores of 70.8% and 94.0% on the test sets of MATH and GSM8K, respectively, surpassing a series of closed-source models, including GPT-4-1106, Claude-3-Opus, and Gemini-1.5-Pro. Our code, data, and models are available at https://github.com/dvlab-research/Step-DPO.
     </details>

314. **Math-LLaVA: Bootstrapping Mathematical Reasoning for Multimodal Large Language Models** [[pdf]](http://arxiv.org/abs/2406.17294) `2024-06-26` (5 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This research addresses the lack of high-quality, diverse multimodal mathematical datasets by collecting 40K high-quality images with question-answer pairs from 24 existing datasets and synthesizing 320K new pairs, creating the MathV360K dataset, which enhances both the breadth and depth of multimodal mathematical questions.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have demonstrated impressive reasoning capabilities, particularly in textual mathematical problem-solving. However, existing open-source image instruction fine-tuning datasets, containing limited question-answer pairs per image, do not fully exploit visual information to enhance the multimodal mathematical reasoning capabilities of Multimodal LLMs (MLLMs). To bridge this gap, we address the lack of high-quality, diverse multimodal mathematical datasets by collecting 40K high-quality images with question-answer pairs from 24 existing datasets and synthesizing 320K new pairs, creating the MathV360K dataset, which enhances both the breadth and depth of multimodal mathematical questions. We introduce Math-LLaVA, a LLaVA-1.5-based model fine-tuned with MathV360K. This novel approach significantly improves the multimodal mathematical reasoning capabilities of LLaVA-1.5, achieving a 19-point increase and comparable performance to GPT-4V on MathVista's minitest split, and yielding leading performance on Math-V and MathVerse. Furthermore, Math-LLaVA demonstrates enhanced generalizability, showing substantial improvements on the MMMU benchmark. Our research highlights the importance of dataset diversity and synthesis in advancing MLLMs' mathematical reasoning abilities. The code and data are available at: \url{https://github.com/HZQ950419/Math-LLaVA}.
     </details>

315. **MindStar: Enhancing Math Reasoning in Pre-trained LLMs at Inference Time** [[pdf]](http://arxiv.org/abs/2405.16265) `2024-06-26` (4 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper proposes a purely inference-based searching method -- MindStar (M*), which significantly enhances the reasoning abilities of open-source models, and achieves comparable performance to GPT-3.5 and Grok-1, but with substantially reduced model size and computational costs.
     </details>


     <details>
          <summary>Abstract</summary>
          Although Large Language Models (LLMs) achieve remarkable performance across various tasks, they often struggle with complex reasoning tasks, such as answering mathematical questions. Recent efforts to address this issue have primarily focused on leveraging mathematical datasets through supervised fine-tuning or self-improvement techniques. However, these methods often depend on high-quality datasets that are difficult to prepare, or they require substantial computational resources for fine-tuning. Inspired by findings that LLMs know how to produce the right answer but struggle to select the correct reasoning path, we propose a purely inference-based searching method -- MindStar (M*). This method formulates reasoning tasks as searching problems and proposes two search ideas to identify the optimal reasoning paths. We evaluate the M* framework on both the GSM8K and MATH datasets, comparing its performance with existing open and closed-source LLMs. Our results demonstrate that M* significantly enhances the reasoning abilities of open-source models, such as Llama-2-13B and Mistral-7B, and achieves comparable performance to GPT-3.5 and Grok-1, but with substantially reduced model size and computational costs.
     </details>

316. **MathOdyssey: Benchmarking Mathematical Problem-Solving Skills in Large Language Models Using Odyssey Math Data** [[pdf]](http://arxiv.org/abs/2406.18321) `ICLR 2025 Submission` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is indicated that while LLMs perform well on routine and moderately difficult tasks, they face significant challenges with Olympiad-level problems and complex university-level questions, particularly with the most demanding problems.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have significantly advanced natural language understanding and demonstrated strong problem-solving abilities. Despite these successes, most LLMs still struggle with solving mathematical problems due to the intricate reasoning required. This paper investigates the mathematical problem-solving capabilities of LLMs using the newly developed "MathOdyssey" dataset. The dataset includes diverse mathematical problems at high school and university levels, created by experts from notable institutions to rigorously test LLMs in advanced problem-solving scenarios and cover a wider range of subject areas. By providing the MathOdyssey dataset as a resource to the AI community, we aim to contribute to the understanding and improvement of AI capabilities in complex mathematical problem-solving. We conduct benchmarking on open-source models, such as Llama-3 and DBRX-Instruct, and closed-source models from the GPT series and Gemini models. Our results indicate that while LLMs perform well on routine and moderately difficult tasks, they face significant challenges with Olympiad-level problems and complex university-level questions. Our analysis shows a narrowing performance gap between open-source and closed-source models, yet substantial challenges remain, particularly with the most demanding problems. This study highlights the ongoing need for research to enhance the mathematical reasoning of LLMs. The dataset, results, and code are publicly available.
     </details>

317. **Multiple-Choice Questions are Efficient and Robust LLM Evaluators** [[pdf]](http://arxiv.org/abs/2405.11966) `2024-06-26` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          GSM-MC, a multiple-choice (MC) dataset constructed by collecting answers and incorrect predictions on GSM8K from 60 open-source models, is presented and MATH-MC, constructed from MATH, and PythonIO, a new program reasoning MC dataset constructed from HumanEval and MBPP are introduced.
     </details>


     <details>
          <summary>Abstract</summary>
          We present GSM-MC, a multiple-choice (MC) dataset constructed by collecting answers and incorrect predictions on GSM8K from 60 open-source models. Through extensive experiments, we show that LLMs' performance on the MC version of this popular benchmark is strongly correlated with their performance on the original version and is quite robust to distractor choices and option orders, while the evaluation time is reduced by a factor of up to 30. Following similar procedures, we introduce MATH-MC, constructed from MATH, and PythonIO, a new program reasoning MC dataset constructed from HumanEval and MBPP. Experimental results indicate that LLMs' performance on these MC benchmarks leaves much room for improvement. Our data and code are available at https://github.com/Geralt-Targaryen/MC-Evaluation.
     </details>

318. **Improving Arithmetic Reasoning Ability of Large Language Models through Relation Tuples, Verification and Dynamic Feedback** [[pdf]](https://arxiv.org/abs/2406.17873v1) `2024-06-25` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper proposes to use a semi-structured form to represent reasoning steps of large language models that uses relation tuples, which are not only human-readable but also machine-friendly and easier to verify than natural language.
     </details>


     <details>
          <summary>Abstract</summary>
          Current representations used in reasoning steps of large language models can mostly be categorized into two main types: (1) natural language, which is difficult to verify; and (2) non-natural language, usually programming code, which is difficult for people who are unfamiliar with coding to read. In this paper, we propose to use a semi-structured form to represent reasoning steps of large language models. Specifically, we use relation tuples, which are not only human-readable but also machine-friendly and easier to verify than natural language. We implement a framework that includes three main components: (1) introducing relation tuples into the reasoning steps of large language models; (2) implementing an automatic verification process of reasoning steps with a local code interpreter based on relation tuples; and (3) integrating a simple and effective dynamic feedback mechanism, which we found helpful for self-improvement of large language models. The experimental results on various arithmetic datasets demonstrate the effectiveness of our method in improving the arithmetic reasoning ability of large language models. The source code is available at https://github.com/gpgg/art.
     </details>

319. **Task Oriented In-Domain Data Augmentation** [[pdf]](http://arxiv.org/abs/2406.16694) `2024-06-24` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          TRAIT, a task-oriented in-domain data augmentation framework, is proposed and adapted to two domains: advertisement and math, which improves LLM performance by 8% in the advertisement domain and 7.5% in the math domain.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have shown superior performance in various applications and fields. To achieve better performance on specialized domains such as law and advertisement, LLMs are often continue pre-trained on in-domain data. However, existing approaches suffer from two major issues. First, in-domain data are scarce compared with general domain-agnostic data. Second, data used for continual pre-training are not task-aware, such that they may not be helpful to downstream applications. We propose TRAIT, a task-oriented in-domain data augmentation framework. Our framework is divided into two parts: in-domain data selection and task-oriented synthetic passage generation. The data selection strategy identifies and selects a large amount of in-domain data from general corpora, and thus significantly enriches domain knowledge in the continual pre-training data. The synthetic passages contain guidance on how to use domain knowledge to answer questions about downstream tasks. By training on such passages, the model aligns with the need of downstream applications. We adapt LLMs to two domains: advertisement and math. On average, TRAIT improves LLM performance by 8% in the advertisement domain and 7.5% in the math domain.
     </details>

320. **The Impact of Reasoning Step Length on Large Language Models** [[pdf]](http://arxiv.org/abs/2401.04925) `ACL 2024 Findings` (28 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The results indicate that lengthening the reasoning steps in prompts, even without adding new information into the prompt, considerably enhances LLMs' reasoning abilities across multiple datasets, and shows that even incorrect rationales can yield favorable outcomes if they maintain the requisite length of inference.
     </details>


     <details>
          <summary>Abstract</summary>
          Chain of Thought (CoT) is significant in improving the reasoning abilities of large language models (LLMs). However, the correlation between the effectiveness of CoT and the length of reasoning steps in prompts remains largely unknown. To shed light on this, we have conducted several empirical experiments to explore the relations. Specifically, we design experiments that expand and compress the rationale reasoning steps within CoT demonstrations, while keeping all other factors constant. We have the following key findings. First, the results indicate that lengthening the reasoning steps in prompts, even without adding new information into the prompt, considerably enhances LLMs’ reasoning abilities across multiple datasets. Alternatively, shortening the reasoning steps, even while preserving the key information, significantly diminishes the reasoning abilities of models. This finding highlights the importance of the number of steps in CoT prompts and provides practical guidance to make better use of LLMs’ potential in complex problem-solving scenarios. Second, we also investigated the relationship between the performance of CoT and the rationales used in demonstrations. Surprisingly, the result shows that even incorrect rationales can yield favorable outcomes if they maintain the requisite length of inference. Third, we observed that the advantages of increasing reasoning steps are task-dependent: simpler tasks require fewer steps, whereas complex tasks gain significantly from longer inference sequences.
     </details>

321. **RL on Incorrect Synthetic Data Scales the Efficiency of LLM Math Reasoning by Eight-Fold** [[pdf]](https://arxiv.org/abs/2406.14532v1) `2024-06-20` (10 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is shown that training on per-step negatives can help to unlearn spurious correlations in the positive data, and is equivalent to advantage-weighted reinforcement learning (RL), implying that it inherits robustness benefits of RL over imitating positive data alone.
     </details>


     <details>
          <summary>Abstract</summary>
          Training on model-generated synthetic data is a promising approach for finetuning LLMs, but it remains unclear when it helps or hurts. In this paper, we investigate this question for math reasoning via an empirical study, followed by building a conceptual understanding of our observations. First, we find that while the typical approach of finetuning a model on synthetic correct or positive problem-solution pairs generated by capable models offers modest performance gains, sampling more correct solutions from the finetuned learner itself followed by subsequent fine-tuning on this self-generated data $\textbf{doubles}$ the efficiency of the same synthetic problems. At the same time, training on model-generated positives can amplify various spurious correlations, resulting in flat or even inverse scaling trends as the amount of data increases. Surprisingly, we find that several of these issues can be addressed if we also utilize negative responses, i.e., model-generated responses that are deemed incorrect by a final answer verifier. Crucially, these negatives must be constructed such that the training can appropriately recover the utility or advantage of each intermediate step in the negative response. With this per-step scheme, we are able to attain consistent gains over only positive data, attaining performance similar to amplifying the amount of synthetic data by $\mathbf{8 \times}$. We show that training on per-step negatives can help to unlearn spurious correlations in the positive data, and is equivalent to advantage-weighted reinforcement learning (RL), implying that it inherits robustness benefits of RL over imitating positive data alone.
     </details>

322. **Can LLMs Reason in the Wild with Programs?** [[pdf]](http://arxiv.org/abs/2406.13764) `2024-06-19` (4 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces the task of reasoning in the wild, where an LLM is tasked to solve a reasoning problem of unknown type by identifying the subproblems and their corresponding formalisms, and writing a program to solve each subproblem, guided by a tactic.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have shown superior capability to solve reasoning problems with programs. While being a promising direction, most of such frameworks are trained and evaluated in settings with a prior knowledge of task requirements. However, as LLMs become more capable, it is necessary to assess their reasoning abilities in more realistic scenarios where many real-world problems are open-ended with ambiguous scope, and often require multiple formalisms to solve. To investigate this, we introduce the task of reasoning in the wild, where an LLM is tasked to solve a reasoning problem of unknown type by identifying the subproblems and their corresponding formalisms, and writing a program to solve each subproblem, guided by a tactic. We create a large tactic-guided trajectory dataset containing detailed solutions to a diverse set of reasoning problems, ranging from well-defined single-form reasoning (e.g., math, logic), to ambiguous and hybrid ones (e.g., commonsense, combined math and logic). This allows us to test various aspects of LLMs reasoning at the fine-grained level such as the selection and execution of tactics, and the tendency to take undesired shortcuts. In experiments, we highlight that existing LLMs fail significantly on problems with ambiguous and mixed scope, revealing critical limitations and overfitting issues (e.g. accuracy on GSM8K drops by at least 50\%). We further show the potential of finetuning a local LLM on the tactic-guided trajectories in achieving better performance. Project repo is available at github.com/gblackout/Reason-in-the-Wild
     </details>

323. **Mathador-LM: A Dynamic Benchmark for Mathematical Reasoning on Large Language Models** [[pdf]](http://arxiv.org/abs/2406.12572) `2024-06-19` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is shown that, across leading LLMs, it is possible to obtain stable average performance while generating benchmark instances dynamically, following a target difficulty level, which alleviates concerns about test-set leakage into training data, an issue that often undermines popular benchmarks.
     </details>


     <details>
          <summary>Abstract</summary>
          We introduce Mathador-LM, a new benchmark for evaluating the mathematical reasoning on large language models (LLMs), combining ruleset interpretation, planning, and problem-solving. This benchmark is inspired by the Mathador game, where the objective is to reach a target number using basic arithmetic operations on a given set of base numbers, following a simple set of rules. We show that, across leading LLMs, we obtain stable average performance while generating benchmark instances \emph{dynamically}, following a target difficulty level. Thus, our benchmark alleviates concerns about test-set leakage into training data, an issue that often undermines popular benchmarks. Additionally, we conduct a comprehensive evaluation of both open and closed-source state-of-the-art LLMs on Mathador-LM. Our findings reveal that contemporary models struggle with Mathador-LM, scoring significantly lower than average 3rd graders. This stands in stark contrast to their strong performance on popular mathematical reasoning benchmarks. The implementation of Mathador-LM benchmark is available at \href{https://github.com/IST-DASLab/Mathador-LM}{github.com/IST-DASLab/Mathador-LM}.
     </details>

324. **GeoGPT4V: Towards Geometric Multi-modal Large Language Models with Geometric Image Generation** [[pdf]](https://arxiv.org/abs/2406.11503v1) `2024-06-17` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel pipeline is introduced that leverages GPT-4 and GPT-4V to generate relatively basic geometry problems with aligned text and images, facilitating model learning and demonstrates that the GeoGPT4V dataset significantly improves the geometry performance of various models on the MathVista and MathVision benchmarks.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models have seen widespread adoption in math problem-solving. However, in geometry problems that usually require visual aids for better understanding, even the most advanced multi-modal models currently still face challenges in effectively using image information. High-quality data is crucial for enhancing the geometric capabilities of multi-modal models, yet existing open-source datasets and related efforts are either too challenging for direct model learning or suffer from misalignment between text and images. To overcome this issue, we introduce a novel pipeline that leverages GPT-4 and GPT-4V to generate relatively basic geometry problems with aligned text and images, facilitating model learning. We have produced a dataset of 4.9K geometry problems and combined it with 19K open-source data to form our GeoGPT4V dataset. Experimental results demonstrate that the GeoGPT4V dataset significantly improves the geometry performance of various models on the MathVista and MathVision benchmarks. The code is available at https://github.com/Lanyu0303/GeoGPT4V_Project
     </details>

325. **On the Empirical Complexity of Reasoning and Planning in LLMs** [[pdf]](http://arxiv.org/abs/2404.11041) `2024-06-17` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Experimental case studies suggest that both CoT and ToT benefit significantly from task decomposition, and for computationally hard reasoning tasks, the more sophisticated tree structure of ToT outperforms the linear structure of CoT.
     </details>


     <details>
          <summary>Abstract</summary>
          Chain-of-thought (CoT), tree-of-thought (ToT), and related techniques work surprisingly well in practice for some complex reasoning tasks with Large Language Models (LLMs), but why? This work seeks the underlying reasons by conducting experimental case studies and linking the performance benefits to well-established sample and computational complexity principles in machine learning. We experimented with 6 reasoning tasks, ranging from grade school math, air travel planning, ..., to Blocksworld. The results suggest that (i) both CoT and ToT benefit significantly from task decomposition, which breaks a complex reasoning task into a sequence of steps with low sample complexity and explicitly outlines the reasoning structure, and (ii) for computationally hard reasoning tasks, the more sophisticated tree structure of ToT outperforms the linear structure of CoT. These findings provide useful guidelines for the use of LLM in solving reasoning tasks in practice.
     </details>

326. **Probabilistic Reasoning in Generative Large Language Models** [[pdf]](http://arxiv.org/abs/2402.09614) `2024-06-17` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduces the Bayesian Linguistic Inference Dataset (BLInD), a new dataset specifically designed to test the probabilistic reasoning capabilities of LLMs, and presents several prompting strategies that map the problem to different formal representations, including Python code, probabilistic algorithms, and probabilistic logical programming.
     </details>


     <details>
          <summary>Abstract</summary>
          This paper considers the challenges Large Language Models (LLMs) face when reasoning over text that includes information involving uncertainty explicitly quantified via probability values. This type of reasoning is relevant to a variety of contexts ranging from everyday conversations to medical decision-making. Despite improvements in the mathematical reasoning capabilities of LLMs, they still exhibit significant difficulties when it comes to probabilistic reasoning. To deal with this problem, we introduce the Bayesian Linguistic Inference Dataset (BLInD), a new dataset specifically designed to test the probabilistic reasoning capabilities of LLMs. We use BLInD to find out the limitations of LLMs for tasks involving probabilistic reasoning. In addition, we present several prompting strategies that map the problem to different formal representations, including Python code, probabilistic algorithms, and probabilistic logical programming. We conclude by providing an evaluation of our methods on BLInD and an adaptation of a causal reasoning question-answering dataset. Our empirical results highlight the effectiveness of our proposed strategies for multiple LLMs.
     </details>

327. **How Do Humans Write Code? Large Models Do It the Same Way Too** [[pdf]](http://arxiv.org/abs/2402.15729) `2024-06-17` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Human-Think Language is proposed, which leverages a suite of strategies that help integrate PoT and CoT, encompassing a new generation paradigm that uses full CoT reasoning to control code generation and shows the most significant improvement in non-mathematical natural language inference task.
     </details>


     <details>
          <summary>Abstract</summary>
          Program-of-Thought (PoT) replaces natural language-based Chain-of-Thought (CoT) as the most popular method in Large Language Models (LLMs) mathematical reasoning tasks by utilizing external tool calls to circumvent computational errors. However, our evaluation of the GPT-4 and Llama series reveals that using PoT introduces more reasoning errors, such as incorrect formulas or flawed logic, compared to CoT. To address this issue, we propose Human-Think Language (HTL), which leverages a suite of strategies that help integrate PoT and CoT, encompassing: (1) a new generation paradigm that uses full CoT reasoning to control code generation. (2) Focus Attention, that directs model attention to the CoT reasoning during PoT to generate more logical code. (3) reinforcement learning that utilizes the accuracy of both CoT and PoT responses as rewards to prevent repetitive reasoning steps in LLMs when solving difficult math problems. Our method achieves an average improvement of 6.5% on the Llama-Base model and 4.3% on the Mistral-Base model across 8 mathematical calculation datasets. It also shows significant effectiveness on five out-of-domain datasets by controlling the model's information flow, exhibiting strong transferability. Additionally, HTL shows the most significant improvement in non-mathematical natural language inference task, contributing to a unified reasoning task framework
     </details>

328. **Learn Beyond The Answer: Training Language Models with Reflection for Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2406.12050) `2024-06-17` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          reflective augmentation is proposed, a method that embeds problem reflection into each training instance, thereby fostering a thorough comprehension through reflective reasoning and its complementary nature relative to existing augmentation techniques.
     </details>


     <details>
          <summary>Abstract</summary>
          Supervised fine-tuning enhances the problem-solving abilities of language models across various mathematical reasoning tasks. To maximize such benefits, existing research focuses on broadening the training set with various data augmentation techniques, which is effective for standard single-round question-answering settings. Our work introduces a novel technique aimed at cultivating a deeper understanding of the training problems at hand, enhancing performance not only in standard settings but also in more complex scenarios that require reflective thinking. Specifically, we propose reflective augmentation, a method that embeds problem reflection into each training instance. It trains the model to consider alternative perspectives and engage with abstractions and analogies, thereby fostering a thorough comprehension through reflective reasoning. Extensive experiments validate the achievement of our aim, underscoring the unique advantages of our method and its complementary nature relative to existing augmentation techniques.
     </details>

329. **Mathematical Entities: Corpora and Benchmarks** [[pdf]](http://arxiv.org/abs/2406.11577) `2024-06-17` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Mathematics is a highly specialized domain with its own unique set of challenges. Despite this, there has been relatively little research on natural language processing for mathematical texts, and there are few mathematical language resources aimed at NLP. In this paper, we aim to provide annotated corpora that can be used to study the language of mathematics in different contexts, ranging from fundamental concepts found in textbooks to advanced research mathematics. We preprocess the corpora with a neural parsing model and some manual intervention to provide part-of-speech tags, lemmas, and dependency trees. In total, we provide 182397 sentences across three corpora. We then aim to test and evaluate several noteworthy natural language processing models using these corpora, to show how well they can adapt to the domain of mathematics and provide useful tools for exploring mathematical language. We evaluate several neural and symbolic models against benchmarks that we extract from the corpus metadata to show that terminology extraction and definition extraction do not easily generalize to mathematics, and that additional work is needed to achieve good performance on these metrics. Finally, we provide a learning assistant that grants access to the content of these corpora in a context-sensitive manner, utilizing text search and entity linking. Though our corpora and benchmarks provide useful metrics for evaluating mathematical language processing, further work is necessary to adapt models to mathematics in order to provide more effective learning assistants and apply NLP methods to different mathematical domains.
     </details>

330. **Step-level Value Preference Optimization for Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2406.10858) `2024-06-16` (4 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces a novel algorithm called Step-level Value Preference Optimization (SVPO), which employs Monte Carlo Tree Search (MCTS) and an explicit value model to replicate the behavior of the implicit reward model, complementing standard preference optimization.
     </details>


     <details>
          <summary>Abstract</summary>
          Direct Preference Optimization (DPO) using an implicit reward model has proven to be an effective alternative to reinforcement learning from human feedback (RLHF) for fine-tuning preference aligned large language models (LLMs). However, the overall preference annotations of responses do not fully capture the fine-grained quality of model outputs in complex multi-step reasoning tasks, such as mathematical reasoning. To address this limitation, we introduce a novel algorithm called Step-level Value Preference Optimization (SVPO). Our approach employs Monte Carlo Tree Search (MCTS) to automatically annotate step-level preferences for multi-step reasoning. Furthermore, from the perspective of learning-to-rank, we train an explicit value model to replicate the behavior of the implicit reward model, complementing standard preference optimization. This value model enables the LLM to generate higher reward responses with minimal cost during inference. Experimental results demonstrate that our method achieves state-of-the-art performance on both in-domain and out-of-domain mathematical reasoning benchmarks. Our code is available at \url{https://github.com/MARIO-Math-Reasoning/Super_MARIO}.
     </details>

331. **Exposing the Achilles' Heel: Evaluating LLMs Ability to Handle Mistakes in Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2406.10834) `ICLR 2025 Submission` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces a novel dataset MWP-MISTAKE, incorporating MWPs with both correct and incorrect reasoning steps generated through rule-based methods and smaller language models, and highlights GPT-$o's superior performance in mistake detection and rectification and the persistent challenges faced by smaller models.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have been applied to Math Word Problems (MWPs) with transformative impacts, revolutionizing how these complex problems are approached and solved in various domains including educational settings. However, the evaluation of these models often prioritizes final accuracy, overlooking the crucial aspect of reasoning capabilities. This work addresses this gap by focusing on the ability of LLMs to detect and correct reasoning mistakes. We introduce a novel dataset MWP-MISTAKE, incorporating MWPs with both correct and incorrect reasoning steps generated through rule-based methods and smaller language models. Our comprehensive benchmarking reveals significant insights into the strengths and weaknesses of state-of-the-art models, such as GPT-4o, GPT-4, GPT-3.5Turbo, and others. We highlight GPT-$o's superior performance in mistake detection and rectification and the persistent challenges faced by smaller models. Additionally, we identify issues related to data contamination and memorization, impacting the reliability of LLMs in real-world applications. Our findings emphasize the importance of rigorous evaluation of reasoning processes and propose future directions to enhance the generalization and robustness of LLMs in mathematical problem-solving.
     </details>

332. **RUPBench: Benchmarking Reasoning Under Perturbations for Robustness Evaluation in Large Language Models** [[pdf]](http://arxiv.org/abs/2406.11020) `2024-06-16` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work presents RUPBench, a comprehensive benchmark designed to evaluate LLM robustness across diverse reasoning tasks, and highlights that larger models tend to exhibit greater robustness to perturbations.
     </details>


     <details>
          <summary>Abstract</summary>
          With the increasing use of large language models (LLMs), ensuring reliable performance in diverse, real-world environments is essential. Despite their remarkable achievements, LLMs often struggle with adversarial inputs, significantly impacting their effectiveness in practical applications. To systematically understand the robustness of LLMs, we present RUPBench, a comprehensive benchmark designed to evaluate LLM robustness across diverse reasoning tasks. Our benchmark incorporates 15 reasoning datasets, categorized into commonsense, arithmetic, logical, and knowledge-intensive reasoning, and introduces nine types of textual perturbations at lexical, syntactic, and semantic levels. By examining the performance of state-of-the-art LLMs such as GPT-4o, Llama3, Phi-3, and Gemma on both original and perturbed datasets, we provide a detailed analysis of their robustness and error patterns. Our findings highlight that larger models tend to exhibit greater robustness to perturbations. Additionally, common error types are identified through manual inspection, revealing specific challenges faced by LLMs in different reasoning contexts. This work provides insights into areas where LLMs need further improvement to handle diverse and noisy inputs effectively.
     </details>

333. **Teaching Large Language Models to Reason with Reinforcement Learning** [[pdf]](https://openreview.net/forum?id=mjqoceuMnI) `ICML 2024 Workshop AI for Math` (24 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is concluded that during RL training models fail to explore significantly beyond solutions already produced by SFT models, and the implications of these findings for RLHF and the future role of RL in LLM fine-tuning are discussed.
     </details>


     <details>
          <summary>Abstract</summary>
          Reinforcement Learning from Human Feedback (\textbf{RLHF}) has emerged as a dominant approach for aligning LLM outputs with human preferences. Inspired by the success of RLHF, we study the performance of multiple algorithms that learn from feedback (Expert Iteration, Proximal Policy Optimization (\textbf{PPO}), Return-Conditioned RL) on improving LLM reasoning capabilities. We investigate both sparse and dense rewards provided to the LLM both heuristically and via a learned reward model. We additionally start from multiple model sizes and initializations both with and without supervised fine-tuning (\textbf{SFT}) data. Overall, we find all algorithms perform comparably, with Expert Iteration performing best in most cases. Surprisingly, we find the sample complexity of Expert Iteration is similar to that of PPO, requiring at most on the order of $10^6$ samples to converge from a pretrained checkpoint. We investigate why this is the case, concluding that during RL training models fail to explore significantly beyond solutions already produced by SFT models. Additionally, we discuss a trade off between maj@1 and pass@96 metric performance during SFT training and how conversely RL training improves both simultaneously. We then conclude by discussing the implications of our findings for RLHF and the future role of RL in LLM fine-tuning.
     </details>

334. **Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B** [[pdf]](http://arxiv.org/abs/2406.07394) `2024-06-13` (9 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          This paper introduces the MCT Self-Refine (MCTSr) algorithm, an innovative integration of Large Language Models (LLMs) with Monte Carlo Tree Search (MCTS), designed to enhance performance in complex mathematical reasoning tasks. Addressing the challenges of accuracy and reliability in LLMs, particularly in strategic and mathematical reasoning, MCTSr leverages systematic exploration and heuristic self-refine mechanisms to improve decision-making frameworks within LLMs. The algorithm constructs a Monte Carlo search tree through iterative processes of Selection, self-refine, self-evaluation, and Backpropagation, utilizing an improved Upper Confidence Bound (UCB) formula to optimize the exploration-exploitation balance. Extensive experiments demonstrate MCTSr's efficacy in solving Olympiad-level mathematical problems, significantly improving success rates across multiple datasets, including GSM8K, GSM Hard, MATH, and Olympiad-level benchmarks, including Math Odyssey, AIME, and OlympiadBench. The study advances the application of LLMs in complex reasoning tasks and sets a foundation for future AI integration, enhancing decision-making accuracy and reliability in LLM-driven applications.
     </details>

335. **VerityMath: Advancing Mathematical Reasoning by Self-Verification Through Unit Consistency** [[pdf]](https://openreview.net/forum?id=S9utaRXaZt&name=pdf) `2024-06-13` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper studies the performance of strong open-source LLMs, including Llama 2 (7B), Code Llama (7B), and Mistral (7B) on math word problems using program-based solving techniques, and proposes a systematic approach that incorporates unit consistency programs.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs), combined with program-based solving techniques, are increasingly demonstrating proficiency in mathematical reasoning. For example, closed-source models such as OpenAI GPT-4 and Claude show excellent results in solving math word problems. However, progress in math word problem-solving for open-source LLMs is limited, and the challenges these models face are not well-studied. In this paper, we study the performance of strong open-source LLMs, including Llama 2 (7B), Code Llama (7B), and Mistral (7B) on math word problems using program-based solving techniques. Specifically, we analyze the outputs of these models when applied to math word problems and identify a category of problems that pose a significant challenge, particularly those involving quantities spanning multiple units. To address this issue, we propose a systematic approach by defining the units for each quantity and ensuring the consistency of these units during mathematical operations. We developed Unit Consistency Programs (UCPs), an annotated dataset of math word problems, each paired with programs containing unit specifications and unit verification routines. We fine-tuned Llama 2 (7B), Code Llama (7B), and Mistral (7B) models with UCPs to produce theirVerityMath variants. Our findings indicate that our approach, which incorporates unit consistency, currently slightly underperforms compared to an approach that does not. To understand the reasons behind this, we conduct an in-depth error analysis and suggest options for future improvements. Our code and dataset are available at https://github.com/vernontoh/VerityMath.
     </details>

336. **Distilling LLMs' Decomposition Abilities into Compact Language Models** [[pdf]](https://openreview.net/forum?id=XM44SZM3VO&name=pdf) `2024-06-13` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This study focuses on distilling the LLMs' decomposition skills into compact models using offline reinforcement learning, leveraging the advancements in the LLM`s capabilities to provide feedback and generate a specialized task-specific dataset for training compact models.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have demonstrated proficiency in their reasoning abilities, yet their large size presents scalability challenges and limits any further customization. In contrast, compact models offer customized training but often fall short in solving complex reasoning tasks. This study focuses on distilling the LLMs' decomposition skills into compact models using offline reinforcement learning. We leverage the advancements in the LLM`s capabilities to provide feedback and generate a specialized task-specific dataset for training compact models. The development of an AI-generated dataset and the establishment of baselines constitute the primary contributions of our work, underscoring the potential of compact models in replicating complex problem-solving skills.
     </details>

337. **Progress or Regress? Self-Improvement Reversal in Post-training** [[pdf]](https://openreview.net/forum?id=MG18DR2dAN) `ICLR 2025 Submission` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Findings indicate that current self-improvement practices through post-training are inadequate for equipping models to tackle more complex problems and underscore the necessity of critical evaluation metrics in discerning the progress or regress dichotomy for self-improving LLMs.
     </details>


     <details>
          <summary>Abstract</summary>
          Self-improvement through post-training methods such as iterative preference learning has been acclaimed for enhancing the problem-solving capabilities (e.g., mathematical reasoning) of Large Language Models (LLMs) without human intervention. However, as exploration deepens, it becomes crucial to assess whether these improvements genuinely signify progress in solving more challenging problems or if they could lead to unintended regressions. To address this, we propose a comprehensive evaluative framework that goes beyond the superficial pass@1 metric to scrutinize the underlying enhancements of post-training paradigms for self-improvement. Through rigorous experimentation and analysis across diverse problem-solving tasks, the empirical results point out the phenomenon of \emph{self-improvement reversal}, where models showing improved performance across benchmarks will paradoxically exhibit declines in broader, essential capabilities, like output diversity and out-of-distribution (OOD) generalization. These findings indicate that current self-improvement practices through post-training are inadequate for equipping models to tackle more complex problems. Furthermore, they underscore the necessity of our critical evaluation metrics in discerning the \emph{progress or regress} dichotomy for self-improving LLMs.
     </details>

338. **Pre-Calc: Learning to Use the Calculator Improves Numeracy in Language Models** [[pdf]](https://openreview.net/forum?id=Hb5gA02FyR&name=pdf) `2024-06-13` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper proposes Pre-Calc, a simple pre-finetuning objective of learning to use the calculator for both encoder-only and encoder-decoder architectures, formulated as a discriminative and generative task respectively.
     </details>


     <details>
          <summary>Abstract</summary>
          Quantitative and numerical comprehension in language is an important task in many fields like education and finance, but still remains a challenging task for language models. While tool and calculator usage has shown to be helpful to improve mathematical reasoning in large pretrained decoder-only language models, this remains unexplored for smaller language models with encoders. In this paper, we propose Pre-Calc, a simple pre-finetuning objective of learning to use the calculator for both encoder-only and encoder-decoder architectures, formulated as a discriminative and generative task respectively. We pre-train BERT and RoBERTa for discriminative calculator use and Flan-T5 for generative calculator use on the MAWPS, SVAMP, and AsDiv-A datasets, which improves performance on downstream tasks that require numerical understanding. Our code and data are available at https://github.com/calc-cmu/pre-calc.
     </details>

339. **Smart Vision-Language Reasoners** [[pdf]](https://openreview.net/forum?id=Mf6ot5U7ni&name=pdf) `2024-06-13` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This article employs the abstractions given in the SMART task (Simple Multimodal Algorithmic Reasoning Task) introduced incherian2022deep as meta-reasoning and problem-solving skills along eight axes: math, counting, path, measure, logic, spatial, and pattern.
     </details>


     <details>
          <summary>Abstract</summary>
          In this article, we investigate vision-language models (VLM) as reasoners. The ability to form abstractions underlies mathematical reasoning, problem-solving, and other Math AI tasks. Several formalisms have been given to these underlying abstractions and skills utilized by humans and intelligent systems for reasoning. Furthermore, human reasoning is inherently multimodal, and as such, we focus our investigations on multimodal AI. In this article, we employ the abstractions given in the SMART task (Simple Multimodal Algorithmic Reasoning Task) introduced in \cite{cherian2022deep} as meta-reasoning and problem-solving skills along eight axes: math, counting, path, measure, logic, spatial, and pattern. We investigate the ability of vision-language models to reason along these axes and seek avenues of improvement. Including composite representations with vision-language cross-attention enabled learning multimodal representations adaptively from fused frozen pretrained backbones for better visual grounding. Furthermore, proper hyperparameter and other training choices led to strong improvements (up to $48\%$ gain in accuracy) on the SMART task, further underscoring the power of deep multimodal learning. The smartest VLM, which includes a novel QF multimodal layer, improves upon the best previous baselines in every one of the eight fundamental reasoning skills. End-to-end code is available at https://github.com/smarter-vlm/smarter.
     </details>

340. **Improving Autoformalization using Type Checking** [[pdf]](http://arxiv.org/abs/2406.07222) `2024-06-11` `Lean` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a method to fix the performance of large language models for autoformalization through decoding with type-check filtering, where they initially sample a diverse set of candidate formalizations for an informal statement, then use the Lean proof assistant to filter out candidates that do not type-check.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models show promise for autoformalization, the task of automatically translating natural language into formal languages. However, current autoformalization methods remain limited. The last reported state-of-the-art performance on the ProofNet formalization benchmark for the Lean proof assistant, achieved using Codex for Lean 3, only showed successful formalization of 16.1% of informal statements. Similarly, our evaluation of GPT-4o for Lean 4 only produces successful translations 34.9% of the time. Our analysis shows that the performance of these models is largely limited by their inability to generate formal statements that successfully type-check (i.e., are syntactically correct and consistent with types) - with a whopping 86.6% of GPT-4o errors starting from a type-check failure. In this work, we propose a method to fix this issue through decoding with type-check filtering, where we initially sample a diverse set of candidate formalizations for an informal statement, then use the Lean proof assistant to filter out candidates that do not type-check. Using GPT-4o as a base model, and combining our method with self-consistency, we obtain a +18.3% absolute increase in formalization accuracy, and achieve a new state-of-the-art of 53.2% on ProofNet with Lean 4.
     </details>

341. **Autograding Mathematical Induction Proofs with Natural Language Processing** [[pdf]](http://arxiv.org/abs/2406.10268) `2024-06-11` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper presents a set of training methods and models capable of autograding freeform mathematical proofs by leveraging existing large language models and other machine learning techniques and finds that the best grading model is also more accurate than most human graders.
     </details>


     <details>
          <summary>Abstract</summary>
          In mathematical proof education, there remains a need for interventions that help students learn to write mathematical proofs. Research has shown that timely feedback can be very helpful to students learning new skills. While for many years natural language processing models have struggled to perform well on tasks related to mathematical texts, recent developments in natural language processing have created the opportunity to complete the task of giving students instant feedback on their mathematical proofs. In this paper, we present a set of training methods and models capable of autograding freeform mathematical proofs by leveraging existing large language models and other machine learning techniques. The models are trained using proof data collected from four different proof by induction problems. We use four different robust large language models to compare their performances, and all achieve satisfactory performances to various degrees. Additionally, we recruit human graders to grade the same proofs as the training data, and find that the best grading model is also more accurate than most human graders. With the development of these grading models, we create and deploy an autograder for proof by induction problems and perform a user study with students. Results from the study shows that students are able to make significant improvements to their proofs using the feedback from the autograder, but students still do not trust the AI autograders as much as they trust human graders. Future work can improve on the autograder feedback and figure out ways to help students trust AI autograders.
     </details>

342. **Husky: A Unified, Open-Source Language Agent for Multi-Step Reasoning** [[pdf]](http://arxiv.org/abs/2406.06469) `2024-06-10` (4 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Husky is introduced, a holistic, open-source language agent that learns to reason over a unified action space to address a diverse set of complex tasks involving numerical, tabular, and knowledge-based reasoning.
     </details>


     <details>
          <summary>Abstract</summary>
          Language agents perform complex tasks by using tools to execute each step precisely. However, most existing agents are based on proprietary models or designed to target specific tasks, such as mathematics or multi-hop question answering. We introduce Husky, a holistic, open-source language agent that learns to reason over a unified action space to address a diverse set of complex tasks involving numerical, tabular, and knowledge-based reasoning. Husky iterates between two stages: 1) generating the next action to take towards solving a given task and 2) executing the action using expert models and updating the current solution state. We identify a thorough ontology of actions for addressing complex tasks and curate high-quality data to train expert models for executing these actions. Our experiments show that Husky outperforms prior language agents across 14 evaluation datasets. Moreover, we introduce HuskyQA, a new evaluation set which stress tests language agents for mixed-tool reasoning, with a focus on retrieving missing knowledge and performing numerical reasoning. Despite using 7B models, Husky matches or even exceeds frontier LMs such as GPT-4 on these tasks, showcasing the efficacy of our holistic approach in addressing complex reasoning problems. Our code and models are available at https://github.com/agent-husky/Husky-v1.
     </details>

343. **LLMs Are Not Intelligent Thinkers: Introducing Mathematical Topic Tree Benchmark for Comprehensive Evaluation of LLMs** [[pdf]](http://arxiv.org/abs/2406.05194) `2024-06-07` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) demonstrate impressive capabilities in mathematical reasoning. However, despite these achievements, current evaluations are mostly limited to specific mathematical topics, and it remains unclear whether LLMs are genuinely engaging in reasoning. To address these gaps, we present the Mathematical Topics Tree (MaTT) benchmark, a challenging and structured benchmark that offers 1,958 questions across a wide array of mathematical subjects, each paired with a detailed hierarchical chain of topics. Upon assessing different LLMs using the MaTT benchmark, we find that the most advanced model, GPT-4, achieved a mere 54\% accuracy in a multiple-choice scenario. Interestingly, even when employing Chain-of-Thought prompting, we observe mostly no notable improvement. Moreover, LLMs accuracy dramatically reduced by up to 24.2 percentage point when the questions were presented without providing choices. Further detailed analysis of the LLMs' performance across a range of topics showed significant discrepancy even for closely related subtopics within the same general mathematical area. In an effort to pinpoint the reasons behind LLMs performances, we conducted a manual evaluation of the completeness and correctness of the explanations generated by GPT-4 when choices were available. Surprisingly, we find that in only 53.3\% of the instances where the model provided a correct answer, the accompanying explanations were deemed complete and accurate, i.e., the model engaged in genuine reasoning.
     </details>

344. **Robustness Assessment of Mathematical Reasoning in the Presence of Missing and Contradictory Conditions** [[pdf]](http://arxiv.org/abs/2406.05055) `2024-06-07` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have demonstrated impressive performance on reasoning tasks, which can be further improved through few-shot prompting techniques. However, the current evaluation primarily focuses on carefully constructed benchmarks and neglects the consideration of real-world reasoning problems that present missing and contradictory conditions, known as ill-defined problems. Our observations suggest that existing few-shot prompting techniques are ineffective in such scenarios, often providing overconfident answers or hallucination. To further study this problem, we develop a benchmark called Problems with Missing and Contradictory conditions (PMC) and introduce two novel metrics to evaluate the performance of few-shot prompting methods in these scenarios. Our analysis using the PMC benchmark reveals a trade-off dilemma between the performance of mathematical reasoning for well-defined problems and the ability to recognize ill-defined problems. To address the challenges posed by PMC, we propose a novel few-shot prompting method called SMT-LIB Prompting (SLP), which utilizes the SMT-LIB language to model the problems instead of solving them directly. Subsequently, a double-check solving strategy checks the satisfiability and uniqueness of the solution and provides final feedback. Extensive experiments demonstrate the superiority of our SLP approach compared to existing few-shot prompting methods when dealing with problems with missing and contradictory conditions. We will open-source our benchmark and code to facilitate future research.
     </details>

345. **Improve Mathematical Reasoning in Language Models by Automated Process Supervision** [[pdf]](http://arxiv.org/abs/2406.06592) `2024-06-05` (23 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel divide-and-conquer style Monte Carlo Tree Search (MCTS) algorithm named OmegaPRM is proposed, which swiftly identifies the first error in the Chain of Thought with binary search and balances the positive and negative examples, thereby ensuring both efficiency and quality.
     </details>


     <details>
          <summary>Abstract</summary>
          Complex multi-step reasoning tasks, such as solving mathematical problems or generating code, remain a significant hurdle for even the most advanced large language models (LLMs). Verifying LLM outputs with an Outcome Reward Model (ORM) is a standard inference-time technique aimed at enhancing the reasoning performance of LLMs. However, this still proves insufficient for reasoning tasks with a lengthy or multi-hop reasoning chain, where the intermediate outcomes are neither properly rewarded nor penalized. Process supervision addresses this limitation by assigning intermediate rewards during the reasoning process. To date, the methods used to collect process supervision data have relied on either human annotation or per-step Monte Carlo estimation, both prohibitively expensive to scale, thus hindering the broad application of this technique. In response to this challenge, we propose a novel divide-and-conquer style Monte Carlo Tree Search (MCTS) algorithm named \textit{OmegaPRM} for the efficient collection of high-quality process supervision data. This algorithm swiftly identifies the first error in the Chain of Thought (CoT) with binary search and balances the positive and negative examples, thereby ensuring both efficiency and quality. As a result, we are able to collect over 1.5 million process supervision annotations to train a Process Reward Model (PRM). Utilizing this fully automated process supervision alongside the weighted self-consistency algorithm, we have enhanced the instruction tuned Gemini Pro model's math reasoning performance, achieving a 69.4\% success rate on the MATH benchmark, a 36\% relative improvement from the 51\% base model performance. Additionally, the entire process operates without any human intervention, making our method both financially and computationally cost-effective compared to existing methods.
     </details>

346. **MR-GSM8K: A Meta-Reasoning Benchmark for Large Language Model Evaluation** [[pdf]](http://arxiv.org/abs/2312.17080) `ICLR 2025 Submission` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces a novel evaluation paradigm for Large Language Models (LLMs) that compels them to transition from a traditional question-answering role, akin to a student, to a solution-scoring role, akin to a teacher, and develops the MR-GSM8K benchmark.
     </details>


     <details>
          <summary>Abstract</summary>
          In this work, we introduce a novel evaluation paradigm for Large Language Models (LLMs) that compels them to transition from a traditional question-answering role, akin to a student, to a solution-scoring role, akin to a teacher. This paradigm, focusing on "reasoning about reasoning," hence termed meta-reasoning, shifts the emphasis from result-oriented assessments, which often neglect the reasoning process, to a more comprehensive evaluation that effectively distinguishes between the cognitive capabilities of different models. By applying this paradigm in the GSM8K dataset, we have developed the MR-GSM8K benchmark. Our extensive analysis includes several state-of-the-art models from both open-source and commercial domains, uncovering fundamental deficiencies in their training and evaluation methodologies. Notably, while models like Deepseek-v2 and Claude3-Sonnet closely competed with GPT-4 in GSM8K, their performance disparities expanded dramatically in MR-GSM8K, with differences widening to over 20 absolute points, underscoring the significant challenge posed by our meta-reasoning approach.
     </details>

347. **Assessing the Emergent Symbolic Reasoning Abilities of Llama Large Language Models** [[pdf]](http://arxiv.org/abs/2406.06588) `2024-06-05` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is observed that both increasing the scale of the model and fine-tuning it on relevant tasks lead to significant performance gains, which are mostly observed with mathematical formulas of low complexity, which nevertheless often remain challenging even for the largest fine-tuned models.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) achieve impressive performance in a wide range of tasks, even if they are often trained with the only objective of chatting fluently with users. Among other skills, LLMs show emergent abilities in mathematical reasoning benchmarks, which can be elicited with appropriate prompting methods. In this work, we systematically investigate the capabilities and limitations of popular open-source LLMs on different symbolic reasoning tasks. We evaluate three models of the Llama 2 family on two datasets that require solving mathematical formulas of varying degrees of difficulty. We test a generalist LLM (Llama 2 Chat) as well as two fine-tuned versions of Llama 2 (MAmmoTH and MetaMath) specifically designed to tackle mathematical problems. We observe that both increasing the scale of the model and fine-tuning it on relevant tasks lead to significant performance gains. Furthermore, using fine-grained evaluation measures, we find that such performance gains are mostly observed with mathematical formulas of low complexity, which nevertheless often remain challenging even for the largest fine-tuned models.
     </details>

348. **Forward-Backward Reasoning in Large Language Models for Mathematical Verification** [[pdf]](http://arxiv.org/abs/2308.07758) `ACL 2024 Findings` (9 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes FOBAR to combine FOrward and BAckward Reasoning for verification for verification, and shows that FOBAR outperforms Self-Consistency, which uses forward reasoning alone, demonstrating that combining forward and forward reasoning is better.
     </details>


     <details>
          <summary>Abstract</summary>
          Self-Consistency samples diverse reasoning chains with answers and chooses the final answer by majority voting. It is based on forward reasoning and cannot further improve performance by sampling more reasoning chains when saturated. To further boost performance, we introduce backward reasoning to verify candidate answers. Specifically, for mathematical tasks, we mask a number in the question and ask the LLM to answer a backward question created by a simple template, i.e., to predict the masked number when a candidate answer is provided. Instead of using forward or backward reasoning alone, we propose **FOBAR** to combine **FO**rward and **BA**ckward **R**easoning for verification. Extensive experiments on six standard mathematical data sets and three LLMs show that FOBAR achieves state-of-the-art performance. In particular, FOBAR outperforms Self-Consistency, which uses forward reasoning alone, demonstrating that combining forward and backward reasoning is more accurate in verification. In addition, FOBAR achieves higher accuracy than existing verification methods, showing the effectiveness of the simple template used in backward reasoning and the proposed combination.
     </details>

349. **Break the Chain: Large Language Models Can be Shortcut Reasoners** [[pdf]](http://arxiv.org/abs/2406.06580) `2024-06-04` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A critical evaluation of CoT prompting is conducted, extending beyond arithmetic to include complex logical and commonsense reasoning tasks, areas where standard CoT methods fall short.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent advancements in Chain-of-Thought (CoT) reasoning utilize complex modules but are hampered by high token consumption, limited applicability, and challenges in reproducibility. This paper conducts a critical evaluation of CoT prompting, extending beyond arithmetic to include complex logical and commonsense reasoning tasks, areas where standard CoT methods fall short. We propose the integration of human-like heuristics and shortcuts into language models (LMs) through "break the chain" strategies. These strategies disrupt traditional CoT processes using controlled variables to assess their efficacy. Additionally, we develop innovative zero-shot prompting strategies that encourage the use of shortcuts, enabling LMs to quickly exploit reasoning clues and bypass detailed procedural steps. Our comprehensive experiments across various LMs, both commercial and open-source, reveal that LMs maintain effective performance with "break the chain" strategies. We also introduce ShortcutQA, a dataset specifically designed to evaluate reasoning through shortcuts, compiled from competitive tests optimized for heuristic reasoning tasks such as forward/backward reasoning and simplification. Our analysis confirms that ShortcutQA not only poses a robust challenge to LMs but also serves as an essential benchmark for enhancing reasoning efficiency in AI.
     </details>

350. **Exploring Mathematical Extrapolation of Large Language Models with Synthetic Data** [[pdf]](http://arxiv.org/abs/2406.02100) `ACL 2024 Findings` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper has designed two out-of-domain datasets in the form of extending the numerical range and the composing components of the arithmetical puzzle problem separately to show that the open-llama-3B model can perform well on multi-step reasoning tasks via fine-tuning on high-quality synthetic data.
     </details>


     <details>
          <summary>Abstract</summary>
          While large language models (LLMs) have shown excellent capabilities in language understanding, text generation and many other tasks, they still struggle in complex multi-step reasoning problems such as mathematical reasoning. In this paper, through a newly proposed arithmetical puzzle problem, we show that the model can perform well on multi-step reasoning tasks via fine tuning on high-quality synthetic data. Experiments with the open-llama-3B model on three different test datasets show that not only the model can reach a zero-shot pass@1 at 0.44 on the in-domain dataset, it also demonstrates certain generalization capabilities on the out-of-domain datasets. Specifically, this paper has designed two out-of-domain datasets in the form of extending the numerical range and the composing components of the arithmetical puzzle problem separately. The fine-tuned model have shown encouraging performance on these two far more difficult tasks with the zero-shot pass@1 at 0.33 and 0.35 correspondingly.
     </details>

351. **Process-Driven Autoformalization in Lean 4** [[pdf]](http://arxiv.org/abs/2406.01940) `ICLR 2025 Submission` `Lean` (4 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A new benchmark for autoformalization for large language models designed to evaluate the autoformalization capabilities of large language models (LLMs), and introduces a model that leverages the precise feedback from Lean 4 compilers to enhance autoformalization.
     </details>


     <details>
          <summary>Abstract</summary>
          Autoformalization, the conversion of natural language mathematics into formal languages, offers significant potential for advancing mathematical reasoning. However, existing efforts are limited to formal languages with substantial online corpora and struggle to keep pace with rapidly evolving languages like Lean 4. To bridge this gap, we propose a new benchmark \textbf{Form}alization for \textbf{L}ean~\textbf{4} (\textbf{\name}) designed to evaluate the autoformalization capabilities of large language models (LLMs). This benchmark encompasses a comprehensive assessment of questions, answers, formal statements, and proofs. Additionally, we introduce a \textbf{P}rocess-\textbf{S}upervised \textbf{V}erifier (\textbf{PSV}) model that leverages the precise feedback from Lean 4 compilers to enhance autoformalization. Our experiments demonstrate that the PSV method improves autoformalization, enabling higher accuracy using less filtered training data. Furthermore, when fine-tuned with data containing detailed process information, PSV can leverage the data more effectively, leading to more significant improvements in autoformalization for Lean 4. Our dataset and code are available at \url{https://github.com/rookie-joe/PDA}.
     </details>

352. **Explicitly Encoding Structural Symmetry is Key to Length Generalization in Arithmetic Tasks** [[pdf]](http://arxiv.org/abs/2406.01895) `2024-06-03` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is proved that explicit incorporation of structure via positional encodings is necessary for out-of-distribution generalization, and pinpoint other challenges inherent to length generalization beyond capturing symmetries, in particular complexity of the underlying task.
     </details>


     <details>
          <summary>Abstract</summary>
          Despite the success of Transformers on language understanding, code generation, and logical reasoning, they still fail to generalize over length on basic arithmetic tasks such as addition and multiplication. A major reason behind this failure is the vast difference in structure between numbers and text; For example, the numbers are typically parsed from right to left, and there is a correspondence between digits at the same position across different numbers. In contrast, for text, such symmetries are quite unnatural. In this work, we propose to encode these semantics explicitly into the model via modified number formatting and custom positional encodings. Empirically, our method allows a Transformer trained on numbers with at most 5-digits for addition and multiplication to generalize up to 50-digit numbers, without using additional data for longer sequences. We further demonstrate that traditional absolute positional encodings (APE) fail to generalize to longer sequences, even when trained with augmented data that captures task symmetries. To elucidate the importance of explicitly encoding structure, we prove that explicit incorporation of structure via positional encodings is necessary for out-of-distribution generalization. Finally, we pinpoint other challenges inherent to length generalization beyond capturing symmetries, in particular complexity of the underlying task, and propose changes in the training distribution to address them.
     </details>

353. **Evaluating Mathematical Reasoning of Large Language Models: A Focus on Error Identification and Correction** [[pdf]](http://arxiv.org/abs/2406.00755) `ACL 2024 Findings` (5 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The principal findings indicate that GPT-4 outperforms all models, while open-source model LLaMA-2-7B demonstrates comparable abilities to closed-source models GPT-3.5 and Gemini Pro.
     </details>


     <details>
          <summary>Abstract</summary>
          The rapid advancement of Large Language Models (LLMs) in the realm of mathematical reasoning necessitates comprehensive evaluations to gauge progress and inspire future directions. Existing assessments predominantly focus on problem-solving from the examinee perspective, overlooking a dual perspective of examiner regarding error identification and correction.From the examiner perspective, we define four evaluation tasks for error identification and correction along with a new dataset with annotated error types and steps. We also design diverse prompts to thoroughly evaluate eleven representative LLMs. Our principal findings indicate that GPT-4 outperforms all models, while open-source model LLaMA-2-7B demonstrates comparable abilities to closed-source models GPT-3.5 and Gemini Pro.Notably, calculation error proves the most challenging error type. Moreover, prompting LLMs with the error types can improve the average correction accuracy by 47.9%. These results reveal potential directions for developing the mathematical reasoning abilities of LLMs.Our code and dataset is available on https://github.com/LittleCirc1e/EIC.
     </details>

354. **What Makes Math Word Problems Challenging for LLMs?** [[pdf]](https://aclanthology.org/2024.findings-naacl.72) `NAACL 2024 Findings` (4 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          This paper investigates the question of what makes math word problems (MWPs) in English challenging for large language models (LLMs). We conduct an in-depth analysis of the key linguistic and mathematical characteristics of MWPs. In addition, we train feature-based classifiers to better understand the impact of each feature on the overall difficulty of MWPs for prominent LLMs and investigate whether this helps predict how well LLMs fare against specific categories of MWPs.
     </details>

355. **Arithmetic Reasoning with LLM: Prolog Generation & Permutation** [[pdf]](https://aclanthology.org/2024.naacl-short.61) `NAACL 2024 Short Papers` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work investigates using large language models to generate Prolog programs to solve mathematical questions and proposes to permute the ground truth predicates for more robust LLM training via data augmentation.
     </details>


     <details>
          <summary>Abstract</summary>
          Instructing large language models (LLMs) to solve elementary school math problems has shown great success using Chain of Thought (CoT). However, the CoT approach relies on an LLM to generate a sequence of arithmetic calculations which can be prone to cascaded calculation errors. We hypothesize that an LLM should focus on extracting predicates and generating symbolic formulas from the math problem description so that the underlying calculation can be done via an external code interpreter. We investigate using LLM to generate Prolog programs to solve mathematical questions. Experimental results show that our Prolog-based arithmetic problem-solving outperforms CoT generation in the GSM8K benchmark across three distinct LLMs. In addition, given the insensitive ordering of predicates and symbolic formulas in Prolog, we propose to permute the ground truth predicates for more robust LLM training via data augmentation.
     </details>

356. **Laying Anchors: Semantically Priming Numerals in Language Modeling** [[pdf]](https://aclanthology.org/2024.findings-naacl.169) `NAACL 2024 Findings` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces strategies to semantically prime numerals in any corpus by generating anchors governed by the distribution of numerals in said corpus, thereby enabling mathematically grounded representations of these numeral tokens in any corpus.
     </details>


     <details>
          <summary>Abstract</summary>
          Off-the-shelf pre-trained language models have become the de facto standard in NLP pipelines for a multitude of downstream tasks. However, the inability of these models to properly encode numerals limits their performance on tasks requiring numeric comprehension. We introduce strategies to semantically prime numerals in any corpus by generating anchors governed by the distribution of numerals in said corpus, thereby enabling mathematically grounded representations of these numeral tokens. We establish the superiority of our proposed techniques through evaluation on a range of numeracy tasks for both in-domain (seen) and out-domain (unseen) numerals. Further, we expand our empirical evaluations to numerals ranging from 1 to 10 billion, a significantly broader range compared to previous studies of the same nature, and we demonstrate significant improvements in the mathematical grounding of our learned embeddings.
     </details>

357. **An Evaluation Benchmark for Autoformalization in Lean4** [[pdf]](http://arxiv.org/abs/2406.06555) `2024-06-01` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel evaluation benchmark designed for Lean4 is introduced, applying it to test the abilities of state-of-the-art LLMs, including GPT-3.5, GPT-4, and Gemini Pro, revealing that these LLMs still exhibit limitations in autoformalization, particularly in more complex areas of mathematics.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) hold the potential to revolutionize autoformalization. The introduction of Lean4, a mathematical programming language, presents an unprecedented opportunity to rigorously assess the autoformalization capabilities of LLMs. This paper introduces a novel evaluation benchmark designed for Lean4, applying it to test the abilities of state-of-the-art LLMs, including GPT-3.5, GPT-4, and Gemini Pro. Our comprehensive analysis reveals that, despite recent advancements, these LLMs still exhibit limitations in autoformalization, particularly in more complex areas of mathematics. These findings underscore the need for further development in LLMs to fully harness their potential in scientific research and development. This study not only benchmarks current LLM capabilities but also sets the stage for future enhancements in autoformalization.
     </details>

358. **GOLD: Geometry Problem Solver with Natural Language Description** [[pdf]](https://aclanthology.org/2024.findings-naacl.19) `NAACL 2024 Findings` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          GOLD enhances the extraction of geometric relations by separately processing symbols and geometric primitives within the diagram, and converts the extracted relations into natural language descriptions, efficiently utilizing large language models to solve geometry math problems.
     </details>


     <details>
          <summary>Abstract</summary>
          Addressing the challenge of automated geometry math problem-solving in artificial intelligence (AI) involves understanding multi-modal information and mathematics. blackCurrent methods struggle with accurately interpreting geometry diagrams, which hinders effective problem-solving. To tackle this issue, we present the Geometry problem sOlver with natural Language Description (GOLD) model. GOLD enhances the extraction of geometric relations by separately processing symbols and geometric primitives within the diagram. Subsequently, it converts the extracted relations into natural language descriptions, efficiently utilizing large language models to solve geometry math problems. Experiments show that the GOLD model outperforms the Geoformer model, the previous best method on the UniGeo dataset, by achieving accuracy improvements of 12.7% and 42.1% in calculation and proving subsets. Additionally, it surpasses the former best model on the PGPS9K and Geometry3K datasets, PGPSNet, by obtaining accuracy enhancements of 1.8% and 3.2%, respectively.
     </details>

359. **Self-Demos: Eliciting Out-of-Demonstration Generalizability in Large Language Models** [[pdf]](https://aclanthology.org/2024.findings-naacl.243) `NAACL 2024 Findings` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes Self-Demos, a novel prompting method that elicits the inherent generalizability in LLMs by query-aware demo generation that can outperform state-of-the-art baselines in the OOD setting.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have shown promising abilities of in-context learning (ICL), adapting swiftly to new tasks with only few-shot demonstrations. However, current few-shot methods heavily depend on high-quality, query-specific demos, which are often lacking. When faced with out-of-demonstration (OOD) queries, methods that rely on hand-crafted demos or external retrievers might fail. To bridge the gap between limited demos and OOD queries, we propose Self-Demos, a novel prompting method that elicits the inherent generalizability in LLMs by query-aware demo generation. The generated demos strategically interpolate between existing demos and the given query, transforming the query from OOD to ID. To evaluate the effectiveness of our approach, we manually constructed OOD-Toolset, a dataset in the tool-using scenario with over 300 real-world APIs and 1000 instances, each consisting of three tool-use cases as demos and an OOD query. Thorough experiments on our dataset and two public math benchmarks have shown that our method can outperform state-of-the-art baselines in the OOD setting. Moreover, we conduct a range of analyses to validate Self-Demos’s generalization and provide more insights.
     </details>

360. **Beyond Imitation: Learning Key Reasoning Steps from Dual Chain-of-Thoughts in Reasoning Distillation** [[pdf]](http://arxiv.org/abs/2405.19737) `ICLR 2025 Submission` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel method that further aids SLMs learning key reasoning steps rather than mere simple fine-tuning is proposed, and it is found that EDIT benefits more from logical errors than from knowledge or mathematical calculation errors in dual CoTs.
     </details>


     <details>
          <summary>Abstract</summary>
          As Large Language Models (LLMs) scale up and gain powerful Chain-of-Thoughts (CoTs) reasoning abilities, practical resource constraints drive efforts to distill these capabilities into more compact Smaller Language Models (SLMs). We find that CoTs consist mainly of simple reasoning forms, with a small proportion ($\approx 4.7\%$) of key reasoning steps that truly impact conclusions. However, previous distillation methods typically involve supervised fine-tuning student SLMs only on correct CoTs data produced by teacher LLMs, resulting in students struggling to learn the key reasoning steps, instead imitating the teacher's reasoning forms and making errors or omissions on these steps. To address these issues, drawing an analogy to human learning, where analyzing mistakes according to correct solutions often reveals the crucial steps leading to successes or failures, we propose mistak\textbf{E}-\textbf{D}riven key reason\textbf{I}ng step distilla\textbf{T}ion (\textbf{EDIT}), a novel method that further aids SLMs learning key reasoning steps rather than mere simple fine-tuning. Firstly, to expose these crucial steps in CoTs, we design specific prompts to generate dual CoTs data with similar reasoning paths but divergent conclusions. Then, we apply the minimum edit distance algorithm on the dual CoTs data to locate these key steps and optimize the likelihood of these steps. Extensive experiments validate the effectiveness of EDIT across both in-domain and out-of-domain benchmark reasoning datasets. Further analysis shows that EDIT can generate high-quality CoTs with more correct key reasoning steps. Notably, we also explore how different mistake patterns affect performance and find that EDIT benefits more from logical errors than from knowledge or mathematical calculation errors in dual CoTs\footnote{Code can be found at \url{https://github.com/C-W-D/EDIT}}.
     </details>

361. **Investigating the Robustness of LLMs on Math Word Problems** [[pdf]](http://arxiv.org/abs/2406.15444) `2024-05-30` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A prompting framework that generates adversarial variants of MWPs by adding irrelevant variables is proposed, and fine-tuning on adversarial training instances improves performance on adversarial MWPs by ~8%, indicating increased robustness to noise and better ability to identify relevant data for reasoning.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) excel at various tasks, including solving math word problems (MWPs), but struggle with real-world problems containing irrelevant information. To address this, we propose a prompting framework that generates adversarial variants of MWPs by adding irrelevant variables. We introduce a dataset, ProbleMATHIC, containing both adversarial and non-adversarial MWPs. Our experiments reveal that LLMs are susceptible to distraction by numerical noise, resulting in an average relative performance drop of ~26% on adversarial MWPs. To mitigate this, we fine-tune LLMs (Llama-2, Mistral) on the adversarial samples from our dataset. Fine-tuning on adversarial training instances improves performance on adversarial MWPs by ~8%, indicating increased robustness to noise and better ability to identify relevant data for reasoning. Finally, to assess the generalizability of our prompting framework, we introduce GSM-8K-Adv, an adversarial variant of the GSM-8K benchmark. LLMs continue to struggle when faced with adversarial information, reducing performance by up to ~6%.
     </details>

362. **MathChat: Benchmarking Mathematical Reasoning and Instruction Following in Multi-Turn Interactions** [[pdf]](http://arxiv.org/abs/2405.19444) `2024-05-29` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduces MathChat, a comprehensive benchmark specifically designed to evaluate LLMs across a broader spectrum of mathematical tasks, and develops MathChat sync, a synthetic dialogue based math dataset for LLM finetuning, focusing on improving models' interaction and instruction following capabilities in conversations.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have demonstrated impressive capabilities in mathematical problem solving, particularly in single turn question answering formats. However, real world scenarios often involve mathematical question answering that requires multi turn or interactive information exchanges, and the performance of LLMs on these tasks is still underexplored. This paper introduces MathChat, a comprehensive benchmark specifically designed to evaluate LLMs across a broader spectrum of mathematical tasks. These tasks are structured to assess the models' abilities in multiturn interactions and open ended generation. We evaluate the performance of various SOTA LLMs on the MathChat benchmark, and we observe that while these models excel in single turn question answering, they significantly underperform in more complex scenarios that require sustained reasoning and dialogue understanding. To address the above limitations of existing LLMs when faced with multiturn and open ended tasks, we develop MathChat sync, a synthetic dialogue based math dataset for LLM finetuning, focusing on improving models' interaction and instruction following capabilities in conversations. Experimental results emphasize the need for training LLMs with diverse, conversational instruction tuning datasets like MathChatsync. We believe this work outlines one promising direction for improving the multiturn mathematical reasoning abilities of LLMs, thus pushing forward the development of LLMs that are more adept at interactive mathematical problem solving and real world applications.
     </details>

363. **Achieving >97% on GSM8K: Deeply Understanding the Problems Makes LLMs Better Solvers for Math Word Problems** [[pdf]](http://arxiv.org/abs/2404.14963) `2024-05-29` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Chain-of-Thought (CoT) prompting has enhanced the performance of Large Language Models (LLMs) across various reasoning tasks. However, CoT still falls short in dealing with complex math word problems, as it usually suffers from three pitfalls: semantic misunderstanding errors, calculation errors and step-missing errors. Prior studies involve addressing the calculation errors and step-missing errors, but neglect the semantic misunderstanding errors, which is the major factor limiting the LLMs' performance. To this end, we propose a simple-yet-effective method, namely Deeply Understanding the Problems (DUP), to improve the LLMs' math problem-solving ability by addressing semantic misunderstanding errors. The core of our method is to encourage the LLMs to deeply understand the problems and extract the key problem-solving information used for better reasoning. Extensive experiments on 10 diverse reasoning benchmarks show that our DUP method consistently outperforms the other counterparts by a large margin. More encouragingly, DUP achieves a new SOTA result on the GSM8K benchmark, with an accuracy of 97.1% under zero-shot setting.
     </details>

364. **Efficient Model-agnostic Alignment via Bayesian Persuasion** [[pdf]](http://arxiv.org/abs/2405.18718) `2024-05-28` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          An efficient method for aligning black-box large models using smaller models using smaller models is explored, introducing a model-agnostic and lightweight Bayesian Persuasion Alignment framework and an upper bound on the Advisor's regret is provided, confirming its effectiveness in learning the optimal signaling strategy.
     </details>


     <details>
          <summary>Abstract</summary>
          With recent advancements in large language models (LLMs), alignment has emerged as an effective technique for keeping LLMs consensus with human intent. Current methods primarily involve direct training through Supervised Fine-tuning (SFT) or Reinforcement Learning from Human Feedback (RLHF), both of which require substantial computational resources and extensive ground truth data. This paper explores an efficient method for aligning black-box large models using smaller models, introducing a model-agnostic and lightweight Bayesian Persuasion Alignment framework. We formalize this problem as an optimization of the signaling strategy from the small model's perspective. In the persuasion process, the small model (Advisor) observes the information item (i.e., state) and persuades large models (Receiver) to elicit improved responses. The Receiver then generates a response based on the input, the signal from the Advisor, and its updated belief about the information item. Through training using our framework, we demonstrate that the Advisor can significantly enhance the performance of various Receivers across a range of tasks. We theoretically analyze our persuasion framework and provide an upper bound on the Advisor's regret, confirming its effectiveness in learning the optimal signaling strategy. Our Empirical results demonstrates that GPT-2 can significantly improve the performance of various models, achieving an average enhancement of 16.1% in mathematical reasoning ability and 13.7% in code generation. We hope our work can provide an initial step toward rethinking the alignment framework from the Bayesian Persuasion perspective.
     </details>

365. **Learning Beyond Pattern Matching? Assaying Mathematical Understanding in LLMs** [[pdf]](http://arxiv.org/abs/2405.15485) `2024-05-24` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper assesses the domain knowledge of LLMs through its understanding of different mathematical skills required to solve problems, and finds evidence of domain understanding during in-context learning.
     </details>


     <details>
          <summary>Abstract</summary>
          We are beginning to see progress in language model assisted scientific discovery. Motivated by the use of LLMs as a general scientific assistant, this paper assesses the domain knowledge of LLMs through its understanding of different mathematical skills required to solve problems. In particular, we look at not just what the pre-trained model already knows, but how it learned to learn from information during in-context learning or instruction-tuning through exploiting the complex knowledge structure within mathematics. Motivated by the Neural Tangent Kernel (NTK), we propose \textit{NTKEval} to assess changes in LLM's probability distribution via training on different kinds of math data. Our systematic analysis finds evidence of domain understanding during in-context learning. By contrast, certain instruction-tuning leads to similar performance changes irrespective of training on different data, suggesting a lack of domain understanding across different skills.
     </details>

366. **Rho-1: Not All Tokens Are What You Need** [[pdf]](http://arxiv.org/abs/2404.07965) `2024-05-23` (24 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Rho-1 is introduced, a new language model that employs Selective Language Modeling (SLM), which selectively trains on useful tokens that aligned with the desired distribution, increasing both efficiency and performance of the language model pre-training.
     </details>


     <details>
          <summary>Abstract</summary>
          Previous language model pre-training methods have uniformly applied a next-token prediction loss to all training tokens. Challenging this norm, we posit that ''Not all tokens in a corpus are equally important for language model training''. Our initial analysis examines token-level training dynamics of language model, revealing distinct loss patterns for different tokens. Leveraging these insights, we introduce a new language model called Rho-1. Unlike traditional LMs that learn to predict every next token in a corpus, Rho-1 employs Selective Language Modeling (SLM), which selectively trains on useful tokens that aligned with the desired distribution. This approach involves scoring pretraining tokens using a reference model, and then training the language model with a focused loss on tokens with higher scores. When continual pretraining on 15B OpenWebMath corpus, Rho-1 yields an absolute improvement in few-shot accuracy of up to 30% in 9 math tasks. After fine-tuning, Rho-1-1B and 7B achieved state-of-the-art results of 40.6% and 51.8% on MATH dataset, respectively - matching DeepSeekMath with only 3% of the pretraining tokens. Furthermore, when pretraining on 80B general tokens, Rho-1 achieves 6.8% average enhancement across 15 diverse tasks, increasing both efficiency and performance of the language model pre-training.
     </details>

367. **DeepSeek-Prover: Advancing Theorem Proving in LLMs through Large-Scale Synthetic Data** [[pdf]](http://arxiv.org/abs/2405.14333) `2024-05-23` `Lean` (15 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces an approach to generate extensive Lean 4 proof data derived from high-school and undergraduate-level mathematical competition problems to enhance theorem-proving capabilities in LLMs and demonstrates the potential of leveraging large-scale synthetic data to enhance theorem-proving capabilities in LLMs.
     </details>


     <details>
          <summary>Abstract</summary>
          Proof assistants like Lean have revolutionized mathematical proof verification, ensuring high accuracy and reliability. Although large language models (LLMs) show promise in mathematical reasoning, their advancement in formal theorem proving is hindered by a lack of training data. To address this issue, we introduce an approach to generate extensive Lean 4 proof data derived from high-school and undergraduate-level mathematical competition problems. This approach involves translating natural language problems into formal statements, filtering out low-quality statements, and generating proofs to create synthetic data. After fine-tuning the DeepSeekMath 7B model on this synthetic dataset, which comprises 8 million formal statements with proofs, our model achieved whole-proof generation accuracies of 46.3% with 64 samples and 52% cumulatively on the Lean 4 miniF2F test, surpassing the baseline GPT-4 at 23.0% with 64 samples and a tree search reinforcement learning method at 41.0%. Additionally, our model successfully proved 5 out of 148 problems in the Lean 4 Formalized International Mathematical Olympiad (FIMO) benchmark, while GPT-4 failed to prove any. These results demonstrate the potential of leveraging large-scale synthetic data to enhance theorem-proving capabilities in LLMs. Both the synthetic dataset and the model will be made available to facilitate further research in this promising field.
     </details>

368. **Can LLMs Solve longer Math Word Problems Better?** [[pdf]](http://arxiv.org/abs/2405.14804) `2024-05-23` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This study pioneers the exploration of Context Length Generalizability (CoLeG), the ability of LLMs to solve long MWPs, and introduces Extended Grade-School Math (E-GSM), a collection of MWPs with lengthy narratives.
     </details>


     <details>
          <summary>Abstract</summary>
          Math Word Problems (MWPs) are crucial for evaluating the capability of Large Language Models (LLMs), with current research primarily focusing on questions with concise contexts. However, as real-world math problems often involve complex circumstances, LLMs' ability to solve long MWPs is vital for their applications in these scenarios, yet remains under-explored. This study pioneers the exploration of Context Length Generalizability (CoLeG), the ability of LLMs to solve long MWPs. We introduce Extended Grade-School Math (E-GSM), a collection of MWPs with lengthy narratives. Two novel metrics are proposed to assess the efficacy and resilience of LLMs in solving these problems. Our examination of existing zero-shot prompting techniques and both proprietary and open-source LLMs reveals a general deficiency in CoLeG. To alleviate these challenges, we propose distinct approaches for different categories of LLMs. For proprietary LLMs, a new instructional prompt is proposed to mitigate the influence of long context. For open-source LLMs, a new data augmentation task is developed to improve CoLeG. Our comprehensive results demonstrate the effectiveness of our proposed methods, showing not only improved performance on E-GSM but also generalizability across several other MWP benchmarks. Our findings pave the way for future research in employing LLMs for complex, real-world applications, offering practical solutions to current limitations and opening avenues for further exploration of model generalizability and training methodologies.
     </details>

369. **Large Language Models are Contrastive Reasoners** [[pdf]](http://arxiv.org/abs/2403.08211) `2024-05-22` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work explores how contrastive prompting (CP) significantly improves the ability of large language models to perform complex reasoning and demonstrates that LLMs are decent contrastive reasoners by simply adding "Let's give a correct and a wrong answer" before LLMs provide answers.
     </details>


     <details>
          <summary>Abstract</summary>
          Prompting methods play a crucial role in enhancing the capabilities of pre-trained large language models (LLMs). We explore how contrastive prompting (CP) significantly improves the ability of large language models to perform complex reasoning. We demonstrate that LLMs are decent contrastive reasoners by simply adding "Let's give a correct and a wrong answer." before LLMs provide answers. Experiments on various large language models show that zero-shot contrastive prompting improves performance on a range of arithmetic, commonsense, and symbolic reasoning tasks without any hand-crafted few-shot examples, such as increasing the accuracy on GSM8K from 35.9% to 88.8% and AQUA-RAT from 41.3% to 62.2% with the state-of-the-art GPT-4 model. Our method not only surpasses zero-shot CoT and few-shot CoT in most arithmetic and commonsense reasoning tasks but also can seamlessly integrate with existing prompting methods, resulting in improved or comparable results when compared to state-of-the-art methods. Our code is available at https://github.com/yao8839836/cp
     </details>

370. **Investigating Symbolic Capabilities of Large Language Models** [[pdf]](http://arxiv.org/abs/2405.13209) `2024-05-21` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Evaluating Large Language Models on a series of symbolic tasks reveals a significant decline in LLMs' performance on context-free and context-sensitive symbolic tasks as the complexity, represented by the number of symbols, increases.
     </details>


     <details>
          <summary>Abstract</summary>
          Prompting techniques have significantly enhanced the capabilities of Large Language Models (LLMs) across various complex tasks, including reasoning, planning, and solving math word problems. However, most research has predominantly focused on language-based reasoning and word problems, often overlooking the potential of LLMs in handling symbol-based calculations and reasoning. This study aims to bridge this gap by rigorously evaluating LLMs on a series of symbolic tasks, such as addition, multiplication, modulus arithmetic, numerical precision, and symbolic counting. Our analysis encompasses eight LLMs, including four enterprise-grade and four open-source models, of which three have been pre-trained on mathematical tasks. The assessment framework is anchored in Chomsky's Hierarchy, providing a robust measure of the computational abilities of these models. The evaluation employs minimally explained prompts alongside the zero-shot Chain of Thoughts technique, allowing models to navigate the solution process autonomously. The findings reveal a significant decline in LLMs' performance on context-free and context-sensitive symbolic tasks as the complexity, represented by the number of symbols, increases. Notably, even the fine-tuned GPT3.5 exhibits only marginal improvements, mirroring the performance trends observed in other models. Across the board, all models demonstrated a limited generalization ability on these symbol-intensive tasks. This research underscores LLMs' challenges with increasing symbolic complexity and highlights the need for specialized training, memory and architectural adjustments to enhance their proficiency in symbol-based reasoning tasks.
     </details>

371. **DOP: Diagnostic-Oriented Prompting for Large Language Models in Mathematical Correction** [[pdf]](http://arxiv.org/abs/2405.12100) `2024-05-20` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Math world problems correction(MWPC) is a novel task dedicated to rectifying reasoning errors in the process of solving mathematical problems. In this paper, leveraging the advancements in large language models (LLMs), we address two key objectives:(1) Distinguishing between mathematical reasoning and error correction; (2) Exploring strategies to enhance the error correction capabilities of LLMs in mathematics to solve MWPC task. We noticed that, in real-time education,assisting students in recognizing their mistakes is more crucial than simply providing correct answers. However, current research tends to prioritize obtaining accurate solutions to math problems rather than correcting potentially incorrect ones. Therefore, we modify the research paradigm, demonstrating that improving mathematical reasoning abilities does not equate to mastery in error correction. Meanwhile, we propose a novel method called diagnostic-oriented promping(DOP) aimed at facilitating LLMs to excel in error correction. In experiments, DOP has shown outstanding performance, highlighting its significant impact. We argue that in mathematical education, the demand for outstanding correctors surpasses that for proficient reasoners. Codes and data are available on https://github.com/ChenhaoEcnuCS/Reason-Correct.
     </details>

372. **Self-Explore to Avoid the Pit: Improving the Reasoning Capabilities of Language Models with Fine-grained Rewards** [[pdf]](http://arxiv.org/abs/2404.10346) `2024-05-16` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Training on large amounts of rationales (i.e., CoT Fine-tuning) is effective at improving the reasoning capabilities of large language models (LLMs). However, acquiring human-authored rationales or augmenting rationales from proprietary models is costly and not scalable. In this paper, we study the problem of whether LLMs could self-improve their reasoning capabilities. To this end, we propose Self-Explore, where the LLM is tasked to explore the first wrong step (i.e., the first pit) within the rationale and use such signals as fine-grained rewards for further improvement. On the GSM8K and MATH test set, Self-Explore achieves 11.57% and 2.89% improvement on average across three LLMs compared to supervised fine-tuning (SFT). Our code is available at https://github.com/hbin0701/Self-Explore.
     </details>

373. **Improving Automated Distractor Generation for Math Multiple-choice Questions with Overgenerate-and-rank** [[pdf]](http://arxiv.org/abs/2405.05144) `2024-05-13` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel method to enhance the quality of generated distractors through overgenerate-and-rank, training a ranking model to predict how likely distractors are to be selected by real students is proposed.
     </details>


     <details>
          <summary>Abstract</summary>
          Multiple-choice questions (MCQs) are commonly used across all levels of math education since they can be deployed and graded at a large scale. A critical component of MCQs is the distractors, i.e., incorrect answers crafted to reflect student errors or misconceptions. Automatically generating them in math MCQs, e.g., with large language models, has been challenging. In this work, we propose a novel method to enhance the quality of generated distractors through overgenerate-and-rank, training a ranking model to predict how likely distractors are to be selected by real students. Experimental results on a real-world dataset and human evaluation with math teachers show that our ranking model increases alignment with human-authored distractors, although human-authored ones are still preferred over generated ones.
     </details>

374. **MuMath-Code: Combining Tool-Use Large Language Models with Multi-perspective Data Augmentation for Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2405.07551) `NAACL 2024 Findings` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work firstly includes new math questions via multi-perspective data augmenting methods and then synthesize code-nested solutions to them and proposes a two-stage training strategy, which leverages the advantages of both the external tool and data augmentation.
     </details>


     <details>
          <summary>Abstract</summary>
          The tool-use Large Language Models (LLMs) that integrate with external Python interpreters have significantly enhanced mathematical reasoning capabilities for open-source LLMs, while tool-free methods chose another track: augmenting math reasoning data. However, a great method to integrate the above two research paths and combine their advantages remains to be explored. In this work, we firstly include new math questions via multi-perspective data augmenting methods and then synthesize code-nested solutions to them. The open LLMs (i.e., Llama-2) are finetuned on the augmented dataset to get the resulting models, MuMath-Code ($\mu$-Math-Code). During the inference phase, our MuMath-Code generates code and interacts with the external python interpreter to get the execution results. Therefore, MuMath-Code leverages the advantages of both the external tool and data augmentation. To fully leverage the advantages of our augmented data, we propose a two-stage training strategy: In Stage-1, we finetune Llama-2 on pure CoT data to get an intermediate model, which then is trained on the code-nested data in Stage-2 to get the resulting MuMath-Code. Our MuMath-Code-7B achieves 83.8 on GSM8K and 52.4 on MATH, while MuMath-Code-70B model achieves new state-of-the-art performance among open methods -- achieving 90.7% on GSM8K and 55.1% on MATH. Extensive experiments validate the combination of tool use and data augmentation, as well as our two-stage training strategy. We release the proposed dataset along with the associated code for public use.
     </details>

375. **MathDivide: Improved mathematical reasoning by large language models** [[pdf]](http://arxiv.org/abs/2405.13004) `2024-05-12` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A prompting technique called MathDivide is proposed that breaks down the mathematical problem into simpler subproblems and demonstrates that MathDivide was able to significantly outperform the leading prompting technique called Math-prompter.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models have been proven to be capable of handling complex linguistic and cognitive tasks. Therefore their usage has been extended to tasks requiring logical reasoning ability such as Mathematics. In this paper, we propose a prompting technique called MathDivide that breaks down the mathematical problem into simpler subproblems. Each of the subproblems is formulated as an algebraic expression whose value is evaluated by the Python code generated by the LLM for the corresponding algebraic expression. The values fed to the Python code are the numerical values provided in the problem statement. The solutions for the subproblems are composed together to obtain the final answer for the problem statement. Finally, the final answer is compared to the correct answer. If the final answer matches the correct answer, it is produced as output else a refinement prompt is fed to the LLM. We experiment with this prompting technique on both closed-source LLM models and open-source LLM models using GSM8K dataset. The results obtained demonstrate that MathDivide was able to significantly outperform the leading prompting technique called Math-prompter.
     </details>

376. **Natural Language Reasoning, A Survey** [[pdf]](https://dl.acm.org/doi/10.1145/3664194) `2024-05-09` (36 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This survey paper proposes a clearer view of natural language reasoning in the field of Natural Language Processing (NLP), and identifies and views backward reasoning, a powerful paradigm for multi-step reasoning, and introduces defeasible reasoning as one of the most important future directions in natural language reasoning research.
     </details>


     <details>
          <summary>Abstract</summary>
          This survey paper proposes a clearer view of natural language reasoning in the field of Natural Language Processing (NLP), both conceptually and practically. Conceptually, we provide a distinct definition for natural language reasoning in NLP, based on both philosophy and NLP scenarios, discuss what types of tasks require reasoning, and introduce a taxonomy of reasoning. Practically, we conduct a comprehensive literature review on natural language reasoning in NLP, mainly covering classical logical reasoning, natural language inference, multi-hop question answering, and commonsense reasoning. The paper also identifies and views backward reasoning, a powerful paradigm for multi-step reasoning, and introduces defeasible reasoning as one of the most important future directions in natural language reasoning research. We focus on single-modality unstructured natural language text, excluding neuro-symbolic techniques and mathematical reasoning.
     </details>

377. **LLMs Can Find Mathematical Reasoning Mistakes by Pedagogical Chain-of-Thought** [[pdf]](http://arxiv.org/abs/2405.06705) `IJCAI 2024 Knowledge Representation and Reasoning` (1 cite) (1 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Self-correction is emerging as a promising approach to mitigate the issue of hallucination in Large Language Models (LLMs). To facilitate effective self-correction, recent research has proposed mistake detection as its initial step. However, current literature suggests that LLMs often struggle with reliably identifying reasoning mistakes when using simplistic prompting strategies. To address this challenge, we introduce a unique prompting strategy, termed the Pedagogical Chain-of-Thought (PedCoT), which is specifically designed to guide the identification of reasoning mistakes, particularly mathematical reasoning mistakes. PedCoT consists of pedagogical principles for prompts (PPP) design, two-stage interaction process (TIP) and grounded PedCoT prompts, all inspired by the educational theory of the Bloom Cognitive Model (BCM). We evaluate our approach on two public datasets featuring math problems of varying difficulty levels. The experiments demonstrate that our zero-shot prompting strategy significantly outperforms strong baselines. The proposed method can achieve the goal of reliable mathematical mistake identification and provide a foundation for automatic math answer grading. The results underscore the significance of educational theory, serving as domain knowledge, in guiding prompting strategy design for addressing challenging tasks with LLMs effectively.
     </details>

378. **Learning to Solve Geometry Problems via Simulating Human Dual-Reasoning Process** [[pdf]](http://arxiv.org/abs/2405.06232) `IJCAI 2024 Natural Language Processing` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A Dual-Reasoning Geometry Solver (DualGeoSolver) to simulate the dual-reasoning process of humans for GPS and demonstrates the superiority of DualGeoSolver in both solving accuracy and robustness from explicitly modeling human reasoning process and knowledge application.
     </details>


     <details>
          <summary>Abstract</summary>
          Geometry Problem Solving (GPS), which is a classic and challenging math problem, has attracted much attention in recent years. It requires a solver to comprehensively understand both text and diagram, master essential geometry knowledge, and appropriately apply it in reasoning. However, existing works follow a paradigm of neural machine translation and only focus on enhancing the capability of encoders, which neglects the essential characteristics of human geometry reasoning. In this paper, inspired by dual-process theory, we propose a Dual-Reasoning Geometry Solver (DualGeoSolver) to simulate the dual-reasoning process of humans for GPS. Specifically, we construct two systems in DualGeoSolver, namely Knowledge System and Inference System. Knowledge System controls an implicit reasoning process, which is responsible for providing diagram information and geometry knowledge according to a step-wise reasoning goal generated by Inference System. Inference System conducts an explicit reasoning process, which specifies the goal in each reasoning step and applies the knowledge to generate program tokens for resolving it. The two systems carry out the above process iteratively, which behaves more in line with human cognition. We conduct extensive experiments on two benchmark datasets, GeoQA and GeoQA+. The results demonstrate the superiority of DualGeoSolver in both solving accuracy and robustness from explicitly modeling human reasoning process and knowledge application.
     </details>

379. **Key-Point-Driven Data Synthesis with its Enhancement on Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2403.02333) `2024-05-07` (13 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes Key-Point-Driven Data Synthesis (KPDDS), a novel data synthesis framework that synthesizes question-answer pairs by leveraging key points and exemplar practices from authentic data sources to ensure the generation of novel questions with rigorous quality control and substantial scalability.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have shown great potential in complex reasoning tasks, yet their performance is often hampered by the scarcity of high-quality and reasoning-focused training datasets. Addressing this challenge, we propose Key-Point-Driven Data Synthesis (KPDDS), a novel data synthesis framework that synthesizes question-answer pairs by leveraging key points and exemplar practices from authentic data sources. KPDDS ensures the generation of novel questions with rigorous quality control and substantial scalability. As a result, we present KPMath, an extensive synthetic dataset tailored for mathematical reasoning, comprising over 800K question-answer pairs. Utilizing KPMath and augmenting it with additional reasoning-intensive corpora, we create the comprehensive KPMath-Plus dataset. The Qwen1.5-72B model, fine-tuned on KPMath-Plus, achieves 87.0% PASS@1 accuracy on GSM8K and 58.3% on MATH, surpassing competitors in the 7B to 70B range and best commercial models like GPT-4 across multiple math reasoning datasets.
     </details>

380. **ATG: Benchmarking Automated Theorem Generation for Generative Language Models** [[pdf]](http://arxiv.org/abs/2405.06677) `NAACL 2024 Findings` `MetaMath` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          An Automated Theorem Generation (ATG) benchmark that evaluates whether an agent can automatically generate valuable (and possibly brand new) theorems that are applicable for downstream theorem proving as reusable knowledge is proposed.
     </details>


     <details>
          <summary>Abstract</summary>
          Humans can develop new theorems to explore broader and more complex mathematical results.While current generative language models (LMs) have achieved significant improvement in automatically proving theorems, their ability to generate new or reusable theorems is still under-explored. Without the new theorems, current LMs struggle to prove harder theorems that are distant from the given hypotheses with the exponentially growing search space.More advanced theorem proving is if an agent (for instance, a generative LM) can leverage its creativity to generate new but also reasonable theorems that properly substitute part of a proof and also be saved as reusable knowledge for future theorem proving.Therefore, this paper proposes an Automated Theorem Generation (ATG) benchmark that evaluates whether an agent can automatically generate valuable (and possibly brand new) theorems that are applicable for downstream theorem proving as reusable knowledge. Specifically, we construct the ATG benchmark by splitting the Metamath library into three sets: axioms, library, and problem based on their proving depth.We conduct extensive experiments to investigate whether current LMs can generate theorems in the library and benefit the problem theorems proving. The results demonstrate that high-quality ATG data facilitates models’ performances on downstream ATP. However, there is still room for current LMs to develop better ATG and generate more advanced and human-like theorems. We hope the new ATG challenge can shed some light on advanced complex theorem proving.
     </details>

381. **Monte Carlo Tree Search Boosts Reasoning via Iterative Preference Learning** [[pdf]](https://arxiv.org/abs/2405.00451v2) `2024-05-01` (14 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work leverages Monte Carlo Tree Search (MCTS) to iteratively collect preference data, utilizing its look-ahead ability to break down instance-level rewards into more granular step-level signals, to enhance consistency in intermediate steps.
     </details>


     <details>
          <summary>Abstract</summary>
          We introduce an approach aimed at enhancing the reasoning capabilities of Large Language Models (LLMs) through an iterative preference learning process inspired by the successful strategy employed by AlphaZero. Our work leverages Monte Carlo Tree Search (MCTS) to iteratively collect preference data, utilizing its look-ahead ability to break down instance-level rewards into more granular step-level signals. To enhance consistency in intermediate steps, we combine outcome validation and stepwise self-evaluation, continually updating the quality assessment of newly generated data. The proposed algorithm employs Direct Preference Optimization (DPO) to update the LLM policy using this newly generated step-level preference data. Theoretical analysis reveals the importance of using on-policy sampled data for successful self-improving. Extensive evaluations on various arithmetic and commonsense reasoning tasks demonstrate remarkable performance improvements over existing models. For instance, our approach outperforms the Mistral-7B Supervised Fine-Tuning (SFT) baseline on GSM8K, MATH, and ARC-C, with substantial increases in accuracy to $81.8\%$ (+$5.9\%$), $34.7\%$ (+$5.8\%$), and $76.4\%$ (+$15.8\%$), respectively. Additionally, our research delves into the training and inference compute tradeoff, providing insights into how our method effectively maximizes performance gains. Our code is publicly available at https://github.com/YuxiXie/MCTS-DPO.
     </details>

382. **Self-Refine Instruction-Tuning for Aligning Reasoning in Language Models** [[pdf]](http://arxiv.org/abs/2405.00402) `2024-05-01` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper proposes the Self-refine Instruction-tuning method that elicits Smaller Language Models to self-refine their abilities, and shows that this approach significantly outperforms Instruction-tuning in both in-domain and out-domain scenarios, aligning the reasoning abilities of Smaller and Larger Language Models.
     </details>


     <details>
          <summary>Abstract</summary>
          The alignments of reasoning abilities between smaller and larger Language Models are largely conducted via Supervised Fine-Tuning (SFT) using demonstrations generated from robust Large Language Models (LLMs). Although these approaches deliver more performant models, they do not show sufficiently strong generalization ability as the training only relies on the provided demonstrations.   In this paper, we propose the Self-refine Instruction-tuning method that elicits Smaller Language Models to self-refine their abilities. Our approach is based on a two-stage process, where reasoning abilities are first transferred between LLMs and Small Language Models (SLMs) via Instruction-tuning on demonstrations provided by LLMs, and then the instructed models Self-refine their abilities through preference optimization strategies. In particular, the second phase operates refinement heuristics based on the Direct Preference Optimization algorithm, where the SLMs are elicited to deliver a series of reasoning paths by automatically sampling the generated responses and providing rewards using ground truths from the LLMs. Results obtained on commonsense and math reasoning tasks show that this approach significantly outperforms Instruction-tuning in both in-domain and out-domain scenarios, aligning the reasoning abilities of Smaller and Larger Language Models.
     </details>

383. **General Purpose Verification for Chain of Thought Prompting** [[pdf]](http://arxiv.org/abs/2405.00204) `2024-04-30` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper explores ways to improve reasoning capabilities of LLMs through (1) exploration of different chains of thought and (2) validation of the individual steps of the reasoning process through three general principles that a model should adhere to while reasoning.
     </details>


     <details>
          <summary>Abstract</summary>
          Many of the recent capabilities demonstrated by Large Language Models (LLMs) arise primarily from their ability to exploit contextual information. In this paper, we explore ways to improve reasoning capabilities of LLMs through (1) exploration of different chains of thought and (2) validation of the individual steps of the reasoning process. We propose three general principles that a model should adhere to while reasoning: (i) Relevance, (ii) Mathematical Accuracy, and (iii) Logical Consistency. We apply these constraints to the reasoning steps generated by the LLM to improve the accuracy of the final generation. The constraints are applied in the form of verifiers: the model itself is asked to verify if the generated steps satisfy each constraint. To further steer the generations towards high-quality solutions, we use the perplexity of the reasoning steps as an additional verifier. We evaluate our method on 4 distinct types of reasoning tasks, spanning a total of 9 different datasets. Experiments show that our method is always better than vanilla generation, and, in 6 out of the 9 datasets, it is better than best-of N sampling which samples N reasoning chains and picks the lowest perplexity generation.
     </details>

384. **Can Large Language Models put 2 and 2 together? Probing for Entailed Arithmetical Relationships** [[pdf]](https://arxiv.org/abs/2404.19432v1) `2024-04-30` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is empirical demonstrate that although LLMs make steady progress in knowledge acquisition and (pseudo)reasoning with each new GPT release, their capabilities are limited to statistical inference only, and it is argued that bigger is not always better and chasing purely statistical improvements is flawed at the core.
     </details>


     <details>
          <summary>Abstract</summary>
          Two major areas of interest in the era of Large Language Models regard questions of what do LLMs know, and if and how they may be able to reason (or rather, approximately reason). Since to date these lines of work progressed largely in parallel (with notable exceptions), we are interested in investigating the intersection: probing for reasoning about the implicitly-held knowledge. Suspecting the performance to be lacking in this area, we use a very simple set-up of comparisons between cardinalities associated with elements of various subjects (e.g. the number of legs a bird has versus the number of wheels on a tricycle). We empirically demonstrate that although LLMs make steady progress in knowledge acquisition and (pseudo)reasoning with each new GPT release, their capabilities are limited to statistical inference only. It is difficult to argue that pure statistical learning can cope with the combinatorial explosion inherent in many commonsense reasoning tasks, especially once arithmetical notions are involved. Further, we argue that bigger is not always better and chasing purely statistical improvements is flawed at the core, since it only exacerbates the dangerous conflation of the production of correct answers with genuine reasoning ability.
     </details>

385. **Exploring Internal Numeracy in Language Models: A Case Study on ALBERT** [[pdf]](http://arxiv.org/abs/2404.16574) `2024-04-25` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          These findings illustrate that language models, trained purely to model text, can intuit basic mathematical concepts, opening avenues for NLP applications that intersect with quantitative reasoning.
     </details>


     <details>
          <summary>Abstract</summary>
          It has been found that Transformer-based language models have the ability to perform basic quantitative reasoning. In this paper, we propose a method for studying how these models internally represent numerical data, and use our proposal to analyze the ALBERT family of language models. Specifically, we extract the learned embeddings these models use to represent tokens that correspond to numbers and ordinals, and subject these embeddings to Principal Component Analysis (PCA). PCA results reveal that ALBERT models of different sizes, trained and initialized separately, consistently learn to use the axes of greatest variation to represent the approximate ordering of various numerical concepts. Numerals and their textual counterparts are represented in separate clusters, but increase along the same direction in 2D space. Our findings illustrate that language models, trained purely to model text, can intuit basic mathematical concepts, opening avenues for NLP applications that intersect with quantitative reasoning.
     </details>

386. **SAAS: Solving Ability Amplification Strategy for Enhanced Mathematical Reasoning in Large Language Models** [[pdf]](http://arxiv.org/abs/2404.03887) `2024-04-24` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This study proposes a sequential learning approach, named SAAS (Solving Ability Amplification Strategy), which strategically transitions from CoT learning to PoT learning, and demonstrates that the SAAS achieves state-of-the-art (SOTA) performance.
     </details>


     <details>
          <summary>Abstract</summary>
          This study presents a novel learning approach designed to enhance both mathematical reasoning and problem-solving abilities of Large Language Models (LLMs). We focus on integrating the Chain-of-Thought (CoT) and the Program-of-Thought (PoT) learning, hypothesizing that prioritizing the learning of mathematical reasoning ability is helpful for the amplification of problem-solving ability. Thus, the initial learning with CoT is essential for solving challenging mathematical problems. To this end, we propose a sequential learning approach, named SAAS (Solving Ability Amplification Strategy), which strategically transitions from CoT learning to PoT learning. Our empirical study, involving an extensive performance comparison using several benchmarks, demonstrates that our SAAS achieves state-of-the-art (SOTA) performance. The results underscore the effectiveness of our sequential learning approach, marking a significant advancement in the field of mathematical reasoning in LLMs.
     </details>

387. **Describe-then-Reason: Improving Multimodal Mathematical Reasoning through Visual Comprehension Training** [[pdf]](https://arxiv.org/abs/2404.14604v3) `2024-04-22` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A two-step training pipeline VCAR, which improves the visual comprehension ability of MLLMs through the visual description generation task, followed by another training step on generating rationales with the assistance of descriptions, demonstrates that VCAR substantially outperforms baseline methods solely relying on rationale supervision.
     </details>


     <details>
          <summary>Abstract</summary>
          Open-source multimodal large language models (MLLMs) excel in various tasks involving textual and visual inputs but still struggle with complex multimodal mathematical reasoning, lagging behind proprietary models like GPT-4V(ision) and Gemini-Pro. Although fine-tuning with intermediate steps (i.e., rationales) elicits some mathematical reasoning skills, the resulting models still fall short in visual comprehension due to inadequate visual-centric supervision, which leads to inaccurate interpretation of math figures. To address this issue, we propose a two-step training pipeline VCAR, which emphasizes the Visual Comprehension training in Addition to mathematical Reasoning learning. It first improves the visual comprehension ability of MLLMs through the visual description generation task, followed by another training step on generating rationales with the assistance of descriptions. Experimental results on two popular benchmarks demonstrate that VCAR substantially outperforms baseline methods solely relying on rationale supervision, especially on problems with high visual demands.
     </details>

388. **FGeo-HyperGNet: Geometric Problem Solving Integrating Formal Symbolic System and Hypergraph Neural Network** [[pdf]](http://arxiv.org/abs/2402.11461) `2024-04-22` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Geometric problem solving has always been a long-standing challenge in the fields of automated reasoning and artificial intelligence. We built a neural-symbolic system to automatically perform human-like geometric deductive reasoning. The symbolic part is a formal system built on FormalGeo, which can automatically perform geomertic relational reasoning and algebraic calculations and organize the solving process into a solution hypertree with conditions as hypernodes and theorems as hyperedges. The neural part, called HyperGNet, is a hypergraph neural network based on the attention mechanism, including a encoder to effectively encode the structural and semantic information of the hypertree, and a solver to provide problem-solving guidance. The neural part predicts theorems according to the hypertree, and the symbolic part applies theorems and updates the hypertree, thus forming a predict-apply cycle to ultimately achieve readable and traceable automatic solving of geometric problems. Experiments demonstrate the correctness and effectiveness of this neural-symbolic architecture. We achieved a step-wised accuracy of 87.65% and an overall accuracy of 85.53% on the formalgeo7k datasets.
     </details>

389. **MARIO Eval: Evaluate Your Math LLM with your Math LLM--A mathematical dataset evaluation toolkit** [[pdf]](http://arxiv.org/abs/2404.13925) `2024-04-22` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have been explored in a variety of reasoning tasks including solving of mathematical problems. Each math dataset typically includes its own specially designed evaluation script, which, while suitable for its intended use, lacks generalizability across different datasets. Consequently, updates and adaptations to these evaluation tools tend to occur without being systematically reported, leading to inconsistencies and obstacles to fair comparison across studies. To bridge this gap, we introduce a comprehensive mathematical evaluation toolkit that not only utilizes a python computer algebra system (CAS) for its numerical accuracy, but also integrates an optional LLM, known for its considerable natural language processing capabilities. To validate the effectiveness of our toolkit, we manually annotated two distinct datasets. Our experiments demonstrate that the toolkit yields more robust evaluation results compared to prior works, even without an LLM. Furthermore, when an LLM is incorporated, there is a notable enhancement. The code for our method will be made available at \url{https://github.com/MARIO-Math-Reasoning/math_evaluation}.
     </details>

390. **PARAMANU-GANITA: Language Model with Mathematical Capabilities** [[pdf]](http://arxiv.org/abs/2404.14395) `2024-04-22` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Paramanu-Ganita despite being 35 times smaller than 7B LLMs, outperformed generalist LLMs and math specialised LLMs such as Minerva 8B, LLEMMA 7B, and LLEMMA-7B in GSM8k test accuracy metric respectively and concludes that for strong mathematical reasoning abilities of language model, the authors do not need giant LLMs and immense computing power to their end.
     </details>


     <details>
          <summary>Abstract</summary>
          In this paper, we present Paramanu-Ganita, a 208 million parameter novel Auto Regressive (AR) decoder based language model on mathematics. The model is pretrained from scratch at context size of 4096 on our curated mixed mathematical corpus. We evaluate our model on both perplexity metric and GSM8k mathematical benchmark. Paramanu-Ganita despite being 35 times smaller than 7B LLMs, outperformed generalist LLMs such as LLaMa-1 7B by 28.4% points, LLaMa-2 7B by 27.6% points, Falcon 7B by 32.6% points, PaLM 8B by 35.3% points, and math specialised LLMs such as Minerva 8B by 23.2% points, and LLEMMA-7B by 3.0% points in GSM8k test accuracy metric respectively. Paramanu-Ganita also outperformed giant LLMs like PaLM 62B by 6.4% points, Falcon 40B by 19.8% points, LLaMa-1 33B by 3.8% points and Vicuna 13B by 11.8% points respectively. The large significant margin improvement in performance of our math model over the existing LLMs signifies that reasoning capabilities of language model are just not restricted to LLMs with humongous number of parameters. Paramanu-Ganita took 146 hours of A100 training whereas math specialised LLM, LLEMMA 7B, was trained for 23,000 A100 hours of training equivalent. Thus, our approach of pretraining powerful domain specialised language models from scratch for domain adaptation is much more cost-effective than performing continual training of LLMs for domain adaptation. Hence, we conclude that for strong mathematical reasoning abilities of language model, we do not need giant LLMs and immense computing power to our end. In the end, we want to point out that we have only trained Paramanu-Ganita only on a part of our entire mathematical corpus and yet to explore the full potential of our model.
     </details>

391. **Mathify: Evaluating Large Language Models on Mathematical Problem Solving Tasks** [[pdf]](http://arxiv.org/abs/2404.13099) `NeurIPS 2023 Workshop on Generative AI for Education` (7 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          An extensive mathematics dataset called "MathQuest"ourced from the 11th and 12th standard Mathematics NCERT textbooks is introduced, and MAmmoTH-13B establishes itself as a robust and dependable benchmark for addressing NCERT mathematics problems.
     </details>


     <details>
          <summary>Abstract</summary>
          The rapid progress in the field of natural language processing (NLP) systems and the expansion of large language models (LLMs) have opened up numerous opportunities in the field of education and instructional methods. These advancements offer the potential for tailored learning experiences and immediate feedback, all delivered through accessible and cost-effective services. One notable application area for this technological advancement is in the realm of solving mathematical problems. Mathematical problem-solving not only requires the ability to decipher complex problem statements but also the skill to perform precise arithmetic calculations at each step of the problem-solving process. However, the evaluation of the arithmetic capabilities of large language models remains an area that has received relatively little attention. In response, we introduce an extensive mathematics dataset called "MathQuest" sourced from the 11th and 12th standard Mathematics NCERT textbooks. This dataset encompasses mathematical challenges of varying complexity and covers a wide range of mathematical concepts. Utilizing this dataset, we conduct fine-tuning experiments with three prominent LLMs: LLaMA-2, WizardMath, and MAmmoTH. These fine-tuned models serve as benchmarks for evaluating their performance on our dataset. Our experiments reveal that among the three models, MAmmoTH-13B emerges as the most proficient, achieving the highest level of competence in solving the presented mathematical problems. Consequently, MAmmoTH-13B establishes itself as a robust and dependable benchmark for addressing NCERT mathematics problems.
     </details>

392. **Enhancing Length Extrapolation in Sequential Models with Pointer-Augmented Neural Memory** [[pdf]](https://arxiv.org/abs/2404.11870v1) `2024-04-18` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          PANM helps Transformer achieve up to 100% generalization accuracy in compositional learning tasks and significantly better results in mathematical reasoning, question answering and machine translation tasks.
     </details>


     <details>
          <summary>Abstract</summary>
          We propose Pointer-Augmented Neural Memory (PANM) to help neural networks understand and apply symbol processing to new, longer sequences of data. PANM integrates an external neural memory that uses novel physical addresses and pointer manipulation techniques to mimic human and computer symbol processing abilities. PANM facilitates pointer assignment, dereference, and arithmetic by explicitly using physical pointers to access memory content. Remarkably, it can learn to perform these operations through end-to-end training on sequence data, powering various sequential models. Our experiments demonstrate PANM's exceptional length extrapolating capabilities and improved performance in tasks that require symbol processing, such as algorithmic reasoning and Dyck language recognition. PANM helps Transformer achieve up to 100% generalization accuracy in compositional learning tasks and significantly better results in mathematical reasoning, question answering and machine translation tasks.
     </details>

393. **Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models** [[pdf]](http://arxiv.org/abs/2312.06585) `2024-04-17` (72 cite) (7 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Testing on advanced MATH reasoning and APPS coding benchmarks using PaLM-2 models, it is found that ReST$^{EM}$ scales favorably with model size and significantly surpasses fine-tuning only on human data.
     </details>


     <details>
          <summary>Abstract</summary>
          Fine-tuning language models~(LMs) on human-generated data remains a prevalent practice. However, the performance of such models is often limited by the quantity and diversity of high-quality human data. In this paper, we explore whether we can go beyond human data on tasks where we have access to scalar feedback, for example, on math problems where one can verify correctness. To do so, we investigate a simple self-training method based on expectation-maximization, which we call ReST$^{EM}$, where we (1) generate samples from the model and filter them using binary feedback, (2) fine-tune the model on these samples, and (3) repeat this process a few times. Testing on advanced MATH reasoning and APPS coding benchmarks using PaLM-2 models, we find that ReST$^{EM}$ scales favorably with model size and significantly surpasses fine-tuning only on human data. Overall, our findings suggest self-training with feedback can substantially reduce dependence on human-generated data.
     </details>

394. **Reformatted Alignment** [[pdf]](http://arxiv.org/abs/2402.12219) `2024-04-17` (6 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A simple and effective approach named ReAlign is introduced, which reformats the responses of instruction data into a format that better aligns with pre-established criteria and the collated evidence, and significantly boosts the general alignment ability, math reasoning, factuality, and readability of the LLMs.
     </details>


     <details>
          <summary>Abstract</summary>
          The quality of finetuning data is crucial for aligning large language models (LLMs) with human values. Current methods to improve data quality are either labor-intensive or prone to factual errors caused by LLM hallucinations. This paper explores elevating the quality of existing instruction data to better align with human values, introducing a simple and effective approach named ReAlign, which reformats the responses of instruction data into a format that better aligns with pre-established criteria and the collated evidence. This approach minimizes human annotation, hallucination, and the difficulty in scaling, remaining orthogonal to existing alignment techniques. Experimentally, ReAlign significantly boosts the general alignment ability, math reasoning, factuality, and readability of the LLMs.   Encouragingly, without introducing any additional data or advanced training techniques, and merely by reformatting the response, LLaMA-2-13B's mathematical reasoning ability on GSM8K can be improved from 46.77% to 56.63% in accuracy. Additionally, a mere 5% of ReAlign data yields a 67% boost in general alignment ability measured by the Alpaca dataset. This work highlights the need for further research into the science and mechanistic interpretability of LLMs. We have made the associated code and data publicly accessible to support future studies at https://github.com/GAIR-NLP/ReAlign.
     </details>

395. **A Survey on Deep Learning for Theorem Proving** [[pdf]](http://arxiv.org/abs/2404.09939) `COLM 2024` (6 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper presents a comprehensive survey of deep learning for theorem proving by offering a thorough review of existing approaches across various tasks such as autoformalization, premise selection, proofstep generation, and proof search.
     </details>


     <details>
          <summary>Abstract</summary>
          Theorem proving is a fundamental aspect of mathematics, spanning from informal reasoning in natural language to rigorous derivations in formal systems. In recent years, the advancement of deep learning, especially the emergence of large language models, has sparked a notable surge of research exploring these techniques to enhance the process of theorem proving. This paper presents a comprehensive survey of deep learning for theorem proving by offering (i) a thorough review of existing approaches across various tasks such as autoformalization, premise selection, proofstep generation, and proof search; (ii) an extensive summary of curated datasets and strategies for synthetic data generation; (iii) a detailed analysis of evaluation metrics and the performance of state-of-the-art methods; and (iv) a critical discussion on the persistent challenges and the promising avenues for future exploration. Our survey aims to serve as a foundational reference for deep learning approaches in theorem proving, inspiring and catalyzing further research endeavors in this rapidly growing field. A curated list of papers is available at https://github.com/zhaoyu-li/DL4TP.
     </details>

396. **Wu's Method can Boost Symbolic AI to Rival Silver Medalists and AlphaGeometry to Outperform Gold Medalists at IMO Geometry** [[pdf]](http://arxiv.org/abs/2404.06405) `2024-04-11` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          By combining AlphaGeometry with Wu's method, the IMO-AG-30 Challenge is revisited, and Wu's method is surprisingly strong, setting a new state-of-the-art for automated theorem proving on IMO-AG-30, solving 27 out of 30 problems, the first AI method which outperforms an IMO gold medalist.
     </details>


     <details>
          <summary>Abstract</summary>
          Proving geometric theorems constitutes a hallmark of visual reasoning combining both intuitive and logical skills. Therefore, automated theorem proving of Olympiad-level geometry problems is considered a notable milestone in human-level automated reasoning. The introduction of AlphaGeometry, a neuro-symbolic model trained with 100 million synthetic samples, marked a major breakthrough. It solved 25 of 30 International Mathematical Olympiad (IMO) problems whereas the reported baseline based on Wu's method solved only ten. In this note, we revisit the IMO-AG-30 Challenge introduced with AlphaGeometry, and find that Wu's method is surprisingly strong. Wu's method alone can solve 15 problems, and some of them are not solved by any of the other methods. This leads to two key findings: (i) Combining Wu's method with the classic synthetic methods of deductive databases and angle, ratio, and distance chasing solves 21 out of 30 methods by just using a CPU-only laptop with a time limit of 5 minutes per problem. Essentially, this classic method solves just 4 problems less than AlphaGeometry and establishes the first fully symbolic baseline strong enough to rival the performance of an IMO silver medalist. (ii) Wu's method even solves 2 of the 5 problems that AlphaGeometry failed to solve. Thus, by combining AlphaGeometry with Wu's method we set a new state-of-the-art for automated theorem proving on IMO-AG-30, solving 27 out of 30 problems, the first AI method which outperforms an IMO gold medalist.
     </details>

397. **A Survey in Mathematical Language Processing** [[pdf]](http://arxiv.org/abs/2205.15231) `2024-04-08` (11 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work tracks the development of informal mathematical language processing approaches across five strategic sub-areas in recent years, highlighting the prevailing successful methodological elements along with existing limitations.
     </details>


     <details>
          <summary>Abstract</summary>
          Informal mathematical text underpins real-world quantitative reasoning and communication. Developing sophisticated methods of retrieval and abstraction from this dual modality is crucial in the pursuit of the vision of automating discovery in quantitative science and mathematics. We track the development of informal mathematical language processing approaches across five strategic sub-areas in recent years, highlighting the prevailing successful methodological elements along with existing limitations.
     </details>

398. **Evaluating Mathematical Reasoning Beyond Accuracy** [[pdf]](http://arxiv.org/abs/2404.05692) `2024-04-08` (5 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          The leaderboard of Large Language Models (LLMs) in mathematical tasks has been continuously updated. However, the majority of evaluations focus solely on the final results, neglecting the quality of the intermediate steps. This oversight can mask underlying problems, such as logical errors or unnecessary steps in the reasoning process. To measure reasoning beyond final-answer accuracy, we introduce ReasonEval, a new methodology for evaluating the quality of reasoning steps. ReasonEval employs $\textit{validity}$ and $\textit{redundancy}$ to characterize the reasoning quality, as well as accompanying LLMs to assess them automatically. Instantiated by base models that possess strong mathematical knowledge and trained with high-quality labeled data, ReasonEval achieves state-of-the-art performance on human-labeled datasets and can accurately detect different types of errors generated by perturbation. When applied to evaluate LLMs specialized in math, we find that an increase in final-answer accuracy does not necessarily guarantee an improvement in the overall quality of the reasoning steps for challenging mathematical problems. Additionally, we observe that ReasonEval can play a significant role in data selection. We release the best-performing model, meta-evaluation script, and all evaluation results at https://github.com/GAIR-NLP/ReasonEval.
     </details>

399. **FRACTAL: Fine-Grained Scoring from Aggregate Text Labels** [[pdf]](http://arxiv.org/abs/2404.04817) `2024-04-07` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work is the first to develop response-level feedback to sentence-level scoring techniques, leveraging sentence-level prior information, along with comprehensive evaluations on multiple tasks as well as end-to-end finetuning evaluation showing performance comparable to a model trained on fine-grained human annotated labels.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) are being increasingly tuned to power complex generation tasks such as writing, fact-seeking, querying and reasoning. Traditionally, human or model feedback for evaluating and further tuning LLM performance has been provided at the response level, enabling faster and more cost-effective assessments. However, recent works (Amplayo et al. [2022], Wu et al. [2023]) indicate that sentence-level labels may provide more accurate and interpretable feedback for LLM optimization. In this work, we introduce methods to disaggregate response-level labels into sentence-level (pseudo-)labels. Our approach leverages multiple instance learning (MIL) and learning from label proportions (LLP) techniques in conjunction with prior information (e.g., document-sentence cosine similarity) to train a specialized model for sentence-level scoring. We also employ techniques which use model predictions to pseudo-label the train-set at the sentence-level for model training to further improve performance.   We conduct extensive evaluations of our methods across six datasets and four tasks: retrieval, question answering, summarization, and math reasoning. Our results demonstrate improved performance compared to multiple baselines across most of these tasks. Our work is the first to develop response-level feedback to sentence-level scoring techniques, leveraging sentence-level prior information, along with comprehensive evaluations on multiple tasks as well as end-to-end finetuning evaluation showing performance comparable to a model trained on fine-grained human annotated labels.
     </details>

400. **MM-MATH: Advancing Multimodal Math Evaluation with Process Evaluation and Fine-grained Classification** [[pdf]](https://arxiv.org/abs/2404.05091v4) `2024-04-07` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduces a novel benchmark, MM-MATH, consisting of 5,929 open-ended middle school math problems with visual contexts, with fine-grained classification across difficulty, grade level, and knowledge points, which incorporates both outcome and process evaluations.
     </details>


     <details>
          <summary>Abstract</summary>
          To advance the evaluation of multimodal math reasoning in large multimodal models (LMMs), this paper introduces a novel benchmark, MM-MATH. MM-MATH consists of 5,929 open-ended middle school math problems with visual contexts, with fine-grained classification across difficulty, grade level, and knowledge points. Unlike existing benchmarks relying on binary answer comparison, MM-MATH incorporates both outcome and process evaluations. Process evaluation employs LMM-as-a-judge to automatically analyze solution steps, identifying and categorizing errors into specific error types. Extensive evaluation of ten models on MM-MATH reveals significant challenges for existing LMMs, highlighting their limited utilization of visual information and struggles with higher-difficulty problems. The best-performing model achieves only 31% accuracy on MM-MATH, compared to 82% for humans. This highlights the challenging nature of our benchmark for existing models and the significant gap between the multimodal reasoning capabilities of current models and humans. Our process evaluation reveals that diagram misinterpretation is the most common error, accounting for more than half of the total error cases, underscoring the need for improved image comprehension in multimodal reasoning.
     </details>

401. **Large Language Models for Mathematical Reasoning: Progresses and Challenges** [[pdf]](http://arxiv.org/abs/2402.00157) `EACL 2024 student research workshop` (53 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This survey stands as one of the first extensive examinations of the landscape of LLM-oriented techniques in the realm of mathematics, providing a holistic perspective on the current state, accomplishments, and future challenges in this rapidly evolving field.
     </details>


     <details>
          <summary>Abstract</summary>
          Mathematical reasoning serves as a cornerstone for assessing the fundamental cognitive capabilities of human intelligence. In recent times, there has been a notable surge in the development of Large Language Models (LLMs) geared towards the automated resolution of mathematical problems. However, the landscape of mathematical problem types is vast and varied, with LLM-oriented techniques undergoing evaluation across diverse datasets and settings. This diversity makes it challenging to discern the true advancements and obstacles within this burgeoning field. This survey endeavors to address four pivotal dimensions: i) a comprehensive exploration of the various mathematical problems and their corresponding datasets that have been investigated; ii) an examination of the spectrum of LLM-oriented techniques that have been proposed for mathematical problem-solving; iii) an overview of factors and concerns affecting LLMs in solving math; and iv) an elucidation of the persisting challenges within this domain. To the best of our knowledge, this survey stands as one of the first extensive examinations of the landscape of LLMs in the realm of mathematics, providing a holistic perspective on the current state, accomplishments, and future challenges in this rapidly evolving field.
     </details>

402. **Data Augmentation with In-Context Learning and Comparative Evaluation in Math Word Problem Solving** [[pdf]](https://arxiv.org/abs/2404.03938v1) `2024-04-05` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This study aims to provide MWP solvers with a more diverse training set, ultimately improving their ability to solve various math problems by introducing a new in-context learning augmentation method, employing the Llama-7b language model.
     </details>


     <details>
          <summary>Abstract</summary>
          Math Word Problem (MWP) solving presents a challenging task in Natural Language Processing (NLP). This study aims to provide MWP solvers with a more diverse training set, ultimately improving their ability to solve various math problems. We propose several methods for data augmentation by modifying the problem texts and equations, such as synonym replacement, rule-based: question replacement, and rule based: reversing question methodologies over two English MWP datasets. This study extends by introducing a new in-context learning augmentation method, employing the Llama-7b language model. This approach involves instruction-based prompting for rephrasing the math problem texts. Performance evaluations are conducted on 9 baseline models, revealing that augmentation methods outperform baseline models. Moreover, concatenating examples generated by various augmentation methods further improves performance.
     </details>

403. **Proceedings 12th International Workshop on Theorem proving components for Educational software** [[pdf]](https://arxiv.org/abs/2404.03709v1) `2024-04-04` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The volume editors hope that this collection of papers will further promote the development of theorem-proving based software, and that it will allow to improve the mutual understanding between computer scientists, mathematicians and stakeholders in education.
     </details>


     <details>
          <summary>Abstract</summary>
          The ThEdu series pursues the smooth transition from an intuitive way of doing mathematics at secondary school to a more formal approach to the subject in STEM education, while favouring software support for this transition by exploiting the power of theorem-proving technologies. What follows is a brief description of how the present volume contributes to this enterprise.   The 12th International Workshop on Theorem Proving Components for Educational Software(ThEdu'23), was a satellite event of the 29th international Conference on Automated Deduction (CADE 2023), July 1-4, 2023, Rome, Italy. ThEdu'23 was very successful, with one invited talk, by Yves Bertot (Inria, France), "The challenges of using Type Theory to teach Mathematics", and seven regular contributions. An open call for papers was then issued, to which eight contributions were submitted. Seven submissions have been accepted by our reviewers, who jointly produced at least three careful reports on each of the contributions. The resulting revised papers are collected in the present volume.   We, the volume editors, hope that this collection of papers will further promote the development of theorem-proving based software, and that it will allow to improve the mutual understanding between computer scientists, mathematicians and stakeholders in education.   PC Chairs:Julien Narboux (University of Strasbourg, France); Walther Neuper (JKU, Johannes Kepler University, Linz, Austria); Pedro Quaresma (University of Coimbra, Portugal)
     </details>

404. **ChatGLM-Math: Improving Math Problem-Solving in Large Language Models with a Self-Critique Pipeline** [[pdf]](https://arxiv.org/abs/2404.02893v1) `2024-04-03` (18 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work tailor the Self-Critique pipeline, which addresses the challenge in the feedback learning stage of LLM alignment, and significantly enhances the LLM's mathematical problem-solving while still improving its language ability, outperforming LLMs that could be two times larger.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have shown excellent mastering of human language, but still struggle in real-world applications that require mathematical problem-solving. While many strategies and datasets to enhance LLMs' mathematics are developed, it remains a challenge to simultaneously maintain and improve both language and mathematical capabilities in deployed LLM systems.In this work, we tailor the Self-Critique pipeline, which addresses the challenge in the feedback learning stage of LLM alignment. We first train a general Math-Critique model from the LLM itself to provide feedback signals. Then, we sequentially employ rejective fine-tuning and direct preference optimization over the LLM's own generations for data collection. Based on ChatGLM3-32B, we conduct a series of experiments on both academic and our newly created challenging dataset, MathUserEval. Results show that our pipeline significantly enhances the LLM's mathematical problem-solving while still improving its language ability, outperforming LLMs that could be two times larger. Related techniques have been deployed to ChatGLM\footnote{\url{https://chatglm.cn}}, an online serving LLM. Related evaluation dataset and scripts are released at \url{https://github.com/THUDM/ChatGLM-Math}.
     </details>

405. **Advancing LLM Reasoning Generalists with Preference Trees** [[pdf]](http://arxiv.org/abs/2404.02078) `ICLR 2025 Submission` (41 cite) (3 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces Eurus, a suite of large language models (LLMs) optimized for reasoning, and derives a novel reward modeling objective which, together with UltraInteract, leads to a strong reward model.
     </details>


     <details>
          <summary>Abstract</summary>
          We introduce Eurus, a suite of large language models (LLMs) optimized for reasoning. Finetuned from Mistral-7B and CodeLlama-70B, Eurus models achieve state-of-the-art results among open-source models on a diverse set of benchmarks covering mathematics, code generation, and logical reasoning problems. Notably, Eurus-70B beats GPT-3.5 Turbo in reasoning through a comprehensive benchmarking across 12 tests covering five tasks, and achieves a 33.3% pass@1 accuracy on LeetCode and 32.6% on TheoremQA, two challenging benchmarks, substantially outperforming existing open-source models by margins more than 13.3%. The strong performance of Eurus can be primarily attributed to UltraInteract, our newly-curated large-scale, high-quality alignment dataset specifically designed for complex reasoning tasks. UltraInteract can be used in both supervised fine-tuning and preference learning. For each instruction, it includes a preference tree consisting of (1) reasoning chains with diverse planning strategies in a unified format, (2) multi-turn interaction trajectories with the environment and the critique, and (3) pairwise data to facilitate preference learning. UltraInteract allows us to conduct an in-depth exploration of preference learning for reasoning tasks. Our investigation reveals that some well-established preference learning algorithms may be less suitable for reasoning tasks compared to their effectiveness in general conversations. Inspired by this, we derive a novel reward modeling objective which, together with UltraInteract, leads to a strong reward model.
     </details>

406. **Autonomous Data Selection with Language Models for Mathematical Texts** [[pdf]](http://arxiv.org/abs/2402.07625) `ICLR 2024 Workshop DPFM` (15 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work continuously pretrained a 7B-parameter language model on the authors' curated dataset, achieving substantial improvements in downstream performance on the MATH, GSM8K, and BIG-Bench Hard (BBH) tasks with a token amount reduced by orders of magnitude compared to previous continual pretraining works.
     </details>


     <details>
          <summary>Abstract</summary>
          To improve language models' proficiency in mathematical reasoning via continual pretraining, we introduce a novel strategy that leverages base language models for autonomous data selection. Departing from conventional supervised fine-tuning or trained classifiers with human-annotated data, our approach Autonomous Data Selection (AutoDS) utilizes meta-prompted language models as zero-shot verifiers to evaluate and select high-quality mathematical content autonomously. To demonstrate the efficacy of our method, we continuously pretrained a 7B-parameter language model on our curated dataset, achieving substantial improvements in downstream performance on the MATH, GSM8K, and BIG-Bench Hard (BBH) tasks with a token amount reduced by orders of magnitude compared to previous continual pretraining works. Our method showcases a 2 times increase in pretraining token efficiency compared to state-of-the-art baselines, underscoring the potential of our approach in enhancing models' mathematical reasoning capabilities. The AutoMathText dataset is available at https://huggingface.co/datasets/math-ai/AutoMathText. The code is available at https://github.com/yifanzhang-pro/AutoMathText.
     </details>

407. **Large Language Models for Mathematicians** [[pdf]](http://arxiv.org/abs/2312.04556) `2024-04-02` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A mathematical description of the transformer model used in all modern language models is provided and light is shed on the potential of LLMs to change how mathematicians work.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) such as ChatGPT have received immense interest for their general-purpose language understanding and, in particular, their ability to generate high-quality text or computer code. For many professions, LLMs represent an invaluable tool that can speed up and improve the quality of work. In this note, we discuss to what extent they can aid professional mathematicians. We first provide a mathematical description of the transformer model used in all modern language models. Based on recent studies, we then outline best practices and potential issues and report on the mathematical abilities of language models. Finally, we shed light on the potential of LLMs to change how mathematicians work.
     </details>

408. **$\texttt{LM}^\texttt{2}$: A Simple Society of Language Models Solves Complex Reasoning** [[pdf]](https://arxiv.org/abs/2404.02255v1) `2024-04-02` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Despite demonstrating emergent reasoning abilities, Large Language Models (LLMS) often lose track of complex, multi-step reasoning. Existing studies show that providing guidance via decomposing the original question into multiple subproblems elicits more robustness in LLM reasoning -- a decomposer generates the subproblems, and a solver solves each of these subproblems. However, these techniques fail to accommodate coordination between the decomposer and the solver modules (either in a single model or different specialized ones) -- the decomposer does not keep track of the ability of the solver to follow the decomposed reasoning. In this paper, we propose LM2 to address these challenges. LM2 modularizes the decomposition, solution, and verification into three different language models. The decomposer module identifies the key concepts necessary to solve the problem and generates step-by-step subquestions according to the reasoning requirement. The solver model generates the solution to the subproblems that are then checked by the verifier module; depending upon the feedback from the verifier, the reasoning context is constructed using the subproblems and the solutions. These models are trained to coordinate using policy learning. Exhaustive experimentation suggests the superiority of LM2 over existing methods on in- and out-domain reasoning problems, outperforming the best baselines by $8.1\%$ on MATH, $7.71\%$ on JEEBench, and $9.7\%$ on MedQA problems (code available at https://github.com/LCS2-IIITD/Language_Model_Multiplex).
     </details>

409. **Enhancing Formal Theorem Proving: A Comprehensive Dataset for Training AI Models on Coq Code** [[pdf]](http://arxiv.org/abs/2403.12627) `2024-04-02` `Coq` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper presents a comprehensive dataset specifically designed to enhance LLMs' proficiency in interpreting and generating Coq code, and discusses the dataset's composition, the methodology behind its creation, and the implications for the future of machine learning in formal verification.
     </details>


     <details>
          <summary>Abstract</summary>
          In the realm of formal theorem proving, the Coq proof assistant stands out for its rigorous approach to verifying mathematical assertions and software correctness. Despite the advances in artificial intelligence and machine learning, the specialized nature of Coq syntax and semantics poses unique challenges for Large Language Models (LLMs). Addressing this gap, we present a comprehensive dataset specifically designed to enhance LLMs' proficiency in interpreting and generating Coq code. This dataset, derived from a collection of over 10,000 Coq source files, encompasses a wide array of propositions, proofs, and definitions, enriched with metadata including source references and licensing information. Our primary aim is to facilitate the development of LLMs capable of generating syntactically correct and semantically meaningful Coq constructs, thereby advancing the frontier of automated theorem proving. Initial experiments with this dataset have showcased its significant potential; models trained on this data exhibited enhanced accuracy in Coq code generation. Notably, a particular experiment revealed that a fine-tuned LLM was capable of generating 141 valid proofs for a basic lemma, highlighting the dataset's utility in facilitating the discovery of diverse and valid proof strategies. This paper discusses the dataset's composition, the methodology behind its creation, and the implications of our findings for the future of machine learning in formal verification. The dataset is accessible for further research and exploration: https://huggingface.co/datasets/florath/coq-facts-props-proofs-gen0-v1
     </details>

410. **Can Large Language Models Reason and Plan?** [[pdf]](http://arxiv.org/abs/2403.04121) `2024-04-01` (27 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          While humans sometimes do show the capability of correcting their own erroneous guesses with self-critiquing, there seems to be no basis for that assumption in the case of LLMs.
     </details>

411. **OVM, Outcome-supervised Value Models for Planning in Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2311.09724) `NAACL 2024 Findings` (18 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Outcome-supervised Value Model (OVM) is proposed that employs outcome supervision for training a value model, which prioritizes steps that lead to accurate conclusions, thereby significantly enhancing its scalability and eliminating the need for labor-intensive annotations of step-level correctness.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) often struggle with maintaining accuracy throughout multiple multiple reasoning steps, especially in mathematical reasoning where an error in earlier steps can propagate to subsequent ones and it ultimately leading to an incorrect answer.To reduce error propagation, guided decoding is employed to direct the LM decoding on a step-by-step basis. We argue that in guided decoding, assessing the potential of an incomplete reasoning path can be more advantageous than simply ensuring per-step correctness, as the former approach leads towards a correct final answer. This transforms the task into a value estimation problem in planning.Inspired by the findings that outcome supervision for guided decoding essentially acts as a value model, we propose Outcome-supervised Value Model (OVM) that employs outcome supervision for training a value model, which prioritizes steps that lead to accurate conclusions. Furthermore, the OVM eliminates the need for labor-intensive annotations of step-level correctness, thereby significantly enhancing its scalability. Our experiments on two multi-step mathematical reasoning datasets, GSM8K and Game of 24, demonstrate the superior performance of the OVM model. Notably, in GSM8K, our OVM-7B model achieves state-of-the-art results among LLMs up to 13B parameters; especially it does not utilize GPT-4 or code execution. These findings offer a novel perspective on the role of outcome supervision in training value models for multi-step reasoning tasks and provide theoretical justification for its advantage in value estimation for guided decoding.
     </details>

412. **Do language models plan ahead for future tokens?** [[pdf]](https://arxiv.org/abs/2404.00859v2) `2024-04-01` (10 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          In the autoregressive language modeling setting, the experiments are more suggestive of the breadcrumbs hypothesis, though pre-caching increases with model scale, and in a constructed synthetic data setting, they find clear evidence for pre-caching.
     </details>


     <details>
          <summary>Abstract</summary>
          Do transformers "think ahead" during inference at a given position? It is known transformers prepare information in the hidden states of the forward pass at time step $t$ that is then used in future forward passes $t+\tau$. We posit two explanations for this phenomenon: pre-caching, in which off-diagonal gradient terms present during training result in the model computing features at $t$ irrelevant to the present inference task but useful for the future, and breadcrumbs, in which features most relevant to time step $t$ are already the same as those that would most benefit inference at time $t+\tau$. We test these hypotheses by training language models without propagating gradients to past timesteps, a scheme we formalize as myopic training. In a constructed synthetic data setting, we find clear evidence for pre-caching. In the autoregressive language modeling setting, our experiments are more suggestive of the breadcrumbs hypothesis, though pre-caching increases with model scale.
     </details>

413. **Exploring the Mystery of Influential Data for Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2404.01067) `2024-04-01` (5 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a Quality-aware Diverse Selection (QaDS) strategy adaptable for mathematical reasoning, and showcases the use of QaDS in creating efficient fine-tuning mixtures with various selection ratios.
     </details>


     <details>
          <summary>Abstract</summary>
          Selecting influential data for fine-tuning on downstream tasks is a key factor for both performance and computation efficiency. Recent works have shown that training with only limited data can show a superior performance on general tasks. However, the feasibility on mathematical reasoning tasks has not been validated. To go further, there exist two open questions for mathematical reasoning: how to select influential data and what is an influential data composition. For the former one, we propose a Quality-aware Diverse Selection (QaDS) strategy adaptable for mathematical reasoning. A comparison with other selection strategies validates the superiority of QaDS. For the latter one, we first enlarge our setting and explore the influential data composition. We conduct a series of experiments and highlight: scaling up reasoning data, and training with general data selected by QaDS is helpful. Then, we define our optimal mixture as OpenMathMix, an influential data mixture with open-source data selected by QaDS. With OpenMathMix, we achieve a state-of-the-art 48.8% accuracy on MATH with 7B base model. Additionally, we showcase the use of QaDS in creating efficient fine-tuning mixtures with various selection ratios, and analyze the quality of a wide range of open-source datasets, which can perform as a reference for future works on mathematical reasoning tasks.
     </details>

414. **GFLean: An Autoformalisation Framework for Lean via GF** [[pdf]](http://arxiv.org/abs/2404.01234) `2024-04-01` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          An autoformalisation framework for the Lean theorem prover, called GFLean, which uses a high-level grammar writing tool called Grammatical Framework for parsing and linearisation and is implemented in Haskell.
     </details>


     <details>
          <summary>Abstract</summary>
          We present an autoformalisation framework for the Lean theorem prover, called GFLean. GFLean uses a high-level grammar writing tool called Grammatical Framework (GF) for parsing and linearisation. GFLean is implemented in Haskell. We explain the functionalities of GFLean, its inner working and discuss its limitations. We also discuss how we can use neural network based translation programs and rule based translation programs together complimenting each other to build robust autoformalisation frameworks.
     </details>

415. **NumeroLogic: Number Encoding for Enhanced LLMs' Numerical Reasoning** [[pdf]](http://arxiv.org/abs/2404.00459) `2024-03-30` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a simple adjustment to how numbers are represented by including the count of digits before each number, which offers an added advantage in number generation by serving as a Chain of Thought (CoT) by requiring the model to consider the number of digits first.
     </details>


     <details>
          <summary>Abstract</summary>
          Language models struggle with handling numerical data and performing arithmetic operations. We hypothesize that this limitation can be partially attributed to non-intuitive textual numbers representation. When a digit is read or generated by a causal language model it does not know its place value (e.g. thousands vs. hundreds) until the entire number is processed. To address this issue, we propose a simple adjustment to how numbers are represented by including the count of digits before each number. For instance, instead of "42", we suggest using "{2:42}" as the new format. This approach, which we term NumeroLogic, offers an added advantage in number generation by serving as a Chain of Thought (CoT). By requiring the model to consider the number of digits first, it enhances the reasoning process before generating the actual number. We use arithmetic tasks to demonstrate the effectiveness of the NumeroLogic formatting. We further demonstrate NumeroLogic applicability to general natural language modeling, improving language understanding performance in the MMLU benchmark.
     </details>

416. **Can LLMs Master Math? Investigating Large Language Models on Math Stack Exchange** [[pdf]](https://arxiv.org/abs/2404.00344v1) `2024-03-30` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The case analysis indicates that while the GPT-4 can generate relevant responses in certain instances, it does not consistently answer all questions accurately, and shed light on the gaps in LLM capabilities within mathematics, thereby setting the stage for future research and advancements in AI-driven mathematical reasoning.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have demonstrated exceptional capabilities in various natural language tasks, often achieving performances that surpass those of humans. Despite these advancements, the domain of mathematics presents a distinctive challenge, primarily due to its specialized structure and the precision it demands. In this study, we adopted a two-step approach for investigating the proficiency of LLMs in answering mathematical questions. First, we employ the most effective LLMs, as identified by their performance on math question-answer benchmarks, to generate answers to 78 questions from the Math Stack Exchange (MSE). Second, a case analysis is conducted on the LLM that showed the highest performance, focusing on the quality and accuracy of its answers through manual evaluation. We found that GPT-4 performs best (nDCG of 0.48 and P@10 of 0.37) amongst existing LLMs fine-tuned for answering mathematics questions and outperforms the current best approach on ArqMATH3 Task1, considering P@10. Our Case analysis indicates that while the GPT-4 can generate relevant responses in certain instances, it does not consistently answer all questions accurately. This paper explores the current limitations of LLMs in navigating complex mathematical problem-solving. Through case analysis, we shed light on the gaps in LLM capabilities within mathematics, thereby setting the stage for future research and advancements in AI-driven mathematical reasoning. We make our code and findings publicly available for research: \url{https://github.com/gipplab/LLM-Investig-MathStackExchange}
     </details>

417. **Large Language Models Are Unconscious of Unreasonability in Math Problems** [[pdf]](https://arxiv.org/abs/2403.19346v2) `2024-03-28` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          With a strategic prompt template called Critical Calculation and Conclusion (CCC), LLMs can better self-evaluate and detect unreasonable errors in math questions, making them more reliable and safe in practical application scenarios.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) demonstrate substantial capabilities in solving math problems. However, they tend to produce hallucinations when given questions containing unreasonable errors. In this paper, we study the behavior of LLMs when faced with unreasonable math problems and further explore their potential to address these problems. We construct the Unreasonable Math Problem (UMP) benchmark to examine the error detection ability of LLMs. Experiments show that LLMs are able to detect unreasonable errors, but still fail in generating non-hallucinatory content. In order to improve their ability of error detection and correction, we further design a strategic prompt template called Critical Calculation and Conclusion(CCC). With CCC, LLMs can better self-evaluate and detect unreasonable errors in math questions, making them more reliable and safe in practical application scenarios.
     </details>

418. **Dual Instruction Tuning with Large Language Models for Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2403.18295) `2024-03-27` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a dual instruction tuning strategy to meticulously model mathematical reasoning from both forward and reverse directions, and introduces the Intermediate Reasoning State Prediction task (forward reasoning) and the Instruction Reconstruction task (reverse reasoning) to enhance the LLMs' understanding and execution of instructions.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent advancements highlight the success of instruction tuning with large language models (LLMs) utilizing Chain-of-Thought (CoT) data for mathematical reasoning tasks. Despite the fine-tuned LLMs, challenges persist, such as incorrect, missing, and redundant steps in CoT generation leading to inaccuracies in answer predictions. To alleviate this problem, we propose a dual instruction tuning strategy to meticulously model mathematical reasoning from both forward and reverse directions. This involves introducing the Intermediate Reasoning State Prediction task (forward reasoning) and the Instruction Reconstruction task (reverse reasoning) to enhance the LLMs' understanding and execution of instructions. Training instances for these tasks are constructed based on existing mathematical instruction tuning datasets. Subsequently, LLMs undergo multi-task fine-tuning using both existing mathematical instructions and the newly created data. Comprehensive experiments validate the effectiveness and domain generalization of the dual instruction tuning strategy across various mathematical reasoning tasks.
     </details>

419. **Look Before You Leap: Problem Elaboration Prompting Improves Mathematical Reasoning in Large Language Models** [[pdf]](http://arxiv.org/abs/2402.15764) `2024-03-26` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A new approach named Problem Elaboration Prompting (PEP) is proposed to enhance the mathematical capacities of LLMs by decomposing and elucidates the problem context before reasoning, therefore enhancing the context modeling and parsing efficiency.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) still grapple with complex tasks like mathematical reasoning. Despite significant efforts invested in improving prefix prompts or reasoning process, the crucial role of problem context might have been neglected. Accurate recognition of inputs is fundamental for solving mathematical tasks, as ill-formed problems could potentially mislead LLM's reasoning. In this study, we propose a new approach named Problem Elaboration Prompting (PEP) to enhance the mathematical capacities of LLMs. Specifically, PEP decomposes and elucidates the problem context before reasoning, therefore enhancing the context modeling and parsing efficiency. Experiments across datasets and models demonstrate promising performances: (1) PEP demonstrates an overall enhancement in various mathematical tasks. For instance, with the GPT-3.5 model, PEP exhibits improvements of 9.93% and 8.80% on GSM8k through greedy decoding and self-consistency, respectively. (2) PEP can be easily implemented and integrated with other prompting methods. (3) PEP shows particular strength in handling distraction problems.
     </details>

420. **Graph of Thoughts: Solving Elaborate Problems with Large Language Models** [[pdf]](http://arxiv.org/abs/2308.09687) `AAAI 2024 Natural Language Processing` (335 cite) (14 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Graph of Thoughts is introduced: a framework that advances prompting capabilities in large language models (LLMs) beyond those offered by paradigms such as Chain-of-Thought or Tree of Thoughts, and is ensured that GoT is extensible with new thought transformations and thus can be used to spearhead new prompting schemes.
     </details>


     <details>
          <summary>Abstract</summary>
          We introduce Graph of Thoughts (GoT): a framework that advances prompting capabilities in large language models (LLMs) beyond those offered by paradigms such as Chain-of-Thought or Tree of Thoughts (ToT). The key idea and primary advantage of GoT is the ability to model the information generated by an LLM as an arbitrary graph, where units of information ("LLM thoughts") are vertices, and edges correspond to dependencies between these vertices. This approach enables combining arbitrary LLM thoughts into synergistic outcomes, distilling the essence of whole networks of thoughts, or enhancing thoughts using feedback loops. We illustrate that GoT offers advantages over state of the art on different tasks, for example increasing the quality of sorting by 62% over ToT, while simultaneously reducing costs by >31%. We ensure that GoT is extensible with new thought transformations and thus can be used to spearhead new prompting schemes. This work brings the LLM reasoning closer to human thinking or brain mechanisms such as recurrence, both of which form complex networks
     </details>

421. **Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Language Models** [[pdf]](http://arxiv.org/abs/2305.16582) `2024-03-22` (6 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Graph-of-Thought reasoning is proposed, which models human thought processes not only as a chain but also as a graph, and achieves significant improvement over the strong CoT baseline on the AQUA-RAT test set and boosts accuracy from 85.19% to 87.59% using the T5-base model.
     </details>


     <details>
          <summary>Abstract</summary>
          With the widespread use of language models (LMs) in NLP tasks, researchers have discovered the potential of Chain-of-thought (CoT) to assist LMs in accomplishing complex reasoning tasks by generating intermediate steps. However, human thought processes are often non-linear, rather than simply sequential chains of thoughts. Therefore, we propose Graph-of-Thought (GoT) reasoning, which models human thought processes not only as a chain but also as a graph. By representing thought units as nodes and connections between them as edges, our approach captures the non-sequential nature of human thinking and allows for a more realistic modeling of thought processes. GoT adopts a two-stage framework with an additional GoT encoder for thought graph representation and fuses the graph representation with the original input representation through a gated fusion mechanism. We evaluate GoT's performance on a text-only reasoning task (AQUA-RAT) and a multimodal reasoning task (ScienceQA). Our model achieves significant improvement over the strong CoT baseline on the AQUA-RAT test set and boosts accuracy from 85.19% to 87.59% using the T5-base model over the state-of-the-art Multimodal-CoT on the ScienceQA test set.
     </details>

422. **RankPrompt: Step-by-Step Comparisons Make Language Models Better Reasoners** [[pdf]](http://arxiv.org/abs/2403.12373) `2024-03-22` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          RankPrompt breaks down the ranking problem into a series of comparisons among diverse responses, leveraging the inherent capabilities of LLMs to generate chains of comparison as contextual exemplars and validate RankPrompt as an effective method for eliciting high-quality feedback from language models.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have achieved impressive performance across various reasoning tasks. However, even state-of-the-art LLMs such as ChatGPT are prone to logical errors during their reasoning processes. Existing solutions, such as deploying task-specific verifiers or voting over multiple reasoning paths, either require extensive human annotations or fail in scenarios with inconsistent responses. To address these challenges, we introduce RankPrompt, a new prompting method that enables LLMs to self-rank their responses without additional resources. RankPrompt breaks down the ranking problem into a series of comparisons among diverse responses, leveraging the inherent capabilities of LLMs to generate chains of comparison as contextual exemplars. Our experiments across 11 arithmetic and commonsense reasoning tasks show that RankPrompt significantly enhances the reasoning performance of ChatGPT and GPT-4, with improvements of up to 13%. Moreover, RankPrompt excels in LLM-based automatic evaluations for open-ended tasks, aligning with human judgments 74% of the time in the AlpacaEval dataset. It also exhibits robustness to variations in response order and consistency. Collectively, our results validate RankPrompt as an effective method for eliciting high-quality feedback from language models.
     </details>

423. **MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems?** [[pdf]](https://arxiv.org/abs/2403.14624v2) `2024-03-21` (55 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The MathVerse benchmark is introduced, an all-around visual math benchmark designed for an equitable and in-depth evaluation of MLLMs, and a Chain-of-Thought evaluation strategy is proposed for a fine-grained assessment of the output answers.
     </details>


     <details>
          <summary>Abstract</summary>
          The remarkable progress of Multi-modal Large Language Models (MLLMs) has garnered unparalleled attention, due to their superior performance in visual contexts. However, their capabilities in visual math problem-solving remain insufficiently evaluated and understood. We investigate current benchmarks to incorporate excessive visual content within textual questions, which potentially assist MLLMs in deducing answers without truly interpreting the input diagrams. To this end, we introduce MathVerse, an all-around visual math benchmark designed for an equitable and in-depth evaluation of MLLMs. We meticulously collect 2,612 high-quality, multi-subject math problems with diagrams from publicly available sources. Each problem is then transformed by human annotators into six distinct versions, each offering varying degrees of information content in multi-modality, contributing to 15K test samples in total. This approach allows MathVerse to comprehensively assess whether and how much MLLMs can truly understand the visual diagrams for mathematical reasoning. In addition, we propose a Chain-of-Thought (CoT) evaluation strategy for a fine-grained assessment of the output answers. Rather than naively judging True or False, we employ GPT-4(V) to adaptively extract crucial reasoning steps, and then score each step with detailed error analysis, which can reveal the intermediate CoT reasoning quality by MLLMs. We hope the MathVerse benchmark may provide unique insights to guide the future development of MLLMs. Project page: https://mathverse-cuhk.github.io
     </details>

424. **Reinforcement Learning from Reflective Feedback (RLRF): Aligning and Improving LLMs via Fine-Grained Self-Reflection** [[pdf]](https://arxiv.org/abs/2403.14238v1) `2024-03-21` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel framework: Reinforcement Learning from Reflective Feedback (RLRF), which leverages fine-grained feedback based on detailed criteria to improve the core capabilities of LLMs beyond superficial surface-level adjustment is proposed.
     </details>


     <details>
          <summary>Abstract</summary>
          Despite the promise of RLHF in aligning LLMs with human preferences, it often leads to superficial alignment, prioritizing stylistic changes over improving downstream performance of LLMs. Underspecified preferences could obscure directions to align the models. Lacking exploration restricts identification of desirable outputs to improve the models. To overcome these challenges, we propose a novel framework: Reinforcement Learning from Reflective Feedback (RLRF), which leverages fine-grained feedback based on detailed criteria to improve the core capabilities of LLMs. RLRF employs a self-reflection mechanism to systematically explore and refine LLM responses, then fine-tuning the models via a RL algorithm along with promising responses. Our experiments across Just-Eval, Factuality, and Mathematical Reasoning demonstrate the efficacy and transformative potential of RLRF beyond superficial surface-level adjustment.
     </details>

425. **From Large to Tiny: Distilling and Refining Mathematical Expertise for Math Word Problems with Weakly Supervision** [[pdf]](http://arxiv.org/abs/2403.14390) `2024-03-21` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          An innovative two-stage framework that adeptly transfers mathematical Expertise from large to tiny language models is introduced that demonstrates significantly improved performance on the Math23K and Weak12K datasets compared to existing small model methods, while maintaining a much lower computational cost than ChatGPT.
     </details>


     <details>
          <summary>Abstract</summary>
          Addressing the challenge of high annotation costs in solving Math Word Problems (MWPs) through full supervision with intermediate equations, recent works have proposed weakly supervised task settings that rely solely on the final answer as a supervised signal. Existing leading approaches typically employ various search techniques to infer intermediate equations, but cannot ensure their semantic consistency with natural language descriptions. The rise of Large Language Models (LLMs) like ChatGPT has opened up new possibilities for addressing MWPs directly. However, the computational demands of LLMs make them less than ideal for use in settings where resources are tight. In light of these challenges, we introduce an innovative two-stage framework that adeptly transfers mathematical Expertise from large to tiny language models. In \emph{Distillation Stage}, we propose a series of extraction processes that satisfy the properties of MWPs to distill mathematical knowledge from LLMs to construct problem-equation pairs required for supervised training. In \emph{Refinement Stage}, Due to Knowledge distilling method cannot guarantee the full utilization of all data, we further utilize the unsuccessfully searched data effectively by Knowledge Refine method. Finally, We train a small model using distilled data generated through two-stage methods. As our method fully leverages the semantic understanding capabilities during the searching 'problem-equation' pair, it demonstrates significantly improved performance on the Math23K and Weak12K datasets compared to existing small model methods, while maintaining a much lower computational cost than ChatGPT.
     </details>

426. **A Semantic Search Engine for Mathlib4** [[pdf]](http://arxiv.org/abs/2403.13310) `2024-03-20` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper presents a semantic search engine for mathlib4 that accepts informal queries and finds the relevant theorems, and establishes a benchmark for assessing the performance of various search engines for mathlib4.
     </details>


     <details>
          <summary>Abstract</summary>
          The interactive theorem prover, Lean, enables the verification of formal mathematical proofs and is backed by an expanding community. Central to this ecosystem is its mathematical library, mathlib4, which lays the groundwork for the formalization of an expanding range of mathematical theories. However, searching for theorems in mathlib4 can be challenging. To successfully search in mathlib4, users often need to be familiar with its naming conventions or documentation strings. Therefore, creating a semantic search engine that can be used easily by individuals with varying familiarity with mathlib4 is very important. In this paper, we present a semantic search engine for mathlib4 that accepts informal queries and finds the relevant theorems. We also establish a benchmark for assessing the performance of various search engines for mathlib4.
     </details>

427. **Arithmetic Feature Interaction Is Necessary for Deep Tabular Learning** [[pdf]](http://arxiv.org/abs/2402.02334) `AAAI 2024 Machine Learning` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Results show that AMFormer outperforms strong counterparts in fine-grained tabular data modeling, data efficiency in training, and generalization, suggesting it has established a strong inductive bias for deep learning on tabular data.
     </details>


     <details>
          <summary>Abstract</summary>
          Until recently, the question of the effective inductive bias of deep models on tabular data has remained unanswered. This paper investigates the hypothesis that arithmetic feature interaction is necessary for deep tabular learning. To test this point, we create a synthetic tabular dataset with a mild feature interaction assumption and examine a modified transformer architecture enabling arithmetical feature interactions, referred to as AMFormer. Results show that AMFormer outperforms strong counterparts in fine-grained tabular data modeling, data efficiency in training, and generalization. This is attributed to its parallel additive and multiplicative attention operators and prompt-based optimization, which facilitate the separation of tabular samples in an extended space with arithmetically-engineered features. Our extensive experiments on real-world data also validate the consistent effectiveness, efficiency, and rationale of AMFormer, suggesting it has established a strong inductive bias for deep learning on tabular data. Code is available at https://github.com/aigc-apps/AMFormer.
     </details>

428. **Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking** [[pdf]](http://arxiv.org/abs/2403.09629) `COLM 2024` (25 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Quiet-STaR is presented, a generalization of STaR in which LMs learn to generate rationales at each token to explain future text, improving their predictions and improving the LM's ability to directly answer difficult questions.
     </details>


     <details>
          <summary>Abstract</summary>
          When writing and talking, people sometimes pause to think. Although reasoning-focused works have often framed reasoning as a method of answering questions or completing agentic tasks, reasoning is implicit in almost all written text. For example, this applies to the steps not stated between the lines of a proof or to the theory of mind underlying a conversation. In the Self-Taught Reasoner (STaR, Zelikman et al. 2022), useful thinking is learned by inferring rationales from few-shot examples in question-answering and learning from those that lead to a correct answer. This is a highly constrained setting -- ideally, a language model could instead learn to infer unstated rationales in arbitrary text. We present Quiet-STaR, a generalization of STaR in which LMs learn to generate rationales at each token to explain future text, improving their predictions. We address key challenges, including 1) the computational cost of generating continuations, 2) the fact that the LM does not initially know how to generate or use internal thoughts, and 3) the need to predict beyond individual next tokens. To resolve these, we propose a tokenwise parallel sampling algorithm, using learnable tokens indicating a thought's start and end, and an extended teacher-forcing technique. Encouragingly, generated rationales disproportionately help model difficult-to-predict tokens and improve the LM's ability to directly answer difficult questions. In particular, after continued pretraining of an LM on a corpus of internet text with Quiet-STaR, we find zero-shot improvements on GSM8K (5.9%$\rightarrow$10.9%) and CommonsenseQA (36.3%$\rightarrow$47.2%) and observe a perplexity improvement of difficult tokens in natural text. Crucially, these improvements require no fine-tuning on these tasks. Quiet-STaR marks a step towards LMs that can learn to reason in a more general and scalable way.
     </details>

429. **Self-Consistency Boosts Calibration for Math Reasoning** [[pdf]](http://arxiv.org/abs/2403.09849) `2024-03-14` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Three off-the-shelf calibration methods based on self-consistency for math reasoning tasks better bridge model confidence and accuracy than existing methods based on p(True) (Kadavath et al., 2022) or logit (Kadavath et al., 2022).
     </details>


     <details>
          <summary>Abstract</summary>
          Calibration, which establishes the correlation between accuracy and model confidence, is important for LLM development. We design three off-the-shelf calibration methods based on self-consistency (Wang et al., 2022) for math reasoning tasks. Evaluation on two popular benchmarks (GSM8K and MathQA) using strong open-source LLMs (Mistral and LLaMA2), our methods better bridge model confidence and accuracy than existing methods based on p(True) (Kadavath et al., 2022) or logit (Kadavath et al., 2022).
     </details>

430. **Incorporating Graph Attention Mechanism into Geometric Problem Solving Based on Deep Reinforcement Learning** [[pdf]](https://arxiv.org/abs/2403.14690v1) `2024-03-14` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel algorithm, named Automatically Adding Auxiliary Components using Reinforcement Learning framework (A3C-RL), is proposed by forcing an agent to select top strategies, which incorporates the AttnStrategy and BERT as the memory components.
     </details>


     <details>
          <summary>Abstract</summary>
          In the context of online education, designing an automatic solver for geometric problems has been considered a crucial step towards general math Artificial Intelligence (AI), empowered by natural language understanding and traditional logical inference. In most instances, problems are addressed by adding auxiliary components such as lines or points. However, adding auxiliary components automatically is challenging due to the complexity in selecting suitable auxiliary components especially when pivotal decisions have to be made. The state-of-the-art performance has been achieved by exhausting all possible strategies from the category library to identify the one with the maximum likelihood. However, an extensive strategy search have to be applied to trade accuracy for ef-ficiency. To add auxiliary components automatically and efficiently, we present deep reinforcement learning framework based on the language model, such as BERT. We firstly apply the graph attention mechanism to reduce the strategy searching space, called AttnStrategy, which only focus on the conclusion-related components. Meanwhile, a novel algorithm, named Automatically Adding Auxiliary Components using Reinforcement Learning framework (A3C-RL), is proposed by forcing an agent to select top strategies, which incorporates the AttnStrategy and BERT as the memory components. Results from extensive experiments show that the proposed A3C-RL algorithm can substantially enhance the average precision by 32.7% compared to the traditional MCTS. In addition, the A3C-RL algorithm outperforms humans on the geometric questions from the annual University Entrance Mathematical Examination of China.
     </details>

431. **Laying the Foundation First? Investigating the Generalization from Atomic Skills to Complex Reasoning Tasks** [[pdf]](https://arxiv.org/abs/2403.09479v1) `2024-03-14` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          By leveraging hierarchical curriculum learning, this work successfully induce generalization, significantly improve the performance of open-source LMs on complex reasoning tasks, and offers valuable guidance for designing better training strategies for complex reasoning tasks.
     </details>


     <details>
          <summary>Abstract</summary>
          Current language models have demonstrated their capability to develop basic reasoning, but struggle in more complicated reasoning tasks that require a combination of atomic skills, such as math word problem requiring skills like arithmetic and unit conversion. Previous methods either do not improve the inherent atomic skills of models or not attempt to generalize the atomic skills to complex reasoning tasks. In this paper, we first propose a probing framework to investigate whether the atomic skill can spontaneously generalize to complex reasoning tasks. Then, we introduce a hierarchical curriculum learning training strategy to achieve better skill generalization. In our experiments, we find that atomic skills can not spontaneously generalize to compositional tasks. By leveraging hierarchical curriculum learning, we successfully induce generalization, significantly improve the performance of open-source LMs on complex reasoning tasks. Promisingly, the skill generalization exhibit effective in cross-dataset and cross-domain scenarios. Complex reasoning can also help enhance atomic skills. Our findings offer valuable guidance for designing better training strategies for complex reasoning tasks.
     </details>

432. **APOLLO: An Optimized Training Approach for Long-form Numerical Reasoning** [[pdf]](http://arxiv.org/abs/2212.07249) `2024-03-12` (5 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          APOLLO includes a number-aware negative sampling strategy for the retriever to discriminate key numerical facts, and a consistency-based reinforcement learning with target program augmentation for the generator to ultimately increase the execution accuracy.
     </details>


     <details>
          <summary>Abstract</summary>
          Long-form numerical reasoning in financial analysis aims to generate a reasoning program to calculate the correct answer for a given question. Previous work followed a retriever-generator framework, where the retriever selects key facts from a long-form document, and the generator generates a reasoning program based on retrieved facts. However, they treated all facts equally without considering the different contributions of facts with and without numbers. Meanwhile, the program consistency were ignored under supervised training, resulting in lower training accuracy and diversity. To solve these problems, we proposed APOLLO to improve the long-form numerical reasoning framework. For the retriever, we adopt a number-aware negative sampling strategy to enable the retriever to be more discriminative on key numerical facts. For the generator, we design consistency-based reinforcement learning and target program augmentation strategy based on the consistency of program execution results. Experimental results on the FinQA and ConvFinQA leaderboard verify the effectiveness of our proposed method, achieving the new state-of-the-art.
     </details>

433. **FineMath: A Fine-Grained Mathematical Evaluation Benchmark for Chinese Large Language Models** [[pdf]](http://arxiv.org/abs/2403.07747) `2024-03-12` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The proposed FineMath is a fine-grained mathematical evaluation benchmark dataset for assessing Chinese LLMs, created to cover the major key mathematical concepts taught in elementary school math, and finds that there is still considerable room for improvements in terms of mathematical reasoning capability of Chinese LLMs.
     </details>


     <details>
          <summary>Abstract</summary>
          To thoroughly assess the mathematical reasoning abilities of Large Language Models (LLMs), we need to carefully curate evaluation datasets covering diverse mathematical concepts and mathematical problems at different difficulty levels. In pursuit of this objective, we propose FineMath in this paper, a fine-grained mathematical evaluation benchmark dataset for assessing Chinese LLMs. FineMath is created to cover the major key mathematical concepts taught in elementary school math, which are further divided into 17 categories of math word problems, enabling in-depth analysis of mathematical reasoning abilities of LLMs. All the 17 categories of math word problems are manually annotated with their difficulty levels according to the number of reasoning steps required to solve these problems. We conduct extensive experiments on a wide range of LLMs on FineMath and find that there is still considerable room for improvements in terms of mathematical reasoning capability of Chinese LLMs. We also carry out an in-depth analysis on the evaluation process and methods that have been overlooked previously. These two factors significantly influence the model results and our understanding of their mathematical reasoning capabilities. The dataset will be publicly available soon.
     </details>

434. **Reverse That Number! Decoding Order Matters in Arithmetic Learning** [[pdf]](http://arxiv.org/abs/2403.05845) `2024-03-09` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces a novel strategy that not only reevaluates the digit order by prioritizing output from the least significant digit but also incorporates a step-by-step methodology to substantially reduce complexity.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent advancements in pretraining have demonstrated that modern Large Language Models (LLMs) possess the capability to effectively learn arithmetic operations. However, despite acknowledging the significance of digit order in arithmetic computation, current methodologies predominantly rely on sequential, step-by-step approaches for teaching LLMs arithmetic, resulting in a conclusion where obtaining better performance involves fine-grained step-by-step. Diverging from this conventional path, our work introduces a novel strategy that not only reevaluates the digit order by prioritizing output from the least significant digit but also incorporates a step-by-step methodology to substantially reduce complexity. We have developed and applied this method in a comprehensive set of experiments. Compared to the previous state-of-the-art (SOTA) method, our findings reveal an overall improvement of in accuracy while requiring only a third of the tokens typically used during training. For the purpose of facilitating replication and further research, we have made our code and dataset publicly available at \url{https://anonymous.4open.science/r/RAIT-9FB7/}.
     </details>

435. **RAT: Retrieval Augmented Thoughts Elicit Context-Aware Reasoning in Long-Horizon Generation** [[pdf]](http://arxiv.org/abs/2403.05313) `2024-03-08` (20 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work explores how iterative revising a chain of thoughts with the help of information retrieval significantly improves large language models' reasoning and generation ability in long-horizon generation tasks, while hugely mitigating hallucination.
     </details>


     <details>
          <summary>Abstract</summary>
          We explore how iterative revising a chain of thoughts with the help of information retrieval significantly improves large language models' reasoning and generation ability in long-horizon generation tasks, while hugely mitigating hallucination. In particular, the proposed method -- *retrieval-augmented thoughts* (RAT) -- revises each thought step one by one with retrieved information relevant to the task query, the current and the past thought steps, after the initial zero-shot CoT is generated. Applying RAT to GPT-3.5, GPT-4, and CodeLLaMA-7b substantially improves their performances on various long-horizon generation tasks; on average of relatively increasing rating scores by 13.63% on code generation, 16.96% on mathematical reasoning, 19.2% on creative writing, and 42.78% on embodied task planning. The demo page can be found at https://craftjarvis.github.io/RAT
     </details>

436. **Common 7B Language Models Already Possess Strong Math Capabilities** [[pdf]](http://arxiv.org/abs/2403.04706) `ICLR 2025 Submission` (33 cite) (3 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper shows that the LLaMA-2 7B model with common pre-training already exhibits strong mathematical abilities, as evidenced by its impressive accuracy when selecting the best response from 256 random generations, and finds that simply scaling up the SFT data can significantly enhance the reliability of generating correct answers.
     </details>


     <details>
          <summary>Abstract</summary>
          Mathematical capabilities were previously believed to emerge in common language models only at a very large scale or require extensive math-related pre-training. This paper shows that the LLaMA-2 7B model with common pre-training already exhibits strong mathematical abilities, as evidenced by its impressive accuracy of 97.7% and 72.0% on the GSM8K and MATH benchmarks, respectively, when selecting the best response from 256 random generations. The primary issue with the current base model is the difficulty in consistently eliciting its inherent mathematical capabilities. Notably, the accuracy for the first answer drops to 49.5% and 7.9% on the GSM8K and MATH benchmarks, respectively. We find that simply scaling up the SFT data can significantly enhance the reliability of generating correct answers. However, the potential for extensive scaling is constrained by the scarcity of publicly available math questions. To overcome this limitation, we employ synthetic data, which proves to be nearly as effective as real data and shows no clear saturation when scaled up to approximately one million samples. This straightforward approach achieves an accuracy of 82.6% on GSM8K and 40.6% on MATH using LLaMA-2 7B models, surpassing previous models by 14.2% and 20.8%, respectively. We also provide insights into scaling behaviors across different reasoning complexities and error types.
     </details>

437. **Machine learning and information theory concepts towards an AI Mathematician** [[pdf]](http://arxiv.org/abs/2403.04571) `2024-03-07` (6 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This essay builds on the idea that current deep learning mostly succeeds at system 1 abilities—which correspond to the authors' intuition and habitual behaviors—but still lacks something important regarding system 2 abilities—which include reasoning and robust uncertainty estimation.
     </details>


     <details>
          <summary>Abstract</summary>
          The current state-of-the-art in artificial intelligence is impressive, especially in terms of mastery of language, but not so much in terms of mathematical reasoning. What could be missing? Can we learn something useful about that gap from how the brains of mathematicians go about their craft? This essay builds on the idea that current deep learning mostly succeeds at system 1 abilities -- which correspond to our intuition and habitual behaviors -- but still lacks something important regarding system 2 abilities -- which include reasoning and robust uncertainty estimation. It takes an information-theoretical posture to ask questions about what constitutes an interesting mathematical statement, which could guide future work in crafting an AI mathematician. The focus is not on proving a given theorem but on discovering new and interesting conjectures. The central hypothesis is that a desirable body of theorems better summarizes the set of all provable statements, for example by having a small description length while at the same time being close (in terms of number of derivation steps) to many provable statements.
     </details>

438. **Benchmarking Hallucination in Large Language Models based on Unanswerable Math Word Problem** [[pdf]](http://arxiv.org/abs/2403.03558) `2024-03-06` (5 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is shown that utilizing MWP is a reliable and effective approach to assess hallucination and in-context learning and reinforcement learning with human feedback (RLHF) training significantly enhance the model’s ability to avoid hallucination.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) are highly effective in various natural language processing (NLP) tasks. However, they are susceptible to producing unreliable conjectures in ambiguous contexts called hallucination. This paper presents a new method for evaluating LLM hallucination in Question Answering (QA) based on the unanswerable math word problem (MWP). To support this approach, we innovatively develop a dataset called Unanswerable Math Word Problem (UMWP) which comprises 5200 questions across five categories. We developed an evaluation methodology combining text similarity and mathematical expression detection to determine whether LLM considers the question unanswerable. The results of extensive experiments conducted on 31 LLMs, including GPT-3, InstructGPT, LLaMA, and Claude, demonstrate that in-context learning and reinforcement learning with human feedback (RLHF) training significantly enhance the model's ability to avoid hallucination. We show that utilizing MWP is a reliable and effective approach to assess hallucination. Our code and data are available at https://github.com/Yuki-Asuuna/UMWP.
     </details>

439. **Exploring the Limitations of Large Language Models in Compositional Relation Reasoning** [[pdf]](http://arxiv.org/abs/2403.02615) `2024-03-04` (4 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The Multilingual Composition Relation (MCR) benchmark aims at investigating the robustness and adaptability of LLMs in handling composition relation reasoning across diverse linguistic contexts.
     </details>


     <details>
          <summary>Abstract</summary>
          We present a comprehensive evaluation of large language models(LLMs)' ability to reason about composition relations through a benchmark encompassing 1,500 test cases in English, designed to cover six distinct types of composition relations: Positional, Comparative, Personal, Mathematical, Identity, and Other. Acknowledging the significance of multilingual capabilities, we expanded our assessment to include translations of these cases into Chinese, Japanese, French, and Korean. Our Multilingual Composition Relation (MCR) benchmark aims at investigating the robustness and adaptability of LLMs in handling composition relation reasoning across diverse linguistic contexts.
     </details>

440. **Functional Benchmarks for Robust Evaluation of Reasoning Performance, and the Reasoning Gap** [[pdf]](http://arxiv.org/abs/2402.19450) `2024-02-29` (22 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Here it is shown that models which anecdotally have good reasoning performance over real-world tasks, have quantifiable lower gaps, motivating the open problem of building"gap 0"models.
     </details>


     <details>
          <summary>Abstract</summary>
          We propose a framework for robust evaluation of reasoning capabilities of language models, using functional variants of benchmarks. Models that solve a reasoning test should exhibit no difference in performance over the static version of a problem compared to a snapshot of the functional variant. We have rewritten the relevant fragment of the MATH benchmark into its functional variant MATH(), with functionalization of other benchmarks to follow. When evaluating current state-of-the-art models over snapshots of MATH(), we find a reasoning gap -- the percentage difference between the static and functional accuracies. We find reasoning gaps from 58.35% to 80.31% among the state-of-the-art closed and open weights models that perform well on static benchmarks, with the caveat that the gaps are likely to be smaller with more sophisticated prompting strategies. Here we show that models which anecdotally have good reasoning performance over real-world tasks, have quantifiable lower gaps, motivating the open problem of building "gap 0" models. Code for evaluation and new evaluation datasets, three MATH() snapshots, are publicly available at https://github.com/consequentai/fneval/.
     </details>

441. **Automatic Prompt Augmentation and Selection with Chain-of-Thought from Labeled Data** [[pdf]](http://arxiv.org/abs/2302.12822) `EMNLP 2023 Findings` (99 cite) (4 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A new strategy, Automate-CoT (Automatic Prompt Augmentation and Selection with Chain-of-Thought), that can bypass human engineering of CoT by automatically augmenting rational chains from a small labeled dataset, and then pruning low-quality chains to construct a candidate pool of machine-generated rationale chains based on the labels.
     </details>


     <details>
          <summary>Abstract</summary>
          Chain-of-thought (CoT) advances the reasoning abilities of large language models (LLMs) and achieves superior performance in complex reasoning tasks. However, most CoT studies rely on carefully designed human-annotated rational chains to prompt LLMs, posing challenges for real-world applications where labeled data is available without rational chains. This paper proposes a new strategy, AutomateCoT (Automatic Prompt Augmentation and Selection with Chain-of-Thought), that can bypass human engineering of CoT by automatically augmenting rational chains from a small labeled dataset, and then pruning low-quality chains to construct a candidate pool of machinegenerated rationale chains based on the labels. Finally, it selects the optimal combination of several rationale chains from the pool for CoT prompting by employing a variance-reduced policy gradient strategy to estimate the significance of each example. Automate-CoT enables a quick adaptation of the CoT technique to different tasks. Experimental results demonstrate the effectiveness of our method, where competitive results are achieved on arithmetic reasoning (+2.7%), commonsense reasoning (+3.4%), symbolic reasoning (+3.2%), and non-reasoning tasks (+2.5%).
     </details>

442. **Stepwise Self-Consistent Mathematical Reasoning with Large Language Models** [[pdf]](http://arxiv.org/abs/2402.17786) `ICML 2024 Workshop ICL` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces a novel algorithm, namely Stepwise Self-Consistent Chain-of-Thought (SSC-CoT), which employs a strategy of selecting intermediate steps based on the intersection of various reasoning chains and enables the model to discover critical intermediate steps by querying a knowledge graph comprising relevant domain knowledge.
     </details>


     <details>
          <summary>Abstract</summary>
          Using Large Language Models for complex mathematical reasoning is difficult, primarily due to the complexity of multi-step reasoning. The main challenges of this process include (1) selecting critical intermediate results to advance the procedure, and (2) limited exploration of potential solutions. To address these issues, we introduce a novel algorithm, namely Stepwise Self-Consistent Chain-of-Thought (SSC-CoT). SSC-CoT employs a strategy of selecting intermediate steps based on the intersection of various reasoning chains. Additionally, SSC-CoT enables the model to discover critical intermediate steps by querying a knowledge graph comprising relevant domain knowledge. To validate SSC-CoT, we present a new dataset, TriMaster100, tailored for complex trigonometry problems. This dataset contains 100 questions, with each solution broken down into scored intermediate steps, facilitating a comprehensive evaluation of the mathematical reasoning process. On TriMaster100, SSC-CoT triples the effectiveness of the state-of-the-art methods. Furthermore, we benchmark SSC-CoT on the widely recognized complex mathematical question dataset, MATH level 5, and it surpasses the second-best method by 7.2% in accuracy. Code and the TriMaster100 dataset can be found at: https://github.com/zhao-zilong/ssc-cot.
     </details>

443. **ConceptMath: A Bilingual Concept-wise Benchmark for Measuring Mathematical Reasoning of Large Language Models** [[pdf]](http://arxiv.org/abs/2402.14660) `ACL 2024 Findings` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          ConceptMath is introduced, a bilingual (English and Chinese), fine-grained benchmark that evaluates concept-wise mathematical reasoning of Large Language Models (LLMs) and an efficient fine-tuning strategy to enhance the weaknesses of existing LLMs is introduced.
     </details>


     <details>
          <summary>Abstract</summary>
          This paper introduces ConceptMath, a bilingual (English and Chinese), fine-grained benchmark that evaluates concept-wise mathematical reasoning of Large Language Models (LLMs). Unlike traditional benchmarks that evaluate general mathematical reasoning with an average accuracy, ConceptMath systemically organizes math problems under a hierarchy of math concepts, so that mathematical reasoning can be evaluated at different granularity with concept-wise accuracies. Based on our ConcepthMath, we then evaluate a broad range of LLMs, and we observe existing LLMs, though achieving high average accuracies on traditional benchmarks, exhibit significant performance variations across different math concepts and may even fail catastrophically on the most basic ones. Besides, we also introduce an efficient fine-tuning strategy to enhance the weaknesses of existing LLMs. Finally, we hope ConceptMath could guide the developers to understand the fine-grained mathematical abilities of their models and facilitate the growth of foundation models. Code is available at https://github.com/conceptmath/conceptmath.
     </details>

444. **Mathematical Language Models: A Survey** [[pdf]](http://arxiv.org/abs/2312.07622) `2024-02-23` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A comprehensive survey of mathematical LMs is conducted, systematically categorizing pivotal research endeavors from two distinct perspectives: tasks and methodologies, revealing a large number of proposed mathematical LLMs.
     </details>


     <details>
          <summary>Abstract</summary>
          In recent years, there has been remarkable progress in leveraging Language Models (LMs), encompassing Pre-trained Language Models (PLMs) and Large-scale Language Models (LLMs), within the domain of mathematics. This paper conducts a comprehensive survey of mathematical LMs, systematically categorizing pivotal research endeavors from two distinct perspectives: tasks and methodologies. The landscape reveals a large number of proposed mathematical LLMs, which are further delineated into instruction learning, tool-based methods, fundamental CoT techniques, and advanced CoT methodologies. In addition, our survey entails the compilation of over 60 mathematical datasets, including training datasets, benchmark datasets, and augmented datasets. Addressing the primary challenges and delineating future trajectories within the field of mathematical LMs, this survey is positioned as a valuable resource, poised to facilitate and inspire future innovation among researchers invested in advancing this domain.
     </details>

445. **An Empirical Study of Data Ability Boundary in LLMs' Math Reasoning** [[pdf]](http://arxiv.org/abs/2403.00799) `2024-02-23` (0 cite) (1 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) are displaying emergent abilities for math reasoning tasks,and there is a growing attention on enhancing the ability of open-source LLMs through supervised fine-tuning (SFT).In this paper, we aim to explore a general data strategy for supervised data to help optimize and expand math reasoning ability.Firstly, we determine the ability boundary of reasoning paths augmentation by identifying these paths' minimal optimal set.Secondly, we validate that different abilities of the model can be cumulatively enhanced by Mix of Minimal Optimal Sets of corresponding types of data, while our models MMOS achieve SOTA performance on series base models under much lower construction costs.Besides, we point out GSM-HARD is not really hard and today's LLMs no longer lack numerical robustness.Also, we provide an Auto Problem Generator for robustness testing and educational applications.Our code and data are publicly available at https://github.com/cyzhh/MMOS.
     </details>

446. **Brain-Inspired Two-Stage Approach: Enhancing Mathematical Reasoning by Imitating Human Thought Processes** [[pdf]](http://arxiv.org/abs/2403.00800) `2024-02-23` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel approach, named Brain, to imitate human thought processes to enhance mathematical reasoning abilities, using the Frontal Lobe Model to generate plans, and then employing the Parietal Lobe Model to generate code and execute to obtain answers.
     </details>


     <details>
          <summary>Abstract</summary>
          Although large language models demonstrate emergent abilities in solving math word problems, there is a challenging task in complex multi-step mathematical reasoning tasks. To improve model performance on mathematical reasoning tasks, previous work has conducted supervised fine-tuning on open-source models by improving the quality and quantity of data. In this paper, we propose a novel approach, named Brain, to imitate human thought processes to enhance mathematical reasoning abilities, using the Frontal Lobe Model to generate plans, and then employing the Parietal Lobe Model to generate code and execute to obtain answers. First, we achieve SOTA performance in comparison with Code LLaMA 7B based models through this method. Secondly, we find that plans can be explicitly extracted from natural language, code, or formal language. Our code and data are publicly available at https://github.com/cyzhh/Brain.
     </details>

447. **Diversity of Thought Improves Reasoning Abilities of LLMs** [[pdf]](http://arxiv.org/abs/2310.07088) `2024-02-23` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a method that automatically improves prompt diversity by soliciting feedback from the LLM to ideate approaches that are apt for the problem, and improves the Pareto frontier of the accuracy-cost trade-off.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) are documented to struggle in settings that require complex reasoning. Nevertheless, instructing the model to break down the problem into smaller reasoning steps, or ensembling various generations through modifying decoding steps boosts performance. However, these methods assume that the input prompt is fixed and expect the decoding strategies to introduce the diversity needed for ensembling. In this work, we discuss how one can create and leverage variations of the input prompt as a means of diversity of thought. We propose a method that automatically improves prompt diversity by soliciting feedback from the LLM to ideate approaches that are apt for the problem. We then ensemble the diverse prompts in our method DIVSE (DIVerse reasoning path Self-Ensemble) across multiple inference calls, or use diverse approaches within a single inference call; we call the latter IDIV-SE (In-call DIVerse reasoning path Self-Ensemble). Apart from our approaches outperforming prior work, DIV-SE(in particular) advances state-of-the-art performance on the challenging planning and graph coloring benchmarks. Our results improve the Pareto frontier of the accuracy-cost trade-off.
     </details>

448. **RevOrder: A Novel Method for Enhanced Arithmetic in Language Models** [[pdf]](http://arxiv.org/abs/2402.03822) `2024-02-23` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          RevOrder, a novel technique aimed at improving arithmetic operations in large language models (LLMs) by reversing the output digits in addition, subtraction, and n-digit by 1-digit (nD by 1D) multiplication tasks, significantly reduces the Count of Sequential Intermediate Digits to $\mathcal{O}(1)$, a new metric to assess equation complexity.
     </details>


     <details>
          <summary>Abstract</summary>
          This paper presents RevOrder, a novel technique aimed at improving arithmetic operations in large language models (LLMs) by reversing the output digits in addition, subtraction, and n-digit by 1-digit (nD by 1D) multiplication tasks. Our method significantly reduces the Count of Sequential Intermediate Digits (CSID) to $\mathcal{O}(1)$, a new metric we introduce to assess equation complexity. Through comprehensive testing, RevOrder not only achieves perfect accuracy in basic arithmetic operations but also substantially boosts LLM performance in division tasks, particularly with large numbers where traditional models struggle. Implementation of RevOrder is cost-effective for both training and inference phases. Moreover, applying RevOrder to fine-tune the LLaMA2-7B model on the GSM8K math task results in a considerable improvement, reducing equation calculation errors by 46% and increasing overall scores from 41.6 to 44.4.
     </details>

449. **Tokenization counts: the impact of tokenization on arithmetic in frontier LLMs** [[pdf]](http://arxiv.org/abs/2402.14903) `2024-02-22` (24 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work performs the first study of how number tokenization choices lead to differences in model performance on arithmetic tasks, accompanied by a thorough analysis of error patterns.
     </details>


     <details>
          <summary>Abstract</summary>
          Tokenization, the division of input text into input tokens, is an often overlooked aspect of the large language model (LLM) pipeline and could be the source of useful or harmful inductive biases. Historically, LLMs have relied on byte pair encoding, without care to specific input domains. With the increased use of LLMs for reasoning, various number-specific tokenization schemes have been adopted, with popular models like LLaMa and PaLM opting for single-digit tokenization while GPT-3.5 and GPT-4 have separate tokens for each 1-, 2-, and 3-digit numbers. In this work, we study the effect this choice has on numerical reasoning through the use of arithmetic tasks. We consider left-to-right and right-to-left tokenization for GPT-3.5 and -4, finding that right-to-left tokenization (enforced by comma separating numbers at inference time) leads to largely improved performance. Furthermore, we find that model errors when using standard left-to-right tokenization follow stereotyped error patterns, suggesting that model computations are systematic rather than approximate. We show that the model is able to convert between tokenizations easily, thus allowing chain-of-thought-inspired approaches to recover performance on left-to-right tokenized inputs. We also find the gap between tokenization directions decreases when models are scaled, possibly indicating that larger models are better able to override this tokenization-dependent inductive bias. In summary, our work performs the first study of how number tokenization choices lead to differences in model performance on arithmetic tasks, accompanied by a thorough analysis of error patterns. We hope this work inspires practitioners to more carefully ablate number tokenization-related choices when working towards general models of numerical reasoning.
     </details>

450. **SciAgent: Tool-augmented Language Models for Scientific Reasoning** [[pdf]](http://arxiv.org/abs/2402.11451) `2024-02-20` (8 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work develops SciAgent to retrieve, understand and, if necessary, use tools for scientific problem solving, andCrafts a benchmark, SciToolBench, spanning five scientific domains to evaluate LLMs' abilities with tool assistance.
     </details>


     <details>
          <summary>Abstract</summary>
          Scientific reasoning poses an excessive challenge for even the most advanced Large Language Models (LLMs). To make this task more practical and solvable for LLMs, we introduce a new task setting named tool-augmented scientific reasoning. This setting supplements LLMs with scalable toolsets, and shifts the focus from pursuing an omniscient problem solver to a proficient tool-user. To facilitate the research of such setting, we construct a tool-augmented training corpus named MathFunc which encompasses over 30,000 samples and roughly 6,000 tools. Building on MathFunc, we develop SciAgent to retrieve, understand and, if necessary, use tools for scientific problem solving. Additionally, we craft a benchmark, SciToolBench, spanning five scientific domains to evaluate LLMs' abilities with tool assistance. Extensive experiments on SciToolBench confirm the effectiveness of SciAgent. Notably, SciAgent-Mistral-7B surpasses other LLMs with the same size by more than 13% in absolute accuracy. Furthermore, SciAgent-DeepMath-7B shows much superior performance than ChatGPT.
     </details>

451. **SymBa: Symbolic Backward Chaining for Structured Natural Language Reasoning** [[pdf]](https://arxiv.org/abs/2402.12806v2) `2024-02-20` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a novel backward chaining framework, SymBa (Symbolic Backward Chaining), in which a symbolic solver controls the whole proof process, and an LLM searches for the relevant natural language premises and translates them into a symbolic form for the solver.
     </details>


     <details>
          <summary>Abstract</summary>
          To improve the performance and explainability of LLM-based natural language reasoning, structured reasoning can be applied to generate explicitly structured proofs. Among different methods for structured reasoning, we specifically focus on backward chaining, where the proof goal is recursively decomposed to subgoals by searching and applying rules. We argue that current LLM-based backward chaining systems (e.g. Least-to-most prompting and LAMBADA) are incomplete, as they omit crucial algorithmic components identified from the classic backward chaining algorithm (SLD Resolution) in computational logic. To this end, we propose a novel backward chaining system, SymBa (Symbolic Backward Chaining), which integrates a symbolic solver and an LLM. In SymBa, the solver controls the proof process, and the LLM is only called when the solver requires new information to complete the proof. Empowered by completeness, SymBa achieves a significant improvement in deductive, relational, and arithmetic reasoning benchmarks compared to the baselines.
     </details>

452. **Can LLMs Compute with Reasons?** [[pdf]](http://arxiv.org/abs/2402.12080) `2024-02-19` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The goal is to provide a framework that empowers SLMs to approach the level of logic-based applications achieved by high-parameter models, potentially benefiting any language model.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) often struggle with complex mathematical tasks, prone to "hallucinating" incorrect answers due to their reliance on statistical patterns. This limitation is further amplified in average Small LangSLMs with limited context and training data. To address this challenge, we propose an "Inductive Learning" approach utilizing a distributed network of SLMs. This network leverages error-based learning and hint incorporation to refine the reasoning capabilities of SLMs. Our goal is to provide a framework that empowers SLMs to approach the level of logic-based applications achieved by high-parameter models, potentially benefiting any language model. Ultimately, this novel concept paves the way for bridging the logical gap between humans and LLMs across various fields.
     </details>

453. **Learning From Failure: Integrating Negative Examples when Fine-tuning Large Language Models as Agents** [[pdf]](https://arxiv.org/abs/2402.11651v2) `2024-02-18` (6 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is argued that unsuccessful trajectories offer valuable insights, and LLMs can learn from these trajectories through appropriate quality control and fine-tuning strategies, and the first to demonstrate the value of negative trajectories and their application in agent-tunning scenarios is demonstrated.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have achieved success in acting as agents, which interact with environments through tools such as search engines. However, LLMs are optimized for language generation instead of tool use during training or alignment, limiting their effectiveness as agents. To resolve this problem, previous work has first collected interaction trajectories between LLMs and environments, using only trajectories that successfully finished the task to fine-tune smaller models, making fine-tuning data scarce and acquiring it both difficult and costly. Discarding failed trajectories also leads to significant wastage of data and resources and limits the possible optimization paths during fine-tuning. In this paper, we argue that unsuccessful trajectories offer valuable insights, and LLMs can learn from these trajectories through appropriate quality control and fine-tuning strategies. By simply adding a prefix or suffix that tells the model whether to generate a successful trajectory during training, we improve model performance by a large margin on mathematical reasoning, multi-hop question answering, and strategic question answering tasks. We further analyze the inference results and find that our method provides a better trade-off between valuable information and errors in unsuccessful trajectories. To our knowledge, we are the first to demonstrate the value of negative trajectories and their application in agent-tunning scenarios. Our findings offer guidance for developing better agent-tuning methods and low-resource data usage techniques.
     </details>

454. **Orca-Math: Unlocking the potential of SLMs in Grade School Math** [[pdf]](http://arxiv.org/abs/2402.14830) `2024-02-16` (31 cite) (3 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Mathematical word problem-solving has long been recognized as a complex task for small language models (SLMs). A recent study hypothesized that the smallest model size, needed to achieve over 80% accuracy on the GSM8K benchmark, is 34 billion parameters. To reach this level of performance with smaller models, researcher often train SLMs to generate Python code or use tools to help avoid calculation errors. Additionally, they employ ensembling, where outputs of up to 100 model runs are combined to arrive at a more accurate result. Result selection is done using consensus, majority vote or a separate a verifier model used in conjunction with the SLM. Ensembling provides a substantial boost in accuracy but at a significant cost increase with multiple calls to the model (e.g., Phi-GSM uses top-48 to boost the performance from 68.2 to 81.5).   In this work, we present Orca-Math, a 7-billion-parameter SLM based on the Mistral-7B, which achieves 86.81% on GSM8k without the need for multiple model calls or the use of verifiers, code execution or any other external tools. Our approach has the following key elements: (1) A high quality synthetic dataset of 200K math problems created using a multi-agent setup where agents collaborate to create the data, (2) An iterative learning techniques that enables the SLM to practice solving problems, receive feedback on its solutions and learn from preference pairs incorporating the SLM solutions and the feedback. When trained with Supervised Fine-Tuning alone, Orca-Math achieves 81.50% on GSM8k pass@1 metric. With iterative preference learning, Orca-Math achieves 86.81% pass@1. Orca-Math surpasses the performance of significantly larger models such as LLAMA-2-70B, WizardMath-70B, Gemini-Pro, ChatGPT-3.5. It also significantly outperforms other smaller models while using much smaller data (hundreds of thousands vs. millions of problems).
     </details>

455. **AutoTutor meets Large Language Models: A Language Model Tutor with Rich Pedagogy and Guardrails** [[pdf]](https://arxiv.org/abs/2402.09216v3) `2024-02-14` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper creates a sample end-to-end tutoring system named MWPTutor, which uses LLMs to fill in the state space of a pre-defined finite state transducer, and shows that this hybrid approach achieves a better overall tutoring score than an instructed, but otherwise free-form, GPT-4.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have found several use cases in education, ranging from automatic question generation to essay evaluation. In this paper, we explore the potential of using Large Language Models (LLMs) to author Intelligent Tutoring Systems. A common pitfall of LLMs is their straying from desired pedagogical strategies such as leaking the answer to the student, and in general, providing no guarantees. We posit that while LLMs with certain guardrails can take the place of subject experts, the overall pedagogical design still needs to be handcrafted for the best learning results. Based on this principle, we create a sample end-to-end tutoring system named MWPTutor, which uses LLMs to fill in the state space of a pre-defined finite state transducer. This approach retains the structure and the pedagogy of traditional tutoring systems that has been developed over the years by learning scientists but brings in additional flexibility of LLM-based approaches. Through a human evaluation study on two datasets based on math word problems, we show that our hybrid approach achieves a better overall tutoring score than an instructed, but otherwise free-form, GPT-4. MWPTutor is completely modular and opens up the scope for the community to improve its performance by improving individual modules or using different teaching strategies that it can follow.
     </details>

456. **FGeo-DRL: Deductive Reasoning for Geometric Problems through Deep Reinforcement Learning** [[pdf]](http://arxiv.org/abs/2402.09051) `2024-02-14` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A neural-symbolic system, named FGeo-DRL, is built to automatically perform human-like geometric deductive reasoning, capable of autonomously learning problem-solving methods from the feedback of a formalized environment, without the need for human supervision.
     </details>


     <details>
          <summary>Abstract</summary>
          The human-like automatic deductive reasoning has always been one of the most challenging open problems in the interdiscipline of mathematics and artificial intelligence. This paper is the third in a series of our works. We built a neural-symbolic system, called FGeoDRL, to automatically perform human-like geometric deductive reasoning. The neural part is an AI agent based on reinforcement learning, capable of autonomously learning problem-solving methods from the feedback of a formalized environment, without the need for human supervision. It leverages a pre-trained natural language model to establish a policy network for theorem selection and employ Monte Carlo Tree Search for heuristic exploration. The symbolic part is a reinforcement learning environment based on geometry formalization theory and FormalGeo, which models GPS as a Markov Decision Process. In this formal symbolic system, the known conditions and objectives of the problem form the state space, while the set of theorems forms the action space. Leveraging FGeoDRL, we have achieved readable and verifiable automated solutions to geometric problems. Experiments conducted on the formalgeo7k dataset have achieved a problem-solving success rate of 86.40%. The project is available at https://github.com/PersonNoName/FGeoDRL.
     </details>

457. **Beyond Lines and Circles: Unveiling the Geometric Reasoning Gap in Large Language Models** [[pdf]](http://arxiv.org/abs/2402.03877) `2024-02-14` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces a framework that formulates an LLMs-based multi-agents system that enhances their existing reasoning potential by conducting an internal dialogue and underscores LLMs' current limitations in geometric reasoning and improves geometric reasoning capabilities through self-correction, collaboration, and diverse role specializations.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) demonstrate ever-increasing abilities in mathematical and algorithmic tasks, yet their geometric reasoning skills are underexplored. We investigate LLMs' abilities in constructive geometric problem-solving one of the most fundamental steps in the development of human mathematical reasoning. Our work reveals notable challenges that the state-of-the-art LLMs face in this domain despite many successes in similar areas. LLMs exhibit biases in target variable selection and struggle with 2D spatial relationships, often misrepresenting and hallucinating objects and their placements. To this end, we introduce a framework that formulates an LLMs-based multi-agents system that enhances their existing reasoning potential by conducting an internal dialogue. This work underscores LLMs' current limitations in geometric reasoning and improves geometric reasoning capabilities through self-correction, collaboration, and diverse role specializations.
     </details>

458. **FGeo-TP: A Language Model-Enhanced Solver for Geometry Problems** [[pdf]](http://arxiv.org/abs/2402.09047) `2024-02-14` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduced FGeo-TP (Theorem Predictor), which utilizes the language model to predict theorem sequences for solving geometry problems, and compared the effectiveness of various Transformer architectures in theorem prediction, implementing pruning in the search process of FGPS, thereby improving its performance in solving geometry problems.
     </details>


     <details>
          <summary>Abstract</summary>
          The application of contemporary artificial intelligence techniques to address geometric problems and automated deductive proof has always been a grand challenge to the interdiscipline field of mathematics and artificial Intelligence. This is the fourth article in a series of our works, in our previous work, we established of a geometric formalized system known as FormalGeo. Moreover we annotated approximately 7000 geometric problems, forming the FormalGeo7k dataset. Despite the FGPS (Formal Geometry Problem Solver) can achieve interpretable algebraic equation solving and human-like deductive reasoning, it often experiences timeouts due to the complexity of the search strategy. In this paper, we introduced FGeo-TP (Theorem Predictor), which utilizes the language model to predict theorem sequences for solving geometry problems. We compared the effectiveness of various Transformer architectures, such as BART or T5, in theorem prediction, implementing pruning in the search process of FGPS, thereby improving its performance in solving geometry problems. Our results demonstrate a significant increase in the problem-solving rate of the language model-enhanced FGeo-TP on the FormalGeo7k dataset, rising from 39.7% to 80.86%. Furthermore, FGeo-TP exhibits notable reductions in solving time and search steps across problems of varying difficulty levels.
     </details>

459. **FormalGeo: An Extensible Formalized Framework for Olympiad Geometric Problem Solving** [[pdf]](http://arxiv.org/abs/2310.18021) `2024-02-14` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper has constructed a consistent formal plane geometry system that will serve as a crucial bridge between IMO-level plane geometry challenges and readable AI automated reasoning, and has been able to seamlessly integrate modern AI models with this formal system.
     </details>


     <details>
          <summary>Abstract</summary>
          This is the first paper in a series of work we have accomplished over the past three years. In this paper, we have constructed a consistent formal plane geometry system. This will serve as a crucial bridge between IMO-level plane geometry challenges and readable AI automated reasoning. Within this formal framework, we have been able to seamlessly integrate modern AI models with our formal system. AI is now capable of providing deductive reasoning solutions to IMO-level plane geometry problems, just like handling other natural languages, and these proofs are readable, traceable, and verifiable. We propose the geometry formalization theory (GFT) to guide the development of the geometry formal system. Based on the GFT, we have established the FormalGeo, which consists of 88 geometric predicates and 196 theorems. It can represent, validate, and solve IMO-level geometry problems. we also have crafted the FGPS (formal geometry problem solver) in Python. It serves as both an interactive assistant for verifying problem-solving processes and an automated problem solver. We've annotated the formalgeo7k and formalgeo-imo datasets. The former contains 6,981 (expand to 133,818 through data augmentation) geometry problems, while the latter includes 18 (expand to 2,627 and continuously increasing) IMO-level challenging geometry problems. All annotated problems include detailed formal language descriptions and solutions. Implementation of the formal system and experiments validate the correctness and utility of the GFT. The backward depth-first search method only yields a 2.42% problem-solving failure rate, and we can incorporate deep learning techniques to achieve lower one. The source code of FGPS and datasets are available at https://github.com/BitSecret/FGPS.
     </details>

460. **InternLM-Math: Open Math Large Language Models Toward Verifiable Reasoning** [[pdf]](http://arxiv.org/abs/2402.06332) `2024-02-09` `Lean` (33 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduces and open-source their math reasoning LLMs InternLM-Math which is continue pre-trained from InternLM2 and unify chain-of-thought reasoning, reward modeling, formal reasoning, data augmentation, and code interpreter in a unified seq2seq format and supervise the authors' model to be a versatile math reasoner, verifier, prover, and augmenter.
     </details>


     <details>
          <summary>Abstract</summary>
          The math abilities of large language models can represent their abstract reasoning ability. In this paper, we introduce and open-source our math reasoning LLMs InternLM-Math which is continue pre-trained from InternLM2. We unify chain-of-thought reasoning, reward modeling, formal reasoning, data augmentation, and code interpreter in a unified seq2seq format and supervise our model to be a versatile math reasoner, verifier, prover, and augmenter. These abilities can be used to develop the next math LLMs or self-iteration. InternLM-Math obtains open-sourced state-of-the-art performance under the setting of in-context learning, supervised fine-tuning, and code-assisted reasoning in various informal and formal benchmarks including GSM8K, MATH, Hungary math exam, MathBench-ZH, and MiniF2F. Our pre-trained model achieves 30.3 on the MiniF2F test set without fine-tuning. We further explore how to use LEAN to solve math problems and study its performance under the setting of multi-task learning which shows the possibility of using LEAN as a unified platform for solving and proving in math. Our models, codes, and data are released at \url{https://github.com/InternLM/InternLM-Math}.
     </details>

461. **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models** [[pdf]](http://arxiv.org/abs/2402.03300) `2024-02-06` `Isabelle` (123 cite) (7 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Group Relative Policy Optimization (GRPO), a variant of Proximal Policy Optimization (PPO), that enhances mathematical reasoning abilities while concurrently optimizing the memory usage of PPO is introduced.
     </details>


     <details>
          <summary>Abstract</summary>
          Mathematical reasoning poses a significant challenge for language models due to its complex and structured nature. In this paper, we introduce DeepSeekMath 7B, which continues pre-training DeepSeek-Coder-Base-v1.5 7B with 120B math-related tokens sourced from Common Crawl, together with natural language and code data. DeepSeekMath 7B has achieved an impressive score of 51.7% on the competition-level MATH benchmark without relying on external toolkits and voting techniques, approaching the performance level of Gemini-Ultra and GPT-4. Self-consistency over 64 samples from DeepSeekMath 7B achieves 60.9% on MATH. The mathematical reasoning capability of DeepSeekMath is attributed to two key factors: First, we harness the significant potential of publicly available web data through a meticulously engineered data selection pipeline. Second, we introduce Group Relative Policy Optimization (GRPO), a variant of Proximal Policy Optimization (PPO), that enhances mathematical reasoning abilities while concurrently optimizing the memory usage of PPO.
     </details>

462. **Large Language Models as an Indirect Reasoner: Contrapositive and Contradiction for Automated Reasoning** [[pdf]](http://arxiv.org/abs/2402.03667) `2024-02-05` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel Indirect Reasoning (IR) method that employs the logic of contrapositives and contradictions to tackle IR tasks such as factual reasoning and mathematic proof to strengthen the reasoning power of LLMs is proposed.
     </details>


     <details>
          <summary>Abstract</summary>
          Recently, increasing attention has been focused drawn on to improve the ability of Large Language Models (LLMs) to perform complex reasoning. However, previous methods, such as Chain-of-Thought and Self-Consistency, mainly follow Direct Reasoning (DR) frameworks, so they will meet difficulty in solving numerous real-world tasks which can hardly be solved via DR. Therefore, to strengthen the reasoning power of LLMs, this paper proposes a novel Indirect Reasoning (IR) method that employs the logic of contrapositives and contradictions to tackle IR tasks such as factual reasoning and mathematic proof. Specifically, our methodology comprises two steps. Firstly, we leverage the logical equivalence of contrapositive to augment the data and rules to enhance the comprehensibility of LLMs. Secondly, we design a set of prompt templates to trigger LLMs to conduct IR based on proof by contradiction that is logically equivalent to the original DR process. Our IR method is simple yet effective and can be straightforwardly integrated with existing DR methods to further boost the reasoning abilities of LLMs. The experimental results on popular LLMs, such as GPT-3.5-turbo and Gemini-pro, show that our IR method enhances the overall accuracy of factual reasoning by 27.33% and mathematical proof by 31.43%, when compared with traditional DR methods. Moreover, the methods combining IR and DR significantly outperform the methods solely using IR or DR, further demonstrating the effectiveness of our strategy.
     </details>

463. **REFINER: Reasoning Feedback on Intermediate Representations** [[pdf]](http://arxiv.org/abs/2304.01904) `2024-02-04` (108 cite) (17 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          REFINER is a framework for finetuning LMs to explicitly generate intermediate reasoning steps while interacting with a critic model that provides automated feedback on the reasoning that provides structured feedback that the reasoning LM uses to iteratively improve its intermediate arguments.
     </details>


     <details>
          <summary>Abstract</summary>
          Language models (LMs) have recently shown remarkable performance on reasoning tasks by explicitly generating intermediate inferences, e.g., chain-of-thought prompting. However, these intermediate inference steps may be inappropriate deductions from the initial context and lead to incorrect final predictions. Here we introduce REFINER, a framework for finetuning LMs to explicitly generate intermediate reasoning steps while interacting with a critic model that provides automated feedback on the reasoning. Specifically, the critic provides structured feedback that the reasoning LM uses to iteratively improve its intermediate arguments. Empirical evaluations of REFINER on three diverse reasoning tasks show significant improvements over baseline LMs of comparable scale. Furthermore, when using GPT-3.5 or ChatGPT as the reasoner, the trained critic significantly improves reasoning without finetuning the reasoner. Finally, our critic model is trained without expensive human-in-the-loop data but can be substituted with humans at inference time.
     </details>

464. **Multi-step Problem Solving Through a Verifier: An Empirical Analysis on Model-induced Process Supervision** [[pdf]](http://arxiv.org/abs/2402.02658) `2024-02-04` (5 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Model-induced Process Supervision (MiPS), a novel method for automating data curation, is introduced and verification focusing on high predicted scores of the verifier shall be preferred over that of low predicted scores, contrary to prior work.
     </details>


     <details>
          <summary>Abstract</summary>
          Process supervision, using a trained verifier to evaluate the intermediate steps generated by reasoner, has demonstrated significant improvements in multi-step problem solving. In this paper, to avoid expensive human annotation effort on the verifier training data, we introduce Model-induced Process Supervision (MiPS), a novel method for automating data curation. MiPS annotates an intermediate step by sampling completions of this solution through the reasoning model, and obtaining an accuracy defined as the proportion of correct completions. Errors in the reasoner would cause MiPS to underestimate the accuracy of intermediate steps, therefore, we suggest and empirically show that verification focusing on high predicted scores of the verifier shall be preferred over that of low predicted scores, contrary to prior work. Our approach significantly improves the performance of PaLM 2 on math and coding tasks (accuracy +0.67% on GSM8K, +4.16% on MATH, +0.92% on MBPP compared with an output supervision trained verifier). Additionally, our study demonstrates that the verifier exhibits strong generalization ability across different reasoning models.
     </details>

