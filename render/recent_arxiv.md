# Recent ArXiv 

ArXiv papers in recent months, sorted by time.

1. **Deliberate Reasoning for LLMs as Structure-aware Planning with Accurate World Model** [[pdf]](https://arxiv.org/abs/2410.03136v1) `2024-10-04` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a novel multi-step reasoning framework for LLMs, referred to as Structure-aware Planning with Accurate World Model (SWAP), which incorporates structural information to guide the reasoning process via a world model and provides a soft verification mechanism over the steps.
     </details>


     <details>
          <summary>Abstract</summary>
          Enhancing the reasoning capabilities of large language models (LLMs) remains a key challenge, especially for tasks that require complex, multi-step decision-making. Humans excel at these tasks by leveraging deliberate planning with an internal world model to simulate the potential outcomes of various actions. Inspired by this, we propose a novel multi-step reasoning framework for LLMs, referred to as Structure-aware Planning with Accurate World Model (SWAP). Unlike previous approaches that rely solely on Chain-of-Thought (CoT) reasoning in natural language, SWAP incorporates structural information to guide the reasoning process via a world model and provides a soft verification mechanism over the steps. Moreover, SWAP overcomes the challenge of accurate world state predictions in complex reasoning tasks by introducing a Generator-Discriminator architecture, which enables more reliable world modeling. Specifically, the generator predicts the next state, and the discriminator ensures alignment with the logical consistency required by the problem context. SWAP also encourages the policy model to explore a broad range of potential actions to prevent premature convergence. By resolving the bottlenecks of generation diversity for both actions and states using diversity-based modeling (DBM) and improving discrimination accuracy through contrastive ranking (CR), SWAP significantly enhances the reasoning performance of LLMs. We evaluate SWAP across diverse reasoning-intensive benchmarks including math reasoning, logical reasoning, and coding tasks. Extensive experiments demonstrate that SWAP achieves substantial improvements over the baselines and consistently outperforms existing LLMs of similar sizes.
     </details>

2. **Enhance Reasoning by Learning from Mistakes: Peer-Review Knowledge Distillation from Multiple Large Language Models** [[pdf]](https://arxiv.org/abs/2410.03663v1) `2024-10-04` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduces a novel Mistake-Aware Peer-Review Distillation (MAPD) approach, which asks teachers to identify and explain the student's mistakes, providing customized instruction learning data.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have exhibited complex reasoning abilities by generating question rationales and demonstrated exceptional performance in natural language processing (NLP) tasks. However, these reasoning capabilities generally emerge in models with tens of billions of parameters, creating significant computational challenges for real-world deployment. Recent research has concentrated on improving open-source smaller models through knowledge distillation (KD) from commercial LLMs. Nevertheless, most of these studies rely solely on the responses from one single LLM as the gold rationale for training. In this paper, we introduce a novel Mistake-Aware Peer-Review Distillation (MAPD) approach: 1) Instead of merely obtaining gold rationales from teachers, our method asks teachers to identify and explain the student's mistakes, providing customized instruction learning data. 2) We design a simulated peer-review process between teacher LLMs, which selects only the generated rationales above the acceptance threshold. This reduces the chance of teachers guessing correctly with flawed rationale, improving instructional data quality. Comprehensive experiments and analysis on mathematical, commonsense, and logical reasoning tasks demonstrate the effectiveness of our method.
     </details>

3. **Steering Large Language Models between Code Execution and Textual Reasoning** [[pdf]](https://arxiv.org/abs/2410.03524v1) `2024-10-04` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is discovered that results from LLM written code are not always better than using textual reasoning, even if the task could be solved through code, so three methods to better steer LLM code/text generation are proposed and achieve a notable improvement.
     </details>


     <details>
          <summary>Abstract</summary>
          While a lot of recent research focuses on enhancing the textual reasoning capabilities of Large Language Models (LLMs) by optimizing the multi-agent framework or reasoning chains, several benchmark tasks can be solved with 100% success through direct coding, which is more scalable and avoids the computational overhead associated with textual iterating and searching. Textual reasoning has inherent limitations in solving tasks with challenges in math, logics, optimization, and searching, which is unlikely to be solved by simply scaling up the model and data size. The recently released OpenAI GPT Code Interpreter and multi-agent frameworks such as AutoGen have demonstrated remarkable proficiency of integrating code generation and execution to solve complex tasks using LLMs. However, based on our experiments on 7 existing popular methods for steering code/text generation in both single- and multi-turn settings with 14 tasks and 6 types of LLMs (including the new O1-preview), currently there is no optimal method to correctly steer LLMs to write code when needed. We discover some interesting patterns on when models use code vs. textual reasoning with the evolution to task complexity and model sizes, which even result in an astonishingly inverse scaling law. We also discover that results from LLM written code are not always better than using textual reasoning, even if the task could be solved through code. To mitigate the above issues, we propose three methods to better steer LLM code/text generation and achieve a notable improvement. The costs of token lengths and runtime are thoroughly discussed for all the methods. We believe the problem of steering LLM code/text generation is critical for future research and has much space for further improvement. Project Page, Datasets, and Codes are available at https://yongchao98.github.io/CodeSteer/.
     </details>

4. **System 2 reasoning capabilities are nigh** [[pdf]](https://arxiv.org/abs/2410.03662v1) `2024-10-04` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is argued that if current models are insufficient to be classed as performing reasoning, there remains very little additional progress needed to attain that goal.
     </details>


     <details>
          <summary>Abstract</summary>
          In recent years, machine learning models have made strides towards human-like reasoning capabilities from several directions. In this work, we review the current state of the literature and describe the remaining steps to achieve a neural model which can perform System 2 reasoning analogous to a human. We argue that if current models are insufficient to be classed as performing reasoning, there remains very little additional progress needed to attain that goal.
     </details>

5. **Adaptive Inference-Time Compute: LLMs Can Predict if They Can Do Better, Even Mid-Generation** [[pdf]](https://arxiv.org/abs/2410.02725v1) `2024-10-03` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A new generative self-evaluation scheme designed to adaptively reduce the number of generated samples while maintaining or even improving performance and enabling more efficient and scalable compute utilization during inference for LLMs is introduced.
     </details>


     <details>
          <summary>Abstract</summary>
          Inference-time computation is a powerful paradigm to enhance the performance of large language models (LLMs), with Best-of-N sampling being a widely used technique. However, this method is computationally expensive, requiring both (1) an external reward model and (2) the generation of multiple samples. In this work, we introduce a new generative self-evaluation scheme designed to adaptively reduce the number of generated samples while maintaining or even improving performance. We use a generative reward model formulation, allowing the LLM to predict mid-generation the probability that restarting the generation will yield a better response. These predictions are obtained without an external reward model and can be used to decide whether or not to generate more samples, prune unpromising samples early on, or to pick the best sample. This capability is very inexpensive as it involves generating a single predefined token. Trained using a dataset constructed with real unfiltered LMSYS user prompts, Llama 3.1 8B's win rate against GPT-4 on AlpacaEval increases from 21% to 34% with 16 samples and math performance on GSM8K improves from 84% to 91%. By sampling only when the LLM determines that it is beneficial to do so and adaptively adjusting temperature annealing, we demonstrate that 74% of the improvement from using 16 samples can be achieved with only 1.2 samples on average. We further demonstrate that 50-75% of samples can be pruned early in generation with minimal degradation in performance. Overall, our methods enable more efficient and scalable compute utilization during inference for LLMs.
     </details>

6. **CodePMP: Scalable Preference Model Pretraining for Large Language Model Reasoning** [[pdf]](http://arxiv.org/abs/2410.02229) `2024-10-03` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          CodePMP improves RM finetuning efficiency by pretraining preference models on large-scale synthesized code-preference pairs from publicly available high-quality source code, and highlighting the importance of scalable preference model pretraining for efficient reward modeling.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have made significant progress in natural language understanding and generation, driven by scalable pretraining and advanced finetuning. However, enhancing reasoning abilities in LLMs, particularly via reinforcement learning from human feedback (RLHF), remains challenging due to the scarcity of high-quality preference data, which is labor-intensive to annotate and crucial for reward model (RM) finetuning. To alleviate this issue, we introduce CodePMP, a scalable preference model pretraining (PMP) pipeline that utilizes a large corpus of synthesized code-preference pairs from publicly available high-quality source code. CodePMP improves RM finetuning efficiency by pretraining preference models on large-scale synthesized code-preference pairs. We evaluate CodePMP on mathematical reasoning tasks (GSM8K, MATH) and logical reasoning tasks (ReClor, LogiQA2.0), consistently showing significant improvements in reasoning performance of LLMs and highlighting the importance of scalable preference model pretraining for efficient reward modeling.
     </details>

7. **GraphIC: A Graph-Based In-Context Example Retrieval Model for Multi-Step Reasoning** [[pdf]](http://arxiv.org/abs/2410.02203) `2024-10-03` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          GraphIC is presented, a novel approach that leverages graph-based representations of reasoning processes, coupled with Bayesian Networks (BNs) to select ICEs, and outperforms both training-free and training-based models in selecting ICEs, excelling in terms of both effectiveness and efficiency.
     </details>


     <details>
          <summary>Abstract</summary>
          In-context learning (ICL) enables large language models (LLMs) to generalize to new tasks by incorporating a few in-context examples (ICEs) directly in the input, without updating parameters. However, the effectiveness of ICL heavily relies on the selection of ICEs, and conventional text-based embedding methods are often inadequate for tasks that require multi-step reasoning, such as mathematical and logical problem solving. This is due to the bias introduced by shallow semantic similarities that fail to capture the deeper reasoning structures required for these tasks. We present GraphIC, a novel approach that leverages graph-based representations of reasoning processes, coupled with Bayesian Networks (BNs) to select ICEs. Graph structures inherently filter out shallow semantics while preserving the core reasoning structure. Importantly, BNs capture the dependency of a node's attributes on its parent nodes, closely mirroring the hierarchical nature of human cognition-where each thought is shaped by preceding ones. This makes BNs particularly well-suited for multi-step reasoning tasks, aligning the process more closely with human-like reasoning. Extensive experiments across three types of reasoning tasks (mathematical reasoning, code generation, and logical reasoning) demonstrate that GraphIC outperforms both training-free and training-based models in selecting ICEs, excelling in terms of both effectiveness and efficiency. We show that GraphIC enhances ICL's performance and interoperability, significantly advancing ICE selection for multi-step reasoning tasks.
     </details>

8. **LLaMA-Berry: Pairwise Optimization for O1-like Olympiad-Level Mathematical Reasoning** [[pdf]](https://arxiv.org/abs/2410.02884v1) `2024-10-03` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          An advanced mathematical problem-solving framework, LLaMA-Berry, for enhancing the mathematical reasoning ability of Large Language Models (LLMs), which combines Monte Carlo Tree Search (MCTS) with iterative Self-Refine to optimize the reasoning path and utilizes a pairwise reward model to evaluate different paths globally.
     </details>


     <details>
          <summary>Abstract</summary>
          This paper presents an advanced mathematical problem-solving framework, LLaMA-Berry, for enhancing the mathematical reasoning ability of Large Language Models (LLMs). The framework combines Monte Carlo Tree Search (MCTS) with iterative Self-Refine to optimize the reasoning path and utilizes a pairwise reward model to evaluate different paths globally. By leveraging the self-critic and rewriting capabilities of LLMs, Self-Refine applied to MCTS (SR-MCTS) overcomes the inefficiencies and limitations of conventional step-wise and greedy search algorithms by fostering a more efficient exploration of solution spaces. Pairwise Preference Reward Model~(PPRM), inspired by Reinforcement Learning from Human Feedback (RLHF), is then used to model pairwise preferences between solutions, utilizing an Enhanced Borda Count (EBC) method to synthesize these preferences into a global ranking score to find better answers. This approach addresses the challenges of scoring variability and non-independent distributions in mathematical reasoning tasks. The framework has been tested on general and advanced benchmarks, showing superior performance in terms of search efficiency and problem-solving capability compared to existing methods like ToT and rStar, particularly in complex Olympiad-level benchmarks, including GPQA, AIME24 and AMC23.
     </details>

9. **The Role of Deductive and Inductive Reasoning in Large Language Models** [[pdf]](https://arxiv.org/abs/2410.02892v1) `2024-10-03` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The Deductive and InDuctive method is proposed, which enhances LLM reasoning by dynamically integrating both deductive and inductive reasoning within the prompt construction process, and provides a more robust and cognitively aligned framework for reasoning in LLMs.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have achieved substantial progress in artificial intelligence, particularly in reasoning tasks. However, their reliance on static prompt structures, coupled with limited dynamic reasoning capabilities, often constrains their adaptability to complex and evolving problem spaces. In this paper, we propose the Deductive and InDuctive(DID) method, which enhances LLM reasoning by dynamically integrating both deductive and inductive reasoning within the prompt construction process. Drawing inspiration from cognitive science, the DID approach mirrors human adaptive reasoning mechanisms, offering a flexible framework that allows the model to adjust its reasoning pathways based on task context and performance. We empirically validate the efficacy of DID on established datasets such as AIW and MR-GSM8K, as well as on our custom dataset, Holiday Puzzle, which presents tasks about different holiday date calculating challenges. By leveraging DID's hybrid prompt strategy, we demonstrate significant improvements in both solution accuracy and reasoning quality, achieved without imposing substantial computational overhead. Our findings suggest that DID provides a more robust and cognitively aligned framework for reasoning in LLMs, contributing to the development of advanced LLM-driven problem-solving strategies informed by cognitive science models.
     </details>

10. **Unlocking Structured Thinking in Language Models with Cognitive prompting** [[pdf]](https://arxiv.org/abs/2410.02953v1) `2024-10-03` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Cognitive prompting, particularly when dynamically adapted, significantly improves the performance of larger models, such as LLaMA3.1 70B, and enhances their ability to handle multi-step reasoning tasks, highlighting cognitive prompting as a promising strategy for general-purpose AI reasoning.
     </details>


     <details>
          <summary>Abstract</summary>
          We propose cognitive prompting as a novel approach to guide problem-solving in large language models (LLMs) through structured, human-like cognitive operations such as goal clarification, decomposition, filtering, abstraction, and pattern recognition. By employing systematic, step-by-step reasoning, cognitive prompting enables LLMs to efficiently tackle complex, multi-step tasks. We evaluate the effectiveness of cognitive prompting on Meta's LLaMA models, comparing performance on arithmetic reasoning tasks using the GSM8K dataset and on commonsense reasoning benchmarks. Our analysis includes comparisons between models without cognitive prompting, models with a static sequence of cognitive operations, and models using reflective cognitive prompting, where the LLM dynamically self-selects the sequence of cognitive operations. The results show that cognitive prompting, particularly when dynamically adapted, significantly improves the performance of larger models, such as LLaMA3.1 70B, and enhances their ability to handle multi-step reasoning tasks. This approach also improves interpretability and flexibility, highlighting cognitive prompting as a promising strategy for general-purpose AI reasoning.
     </details>

11. **Can We Further Elicit Reasoning in LLMs? Critic-Guided Planning with Retrieval-Augmentation for Solving Challenging Tasks** [[pdf]](http://arxiv.org/abs/2410.01428) `2024-10-02` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Critic-guided planning with Retrieval-augmentation, CR-Planner is introduced, a novel framework that leverages fine-tuned critic models to guide both reasoning and retrieval processes through planning that significantly outperforms baselines on challenging domain-knowledge-intensive and reasoning-heavy tasks.
     </details>


     <details>
          <summary>Abstract</summary>
          State-of-the-art large language models (LLMs) exhibit impressive problem-solving capabilities but may struggle with complex reasoning and factual correctness. Existing methods harness the strengths of chain-of-thought and retrieval-augmented generation (RAG) to decompose a complex problem into simpler steps and apply retrieval to improve factual correctness. These methods work well on straightforward reasoning tasks but often falter on challenging tasks such as competitive programming and mathematics, due to frequent reasoning errors and irrelevant knowledge retrieval. To address this, we introduce Critic-guided planning with Retrieval-augmentation, CR-Planner, a novel framework that leverages fine-tuned critic models to guide both reasoning and retrieval processes through planning. CR-Planner solves a problem by iteratively selecting and executing sub-goals. Initially, it identifies the most promising sub-goal from reasoning, query generation, and retrieval, guided by rewards given by a critic model named sub-goal critic. It then executes this sub-goal through sampling and selecting the optimal output based on evaluations from another critic model named execution critic. This iterative process, informed by retrieved information and critic models, enables CR-Planner to effectively navigate the solution space towards the final answer. We employ Monte Carlo Tree Search to collect the data for training the critic models, allowing for a systematic exploration of action sequences and their long-term impacts. We validate CR-Planner on challenging domain-knowledge-intensive and reasoning-heavy tasks, including competitive programming, theorem-driven math reasoning, and complex domain retrieval problems. Our experiments demonstrate that CR-Planner significantly outperforms baselines, highlighting its effectiveness in addressing challenging problems by improving both reasoning and retrieval.
     </details>

12. **CreDes: Causal Reasoning Enhancement and Dual-End Searching for Solving Long-Range Reasoning Problems using LLMs** [[pdf]](http://arxiv.org/abs/2410.01696) `2024-10-02` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          By integrating CRE and DES (CreDes), the model has realized simultaneous multi-step reasoning, circumventing the inefficiencies from cascading multiple one-step reasoning like the Chain-of-Thought (CoT).
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have demonstrated limitations in handling combinatorial optimization problems involving long-range reasoning, partially due to causal hallucinations and huge search space. As for causal hallucinations, i.e., the inconsistency between reasoning and corresponding state transition, this paper introduces the Causal Relationship Enhancement (CRE) mechanism combining cause-effect interventions and the Individual Treatment Effect (ITE) to guarantee the solid causal rightness between each step of reasoning and state transition. As for the long causal range and huge search space limiting the performances of existing models featuring single-direction search, a Dual-End Searching (DES) approach is proposed to seek solutions by simultaneously starting from both the initial and goal states on the causal probability tree. By integrating CRE and DES (CreDes), our model has realized simultaneous multi-step reasoning, circumventing the inefficiencies from cascading multiple one-step reasoning like the Chain-of-Thought (CoT). Experiments demonstrate that CreDes significantly outperforms existing State-Of-The-Art (SOTA) solutions in long-range reasoning tasks in terms of both accuracy and time efficiency.
     </details>

13. **Evaluating Robustness of Reward Models for Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2410.01729) `2024-10-02` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Reward models are key in reinforcement learning from human feedback (RLHF) systems, aligning the model behavior with human preferences. Particularly in the math domain, there have been plenty of studies using reward models to align policies for improving reasoning capabilities. Recently, as the importance of reward models has been emphasized, RewardBench is proposed to understand their behavior. However, we figure out that the math subset of RewardBench has different representations between chosen and rejected completions, and relies on a single comparison, which may lead to unreliable results as it only see an isolated case. Therefore, it fails to accurately present the robustness of reward models, leading to a misunderstanding of its performance and potentially resulting in reward hacking. In this work, we introduce a new design for reliable evaluation of reward models, and to validate this, we construct RewardMATH, a benchmark that effectively represents the robustness of reward models in mathematical reasoning tasks. We demonstrate that the scores on RewardMATH strongly correlate with the results of optimized policy and effectively estimate reward overoptimization, whereas the existing benchmark shows almost no correlation. The results underscore the potential of our design to enhance the reliability of evaluation, and represent the robustness of reward model. We make our code and data publicly available.
     </details>

14. **Interpretable Contrastive Monte Carlo Tree Search Reasoning** [[pdf]](http://arxiv.org/abs/2410.01707) `2024-10-02` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work conducted extensive ablation studies and quantitative analysis on components of MCTS, revealing the impact of each component on the MCTS reasoning performance of LLMs and designed a highly interpretable reward model based on the principle of contrastive decoding.
     </details>


     <details>
          <summary>Abstract</summary>
          We propose SC-MCTS*: a novel Monte Carlo Tree Search (MCTS) reasoning algorithm for Large Language Models (LLMs), significantly improves both reasoning accuracy and speed. Our motivation comes from: 1. Previous MCTS LLM reasoning works often overlooked its biggest drawback--slower speed compared to CoT; 2. Previous research mainly used MCTS as a tool for LLM reasoning on various tasks with limited quantitative analysis or ablation studies of its components from reasoning interpretability perspective. 3. The reward model is the most crucial component in MCTS, however previous work has rarely conducted in-depth study or improvement of MCTS's reward models. Thus, we conducted extensive ablation studies and quantitative analysis on components of MCTS, revealing the impact of each component on the MCTS reasoning performance of LLMs. Building on this, (i) we designed a highly interpretable reward model based on the principle of contrastive decoding and (ii) achieved an average speed improvement of 51.9% per node using speculative decoding. Additionally, (iii) we improved UCT node selection strategy and backpropagation used in previous works, resulting in significant performance improvement. We outperformed o1-mini by an average of 17.4% on the Blocksworld multi-step reasoning dataset using Llama-3.1-70B with SC-MCTS*.
     </details>

15. **Layer Swapping for Zero-Shot Cross-Lingual Transfer in Large Language Models** [[pdf]](http://arxiv.org/abs/2410.01335) `2024-10-02` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The ability to successfully re-compose LLMs for cross-lingual transfer in this manner opens up future possibilities to combine model expertise, create modular solutions, and transfer reasoning capabilities across languages all post hoc.
     </details>


     <details>
          <summary>Abstract</summary>
          Model merging, such as model souping, is the practice of combining different models with the same architecture together without further training. In this work, we present a model merging methodology that addresses the difficulty of fine-tuning Large Language Models (LLMs) for target tasks in non-English languages, where task-specific data is often unavailable. We focus on mathematical reasoning and without in-language math data, facilitate cross-lingual transfer by composing language and math capabilities. Starting from the same pretrained model, we fine-tune separate "experts" on math instruction data in English and on generic instruction data in the target language. We then replace the top and bottom transformer layers of the math expert directly with layers from the language expert, which consequently enhances math performance in the target language. The resulting merged models outperform the individual experts and other merging methods on the math benchmark, MGSM, by 10% across four major languages where math instruction data is scarce. In addition, this layer swapping is simple, inexpensive, and intuitive, as it is based on an interpretative analysis of the most important parameter changes during the fine-tuning of each expert. The ability to successfully re-compose LLMs for cross-lingual transfer in this manner opens up future possibilities to combine model expertise, create modular solutions, and transfer reasoning capabilities across languages all post hoc.
     </details>

16. **Not All LLM Reasoners Are Created Equal** [[pdf]](http://arxiv.org/abs/2410.01748) `2024-10-02` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Overall, LLMs exhibit systematic differences in their reasoning abilities, despite what their performance on standard benchmarks indicates, with a significant reasoning gap in smaller, more cost-efficient, and math-specialized models.
     </details>


     <details>
          <summary>Abstract</summary>
          We study the depth of grade-school math (GSM) problem-solving capabilities of LLMs. To this end, we evaluate their performance on pairs of existing math word problems together so that the answer to the second problem depends on correctly answering the first problem. Our findings reveal a significant reasoning gap in most LLMs, that is performance difference between solving the compositional pairs and solving each question independently. This gap is more pronounced in smaller, more cost-efficient, and math-specialized models. Moreover, instruction-tuning recipes and code generation have varying effects across LLM sizes, while finetuning on GSM can lead to task overfitting. Our analysis indicates that large reasoning gaps are not because of test-set leakage, but due to distraction from additional context and poor second-hop reasoning. Overall, LLMs exhibit systematic differences in their reasoning abilities, despite what their performance on standard benchmarks indicates.
     </details>

17. **OpenMathInstruct-2: Accelerating AI for Math with Massive Open-Source Instruction Data** [[pdf]](http://arxiv.org/abs/2410.01560) `2024-10-02` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The OpenMathInstruct-2 dataset is created, which consists of 14M question-solution pairs, making it nearly eight times larger than the previous largest open-source math reasoning dataset, and is released under a commercially permissive license.
     </details>


     <details>
          <summary>Abstract</summary>
          Mathematical reasoning continues to be a critical challenge in large language model (LLM) development with significant interest. However, most of the cutting-edge progress in mathematical reasoning with LLMs has become \emph{closed-source} due to lack of access to training data. This lack of data access limits researchers from understanding the impact of different choices for synthesizing and utilizing the data. With the goal of creating a high-quality finetuning (SFT) dataset for math reasoning, we conduct careful ablation experiments on data synthesis using the recently released \texttt{Llama3.1} family of models. Our experiments show that: (a) solution format matters, with excessively verbose solutions proving detrimental to SFT performance, (b) data generated by a strong teacher outperforms equally-sized data generated by a weak student model, (c) SFT is robust to low-quality solutions, allowing for imprecise data filtering, and (d) question diversity is crucial for achieving data scaling gains. Based on these insights, we create the OpenMathInstruct-2 dataset, which consists of 14M question-solution pairs ($\approx$ 600K unique questions), making it nearly eight times larger than the previous largest open-source math reasoning dataset. Finetuning the \texttt{Llama-3.1-8B-Base} using OpenMathInstruct-2 outperforms \texttt{Llama3.1-8B-Instruct} on MATH by an absolute 15.9\% (51.9\% $\rightarrow$ 67.8\%). Finally, to accelerate the open-source efforts, we release the code, the finetuned models, and the OpenMathInstruct-2 dataset under a commercially permissive license.
     </details>

18. **PersonaMath: Enhancing Math Reasoning through Persona-Driven Data Augmentation** [[pdf]](http://arxiv.org/abs/2410.01504) `2024-10-02` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a data augmentation approach and introduces PersonaMathQA, a dataset derived from MATH and GSM8K, on which the PersonaMath models are trained, and introduces a novel persona-driven data augmentation technique to enhance the dataset's quantity and diversity.
     </details>


     <details>
          <summary>Abstract</summary>
          While closed-source Large Language Models (LLMs) demonstrate strong mathematical problem-solving abilities, open-source models continue to struggle with such tasks. To bridge this gap, we propose a data augmentation approach and introduce PersonaMathQA, a dataset derived from MATH and GSM8K, on which we train the PersonaMath models. Our approach consists of two stages: the first stage is learning from Persona Diversification, and the second stage is learning from Reflection. In the first stage, we regenerate detailed chain-of-thought (CoT) solutions as instructions using a closed-source LLM and introduce a novel persona-driven data augmentation technique to enhance the dataset's quantity and diversity. In the second stage, we incorporate reflection to fully leverage more challenging and valuable questions. Evaluation of our PersonaMath models on MATH and GSM8K reveals that the PersonaMath-7B model (based on LLaMA-2-7B) achieves an accuracy of 24.2% on MATH and 68.7% on GSM8K, surpassing all baseline methods and achieving state-of-the-art performance. Notably, our dataset contains only 70.3K data points-merely 17.8% of MetaMathQA and 27% of MathInstruct-yet our model outperforms these baselines, demonstrating the high quality and diversity of our dataset, which enables more efficient model training. We open-source the PersonaMathQA dataset, PersonaMath models, and our code for public usage.
     </details>

19. **ReGenesis: LLMs can Grow into Reasoning Generalists via Self-Improvement** [[pdf]](http://arxiv.org/abs/2410.02108) `2024-10-02` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          ReGenesis self-synthesizes reasoning paths by converting general reasoning guidelines into task-specific ones, generating reasoning structures, and subsequently transforming these structures into reasoning paths, without the need for human-designed task-specific examples used in existing methods.
     </details>


     <details>
          <summary>Abstract</summary>
          Post-training Large Language Models (LLMs) with explicit reasoning trajectories can enhance their reasoning abilities. However, acquiring such high-quality trajectory data typically demands meticulous supervision from humans or superior models, which can be either expensive or license-constrained. In this paper, we explore how far an LLM can improve its reasoning by self-synthesizing reasoning paths as training data without any additional supervision. Existing self-synthesizing methods, such as STaR, suffer from poor generalization to out-of-domain (OOD) reasoning tasks. We hypothesize it is due to that their self-synthesized reasoning paths are too task-specific, lacking general task-agnostic reasoning guidance. To address this, we propose Reasoning Generalist via Self-Improvement (ReGenesis), a method to self-synthesize reasoning paths as post-training data by progressing from abstract to concrete. More specifically, ReGenesis self-synthesizes reasoning paths by converting general reasoning guidelines into task-specific ones, generating reasoning structures, and subsequently transforming these structures into reasoning paths, without the need for human-designed task-specific examples used in existing methods. We show that ReGenesis achieves superior performance on all in-domain and OOD settings tested compared to existing methods. For six OOD tasks specifically, while previous methods exhibited an average performance decrease of approximately 4.6% after post training, ReGenesis delivers around 6.1% performance improvement. We also conduct in-depth analysis of our framework and show ReGenesis is effective across various LLMs and design choices.
     </details>

20. **Step-by-Step Reasoning for Math Problems via Twisted Sequential Monte Carlo** [[pdf]](http://arxiv.org/abs/2410.01920) `2024-10-02` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduces a novel verification method based on Twisted Sequential Monte Carlo (TSMC), which sequentially refines its sampling effort to focus exploration on promising candidates, resulting in more efficient generation of high-quality solutions.
     </details>


     <details>
          <summary>Abstract</summary>
          Augmenting the multi-step reasoning abilities of Large Language Models (LLMs) has been a persistent challenge. Recently, verification has shown promise in improving solution consistency by evaluating generated outputs. However, current verification approaches suffer from sampling inefficiencies, requiring a large number of samples to achieve satisfactory performance. Additionally, training an effective verifier often depends on extensive process supervision, which is costly to acquire. In this paper, we address these limitations by introducing a novel verification method based on Twisted Sequential Monte Carlo (TSMC). TSMC sequentially refines its sampling effort to focus exploration on promising candidates, resulting in more efficient generation of high-quality solutions. We apply TSMC to LLMs by estimating the expected future rewards at partial solutions. This approach results in a more straightforward training target that eliminates the need for step-wise human annotations. We empirically demonstrate the advantages of our method across multiple math benchmarks, and also validate our theoretical analysis of both our approach and existing verification methods.
     </details>

21. **TypedThinker: Typed Thinking Improves Large Language Model Reasoning** [[pdf]](http://arxiv.org/abs/2410.01952) `2024-10-02` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          TypedThinker is a novel framework that enhances LLMs' problem-solving abilities by incorporating multiple reasoning types (deductive, inductive, abductive, and analogical) and shows effective generalization to new benchmarks and can further enhance the reasoning capability of powerful models like GPT-4o.
     </details>


     <details>
          <summary>Abstract</summary>
          Despite significant advancements in the reasoning capabilities of Large Language Models (LLMs), the lack of diverse reasoning solutions often makes them trapped in a limited solution search area. In this paper, we propose TypedThinker, a novel framework that enhances LLMs' problem-solving abilities by incorporating multiple reasoning types (deductive, inductive, abductive, and analogical). Our analysis across four benchmarks reveals that different reasoning types uniquely solve distinct sets of problems, highlighting the importance of diverse thinking approaches. TypedThinker addresses two key challenges: selecting appropriate reasoning types for given problems and effectively implementing specific reasoning types. Through self-training on successful experiences, TypedThinker learns an implicit policy for reasoning type selection and application. Experimental results demonstrate significant improvements over baseline models, with accuracy increases of 3.4% for Mistral 7B and 16.7% for LLaMA3 8B across four reasoning benchmarks. Notably, TypedThinker shows effective generalization to new benchmarks and can further enhance the reasoning capability of powerful models like GPT-4o. The code is released at https://github.com/dqwang122/ThinkHub.
     </details>

22. **VinePPO: Unlocking RL Potential For LLM Reasoning Through Refined Credit Assignment** [[pdf]](http://arxiv.org/abs/2410.01679) `2024-10-02` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          VinePPO is proposed, a straightforward approach that leverages the flexibility of language environments to compute unbiased Monte Carlo-based estimates, bypassing the need for large value networks, and consistently outperforms PPO and other RL-free baselines across MATH and GSM8K datasets.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) are increasingly applied to complex reasoning tasks that require executing several complex steps before receiving any reward. Properly assigning credit to these steps is essential for enhancing model performance. Proximal Policy Optimization (PPO), a state-of-the-art reinforcement learning (RL) algorithm used for LLM finetuning, employs value networks to tackle credit assignment. However, value networks face challenges in predicting the expected cumulative rewards accurately in complex reasoning tasks, often leading to high-variance updates and suboptimal performance. In this work, we systematically evaluate the efficacy of value networks and reveal their significant shortcomings in reasoning-heavy LLM tasks, showing that they barely outperform a random baseline when comparing alternative steps. To address this, we propose VinePPO, a straightforward approach that leverages the flexibility of language environments to compute unbiased Monte Carlo-based estimates, bypassing the need for large value networks. Our method consistently outperforms PPO and other RL-free baselines across MATH and GSM8K datasets with fewer gradient updates (up to 9x), less wall-clock time (up to 3.0x). These results emphasize the importance of accurate credit assignment in RL finetuning of LLM and demonstrate VinePPO's potential as a superior alternative.
     </details>

23. **When a language model is optimized for reasoning, does it still show embers of autoregression? An analysis of OpenAI o1** [[pdf]](http://arxiv.org/abs/2410.01792) `2024-10-02` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is found that o1 substantially outperforms previous LLMs in many cases, with particularly large improvements on rare variants of common tasks, and shows that optimizing a language model for reasoning can mitigate but might not fully overcome the language model's probability sensitivity.
     </details>


     <details>
          <summary>Abstract</summary>
          In "Embers of Autoregression" (McCoy et al., 2023), we showed that several large language models (LLMs) have some important limitations that are attributable to their origins in next-word prediction. Here we investigate whether these issues persist with o1, a new system from OpenAI that differs from previous LLMs in that it is optimized for reasoning. We find that o1 substantially outperforms previous LLMs in many cases, with particularly large improvements on rare variants of common tasks (e.g., forming acronyms from the second letter of each word in a list, rather than the first letter). Despite these quantitative improvements, however, o1 still displays the same qualitative trends that we observed in previous systems. Specifically, o1 -- like previous LLMs -- is sensitive to the probability of examples and tasks, performing better and requiring fewer "thinking tokens" in high-probability settings than in low-probability ones. These results show that optimizing a language model for reasoning can mitigate but might not fully overcome the language model's probability sensitivity.
     </details>

24. **RATIONALYST: Pre-training Process-Supervision for Improving Reasoning** [[pdf]](http://arxiv.org/abs/2410.01044) `2024-10-01` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          RATIONALYST, a model for process-supervision of reasoning based on pre-training on a vast collection of rationale annotations extracted from unlabeled data, is introduced, which demonstrates superior performance compared to significantly larger verifiers like GPT-4 and similarly sized models fine-tuned on matching training sets.
     </details>


     <details>
          <summary>Abstract</summary>
          The reasoning steps generated by LLMs might be incomplete, as they mimic logical leaps common in everyday communication found in their pre-training data: underlying rationales are frequently left implicit (unstated). To address this challenge, we introduce RATIONALYST, a model for process-supervision of reasoning based on pre-training on a vast collection of rationale annotations extracted from unlabeled data. We extract 79k rationales from web-scale unlabelled dataset (the Pile) and a combination of reasoning datasets with minimal human intervention. This web-scale pre-training for reasoning allows RATIONALYST to consistently generalize across diverse reasoning tasks, including mathematical, commonsense, scientific, and logical reasoning. Fine-tuned from LLaMa-3-8B, RATIONALYST improves the accuracy of reasoning by an average of 3.9% on 7 representative reasoning benchmarks. It also demonstrates superior performance compared to significantly larger verifiers like GPT-4 and similarly sized models fine-tuned on matching training sets.
     </details>

25. **Scheherazade: Evaluating Chain-of-Thought Math Reasoning in LLMs with Chain-of-Problems** [[pdf]](http://arxiv.org/abs/2410.00151) `2024-09-30` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Benchmarks are critical for measuring progress of math reasoning abilities of Large Language Models (LLMs). However, existing widely-used benchmarks such as GSM8K have been rendered less useful as multiple cutting-edge LLMs achieve over 94% accuracy. While harder benchmarks have been proposed, their creation is often manual and expensive. We present Scheherazade, an automated approach for producing challenging mathematical reasoning benchmarks by logically chaining mathematical reasoning problems. We propose two different chaining methods, forward chaining and backward chaining, which require reasoning forward and backward through the chain respectively. We apply Scheherazade on GSM8K to create GSM8K-Scheherazade and evaluate 3 frontier LLMs and OpenAI's o1-preview on it. We show that while frontier models' performance declines precipitously at only a few questions chained, a preliminary evaluation suggests o1-preview performance persists up to 5 questions chained backwards. In addition, while all other models perform worse when problems are chained backwards, o1-preview performs better on backward-chained benchmarks. We will release the dataset and code publicly.
     </details>

26. **The Perfect Blend: Redefining RLHF with Mixture of Judges** [[pdf]](http://arxiv.org/abs/2409.20370) `2024-09-30` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel post-training paradigm which can detect and mitigate reward hacking behaviors while reaching a pareto-optimal point across an extremely large number of objectives is introduced, called Constrained Generative Policy Optimization (CGPO).
     </details>


     <details>
          <summary>Abstract</summary>
          Reinforcement learning from human feedback (RLHF) has become the leading approach for fine-tuning large language models (LLM). However, RLHF has limitations in multi-task learning (MTL) due to challenges of reward hacking and extreme multi-objective optimization (i.e., trade-off of multiple and/or sometimes conflicting objectives). Applying RLHF for MTL currently requires careful tuning of the weights for reward model and data combinations. This is often done via human intuition and does not generalize. In this work, we introduce a novel post-training paradigm which we called Constrained Generative Policy Optimization (CGPO). The core of CGPO is Mixture of Judges (MoJ) with cost-efficient constrained policy optimization with stratification, which can identify the perfect blend in RLHF in a principled manner. It shows strong empirical results with theoretical guarantees, does not require extensive hyper-parameter tuning, and is plug-and-play in common post-training pipelines. Together, this can detect and mitigate reward hacking behaviors while reaching a pareto-optimal point across an extremely large number of objectives.   Our empirical evaluations demonstrate that CGPO significantly outperforms standard RLHF algorithms like PPO and DPO across various tasks including general chat, STEM questions, instruction following, and coding. Specifically, CGPO shows improvements of 7.4% in AlpacaEval-2 (general chat), 12.5% in Arena-Hard (STEM & reasoning), and consistent gains in other domains like math and coding. Notably, PPO, while commonly used, is prone to severe reward hacking in popular coding benchmarks, which CGPO successfully addresses. This breakthrough in RLHF not only tackles reward hacking and extreme multi-objective optimization challenges but also advances the state-of-the-art in aligning general-purpose LLMs for diverse applications.
     </details>

27. **MetaMath: Integrating Natural Language and Code for Enhanced Mathematical Reasoning in Large Language Models** [[pdf]](http://arxiv.org/abs/2409.19381) `2024-09-28` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Findings show that LLMs are better at reasoning in natural language compared to code, and motivate the development of a new prompting method, MetaMath, which leverages an LLM to dynamically select the most appropriate reasoning form, resulting in improved performance over comparable baselines with GPT-4o-mini.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) are commonly used to generate solutions for mathematical reasoning problems in the following formats: natural language, code, or a combination of both. In this paper, we explore fundamental questions related to solving mathematical reasoning problems using natural language and code with state-of-the-art LLMs, including GPT-4o-mini and LLama-3.1-8b-Turbo. Our findings show that LLMs are better at reasoning in natural language compared to code. Additionally, although natural language and code serve as complementary forms of reasoning, they can affect each other in a negative way in certain scenarios. These insights motivate our development of a new prompting method, MetaMath, which leverages an LLM to dynamically select the most appropriate reasoning form, resulting in improved performance over comparable baselines with GPT-4o-mini.
     </details>

28. **BEATS: Optimizing LLM Mathematical Capabilities with BackVerify and Adaptive Disambiguate based Efficient Tree Search** [[pdf]](http://arxiv.org/abs/2409.17972) `2024-09-26` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a novel approach, BEATS, to enhance mathematical problem-solving abilities of large Language Models, which leverages newly designed prompts that guide the model to iteratively rewrite, advance by one step, and generate answers based on previous steps.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have exhibited exceptional performance across a broad range of tasks and domains. However, they still encounter difficulties in solving mathematical problems due to the rigorous and logical nature of mathematics. Previous studies have employed techniques such as supervised fine-tuning (SFT), prompt engineering, and search-based methods to improve the mathematical problem-solving abilities of LLMs. Despite these efforts, their performance remains suboptimal and demands substantial computational resources. To address this issue, we propose a novel approach, BEATS, to enhance mathematical problem-solving abilities. Our method leverages newly designed prompts that guide the model to iteratively rewrite, advance by one step, and generate answers based on previous steps. Additionally, we introduce a new back-verification technique that uses LLMs to validate the correctness of the generated answers. Furthermore, we employ a pruning tree search to optimize search time while achieving strong performance. Notably, our method improves Qwen2-7b-Instruct's score from 36.94 to 61.52, outperforming GPT4's 42.5 on the MATH benchmark.
     </details>

29. **Learning to Love Edge Cases in Formative Math Assessment: Using the AMMORE Dataset and Chain-of-Thought Prompting to Improve Grading Accuracy** [[pdf]](http://arxiv.org/abs/2409.17904) `2024-09-26` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          AMMORE, a new dataset of 53,000 math open-response question-answer pairs from Rori, is introduced and two experiments to evaluate the use of large language models (LLM) for grading particularly challenging student answers suggest that LLMs could be a valuable tool for grading open-response questions in K-12 mathematics education.
     </details>


     <details>
          <summary>Abstract</summary>
          This paper introduces AMMORE, a new dataset of 53,000 math open-response question-answer pairs from Rori, a learning platform used by students in several African countries and conducts two experiments to evaluate the use of large language models (LLM) for grading particularly challenging student answers. The AMMORE dataset enables various potential analyses and provides an important resource for researching student math acquisition in understudied, real-world, educational contexts. In experiment 1 we use a variety of LLM-driven approaches, including zero-shot, few-shot, and chain-of-thought prompting, to grade the 1% of student answers that a rule-based classifier fails to grade accurately. We find that the best-performing approach -- chain-of-thought prompting -- accurately scored 92% of these edge cases, effectively boosting the overall accuracy of the grading from 98.7% to 99.9%. In experiment 2, we aim to better understand the consequential validity of the improved grading accuracy, by passing grades generated by the best-performing LLM-based approach to a Bayesian Knowledge Tracing (BKT) model, which estimated student mastery of specific lessons. We find that relatively modest improvements in model accuracy at the individual question level can lead to significant changes in the estimation of student mastery. Where the rules-based classifier currently used to grade student, answers misclassified the mastery status of 6.9% of students across their completed lessons, using the LLM chain-of-thought approach this misclassification rate was reduced to 2.6% of students. Taken together, these findings suggest that LLMs could be a valuable tool for grading open-response questions in K-12 mathematics education, potentially enabling encouraging wider adoption of open-ended questions in formative assessment.
     </details>

30. **SciDFM: A Large Language Model with Mixture-of-Experts for Science** [[pdf]](http://arxiv.org/abs/2409.18412) `2024-09-26` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          SciDFM, a mixture-of-experts LLM, is introduced, which is trained from scratch and is able to conduct college-level scientific reasoning and understand molecules and amino acid sequences and reaches a SOTA performance on domain-specific benchmarks among models of similar size.
     </details>


     <details>
          <summary>Abstract</summary>
          Recently, there has been a significant upsurge of interest in leveraging large language models (LLMs) to assist scientific discovery. However, most LLMs only focus on general science, while they lack domain-specific knowledge, such as chemical molecules and amino acid sequences. To bridge these gaps, we introduce SciDFM, a mixture-of-experts LLM, which is trained from scratch and is able to conduct college-level scientific reasoning and understand molecules and amino acid sequences. We collect a large-scale training corpus containing numerous scientific papers and books from different disciplines as well as data from domain-specific databases. We further fine-tune the pre-trained model on lots of instruction data to improve performances on downstream benchmarks. From experiment results, we show that SciDFM achieves strong performance on general scientific benchmarks such as SciEval and SciQ, and it reaches a SOTA performance on domain-specific benchmarks among models of similar size. We further analyze the expert layers and show that the results of expert selection vary with data from different disciplines. To benefit the broader research community, we open-source SciDFM at https://huggingface.co/OpenDFM/SciDFM-MoE-A5.6B-v1.0.
     </details>

31. **HDFlow: Enhancing LLM Complex Problem-Solving with Hybrid Thinking and Dynamic Workflows** [[pdf]](http://arxiv.org/abs/2409.17433) `2024-09-25` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel framework HDFlow for complex reasoning with LLMs that combines fast and slow thinking modes in an adaptive manner and hybrid thinking achieves the highest accuracy while providing an effective balance between computational efficiency and performance is proposed.
     </details>


     <details>
          <summary>Abstract</summary>
          Despite recent advancements in large language models (LLMs), their performance on complex reasoning problems requiring multi-step thinking and combining various skills is still limited. To address this, we propose a novel framework HDFlow for complex reasoning with LLMs that combines fast and slow thinking modes in an adaptive manner. Our approach consists of two key components: 1) a new approach for slow, deliberate reasoning called Dynamic Workflow, which automatically decomposes complex problems into more manageable sub-tasks and dynamically designs a workflow to assemble specialized LLM or symbolic reasoning tools to solve sub-tasks; 2) Hybrid Thinking, a general framework that dynamically combines fast and slow thinking based on problem complexity. Finally, we propose an easy-to-scale method for automatically synthesizing a large-scale dataset of 27K challenging reasoning problems for complex reasoning and a hybrid thinking tuning method that trains smaller LLMs on this dataset to internalize the fast/slow hybrid reasoning strategies. Experiments on four reasoning benchmark datasets demonstrate that our slow thinking with dynamic workflows significantly outperforms Chain-of-Thought, and hybrid thinking achieves the highest accuracy while providing an effective balance between computational efficiency and performance. Fine-tuning using our hybrid thinking approach also significantly boosts the complex reasoning capabilities of open-source language models. The results showcase the promise of slow thinking, dynamic workflows, and hybrid thinking in expanding the frontier of complex problem-solving with LLMs\footnote{Code and data will be released at \url{https://github.com/wenlinyao/HDFlow}.}.
     </details>

32. **LLaMa-SciQ: An Educational Chatbot for Answering Science MCQ** [[pdf]](http://arxiv.org/abs/2409.16779) `2024-09-25` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) often struggle with tasks requiring mathematical reasoning, particularly multiple-choice questions (MCQs). To address this issue, we developed LLaMa-SciQ, an educational chatbot designed to assist college students in solving and understanding MCQs in STEM fields. We begin by fine-tuning and aligning the models to human preferences. After comparing the performance of Mistral-7B and LLaMa-8B, we selected the latter as the base model due to its higher evaluation accuracy. To further enhance accuracy, we implement Retrieval-Augmented Generation (RAG) and apply quantization to compress the model, reducing inference time and increasing accessibility for students. For mathematical reasoning, LLaMa-SciQ achieved 74.5% accuracy on the GSM8k dataset and 30% on the MATH dataset. However, RAG does not improve performance and even reduces it, likely due to retriever issues or the model's unfamiliarity with context. Despite this, the quantized model shows only a 5% loss in performance, demonstrating significant efficiency improvements.
     </details>

33. **MathDSL: A Domain-Specific Language for Concise Mathematical Solutions Via Program Synthesis** [[pdf]](http://arxiv.org/abs/2409.17490) `2024-09-25` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is demonstrated that a program synthesis system (DreamCoder) using MathDSL can generate programs that solve linear equations with greater accuracy and conciseness than using reinforcement learning systems.
     </details>


     <details>
          <summary>Abstract</summary>
          We present MathDSL, a Domain-Specific Language (DSL) for mathematical equation solving, which, when deployed in program synthesis models, outperforms state-of-the-art reinforcement-learning-based methods. We also introduce a quantitative metric for measuring the conciseness of a mathematical solution and demonstrate the improvement in the quality of generated solutions compared to other methods. Our system demonstrates that a program synthesis system (DreamCoder) using MathDSL can generate programs that solve linear equations with greater accuracy and conciseness than using reinforcement learning systems. Additionally, we demonstrate that if we use the action spaces of previous reinforcement learning systems as DSLs, MathDSL outperforms the action-space-DSLs. We use DreamCoder to store equation-solving strategies as learned abstractions in its program library and demonstrate that by using MathDSL, these can be converted into human-interpretable solution strategies that could have applications in mathematical education.
     </details>

34. **Models Can and Should Embrace the Communicative Nature of Human-Generated Math** [[pdf]](http://arxiv.org/abs/2409.17005) `2024-09-25` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Math is constructed by people for people: just as natural language corpora reflect not just propositions but the communicative goals of language users, the math data that models are trained on reflects not just idealized mathematical entities but rich communicative intentions. While there are important advantages to treating math in a purely symbolic manner, we here hypothesize that there are benefits to treating math as situated linguistic communication and that language models are well suited for this goal, in ways that are not fully appreciated. We illustrate these points with two case studies. First, we ran an experiment in which we found that language models interpret the equals sign in a humanlike way -- generating systematically different word problems for the same underlying equation arranged in different ways. Second, we found that language models prefer proofs to be ordered in naturalistic ways, even though other orders would be logically equivalent. We advocate for AI systems that learn from and represent the communicative intentions latent in human-generated math.
     </details>

35. **Proof Automation with Large Language Models** [[pdf]](https://arxiv.org/abs/2409.14274v1) `2024-09-22` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          PALM is a novel generate-then-repair approach that first prompts an LLM to generate an initial proof and then leverages targeted symbolic methods to iteratively repair low-level problems, and significantly outperforms other state-of-the-art approaches.
     </details>


     <details>
          <summary>Abstract</summary>
          Interactive theorem provers such as Coq are powerful tools to formally guarantee the correctness of software. However, using these tools requires significant manual effort and expertise. While Large Language Models (LLMs) have shown promise in automatically generating informal proofs in natural language, they are less effective at generating formal proofs in interactive theorem provers. In this paper, we conduct a formative study to identify common mistakes made by LLMs when asked to generate formal proofs. By analyzing 520 proof generation errors made by GPT-3.5, we found that GPT-3.5 often identified the correct high-level structure of a proof, but struggled to get the lower-level details correct. Based on this insight, we propose PALM, a novel generate-then-repair approach that first prompts an LLM to generate an initial proof and then leverages targeted symbolic methods to iteratively repair low-level problems. We evaluate PALM on a large dataset that includes more than 10K theorems. Our results show that PALM significantly outperforms other state-of-the-art approaches, successfully proving 76.6% to 180.4% more theorems. Moreover, PALM proves 1270 theorems beyond the reach of existing approaches. We also demonstrate the generalizability of PALM across different LLMs.
     </details>

36. **GroupDebate: Enhancing the Efficiency of Multi-Agent Debate Using Group Discussion** [[pdf]](http://arxiv.org/abs/2409.14051) `2024-09-21` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A method to significantly reduce token cost in multi-agent debates by dividing all agents into multiple debate groups, with agents engaging in debates within their respective groups and sharing interim debate results between groups.
     </details>


     <details>
          <summary>Abstract</summary>
          In recent years, Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse NLP tasks. Extensive research has explored how to enhance the logical reasoning abilities such as Chain-of-Thought, Chain-of-Thought with Self-Consistency, Tree-Of-Thoughts, and multi-agent debates. In the context of multi-agent debates, significant performance improvements can be achieved with an increasing number of agents and debate rounds. However, the escalation in the number of agents and debate rounds can drastically raise the tokens cost of debates, thereby limiting the scalability of the multi-agent debate technique. To better harness the advantages of multi-agent debates in logical reasoning tasks, this paper proposes a method to significantly reduce token cost in multi-agent debates. This approach involves dividing all agents into multiple debate groups, with agents engaging in debates within their respective groups and sharing interim debate results between groups. Comparative experiments across multiple datasets have demonstrated that this method can reduce the total tokens by up to 51.7% during debates and while potentially enhancing accuracy by as much as 25%. Our method significantly enhances the performance and efficiency of interactions in the multi-agent debate.
     </details>

37. **CSCE: Boosting LLM Reasoning by Simultaneous Enhancing of Casual Significance and Consistency** [[pdf]](http://arxiv.org/abs/2409.17174) `2024-09-20` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A non-chain-based reasoning framework for simultaneous consideration of causal significance and consistency, i.e., the Causal Significance and Consistency Enhancer (CSCE) is proposed and extensive experiments show that this method improves both the reasoning success rate and speed.
     </details>


     <details>
          <summary>Abstract</summary>
          Chain-based reasoning methods like chain of thought (CoT) play a rising role in solving reasoning tasks for large language models (LLMs). However, the causal illusions between \textit{a step of reasoning} and \textit{corresponding state transitions} are becoming a significant obstacle to advancing LLMs' reasoning capabilities, especially in long-range reasoning tasks. This paper proposes a non-chain-based reasoning framework for simultaneous consideration of causal significance and consistency, i.e., the Causal Significance and Consistency Enhancer (CSCE). We customize LLM's loss function utilizing treatment effect assessments to enhance its reasoning ability from two aspects: causal significance and consistency. This ensures that the model captures essential causal relationships and maintains robust and consistent performance across various scenarios. Additionally, we transform the reasoning process from the cascading multiple one-step reasoning commonly used in Chain-Based methods, like CoT, to a causal-enhanced method that outputs the entire reasoning process in one go, further improving the model's reasoning efficiency. Extensive experiments show that our method improves both the reasoning success rate and speed. These improvements further demonstrate that non-chain-based methods can also aid LLMs in completing reasoning tasks.
     </details>

38. **Neural-Symbolic Collaborative Distillation: Advancing Small Language Models for Complex Reasoning Tasks** [[pdf]](http://arxiv.org/abs/2409.13203) `2024-09-20` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The proposed NesyCD can achieve superior performance cost-effectively, utilizing smaller models and blending parameterized neural networks with symbolic KB, which generalizes well and is comprehended and manipulated by humans.
     </details>


     <details>
          <summary>Abstract</summary>
          In this paper, we propose $\textbf{Ne}$ural-$\textbf{Sy}$mbolic $\textbf{C}$ollaborative $\textbf{D}$istillation ($\textbf{NesyCD}$), a novel knowledge distillation method for learning the complex reasoning abilities of Large Language Models (LLMs, e.g., \textgreater 13B). We argue that complex reasoning tasks are difficult for Small Language Models (SLMs, e.g., $\leq$ 7B), as these tasks demand not only general cognitive abilities but also specialized knowledge, which is often sparse and difficult for these neural-based SLMs to effectively capture. Therefore, NesyCD distills the general capabilities and specialized knowledge in LLMs using different manners. On the one hand, we distill only general abilities from teacher LLMs into the student SLMs of parameterized neural networks. On the other hand, for the specialized abilities and uncommon knowledge of a complex reasoning task, we employ a symbolic knowledge distillation approach to obtain and store the specialized knowledge within a symbolic knowledge base (KB). By decoupling general and specialized capabilities, the proposed NesyCD can achieve superior performance cost-effectively, utilizing smaller models and blending parameterized neural networks with symbolic KB. Moreover, the specialized KB generalizes well and is comprehended and manipulated by humans. Our experiments show that NesyCD significantly boosts SLMs' complex reasoning performance on in-domain (BBH, GSM8K) and out-of-domain (AGIEval, ARC) datasets. Notably, our approach enabled the LLaMA3-8B and Qwen2-7B to surpass GPT-3.5-turbo in performance and come close to matching LLaMA3-70B, despite the latter having nine times more parameters. Our code will be available at https://github.com/Xnhyacinth/NesyCD.
     </details>

39. **Training Language Models to Self-Correct via Reinforcement Learning** [[pdf]](http://arxiv.org/abs/2409.12917) `2024-09-19` (9 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A multi-turn online reinforcement learning (RL) approach that significantly improves an LLM's self-correction ability using entirely self-generated data, and uses appropriate regularization to steer the learning process into learning a self-correction behavior that is effective at test time as opposed to fitting high-reward responses for a given prompt.
     </details>


     <details>
          <summary>Abstract</summary>
          Self-correction is a highly desirable capability of large language models (LLMs), yet it has consistently been found to be largely ineffective in modern LLMs. Existing approaches for training self-correction either require multiple models or rely on a more capable model or other forms of supervision. To this end, we develop a multi-turn online reinforcement learning (RL) approach, SCoRe, that significantly improves an LLM's self-correction ability using entirely self-generated data. To build SCoRe, we first show that variants of supervised fine-tuning (SFT) on offline model-generated correction traces are insufficient for instilling self-correction behavior. In particular, we observe that training via SFT either suffers from a distribution mismatch between the training data and the model's own responses or implicitly prefers only a certain mode of correction behavior that is often not effective at test time. SCoRe addresses these challenges by training under the model's own distribution of self-generated correction traces and using appropriate regularization to steer the learning process into learning a self-correction strategy that is effective at test time as opposed to simply fitting high-reward responses for a given prompt. This regularization prescribes running a first phase of RL on a base model to generate a policy initialization that is less susceptible to collapse and then using a reward bonus to amplify self-correction during training. When applied to Gemini 1.0 Pro and 1.5 Flash models, we find that SCoRe achieves state-of-the-art self-correction performance, improving the base models' self-correction by 15.6% and 9.1% respectively on the MATH and HumanEval benchmarks.
     </details>

40. **CodePlan: Unlocking Reasoning Potential in Large Langauge Models by Scaling Code-form Planning** [[pdf]](http://arxiv.org/abs/2409.12452) `2024-09-19` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Despite the remarkable success of large language models (LLMs) on traditional natural language processing tasks, their planning ability remains a critical bottleneck in tackling complex multi-step reasoning tasks. Existing approaches mainly rely on prompting or task-specific fine-tuning, often suffering from weak robustness and cross-task generalization. To address the limitation, we introduce CODEPLAN, a scalable paradigm that empowers LLMs to generate and follow code-form plans pseudocode that outlines high-level, structured reasoning processes. By leveraging the structured and versatile nature of code, CODEPLAN effectively captures the rich semantics and control flows inherent to sophisticated reasoning. Importantly, CODEPLAN allows the automatic extraction of code-form plans from massive, wide-ranging text corpora without the need for curated, task-specific datasets. This enables it to scale up efficiently and improve reasoning capabilities across diverse scenarios. To train CODEPLAN, we construct a large-scale dataset of 2M examples that integrate code-form plans with standard prompt-response pairs from existing corpora. With minimal computation overhead during both training and inference, CODEPLAN achieves a 25.1% relative improvement compared with directly generating responses, averaged across 13 challenging multi-step reasoning benchmarks, spanning mathematical reasoning, symbolic reasoning, instruction-following, multi-hop QA, and decision-making tasks. Further analysis reveals CODEPLAN's increasing performance gains on more complex reasoning tasks, as well as significant data efficiency thanks to its generalization ability.
     </details>

41. **ControlMath: Controllable Data Generation Promotes Math Generalist Models** [[pdf]](http://arxiv.org/abs/2409.15376) `2024-09-19` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          ControlMath is proposed, an iterative method involving an equation-generator module and two LLM-based agents that enables the generation of diverse math problems, not limited to specific domains or distributions.
     </details>


     <details>
          <summary>Abstract</summary>
          Utilizing large language models (LLMs) for data augmentation has yielded encouraging results in mathematical reasoning. However, these approaches face constraints in problem diversity, potentially restricting them to in-domain/distribution data generation. To this end, we propose ControlMath, an iterative method involving an equation-generator module and two LLM-based agents. The module creates diverse equations, which the Problem-Crafter agent then transforms into math word problems. The Reverse-Agent filters and selects high-quality data, adhering to the "less is more" principle, achieving better results with fewer data points. This approach enables the generation of diverse math problems, not limited to specific domains or distributions. As a result, we collect ControlMathQA, which involves 190k math word problems. Extensive results prove that combining our dataset with in-domain datasets like GSM8K can help improve the model's mathematical ability to generalize, leading to improved performances both within and beyond specific domains.
     </details>

42. **InfiMM-WebMath-40B: Advancing Multimodal Pre-Training for Enhanced Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2409.12568) `2024-09-19` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces InfiMM-WebMath-40B, a high-quality dataset of interleaved image-text documents that significantly enhances the performance of the 1.3B model, and sets a new state-of-the-art among open-source models on multi-modal math benchmarks such as MathVerse and The authors-Math.
     </details>


     <details>
          <summary>Abstract</summary>
          Pre-training on large-scale, high-quality datasets is crucial for enhancing the reasoning capabilities of Large Language Models (LLMs), especially in specialized domains such as mathematics. Despite the recognized importance, the Multimodal LLMs (MLLMs) field currently lacks a comprehensive open-source pre-training dataset specifically designed for mathematical reasoning. To address this gap, we introduce InfiMM-WebMath-40B, a high-quality dataset of interleaved image-text documents. It comprises 24 million web pages, 85 million associated image URLs, and 40 billion text tokens, all meticulously extracted and filtered from CommonCrawl. We provide a detailed overview of our data collection and processing pipeline. To demonstrate the robustness of InfiMM-WebMath-40B, we conducted evaluations in both text-only and multimodal settings. Our evaluations on text-only benchmarks show that, despite utilizing only 40 billion tokens, our dataset significantly enhances the performance of our 1.3B model, delivering results comparable to DeepSeekMath-1.3B, which uses 120 billion tokens for the same model size. Nevertheless, with the introduction of our multi-modal math pre-training dataset, our models set a new state-of-the-art among open-source models on multi-modal math benchmarks such as MathVerse and We-Math. We release our data at https://huggingface.co/datasets/Infi-MM/InfiMM-WebMath-40B.
     </details>

43. **Iteration of Thought: Leveraging Inner Dialogue for Autonomous Large Language Model Reasoning** [[pdf]](http://arxiv.org/abs/2409.12618) `2024-09-19` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The Iteration of Thought (IoT) framework for enhancing LLM responses by generating "thought"-provoking prompts vis a vis an input query and the current iteration of an LLM's response is proposed.
     </details>


     <details>
          <summary>Abstract</summary>
          Iterative human engagement is a common and effective means of leveraging the advanced language processing power of large language models (LLMs). Using well-structured prompts in a conversational manner, human users can effectively influence an LLM to develop more thoughtful and accurate responses. Motivated by this insight, we propose the Iteration of Thought (IoT) framework for enhancing LLM responses by generating "thought"-provoking prompts vis a vis an input query and the current iteration of an LLM's response. Unlike static or semi-static approaches, e.g. Chain of Thought (CoT) or Tree of Thoughts (ToT), IoT adapts its reasoning path dynamically, based on evolving context, and without generating alternate explorative thoughts which are ultimately discarded. The three components of the IoT framework are (1) an Inner Dialogue Agent (IDA) responsible for generating instructive, context-specific prompts; (2) an LLM Agent (LLMA) that processes these prompts to refine its responses; and (3) an iterative prompting loop that implements a conversation between the former two components. We introduce two variants of our framework: Autonomous Iteration of Thought (AIoT), where an LLM decides when to stop iterating, and Guided Iteration of Thought (GIoT), which always forces a fixed number iterations. We investigate the performance of IoT across various datasets, spanning complex reasoning tasks from the GPQA dataset, explorative problem-solving in Game of 24, puzzle solving in Mini Crosswords, and multi-hop question answering from the HotpotQA dataset. Our results show that IoT represents a viable paradigm for autonomous response refinement in LLMs, showcasing significant improvements over CoT and thereby enabling more adaptive and efficient reasoning systems that minimize human intervention.
     </details>

44. **LogicPro: Improving Complex Logical Reasoning via Program-Guided Learning** [[pdf]](http://arxiv.org/abs/2409.12929) `2024-09-19` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This approach achieves significant improvements in multiple models for the BBH$^{27}$, GSM8K, HellSwag, Logicqa, Reclor, and RTE datasets, outperforming a wide range of existing reasoning datasets.
     </details>


     <details>
          <summary>Abstract</summary>
          In this paper, we present a novel approach, called LogicPro, to enhance Large Language Models (LLMs) complex Logical reasoning through Program Examples. We do this effectively by simply utilizing widely available algorithmic problems and their code solutions. First, we constructed diverse test samples input based on algorithmic questions and code solutions. Then, we designed different complex reasoning questions based on algorithmic problems and test samples. Finally, combining the intermediate variable outputs of the code solutions and the complex reasoning questions, we derived the reasoning process and the final answer. With this approach, we can construct a dataset that is sufficiently difficult (all models are ineffective), diverse (synthesized from 2,360 different algorithmic questions), and scalable (building different test samples and collecting more algorithmic questions). In addition, we obtain a high-quality reasoning process guided by the values of intermediate variables. As a result, our approach achieves significant improvements in multiple models for the BBH$^{27}$, GSM8K, HellSwag, Logicqa, Reclor, and RTE datasets, outperforming a wide range of existing reasoning datasets.
     </details>

45. **Textualized Agent-Style Reasoning for Complex Tasks by Multiple Round LLM Generation** [[pdf]](https://arxiv.org/abs/2409.12411v1) `2024-09-19` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work presents AgentCOT, a llm-based autonomous agent framework, which can solve complex problems in an agent-style manner by multiple round LLM generation by integrating the step's index into the reasoning process to form a graph structure for complex inference logic.
     </details>


     <details>
          <summary>Abstract</summary>
          Chain-of-thought prompting significantly boosts the reasoning ability of large language models but still faces three issues: hallucination problem, restricted interpretability, and uncontrollable generation. To address these challenges, we present AgentCOT, a llm-based autonomous agent framework, which can solve complex problems in an agent-style manner by multiple round LLM generation. At each step, AgentCOT selects an action and executes it to yield an intermediate result with supporting evidence. In addition, we integrate the step's index into the reasoning process to form a graph structure for complex inference logic. We introduce two new strategies to enhance the performance of AgentCOT.We conduct extensive experiments to verify the effectiveness of our method on six common benchmarks. Results exhibit that our method brings in substantial improvements over current competitive approaches.
     </details>

46. **To CoT or not to CoT? Chain-of-thought helps mainly on math and symbolic reasoning** [[pdf]](https://arxiv.org/abs/2409.12183v1) `2024-09-18` (5 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The results show that CoT gives strong performance benefits primarily on tasks involving math or logic, with much smaller gains on other types of tasks, and suggest a need to move beyond prompt-based CoT to new paradigms that better leverage intermediate computation across the whole range of LLM applications.
     </details>


     <details>
          <summary>Abstract</summary>
          Chain-of-thought (CoT) via prompting is the de facto method for eliciting reasoning capabilities from large language models (LLMs). But for what kinds of tasks is this extra ``thinking'' really helpful? To analyze this, we conducted a quantitative meta-analysis covering over 100 papers using CoT and ran our own evaluations of 20 datasets across 14 models. Our results show that CoT gives strong performance benefits primarily on tasks involving math or logic, with much smaller gains on other types of tasks. On MMLU, directly generating the answer without CoT leads to almost identical accuracy as CoT unless the question or model's response contains an equals sign, indicating symbolic operations and reasoning. Following this finding, we analyze the behavior of CoT on these problems by separating planning and execution and comparing against tool-augmented LLMs. Much of CoT's gain comes from improving symbolic execution, but it underperforms relative to using a symbolic solver. Our results indicate that CoT can be applied selectively, maintaining performance while saving inference costs. Furthermore, they suggest a need to move beyond prompt-based CoT to new paradigms that better leverage intermediate computation across the whole range of LLM applications.
     </details>

47. **Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement** [[pdf]](https://arxiv.org/abs/2409.12122v1) `2024-09-18` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A series of math-specific large language models that possess advanced mathematical reasoning capabilities, including Chain-of-Thought (CoT) and Tool-Integrated Reasoning (TIR), and are evaluated on 10 mathematics datasets in both English and Chinese.
     </details>


     <details>
          <summary>Abstract</summary>
          In this report, we present a series of math-specific large language models: Qwen2.5-Math and Qwen2.5-Math-Instruct-1.5B/7B/72B. The core innovation of the Qwen2.5 series lies in integrating the philosophy of self-improvement throughout the entire pipeline, from pre-training and post-training to inference: (1) During the pre-training phase, Qwen2-Math-Instruct is utilized to generate large-scale, high-quality mathematical data. (2) In the post-training phase, we develop a reward model (RM) by conducting massive sampling from Qwen2-Math-Instruct. This RM is then applied to the iterative evolution of data in supervised fine-tuning (SFT). With a stronger SFT model, it's possible to iteratively train and update the RM, which in turn guides the next round of SFT data iteration. On the final SFT model, we employ the ultimate RM for reinforcement learning, resulting in the Qwen2.5-Math-Instruct. (3) Furthermore, during the inference stage, the RM is used to guide sampling, optimizing the model's performance.   Qwen2.5-Math-Instruct supports both Chinese and English, and possess advanced mathematical reasoning capabilities, including Chain-of-Thought (CoT) and Tool-Integrated Reasoning (TIR). We evaluate our models on 10 mathematics datasets in both English and Chinese, such as GSM8K, MATH, GaoKao, AMC23, and AIME24, covering a range of difficulties from grade school level to math competition problems.
     </details>

48. **Enhancing Logical Reasoning in Large Language Models through Graph-based Synthetic Data** [[pdf]](http://arxiv.org/abs/2409.12437) `2024-09-18` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is demonstrated that supervised fine-tuning (SFT) with synthetic graph-based reasoning data effectively enhances LLMs' reasoning performance without compromising their effectiveness on other standard evaluation benchmarks.
     </details>


     <details>
          <summary>Abstract</summary>
          Despite recent advances in training and prompting strategies for Large Language Models (LLMs), these models continue to face challenges with complex logical reasoning tasks that involve long reasoning chains. In this work, we explore the potential and limitations of using graph-based synthetic reasoning data as training signals to enhance LLMs' reasoning capabilities. Our extensive experiments, conducted on two established natural language reasoning tasks -- inductive reasoning and spatial reasoning -- demonstrate that supervised fine-tuning (SFT) with synthetic graph-based reasoning data effectively enhances LLMs' reasoning performance without compromising their effectiveness on other standard evaluation benchmarks.
     </details>

49. **MAgICoRe: Multi-Agent, Iterative, Coarse-to-Fine Refinement for Reasoning** [[pdf]](https://arxiv.org/abs/2409.12147v1) `2024-09-18` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes MAgICoRe, which avoids excessive refinement by categorizing problem difficulty as easy or hard, solving easy problems with coarse-grained aggregation and hard ones with fine-grained and iterative multi-agent refinement, and employs a multi-agent loop with three agents.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models' (LLM) reasoning can be improved using test-time aggregation strategies, i.e., generating multiple samples and voting among generated samples. While these improve performance, they often reach a saturation point. Refinement offers an alternative by using LLM-generated feedback to improve solution quality. However, refinement introduces 3 key challenges: (1) Excessive refinement: Uniformly refining all instances can over-correct and reduce the overall performance. (2) Inability to localize and address errors: LLMs have a limited ability to self-correct and struggle to identify and correct their own mistakes. (3) Insufficient refinement: Deciding how many iterations of refinement are needed is non-trivial, and stopping too soon could leave errors unaddressed. To tackle these issues, we propose MAgICoRe, which avoids excessive refinement by categorizing problem difficulty as easy or hard, solving easy problems with coarse-grained aggregation and hard ones with fine-grained and iterative multi-agent refinement. To improve error localization, we incorporate external step-wise reward model (RM) scores. Moreover, to ensure effective refinement, we employ a multi-agent loop with three agents: Solver, Reviewer (which generates targeted feedback based on step-wise RM scores), and the Refiner (which incorporates feedback). To ensure sufficient refinement, we re-evaluate updated solutions, iteratively initiating further rounds of refinement. We evaluate MAgICoRe on Llama-3-8B and GPT-3.5 and show its effectiveness across 5 math datasets. Even one iteration of MAgICoRe beats Self-Consistency by 3.4%, Best-of-k by 3.2%, and Self-Refine by 4.0% while using less than half the samples. Unlike iterative refinement with baselines, MAgICoRe continues to improve with more iterations. Finally, our ablations highlight the importance of MAgICoRe's RMs and multi-agent communication.
     </details>

50. **Small Language Models are Equation Reasoners** [[pdf]](http://arxiv.org/abs/2409.12393) `2024-09-18` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper investigates why sLM perform poorly on arithmetic reasoning tasks and hypothesizes that natural language format variability introduces high ambiguity for these smaller models, and conducts experiments with equation-only format, which is a reasoning format that unifies arithmetic reasoning previously expressed in natural language formats into mathematical equations.
     </details>


     <details>
          <summary>Abstract</summary>
          Chain-of-Thought (CoT) reasoning has enabled Large Language Model (LLM) to achieve remarkable performance in various NLP tasks, including arithmetic problem-solving. However, this success does not generalize to small language model (sLM) like T5, due to their limited capacity and absence of emergent abilities associated with larger models. Recent works to enhance sLM through knowledge distillation have yielded some improvements but still face significant limitations, particularly high ambiguity from the variability in natural language expressions and substantial computational costs. In this paper, we investigate why sLM perform poorly on arithmetic reasoning tasks and hypothesize that natural language format variability introduces high ambiguity for these smaller models. Based on this hypothesis, we conduct experiments with equation-only format, which is a reasoning format that unifies arithmetic reasoning previously expressed in natural language formats into mathematical equations. Experiment results demonstrate that equation-only format effectively boosts the arithmetic reasoning abilities of sLM, especially in very small models like T5-Tiny.
     </details>

51. **Improving LLM Reasoning with Multi-Agent Tree-of-Thought Validator Agent** [[pdf]](https://arxiv.org/abs/2409.11527v1) `2024-09-17` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces a novel approach combining ToT-based Reasoner agents with a Thought Validator agent, enabling a more robust voting strategy by discarding faulty reasoning paths, enhancing the system's ability to tackle tasks requiring systematic and trustworthy reasoning.
     </details>


     <details>
          <summary>Abstract</summary>
          Multi-agent strategies have emerged as a promising approach to enhance the reasoning abilities of Large Language Models (LLMs) by assigning specialized roles in the problem-solving process. Concurrently, Tree of Thoughts (ToT) methods have shown potential in improving reasoning for complex question-answering tasks by exploring diverse reasoning paths. A critical limitation in multi-agent reasoning is the 'Reasoner' agent's shallow exploration of reasoning paths. While ToT strategies could help mitigate this problem, they may generate flawed reasoning branches, which could harm the trustworthiness of the final answer. To leverage the strengths of both multi-agent reasoning and ToT strategies, we introduce a novel approach combining ToT-based Reasoner agents with a Thought Validator agent. Multiple Reasoner agents operate in parallel, employing ToT to explore diverse reasoning paths. The Thought Validator then scrutinizes these paths, considering a Reasoner's conclusion only if its reasoning is valid. This method enables a more robust voting strategy by discarding faulty reasoning paths, enhancing the system's ability to tackle tasks requiring systematic and trustworthy reasoning. Our method demonstrates superior performance compared to existing techniques when evaluated on the GSM8K dataset, outperforming the standard ToT strategy by an average 5.6\% across four LLMs.
     </details>

52. **Reasoning Graph Enhanced Exemplars Retrieval for In-Context Learning** [[pdf]](http://arxiv.org/abs/2409.11147) `2024-09-17` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel method named Reasoning Graph-enhanced Exemplar Retrieval (RGER), which first quires LLM to generate an initial response, then expresses intermediate problem-solving steps to a graph structure, and employs graph kernel to select exemplars with semantic and structural similarity.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models(LLMs) have exhibited remarkable few-shot learning capabilities and unified the paradigm of NLP tasks through the in-context learning(ICL) technique. Despite the success of ICL, the quality of the exemplar demonstrations can significantly influence the LLM's performance. Existing exemplar selection methods mainly focus on the semantic similarity between queries and candidate exemplars. On the other hand, the logical connections between reasoning steps can be beneficial to depict the problem-solving process as well. In this paper, we proposes a novel method named Reasoning Graph-enhanced Exemplar Retrieval(RGER). RGER first quires LLM to generate an initial response, then expresses intermediate problem-solving steps to a graph structure. After that, it employs graph kernel to select exemplars with semantic and structural similarity. Extensive experiments demonstrate the structural relationship is helpful to the alignment of queries and candidate exemplars. The efficacy of RGER on math and logit reasoning tasks showcases its superiority over state-of-the-art retrieval-based approaches. Our code is released at https://github.com/Yukang-Lin/RGER.
     </details>

53. **RoMath: A Mathematical Reasoning Benchmark in Romanian** [[pdf]](http://arxiv.org/abs/2409.11074) `2024-09-17` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Mathematics has long been conveyed through natural language, primarily for human understanding. With the rise of mechanized mathematics and proof assistants, there is a growing need to understand informal mathematical text, yet most existing benchmarks focus solely on English, overlooking other languages. This paper introduces RoMath, a Romanian mathematical reasoning benchmark suite comprising three datasets: RoMath-Baccalaureate, RoMath-Competitions and RoMath-Synthetic, which cover a range of mathematical domains and difficulty levels, aiming to improve non-English language models and promote multilingual AI development. By focusing on Romanian, a low-resource language with unique linguistic features, RoMath addresses the limitations of Anglo-centric models and emphasizes the need for dedicated resources beyond simple automatic translation. We benchmark several open-weight language models, highlighting the importance of creating resources for underrepresented languages. We make the code and dataset available.
     </details>

54. **On the Diagram of Thought** [[pdf]](http://arxiv.org/abs/2409.10038) `2024-09-16` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Diagram of Thought is introduced, a framework that models iterative reasoning in large language models (LLMs) as the construction of a directed acyclic graph (DAG) within a single model, eliminating the need for multiple models or external control mechanisms.
     </details>


     <details>
          <summary>Abstract</summary>
          We introduce Diagram of Thought (DoT), a framework that models iterative reasoning in large language models (LLMs) as the construction of a directed acyclic graph (DAG) within a single model. Unlike traditional approaches that represent reasoning as linear chains or trees, DoT organizes propositions, critiques, refinements, and verifications into a cohesive DAG structure, allowing the model to explore complex reasoning pathways while maintaining logical consistency. Each node in the diagram corresponds to a proposition that has been proposed, critiqued, refined, or verified, enabling the LLM to iteratively improve its reasoning through natural language feedback. By leveraging auto-regressive next-token prediction with role-specific tokens, DoT facilitates seamless transitions between proposing ideas and critically evaluating them, providing richer feedback than binary signals. Furthermore, we formalize the DoT framework using Topos Theory, providing a mathematical foundation that ensures logical consistency and soundness in the reasoning process. This approach enhances both the training and inference processes within a single LLM, eliminating the need for multiple models or external control mechanisms. DoT offers a conceptual framework for designing next-generation reasoning-specialized models, emphasizing training efficiency, robust reasoning capabilities, and theoretical grounding. The code is available at https://github.com/diagram-of-thought/diagram-of-thought.
     </details>

55. **CPL: Critical Planning Step Learning Boosts LLM Generalization in Reasoning Tasks** [[pdf]](http://arxiv.org/abs/2409.08642) `2024-09-13` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Post-training large language models (LLMs) to develop reasoning capabilities has proven effective across diverse domains, such as mathematical reasoning and code generation. However, existing methods primarily focus on improving task-specific reasoning but have not adequately addressed the model's generalization capabilities across a broader range of reasoning tasks. To tackle this challenge, we introduce Critical Planning Step Learning (CPL), which leverages Monte Carlo Tree Search (MCTS) to explore diverse planning steps in multi-step reasoning tasks. Based on long-term outcomes, CPL learns step-level planning preferences to improve the model's planning capabilities and, consequently, its general reasoning capabilities. Furthermore, while effective in many scenarios for aligning LLMs, existing preference learning approaches like Direct Preference Optimization (DPO) struggle with complex multi-step reasoning tasks due to their inability to capture fine-grained supervision at each step. We propose Step-level Advantage Preference Optimization (Step-APO), which integrates an advantage estimate for step-level preference pairs obtained via MCTS into the DPO. This enables the model to more effectively learn critical intermediate planning steps, thereby further improving its generalization in reasoning tasks. Experimental results demonstrate that our method, trained exclusively on GSM8K and MATH, not only significantly improves performance on GSM8K (+10.5%) and MATH (+6.5%), but also enhances out-of-domain reasoning benchmarks, such as ARC-C (+4.0%), BBH (+1.8%), MMLU-STEM (+2.2%), and MMLU (+0.9%).
     </details>

56. **Expediting and Elevating Large Language Model Reasoning via Hidden Chain-of-Thought Decoding** [[pdf]](http://arxiv.org/abs/2409.08561) `2024-09-13` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a novel approach to compress the CoT process through semantic alignment, enabling more efficient decoding while preserving the benefits of CoT reasoning and paves the way for more efficient exploitation of multi-step reasoning capabilities in LLMs across a wide range of applications.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have demonstrated remarkable capabilities in tasks requiring reasoning and multi-step problem-solving through the use of chain-of-thought (CoT) prompting. However, generating the full CoT process results in significantly longer output sequences, leading to increased computational costs and latency during inference. To address this challenge, we propose a novel approach to compress the CoT process through semantic alignment, enabling more efficient decoding while preserving the benefits of CoT reasoning. Our method introduces an auxiliary CoT model that learns to generate and compress the full thought process into a compact special token representation semantically aligned with the original CoT output. This compressed representation is then integrated into the input of the Hidden Chain-of-Thought (HCoT) model. The training process follows a two-stage procedure: First, the CoT model is optimized to generate the compressed token representations aligned with the ground-truth CoT outputs using a contrastive loss. Subsequently, with the CoT model parameters frozen, the HCoT model is fine-tuned to generate accurate subsequent predictions conditioned on the prefix instruction and the compressed CoT representations from the CoT model. Extensive experiments across three challenging domains - mathematical reasoning, agent invocation, and question answering - demonstrate that our semantic compression approach achieves competitive or improved performance compared to the full CoT baseline, while providing significant speedups of at least 1.5x in decoding time. Moreover, incorporating contrastive learning objectives further enhances the quality of the compressed representations, leading to better CoT prompting and improved task accuracy. Our work paves the way for more efficient exploitation of multi-step reasoning capabilities in LLMs across a wide range of applications.
     </details>

57. **Can We Count on LLMs? The Fixed-Effect Fallacy and Claims of GPT-4 Capabilities** [[pdf]](http://arxiv.org/abs/2409.07638) `2024-09-11` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Evaluation of LLM capabilities is explored, finding that efforts to quantify LLM capabilities easily succumb to the language-as-fixed-effect fallacy, where experimental observations are improperly generalized beyond what the data supports.
     </details>


     <details>
          <summary>Abstract</summary>
          In this paper we explore evaluation of LLM capabilities. We present measurements of GPT-4 performance on several deterministic tasks; each task involves a basic calculation and takes as input parameter some element drawn from a large well-defined population (e.g., count elements in a list, multiply two k-digit numbers, etc). We examine several conditions per-task and perform enough trials so that statistically significant differences can be detected. This allows us to investigate the sensitivity of task-accuracy both to query phrasing and input parameter population. We find that seemingly trivial modifications in the task-prompt or input population can yield differences far larger than can be explained by sampling effects. For example, performance on a simple list-counting task varies with query-phrasing and list-length, but also with list composition (i.e., the thing-to-be-counted) and object frequency (e.g., success when an element accounts for $\approx$ 50\% of a list is different from when it accounts for $\approx$ 70\% etc).   We conclude that efforts to quantify LLM capabilities easily succumb to the language-as-fixed-effect fallacy, where experimental observations are improperly generalized beyond what the data supports. A consequence appears to be that intuitions that have been formed based on interactions with humans form a very unreliable guide as to which input modifications should ``make no difference'' to LLM performance.
     </details>

58. **MathGLM-Vision: Solving Mathematical Problems with Multi-Modal Large Language Model** [[pdf]](http://arxiv.org/abs/2409.13729) `2024-09-09` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have demonstrated significant capabilities in mathematical reasoning, particularly with text-based mathematical problems. However, current multi-modal large language models (MLLMs), especially those specialized in mathematics, tend to focus predominantly on solving geometric problems but ignore the diversity of visual information available in other areas of mathematics. Moreover, the geometric information for these specialized mathematical MLLMs is derived from several public datasets, which are typically limited in diversity and complexity. To address these limitations, we aim to construct a fine-tuning dataset named MathVL, and develop a series of specialized mathematical MLLMs termed MathGLM-Vision by conducting Supervised Fine-Tuning (SFT) on MathVL with various parameter-scale backbones. To extensively evaluate the effectiveness of MathGLM-Vision, we conduct experiments on several public benchmarks and our curated MathVL-test consisting of 2,000 problems. Experimental results demonstrate that MathGLM-Vision achieves significant improvements compared with some existing models, including backbone models and open-source mathematical MLLMs. These findings indicate the importance of diversity dataset in enhancing the mathematical reasoning abilities of MLLMs.
     </details>

59. **VisScience: An Extensive Benchmark for Evaluating K12 Educational Multi-modal Scientific Reasoning** [[pdf]](http://arxiv.org/abs/2409.13730) `2024-09-09` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A comprehensive benchmark is constructed, named VisScience, which is utilized to assess the multi-modal scientific reasoning across the three disciplines of mathematics, physics, and chemistry and demonstrates that closed-source MLLMs generally outperform open-source models.
     </details>


     <details>
          <summary>Abstract</summary>
          Multi-modal large language models (MLLMs) have demonstrated promising capabilities across various tasks by integrating textual and visual information to achieve visual understanding in complex scenarios. Despite the availability of several benchmarks aims to evaluating MLLMs in tasks from visual question answering to complex problem-solving, most focus predominantly on mathematics or general visual understanding tasks. This reveals a critical gap in current benchmarks, which often overlook the inclusion of other key scientific disciplines such as physics and chemistry. To address this gap, we meticulously construct a comprehensive benchmark, named VisScience, which is utilized to assess the multi-modal scientific reasoning across the three disciplines of mathematics, physics, and chemistry. This benchmark comprises 3,000 questions drawn from K12 education - spanning elementary school through high school - equally distributed across three disciplines, with 1,000 questions per discipline. The questions within VisScience span 21 distinct subjects and are categorized into five difficulty levels, offering a broad spectrum of topics within each discipline. With VisScience, we present a detailed evaluation of the performance of 25 representative MLLMs in scientific reasoning. Experimental results demonstrate that closed-source MLLMs generally outperform open-source models. The best performance observed include a 53.4\% accuracy in mathematics by Claude3.5-Sonnet, 38.2\% in physics by GPT-4o, and 47.0\% in chemistry by Gemini-1.5-Pro. These results underscore the strengths and limitations of MLLMs, suggesting areas for future improvement and highlighting the importance of developing models that can effectively handle the diverse demands of multi-modal scientific reasoning.
     </details>

60. **Diagram Formalization Enhanced Multi-Modal Geometry Problem Solver** [[pdf]](http://arxiv.org/abs/2409.04214) `2024-09-06` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The Diagram Formalization Enhanced Geometry Problem Solver (DFE-GPS) is introduced, a new framework that integrates visual features, geometric formal language, and natural language representations that improves MLLMs' ability to process geometric diagrams and extends their application to open-ended tasks on the formalgeo7k dataset.
     </details>


     <details>
          <summary>Abstract</summary>
          Mathematical reasoning remains an ongoing challenge for AI models, especially for geometry problems that require both linguistic and visual signals. As the vision encoders of most MLLMs are trained on natural scenes, they often struggle to understand geometric diagrams, performing no better in geometry problem solving than LLMs that only process text. This limitation is amplified by the lack of effective methods for representing geometric relationships. To address these issues, we introduce the Diagram Formalization Enhanced Geometry Problem Solver (DFE-GPS), a new framework that integrates visual features, geometric formal language, and natural language representations. We propose a novel synthetic data approach and create a large-scale geometric dataset, SynthGeo228K, annotated with both formal and natural language captions, designed to enhance the vision encoder for a better understanding of geometric structures. Our framework improves MLLMs' ability to process geometric diagrams and extends their application to open-ended tasks on the formalgeo7k dataset.
     </details>

61. **From Calculation to Adjudication: Examining LLM judges on Mathematical Reasoning Tasks** [[pdf]](http://arxiv.org/abs/2409.04168) `2024-09-06` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The analysis uncovers a strong correlation between judgment performance and the candidate model task performance and shows that regularities in the judgments are quantifiable using statistical measures and provides various angles on exploiting them.
     </details>


     <details>
          <summary>Abstract</summary>
          To reduce the need for human annotations, large language models (LLMs) have been proposed as judges of the quality of other candidate models. LLM judges are typically evaluated by measuring the correlation with human judgments on generation tasks such as summarization or machine translation. In contrast, we study LLM judges on mathematical reasoning tasks. These tasks require multi-step reasoning, and the correctness of their solutions is verifiable, enabling a more objective evaluation. We perform a detailed performance analysis and find that the used judges are mostly unable to improve task performance but are able to pick the better model. Our analysis uncovers a strong correlation between judgment performance and the candidate model task performance. We observe that judges tend to choose the model of higher quality even if its answer is incorrect. Further, we show that it is possible to use statistics, such as the task performances of the individual models, to predict judgment performance. In an ablation, we either swap or mask the candidate answers and observe that judges often keep the original judgment, providing evidence that judges incorporate writing style in their judgments. In summary, we find that regularities in the judgments are quantifiable using statistical measures and provide various angles on exploiting them.
     </details>

62. **Self-Harmonized Chain of Thought** [[pdf]](http://arxiv.org/abs/2409.04057) `2024-09-06` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          ECHO is proposed, a self-harmonized chain-of-thought prompting method that consolidates diverse solution paths into a uniform and effective solution pattern and demonstrates the best overall performance across three reasoning domains.
     </details>


     <details>
          <summary>Abstract</summary>
          Chain-of-Thought (CoT) prompting reveals that large language models are capable of performing complex reasoning via intermediate steps. CoT prompting is primarily categorized into three approaches. The first approach utilizes straightforward prompts like ``Let's think step by step'' to generate a sequential thought process before yielding an answer. The second approach makes use of human-crafted, step-by-step demonstrations to guide the model's reasoning process. The third automates the generation of reasoned demonstrations with the 'Let's think step by step'.This approach sometimes leads to reasoning errors, highlighting the need to diversify demonstrations to mitigate its misleading effects. However, diverse demonstrations pose challenges for effective representations. In this work, we propose ECHO, a self-harmonized chain-of-thought prompting method. It consolidates diverse solution paths into a uniform and effective solution pattern.ECHO demonstrates the best overall performance across three reasoning domains.
     </details>

63. **Strategic Chain-of-Thought: Guiding Accurate Reasoning in LLMs through Strategy Elicitation** [[pdf]](http://arxiv.org/abs/2409.03271) `2024-09-05` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The SCoT framework is extended to develop a few-shot method with automatically matched demonstrations, yielding even stronger results, underscore the efficacy of SCoT, highlighting its potential to substantially enhance LLM performance in complex reasoning tasks.
     </details>


     <details>
          <summary>Abstract</summary>
          The Chain-of-Thought (CoT) paradigm has emerged as a critical approach for enhancing the reasoning capabilities of large language models (LLMs). However, despite their widespread adoption and success, CoT methods often exhibit instability due to their inability to consistently ensure the quality of generated reasoning paths, leading to sub-optimal reasoning performance. To address this challenge, we propose the \textbf{Strategic Chain-of-Thought} (SCoT), a novel methodology designed to refine LLM performance by integrating strategic knowledge prior to generating intermediate reasoning steps. SCoT employs a two-stage approach within a single prompt: first eliciting an effective problem-solving strategy, which is then used to guide the generation of high-quality CoT paths and final answers. Our experiments across eight challenging reasoning datasets demonstrate significant improvements, including a 21.05\% increase on the GSM8K dataset and 24.13\% on the Tracking\_Objects dataset, respectively, using the Llama3-8b model. Additionally, we extend the SCoT framework to develop a few-shot method with automatically matched demonstrations, yielding even stronger results. These findings underscore the efficacy of SCoT, highlighting its potential to substantially enhance LLM performance in complex reasoning tasks.
     </details>

64. **Building Math Agents with Multi-Turn Iterative Preference Learning** [[pdf]](https://arxiv.org/abs/2409.02392v1) `2024-09-04` (5 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces a multi-turn direct preference learning framework, tailored for this context, that leverages feedback from code interpreters and optimizes trajectory-level preferences and includes multi-turn DPO and multi-turn KTO as specific implementations.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent studies have shown that large language models' (LLMs) mathematical problem-solving capabilities can be enhanced by integrating external tools, such as code interpreters, and employing multi-turn Chain-of-Thought (CoT) reasoning. While current methods focus on synthetic data generation and Supervised Fine-Tuning (SFT), this paper studies the complementary direct preference learning approach to further improve model performance. However, existing direct preference learning algorithms are originally designed for the single-turn chat task, and do not fully address the complexities of multi-turn reasoning and external tool integration required for tool-integrated mathematical reasoning tasks. To fill in this gap, we introduce a multi-turn direct preference learning framework, tailored for this context, that leverages feedback from code interpreters and optimizes trajectory-level preferences. This framework includes multi-turn DPO and multi-turn KTO as specific implementations. The effectiveness of our framework is validated through training of various language models using an augmented prompt set from the GSM8K and MATH datasets. Our results demonstrate substantial improvements: a supervised fine-tuned Gemma-1.1-it-7B model's performance increased from 77.5% to 83.9% on GSM8K and from 46.1% to 51.2% on MATH. Similarly, a Gemma-2-it-9B model improved from 84.1% to 86.3% on GSM8K and from 51.0% to 54.5% on MATH.
     </details>

65. **CMM-Math: A Chinese Multimodal Math Dataset To Evaluate and Enhance the Mathematics Reasoning of Large Multimodal Models** [[pdf]](https://arxiv.org/abs/2409.02834v1) `2024-09-04` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper releases a Chinese multimodal math dataset, including benchmark and training parts, to evaluate and enhance the mathematical reasoning of LMMs, and proposes a Multimodal Mathematical LMM (Math-LMM) to handle the problems with mixed input of multiple images and text segments.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have obtained promising results in mathematical reasoning, which is a foundational skill for human intelligence. Most previous studies focus on improving and measuring the performance of LLMs based on textual math reasoning datasets (e.g., MATH, GSM8K). Recently, a few researchers have released English multimodal math datasets (e.g., MATHVISTA and MATH-V) to evaluate the effectiveness of large multimodal models (LMMs). In this paper, we release a Chinese multimodal math (CMM-Math) dataset, including benchmark and training parts, to evaluate and enhance the mathematical reasoning of LMMs. CMM-Math contains over 28,000 high-quality samples, featuring a variety of problem types (e.g., multiple-choice, fill-in-the-blank, and so on) with detailed solutions across 12 grade levels from elementary to high school in China. Specifically, the visual context may be present in the questions or opinions, which makes this dataset more challenging. Through comprehensive analysis, we discover that state-of-the-art LMMs on the CMM-Math dataset face challenges, emphasizing the necessity for further improvements in LMM development. We also propose a Multimodal Mathematical LMM (Math-LMM) to handle the problems with mixed input of multiple images and text segments. We train our model using three stages, including foundational pre-training, foundational fine-tuning, and mathematical fine-tuning. The extensive experiments indicate that our model effectively improves math reasoning performance by comparing it with the SOTA LMMs over three multimodal mathematical datasets.
     </details>

66. **Deconfounded Causality-aware Parameter-Efficient Fine-Tuning for Problem-Solving Improvement of LLMs** [[pdf]](https://arxiv.org/abs/2409.02686v1) `2024-09-04` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Deconfounded Causal Adaptation (DCA), a novel parameter-efficient fine-tuning (PEFT) method to enhance the model's reasoning capabilities by encouraging the model to extract the general problem-solving skills and apply these skills to different questions.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have demonstrated remarkable efficiency in tackling various tasks based on human instructions, but studies reveal that they often struggle with tasks requiring reasoning, such as math or physics. This limitation raises questions about whether LLMs truly comprehend embedded knowledge or merely learn to replicate the token distribution without a true understanding of the content. In this paper, we delve into this problem and aim to enhance the reasoning capabilities of LLMs. First, we investigate if the model has genuine reasoning capabilities by visualizing the text generation process at the attention and representation level. Then, we formulate the reasoning process of LLMs into a causal framework, which provides a formal explanation of the problems observed in the visualization. Finally, building upon this causal framework, we propose Deconfounded Causal Adaptation (DCA), a novel parameter-efficient fine-tuning (PEFT) method to enhance the model's reasoning capabilities by encouraging the model to extract the general problem-solving skills and apply these skills to different questions. Experiments show that our method outperforms the baseline consistently across multiple benchmarks, and with only 1.2M tunable parameters, we achieve better or comparable results to other fine-tuning methods. This demonstrates the effectiveness and efficiency of our method in improving the overall accuracy and reliability of LLMs.
     </details>

67. **S$^3$c-Math: Spontaneous Step-level Self-correction Makes Large Language Models Better Mathematical Reasoners** [[pdf]](http://arxiv.org/abs/2409.01524) `2024-09-02` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Self-correction is a novel method that can stimulate the potential reasoning abilities of large language models (LLMs). It involves detecting and correcting errors during the inference process when LLMs solve reasoning problems. However, recent works do not regard self-correction as a spontaneous and intrinsic capability of LLMs. Instead, such correction is achieved through post-hoc generation, external knowledge introduction, multi-model collaboration, and similar techniques. In this paper, we propose a series of mathematical LLMs called S$^3$c-Math, which are able to perform Spontaneous Step-level Self-correction for Mathematical reasoning. This capability helps LLMs to recognize whether their ongoing inference tends to contain errors and simultaneously correct these errors to produce a more reliable response. We proposed a method, which employs a step-level sampling approach to construct step-wise self-correction data for achieving such ability. Additionally, we implement a training strategy that uses above constructed data to equip LLMs with spontaneous step-level self-correction capacities. Our data and methods have been demonstrated to be effective across various foundation LLMs, consistently showing significant progress in evaluations on GSM8K, MATH, and other mathematical benchmarks. To the best of our knowledge, we are the first to introduce the spontaneous step-level self-correction ability of LLMs in mathematical reasoning.
     </details>

68. **Dynamic Self-Consistency: Leveraging Reasoning Paths for Efficient LLM Sampling** [[pdf]](http://arxiv.org/abs/2408.17017) `2024-08-30` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Reasoning-Aware Self-Consistency (RASC), an innovative early-stopping framework that dynamically adjusts the number of sample generations by considering both the output answer and the RPs from Chain of Thought (CoT) prompting, is proposed.
     </details>


     <details>
          <summary>Abstract</summary>
          Self-Consistency (SC) is a widely used method to mitigate hallucinations in Large Language Models (LLMs) by sampling the LLM multiple times and outputting the most frequent solution. Despite its benefits, SC results in significant computational costs proportional to the number of samples generated. Previous early-stopping approaches, such as Early Stopping Self Consistency and Adaptive Consistency, have aimed to reduce these costs by considering output consistency, but they do not analyze the quality of the reasoning paths (RPs) themselves. To address this issue, we propose Reasoning-Aware Self-Consistency (RASC), an innovative early-stopping framework that dynamically adjusts the number of sample generations by considering both the output answer and the RPs from Chain of Thought (CoT) prompting. RASC assigns confidence scores sequentially to the generated samples, stops when certain criteria are met, and then employs weighted majority voting to optimize sample usage and enhance answer reliability. We comprehensively test RASC with multiple LLMs across varied QA datasets. RASC outperformed existing methods and significantly reduces sample usage by an average of 80% while maintaining or improving accuracy up to 5% compared to the original SC
     </details>

69. **MultiMath: Bridging Visual and Mathematical Reasoning for Large Language Models** [[pdf]](http://arxiv.org/abs/2409.00147) `2024-08-30` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A multimodal large language model that bridges the gap between math and vision, MultiMath-7B achieves state-of-the-art (SOTA) performance among open-source models on existing multimodal mathematical benchmarks and also excels on text-only mathematical benchmarks.
     </details>


     <details>
          <summary>Abstract</summary>
          The rapid development of large language models (LLMs) has spurred extensive research into their domain-specific capabilities, particularly mathematical reasoning. However, most open-source LLMs focus solely on mathematical reasoning, neglecting the integration with visual injection, despite the fact that many mathematical tasks rely on visual inputs such as geometric diagrams, charts, and function plots. To fill this gap, we introduce \textbf{MultiMath-7B}, a multimodal large language model that bridges the gap between math and vision. \textbf{MultiMath-7B} is trained through a four-stage process, focusing on vision-language alignment, visual and math instruction-tuning, and process-supervised reinforcement learning. We also construct a novel, diverse and comprehensive multimodal mathematical dataset, \textbf{MultiMath-300K}, which spans K-12 levels with image captions and step-wise solutions. MultiMath-7B achieves state-of-the-art (SOTA) performance among open-source models on existing multimodal mathematical benchmarks and also excels on text-only mathematical benchmarks. Our model and dataset are available at {\textcolor{blue}{\url{https://github.com/pengshuai-rin/MultiMath}}}.
     </details>

70. **Physics of Language Models: Part 2.2, How to Learn From Mistakes on Grade-School Math Problems** [[pdf]](http://arxiv.org/abs/2408.16293) `2024-08-29` (4 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Language models have demonstrated remarkable performance in solving reasoning tasks; however, even the strongest models still occasionally make reasoning mistakes. Recently, there has been active research aimed at improving reasoning accuracy, particularly by using pretrained language models to "self-correct" their mistakes via multi-round prompting. In this paper, we follow this line of work but focus on understanding the usefulness of incorporating "error-correction" data directly into the pretraining stage. This data consists of erroneous solution steps immediately followed by their corrections. Using a synthetic math dataset, we show promising results: this type of pretrain data can help language models achieve higher reasoning accuracy directly (i.e., through simple auto-regression, without multi-round prompting) compared to pretraining on the same amount of error-free data. We also delve into many details, such as (1) how this approach differs from beam search, (2) how such data can be prepared, (3) whether masking is needed on the erroneous tokens, (4) the amount of error required, (5) whether such data can be deferred to the fine-tuning stage, and many others.
     </details>

71. **Critic-CoT: Boosting the reasoning abilities of large language model via Chain-of-thoughts Critic** [[pdf]](http://arxiv.org/abs/2408.16326) `2024-08-29` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Critic-CoT is proposed, a novel framework that pushes LLMs toward System-2-like critic capability, via step-wise CoT reasoning format and distant-supervision data construction, without the need for human annotation.
     </details>


     <details>
          <summary>Abstract</summary>
          Self-critic has become an important mechanism for enhancing the reasoning performance of LLMs. However, current approaches mainly involve basic prompts without further training, which tend to be over-simplified, leading to limited accuracy.Moreover, there is a lack of in-depth investigation of the relationship between LLM's ability to criticism and its task-solving performance.To address these issues, we propose Critic-CoT, a novel framework that pushes LLMs toward System-2-like critic capability, via step-wise CoT reasoning format and distant-supervision data construction, without the need for human annotation. Experiments on GSM8K and MATH show that via filtering out invalid solutions or iterative refinement, our enhanced model boosts task-solving performance, which demonstrates the effectiveness of our method. Further, we find that training on critique and refinement alone improves the generation. We hope our work could shed light on future research on improving the reasoning and critic ability of LLMs.
     </details>

72. **Logic Contrastive Reasoning with Lightweight Large Language Model for Math Word Problems** [[pdf]](http://arxiv.org/abs/2409.00131) `2024-08-29` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel method for measuring mathematical logic similarity and an automatic screening mechanism to construct a set of reference problems that integrate both semantic and logical similarity are introduced, in the first attempt to utilize retrieval-enhanced generation for mathematical problem-solving.
     </details>


     <details>
          <summary>Abstract</summary>
          This study focuses on improving the performance of lightweight Large Language Models (LLMs) in mathematical reasoning tasks. We introduce a novel method for measuring mathematical logic similarity and design an automatic screening mechanism to construct a set of reference problems that integrate both semantic and logical similarity. By employing carefully crafted positive and negative example prompts, we guide the model towards adopting sound reasoning logic. To the best of our knowledge, this is the first attempt to utilize retrieval-enhanced generation for mathematical problem-solving. Experimental results demonstrate that our method achieves a 15.8% improvement over the Chain of Thought approach on the SVAMP dataset and a 21.5 % improvement on the GSM8K dataset. Further application of this method to a large-scale model with 175 billion parameters yields performance comparable to the best results on both aforementioned datasets. Finally, we conduct an analysis of errors during the reasoning process, providing valuable insights and directions for future research on reasoning tasks using large language models.
     </details>

73. **AutoGeo: Automating Geometric Image Dataset Creation for Enhanced Geometry Understanding** [[pdf]](http://arxiv.org/abs/2409.09039) `2024-08-28` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Experimental results indicate significant improvements in the model's ability in handling geometric images, as evidenced by enhanced accuracy in tasks such as geometric captioning and mathematical reasoning.
     </details>


     <details>
          <summary>Abstract</summary>
          With the rapid advancement of large language models, there has been a growing interest in their capabilities in mathematical reasoning. However, existing research has primarily focused on text-based algebra problems, neglecting the study of geometry due to the lack of high-quality geometric datasets. To address this gap, this paper introduces AutoGeo, a novel approach for automatically generating mathematical geometric images to fulfill the demand for large-scale and diverse geometric datasets. AutoGeo facilitates the creation of AutoGeo-100k, an extensive repository comprising 100k high-quality geometry image-text pairs. By leveraging precisely defined geometric clauses, AutoGeo-100k contains a wide variety of geometric shapes, including lines, polygons, circles, and complex spatial relationships, etc. Furthermore, this paper demonstrates the efficacy of AutoGeo-100k in enhancing the performance of multimodal large language models through fine-tuning. Experimental results indicate significant improvements in the model's ability in handling geometric images, as evidenced by enhanced accuracy in tasks such as geometric captioning and mathematical reasoning. This research not only fills a critical gap in the availability of geometric datasets but also paves the way for the advancement of sophisticated AI-driven tools in education and research. Project page: https://autogeo-official.github.io/.
     </details>

74. **SIaM: Self-Improving Code-Assisted Mathematical Reasoning of Large Language Models** [[pdf]](https://arxiv.org/abs/2408.15565v1) `2024-08-28` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a novel paradigm that uses a code-based critic model to guide steps including question-code data construction, quality control, and complementary evaluation and explores different alignment algorithms with self-generated instruction/preference data to foster continuous improvement.
     </details>


     <details>
          <summary>Abstract</summary>
          There is a growing trend of teaching large language models (LLMs) to solve mathematical problems through coding. Existing studies primarily focus on prompting powerful, closed-source models to generate seed training data followed by in-domain data augmentation, equipping LLMs with considerable capabilities for code-aided mathematical reasoning. However, continually training these models on augmented data derived from a few datasets such as GSM8K may impair their generalization abilities and restrict their effectiveness to a narrow range of question types. Conversely, the potential of improving such LLMs by leveraging large-scale, expert-written, diverse math question-answer pairs remains unexplored. To utilize these resources and tackle unique challenges such as code response assessment, we propose a novel paradigm that uses a code-based critic model to guide steps including question-code data construction, quality control, and complementary evaluation. We also explore different alignment algorithms with self-generated instruction/preference data to foster continuous improvement. Experiments across both in-domain (up to +5.7%) and out-of-domain (+4.4%) benchmarks in English and Chinese demonstrate the effectiveness of the proposed paradigm.
     </details>

75. **Generative Verifiers: Reward Modeling as Next-Token Prediction** [[pdf]](http://arxiv.org/abs/2408.15240) `2024-08-27` (8 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is demonstrated that when using Gemma-based verifiers on algorithmic and grade-school math reasoning tasks, GenRM outperforms discriminative verifiers and LLM-as-a-Judge, and it is shown that GenRM scales favorably across dataset size, model capacity, and inference-time compute.
     </details>


     <details>
          <summary>Abstract</summary>
          Verifiers or reward models are often used to enhance the reasoning performance of large language models (LLMs). A common approach is the Best-of-N method, where N candidate solutions generated by the LLM are ranked by a verifier, and the best one is selected. While LLM-based verifiers are typically trained as discriminative classifiers to score solutions, they do not utilize the text generation capabilities of pretrained LLMs. To overcome this limitation, we instead propose training verifiers using the ubiquitous next-token prediction objective, jointly on verification and solution generation. Compared to standard verifiers, such generative verifiers (GenRM) can benefit from several advantages of LLMs: they integrate seamlessly with instruction tuning, enable chain-of-thought reasoning, and can utilize additional inference-time compute via majority voting for better verification. We demonstrate that when using Gemma-based verifiers on algorithmic and grade-school math reasoning tasks, GenRM outperforms discriminative verifiers and LLM-as-a-Judge, showing a 16-64% improvement in the percentage of problems solved with Best-of-N. Furthermore, we show that GenRM scales favorably across dataset size, model capacity, and inference-time compute.
     </details>

76. **What makes math problems hard for reinforcement learning: a case study** [[pdf]](https://arxiv.org/abs/2408.15332v1) `2024-08-27` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Using a long-standing conjecture from combinatorial group theory, we explore, from multiple angles, the challenges of finding rare instances carrying disproportionately high rewards. Based on lessons learned in the mathematical context defined by the Andrews-Curtis conjecture, we propose algorithmic improvements that can be relevant in other domains with ultra-sparse reward problems. Although our case study can be formulated as a game, its shortest winning sequences are potentially $10^6$ or $10^9$ times longer than those encountered in chess. In the process of our study, we demonstrate that one of the potential counterexamples due to Akbulut and Kirby, whose status escaped direct mathematical methods for 39 years, is stably AC-trivial.
     </details>

77. **Multi-tool Integration Application for Math Reasoning Using Large Language Model** [[pdf]](https://arxiv.org/abs/2408.12148v1) `2024-08-22` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel multi tool application framework for mathematical reasoning is proposed, aiming to achieve more comprehensive and accurate mathematical reasoning by utilizing the collaborative effect of large language models (LLMs) and multiple external tools.
     </details>


     <details>
          <summary>Abstract</summary>
          Mathematical reasoning is an important research direction in the field of artificial intelligence. This article proposes a novel multi tool application framework for mathematical reasoning, aiming to achieve more comprehensive and accurate mathematical reasoning by utilizing the collaborative effect of large language models (LLMs) and multiple external tools. Firstly, use a Math Tool to perform basic mathematical calculations during the inference process through interaction with LLM. Secondly, Code Tool can generate code fragments that comply with syntax rules and execute them, providing support for complex mathematical problems. Then, through the iterative reasoning of the CoT Tool, the logical coherence and accuracy of mathematical reasoning are enhanced. Ultimately, by using self consistency tools to select the final answer based on different parameters, the consistency and reliability of reasoning are improved. Through the synergistic effect of these tools, the framework has achieved significant performance improvement in mathematical reasoning tasks. We conducted experiments on the NumGLUE Task 4 test set, which includes 220 mathematical reasoning fill in the blank questions. The experimental results showed that, based on Math Tool, Code Tool, and CoT Tool, in Task 4 task,our method achieved an accuracy of 89.09,compared with the GPT3+FewShot baseline, Few Shot+ERNIE-4.0+self consistency improved by 49.09%, and compared with fine-tuning the Fine tuning baseline, Few Shot+ERNIE-4.0+self consistency improved by 52.29%
     </details>

78. **Positional Description for Numerical Normalization** [[pdf]](https://arxiv.org/abs/2408.12430v1) `INTERSPEECH 2024 Speech Synthesis` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A Positional Description Scheme tailored for digit sequences, integrating placeholder value information for each digit, which effectively mitigates fatal numerical normalization errors in neural models, requiring only a modest amount of training data without rule-based Finite State Transducers.
     </details>


     <details>
          <summary>Abstract</summary>
          We present a Positional Description Scheme (PDS) tailored for digit sequences, integrating placeholder value information for each digit. Given the structural limitations of subword tokenization algorithms, language models encounter critical Text Normalization (TN) challenges [1] when handling numerical tasks. Our schema addresses this challenge through straightforward pre-processing, preserving the model architecture while significantly simplifying number normalization, rendering the problem tractable. This simplifies the task and facilitates more compact production-ready models capable of learning from smaller datasets. Furthermore, our investigations reveal that PDS enhances the arithmetic processing capabilities of language models, resulting in a relative accuracy improvement of 23% to 51% on complex arithmetic tasks. We demonstrate that PDS effectively mitigates fatal numerical normalization errors in neural models, requiring only a modest amount of training data without rule-based Finite State Transducers (FST). We demonstrate that PDS is essential for both the Text-To-Speech and Speech Recognition text processing, enabling effective TN under production constraints.
     </details>

79. **EAGLE: Elevating Geometric Reasoning through LLM-empowered Visual Instruction Tuning** [[pdf]](https://arxiv.org/abs/2408.11397v1) `2024-08-21` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper proposes EAGLE, a novel two-stage end-to-end visual enhancement MLLM framework designed to ElevAte Geometric reasoning through LLM-Empowered visual instruction tuning, and develops the geometry expert model EAGLE-7B.
     </details>


     <details>
          <summary>Abstract</summary>
          Multi-modal Large Language Models have recently experienced rapid developments and excel in various multi-modal tasks. However, they still struggle with mathematical geometric problem solving, which requires exceptional visual perception proficiency. Existing MLLMs mostly optimize the LLM backbone to acquire geometric reasoning capabilities, while rarely emphasizing improvements in visual comprehension. In this paper, we first investigate the visual perception performance of MLLMs when facing geometric diagrams. Our findings reveal that current MLLMs severely suffer from inaccurate geometric perception and hallucinations. To address these limitations, we propose EAGLE, a novel two-stage end-to-end visual enhancement MLLM framework designed to ElevAte Geometric reasoning through LLM-Empowered visual instruction tuning. Specifically, in the preliminary stage, we feed geometric image-caption pairs into our MLLM that contains a fully fine-tuning CLIP ViT and a frozen LLM, aiming to endow our model with basic geometric knowledge. In the subsequent advanced stage, we incorporate LoRA modules into the vision encoder and unfreeze the LLM backbone. This enables the model to leverage the inherent CoT rationales within question-answer pairs, guiding the MLLM to focus on nuanced visual cues and enhancing its overall perceptual capacity. Moreover, we optimize the cross-modal projector in both stages to foster adaptive visual-linguistic alignments. After the two-stage visual enhancement, we develop the geometry expert model EAGLE-7B. Extensive experiments on popular benchmarks demonstrate the effectiveness of our model. For example, on the GeoQA benchmark, EAGLE-7B not only surpasses the exemplary G-LLaVA 7B model by 2.9%, but also marginally outperforms the larger G-LLaVA 13B model. On the MathVista benchmark, EAGLE-7B achieves remarkable 3.8% improvements compared with the proprietary model GPT-4V.
     </details>

80. **Benchmarking Large Language Models for Math Reasoning Tasks** [[pdf]](https://arxiv.org/abs/2408.10839v1) `2024-08-20` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The results indicate that larger foundation models like GPT-4o and LLaMA 3-70B can solve mathematical reasoning independently from the concrete prompting strategy, while for smaller models the in-context learning approach significantly influences the performance.
     </details>


     <details>
          <summary>Abstract</summary>
          The use of Large Language Models (LLMs) in mathematical reasoning has become a cornerstone of related research, demonstrating the intelligence of these models and enabling potential practical applications through their advanced performance, such as in educational settings. Despite the variety of datasets and in-context learning algorithms designed to improve the ability of LLMs to automate mathematical problem solving, the lack of comprehensive benchmarking across different datasets makes it complicated to select an appropriate model for specific tasks. In this project, we present a benchmark that fairly compares seven state-of-the-art in-context learning algorithms for mathematical problem solving across five widely used mathematical datasets on four powerful foundation models. Furthermore, we explore the trade-off between efficiency and performance, highlighting the practical applications of LLMs for mathematical reasoning. Our results indicate that larger foundation models like GPT-4o and LLaMA 3-70B can solve mathematical reasoning independently from the concrete prompting strategy, while for smaller models the in-context learning approach significantly influences the performance. Moreover, the optimal prompt depends on the chosen foundation model. We open-source our benchmark code to support the integration of additional models in future research.
     </details>

81. **SubgoalXL: Subgoal-based Expert Learning for Theorem Proving** [[pdf]](https://arxiv.org/abs/2408.11172v1) `2024-08-20` `Isabelle` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          SubgoalXL is introduced, a novel approach that synergizes subgoal-based proofs with expert learning to enhance LLMs' capabilities in formal theorem proving within the Isabelle environment, and achieves a new state-of-the-art performance.
     </details>


     <details>
          <summary>Abstract</summary>
          Formal theorem proving, a field at the intersection of mathematics and computer science, has seen renewed interest with advancements in large language models (LLMs). This paper introduces SubgoalXL, a novel approach that synergizes subgoal-based proofs with expert learning to enhance LLMs' capabilities in formal theorem proving within the Isabelle environment. SubgoalXL addresses two critical challenges: the scarcity of specialized mathematics and theorem-proving data, and the need for improved multi-step reasoning abilities in LLMs. By optimizing data efficiency and employing subgoal-level supervision, SubgoalXL extracts richer information from limited human-generated proofs. The framework integrates subgoal-oriented proof strategies with an expert learning system, iteratively refining formal statement, proof, and subgoal generators. Leveraging the Isabelle environment's advantages in subgoal-based proofs, SubgoalXL achieves a new state-of-the-art performance of 56.1\% in Isabelle on the standard miniF2F dataset, marking an absolute improvement of 4.9\%. Notably, SubgoalXL successfully solves 41 AMC12, 9 AIME, and 3 IMO problems from miniF2F. These results underscore the effectiveness of maximizing limited data utility and employing targeted guidance for complex reasoning in formal theorem proving, contributing to the ongoing advancement of AI reasoning capabilities. The implementation is available at \url{https://github.com/zhaoxlpku/SubgoalXL}.
     </details>

82. **Concept Distillation from Strong to Weak Models via Hypotheses-to-Theories Prompting** [[pdf]](https://arxiv.org/abs/2408.09365v1) `2024-08-18` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Hand-crafting high quality prompts to optimize the performance of language models is a complicated and labor-intensive process. Furthermore, when migrating to newer, smaller, or weaker models (possibly due to latency or cost gains), prompts need to be updated to re-optimize the task performance. We propose Concept Distillation (CD), an automatic prompt optimization technique for enhancing weaker models on complex tasks. CD involves: (1) collecting mistakes made by weak models with a base prompt (initialization), (2) using a strong model to generate reasons for these mistakes and create rules/concepts for weak models (induction), and (3) filtering these rules based on validation set performance and integrating them into the base prompt (deduction/verification). We evaluated CD on NL2Code and mathematical reasoning tasks, observing significant performance boosts for small and weaker language models. Notably, Mistral-7B's accuracy on Multi-Arith increased by 20%, and Phi-3-mini-3.8B's accuracy on HumanEval rose by 34%. Compared to other automated methods, CD offers an effective, cost-efficient strategy for improving weak models' performance on complex tasks and enables seamless workload migration across different language models without compromising performance.
     </details>

83. **Math-PUMA: Progressive Upward Multimodal Alignment to Enhance Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2408.08640) `2024-08-16` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This approach is designed to improve the mathematical reasoning skills of MLLMs through a three-stage training process, with the second stage being the critical alignment stage and effectively narrows the performance gap for problems presented in different modalities.
     </details>


     <details>
          <summary>Abstract</summary>
          Multimodal Large Language Models (MLLMs) excel in solving text-based mathematical problems, but they struggle with mathematical diagrams since they are primarily trained on natural scene images. For humans, visual aids generally enhance problem-solving, but MLLMs perform worse as information shifts from textual to visual modality. This decline is mainly due to their shortcomings in aligning images and text. To tackle aforementioned challenges, we propose Math-PUMA, a methodology focused on Progressive Upward Multimodal Alignment. This approach is designed to improve the mathematical reasoning skills of MLLMs through a three-stage training process, with the second stage being the critical alignment stage. We first enhance the language model's mathematical reasoning capabilities with extensive set of textual mathematical problems. We then construct a multimodal dataset with varying degrees of textual and visual information, creating data pairs by presenting each problem in at least two forms. By leveraging the Kullback-Leibler (KL) divergence of next-token prediction distributions to align visual and textual modalities, consistent problem-solving abilities are ensured. Finally, we utilize multimodal instruction tuning for MLLMs with high-quality multimodal data. Experimental results on multiple mathematical reasoning benchmarks demonstrate that the MLLMs trained with Math-PUMA surpass most open-source MLLMs. Our approach effectively narrows the performance gap for problems presented in different modalities. The code and data are available at: \url{https://github.com/wwzhuang01/Math-PUMA}.
     </details>

84. **DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for Reinforcement Learning and Monte-Carlo Tree Search** [[pdf]](http://arxiv.org/abs/2408.08152) `2024-08-15` `Lean` (7 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          An open-source language model designed for theorem proving in Lean 4, which enhances DeepSeek-Prover-V1 by optimizing both training and inference processes, and proposes RMaxTS, a variant of Monte-Carlo tree search that employs an intrinsic-reward-driven exploration strategy to generate diverse proof paths.
     </details>


     <details>
          <summary>Abstract</summary>
          We introduce DeepSeek-Prover-V1.5, an open-source language model designed for theorem proving in Lean 4, which enhances DeepSeek-Prover-V1 by optimizing both training and inference processes. Pre-trained on DeepSeekMath-Base with specialization in formal mathematical languages, the model undergoes supervised fine-tuning using an enhanced formal theorem proving dataset derived from DeepSeek-Prover-V1. Further refinement is achieved through reinforcement learning from proof assistant feedback (RLPAF). Beyond the single-pass whole-proof generation approach of DeepSeek-Prover-V1, we propose RMaxTS, a variant of Monte-Carlo tree search that employs an intrinsic-reward-driven exploration strategy to generate diverse proof paths. DeepSeek-Prover-V1.5 demonstrates significant improvements over DeepSeek-Prover-V1, achieving new state-of-the-art results on the test set of the high school level miniF2F benchmark ($63.5\%$) and the undergraduate level ProofNet benchmark ($25.3\%$).
     </details>

85. **Automated Design of Agentic Systems** [[pdf]](http://arxiv.org/abs/2408.08435) `2024-08-15` (4 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work forms a new research area, Automated Design of Agentic Systems (ADAS), which aims to automatically create powerful agentic system designs, including inventing novel building blocks and/or combining them in new ways, and presents a simple yet effective algorithm named Meta Agent Search, which can progressively invent agents with novel designs that greatly outperform state-of-the-art hand-designed agents.
     </details>


     <details>
          <summary>Abstract</summary>
          Researchers are investing substantial effort in developing powerful general-purpose agents, wherein Foundation Models are used as modules within agentic systems (e.g. Chain-of-Thought, Self-Reflection, Toolformer). However, the history of machine learning teaches us that hand-designed solutions are eventually replaced by learned solutions. We formulate a new research area, Automated Design of Agentic Systems (ADAS), which aims to automatically create powerful agentic system designs, including inventing novel building blocks and/or combining them in new ways. We further demonstrate that there is an unexplored yet promising approach within ADAS where agents can be defined in code and new agents can be automatically discovered by a meta agent programming ever better ones in code. Given that programming languages are Turing Complete, this approach theoretically enables the learning of any possible agentic system: including novel prompts, tool use, control flows, and combinations thereof. We present a simple yet effective algorithm named Meta Agent Search to demonstrate this idea, where a meta agent iteratively programs interesting new agents based on an ever-growing archive of previous discoveries. Through extensive experiments across multiple domains including coding, science, and math, we show that our algorithm can progressively invent agents with novel designs that greatly outperform state-of-the-art hand-designed agents. Importantly, we consistently observe the surprising result that agents invented by Meta Agent Search maintain superior performance even when transferred across domains and models, demonstrating their robustness and generality. Provided we develop it safely, our work illustrates the potential of an exciting new research direction toward automatically designing ever-more powerful agentic systems to benefit humanity.
     </details>

86. **MathScape: Evaluating MLLMs in multimodal Math Scenarios through a Hierarchical Benchmark** [[pdf]](https://arxiv.org/abs/2408.07543v3) `2024-08-14` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A new benchmark that emphasizes the understanding and application of combined visual and textual information is proposed, MathScape, designed to evaluate photo-based math problem scenarios, assessing the theoretical understanding and application ability of MLLMs through a categorical hierarchical approach.
     </details>


     <details>
          <summary>Abstract</summary>
          With the development of Multimodal Large Language Models (MLLMs), the evaluation of multimodal models in the context of mathematical problems has become a valuable research field. Multimodal visual-textual mathematical reasoning serves as a critical indicator for evaluating the comprehension and complex multi-step quantitative reasoning abilities of MLLMs. However, previous multimodal math benchmarks have not sufficiently integrated visual and textual information. To address this gap, we proposed MathScape, a new benchmark that emphasizes the understanding and application of combined visual and textual information. MathScape is designed to evaluate photo-based math problem scenarios, assessing the theoretical understanding and application ability of MLLMs through a categorical hierarchical approach. We conduct a multi-dimensional evaluation on 11 advanced MLLMs, revealing that our benchmark is challenging even for the most sophisticated models. By analyzing the evaluation results, we identify the limitations of MLLMs, offering valuable insights for enhancing model performance.
     </details>

87. **Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers** [[pdf]](https://arxiv.org/abs/2408.06195v1) `2024-08-12` (6 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          This paper introduces rStar, a self-play mutual reasoning approach that significantly improves reasoning capabilities of small language models (SLMs) without fine-tuning or superior models. rStar decouples reasoning into a self-play mutual generation-discrimination process. First, a target SLM augments the Monte Carlo Tree Search (MCTS) with a rich set of human-like reasoning actions to construct higher quality reasoning trajectories. Next, another SLM, with capabilities similar to the target SLM, acts as a discriminator to verify each trajectory generated by the target SLM. The mutually agreed reasoning trajectories are considered mutual consistent, thus are more likely to be correct. Extensive experiments across five SLMs demonstrate rStar can effectively solve diverse reasoning problems, including GSM8K, GSM-Hard, MATH, SVAMP, and StrategyQA. Remarkably, rStar boosts GSM8K accuracy from 12.51% to 63.91% for LLaMA2-7B, from 36.46% to 81.88% for Mistral-7B, from 74.53% to 91.13% for LLaMA3-8B-Instruct. Code will be available at https://github.com/zhentingqi/rStar.
     </details>

88. **InfinityMATH: A Scalable Instruction Tuning Dataset in Programmatic Mathematical Reasoning** [[pdf]](https://arxiv.org/abs/2408.07089v1) `2024-08-09` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          InfinityMATH is introduced, a scalable instruction tuning dataset for programmatic mathematical reasoning that ensures that models are more versatile and effective across a broader range of mathematical problems.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent advancements in Chain-of-Thoughts (CoT) and Program-of-Thoughts (PoT) methods have greatly enhanced language models' mathematical reasoning capabilities, facilitating their integration into instruction tuning datasets with LLMs. However, existing methods for large-scale dataset creation require substantial seed data and high computational costs for data synthesis, posing significant challenges for scalability. We introduce InfinityMATH, a scalable instruction tuning dataset for programmatic mathematical reasoning. The construction pipeline emphasizes decoupling numbers from mathematical problems to synthesize number-independent programs, enabling efficient and flexible scaling while minimizing dependency on specific numerical values. Fine-tuning experiments with open-source language and code models, such as Llama2 and CodeLlama, demonstrate the practical benefits of InfinityMATH. These fine-tuned models, showed significant relative improvements on both in-domain and out-of-domain benchmarks, ranging from 184.7% to 514.3% on average. Additionally, these models exhibited high robustness on the GSM8K+ and MATH+ benchmarks, which are enhanced version of test sets with simply the number variations. InfinityMATH ensures that models are more versatile and effective across a broader range of mathematical problems. The data is available at https://huggingface.co/datasets/flagopen/InfinityMATH.
     </details>

89. **Evaluating Language Model Math Reasoning via Grounding in Educational Curricula** [[pdf]](https://arxiv.org/abs/2408.04226v2) `2024-08-08` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Our work presents a novel angle for evaluating language models' (LMs) mathematical abilities, by investigating whether they can discern skills and concepts enabled by math content. We contribute two datasets: one consisting of 385 fine-grained descriptions of K-12 math skills and concepts, or standards, from Achieve the Core (ATC), and another of 9.9K problems labeled with these standards (MathFish). Working with experienced teachers, we find that LMs struggle to tag and verify standards linked to problems, and instead predict labels that are close to ground truth, but differ in subtle ways. We also show that LMs often generate problems that do not fully align with standards described in prompts. Finally, we categorize problems in GSM8k using math standards, allowing us to better understand why some problems are more difficult to solve for models than others.
     </details>

90. **miniCTX: Neural Theorem Proving with (Long-)Contexts** [[pdf]](https://arxiv.org/abs/2408.03350v1) `2024-08-05` `Lean` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces miniCTX, which tests a model's ability to prove formal mathematical theorems that depend on new context that is not seen during training, and offers ntp-toolkit for automatically extracting and annotating theorem proving data.
     </details>


     <details>
          <summary>Abstract</summary>
          We introduce miniCTX, which tests a model's ability to prove formal mathematical theorems that depend on new definitions, lemmas, or other contextual information that was not observed during training. miniCTX contains theorems sourced from real Lean projects and textbooks, each associated with a context that can span tens of thousands of tokens. Models are tasked with proving a theorem given access to code from the theorem's repository, which contains context that is helpful or needed for the proof. As a baseline for miniCTX, we introduce file-tuning, a simple recipe that trains a model to generate a proof step conditioned on the preceding file contents. File-tuning substantially outperforms the traditional neural theorem proving approach that fine-tunes on states alone. Additionally, our file-tuned model improves performance on the standard miniF2F benchmark, achieving a pass rate of 33.61%, which is a new state-of-the-art for 1.3B parameter models. Alongside miniCTX, we offer ntp-toolkit for automatically extracting and annotating theorem proving data, making it easy to add new projects into miniCTX to ensure that contexts are not seen during training. miniCTX offers a challenging and realistic perspective on evaluating neural theorem provers.
     </details>

91. **MathLearner: A Large Language Model Agent Framework for Learning to Solve Mathematical Problems** [[pdf]](http://arxiv.org/abs/2408.01779) `2024-08-03` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work has proposed an agent framework for learning to solve mathematical problems based on inductive reasoning that improves global accuracy over the baseline method (chain-of-thought) and solves 17.54% of the mathematical problems that the baseline cannot solve.
     </details>


     <details>
          <summary>Abstract</summary>
          With the development of artificial intelligence (AI), large language models (LLM) are widely used in many fields. However, the reasoning ability of LLM is still very limited when it comes to mathematical reasoning. Mathematics plays an important role in all aspects of human society and is a technical guarantee in the fields of healthcare, transport and aerospace, for this reason, the development of AI big language models in the field of mathematics has great potential significance. To improve the mathematical reasoning ability of large language models, we proposed an agent framework for learning to solve mathematical problems based on inductive reasoning. By emulating the human learning process of generalization of learned information and effective application of previous knowledge in new reasoning tasks, this framework has great performance in the mathematical reasoning process. It improves global accuracy over the baseline method (chain-of-thought) by 20.96% and solves 17.54% of the mathematical problems that the baseline cannot solve. Benefiting from the efficient RETRIEVAL method, our model improves the ability of large language models to efficiently use external knowledge, i.e., the mathematical computation of the model can be based on written procedures. In education, our model can be used as a personalised learning aid, thus reducing the inequality of educational resources.
     </details>

92. **GeoEval: Benchmark for Evaluating LLMs and Multi-Modal Models on Geometry Problem-Solving** [[pdf]](https://aclanthology.org/2024.findings-acl.73) `ACL 2024 Findings` (8 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The GeoEval benchmark is introduced, a comprehensive collection that includes a main subset of 2,000 problems, a 750 problems subset focusing on backward reasoning, an augmented subset of 2,000 problems, and a hard subset of 300 problems that facilitates a deeper investigation into the performance of LLMs and MMs in solving geometry math problems.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent advancements in large language models (LLMs) and multi-modal models (MMs) have demonstrated their remarkable capabilities in problem-solving. Yet, their proficiency in tackling geometry math problems, which necessitates an integrated understanding of both textual and visual information, has not been thoroughly evaluated. To address this gap, we introduce the GeoEval benchmark, a comprehensive collection that includes a main subset of 2,000 problems, a 750 problems subset focusing on backward reasoning, an augmented sub- set of 2,000 problems, and a hard subset of 300 problems. This benchmark facilitates a deeper investigation into the performance of LLMs and MMs in solving geometry math problems. Our evaluation of ten LLMs and MMs across these varied subsets reveals that the WizardMath model excels, achieving a 55.67% accuracy rate on the main subset but only a 6.00% accuracy on the hard subset. This highlights the critical need for testing models against datasets on which they have not been pre-trained. Additionally, our findings indicate that GPT-series models perform more effectively on problems they have rephrased, suggesting a promising method for enhancing model capabilities.
     </details>

93. **Small Language Models Need Strong Verifiers to Self-Correct Reasoning** [[pdf]](https://aclanthology.org/2024.findings-acl.924) `ACL 2024 Findings` (7 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work explores whether small (<= 13B) language models (LMs) have the ability of self-correction on reasoning tasks with minimal inputs from stronger LMs and proposes a novel pipeline that prompts smaller LMs to collect self-correction data that supports the training of self-refinement abilities.
     </details>


     <details>
          <summary>Abstract</summary>
          Self-correction has emerged as a promising solution to boost the reasoning performance of large language models (LLMs), where LLMs refine their solutions using self-generated critiques that pinpoint the errors. This work explores whether small (≤ 13B) language models (LMs) have the ability of self-correction on reasoning tasks with minimal inputs from stronger LMs. We propose a novel pipeline that prompts smaller LMs to collect self-correction data that supports the training of self-refinement abilities. First, we leverage correct solutions to guide the model in critiquing their incorrect responses. Second, the generated critiques, after filtering, are used for supervised fine-tuning of the self-correcting reasoner through solution refinement. Our experimental results show improved self-correction abilities of two models on five datasets spanning math and commonsense reasoning, with notable performance gains when paired with a strong GPT-4-based verifier, though limitations are identified when using a weak self-verifier for determining when to correct.
     </details>

94. **SceMQA: A Scientific College Entrance Level Multimodal Question Answering Benchmark** [[pdf]](https://aclanthology.org/2024.acl-short.11) `ACL 2024 Short Papers` (6 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          The paper introduces SceMQA, a novel benchmark for scientific multimodal question answering at the college entrance level. It addresses a critical educational phase often overlooked in existing benchmarks, spanning high school to pre-college levels. SceMQA focuses on core science subjects including Mathematics, Physics, Chemistry, and Biology. It features a blend of multiple-choice and free-response formats, ensuring a comprehensive evaluation of AI models’ abilities. Additionally, our benchmark provides specific knowledge points for each problem and detailed explanations for each answer. SceMQA also uniquely presents problems with identical contexts but varied questions to facilitate a more thorough and accurate assessment of reasoning capabilities. In the experiment, we evaluate both open-source and close-source state-of-the-art Multimodal Large Language Models (MLLMs), across various experimental settings. The results show that further research and development are needed in developing more capable MLLM, as highlighted by only 50% to 60% accuracy achieved by the strongest models.
     </details>

95. **LANS: A Layout-Aware Neural Solver for Plane Geometry Problem** [[pdf]](https://aclanthology.org/2024.findings-acl.153) `ACL 2024 Findings` (5 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A layout-aware neural solver named LANS is proposed, integrated with two new modules: multimodal layout-aware pre-trained language module (MLA-PLM) and layout-aware fusion attention (LA-FA), which employs a layout-aware attention mask to realize point-guided cross-modal fusion for further boosting layout awareness of LANS.
     </details>


     <details>
          <summary>Abstract</summary>
          Geometry problem solving (GPS) is a challenging mathematical reasoning task requiring multi-modal understanding, fusion, and reasoning. Existing neural solvers take GPS as a vision-language task but are short in the representation of geometry diagrams that carry rich and complex layout information. In this paper, we propose a layout-aware neural solver named LANS, integrated with two new modules: multimodal layout-aware pre-trained language module (MLA-PLM) and layout-aware fusion attention (LA-FA). MLA-PLM adopts structural-semantic pre-training (SSP) to implement global relationship modeling, and point-match pre-training (PMP) to achieve alignment between visual points and textual points. LA-FA employs a layout-aware attention mask to realize point-guided cross-modal fusion for further boosting layout awareness of LANS. Extensive experiments on datasets Geometry3K and PGPS9K validate the effectiveness of the layout-aware modules and superior problem-solving performance of our LANS solver, over existing symbolic and neural solvers. We have made our code and data publicly available.
     </details>

96. **Distilling Mathematical Reasoning Capabilities into Small Language Models** [[pdf]](http://arxiv.org/abs/2401.11864) `2024-08-01` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Equation-of-Thought Distillation (EoTD) is introduced, a novel technique that encapsulates the reasoning process into equation-based representations to construct an EoTD dataset for fine-tuning SLMs and the Ensemble Thoughts Distillation framework is proposed to enhance the reasoning performance of SLMs.
     </details>


     <details>
          <summary>Abstract</summary>
          This work addresses the challenge of democratizing advanced Large Language Models (LLMs) by compressing their mathematical reasoning capabilities into sub-billion parameter Small Language Models (SLMs) without compromising performance. We introduce Equation-of-Thought Distillation (EoTD), a novel technique that encapsulates the reasoning process into equation-based representations to construct an EoTD dataset for fine-tuning SLMs. Additionally, we propose the Ensemble Thoughts Distillation (ETD) framework to enhance the reasoning performance of SLMs. This involves creating a reasoning dataset with multiple thought processes, including Chain-of-Thought (CoT), Program-of-Thought (PoT), and Equation-of-Thought (EoT), and using it for fine-tuning. Our experimental performance demonstrates that EoTD significantly boosts the reasoning abilities of SLMs, while ETD enables these models to achieve state-of-the-art reasoning performance.
     </details>

97. **Meta-Reasoning: Semantics-Symbol Deconstruction for Large Language Models** [[pdf]](https://aclanthology.org/2024.findings-acl.34) `ACL 2024 Findings` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Experimental results demonstrate that Meta-Reasoning significantly enhances in-context reasoning accuracy, learning efficiency, out-of-domain generalization, and output stability compared to the Chain-of-Thought technique.
     </details>


     <details>
          <summary>Abstract</summary>
          Neural-symbolic methods have demonstrated efficiency in enhancing the reasoning abilities of large language models (LLMs). However, existing methods mainly rely on syntactically mapping natural languages to complete formal languages like Python and SQL. Those methods require that reasoning tasks be convertible into programs, which cater to the computer execution mindset and deviate from human reasoning habits. To broaden symbolic methods’ applicability and adaptability in the real world, we propose Meta-Reasoning from a linguistic perspective. This method empowers LLMs to deconstruct reasoning-independent semantic information into generic symbolic representations, thereby efficiently capturing more generalized reasoning knowledge. We conduct extensive experiments on more than ten datasets encompassing conventional reasoning tasks like arithmetic, symbolic, and logical reasoning, and the more complex interactive reasoning tasks like theory-of-mind reasoning. Experimental results demonstrate that Meta-Reasoning significantly enhances in-context reasoning accuracy, learning efficiency, out-of-domain generalization, and output stability compared to the Chain-of-Thought technique.
     </details>

98. **Prover-Verifier Games improve legibility of LLM outputs** [[pdf]](http://arxiv.org/abs/2407.13692) `2024-08-01` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A training algorithm inspired by Prover-Verifier Game is proposed, which finds that the helpful prover's accuracy and the verifier's robustness to adversarial attacks increase over the course of training, and that legibility training transfers to time-constrained humans tasked with verifying solution correctness.
     </details>


     <details>
          <summary>Abstract</summary>
          One way to increase confidence in the outputs of Large Language Models (LLMs) is to support them with reasoning that is clear and easy to check -- a property we call legibility. We study legibility in the context of solving grade-school math problems and show that optimizing chain-of-thought solutions only for answer correctness can make them less legible. To mitigate the loss in legibility, we propose a training algorithm inspired by Prover-Verifier Game from Anil et al. (2021). Our algorithm iteratively trains small verifiers to predict solution correctness, "helpful" provers to produce correct solutions that the verifier accepts, and "sneaky" provers to produce incorrect solutions that fool the verifier. We find that the helpful prover's accuracy and the verifier's robustness to adversarial attacks increase over the course of training. Furthermore, we show that legibility training transfers to time-constrained humans tasked with verifying solution correctness. Over course of LLM training human accuracy increases when checking the helpful prover's solutions, and decreases when checking the sneaky prover's solutions. Hence, training for checkability by small verifiers is a plausible technique for increasing output legibility. Our results suggest legibility training against small verifiers as a practical avenue for increasing legibility of large LLMs to humans, and thus could help with alignment of superhuman models.
     </details>

99. **Evaluating LLMs’ Mathematical Reasoning in Financial Document Question Answering** [[pdf]](https://aclanthology.org/2024.findings-acl.231) `ACL 2024 Findings` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel prompting technique is introduced tailored to semi-structured documents, matching or outperforming other baselines in performance while providing a nuanced understanding of LLMs abilities for such a task.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs), excel in natural language understanding, but their capability for complex mathematical reasoning with a hybrid of structured tables and unstructured text remain uncertain. This study explores LLMs’ mathematical reasoning on four financial tabular question-answering datasets: TATQA, FinQA, ConvFinQA, and Multihiertt. Through extensive experiments with various models and prompting techniques, we assess how LLMs adapt to complex tables and mathematical tasks. We focus on sensitivity to table complexity and performance variations with an increasing number of arithmetic reasoning steps. The results provide insights into LLMs’ capabilities and limitations in handling complex mathematical scenarios for semi-structured tables. Ultimately, we introduce a novel prompting technique EEDP tailored to semi-structured documents, matching or outperforming baselines performance while providing a nuanced understanding of LLMs abilities.
     </details>

100. **BBA: Bi-Modal Behavioral Alignment for Reasoning with Large Vision-Language Models** [[pdf]](https://aclanthology.org/2024.findings-acl.433) `ACL 2024 Findings` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The BBA prompting method is introduced, designed to maximize the potential of DSL in augmenting complex multi-modal reasoning tasks, and substantially improves the performance of GPT-4V(ision) on geometry problem solving, chess positional advantage prediction and molecular property prediction.
     </details>


     <details>
          <summary>Abstract</summary>
          Multimodal reasoning stands as a pivotal capability for large vision-language models (LVLMs). The integration with Domain-Specific Languages (DSL), offering precise visual representations, equips these models with the opportunity to execute more accurate reasoning in complex and professional domains. However, the vanilla Chain-of-Thought (CoT) prompting method faces challenges in effectively leveraging the unique strengths of visual and DSL representations, primarily due to their differing reasoning mechanisms. Additionally, it often falls short in addressing critical steps in multi-step reasoning tasks. To mitigate these challenges, we introduce the Bi-Modal Behavioral Alignment (BBA) prompting method, designed to maximize the potential of DSL in augmenting complex multi-modal reasoning tasks. This method initiates by guiding LVLMs to create separate reasoning chains for visual and DSL representations. Subsequently, it aligns these chains by addressing any inconsistencies, thus achieving a cohesive integration of behaviors from different modalities. Our experiments demonstrate that BBA substantially improves the performance of GPT-4V(ision) on geometry problem solving (28.34% → 34.22%), chess positional advantage prediction (42.08% → 46.99%) and molecular property prediction (77.47% → 83.52%).
     </details>

101. **Bi-Chainer: Automated Large Language Models Reasoning with Bidirectional Chaining** [[pdf]](https://aclanthology.org/2024.findings-acl.507) `ACL 2024 Findings` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A bidirectional chaining method, Bi-Chainer, which dynamically switches to depth-first reasoning in the opposite reasoning direction when it encounters multiple branching options within the current direction, resulting in more efficient and accurate reasoning.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have shown human-like reasoning abilities but still face challenges in solving complex logical problems. Existing unidirectional chaining methods, such as forward chaining and backward chaining, suffer from issues like low prediction accuracy and efficiency. To address these, we propose a bidirectional chaining method, Bi-Chainer, which dynamically switches to depth-first reasoning in the opposite reasoning direction when it encounters multiple branching options within the current direction. Thus, the intermediate reasoning results can be utilized as guidance to facilitate the reasoning process. We show that Bi-Chainer achieves sizable accuracy boots over unidirectional chaining frameworks on four challenging logical reasoning datasets. Moreover, Bi-Chainer enhances the accuracy of intermediate proof steps and reduces the average number of inference calls, resulting in more efficient and accurate reasoning.
     </details>

102. **Language Models Do Hard Arithmetic Tasks Easily and Hardly Do Easy Arithmetic Tasks** [[pdf]](https://aclanthology.org/2024.acl-short.8) `ACL 2024 Short Papers` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is shown that LLMs are frequently able to correctly and confidently predict the first digit of n-digit by m-digit multiplication tasks without using chain of thought reasoning, despite these tasks require compounding operations to solve.
     </details>


     <details>
          <summary>Abstract</summary>
          The ability (and inability) of large language models (LLMs) to perform arithmetic tasks has been the subject of much theoretical and practical debate. We show that LLMs are frequently able to correctly and confidently predict the first digit of n-digit by m-digit multiplication tasks without using chain of thought reasoning, despite these tasks require compounding operations to solve. Simultaneously, LLMs in practice often fail to correctly or confidently predict the last digit of an n-digit by m-digit multiplication, a task equivalent to 1-digit by 1-digit multiplication which can be easily learned or memorized. We show that the latter task can be solved more robustly when the LLM is conditioned on all of the correct higher-order digits, which on average increases the confidence of the correct last digit on 5-digit by 5-digit multiplication tasks using Llama 2-13B by over 230% (0.13→0.43) and Mistral-7B by 150% (0.22→0.55).
     </details>

103. **NUMCoT: Numerals and Units of Measurement in Chain-of-Thought Reasoning using Large Language Models** [[pdf]](https://aclanthology.org/2024.findings-acl.848) `ACL 2024 Findings` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper anatomizes the reasoning of math word problems to different sub-procedures like numeral conversions from language to numbers and measurement conversions based on units and demonstrates that LLMs still encounter difficulties in handling numeral and measurement conversions.
     </details>


     <details>
          <summary>Abstract</summary>
          Numeral systems and units of measurement are two conjoined topics in activities of human beings and have mutual effects with the languages expressing them. Currently, the evaluation of Large Language Models (LLMs) often involves mathematical reasoning, yet little attention is given to how minor changes in numbers or units can drastically alter the complexity of problems and the performance of LLMs. In this paper, we scrutinize existing LLMs on processing of numerals and units of measurement by constructing datasets with perturbations. We first anatomize the reasoning of math word problems to different sub-procedures like numeral conversions from language to numbers and measurement conversions based on units. Then we further annotate math word problems from ancient Chinese arithmetic works which are challenging in numerals and units of measurement. Experiments on perturbed datasets demonstrate that LLMs still encounter difficulties in handling numeral and measurement conversions.
     </details>

104. **Question-Analysis Prompting Improves LLM Performance in Reasoning Tasks** [[pdf]](https://aclanthology.org/2024.acl-srw.45) `ACL 2024 Student Research Workshop` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This research proposes a novel prompting strategy called Question Analysis Prompting (QAP), in which the model is prompted to explain the question in words before solving, where detailed responses are beneficial when answering harder questions, but can negatively affect easy questions.
     </details>


     <details>
          <summary>Abstract</summary>
          Although LLMs have the potential to transform many fields, they still underperform humans in reasoning tasks. Existing methods induce the model to produce step-by-step calculations, but this research explores the question: Does making the LLM analyze the question improve its performance? We propose a novel prompting strategy called Question Analysis Prompting (QAP), in which the model is prompted to explain the question in ’n’ words before solving. The value of ’n’ influences the length of response generated by the model. QAP is evaluated on GPT-3.5 Turbo and GPT-4 Turbo on arithmetic datasets GSM8K, AQuA, and SAT and commonsense dataset StrategyQA. QAP is compared with other state-of-the-art prompts including chain-of-thought (CoT), Plan and Solve Prompting (PS+) and Take A Deep Breath (TADB). QAP outperforms all state-of-the-art prompts on AQuA and SAT datasets on both GPT-3.5 and GPT-4. QAP consistently ranks among the top-2 prompts on 75% of the tests. A key factor of QAP performance can be attributed to response length, where detailed responses are beneficial when answering harder questions, but can negatively affect easy questions.
     </details>

105. **Large Language Monkeys: Scaling Inference Compute with Repeated Sampling** [[pdf]](http://arxiv.org/abs/2407.21787) `2024-07-31` (21 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Inference compute is explored as another axis for scaling by increasing the number of generated samples across multiple tasks and models, observing that coverage - the fraction of problems solved by any attempt - scales with the number of samples over four orders of magnitude.
     </details>


     <details>
          <summary>Abstract</summary>
          Scaling the amount of compute used to train language models has dramatically improved their capabilities. However, when it comes to inference, we often limit the amount of compute to only one attempt per problem. Here, we explore inference compute as another axis for scaling by increasing the number of generated samples. Across multiple tasks and models, we observe that coverage - the fraction of problems solved by any attempt - scales with the number of samples over four orders of magnitude. In domains like coding and formal proofs, where all answers can be automatically verified, these increases in coverage directly translate into improved performance. When we apply repeated sampling to SWE-bench Lite, the fraction of issues solved with DeepSeek-V2-Coder-Instruct increases from 15.9% with one sample to 56% with 250 samples, outperforming the single-attempt state-of-the-art of 43% which uses more capable frontier models. Moreover, using current API pricing, amplifying the cheaper DeepSeek model with five samples is more cost-effective and solves more issues than paying a premium for one sample from GPT-4o or Claude 3.5 Sonnet. Interestingly, the relationship between coverage and the number of samples is often log-linear and can be modelled with an exponentiated power law, suggesting the existence of inference-time scaling laws. Finally, we find that identifying correct samples out of many generations remains an important direction for future research in domains without automatic verifiers. When solving math word problems from GSM8K and MATH, coverage with Llama-3 models grows to over 95% with 10,000 samples. However, common methods to pick correct solutions from a sample collection, such as majority voting or reward models, plateau beyond several hundred samples and fail to fully scale with the sample budget.
     </details>

106. **Inductive or Deductive? Rethinking the Fundamental Reasoning Abilities of LLMs** [[pdf]](https://arxiv.org/abs/2408.00114v2) `2024-07-31` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a novel framework, SolverLearner, that enables LLMs to learn the underlying function of inductive reasoning, and reveals that LLMs demonstrate remarkable inductive reasoning capabilities through SolverLearner, achieving near-perfect performance with ACC of 1 in most cases.
     </details>


     <details>
          <summary>Abstract</summary>
          Reasoning encompasses two typical types: deductive reasoning and inductive reasoning. Despite extensive research into the reasoning capabilities of Large Language Models (LLMs), most studies have failed to rigorously differentiate between inductive and deductive reasoning, leading to a blending of the two. This raises an essential question: In LLM reasoning, which poses a greater challenge - deductive or inductive reasoning? While the deductive reasoning capabilities of LLMs, (i.e. their capacity to follow instructions in reasoning tasks), have received considerable attention, their abilities in true inductive reasoning remain largely unexplored. To investigate into the true inductive reasoning capabilities of LLMs, we propose a novel framework, SolverLearner. This framework enables LLMs to learn the underlying function (i.e., $y = f_w(x)$), that maps input data points $(x)$ to their corresponding output values $(y)$, using only in-context examples. By focusing on inductive reasoning and separating it from LLM-based deductive reasoning, we can isolate and investigate inductive reasoning of LLMs in its pure form via SolverLearner. Our observations reveal that LLMs demonstrate remarkable inductive reasoning capabilities through SolverLearner, achieving near-perfect performance with ACC of 1 in most cases. Surprisingly, despite their strong inductive reasoning abilities, LLMs tend to relatively lack deductive reasoning capabilities, particularly in tasks involving ``counterfactual'' reasoning.
     </details>

107. **AI-Assisted Generation of Difficult Math Questions** [[pdf]](http://arxiv.org/abs/2407.21009) `2024-07-30` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A design framework that combines the strengths of LLMs with a human-in-the-loop approach to generate a diverse array of challenging math questions and a striking relationship observed between models' performance on the new dataset, suggesting that successfully solving the question in MATH$^2$ requires a nontrivial combination of two distinct math skills.
     </details>


     <details>
          <summary>Abstract</summary>
          Current LLM training positions mathematical reasoning as a core capability. With publicly available sources fully tapped, there is unmet demand for diverse and challenging math questions. Relying solely on human experts is both time-consuming and costly, while LLM-generated questions often lack the requisite diversity and difficulty. We present a design framework that combines the strengths of LLMs with a human-in-the-loop approach to generate a diverse array of challenging math questions. We leverage LLM metacognition skills [Didolkar et al., 2024] of a strong LLM to extract core "skills" from existing math datasets. These skills serve as the basis for generating novel and difficult questions by prompting the LLM with random pairs of core skills. The use of two different skills within each question makes finding such questions an "out of distribution" task for both LLMs and humans. Our pipeline employs LLMs to iteratively generate and refine questions and solutions through multiturn prompting. Human annotators then verify and further refine the questions, with their efficiency enhanced via further LLM interactions. Applying this pipeline on skills extracted from the MATH dataset [Hendrycks et al., 2021] resulted in MATH$^2$ - a dataset of higher-quality math questions, as evidenced by: (a) Lower performance of all models on MATH$^2$ than on MATH (b) Higher performance on MATH when using MATH$^2$ questions as in-context examples. Although focused on mathematics, our methodology seems applicable to other domains requiring structured reasoning, and potentially as a component of scalable oversight. Also of interest is a striking relationship observed between models' performance on the new dataset: the success rate on MATH$^2$ is the square on MATH, suggesting that successfully solving the question in MATH$^2$ requires a nontrivial combination of two distinct math skills.
     </details>

108. **Key-Point-Driven Mathematical Reasoning Distillation of Large Language Model** [[pdf]](http://arxiv.org/abs/2407.10167) `2024-07-30` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          KPDD enhances the reasoning performance of SLMs by breaking down the problem-solving process into three stages: Core Question Extraction, Problem-Solving Information Extraction, and Step-by-Step Solution, and KPDD-PoT achieves state-of-the-art performance in mathematical reasoning tasks.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have demonstrated exceptional proficiency in mathematical reasoning tasks due to their extensive parameter counts and training on vast datasets. Despite these capabilities, deploying LLMs is hindered by their computational demands. Distilling LLM mathematical reasoning into Smaller Language Models (SLMs) has emerged as a solution to this challenge, although these smaller models often suffer from errors in calculation and semantic understanding. Prior work has proposed Program-of-Thought Distillation (PoTD) to avoid calculation error. To further address semantic understanding errors, we propose Key-Point-Driven Mathematical Reasoning Distillation (KPDD). KPDD enhances the reasoning performance of SLMs by breaking down the problem-solving process into three stages: Core Question Extraction, Problem-Solving Information Extraction, and Step-by-Step Solution. This method is further divided into KPDD-CoT, which generates Chain-of-Thought rationales, and KPDD-PoT, which creates Program-of-Thought rationales. The experiment results show that KPDD-CoT significantly improves reasoning abilities, while KPDD-PoT achieves state-of-the-art performance in mathematical reasoning tasks. Our approach effectively mitigates misunderstanding errors, advancing the deployment of efficient and capable SLMs.
     </details>

109. **Physics of Language Models: Part 2.1, Grade-School Math and the Hidden Reasoning Process** [[pdf]](http://arxiv.org/abs/2407.20311) `2024-07-29` (8 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Recent advances in language models have demonstrated their capability to solve mathematical reasoning problems, achieving near-perfect accuracy on grade-school level math benchmarks like GSM8K. In this paper, we formally study how language models solve these problems. We design a series of controlled experiments to address several fundamental questions: (1) Can language models truly develop reasoning skills, or do they simply memorize templates? (2) What is the model's hidden (mental) reasoning process? (3) Do models solve math questions using skills similar to or different from humans? (4) Do models trained on GSM8K-like datasets develop reasoning skills beyond those necessary for solving GSM8K problems? (5) What mental process causes models to make reasoning mistakes? (6) How large or deep must a model be to effectively solve GSM8K-level math questions?   Our study uncovers many hidden mechanisms by which language models solve mathematical questions, providing insights that extend beyond current understandings of LLMs.
     </details>

110. **PATCH! Psychometrics-AssisTed benCHmarking of Large Language Models: A Case Study of Proficiency in 8th Grade Mathematics** [[pdf]](http://arxiv.org/abs/2404.01799) `2024-07-25` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Many existing benchmarks of large (multimodal) language models (LLMs) focus on measuring LLMs' academic proficiency, often with also an interest in comparing model performance with human test takers. While these benchmarks have proven key to the development of LLMs, they suffer from several limitations, including questionable measurement quality (e.g., Do they measure what they are supposed to in a reliable way?), lack of quality assessment on the item level (e.g., Are some items more important or difficult than others?) and unclear human population reference (e.g., To whom can the model be compared?). In response to these challenges, we propose leveraging knowledge from psychometrics - a field dedicated to the measurement of latent variables like academic proficiency - into LLM benchmarking. We make three primary contributions. First, we introduce PATCH: a novel framework for {P}sychometrics-{A}ssis{T}ed ben{CH}marking of LLMs. PATCH addresses the aforementioned limitations, presenting a new direction for LLM benchmark research. Second, we implement PATCH by measuring GPT-4 and Gemini-Pro-Vision's proficiency in 8th grade mathematics against 56 human populations. We show that adopting a psychometrics-based approach yields evaluation outcomes that diverge from those based on existing benchmarking practices. Third, we release 4 high-quality datasets to support measuring and comparing LLM proficiency in grade school mathematics and science against human populations.
     </details>

111. **Relating the Seemingly Unrelated: Principled Understanding of Generalization for Generative Models in Arithmetic Reasoning Tasks** [[pdf]](https://arxiv.org/abs/2407.17963v1) `2024-07-25` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Extensive experiments with GPT-like models validate the theoretical predictions and deepen the understanding of the generalization mechanisms, and facilitate more data-efficient model training and objective-oriented AI alignment.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have demonstrated impressive versatility across numerous tasks, yet their generalization capabilities remain poorly understood. To investigate these behaviors, arithmetic tasks serve as important venues. In previous studies, seemingly unrelated mysteries still exist -- (1) models with appropriate positional embeddings can correctly perform longer unseen arithmetic operations such as addition, but their effectiveness varies in more complex tasks like multiplication; (2) models perform well for longer unseen cases in modular addition under specific moduli (e.g., modulo 100) but struggle under very close moduli (e.g., modulo 101), regardless of the positional encoding used. We believe previous studies have been treating the symptoms rather than addressing the root cause -- they have paid excessive attention to improving model components, while overlooking the differences in task properties that may be the real drivers. This is confirmed by our unified theoretical framework for different arithmetic scenarios. For example, unlike multiplication, the digital addition task has the property of translation invariance which naturally aligns with the relative positional encoding, and this combination leads to successful generalization of addition to unseen longer domains. The discrepancy in operations modulo 100 and 101 arises from the base. Modulo 100, unlike 101, is compatible with the decimal system (base 10), such that unseen information in digits beyond the units digit and the tens digit is actually not needed for the task. Extensive experiments with GPT-like models validate our theoretical predictions. These findings deepen our understanding of the generalization mechanisms, and facilitate more data-efficient model training and objective-oriented AI alignment.
     </details>

112. **LEAN-GitHub: Compiling GitHub LEAN repositories for a versatile LEAN prover** [[pdf]](http://arxiv.org/abs/2407.17227) `2024-07-24` `Lean` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes LEAN-GitHub, a dataset consisting of large-scale formal data extracted from almost all Lean 4 repositories on GitHub, and achieves state-of-the-art on two other Lean 4 benchmarks targeting different fields/levels of math.
     </details>


     <details>
          <summary>Abstract</summary>
          Recently, large language models have presented promising results in aiding formal mathematical reasoning. However, their performance is restricted due to the scarcity of formal theorem-proving data, which requires additional effort to be extracted from raw formal language corpora. Meanwhile, a significant amount of human-written formal language corpora remains underutilized. To address this issue, we propose LEAN-GitHub, a dataset consisting of large-scale formal data extracted from almost all Lean 4 repositories on GitHub. After fine-tuning InternLM-math-plus on this dataset, our model achieved accuracies of 48.8% with a single pass and 54.5% with 64 passes on the Lean 4 miniF2F test, surpassing state-of-the-art method at 52%. And it also achieves state-of-the-art on two other Lean 4 benchmarks (ProofNet and Putnam) targeting different fields/levels of math. These results demonstrate that our proposed dataset is beneficial for formal reasoning on a wide range of math topics. We open-source our model at https://GitHub. com/InternLM/InternLM-Math and our data at https://huggingface.co/ datasets/InternLM/Lean-GitHub
     </details>

113. **Boosting Large Language Models with Socratic Method for Conversational Mathematics Teaching** [[pdf]](https://arxiv.org/abs/2407.17349v1) `2024-07-24` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper proposes a knowledge-enhanced LLM as a strong baseline to generate reliable responses with review, guidance/heuristic, rectification, and summarization, and shows the great advantages of \texttt{SocraticLLM} by comparing it with several strong generative models.
     </details>


     <details>
          <summary>Abstract</summary>
          With the introduction of large language models (LLMs), automatic math reasoning has seen tremendous success. However, current methods primarily focus on providing solutions or using techniques like Chain-of-Thought to enhance problem-solving accuracy. In this paper, we focus on improving the capability of mathematics teaching via a Socratic teaching-based LLM (\texttt{SocraticLLM}), which guides learners toward profound thinking with clarity and self-discovery via conversation. We collect and release a high-quality mathematical teaching dataset, named \texttt{SocraticMATH}, which provides Socratic-style conversations of problems with extra knowledge. Also, we propose a knowledge-enhanced LLM as a strong baseline to generate reliable responses with review, guidance/heuristic, rectification, and summarization. Experimental results show the great advantages of \texttt{SocraticLLM} by comparing it with several strong generative models. The codes and datasets are available on \url{https://github.com/ECNU-ICALK/SocraticMath}.
     </details>

114. **MathViz-E: A Case-study in Domain-Specialized Tool-Using Agents** [[pdf]](http://arxiv.org/abs/2407.17544) `2024-07-24` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          An automated math visualizer and solver system for mathematical pedagogy that orchestrates mathematical solvers and math graphing tools to produce accurate visualizations from simple natural language commands is presented.
     </details>


     <details>
          <summary>Abstract</summary>
          There has been significant recent interest in harnessing LLMs to control software systems through multi-step reasoning, planning and tool-usage. While some promising results have been obtained, application to specific domains raises several general issues including the control of specialized domain tools, the lack of existing datasets for training and evaluation, and the non-triviality of automated system evaluation and improvement. In this paper, we present a case-study where we examine these issues in the context of a specific domain. Specifically, we present an automated math visualizer and solver system for mathematical pedagogy. The system orchestrates mathematical solvers and math graphing tools to produce accurate visualizations from simple natural language commands. We describe the creation of specialized data-sets, and also develop an auto-evaluator to easily evaluate the outputs of our system by comparing them to ground-truth expressions. We have open sourced the data-sets and code for the proposed system.
     </details>

115. **Q*: Improving Multi-step Reasoning for LLMs with Deliberative Planning** [[pdf]](http://arxiv.org/abs/2406.14283) `2024-07-22` (4 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          By casting multi-step reasoning of LLMs as a heuristic search problem, Q*, a general, versatile and agile framework for guiding LLMs decoding process with deliberative planning is introduced, which can effectively guide LLMs to select the most promising next reasoning step without fine-tuning LLMs for the current task.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have demonstrated impressive capability in many natural language tasks. However, the auto-regressive generation process makes LLMs prone to produce errors, hallucinations and inconsistent statements when performing multi-step reasoning. In this paper, by casting multi-step reasoning of LLMs as a heuristic search problem, we aim to alleviate the pathology by introducing Q*, a general, versatile and agile framework for guiding LLMs decoding process with deliberative planning. By learning a plug-and-play Q-value model as heuristic function for estimating expected future rewards, our Q* can effectively guide LLMs to select the most promising next reasoning step without fine-tuning LLMs for the current task, which avoids the significant computational overhead and potential risk of performance degeneration on other tasks. Extensive experiments on GSM8K, MATH and MBPP demonstrate the superiority of our method, contributing to improving the reasoning performance of existing open-source LLMs.
     </details>

116. **TaskGen: A Task-Based, Memory-Infused Agentic Framework using StrictJSON** [[pdf]](http://arxiv.org/abs/2407.15734) `2024-07-22` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work empirically evaluates TaskGen on various environments such as 40x40 dynamic maze navigation with changing obstacle locations, TextWorld escape room solving with dense rewards and detailed goals, web browsing, and Retrieval Augmented Generation on NaturalQuestions dataset.
     </details>


     <details>
          <summary>Abstract</summary>
          TaskGen is an open-sourced agentic framework which uses an Agent to solve an arbitrary task by breaking them down into subtasks. Each subtask is mapped to an Equipped Function or another Agent to execute. In order to reduce verbosity (and hence token usage), TaskGen uses StrictJSON that ensures JSON output from the Large Language Model (LLM), along with additional features such as type checking and iterative error correction. Key to the philosophy of TaskGen is the management of information/memory on a need-to-know basis. We empirically evaluate TaskGen on various environments such as 40x40 dynamic maze navigation with changing obstacle locations (100% solve rate), TextWorld escape room solving with dense rewards and detailed goals (96% solve rate), web browsing (69% of actions successful), solving the MATH dataset (71% solve rate over 100 Level-5 problems), Retrieval Augmented Generation on NaturalQuestions dataset (F1 score of 47.03%)
     </details>

117. **Reliable Reasoning Beyond Natural Language** [[pdf]](http://arxiv.org/abs/2407.11373) `2024-07-19` `Prolog` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a neurosymbolic approach that prompts LLMs to extract and encode all relevant information from a problem statement as logical code statements, and then uses a logic programming language (Prolog) to conduct the iterative computations of explicit deductive reasoning.
     </details>


     <details>
          <summary>Abstract</summary>
          Despite their linguistic competence, Large Language models (LLMs) often exhibit limitations in their ability to reason reliably and flexibly. To address this, we propose a neurosymbolic approach that prompts LLMs to extract and encode all relevant information from a problem statement as logical code statements, and then use a logic programming language (Prolog) to conduct the iterative computations of explicit deductive reasoning. Our approach significantly enhances the performance of LLMs on the standard mathematical reasoning benchmark, GSM8k, and the Navigate dataset from the BIG-bench dataset. Additionally, we introduce a novel dataset, the Non-Linear Reasoning (NLR) dataset, consisting of 55 unique word problems that target the shortcomings of the next token prediction paradigm of LLMs and require complex non-linear reasoning but only basic arithmetic skills to solve. Our findings demonstrate that the integration of Prolog enables LLMs to achieve high performance on the NLR dataset, which even the most advanced language models (including GPT4) fail to solve using text only.
     </details>

118. **Learning From Correctness Without Prompting Makes LLM Efficient Reasoner** [[pdf]](http://arxiv.org/abs/2403.19094) `2024-07-18` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduces an intrinsic self-correct reasoning framework for LLMs that eliminates the need for human feedback, external tools, and handcraft prompts, and improves reasoning performance without needing to learn from errors.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have demonstrated outstanding performance across various tasks, yet they still exhibit limitations such as hallucination, unfaithful reasoning, and toxic content. One potential approach to mitigate these issues is learning from human or external feedback (e.g. tools). In this paper, we introduce an intrinsic self-correct reasoning framework for LLMs that eliminates the need for human feedback, external tools, and handcraft prompts. The proposed framework, based on a multi-step reasoning paradigm \textbf{Le}arning from \textbf{Co}rrectness (\textsc{LeCo}), improves reasoning performance without needing to learn from errors. This paradigm prioritizes learning from correct reasoning steps, and a unique method to measure confidence for each reasoning step based on generation logits. Experimental results across various multi-step reasoning tasks demonstrate the effectiveness of the framework in improving reasoning performance with reduced token consumption.
     </details>

119. **Weak-to-Strong Reasoning** [[pdf]](http://arxiv.org/abs/2407.13647) `2024-07-18` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A progressive learning framework that enables the strong model to autonomously refine its training data, without requiring input from either a more advanced model or human-annotated data is introduced.
     </details>


     <details>
          <summary>Abstract</summary>
          When large language models (LLMs) exceed human-level capabilities, it becomes increasingly challenging to provide full-scale and accurate supervision for these models. Weak-to-strong learning, which leverages a less capable model to unlock the latent abilities of a stronger model, proves valuable in this context. Yet, the efficacy of this approach for complex reasoning tasks is still untested. Furthermore, tackling reasoning tasks under the weak-to-strong setting currently lacks efficient methods to avoid blindly imitating the weak supervisor including its errors. In this paper, we introduce a progressive learning framework that enables the strong model to autonomously refine its training data, without requiring input from either a more advanced model or human-annotated data. This framework begins with supervised fine-tuning on a selective small but high-quality dataset, followed by preference optimization on contrastive samples identified by the strong model itself. Extensive experiments on the GSM8K and MATH datasets demonstrate that our method significantly enhances the reasoning capabilities of Llama2-70b using three separate weak models. This method is further validated in a forward-looking experimental setup, where Llama3-8b-instruct effectively supervises Llama3-70b on the highly challenging OlympicArena dataset. This work paves the way for a more scalable and sophisticated strategy to enhance AI reasoning powers. All relevant code and resources are available in \url{https://github.com/GAIR-NLP/weak-to-strong-reasoning}.
     </details>

120. **Learning Goal-Conditioned Representations for Language Reward Models** [[pdf]](http://arxiv.org/abs/2407.13887) `2024-07-18` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes training reward models in a contrastive, goal-conditioned fashion by increasing the representation similarity of future states along sampled preferred trajectories and decreasing the similarity along randomly sampled dispreferred trajectories, and finds that these representations can perform fine-grained control by conditioning on desired future goal-states.
     </details>


     <details>
          <summary>Abstract</summary>
          Techniques that learn improved representations via offline data or self-supervised objectives have shown impressive results in traditional reinforcement learning (RL). Nevertheless, it is unclear how improved representation learning can benefit reinforcement learning from human feedback (RLHF) on language models (LMs). In this work, we propose training reward models (RMs) in a contrastive, $\textit{goal-conditioned}$ fashion by increasing the representation similarity of future states along sampled preferred trajectories and decreasing the similarity along randomly sampled dispreferred trajectories. This objective significantly improves RM performance by up to 0.09 AUROC across challenging benchmarks, such as MATH and GSM8k. These findings extend to general alignment as well -- on the Helpful-Harmless dataset, we observe $2.3\%$ increase in accuracy. Beyond improving reward model performance, we show this way of training RM representations enables improved $\textit{steerability}$ because it allows us to evaluate the likelihood of an action achieving a particular goal-state (e.g., whether a solution is correct or helpful). Leveraging this insight, we find that we can filter up to $55\%$ of generated tokens during majority voting by discarding trajectories likely to end up in an "incorrect" state, which leads to significant cost savings. We additionally find that these representations can perform fine-grained control by conditioning on desired future goal-states. For example, we show that steering a Llama 3 model towards helpful generations with our approach improves helpfulness by $9.6\%$ over a supervised-fine-tuning trained baseline. Similarly, steering the model towards complex generations improves complexity by $21.6\%$ over the baseline. Overall, we find that training RMs in this contrastive, goal-conditioned fashion significantly improves performance and enables model steerability.
     </details>

121. **Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate** [[pdf]](http://arxiv.org/abs/2305.19118) `2024-07-17` (204 cite) (8 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A Multi-Agent Debate (MAD) framework is proposed, in which multiple agents express their arguments in the state of"tit for tat"and a judge manages the debate process to obtain a final solution.
     </details>


     <details>
          <summary>Abstract</summary>
          Modern large language models (LLMs) like ChatGPT have shown remarkable performance on general language tasks but still struggle on complex reasoning tasks, which drives the research on cognitive behaviors of LLMs to explore human-like problem-solving strategies. Along this direction, one representative strategy is self-reflection, which asks an LLM to refine the solution with the feedback generated by itself iteratively. However, our study shows that such reflection-style methods suffer from the Degeneration-of-Thought (DoT) problem: once the LLM has established confidence in its solutions, it is unable to generate novel thoughts later through reflection even if its initial stance is incorrect. To address the DoT problem, we propose a Multi-Agent Debate (MAD) framework, in which multiple agents express their arguments in the state of "tit for tat" and a judge manages the debate process to obtain a final solution. Clearly, our MAD framework encourages divergent thinking in LLMs which would be helpful for tasks that require deep levels of contemplation. Experiment results on two challenging datasets, commonsense machine translation and counter-intuitive arithmetic reasoning, demonstrate the effectiveness of our MAD framework. Extensive analyses suggest that the adaptive break of debate and the modest level of "tit for tat" state are required for MAD to obtain good performance. Moreover, we find that LLMs might not be a fair judge if different LLMs are used for agents. Code is available at https://github.com/Skytliang/Multi-Agents-Debate.
     </details>

122. **DotaMath: Decomposition of Thought with Code Assistance and Self-correction for Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2407.04078) `2024-07-17` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduces a series of LLMs that employs the Decomposition of thought with code assistance and self-correction for mathematical reasoning, dubbed as DotaMath, which achieve remarkable performance compared to open-source LLMs across various in-domain and out-of-domain benchmarks.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have made impressive progress in handling simple math problems, yet they still struggle with more challenging and complex mathematical tasks. In this paper, we introduce a series of LLMs that employs the Decomposition of thought with code assistance and self-correction for mathematical reasoning, dubbed as DotaMath. DotaMath models tackle complex mathematical tasks by decomposing them into simpler logical subtasks, leveraging code to solve these subtasks, obtaining fine-grained feedback from the code interpreter, and engaging in self-reflection and correction. By annotating diverse interactive tool-use trajectories and employing query evolution on GSM8K and MATH datasets, we generate an instruction fine-tuning dataset called DotaMathQA with 574K query-response pairs. We train a series of base LLMs using imitation learning on DotaMathQA, resulting in DotaMath models that achieve remarkable performance compared to open-source LLMs across various in-domain and out-of-domain benchmarks. Notably, DotaMath-deepseek-7B showcases an outstanding performance of 64.8% on the competitive MATH dataset and 86.7% on GSM8K. Besides, DotaMath-deepseek-7B maintains strong competitiveness on a series of in-domain and out-of-domain benchmarks (Avg. 80.1%). Looking forward, we anticipate that the DotaMath paradigm will open new pathways for addressing intricate mathematical problems. Our code is publicly available at https://github.com/ChengpengLi1003/DotaMath.
     </details>

123. **Skywork-Math: Data Scaling Laws for Mathematical Reasoning in Large Language Models -- The Story Goes On** [[pdf]](http://arxiv.org/abs/2407.08348) `2024-07-17` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          In this paper, we investigate the underlying factors that potentially enhance the mathematical reasoning capabilities of large language models (LLMs). We argue that the data scaling law for math reasoning capabilities in modern LLMs is far from being saturated, highlighting how the model's quality improves with increases in data quantity. To support this claim, we introduce the Skywork-Math model series, supervised fine-tuned (SFT) on common 7B LLMs using our proposed 2.5M-instance Skywork-MathQA dataset. Skywork-Math 7B has achieved impressive accuracies of 51.2% on the competition-level MATH benchmark and 83.9% on the GSM8K benchmark using only SFT data, outperforming an early version of GPT-4 on MATH. The superior performance of Skywork-Math models contributes to our novel two-stage data synthesis and model SFT pipelines, which include three different augmentation methods and a diverse seed problem set, ensuring both the quantity and quality of Skywork-MathQA dataset across varying difficulty levels. Most importantly, we provide several practical takeaways to enhance math reasoning abilities in LLMs for both research and industry applications.
     </details>

124. **Steamroller Problems: An Evaluation of LLM Reasoning Capability with Automated Theorem Prover Strategies** [[pdf]](https://arxiv.org/abs/2407.20244v1) `2024-07-17` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is confirmed that LLMs have a preference for, and are best able to follow, bottom up reasoning processes, and the reasoning strategies can still be beneficial for deriving small and relevant sets of formulas for external processing by a trusted inference engine.
     </details>


     <details>
          <summary>Abstract</summary>
          This study presents the first examination of the ability of Large Language Models (LLMs) to follow reasoning strategies that are used to guide Automated Theorem Provers (ATPs). We evaluate the performance of GPT4, GPT3.5 Turbo and Google's recent Gemini model on problems from a steamroller domain. In addition to determining accuracy we make use of the Natural Language Processing library spaCy to explore new methods of investigating LLM's reasoning capabilities. This led to one alarming result, the low correlation between correct reasoning and correct answers for any of the tested models. We found that the models' performance when using the ATP reasoning strategies was comparable to one-shot chain of thought and observe that attention to uncertainty in the accuracy results is critical when drawing conclusions about model performance. Consistent with previous speculation we confirm that LLMs have a preference for, and are best able to follow, bottom up reasoning processes. However, the reasoning strategies can still be beneficial for deriving small and relevant sets of formulas for external processing by a trusted inference engine.
     </details>

125. **Reasoning with Large Language Models, a Survey** [[pdf]](http://arxiv.org/abs/2407.11511) `2024-07-16` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is found that self-improvement, self-reflection, and some metacognitive abilities of the reasoning processes are possible through the judicious use of prompts, suggesting that true self-improvement and self-reasoning, to go from reasoning with LLMs to reasoning by LLMs, remains future work.
     </details>


     <details>
          <summary>Abstract</summary>
          Scaling up language models to billions of parameters has opened up possibilities for in-context learning, allowing instruction tuning and few-shot learning on tasks that the model was not specifically trained for. This has achieved breakthrough performance on language tasks such as translation, summarization, and question-answering. Furthermore, in addition to these associative "System 1" tasks, recent advances in Chain-of-thought prompt learning have demonstrated strong "System 2" reasoning abilities, answering a question in the field of artificial general intelligence whether LLMs can reason. The field started with the question whether LLMs can solve grade school math word problems. This paper reviews the rapidly expanding field of prompt-based reasoning with LLMs. Our taxonomy identifies different ways to generate, evaluate, and control multi-step reasoning. We provide an in-depth coverage of core approaches and open problems, and we propose a research agenda for the near future. Finally, we highlight the relation between reasoning and prompt-based learning, and we discuss the relation between reasoning, sequential decision processes, and reinforcement learning. We find that self-improvement, self-reflection, and some metacognitive abilities of the reasoning processes are possible through the judicious use of prompts. True self-improvement and self-reasoning, to go from reasoning with LLMs to reasoning by LLMs, remains future work.
     </details>

126. **Diffusion of Thoughts: Chain-of-Thought Reasoning in Diffusion Language Models** [[pdf]](http://arxiv.org/abs/2402.07754) `2024-07-15` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes Diffusion-of-Thought (DoT), a novel approach that integrates diffusion models with Chain-of-Thought, a well-established technique for improving the reasoning ability of autoregressive language models.
     </details>


     <details>
          <summary>Abstract</summary>
          Recently, diffusion models have garnered significant interest in the field of text processing due to their many potential advantages compared to conventional autoregressive models. In this work, we propose Diffusion-of-Thought (DoT), a novel approach that integrates diffusion models with Chain-of-Thought, a well-established technique for improving the reasoning ability of autoregressive language models. In contrast to autoregressive language models that make decisions in a left-to-right, token-by-token manner, DoT allows reasoning steps to diffuse over time through a diffusion language model and offers greater flexibility in trading-off computation for reasoning performance. Our experimental results demonstrate the effectiveness of DoT in multi-digit multiplication, boolean logic, and grade school math problems, with a small diffusion model outperforming a much larger autoregressive model in both efficiency and accuracy. In addition to that, DoT showcases promising self-correction abilities and benefits from existing reasoning-enhancing techniques like self-consistency decoding. Our findings contribute to the understanding and development of reasoning with diffusion language models.
     </details>

127. **COMET: "Cone of experience" enhanced large multimodal model for mathematical problem generation** [[pdf]](http://arxiv.org/abs/2407.11315) `2024-07-15` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A three-stage fine-turning framework guided by the "Cone of Experience" is proposed, which divides the fine-tuning data into symbolic experience, iconic experience, and direct experience to draw parallels with experiences in the career growth of teachers.
     </details>


     <details>
          <summary>Abstract</summary>
          The automatic generation of high-quality mathematical problems is practically valuable in many educational scenarios. Large multimodal model provides a novel technical approach for the mathematical problem generation because of its wide success in cross-modal data scenarios. However, the traditional method of separating problem solving from problem generation and the mainstream fine-tuning framework of monotonous data structure with homogeneous training objectives limit the application of large multimodal model in mathematical problem generation. Addressing these challenges, this paper proposes COMET, a "Cone of Experience" enhanced large multimodal model for mathematical problem generation. Firstly, from the perspective of mutual ability promotion and application logic, we unify stem generation and problem solving into mathematical problem generation. Secondly, a three-stage fine-turning framework guided by the "Cone of Experience" is proposed. The framework divides the fine-tuning data into symbolic experience, iconic experience, and direct experience to draw parallels with experiences in the career growth of teachers. Several fine-grained data construction and injection methods are designed in this framework. Finally, we construct a Chinese multimodal mathematical problem dataset to fill the vacancy of Chinese multimodal data in this field. Combined with objective and subjective indicators, experiments on multiple datasets fully verify the effectiveness of the proposed framework and model.
     </details>

128. **Step-Controlled DPO: Leveraging Stepwise Error for Enhanced Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2407.00782) `2024-07-14` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes Step-Controlled DPO (SCDPO), a method for automatically providing stepwise error supervision by creating negative samples of mathematical reasoning rationales that start making errors at a specified step, which can better align the model to understand reasoning errors and output accurate reasoning steps.
     </details>


     <details>
          <summary>Abstract</summary>
          Direct Preference Optimization (DPO) has proven effective at improving the performance of large language models (LLMs) on downstream tasks such as reasoning and alignment. In this work, we propose Step-Controlled DPO (SCDPO), a method for automatically providing stepwise error supervision by creating negative samples of mathematical reasoning rationales that start making errors at a specified step. By applying these samples in DPO training, SCDPO can better align the model to understand reasoning errors and output accurate reasoning steps. We apply SCDPO to both code-integrated and chain-of-thought solutions, empirically showing that it consistently improves the performance compared to naive DPO on three different SFT models, including one existing SFT model and two models we finetuned. Qualitative analysis of the credit assignment of SCDPO and DPO demonstrates the effectiveness of SCDPO at identifying errors in mathematical solutions. We then apply SCDPO to an InternLM2-20B model, resulting in a 20B model that achieves high scores of 88.5% on GSM8K and 58.1% on MATH, rivaling all other open-source LLMs, showing the great potential of our method.
     </details>

129. **Stepwise Verification and Remediation of Student Reasoning Errors with Large Language Model Tutors** [[pdf]](https://arxiv.org/abs/2407.09136v1) `2024-07-12` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work focuses on verifying student solutions and shows how grounding to such verification improves the overall quality of tutor response generation and proposes and evaluates several verifiers for detecting student's errors.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) present an opportunity to scale high-quality personalized education to all. A promising approach towards this means is to build dialog tutoring models that scaffold students' problem-solving. However, even though existing LLMs perform well in solving reasoning questions, they struggle to precisely detect student's errors and tailor their feedback to these errors. Inspired by real-world teaching practice where teachers identify student errors and customize their response based on them, we focus on verifying student solutions and show how grounding to such verification improves the overall quality of tutor response generation. We collect a dataset of 1K stepwise math reasoning chains with the first error step annotated by teachers. We show empirically that finding the mistake in a student solution is challenging for current models. We propose and evaluate several verifiers for detecting these errors. Using both automatic and human evaluation we show that the student solution verifiers steer the generation model towards highly targeted responses to student errors which are more often correct with less hallucinations compared to existing baselines.
     </details>

130. **Token-Supervised Value Models for Enhancing Mathematical Reasoning Capabilities of Large Language Models** [[pdf]](http://arxiv.org/abs/2407.12863) `2024-07-12` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Experimental results on mathematical reasoning benchmarks show that Token-Supervised Value Model (TVM) can outperform step-by-step verifiers on GSM8K and MATH with Mistral and Llama.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have demonstrated impressive problem-solving capabilities in mathematics through step-by-step reasoning chains. However, they are susceptible to reasoning errors that impact the quality of subsequent reasoning chains and the final answer due to language models' autoregressive token-by-token generating nature. Recent works have proposed adopting external verifiers to guide the generation of reasoning paths, but existing works utilize models that have been trained with step-by-step labels to assess the correctness of token-by-token reasoning chains. Consequently, they struggle to recognize discriminative details of tokens within a reasoning path and lack the ability to evaluate whether an intermediate reasoning path is on a promising track toward the correct final answer. To amend the lack of sound and token-grained math-verification signals, we devise a novel training scheme for verifiers that apply token-level supervision with the expected cumulative reward (i.e., value). Furthermore, we propose a practical formulation of the cumulative reward by reducing it to finding the probability of future correctness of the final answer and thereby enabling the empirical estimation of the value. Experimental results on mathematical reasoning benchmarks show that Token-Supervised Value Model (TVM) can outperform step-by-step verifiers on GSM8K and MATH with Mistral and Llama.
     </details>

131. **Is Your Model Really A Good Math Reasoner? Evaluating Mathematical Reasoning with Checklist** [[pdf]](http://arxiv.org/abs/2407.08733) `2024-07-11` (4 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is argued that if a model really understands a problem, it should be robustly and readily applied across a diverse array of tasks, and introduced MATHCHECK, a well-designed checklist for testing task generalization and reasoning robustness, as well as an automatic tool to generate checklists efficiently.
     </details>


     <details>
          <summary>Abstract</summary>
          Exceptional mathematical reasoning ability is one of the key features that demonstrate the power of large language models (LLMs). How to comprehensively define and evaluate the mathematical abilities of LLMs, and even reflect the user experience in real-world scenarios, has emerged as a critical issue. Current benchmarks predominantly concentrate on problem-solving capabilities, which presents a substantial risk of model overfitting and fails to accurately represent genuine mathematical reasoning abilities. In this paper, we argue that if a model really understands a problem, it should be robustly and readily applied across a diverse array of tasks. Motivated by this, we introduce MATHCHECK, a well-designed checklist for testing task generalization and reasoning robustness, as well as an automatic tool to generate checklists efficiently. MATHCHECK includes multiple mathematical reasoning tasks and robustness test types to facilitate a comprehensive evaluation of both mathematical reasoning ability and behavior testing. Utilizing MATHCHECK, we develop MATHCHECK-GSM and MATHCHECK-GEO to assess mathematical textual reasoning and multi-modal reasoning capabilities, respectively, serving as upgraded versions of benchmarks including GSM8k, GeoQA, UniGeo, and Geometry3K. We adopt MATHCHECK-GSM and MATHCHECK-GEO to evaluate over 20 LLMs and 11 MLLMs, assessing their comprehensive mathematical reasoning abilities. Our results demonstrate that while frontier LLMs like GPT-4o continue to excel in various abilities on the checklist, many other model families exhibit a significant decline. Further experiments indicate that, compared to traditional math benchmarks, MATHCHECK better reflects true mathematical abilities and represents mathematical intelligence more linearly, thereby supporting our design. On our MATHCHECK, we can easily conduct detailed behavior analysis to deeply investigate models.
     </details>

132. **MAVIS: Mathematical Visual Instruction Tuning** [[pdf]](http://arxiv.org/abs/2407.08739) `2024-07-11` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper proposes MAVIS, the first MAthematical VISual instruction tuning paradigm for MLLMs, involving a series of mathematical visual datasets and specialized MLLMs, and introduces MAVIS-Instruct, which is adopted to instruct-tune the MLLM for robust mathematical reasoning skills.
     </details>


     <details>
          <summary>Abstract</summary>
          Multi-modal Large Language Models (MLLMs) have recently emerged as a significant focus in academia and industry. Despite their proficiency in general multi-modal scenarios, the mathematical problem-solving capabilities in visual contexts remain insufficiently explored. We identify three key areas within MLLMs that need to be improved: visual encoding of math diagrams, diagram-language alignment, and mathematical reasoning skills. This draws forth an urgent demand for large-scale, high-quality data and training pipelines in visual mathematics. In this paper, we propose MAVIS, the first MAthematical VISual instruction tuning paradigm for MLLMs, involving a series of mathematical visual datasets and specialized MLLMs. Targeting the three issues, MAVIS contains three progressive training stages from scratch. First, we curate MAVIS-Caption, consisting of 558K diagram-caption pairs, to fine-tune a math-specific vision encoder (CLIP-Math) through contrastive learning, tailored for improved diagram visual encoding. Second, we utilize MAVIS-Caption to align the CLIP-Math with a large language model (LLM) by a projection layer, enhancing vision-language alignment in mathematical domains. Third, we introduce MAVIS-Instruct, including 900K meticulously collected and annotated visual math problems, which is adopted to finally instruct-tune the MLLM for robust mathematical reasoning skills. In MAVIS-Instruct, we incorporate complete chain-of-thought (CoT) rationales for each problem, and minimize textual redundancy, thereby concentrating the model towards the visual elements. Data and Models are released at https://github.com/ZrrSkywalker/MAVIS
     </details>

133. **Exploring the Compositional Deficiency of Large Language Models in Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2405.06680) `2024-07-11` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work constructs a new dataset \textsc{MathTrap}\footnotemark[3] by introducing carefully designed logical traps into the problem descriptions of MATH and GSM8k and finds that LLMs' performance can be improved through the above external intervention.
     </details>


     <details>
          <summary>Abstract</summary>
          Human cognition exhibits systematic compositionality, the algebraic ability to generate infinite novel combinations from finite learned components, which is the key to understanding and reasoning about complex logic. In this work, we investigate the compositionality of large language models (LLMs) in mathematical reasoning. Specifically, we construct a new dataset \textsc{MathTrap}\footnotemark[3] by introducing carefully designed logical traps into the problem descriptions of MATH and GSM8k. Since problems with logical flaws are quite rare in the real world, these represent ``unseen'' cases to LLMs. Solving these requires the models to systematically compose (1) the mathematical knowledge involved in the original problems with (2) knowledge related to the introduced traps. Our experiments show that while LLMs possess both components of requisite knowledge, they do not \textbf{spontaneously} combine them to handle these novel cases. We explore several methods to mitigate this deficiency, such as natural language prompts, few-shot demonstrations, and fine-tuning. We find that LLMs' performance can be \textbf{passively} improved through the above external intervention. Overall, systematic compositionality remains an open challenge for large language models.
     </details>

134. **Self-training Language Models for Arithmetic Reasoning** [[pdf]](http://arxiv.org/abs/2407.08400) `2024-07-11` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work explores the potential of improving models' reasoning capabilities without new data, merely using automated feedback to the validity of their predictions in arithmetic reasoning (self-training), and finds that models can substantially improve in both single-round (offline) and online self-training.
     </details>


     <details>
          <summary>Abstract</summary>
          Language models achieve impressive results in tasks involving complex multistep reasoning, but scaling these capabilities further traditionally requires expensive collection of more annotated data. In this work, we explore the potential of improving the capabilities of language models without new data, merely using automated feedback to the validity of their predictions in arithmetic reasoning (self-training). We find that models can substantially improve in both single-round (offline) and online self-training. In the offline setting, supervised methods are able to deliver gains comparable to preference optimization, but in online self-training, preference optimization shows to largely outperform supervised training thanks to superior stability and robustness on unseen types of problems.
     </details>

135. **Fuse, Reason and Verify: Geometry Problem Solving with Parsed Clauses from Diagram** [[pdf]](https://arxiv.org/abs/2407.07327v1) `2024-07-10` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper proposes a neural-symbolic model for plane geometry problem solving (PGPS), named PGPSNet-v2, with three key steps: modal fusion, reasoning process and knowledge verification, which outperforms existing symbolic and neural solvers in GPS performance, while maintaining good explainability and reliability.
     </details>


     <details>
          <summary>Abstract</summary>
          Geometry problem solving (GPS) requires capacities of multi-modal understanding, multi-hop reasoning and theorem knowledge application. In this paper, we propose a neural-symbolic model for plane geometry problem solving (PGPS), named PGPSNet-v2, with three key steps: modal fusion, reasoning process and knowledge verification. In modal fusion, we leverage textual clauses to express fine-grained structural and semantic content of geometry diagram, and fuse diagram with textual problem efficiently through structural-semantic pre-training. For reasoning, we design an explicable solution program to describe the geometric reasoning process, and employ a self-limited decoder to generate solution program autoregressively. To reduce solution errors, a multi-level theorem verifier is proposed to eliminate solutions that do not match geometric principles, alleviating the hallucination of the neural model. We also construct a large-scale geometry problem dataset called PGPS9K, containing fine-grained annotations of textual clauses, solution program and involved knowledge tuples. Extensive experiments on datasets Geometry3K and PGPS9K show that our PGPSNet solver outperforms existing symbolic and neural solvers in GPS performance, while maintaining good explainability and reliability, and the solver components (fusion, reasoning, verification) are all justified effective.
     </details>

136. **LLM Critics Help Catch Bugs in Mathematics: Towards a Better Mathematical Verifier with Natural Language Feedback** [[pdf]](http://arxiv.org/abs/2406.14024) `2024-07-08` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper proposes Math-Minos, a natural language feedback enhanced verifier by constructing automatically-generated training data and a two-stage training paradigm for effective training and efficient inference.
     </details>


     <details>
          <summary>Abstract</summary>
          Mathematical verfier achieves success in mathematical reasoning tasks by validating the correctness of solutions. However, existing verifiers are trained with binary classification labels, which are not informative enough for the model to accurately assess the solutions. To mitigate the aforementioned insufficiency of binary labels, we introduce step-wise natural language feedbacks as rationale labels (i.e., the correctness of the current step and the explanations). In this paper, we propose \textbf{Math-Minos}, a natural language feedback enhanced verifier by constructing automatically-generated training data and a two-stage training paradigm for effective training and efficient inference. Our experiments reveal that a small set (30k) of natural language feedbacks can significantly boost the performance of the verifier by the accuracy of 1.6\% (86.6\% $\rightarrow$ 88.2\%) on GSM8K and 0.8\% (37.8\% $\rightarrow$ 38.6\%) on MATH. We have released our code and data for further exploration.
     </details>

137. **Can Large Language Models Replicate ITS Feedback on Open-Ended Math Questions?** [[pdf]](http://arxiv.org/abs/2405.06414) `2024-07-08` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The capabilities of large language models (LLMs) to generate feedback for open-ended math questions, similar to that of an established ITS that uses a template-based approach are examined, finding that open-source and proprietary models both show promise in replicating the feedback they see during training, but do not generalize well to previously unseen student errors.
     </details>


     <details>
          <summary>Abstract</summary>
          Intelligent Tutoring Systems (ITSs) often contain an automated feedback component, which provides a predefined feedback message to students when they detect a predefined error. To such a feedback component, we often resort to template-based approaches. These approaches require significant effort from human experts to detect a limited number of possible student errors and provide corresponding feedback. This limitation is exemplified in open-ended math questions, where there can be a large number of different incorrect errors. In our work, we examine the capabilities of large language models (LLMs) to generate feedback for open-ended math questions, similar to that of an established ITS that uses a template-based approach. We fine-tune both open-source and proprietary LLMs on real student responses and corresponding ITS-provided feedback. We measure the quality of the generated feedback using text similarity metrics. We find that open-source and proprietary models both show promise in replicating the feedback they see during training, but do not generalize well to previously unseen student errors. These results suggest that despite being able to learn the formatting of feedback, LLMs are not able to fully understand mathematical errors made by students.
     </details>

138. **Fill in the Blank: Exploring and Enhancing LLM Capabilities for Backward Reasoning in Math Word Problems** [[pdf]](http://arxiv.org/abs/2310.01991) `2024-07-07` (8 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel Bayesian formulation for creating an ensemble over the base methods to further boost the accuracy of LLMs on the backward reasoning task, with the ensemble-based method resulting in significant performance gains compared to the SOTA forward reasoning strategies the authors adapt.
     </details>


     <details>
          <summary>Abstract</summary>
          While forward reasoning (i.e., find the answer given the question) has been explored extensively in recent literature, backward reasoning is relatively unexplored. We examine the backward reasoning capabilities of LLMs on Math Word Problems (MWPs): given a mathematical question and its answer, with some details omitted from the question, can LLMs effectively retrieve the missing information? On modifying three benchmark datasets for this task, to evaluate this task: GSM8k, SVAMP, and MultiArith, we find a significant drop in the accuracy of models on this task compared to forward reasoning across SOTA LLMs (GPT4, GPT3.5, PaLM-2, and LLaMa). Motivated by the fact backward reasoning can be seen as the ''inverse'' of forward reasoning, we propose variations of three different forward reasoning strategies to improve performance. Rephrase reformulates the given problem into a forward reasoning problem, PAL-Tools combines the idea of Program-Aided LLMs to produce a set of equations that can be solved by an external solver, and Check your Work exploits the availability of natural verifier of high accuracy in the forward direction, interleaving solving and verification steps. Finally, realizing that each of our base methods correctly solves a different set of problems, we propose a novel Bayesian formulation for creating an ensemble over the base methods to further boost the accuracy. Extensive experimentation demonstrates successive improvement in the performance of LLMs on the backward reasoning task, using our strategies, with our ensemble-based method resulting in significant performance gains compared to the SOTA forward reasoning strategies we adapt.
     </details>

139. **LogicVista: Multimodal LLM Logical Reasoning Benchmark in Visual Contexts** [[pdf]](https://arxiv.org/abs/2407.04973v1) `2024-07-06` (5 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          LogicVista is proposed, an evaluation benchmark that assesses the integrated logical reasoning capabilities of multimodal large language models (MLLMs) in Visual contexts using a sample of 448 multiple-choice questions.
     </details>


     <details>
          <summary>Abstract</summary>
          We propose LogicVista, an evaluation benchmark that assesses the integrated logical reasoning capabilities of multimodal large language models (MLLMs) in Visual contexts. Recent advancements in MLLMs have demonstrated various fascinating abilities, from crafting poetry based on an image to performing mathematical reasoning. However, there is still a lack of systematic evaluation of MLLMs' proficiency in logical reasoning tasks, which are essential for activities like navigation and puzzle-solving. Thus we evaluate general logical cognition abilities across 5 logical reasoning tasks encompassing 9 different capabilities, using a sample of 448 multiple-choice questions. Each question is annotated with the correct answer and the human-written reasoning behind the selection, enabling both open-ended and multiple-choice evaluation. A total of 8 MLLMs are comprehensively evaluated using LogicVista. Code and Data Available at https://github.com/Yijia-Xiao/LogicVista.
     </details>

140. **Solving for X and Beyond: Can Large Language Models Solve Complex Math Problems with More-Than-Two Unknowns?** [[pdf]](https://arxiv.org/abs/2407.05134v1) `2024-07-06` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The Formulate-and-Solve strategy, a generalized prompting approach that effectively handles problems with an arbitrary number of unknowns, is proposed, and is revealed to enhance LLM performance on the BeyondX benchmark but also provides deeper insights into the computational limits of LLMs when faced with more complex mathematical challenges.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have demonstrated remarkable performance in solving math problems, a hallmark of human intelligence. Despite high success rates on current benchmarks; however, these often feature simple problems with only one or two unknowns, which do not sufficiently challenge their reasoning capacities. This paper introduces a novel benchmark, BeyondX, designed to address these limitations by incorporating problems with multiple unknowns. Recognizing the challenges in proposing multi-unknown problems from scratch, we developed BeyondX using an innovative automated pipeline that progressively increases complexity by expanding the number of unknowns in simpler problems. Empirical study on BeyondX reveals that the performance of existing LLMs, even those fine-tuned specifically on math tasks, significantly decreases as the number of unknowns increases - with a performance drop of up to 70\% observed in GPT-4. To tackle these challenges, we propose the Formulate-and-Solve strategy, a generalized prompting approach that effectively handles problems with an arbitrary number of unknowns. Our findings reveal that this strategy not only enhances LLM performance on the BeyondX benchmark but also provides deeper insights into the computational limits of LLMs when faced with more complex mathematical challenges.
     </details>

141. **Towards Automated Functional Equation Proving: A Benchmark Dataset and A Domain-Specific In-Context Agent** [[pdf]](https://arxiv.org/abs/2407.14521v1) `2024-07-05` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          FEAS, an agent that enhances the COPRA in-context learning framework within Lean, is introduced, an agent that enhances the COPRA in-context learning framework within Lean and outperforms baselines on FunEq, particularly with the integration of domain-specific heuristics.
     </details>


     <details>
          <summary>Abstract</summary>
          Automated Theorem Proving (ATP) faces challenges due to its complexity and computational demands. Recent work has explored using Large Language Models (LLMs) for ATP action selection, but these methods can be resource-intensive. This study introduces FEAS, an agent that enhances the COPRA in-context learning framework within Lean. FEAS refines prompt generation, response parsing, and incorporates domain-specific heuristics for functional equations. It introduces FunEq, a curated dataset of functional equation problems with varying difficulty. FEAS outperforms baselines on FunEq, particularly with the integration of domain-specific heuristics. The results demonstrate FEAS's effectiveness in generating and formalizing high-level proof strategies into Lean proofs, showcasing the potential of tailored approaches for specific ATP challenges.
     </details>

142. **AgentInstruct: Toward Generative Teaching with Agentic Flows** [[pdf]](http://arxiv.org/abs/2407.03502) `2024-07-03` (4 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces AgentInstruct, an extensible agentic framework for automatically creating large amounts of diverse and high-quality synthetic data, and demonstrates the utility of AgentInstruct by creating a post training dataset of 25M pairs to teach language models different skills.
     </details>


     <details>
          <summary>Abstract</summary>
          Synthetic data is becoming increasingly important for accelerating the development of language models, both large and small. Despite several successful use cases, researchers also raised concerns around model collapse and drawbacks of imitating other models. This discrepancy can be attributed to the fact that synthetic data varies in quality and diversity. Effective use of synthetic data usually requires significant human effort in curating the data. We focus on using synthetic data for post-training, specifically creating data by powerful models to teach a new skill or behavior to another model, we refer to this setting as Generative Teaching. We introduce AgentInstruct, an extensible agentic framework for automatically creating large amounts of diverse and high-quality synthetic data. AgentInstruct can create both the prompts and responses, using only raw data sources like text documents and code files as seeds. We demonstrate the utility of AgentInstruct by creating a post training dataset of 25M pairs to teach language models different skills, such as text editing, creative writing, tool usage, coding, reading comprehension, etc. The dataset can be used for instruction tuning of any base model. We post-train Mistral-7b with the data. When comparing the resulting model Orca-3 to Mistral-7b-Instruct (which uses the same base model), we observe significant improvements across many benchmarks. For example, 40% improvement on AGIEval, 19% improvement on MMLU, 54% improvement on GSM8K, 38% improvement on BBH and 45% improvement on AlpacaEval. Additionally, it consistently outperforms other models such as LLAMA-8B-instruct and GPT-3.5-turbo.
     </details>

143. **TheoremLlama: Transforming General-Purpose LLMs into Lean4 Experts** [[pdf]](http://arxiv.org/abs/2407.03203) `2024-07-03` `Lean` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper proposes TheoremLlama, an end-to-end framework that trains a general-purpose LLM to be a Lean4 expert, and provides Open Bootstrapped Theorems (OBT), an NL-FL aligned and bootstrapped dataset.
     </details>


     <details>
          <summary>Abstract</summary>
          Proving mathematical theorems using computer-verifiable formal languages like Lean significantly impacts mathematical reasoning. One approach to formal theorem proving involves generating complete proofs using Large Language Models (LLMs) based on Natural Language (NL) proofs. Similar methods have shown promising results in code generation. However, most modern LLMs exhibit suboptimal performance due to the scarcity of aligned NL and Formal Language (FL) theorem-proving data. This scarcity results in a paucity of methodologies for training LLMs and techniques to fully utilize their capabilities in composing formal proofs. To address the challenges, this paper proposes **TheoremLlama**, an end-to-end framework to train a general-purpose LLM to become a Lean4 expert. This framework encompasses NL-FL aligned dataset generation methods, training approaches for the LLM formal theorem prover, and techniques for LLM Lean4 proof writing. Using the dataset generation method, we provide *Open Bootstrapped Theorems* (OBT), an NL-FL aligned and bootstrapped dataset. A key innovation in this framework is the NL-FL bootstrapping method, where NL proofs are integrated into Lean4 code for training datasets, leveraging the NL reasoning ability of LLMs for formal reasoning. The **TheoremLlama** framework achieves cumulative accuracies of 36.48% and 33.61% on MiniF2F-Valid and Test datasets respectively, surpassing the GPT-4 baseline of 22.95% and 25.41%. We have also open-sourced our model checkpoints and generated dataset, and will soon make all the code publicly available.
     </details>

144. **Artificial intelligence and machine learning generated conjectures with TxGraffiti** [[pdf]](https://arxiv.org/abs/2407.02731v1) `2024-07-03` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          \emph{TxGraffiti} is a machine learning and heuristic based artificial intelligence designed to automate the task of conjecturing in mathematics. Since its inception, TxGraffiti has generated many surprising conjectures leading to publication in respectable mathematical journals. In this paper we outline the machine learning and heuristic techniques implemented by TxGraffiti. We also recall its contributions to the mathematical literature and announce a new online version of the program available for anyone curious to explore conjectures in graph theory.
     </details>

145. **We-Math: Does Your Large Multimodal Model Achieve Human-like Mathematical Reasoning?** [[pdf]](http://arxiv.org/abs/2407.01284) `2024-07-01` (6 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The IK issue of LMMs can be effectively improved via knowledge augmentation strategies and the primary challenge of GPT-4o has significantly transitioned from IK to IG, establishing it as the first LMM advancing towards the knowledge generalization stage.
     </details>


     <details>
          <summary>Abstract</summary>
          Visual mathematical reasoning, as a fundamental visual reasoning ability, has received widespread attention from the Large Multimodal Models (LMMs) community. Existing benchmarks, such as MathVista and MathVerse, focus more on the result-oriented performance but neglect the underlying principles in knowledge acquisition and generalization. Inspired by human-like mathematical reasoning, we introduce WE-MATH, the first benchmark specifically designed to explore the problem-solving principles beyond end-to-end performance. We meticulously collect and categorize 6.5K visual math problems, spanning 67 hierarchical knowledge concepts and five layers of knowledge granularity. We decompose composite problems into sub-problems according to the required knowledge concepts and introduce a novel four-dimensional metric, namely Insufficient Knowledge (IK), Inadequate Generalization (IG), Complete Mastery (CM), and Rote Memorization (RM), to hierarchically assess inherent issues in LMMs' reasoning process. With WE-MATH, we conduct a thorough evaluation of existing LMMs in visual mathematical reasoning and reveal a negative correlation between solving steps and problem-specific performance. We confirm the IK issue of LMMs can be effectively improved via knowledge augmentation strategies. More notably, the primary challenge of GPT-4o has significantly transitioned from IK to IG, establishing it as the first LMM advancing towards the knowledge generalization stage. In contrast, other LMMs exhibit a marked inclination towards Rote Memorization - they correctly solve composite problems involving multiple knowledge concepts yet fail to answer sub-problems. We anticipate that WE-MATH will open new pathways for advancements in visual mathematical reasoning for LMMs. The WE-MATH data and evaluation code are available at https://github.com/We-Math/We-Math.
     </details>

146. **MalAlgoQA: A Pedagogical Approach for Evaluating Counterfactual Reasoning Abilities** [[pdf]](https://arxiv.org/abs/2407.00938v1) `2024-07-01` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is found that the chain-of-thought prompting technique not only fails to consistently enhance MIA, but can also lead to underperformance compared to simple prompting, which holds significant implications for the development of more cognitively-inspired LLMs to improve their counterfactual reasoning abilities.
     </details>


     <details>
          <summary>Abstract</summary>
          This paper introduces MalAlgoQA, a novel dataset designed to evaluate the counterfactual reasoning capabilities of Large Language Models (LLMs) through a pedagogical approach. The dataset comprises mathematics and reading comprehension questions, each accompanied by four answer choices and their corresponding rationales. We focus on the incorrect answer rationales, termed "malgorithms", which highlights flawed reasoning steps leading to incorrect answers and offers valuable insights into erroneous thought processes. We also propose the Malgorithm Identification task, where LLMs are assessed based on their ability to identify corresponding malgorithm given an incorrect answer choice. To evaluate the model performance, we introduce two metrics: Algorithm Identification Accuracy (AIA) for correct answer rationale identification, and Malgorithm Identification Accuracy (MIA) for incorrect answer rationale identification. The task is challenging since state-of-the-art LLMs exhibit significant drops in MIA as compared to AIA. Moreover, we find that the chain-of-thought prompting technique not only fails to consistently enhance MIA, but can also lead to underperformance compared to simple prompting. These findings hold significant implications for the development of more cognitively-inspired LLMs to improve their counterfactual reasoning abilities, particularly through a pedagogical perspective where understanding and rectifying student misconceptions are crucial.
     </details>

147. **FRoG: Evaluating Fuzzy Reasoning of Generalized Quantifiers in Large Language Models** [[pdf]](https://arxiv.org/abs/2407.01046v2) `2024-07-01` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A new benchmark, FRoG, is introduced for fuzzy reasoning, featuring real-world mathematical word problems that incorporate generalized quantifiers, and it is found that existing methods designed to enhance reasoning do not consistently improve performance in tasks involving fuzzy logic.
     </details>


     <details>
          <summary>Abstract</summary>
          Fuzzy reasoning is vital due to the frequent use of imprecise information in daily contexts. However, the ability of current large language models (LLMs) to handle such reasoning remains largely uncharted. In this paper, we introduce a new benchmark, FRoG, for fuzzy reasoning, featuring real-world mathematical word problems that incorporate generalized quantifiers. Our experimental findings reveal that fuzzy reasoning continues to pose significant challenges for LLMs. Moreover, we find that existing methods designed to enhance reasoning do not consistently improve performance in tasks involving fuzzy logic. Additionally, our results show an inverse scaling effect in the performance of LLMs on FRoG. Interestingly, we also demonstrate that strong mathematical reasoning skills are not necessarily indicative of success on our benchmark.
     </details>

148. **Mamo: a Mathematical Modeling Benchmark with Solvers** [[pdf]](http://arxiv.org/abs/2405.13144) `2024-06-30` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces a new benchmark, Mamo, that transcends traditional result-oriented assessments of Large Language Models' mathematical modeling capabilities and sets a new standard for evaluating their performance in complex problem-solving scenarios.
     </details>


     <details>
          <summary>Abstract</summary>
          Mathematical modeling involves representing real-world phenomena, systems, or problems using mathematical expressions and equations to analyze, understand, and predict their behavior. Given that this process typically requires experienced experts, there is an interest in exploring whether Large Language Models (LLMs) can undertake mathematical modeling to potentially decrease human labor. To evaluate of LLMs in mathematical modeling, we introduce a new benchmark, Mamo, that transcends traditional result-oriented assessments. Unlike conventional methods that primarily assess LLMs based on the accuracy of solutions to mathematical problems, our approach offers deeper insight into the modeling process itself. By focusing on the processes LLMs undertake rather than the correctness of their final solutions, Mamo pioneers a novel evaluation paradigm. This shift underscores the importance of understanding the inherent modeling capabilities of LLMs, paving the way for a more nuanced and comprehensive analysis of their problem-solving strategies. Our work marks a significant advancement in the field, suggesting a new direction for future research by emphasizing the evaluation of LLMs' modeling processes over the mere correctness of answers. This benchmark not only facilitates a better understanding of LLMs' mathematical modeling capabilities but also sets a new standard for evaluating their performance in complex problem-solving scenarios.
     </details>

149. **MathCAMPS: Fine-grained Synthesis of Mathematical Problems From Human Curricula** [[pdf]](http://arxiv.org/abs/2407.00900) `2024-06-30` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes MathCAMPS: a method to synthesize high-quality mathematical problems at scale, grounded on 44 fine-grained standards from the Mathematics Common Core (CC) Standard for K-8 grades, and proposes a cycle-consistency method for validating problem faithfulness.
     </details>


     <details>
          <summary>Abstract</summary>
          Mathematical problem solving is an important skill for Large Language Models (LLMs), both as an important capability and a proxy for a range of reasoning abilities. Existing benchmarks probe a diverse set of skills, but they yield aggregate accuracy metrics, obscuring specific abilities or weaknesses. Furthermore, they are difficult to extend with new problems, risking data contamination over time. To address these challenges, we propose MathCAMPS: a method to synthesize high-quality mathematical problems at scale, grounded on 44 fine-grained "standards" from the Mathematics Common Core (CC) Standard for K-8 grades. We encode each standard in a formal grammar, allowing us to sample diverse symbolic problems and their answers. We then use LLMs to realize the symbolic problems into word problems. We propose a cycle-consistency method for validating problem faithfulness. Finally, we derive follow-up questions from symbolic structures and convert them into follow-up word problems - a novel task of mathematical dialogue that probes for robustness in understanding. Experiments on 23 LLMs show surprising failures even in the strongest models (in particular when asked simple follow-up questions). Moreover, we evaluate training checkpoints of Pythia 12B on MathCAMPS, allowing us to analyze when particular mathematical skills develop during its training. Our framework enables the community to reproduce and extend our pipeline for a fraction of the typical cost of building new high-quality datasets.
     </details>

150. **How to Leverage Digit Embeddings to Represent Numbers?** [[pdf]](http://arxiv.org/abs/2407.00894) `2024-06-30` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper explores the use of mathematical priors to compute aggregated digit embeddings and explicitly incorporate these aggregates into transformer models and evaluates the effectiveness of incorporating this explicit aggregation.
     </details>


     <details>
          <summary>Abstract</summary>
          Apart from performing arithmetic operations, understanding numbers themselves is still a challenge for existing language models. Simple generalisations, such as solving 100+200 instead of 1+2, can substantially affect model performance (Sivakumar and Moosavi, 2023). Among various techniques, character-level embeddings of numbers have emerged as a promising approach to improve number representation. However, this method has limitations as it leaves the task of aggregating digit representations to the model, which lacks direct supervision for this process. In this paper, we explore the use of mathematical priors to compute aggregated digit embeddings and explicitly incorporate these aggregates into transformer models. This can be achieved either by adding a special token to the input embeddings or by introducing an additional loss function to enhance correct predictions. We evaluate the effectiveness of incorporating this explicit aggregation, analysing its strengths and shortcomings, and discuss future directions to better benefit from this approach. Our methods, while simple, are compatible with any pretrained model and require only a few lines of code, which we have made publicly available.
     </details>

151. **LiteSearch: Efficacious Tree Search for LLM** [[pdf]](http://arxiv.org/abs/2407.00320) `2024-06-29` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This study introduces a novel guided tree search algorithm with dynamic node selection and node-level exploration budget (maximum number of children) calculation to tackle the issue of wasteful search strategies in LLM.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent research suggests that tree search algorithms (e.g. Monte Carlo Tree Search) can dramatically boost LLM performance on complex mathematical reasoning tasks. However, they often require more than 10 times the computational resources of greedy decoding due to wasteful search strategies, making them difficult to be deployed in practical applications. This study introduces a novel guided tree search algorithm with dynamic node selection and node-level exploration budget (maximum number of children) calculation to tackle this issue. By considering the search progress towards the final answer (history) and the guidance from a value network (future) trained without any step-wise annotations, our algorithm iteratively selects the most promising tree node before expanding it within the boundaries of the allocated computational budget. Experiments conducted on the GSM8K and TabMWP datasets demonstrate that our approach not only offers competitive performance but also enjoys significantly lower computational costs compared to baseline methods.
     </details>

152. **Advancing Process Verification for Large Language Models via Tree-Based Preference Learning** [[pdf]](https://arxiv.org/abs/2407.00390v1) `2024-06-29` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This study proposes Tree-based Preference Learning Verifier (Tree-PLV), a novel approach that constructs reasoning trees via a best-first search algorithm and collects step-level paired data for preference training and empirically evaluates it across a range of arithmetic and commonsense reasoning tasks, where it significantly outperforms existing benchmarks.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have demonstrated remarkable potential in handling complex reasoning tasks by generating step-by-step rationales.Some methods have proven effective in boosting accuracy by introducing extra verifiers to assess these paths. However, existing verifiers, typically trained on binary-labeled reasoning paths, fail to fully utilize the relative merits of intermediate steps, thereby limiting the effectiveness of the feedback provided. To overcome this limitation, we propose Tree-based Preference Learning Verifier (Tree-PLV), a novel approach that constructs reasoning trees via a best-first search algorithm and collects step-level paired data for preference training. Compared to traditional binary classification, step-level preferences more finely capture the nuances between reasoning steps, allowing for a more precise evaluation of the complete reasoning path. We empirically evaluate Tree-PLV across a range of arithmetic and commonsense reasoning tasks, where it significantly outperforms existing benchmarks. For instance, Tree-PLV achieved substantial performance gains over the Mistral-7B self-consistency baseline on GSM8K (67.55% to 82.79%), MATH (17.00% to 26.80%), CSQA (68.14% to 72.97%), and StrategyQA (82.86% to 83.25%).Additionally, our study explores the appropriate granularity for applying preference learning, revealing that step-level guidance provides feedback that better aligns with the evaluation of the reasoning process.
     </details>

153. **CMMaTH: A Chinese Multi-modal Math Skill Evaluation Benchmark for Foundation Models** [[pdf]](https://arxiv.org/abs/2407.12023v1) `2024-06-28` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Due to the rapid advancements in multimodal large language models, evaluating their multimodal mathematical capabilities continues to receive wide attention. Despite the datasets like MathVista proposed benchmarks for assessing mathematical capabilities in multimodal scenarios, there is still a lack of corresponding evaluation tools and datasets for fine-grained assessment in the context of K12 education in Chinese language. To systematically evaluate the capability of multimodal large models in solving Chinese multimodal mathematical problems, we propose a Chinese Multi-modal Math Skill Evaluation Benchmark, named CMMaTH, contraining 23k multimodal K12 math related questions, forming the largest Chinese multimodal mathematical problem benchmark to date. CMMaTH questions from elementary to high school levels, provide increased diversity in problem types, solution objectives, visual elements, detailed knowledge points, and standard solution annotations. We have constructed an open-source tool GradeGPT integrated with the CMMaTH dataset, facilitating stable, rapid, and cost-free model evaluation. Our data and code are available.
     </details>

154. **LiveBench: A Challenging, Contamination-Free LLM Benchmark** [[pdf]](http://arxiv.org/abs/2406.19314) `2024-06-27` (12 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces a new benchmark for LLMs designed to be immune to both test set contamination and the pitfalls of LLM judging and human crowdsourcing, and releases LiveBench, the first benchmark that contains frequently-updated questions from recent information sources, and scores answers automatically according to objective ground-truth values.
     </details>


     <details>
          <summary>Abstract</summary>
          Test set contamination, wherein test data from a benchmark ends up in a newer model's training set, is a well-documented obstacle for fair LLM evaluation and can quickly render benchmarks obsolete. To mitigate this, many recent benchmarks crowdsource new prompts and evaluations from human or LLM judges; however, these can introduce significant biases, and break down when scoring hard questions. In this work, we introduce a new benchmark for LLMs designed to be immune to both test set contamination and the pitfalls of LLM judging and human crowdsourcing. We release LiveBench, the first benchmark that (1) contains frequently-updated questions from recent information sources, (2) scores answers automatically according to objective ground-truth values, and (3) contains a wide variety of challenging tasks, spanning math, coding, reasoning, language, instruction following, and data analysis. To achieve this, LiveBench contains questions that are based on recently-released math competitions, arXiv papers, news articles, and datasets, and it contains harder, contamination-free versions of tasks from previous benchmarks such as Big-Bench Hard, AMPS, and IFEval. We evaluate many prominent closed-source models, as well as dozens of open-source models ranging from 0.5B to 110B in size. LiveBench is difficult, with top models achieving below 65% accuracy. We release all questions, code, and model answers. Questions will be added and updated on a monthly basis, and we will release new tasks and harder versions of tasks over time so that LiveBench can distinguish between the capabilities of LLMs as they improve in the future. We welcome community engagement and collaboration for expanding the benchmark tasks and models.
     </details>

155. **DiVERT: Distractor Generation with Variational Errors Represented as Text for Math Multiple-choice Questions** [[pdf]](https://arxiv.org/abs/2406.19356v1) `2024-06-27` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduces DiVERT (Distractor Generation with Variational Errors Represented as Text), a novel variational approach that learns an interpretable representation of errors behind distractors in math MCQs and finds that DiVERT leads to error labels that are of comparable quality to human-authored ones.
     </details>


     <details>
          <summary>Abstract</summary>
          High-quality distractors are crucial to both the assessment and pedagogical value of multiple-choice questions (MCQs), where manually crafting ones that anticipate knowledge deficiencies or misconceptions among real students is difficult. Meanwhile, automated distractor generation, even with the help of large language models (LLMs), remains challenging for subjects like math. It is crucial to not only identify plausible distractors but also understand the error behind them. In this paper, we introduce DiVERT (Distractor Generation with Variational Errors Represented as Text), a novel variational approach that learns an interpretable representation of errors behind distractors in math MCQs. Through experiments on a real-world math MCQ dataset with 1,434 questions used by hundreds of thousands of students, we show that DiVERT, despite using a base open-source LLM with 7B parameters, outperforms state-of-the-art approaches using GPT-4o on downstream distractor generation. We also conduct a human evaluation with math educators and find that DiVERT leads to error labels that are of comparable quality to human-authored ones.
     </details>

156. **Evaluating LLMs' Mathematical and Coding Competency through Ontology-guided Interventions** [[pdf]](http://arxiv.org/abs/2401.09395) `2024-06-27` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Through comprehensive evaluations of both closed-source and open-source LLMs, a significant performance drop is shown across all the models against the perturbed questions, suggesting that the current LLMs lack robust problem solving skills and structured reasoning abilities in many areas, as defined by the ontology.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent advancements in Large Language Models (LLMs) have showcased striking results on existing logical reasoning benchmarks, with some models even surpassing human performance. However, the true depth of their competencies and robustness in reasoning tasks remains an open question. To this end, in this paper, we focus on two popular reasoning tasks: arithmetic reasoning and code generation. Particularly, we introduce: (i) a general ontology of perturbations for maths and coding questions, (ii) a semi-automatic method to apply these perturbations, and (iii) two datasets, MORE and CORE, respectively, of perturbed maths and coding problems to probe the limits of LLM capabilities in numeric reasoning and coding tasks. Through comprehensive evaluations of both closed-source and open-source LLMs, we show a significant performance drop across all the models against the perturbed questions, suggesting that the current LLMs lack robust problem solving skills and structured reasoning abilities in many areas, as defined by our ontology. We open source the datasets and source codes at: https://github.com/declare-lab/llm_robustness.
     </details>

157. **Step-DPO: Step-wise Preference Optimization for Long-chain Reasoning of LLMs** [[pdf]](https://arxiv.org/abs/2406.18629v1) `2024-06-26` (7 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a simple, effective, and data-efficient method called Step-DPO, which treats individual reasoning steps as units for preference optimization rather than evaluating answers holistically, and observes that in DPO, self-generated data is more effective than data generated by humans or GPT-4, due to the latter's out-of-distribution nature.
     </details>


     <details>
          <summary>Abstract</summary>
          Mathematical reasoning presents a significant challenge for Large Language Models (LLMs) due to the extensive and precise chain of reasoning required for accuracy. Ensuring the correctness of each reasoning step is critical. To address this, we aim to enhance the robustness and factuality of LLMs by learning from human feedback. However, Direct Preference Optimization (DPO) has shown limited benefits for long-chain mathematical reasoning, as models employing DPO struggle to identify detailed errors in incorrect answers. This limitation stems from a lack of fine-grained process supervision. We propose a simple, effective, and data-efficient method called Step-DPO, which treats individual reasoning steps as units for preference optimization rather than evaluating answers holistically. Additionally, we have developed a data construction pipeline for Step-DPO, enabling the creation of a high-quality dataset containing 10K step-wise preference pairs. We also observe that in DPO, self-generated data is more effective than data generated by humans or GPT-4, due to the latter's out-of-distribution nature. Our findings demonstrate that as few as 10K preference data pairs and fewer than 500 Step-DPO training steps can yield a nearly 3% gain in accuracy on MATH for models with over 70B parameters. Notably, Step-DPO, when applied to Qwen2-72B-Instruct, achieves scores of 70.8% and 94.0% on the test sets of MATH and GSM8K, respectively, surpassing a series of closed-source models, including GPT-4-1106, Claude-3-Opus, and Gemini-1.5-Pro. Our code, data, and models are available at https://github.com/dvlab-research/Step-DPO.
     </details>

158. **Math-LLaVA: Bootstrapping Mathematical Reasoning for Multimodal Large Language Models** [[pdf]](http://arxiv.org/abs/2406.17294) `2024-06-26` (5 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This research addresses the lack of high-quality, diverse multimodal mathematical datasets by collecting 40K high-quality images with question-answer pairs from 24 existing datasets and synthesizing 320K new pairs, creating the MathV360K dataset, which enhances both the breadth and depth of multimodal mathematical questions.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have demonstrated impressive reasoning capabilities, particularly in textual mathematical problem-solving. However, existing open-source image instruction fine-tuning datasets, containing limited question-answer pairs per image, do not fully exploit visual information to enhance the multimodal mathematical reasoning capabilities of Multimodal LLMs (MLLMs). To bridge this gap, we address the lack of high-quality, diverse multimodal mathematical datasets by collecting 40K high-quality images with question-answer pairs from 24 existing datasets and synthesizing 320K new pairs, creating the MathV360K dataset, which enhances both the breadth and depth of multimodal mathematical questions. We introduce Math-LLaVA, a LLaVA-1.5-based model fine-tuned with MathV360K. This novel approach significantly improves the multimodal mathematical reasoning capabilities of LLaVA-1.5, achieving a 19-point increase and comparable performance to GPT-4V on MathVista's minitest split. Furthermore, Math-LLaVA demonstrates enhanced generalizability, showing substantial improvements on the MMMU benchmark. Our research highlights the importance of dataset diversity and synthesis in advancing MLLMs' mathematical reasoning abilities. The code and data are available at: \url{https://github.com/HZQ950419/Math-LLaVA}.
     </details>

159. **MindStar: Enhancing Math Reasoning in Pre-trained LLMs at Inference Time** [[pdf]](http://arxiv.org/abs/2405.16265) `2024-06-26` (4 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper proposes a purely inference-based searching method -- MindStar (M*), which significantly enhances the reasoning abilities of open-source models, and achieves comparable performance to GPT-3.5 and Grok-1, but with substantially reduced model size and computational costs.
     </details>


     <details>
          <summary>Abstract</summary>
          Although Large Language Models (LLMs) achieve remarkable performance across various tasks, they often struggle with complex reasoning tasks, such as answering mathematical questions. Recent efforts to address this issue have primarily focused on leveraging mathematical datasets through supervised fine-tuning or self-improvement techniques. However, these methods often depend on high-quality datasets that are difficult to prepare, or they require substantial computational resources for fine-tuning. Inspired by findings that LLMs know how to produce the right answer but struggle to select the correct reasoning path, we propose a purely inference-based searching method -- MindStar (M*). This method formulates reasoning tasks as searching problems and proposes two search ideas to identify the optimal reasoning paths. We evaluate the M* framework on both the GSM8K and MATH datasets, comparing its performance with existing open and closed-source LLMs. Our results demonstrate that M* significantly enhances the reasoning abilities of open-source models, such as Llama-2-13B and Mistral-7B, and achieves comparable performance to GPT-3.5 and Grok-1, but with substantially reduced model size and computational costs.
     </details>

160. **MathOdyssey: Benchmarking Mathematical Problem-Solving Skills in Large Language Models Using Odyssey Math Data** [[pdf]](http://arxiv.org/abs/2406.18321) `2024-06-26` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is indicated that while LLMs perform well on routine and moderately difficult tasks, they face significant challenges with Olympiad-level problems and complex university-level questions, particularly with the most demanding problems.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have significantly advanced natural language understanding and demonstrated strong problem-solving abilities. Despite these successes, most LLMs still struggle with solving mathematical problems due to the intricate reasoning required. This paper investigates the mathematical problem-solving capabilities of LLMs using the newly developed "MathOdyssey" dataset. The dataset includes diverse mathematical problems at high school and university levels, created by experts from notable institutions to rigorously test LLMs in advanced problem-solving scenarios and cover a wider range of subject areas. By providing the MathOdyssey dataset as a resource to the AI community, we aim to contribute to the understanding and improvement of AI capabilities in complex mathematical problem-solving. We conduct benchmarking on open-source models, such as Llama-3 and DBRX-Instruct, and closed-source models from the GPT series and Gemini models. Our results indicate that while LLMs perform well on routine and moderately difficult tasks, they face significant challenges with Olympiad-level problems and complex university-level questions. Our analysis shows a narrowing performance gap between open-source and closed-source models, yet substantial challenges remain, particularly with the most demanding problems. This study highlights the ongoing need for research to enhance the mathematical reasoning of LLMs. The dataset, results, and code are publicly available.
     </details>

161. **Multiple-Choice Questions are Efficient and Robust LLM Evaluators** [[pdf]](http://arxiv.org/abs/2405.11966) `2024-06-26` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          GSM-MC, a multiple-choice (MC) dataset constructed by collecting answers and incorrect predictions on GSM8K from 60 open-source models, is presented and MATH-MC, constructed from MATH, and PythonIO, a new program reasoning MC dataset constructed from HumanEval and MBPP are introduced.
     </details>


     <details>
          <summary>Abstract</summary>
          We present GSM-MC, a multiple-choice (MC) dataset constructed by collecting answers and incorrect predictions on GSM8K from 60 open-source models. Through extensive experiments, we show that LLMs' performance on the MC version of this popular benchmark is strongly correlated with their performance on the original version and is quite robust to distractor choices and option orders, while the evaluation time is reduced by a factor of up to 30. Following similar procedures, we introduce MATH-MC, constructed from MATH, and PythonIO, a new program reasoning MC dataset constructed from HumanEval and MBPP. Experimental results indicate that LLMs' performance on these MC benchmarks leaves much room for improvement. Our data and code are available at https://github.com/Geralt-Targaryen/MC-Evaluation.
     </details>

162. **Improving Arithmetic Reasoning Ability of Large Language Models through Relation Tuples, Verification and Dynamic Feedback** [[pdf]](https://arxiv.org/abs/2406.17873v1) `2024-06-25` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper proposes to use a semi-structured form to represent reasoning steps of large language models that uses relation tuples, which are not only human-readable but also machine-friendly and easier to verify than natural language.
     </details>


     <details>
          <summary>Abstract</summary>
          Current representations used in reasoning steps of large language models can mostly be categorized into two main types: (1) natural language, which is difficult to verify; and (2) non-natural language, usually programming code, which is difficult for people who are unfamiliar with coding to read. In this paper, we propose to use a semi-structured form to represent reasoning steps of large language models. Specifically, we use relation tuples, which are not only human-readable but also machine-friendly and easier to verify than natural language. We implement a framework that includes three main components: (1) introducing relation tuples into the reasoning steps of large language models; (2) implementing an automatic verification process of reasoning steps with a local code interpreter based on relation tuples; and (3) integrating a simple and effective dynamic feedback mechanism, which we found helpful for self-improvement of large language models. The experimental results on various arithmetic datasets demonstrate the effectiveness of our method in improving the arithmetic reasoning ability of large language models. The source code is available at https://github.com/gpgg/art.
     </details>

163. **Task Oriented In-Domain Data Augmentation** [[pdf]](http://arxiv.org/abs/2406.16694) `2024-06-24` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          TRAIT, a task-oriented in-domain data augmentation framework, is proposed and adapted to two domains: advertisement and math, which improves LLM performance by 8% in the advertisement domain and 7.5% in the math domain.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have shown superior performance in various applications and fields. To achieve better performance on specialized domains such as law and advertisement, LLMs are often continue pre-trained on in-domain data. However, existing approaches suffer from two major issues. First, in-domain data are scarce compared with general domain-agnostic data. Second, data used for continual pre-training are not task-aware, such that they may not be helpful to downstream applications. We propose TRAIT, a task-oriented in-domain data augmentation framework. Our framework is divided into two parts: in-domain data selection and task-oriented synthetic passage generation. The data selection strategy identifies and selects a large amount of in-domain data from general corpora, and thus significantly enriches domain knowledge in the continual pre-training data. The synthetic passages contain guidance on how to use domain knowledge to answer questions about downstream tasks. By training on such passages, the model aligns with the need of downstream applications. We adapt LLMs to two domains: advertisement and math. On average, TRAIT improves LLM performance by 8% in the advertisement domain and 7.5% in the math domain.
     </details>

164. **The Impact of Reasoning Step Length on Large Language Models** [[pdf]](http://arxiv.org/abs/2401.04925) `ACL 2024 Findings` (28 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The results indicate that lengthening the reasoning steps in prompts, even without adding new information into the prompt, considerably enhances LLMs' reasoning abilities across multiple datasets, and shows that even incorrect rationales can yield favorable outcomes if they maintain the requisite length of inference.
     </details>


     <details>
          <summary>Abstract</summary>
          Chain of Thought (CoT) is significant in improving the reasoning abilities of large language models (LLMs). However, the correlation between the effectiveness of CoT and the length of reasoning steps in prompts remains largely unknown. To shed light on this, we have conducted several empirical experiments to explore the relations. Specifically, we design experiments that expand and compress the rationale reasoning steps within CoT demonstrations, while keeping all other factors constant. We have the following key findings. First, the results indicate that lengthening the reasoning steps in prompts, even without adding new information into the prompt, considerably enhances LLMs’ reasoning abilities across multiple datasets. Alternatively, shortening the reasoning steps, even while preserving the key information, significantly diminishes the reasoning abilities of models. This finding highlights the importance of the number of steps in CoT prompts and provides practical guidance to make better use of LLMs’ potential in complex problem-solving scenarios. Second, we also investigated the relationship between the performance of CoT and the rationales used in demonstrations. Surprisingly, the result shows that even incorrect rationales can yield favorable outcomes if they maintain the requisite length of inference. Third, we observed that the advantages of increasing reasoning steps are task-dependent: simpler tasks require fewer steps, whereas complex tasks gain significantly from longer inference sequences.
     </details>

165. **RL on Incorrect Synthetic Data Scales the Efficiency of LLM Math Reasoning by Eight-Fold** [[pdf]](https://arxiv.org/abs/2406.14532v1) `2024-06-20` (10 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is shown that training on per-step negatives can help to unlearn spurious correlations in the positive data, and is equivalent to advantage-weighted reinforcement learning (RL), implying that it inherits robustness benefits of RL over imitating positive data alone.
     </details>


     <details>
          <summary>Abstract</summary>
          Training on model-generated synthetic data is a promising approach for finetuning LLMs, but it remains unclear when it helps or hurts. In this paper, we investigate this question for math reasoning via an empirical study, followed by building a conceptual understanding of our observations. First, we find that while the typical approach of finetuning a model on synthetic correct or positive problem-solution pairs generated by capable models offers modest performance gains, sampling more correct solutions from the finetuned learner itself followed by subsequent fine-tuning on this self-generated data $\textbf{doubles}$ the efficiency of the same synthetic problems. At the same time, training on model-generated positives can amplify various spurious correlations, resulting in flat or even inverse scaling trends as the amount of data increases. Surprisingly, we find that several of these issues can be addressed if we also utilize negative responses, i.e., model-generated responses that are deemed incorrect by a final answer verifier. Crucially, these negatives must be constructed such that the training can appropriately recover the utility or advantage of each intermediate step in the negative response. With this per-step scheme, we are able to attain consistent gains over only positive data, attaining performance similar to amplifying the amount of synthetic data by $\mathbf{8 \times}$. We show that training on per-step negatives can help to unlearn spurious correlations in the positive data, and is equivalent to advantage-weighted reinforcement learning (RL), implying that it inherits robustness benefits of RL over imitating positive data alone.
     </details>

166. **Can LLMs Reason in the Wild with Programs?** [[pdf]](http://arxiv.org/abs/2406.13764) `2024-06-19` (4 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces the task of reasoning in the wild, where an LLM is tasked to solve a reasoning problem of unknown type by identifying the subproblems and their corresponding formalisms, and writing a program to solve each subproblem, guided by a tactic.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have shown superior capability to solve reasoning problems with programs. While being a promising direction, most of such frameworks are trained and evaluated in settings with a prior knowledge of task requirements. However, as LLMs become more capable, it is necessary to assess their reasoning abilities in more realistic scenarios where many real-world problems are open-ended with ambiguous scope, and often require multiple formalisms to solve. To investigate this, we introduce the task of reasoning in the wild, where an LLM is tasked to solve a reasoning problem of unknown type by identifying the subproblems and their corresponding formalisms, and writing a program to solve each subproblem, guided by a tactic. We create a large tactic-guided trajectory dataset containing detailed solutions to a diverse set of reasoning problems, ranging from well-defined single-form reasoning (e.g., math, logic), to ambiguous and hybrid ones (e.g., commonsense, combined math and logic). This allows us to test various aspects of LLMs reasoning at the fine-grained level such as the selection and execution of tactics, and the tendency to take undesired shortcuts. In experiments, we highlight that existing LLMs fail significantly on problems with ambiguous and mixed scope, revealing critical limitations and overfitting issues (e.g. accuracy on GSM8K drops by at least 50\%). We further show the potential of finetuning a local LLM on the tactic-guided trajectories in achieving better performance. Project repo is available at github.com/gblackout/Reason-in-the-Wild
     </details>

167. **Mathador-LM: A Dynamic Benchmark for Mathematical Reasoning on Large Language Models** [[pdf]](http://arxiv.org/abs/2406.12572) `2024-06-19` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is shown that, across leading LLMs, it is possible to obtain stable average performance while generating benchmark instances dynamically, following a target difficulty level, which alleviates concerns about test-set leakage into training data, an issue that often undermines popular benchmarks.
     </details>


     <details>
          <summary>Abstract</summary>
          We introduce Mathador-LM, a new benchmark for evaluating the mathematical reasoning on large language models (LLMs), combining ruleset interpretation, planning, and problem-solving. This benchmark is inspired by the Mathador game, where the objective is to reach a target number using basic arithmetic operations on a given set of base numbers, following a simple set of rules. We show that, across leading LLMs, we obtain stable average performance while generating benchmark instances dynamically, following a target difficulty level. Thus, our benchmark alleviates concerns about test-set leakage into training data, an issue that often undermines popular benchmarks. Additionally, we conduct a comprehensive evaluation of both open and closed-source state-of-the-art LLMs on Mathador-LM. Our findings reveal that contemporary models struggle with Mathador-LM, scoring significantly lower than average 3rd graders. This stands in stark contrast to their strong performance on popular mathematical reasoning benchmarks.
     </details>

168. **GeoGPT4V: Towards Geometric Multi-modal Large Language Models with Geometric Image Generation** [[pdf]](https://arxiv.org/abs/2406.11503v1) `2024-06-17` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel pipeline is introduced that leverages GPT-4 and GPT-4V to generate relatively basic geometry problems with aligned text and images, facilitating model learning and demonstrates that the GeoGPT4V dataset significantly improves the geometry performance of various models on the MathVista and MathVision benchmarks.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models have seen widespread adoption in math problem-solving. However, in geometry problems that usually require visual aids for better understanding, even the most advanced multi-modal models currently still face challenges in effectively using image information. High-quality data is crucial for enhancing the geometric capabilities of multi-modal models, yet existing open-source datasets and related efforts are either too challenging for direct model learning or suffer from misalignment between text and images. To overcome this issue, we introduce a novel pipeline that leverages GPT-4 and GPT-4V to generate relatively basic geometry problems with aligned text and images, facilitating model learning. We have produced a dataset of 4.9K geometry problems and combined it with 19K open-source data to form our GeoGPT4V dataset. Experimental results demonstrate that the GeoGPT4V dataset significantly improves the geometry performance of various models on the MathVista and MathVision benchmarks. The code is available at https://github.com/Lanyu0303/GeoGPT4V_Project
     </details>

169. **On the Empirical Complexity of Reasoning and Planning in LLMs** [[pdf]](http://arxiv.org/abs/2404.11041) `2024-06-17` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Experimental case studies suggest that both CoT and ToT benefit significantly from task decomposition, and for computationally hard reasoning tasks, the more sophisticated tree structure of ToT outperforms the linear structure of CoT.
     </details>


     <details>
          <summary>Abstract</summary>
          Chain-of-thought (CoT), tree-of-thought (ToT), and related techniques work surprisingly well in practice for some complex reasoning tasks with Large Language Models (LLMs), but why? This work seeks the underlying reasons by conducting experimental case studies and linking the performance benefits to well-established sample and computational complexity principles in machine learning. We experimented with 6 reasoning tasks, ranging from grade school math, air travel planning, ..., to Blocksworld. The results suggest that (i) both CoT and ToT benefit significantly from task decomposition, which breaks a complex reasoning task into a sequence of steps with low sample complexity and explicitly outlines the reasoning structure, and (ii) for computationally hard reasoning tasks, the more sophisticated tree structure of ToT outperforms the linear structure of CoT. These findings provide useful guidelines for the use of LLM in solving reasoning tasks in practice.
     </details>

170. **Probabilistic Reasoning in Generative Large Language Models** [[pdf]](http://arxiv.org/abs/2402.09614) `2024-06-17` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduces the Bayesian Linguistic Inference Dataset (BLInD), a new dataset specifically designed to test the probabilistic reasoning capabilities of LLMs, and presents several prompting strategies that map the problem to different formal representations, including Python code, probabilistic algorithms, and probabilistic logical programming.
     </details>


     <details>
          <summary>Abstract</summary>
          This paper considers the challenges Large Language Models (LLMs) face when reasoning over text that includes information involving uncertainty explicitly quantified via probability values. This type of reasoning is relevant to a variety of contexts ranging from everyday conversations to medical decision-making. Despite improvements in the mathematical reasoning capabilities of LLMs, they still exhibit significant difficulties when it comes to probabilistic reasoning. To deal with this problem, we introduce the Bayesian Linguistic Inference Dataset (BLInD), a new dataset specifically designed to test the probabilistic reasoning capabilities of LLMs. We use BLInD to find out the limitations of LLMs for tasks involving probabilistic reasoning. In addition, we present several prompting strategies that map the problem to different formal representations, including Python code, probabilistic algorithms, and probabilistic logical programming. We conclude by providing an evaluation of our methods on BLInD and an adaptation of a causal reasoning question-answering dataset. Our empirical results highlight the effectiveness of our proposed strategies for multiple LLMs.
     </details>

171. **How Do Humans Write Code? Large Models Do It the Same Way Too** [[pdf]](http://arxiv.org/abs/2402.15729) `2024-06-17` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Human-Think Language is proposed, which leverages a suite of strategies that help integrate PoT and CoT, encompassing a new generation paradigm that uses full CoT reasoning to control code generation and shows the most significant improvement in non-mathematical natural language inference task.
     </details>


     <details>
          <summary>Abstract</summary>
          Program-of-Thought (PoT) replaces natural language-based Chain-of-Thought (CoT) as the most popular method in Large Language Models (LLMs) mathematical reasoning tasks by utilizing external tool calls to circumvent computational errors. However, our evaluation of the GPT-4 and Llama series reveals that using PoT introduces more reasoning errors, such as incorrect formulas or flawed logic, compared to CoT. To address this issue, we propose Human-Think Language (HTL), which leverages a suite of strategies that help integrate PoT and CoT, encompassing: (1) a new generation paradigm that uses full CoT reasoning to control code generation. (2) Focus Attention, that directs model attention to the CoT reasoning during PoT to generate more logical code. (3) reinforcement learning that utilizes the accuracy of both CoT and PoT responses as rewards to prevent repetitive reasoning steps in LLMs when solving difficult math problems. Our method achieves an average improvement of 6.5% on the Llama-Base model and 4.3% on the Mistral-Base model across 8 mathematical calculation datasets. It also shows significant effectiveness on five out-of-domain datasets by controlling the model's information flow, exhibiting strong transferability. Additionally, HTL shows the most significant improvement in non-mathematical natural language inference task, contributing to a unified reasoning task framework
     </details>

172. **Learn Beyond The Answer: Training Language Models with Reflection for Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2406.12050) `2024-06-17` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          reflective augmentation is proposed, a method that embeds problem reflection into each training instance, thereby fostering a thorough comprehension through reflective reasoning and its complementary nature relative to existing augmentation techniques.
     </details>


     <details>
          <summary>Abstract</summary>
          Supervised fine-tuning enhances the problem-solving abilities of language models across various mathematical reasoning tasks. To maximize such benefits, existing research focuses on broadening the training set with various data augmentation techniques, which is effective for standard single-round question-answering settings. Our work introduces a novel technique aimed at cultivating a deeper understanding of the training problems at hand, enhancing performance not only in standard settings but also in more complex scenarios that require reflective thinking. Specifically, we propose reflective augmentation, a method that embeds problem reflection into each training instance. It trains the model to consider alternative perspectives and engage with abstractions and analogies, thereby fostering a thorough comprehension through reflective reasoning. Extensive experiments validate the achievement of our aim, underscoring the unique advantages of our method and its complementary nature relative to existing augmentation techniques.
     </details>

173. **Mathematical Entities: Corpora and Benchmarks** [[pdf]](http://arxiv.org/abs/2406.11577) `2024-06-17` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Mathematics is a highly specialized domain with its own unique set of challenges. Despite this, there has been relatively little research on natural language processing for mathematical texts, and there are few mathematical language resources aimed at NLP. In this paper, we aim to provide annotated corpora that can be used to study the language of mathematics in different contexts, ranging from fundamental concepts found in textbooks to advanced research mathematics. We preprocess the corpora with a neural parsing model and some manual intervention to provide part-of-speech tags, lemmas, and dependency trees. In total, we provide 182397 sentences across three corpora. We then aim to test and evaluate several noteworthy natural language processing models using these corpora, to show how well they can adapt to the domain of mathematics and provide useful tools for exploring mathematical language. We evaluate several neural and symbolic models against benchmarks that we extract from the corpus metadata to show that terminology extraction and definition extraction do not easily generalize to mathematics, and that additional work is needed to achieve good performance on these metrics. Finally, we provide a learning assistant that grants access to the content of these corpora in a context-sensitive manner, utilizing text search and entity linking. Though our corpora and benchmarks provide useful metrics for evaluating mathematical language processing, further work is necessary to adapt models to mathematics in order to provide more effective learning assistants and apply NLP methods to different mathematical domains.
     </details>

174. **Step-level Value Preference Optimization for Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2406.10858) `2024-06-16` (4 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces a novel algorithm called Step-level Value Preference Optimization (SVPO), which employs Monte Carlo Tree Search (MCTS) and an explicit value model to replicate the behavior of the implicit reward model, complementing standard preference optimization.
     </details>


     <details>
          <summary>Abstract</summary>
          Direct Preference Optimization (DPO) using an implicit reward model has proven to be an effective alternative to reinforcement learning from human feedback (RLHF) for fine-tuning preference aligned large language models (LLMs). However, the overall preference annotations of responses do not fully capture the fine-grained quality of model outputs in complex multi-step reasoning tasks, such as mathematical reasoning. To address this limitation, we introduce a novel algorithm called Step-level Value Preference Optimization (SVPO). Our approach employs Monte Carlo Tree Search (MCTS) to automatically annotate step-level preferences for multi-step reasoning. Furthermore, from the perspective of learning-to-rank, we train an explicit value model to replicate the behavior of the implicit reward model, complementing standard preference optimization. This value model enables the LLM to generate higher reward responses with minimal cost during inference. Experimental results demonstrate that our method achieves state-of-the-art performance on both in-domain and out-of-domain mathematical reasoning benchmarks. Our code is available at \url{https://github.com/MARIO-Math-Reasoning/Super_MARIO}.
     </details>

175. **Exposing the Achilles' Heel: Evaluating LLMs Ability to Handle Mistakes in Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2406.10834) `2024-06-16` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces a novel dataset MWP-MISTAKE, incorporating MWPs with both correct and incorrect reasoning steps generated through rule-based methods and smaller language models, and highlights GPT-$o's superior performance in mistake detection and rectification and the persistent challenges faced by smaller models.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have been applied to Math Word Problems (MWPs) with transformative impacts, revolutionizing how these complex problems are approached and solved in various domains including educational settings. However, the evaluation of these models often prioritizes final accuracy, overlooking the crucial aspect of reasoning capabilities. This work addresses this gap by focusing on the ability of LLMs to detect and correct reasoning mistakes. We introduce a novel dataset MWP-MISTAKE, incorporating MWPs with both correct and incorrect reasoning steps generated through rule-based methods and smaller language models. Our comprehensive benchmarking reveals significant insights into the strengths and weaknesses of state-of-the-art models, such as GPT-4o, GPT-4, GPT-3.5Turbo, and others. We highlight GPT-$o's superior performance in mistake detection and rectification and the persistent challenges faced by smaller models. Additionally, we identify issues related to data contamination and memorization, impacting the reliability of LLMs in real-world applications. Our findings emphasize the importance of rigorous evaluation of reasoning processes and propose future directions to enhance the generalization and robustness of LLMs in mathematical problem-solving.
     </details>

176. **RUPBench: Benchmarking Reasoning Under Perturbations for Robustness Evaluation in Large Language Models** [[pdf]](http://arxiv.org/abs/2406.11020) `2024-06-16` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work presents RUPBench, a comprehensive benchmark designed to evaluate LLM robustness across diverse reasoning tasks, and highlights that larger models tend to exhibit greater robustness to perturbations.
     </details>


     <details>
          <summary>Abstract</summary>
          With the increasing use of large language models (LLMs), ensuring reliable performance in diverse, real-world environments is essential. Despite their remarkable achievements, LLMs often struggle with adversarial inputs, significantly impacting their effectiveness in practical applications. To systematically understand the robustness of LLMs, we present RUPBench, a comprehensive benchmark designed to evaluate LLM robustness across diverse reasoning tasks. Our benchmark incorporates 15 reasoning datasets, categorized into commonsense, arithmetic, logical, and knowledge-intensive reasoning, and introduces nine types of textual perturbations at lexical, syntactic, and semantic levels. By examining the performance of state-of-the-art LLMs such as GPT-4o, Llama3, Phi-3, and Gemma on both original and perturbed datasets, we provide a detailed analysis of their robustness and error patterns. Our findings highlight that larger models tend to exhibit greater robustness to perturbations. Additionally, common error types are identified through manual inspection, revealing specific challenges faced by LLMs in different reasoning contexts. This work provides insights into areas where LLMs need further improvement to handle diverse and noisy inputs effectively.
     </details>

177. **Teaching Large Language Models to Reason with Reinforcement Learning** [[pdf]](https://openreview.net/forum?id=mjqoceuMnI) `ICML 2024 Workshop AI for Math` (24 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is concluded that during RL training models fail to explore significantly beyond solutions already produced by SFT models, and the implications of these findings for RLHF and the future role of RL in LLM fine-tuning are discussed.
     </details>


     <details>
          <summary>Abstract</summary>
          Reinforcement Learning from Human Feedback (\textbf{RLHF}) has emerged as a dominant approach for aligning LLM outputs with human preferences. Inspired by the success of RLHF, we study the performance of multiple algorithms that learn from feedback (Expert Iteration, Proximal Policy Optimization (\textbf{PPO}), Return-Conditioned RL) on improving LLM reasoning capabilities. We investigate both sparse and dense rewards provided to the LLM both heuristically and via a learned reward model. We additionally start from multiple model sizes and initializations both with and without supervised fine-tuning (\textbf{SFT}) data. Overall, we find all algorithms perform comparably, with Expert Iteration performing best in most cases. Surprisingly, we find the sample complexity of Expert Iteration is similar to that of PPO, requiring at most on the order of $10^6$ samples to converge from a pretrained checkpoint. We investigate why this is the case, concluding that during RL training models fail to explore significantly beyond solutions already produced by SFT models. Additionally, we discuss a trade off between maj@1 and pass@96 metric performance during SFT training and how conversely RL training improves both simultaneously. We then conclude by discussing the implications of our findings for RLHF and the future role of RL in LLM fine-tuning.
     </details>

178. **Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B** [[pdf]](http://arxiv.org/abs/2406.07394) `2024-06-13` (9 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          This paper introduces the MCT Self-Refine (MCTSr) algorithm, an innovative integration of Large Language Models (LLMs) with Monte Carlo Tree Search (MCTS), designed to enhance performance in complex mathematical reasoning tasks. Addressing the challenges of accuracy and reliability in LLMs, particularly in strategic and mathematical reasoning, MCTSr leverages systematic exploration and heuristic self-refine mechanisms to improve decision-making frameworks within LLMs. The algorithm constructs a Monte Carlo search tree through iterative processes of Selection, self-refine, self-evaluation, and Backpropagation, utilizing an improved Upper Confidence Bound (UCB) formula to optimize the exploration-exploitation balance. Extensive experiments demonstrate MCTSr's efficacy in solving Olympiad-level mathematical problems, significantly improving success rates across multiple datasets, including GSM8K, GSM Hard, MATH, and Olympiad-level benchmarks, including Math Odyssey, AIME, and OlympiadBench. The study advances the application of LLMs in complex reasoning tasks and sets a foundation for future AI integration, enhancing decision-making accuracy and reliability in LLM-driven applications.
     </details>

179. **VerityMath: Advancing Mathematical Reasoning by Self-Verification Through Unit Consistency** [[pdf]](https://openreview.net/forum?id=S9utaRXaZt&name=pdf) `2024-06-13` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper studies the performance of strong open-source LLMs, including Llama 2 (7B), Code Llama (7B), and Mistral (7B) on math word problems using program-based solving techniques, and proposes a systematic approach that incorporates unit consistency programs.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs), combined with program-based solving techniques, are increasingly demonstrating proficiency in mathematical reasoning. For example, closed-source models such as OpenAI GPT-4 and Claude show excellent results in solving math word problems. However, progress in math word problem-solving for open-source LLMs is limited, and the challenges these models face are not well-studied. In this paper, we study the performance of strong open-source LLMs, including Llama 2 (7B), Code Llama (7B), and Mistral (7B) on math word problems using program-based solving techniques. Specifically, we analyze the outputs of these models when applied to math word problems and identify a category of problems that pose a significant challenge, particularly those involving quantities spanning multiple units. To address this issue, we propose a systematic approach by defining the units for each quantity and ensuring the consistency of these units during mathematical operations. We developed Unit Consistency Programs (UCPs), an annotated dataset of math word problems, each paired with programs containing unit specifications and unit verification routines. We fine-tuned Llama 2 (7B), Code Llama (7B), and Mistral (7B) models with UCPs to produce theirVerityMath variants. Our findings indicate that our approach, which incorporates unit consistency, currently slightly underperforms compared to an approach that does not. To understand the reasons behind this, we conduct an in-depth error analysis and suggest options for future improvements. Our code and dataset are available at https://github.com/vernontoh/VerityMath.
     </details>

180. **Distilling LLMs' Decomposition Abilities into Compact Language Models** [[pdf]](https://openreview.net/forum?id=XM44SZM3VO&name=pdf) `2024-06-13` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This study focuses on distilling the LLMs' decomposition skills into compact models using offline reinforcement learning, leveraging the advancements in the LLM`s capabilities to provide feedback and generate a specialized task-specific dataset for training compact models.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have demonstrated proficiency in their reasoning abilities, yet their large size presents scalability challenges and limits any further customization. In contrast, compact models offer customized training but often fall short in solving complex reasoning tasks. This study focuses on distilling the LLMs' decomposition skills into compact models using offline reinforcement learning. We leverage the advancements in the LLM`s capabilities to provide feedback and generate a specialized task-specific dataset for training compact models. The development of an AI-generated dataset and the establishment of baselines constitute the primary contributions of our work, underscoring the potential of compact models in replicating complex problem-solving skills.
     </details>

181. **Progress or Regress? Self-Improvement Reversal in Post-training** [[pdf]](https://openreview.net/forum?id=MG18DR2dAN) `2024-06-13` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Findings indicate that current self-improvement practices through post-training are inadequate for equipping models to tackle more complex problems and underscore the necessity of critical evaluation metrics in discerning the progress or regress dichotomy for self-improving LLMs.
     </details>


     <details>
          <summary>Abstract</summary>
          Self-improvement through post-training methods such as iterative preference learning has been acclaimed for enhancing the problem-solving capabilities (e.g., mathematical reasoning) of Large Language Models (LLMs) without human intervention. However, as exploration deepens, it becomes crucial to assess whether these improvements genuinely signify progress in solving more challenging problems or if they could lead to unintended regressions. To address this, we propose a comprehensive evaluative framework that goes beyond the superficial pass@1 metric to scrutinize the underlying enhancements of post-training paradigms for self-improvement. Through rigorous experimentation and analysis across diverse problem-solving tasks, the empirical results point out the phenomenon of \emph{self-improvement reversal}, where models showing improved performance across benchmarks will paradoxically exhibit declines in broader, essential capabilities, like output diversity and out-of-distribution (OOD) generalization. These findings indicate that current self-improvement practices through post-training are inadequate for equipping models to tackle more complex problems. Furthermore, they underscore the necessity of our critical evaluation metrics in discerning the \emph{progress or regress} dichotomy for self-improving LLMs.
     </details>

182. **Pre-Calc: Learning to Use the Calculator Improves Numeracy in Language Models** [[pdf]](https://openreview.net/forum?id=Hb5gA02FyR&name=pdf) `2024-06-13` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper proposes Pre-Calc, a simple pre-finetuning objective of learning to use the calculator for both encoder-only and encoder-decoder architectures, formulated as a discriminative and generative task respectively.
     </details>


     <details>
          <summary>Abstract</summary>
          Quantitative and numerical comprehension in language is an important task in many fields like education and finance, but still remains a challenging task for language models. While tool and calculator usage has shown to be helpful to improve mathematical reasoning in large pretrained decoder-only language models, this remains unexplored for smaller language models with encoders. In this paper, we propose Pre-Calc, a simple pre-finetuning objective of learning to use the calculator for both encoder-only and encoder-decoder architectures, formulated as a discriminative and generative task respectively. We pre-train BERT and RoBERTa for discriminative calculator use and Flan-T5 for generative calculator use on the MAWPS, SVAMP, and AsDiv-A datasets, which improves performance on downstream tasks that require numerical understanding. Our code and data are available at https://github.com/calc-cmu/pre-calc.
     </details>

183. **Smart Vision-Language Reasoners** [[pdf]](https://openreview.net/forum?id=Mf6ot5U7ni&name=pdf) `2024-06-13` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This article employs the abstractions given in the SMART task (Simple Multimodal Algorithmic Reasoning Task) introduced incherian2022deep as meta-reasoning and problem-solving skills along eight axes: math, counting, path, measure, logic, spatial, and pattern.
     </details>


     <details>
          <summary>Abstract</summary>
          In this article, we investigate vision-language models (VLM) as reasoners. The ability to form abstractions underlies mathematical reasoning, problem-solving, and other Math AI tasks. Several formalisms have been given to these underlying abstractions and skills utilized by humans and intelligent systems for reasoning. Furthermore, human reasoning is inherently multimodal, and as such, we focus our investigations on multimodal AI. In this article, we employ the abstractions given in the SMART task (Simple Multimodal Algorithmic Reasoning Task) introduced in \cite{cherian2022deep} as meta-reasoning and problem-solving skills along eight axes: math, counting, path, measure, logic, spatial, and pattern. We investigate the ability of vision-language models to reason along these axes and seek avenues of improvement. Including composite representations with vision-language cross-attention enabled learning multimodal representations adaptively from fused frozen pretrained backbones for better visual grounding. Furthermore, proper hyperparameter and other training choices led to strong improvements (up to $48\%$ gain in accuracy) on the SMART task, further underscoring the power of deep multimodal learning. The smartest VLM, which includes a novel QF multimodal layer, improves upon the best previous baselines in every one of the eight fundamental reasoning skills. End-to-end code is available at https://github.com/smarter-vlm/smarter.
     </details>

184. **Improving Autoformalization using Type Checking** [[pdf]](http://arxiv.org/abs/2406.07222) `2024-06-11` `Lean` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a method to fix the performance of large language models for autoformalization through decoding with type-check filtering, where they initially sample a diverse set of candidate formalizations for an informal statement, then use the Lean proof assistant to filter out candidates that do not type-check.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models show promise for autoformalization, the task of automatically translating natural language into formal languages. However, current autoformalization methods remain limited. The last reported state-of-the-art performance on the ProofNet formalization benchmark for the Lean proof assistant, achieved using Codex for Lean 3, only showed successful formalization of 16.1% of informal statements. Similarly, our evaluation of GPT-4o for Lean 4 only produces successful translations 34.9% of the time. Our analysis shows that the performance of these models is largely limited by their inability to generate formal statements that successfully type-check (i.e., are syntactically correct and consistent with types) - with a whopping 86.6% of GPT-4o errors starting from a type-check failure. In this work, we propose a method to fix this issue through decoding with type-check filtering, where we initially sample a diverse set of candidate formalizations for an informal statement, then use the Lean proof assistant to filter out candidates that do not type-check. Using GPT-4o as a base model, and combining our method with self-consistency, we obtain a +18.3% absolute increase in formalization accuracy, and achieve a new state-of-the-art of 53.2% on ProofNet with Lean 4.
     </details>

185. **Autograding Mathematical Induction Proofs with Natural Language Processing** [[pdf]](http://arxiv.org/abs/2406.10268) `2024-06-11` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper presents a set of training methods and models capable of autograding freeform mathematical proofs by leveraging existing large language models and other machine learning techniques and finds that the best grading model is also more accurate than most human graders.
     </details>


     <details>
          <summary>Abstract</summary>
          In mathematical proof education, there remains a need for interventions that help students learn to write mathematical proofs. Research has shown that timely feedback can be very helpful to students learning new skills. While for many years natural language processing models have struggled to perform well on tasks related to mathematical texts, recent developments in natural language processing have created the opportunity to complete the task of giving students instant feedback on their mathematical proofs. In this paper, we present a set of training methods and models capable of autograding freeform mathematical proofs by leveraging existing large language models and other machine learning techniques. The models are trained using proof data collected from four different proof by induction problems. We use four different robust large language models to compare their performances, and all achieve satisfactory performances to various degrees. Additionally, we recruit human graders to grade the same proofs as the training data, and find that the best grading model is also more accurate than most human graders. With the development of these grading models, we create and deploy an autograder for proof by induction problems and perform a user study with students. Results from the study shows that students are able to make significant improvements to their proofs using the feedback from the autograder, but students still do not trust the AI autograders as much as they trust human graders. Future work can improve on the autograder feedback and figure out ways to help students trust AI autograders.
     </details>

186. **Husky: A Unified, Open-Source Language Agent for Multi-Step Reasoning** [[pdf]](http://arxiv.org/abs/2406.06469) `2024-06-10` (4 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Husky is introduced, a holistic, open-source language agent that learns to reason over a unified action space to address a diverse set of complex tasks involving numerical, tabular, and knowledge-based reasoning.
     </details>


     <details>
          <summary>Abstract</summary>
          Language agents perform complex tasks by using tools to execute each step precisely. However, most existing agents are based on proprietary models or designed to target specific tasks, such as mathematics or multi-hop question answering. We introduce Husky, a holistic, open-source language agent that learns to reason over a unified action space to address a diverse set of complex tasks involving numerical, tabular, and knowledge-based reasoning. Husky iterates between two stages: 1) generating the next action to take towards solving a given task and 2) executing the action using expert models and updating the current solution state. We identify a thorough ontology of actions for addressing complex tasks and curate high-quality data to train expert models for executing these actions. Our experiments show that Husky outperforms prior language agents across 14 evaluation datasets. Moreover, we introduce HuskyQA, a new evaluation set which stress tests language agents for mixed-tool reasoning, with a focus on retrieving missing knowledge and performing numerical reasoning. Despite using 7B models, Husky matches or even exceeds frontier LMs such as GPT-4 on these tasks, showcasing the efficacy of our holistic approach in addressing complex reasoning problems. Our code and models are available at https://github.com/agent-husky/Husky-v1.
     </details>

187. **LLMs Are Not Intelligent Thinkers: Introducing Mathematical Topic Tree Benchmark for Comprehensive Evaluation of LLMs** [[pdf]](http://arxiv.org/abs/2406.05194) `2024-06-07` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) demonstrate impressive capabilities in mathematical reasoning. However, despite these achievements, current evaluations are mostly limited to specific mathematical topics, and it remains unclear whether LLMs are genuinely engaging in reasoning. To address these gaps, we present the Mathematical Topics Tree (MaTT) benchmark, a challenging and structured benchmark that offers 1,958 questions across a wide array of mathematical subjects, each paired with a detailed hierarchical chain of topics. Upon assessing different LLMs using the MaTT benchmark, we find that the most advanced model, GPT-4, achieved a mere 54\% accuracy in a multiple-choice scenario. Interestingly, even when employing Chain-of-Thought prompting, we observe mostly no notable improvement. Moreover, LLMs accuracy dramatically reduced by up to 24.2 percentage point when the questions were presented without providing choices. Further detailed analysis of the LLMs' performance across a range of topics showed significant discrepancy even for closely related subtopics within the same general mathematical area. In an effort to pinpoint the reasons behind LLMs performances, we conducted a manual evaluation of the completeness and correctness of the explanations generated by GPT-4 when choices were available. Surprisingly, we find that in only 53.3\% of the instances where the model provided a correct answer, the accompanying explanations were deemed complete and accurate, i.e., the model engaged in genuine reasoning.
     </details>

188. **Robustness Assessment of Mathematical Reasoning in the Presence of Missing and Contradictory Conditions** [[pdf]](http://arxiv.org/abs/2406.05055) `2024-06-07` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have demonstrated impressive performance on reasoning tasks, which can be further improved through few-shot prompting techniques. However, the current evaluation primarily focuses on carefully constructed benchmarks and neglects the consideration of real-world reasoning problems that present missing and contradictory conditions, known as ill-defined problems. Our observations suggest that existing few-shot prompting techniques are ineffective in such scenarios, often providing overconfident answers or hallucination. To further study this problem, we develop a benchmark called Problems with Missing and Contradictory conditions (PMC) and introduce two novel metrics to evaluate the performance of few-shot prompting methods in these scenarios. Our analysis using the PMC benchmark reveals a trade-off dilemma between the performance of mathematical reasoning for well-defined problems and the ability to recognize ill-defined problems. To address the challenges posed by PMC, we propose a novel few-shot prompting method called SMT-LIB Prompting (SLP), which utilizes the SMT-LIB language to model the problems instead of solving them directly. Subsequently, a double-check solving strategy checks the satisfiability and uniqueness of the solution and provides final feedback. Extensive experiments demonstrate the superiority of our SLP approach compared to existing few-shot prompting methods when dealing with problems with missing and contradictory conditions. We will open-source our benchmark and code to facilitate future research.
     </details>

189. **Improve Mathematical Reasoning in Language Models by Automated Process Supervision** [[pdf]](http://arxiv.org/abs/2406.06592) `2024-06-05` (16 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel divide-and-conquer style Monte Carlo Tree Search (MCTS) algorithm named OmegaPRM is proposed, which swiftly identifies the first error in the Chain of Thought with binary search and balances the positive and negative examples, thereby ensuring both efficiency and quality.
     </details>


     <details>
          <summary>Abstract</summary>
          Complex multi-step reasoning tasks, such as solving mathematical problems or generating code, remain a significant hurdle for even the most advanced large language models (LLMs). Verifying LLM outputs with an Outcome Reward Model (ORM) is a standard inference-time technique aimed at enhancing the reasoning performance of LLMs. However, this still proves insufficient for reasoning tasks with a lengthy or multi-hop reasoning chain, where the intermediate outcomes are neither properly rewarded nor penalized. Process supervision addresses this limitation by assigning intermediate rewards during the reasoning process. To date, the methods used to collect process supervision data have relied on either human annotation or per-step Monte Carlo estimation, both prohibitively expensive to scale, thus hindering the broad application of this technique. In response to this challenge, we propose a novel divide-and-conquer style Monte Carlo Tree Search (MCTS) algorithm named \textit{OmegaPRM} for the efficient collection of high-quality process supervision data. This algorithm swiftly identifies the first error in the Chain of Thought (CoT) with binary search and balances the positive and negative examples, thereby ensuring both efficiency and quality. As a result, we are able to collect over 1.5 million process supervision annotations to train a Process Reward Model (PRM). Utilizing this fully automated process supervision alongside the weighted self-consistency algorithm, we have enhanced the instruction tuned Gemini Pro model's math reasoning performance, achieving a 69.4\% success rate on the MATH benchmark, a 36\% relative improvement from the 51\% base model performance. Additionally, the entire process operates without any human intervention, making our method both financially and computationally cost-effective compared to existing methods.
     </details>

190. **Assessing the Emergent Symbolic Reasoning Abilities of Llama Large Language Models** [[pdf]](http://arxiv.org/abs/2406.06588) `2024-06-05` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is observed that both increasing the scale of the model and fine-tuning it on relevant tasks lead to significant performance gains, which are mostly observed with mathematical formulas of low complexity, which nevertheless often remain challenging even for the largest fine-tuned models.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) achieve impressive performance in a wide range of tasks, even if they are often trained with the only objective of chatting fluently with users. Among other skills, LLMs show emergent abilities in mathematical reasoning benchmarks, which can be elicited with appropriate prompting methods. In this work, we systematically investigate the capabilities and limitations of popular open-source LLMs on different symbolic reasoning tasks. We evaluate three models of the Llama 2 family on two datasets that require solving mathematical formulas of varying degrees of difficulty. We test a generalist LLM (Llama 2 Chat) as well as two fine-tuned versions of Llama 2 (MAmmoTH and MetaMath) specifically designed to tackle mathematical problems. We observe that both increasing the scale of the model and fine-tuning it on relevant tasks lead to significant performance gains. Furthermore, using fine-grained evaluation measures, we find that such performance gains are mostly observed with mathematical formulas of low complexity, which nevertheless often remain challenging even for the largest fine-tuned models.
     </details>

191. **Forward-Backward Reasoning in Large Language Models for Mathematical Verification** [[pdf]](http://arxiv.org/abs/2308.07758) `ACL 2024 Findings` (9 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes FOBAR to combine FOrward and BAckward Reasoning for verification for verification, and shows that FOBAR outperforms Self-Consistency, which uses forward reasoning alone, demonstrating that combining forward and forward reasoning is better.
     </details>


     <details>
          <summary>Abstract</summary>
          Self-Consistency samples diverse reasoning chains with answers and chooses the final answer by majority voting. It is based on forward reasoning and cannot further improve performance by sampling more reasoning chains when saturated. To further boost performance, we introduce backward reasoning to verify candidate answers. Specifically, for mathematical tasks, we mask a number in the question and ask the LLM to answer a backward question created by a simple template, i.e., to predict the masked number when a candidate answer is provided. Instead of using forward or backward reasoning alone, we propose **FOBAR** to combine **FO**rward and **BA**ckward **R**easoning for verification. Extensive experiments on six standard mathematical data sets and three LLMs show that FOBAR achieves state-of-the-art performance. In particular, FOBAR outperforms Self-Consistency, which uses forward reasoning alone, demonstrating that combining forward and backward reasoning is more accurate in verification. In addition, FOBAR achieves higher accuracy than existing verification methods, showing the effectiveness of the simple template used in backward reasoning and the proposed combination.
     </details>

192. **Break the Chain: Large Language Models Can be Shortcut Reasoners** [[pdf]](http://arxiv.org/abs/2406.06580) `2024-06-04` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A critical evaluation of CoT prompting is conducted, extending beyond arithmetic to include complex logical and commonsense reasoning tasks, areas where standard CoT methods fall short.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent advancements in Chain-of-Thought (CoT) reasoning utilize complex modules but are hampered by high token consumption, limited applicability, and challenges in reproducibility. This paper conducts a critical evaluation of CoT prompting, extending beyond arithmetic to include complex logical and commonsense reasoning tasks, areas where standard CoT methods fall short. We propose the integration of human-like heuristics and shortcuts into language models (LMs) through "break the chain" strategies. These strategies disrupt traditional CoT processes using controlled variables to assess their efficacy. Additionally, we develop innovative zero-shot prompting strategies that encourage the use of shortcuts, enabling LMs to quickly exploit reasoning clues and bypass detailed procedural steps. Our comprehensive experiments across various LMs, both commercial and open-source, reveal that LMs maintain effective performance with "break the chain" strategies. We also introduce ShortcutQA, a dataset specifically designed to evaluate reasoning through shortcuts, compiled from competitive tests optimized for heuristic reasoning tasks such as forward/backward reasoning and simplification. Our analysis confirms that ShortcutQA not only poses a robust challenge to LMs but also serves as an essential benchmark for enhancing reasoning efficiency in AI.
     </details>

193. **Exploring Mathematical Extrapolation of Large Language Models with Synthetic Data** [[pdf]](http://arxiv.org/abs/2406.02100) `ACL 2024 Findings` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper has designed two out-of-domain datasets in the form of extending the numerical range and the composing components of the arithmetical puzzle problem separately to show that the open-llama-3B model can perform well on multi-step reasoning tasks via fine-tuning on high-quality synthetic data.
     </details>


     <details>
          <summary>Abstract</summary>
          While large language models (LLMs) have shown excellent capabilities in language understanding, text generation and many other tasks, they still struggle in complex multi-step reasoning problems such as mathematical reasoning. In this paper, through a newly proposed arithmetical puzzle problem, we show that the model can perform well on multi-step reasoning tasks via fine tuning on high-quality synthetic data. Experiments with the open-llama-3B model on three different test datasets show that not only the model can reach a zero-shot pass@1 at 0.44 on the in-domain dataset, it also demonstrates certain generalization capabilities on the out-of-domain datasets. Specifically, this paper has designed two out-of-domain datasets in the form of extending the numerical range and the composing components of the arithmetical puzzle problem separately. The fine-tuned model have shown encouraging performance on these two far more difficult tasks with the zero-shot pass@1 at 0.33 and 0.35 correspondingly.
     </details>

194. **Explicitly Encoding Structural Symmetry is Key to Length Generalization in Arithmetic Tasks** [[pdf]](http://arxiv.org/abs/2406.01895) `2024-06-03` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is proved that explicit incorporation of structure via positional encodings is necessary for out-of-distribution generalization, and pinpoint other challenges inherent to length generalization beyond capturing symmetries, in particular complexity of the underlying task.
     </details>


     <details>
          <summary>Abstract</summary>
          Despite the success of Transformers on language understanding, code generation, and logical reasoning, they still fail to generalize over length on basic arithmetic tasks such as addition and multiplication. A major reason behind this failure is the vast difference in structure between numbers and text; For example, the numbers are typically parsed from right to left, and there is a correspondence between digits at the same position across different numbers. In contrast, for text, such symmetries are quite unnatural. In this work, we propose to encode these semantics explicitly into the model via modified number formatting and custom positional encodings. Empirically, our method allows a Transformer trained on numbers with at most 5-digits for addition and multiplication to generalize up to 50-digit numbers, without using additional data for longer sequences. We further demonstrate that traditional absolute positional encodings (APE) fail to generalize to longer sequences, even when trained with augmented data that captures task symmetries. To elucidate the importance of explicitly encoding structure, we prove that explicit incorporation of structure via positional encodings is necessary for out-of-distribution generalization. Finally, we pinpoint other challenges inherent to length generalization beyond capturing symmetries, in particular complexity of the underlying task, and propose changes in the training distribution to address them.
     </details>

195. **Evaluating Mathematical Reasoning of Large Language Models: A Focus on Error Identification and Correction** [[pdf]](http://arxiv.org/abs/2406.00755) `ACL 2024 Findings` (5 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The principal findings indicate that GPT-4 outperforms all models, while open-source model LLaMA-2-7B demonstrates comparable abilities to closed-source models GPT-3.5 and Gemini Pro.
     </details>


     <details>
          <summary>Abstract</summary>
          The rapid advancement of Large Language Models (LLMs) in the realm of mathematical reasoning necessitates comprehensive evaluations to gauge progress and inspire future directions. Existing assessments predominantly focus on problem-solving from the examinee perspective, overlooking a dual perspective of examiner regarding error identification and correction.From the examiner perspective, we define four evaluation tasks for error identification and correction along with a new dataset with annotated error types and steps. We also design diverse prompts to thoroughly evaluate eleven representative LLMs. Our principal findings indicate that GPT-4 outperforms all models, while open-source model LLaMA-2-7B demonstrates comparable abilities to closed-source models GPT-3.5 and Gemini Pro.Notably, calculation error proves the most challenging error type. Moreover, prompting LLMs with the error types can improve the average correction accuracy by 47.9%. These results reveal potential directions for developing the mathematical reasoning abilities of LLMs.Our code and dataset is available on https://github.com/LittleCirc1e/EIC.
     </details>

196. **What Makes Math Word Problems Challenging for LLMs?** [[pdf]](https://aclanthology.org/2024.findings-naacl.72) `NAACL 2024 Findings` (4 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          This paper investigates the question of what makes math word problems (MWPs) in English challenging for large language models (LLMs). We conduct an in-depth analysis of the key linguistic and mathematical characteristics of MWPs. In addition, we train feature-based classifiers to better understand the impact of each feature on the overall difficulty of MWPs for prominent LLMs and investigate whether this helps predict how well LLMs fare against specific categories of MWPs.
     </details>

197. **Arithmetic Reasoning with LLM: Prolog Generation & Permutation** [[pdf]](https://aclanthology.org/2024.naacl-short.61) `NAACL 2024 Short Papers` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work investigates using large language models to generate Prolog programs to solve mathematical questions and proposes to permute the ground truth predicates for more robust LLM training via data augmentation.
     </details>


     <details>
          <summary>Abstract</summary>
          Instructing large language models (LLMs) to solve elementary school math problems has shown great success using Chain of Thought (CoT). However, the CoT approach relies on an LLM to generate a sequence of arithmetic calculations which can be prone to cascaded calculation errors. We hypothesize that an LLM should focus on extracting predicates and generating symbolic formulas from the math problem description so that the underlying calculation can be done via an external code interpreter. We investigate using LLM to generate Prolog programs to solve mathematical questions. Experimental results show that our Prolog-based arithmetic problem-solving outperforms CoT generation in the GSM8K benchmark across three distinct LLMs. In addition, given the insensitive ordering of predicates and symbolic formulas in Prolog, we propose to permute the ground truth predicates for more robust LLM training via data augmentation.
     </details>

198. **Laying Anchors: Semantically Priming Numerals in Language Modeling** [[pdf]](https://aclanthology.org/2024.findings-naacl.169) `NAACL 2024 Findings` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces strategies to semantically prime numerals in any corpus by generating anchors governed by the distribution of numerals in said corpus, thereby enabling mathematically grounded representations of these numeral tokens in any corpus.
     </details>


     <details>
          <summary>Abstract</summary>
          Off-the-shelf pre-trained language models have become the de facto standard in NLP pipelines for a multitude of downstream tasks. However, the inability of these models to properly encode numerals limits their performance on tasks requiring numeric comprehension. We introduce strategies to semantically prime numerals in any corpus by generating anchors governed by the distribution of numerals in said corpus, thereby enabling mathematically grounded representations of these numeral tokens. We establish the superiority of our proposed techniques through evaluation on a range of numeracy tasks for both in-domain (seen) and out-domain (unseen) numerals. Further, we expand our empirical evaluations to numerals ranging from 1 to 10 billion, a significantly broader range compared to previous studies of the same nature, and we demonstrate significant improvements in the mathematical grounding of our learned embeddings.
     </details>

199. **An Evaluation Benchmark for Autoformalization in Lean4** [[pdf]](http://arxiv.org/abs/2406.06555) `2024-06-01` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel evaluation benchmark designed for Lean4 is introduced, applying it to test the abilities of state-of-the-art LLMs, including GPT-3.5, GPT-4, and Gemini Pro, revealing that these LLMs still exhibit limitations in autoformalization, particularly in more complex areas of mathematics.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) hold the potential to revolutionize autoformalization. The introduction of Lean4, a mathematical programming language, presents an unprecedented opportunity to rigorously assess the autoformalization capabilities of LLMs. This paper introduces a novel evaluation benchmark designed for Lean4, applying it to test the abilities of state-of-the-art LLMs, including GPT-3.5, GPT-4, and Gemini Pro. Our comprehensive analysis reveals that, despite recent advancements, these LLMs still exhibit limitations in autoformalization, particularly in more complex areas of mathematics. These findings underscore the need for further development in LLMs to fully harness their potential in scientific research and development. This study not only benchmarks current LLM capabilities but also sets the stage for future enhancements in autoformalization.
     </details>

200. **GOLD: Geometry Problem Solver with Natural Language Description** [[pdf]](https://aclanthology.org/2024.findings-naacl.19) `NAACL 2024 Findings` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          GOLD enhances the extraction of geometric relations by separately processing symbols and geometric primitives within the diagram, and converts the extracted relations into natural language descriptions, efficiently utilizing large language models to solve geometry math problems.
     </details>


     <details>
          <summary>Abstract</summary>
          Addressing the challenge of automated geometry math problem-solving in artificial intelligence (AI) involves understanding multi-modal information and mathematics. blackCurrent methods struggle with accurately interpreting geometry diagrams, which hinders effective problem-solving. To tackle this issue, we present the Geometry problem sOlver with natural Language Description (GOLD) model. GOLD enhances the extraction of geometric relations by separately processing symbols and geometric primitives within the diagram. Subsequently, it converts the extracted relations into natural language descriptions, efficiently utilizing large language models to solve geometry math problems. Experiments show that the GOLD model outperforms the Geoformer model, the previous best method on the UniGeo dataset, by achieving accuracy improvements of 12.7% and 42.1% in calculation and proving subsets. Additionally, it surpasses the former best model on the PGPS9K and Geometry3K datasets, PGPSNet, by obtaining accuracy enhancements of 1.8% and 3.2%, respectively.
     </details>

201. **Self-Demos: Eliciting Out-of-Demonstration Generalizability in Large Language Models** [[pdf]](https://aclanthology.org/2024.findings-naacl.243) `NAACL 2024 Findings` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes Self-Demos, a novel prompting method that elicits the inherent generalizability in LLMs by query-aware demo generation that can outperform state-of-the-art baselines in the OOD setting.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have shown promising abilities of in-context learning (ICL), adapting swiftly to new tasks with only few-shot demonstrations. However, current few-shot methods heavily depend on high-quality, query-specific demos, which are often lacking. When faced with out-of-demonstration (OOD) queries, methods that rely on hand-crafted demos or external retrievers might fail. To bridge the gap between limited demos and OOD queries, we propose Self-Demos, a novel prompting method that elicits the inherent generalizability in LLMs by query-aware demo generation. The generated demos strategically interpolate between existing demos and the given query, transforming the query from OOD to ID. To evaluate the effectiveness of our approach, we manually constructed OOD-Toolset, a dataset in the tool-using scenario with over 300 real-world APIs and 1000 instances, each consisting of three tool-use cases as demos and an OOD query. Thorough experiments on our dataset and two public math benchmarks have shown that our method can outperform state-of-the-art baselines in the OOD setting. Moreover, we conduct a range of analyses to validate Self-Demos’s generalization and provide more insights.
     </details>

202. **Beyond Imitation: Learning Key Reasoning Steps from Dual Chain-of-Thoughts in Reasoning Distillation** [[pdf]](http://arxiv.org/abs/2405.19737) `2024-05-30` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel method that further aids SLMs learning key reasoning steps rather than mere simple fine-tuning is proposed, and it is found that EDIT benefits more from logical errors than from knowledge or mathematical calculation errors in dual CoTs.
     </details>


     <details>
          <summary>Abstract</summary>
          As Large Language Models (LLMs) scale up and gain powerful Chain-of-Thoughts (CoTs) reasoning abilities, practical resource constraints drive efforts to distill these capabilities into more compact Smaller Language Models (SLMs). We find that CoTs consist mainly of simple reasoning forms, with a small proportion ($\approx 4.7\%$) of key reasoning steps that truly impact conclusions. However, previous distillation methods typically involve supervised fine-tuning student SLMs only on correct CoTs data produced by teacher LLMs, resulting in students struggling to learn the key reasoning steps, instead imitating the teacher's reasoning forms and making errors or omissions on these steps. To address these issues, drawing an analogy to human learning, where analyzing mistakes according to correct solutions often reveals the crucial steps leading to successes or failures, we propose mistak\textbf{E}-\textbf{D}riven key reason\textbf{I}ng step distilla\textbf{T}ion (\textbf{EDIT}), a novel method that further aids SLMs learning key reasoning steps rather than mere simple fine-tuning. Firstly, to expose these crucial steps in CoTs, we design specific prompts to generate dual CoTs data with similar reasoning paths but divergent conclusions. Then, we apply the minimum edit distance algorithm on the dual CoTs data to locate these key steps and optimize the likelihood of these steps. Extensive experiments validate the effectiveness of EDIT across both in-domain and out-of-domain benchmark reasoning datasets. Further analysis shows that EDIT can generate high-quality CoTs with more correct key reasoning steps. Notably, we also explore how different mistake patterns affect performance and find that EDIT benefits more from logical errors than from knowledge or mathematical calculation errors in dual CoTs\footnote{Code can be found at \url{https://github.com/C-W-D/EDIT}}.
     </details>

203. **Investigating the Robustness of LLMs on Math Word Problems** [[pdf]](http://arxiv.org/abs/2406.15444) `2024-05-30` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A prompting framework that generates adversarial variants of MWPs by adding irrelevant variables is proposed, and fine-tuning on adversarial training instances improves performance on adversarial MWPs by ~8%, indicating increased robustness to noise and better ability to identify relevant data for reasoning.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) excel at various tasks, including solving math word problems (MWPs), but struggle with real-world problems containing irrelevant information. To address this, we propose a prompting framework that generates adversarial variants of MWPs by adding irrelevant variables. We introduce a dataset, ProbleMATHIC, containing both adversarial and non-adversarial MWPs. Our experiments reveal that LLMs are susceptible to distraction by numerical noise, resulting in an average relative performance drop of ~26% on adversarial MWPs. To mitigate this, we fine-tune LLMs (Llama-2, Mistral) on the adversarial samples from our dataset. Fine-tuning on adversarial training instances improves performance on adversarial MWPs by ~8%, indicating increased robustness to noise and better ability to identify relevant data for reasoning. Finally, to assess the generalizability of our prompting framework, we introduce GSM-8K-Adv, an adversarial variant of the GSM-8K benchmark. LLMs continue to struggle when faced with adversarial information, reducing performance by up to ~6%.
     </details>

204. **MathChat: Benchmarking Mathematical Reasoning and Instruction Following in Multi-Turn Interactions** [[pdf]](http://arxiv.org/abs/2405.19444) `2024-05-29` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduces MathChat, a comprehensive benchmark specifically designed to evaluate LLMs across a broader spectrum of mathematical tasks, and develops MathChat sync, a synthetic dialogue based math dataset for LLM finetuning, focusing on improving models' interaction and instruction following capabilities in conversations.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have demonstrated impressive capabilities in mathematical problem solving, particularly in single turn question answering formats. However, real world scenarios often involve mathematical question answering that requires multi turn or interactive information exchanges, and the performance of LLMs on these tasks is still underexplored. This paper introduces MathChat, a comprehensive benchmark specifically designed to evaluate LLMs across a broader spectrum of mathematical tasks. These tasks are structured to assess the models' abilities in multiturn interactions and open ended generation. We evaluate the performance of various SOTA LLMs on the MathChat benchmark, and we observe that while these models excel in single turn question answering, they significantly underperform in more complex scenarios that require sustained reasoning and dialogue understanding. To address the above limitations of existing LLMs when faced with multiturn and open ended tasks, we develop MathChat sync, a synthetic dialogue based math dataset for LLM finetuning, focusing on improving models' interaction and instruction following capabilities in conversations. Experimental results emphasize the need for training LLMs with diverse, conversational instruction tuning datasets like MathChatsync. We believe this work outlines one promising direction for improving the multiturn mathematical reasoning abilities of LLMs, thus pushing forward the development of LLMs that are more adept at interactive mathematical problem solving and real world applications.
     </details>

205. **Achieving >97% on GSM8K: Deeply Understanding the Problems Makes LLMs Better Solvers for Math Word Problems** [[pdf]](http://arxiv.org/abs/2404.14963) `2024-05-29` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Chain-of-Thought (CoT) prompting has enhanced the performance of Large Language Models (LLMs) across various reasoning tasks. However, CoT still falls short in dealing with complex math word problems, as it usually suffers from three pitfalls: semantic misunderstanding errors, calculation errors and step-missing errors. Prior studies involve addressing the calculation errors and step-missing errors, but neglect the semantic misunderstanding errors, which is the major factor limiting the LLMs' performance. To this end, we propose a simple-yet-effective method, namely Deeply Understanding the Problems (DUP), to improve the LLMs' math problem-solving ability by addressing semantic misunderstanding errors. The core of our method is to encourage the LLMs to deeply understand the problems and extract the key problem-solving information used for better reasoning. Extensive experiments on 10 diverse reasoning benchmarks show that our DUP method consistently outperforms the other counterparts by a large margin. More encouragingly, DUP achieves a new SOTA result on the GSM8K benchmark, with an accuracy of 97.1% under zero-shot setting.
     </details>

206. **Efficient Model-agnostic Alignment via Bayesian Persuasion** [[pdf]](http://arxiv.org/abs/2405.18718) `2024-05-28` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          An efficient method for aligning black-box large models using smaller models using smaller models is explored, introducing a model-agnostic and lightweight Bayesian Persuasion Alignment framework and an upper bound on the Advisor's regret is provided, confirming its effectiveness in learning the optimal signaling strategy.
     </details>


     <details>
          <summary>Abstract</summary>
          With recent advancements in large language models (LLMs), alignment has emerged as an effective technique for keeping LLMs consensus with human intent. Current methods primarily involve direct training through Supervised Fine-tuning (SFT) or Reinforcement Learning from Human Feedback (RLHF), both of which require substantial computational resources and extensive ground truth data. This paper explores an efficient method for aligning black-box large models using smaller models, introducing a model-agnostic and lightweight Bayesian Persuasion Alignment framework. We formalize this problem as an optimization of the signaling strategy from the small model's perspective. In the persuasion process, the small model (Advisor) observes the information item (i.e., state) and persuades large models (Receiver) to elicit improved responses. The Receiver then generates a response based on the input, the signal from the Advisor, and its updated belief about the information item. Through training using our framework, we demonstrate that the Advisor can significantly enhance the performance of various Receivers across a range of tasks. We theoretically analyze our persuasion framework and provide an upper bound on the Advisor's regret, confirming its effectiveness in learning the optimal signaling strategy. Our Empirical results demonstrates that GPT-2 can significantly improve the performance of various models, achieving an average enhancement of 16.1% in mathematical reasoning ability and 13.7% in code generation. We hope our work can provide an initial step toward rethinking the alignment framework from the Bayesian Persuasion perspective.
     </details>

207. **Learning Beyond Pattern Matching? Assaying Mathematical Understanding in LLMs** [[pdf]](http://arxiv.org/abs/2405.15485) `2024-05-24` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper assesses the domain knowledge of LLMs through its understanding of different mathematical skills required to solve problems, and finds evidence of domain understanding during in-context learning.
     </details>


     <details>
          <summary>Abstract</summary>
          We are beginning to see progress in language model assisted scientific discovery. Motivated by the use of LLMs as a general scientific assistant, this paper assesses the domain knowledge of LLMs through its understanding of different mathematical skills required to solve problems. In particular, we look at not just what the pre-trained model already knows, but how it learned to learn from information during in-context learning or instruction-tuning through exploiting the complex knowledge structure within mathematics. Motivated by the Neural Tangent Kernel (NTK), we propose \textit{NTKEval} to assess changes in LLM's probability distribution via training on different kinds of math data. Our systematic analysis finds evidence of domain understanding during in-context learning. By contrast, certain instruction-tuning leads to similar performance changes irrespective of training on different data, suggesting a lack of domain understanding across different skills.
     </details>

208. **Rho-1: Not All Tokens Are What You Need** [[pdf]](http://arxiv.org/abs/2404.07965) `2024-05-23` (24 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Rho-1 is introduced, a new language model that employs Selective Language Modeling (SLM), which selectively trains on useful tokens that aligned with the desired distribution, increasing both efficiency and performance of the language model pre-training.
     </details>


     <details>
          <summary>Abstract</summary>
          Previous language model pre-training methods have uniformly applied a next-token prediction loss to all training tokens. Challenging this norm, we posit that ''Not all tokens in a corpus are equally important for language model training''. Our initial analysis examines token-level training dynamics of language model, revealing distinct loss patterns for different tokens. Leveraging these insights, we introduce a new language model called Rho-1. Unlike traditional LMs that learn to predict every next token in a corpus, Rho-1 employs Selective Language Modeling (SLM), which selectively trains on useful tokens that aligned with the desired distribution. This approach involves scoring pretraining tokens using a reference model, and then training the language model with a focused loss on tokens with higher scores. When continual pretraining on 15B OpenWebMath corpus, Rho-1 yields an absolute improvement in few-shot accuracy of up to 30% in 9 math tasks. After fine-tuning, Rho-1-1B and 7B achieved state-of-the-art results of 40.6% and 51.8% on MATH dataset, respectively - matching DeepSeekMath with only 3% of the pretraining tokens. Furthermore, when pretraining on 80B general tokens, Rho-1 achieves 6.8% average enhancement across 15 diverse tasks, increasing both efficiency and performance of the language model pre-training.
     </details>

209. **DeepSeek-Prover: Advancing Theorem Proving in LLMs through Large-Scale Synthetic Data** [[pdf]](http://arxiv.org/abs/2405.14333) `2024-05-23` `Lean` (10 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces an approach to generate extensive Lean 4 proof data derived from high-school and undergraduate-level mathematical competition problems to enhance theorem-proving capabilities in LLMs and demonstrates the potential of leveraging large-scale synthetic data to enhance theorem-proving capabilities in LLMs.
     </details>


     <details>
          <summary>Abstract</summary>
          Proof assistants like Lean have revolutionized mathematical proof verification, ensuring high accuracy and reliability. Although large language models (LLMs) show promise in mathematical reasoning, their advancement in formal theorem proving is hindered by a lack of training data. To address this issue, we introduce an approach to generate extensive Lean 4 proof data derived from high-school and undergraduate-level mathematical competition problems. This approach involves translating natural language problems into formal statements, filtering out low-quality statements, and generating proofs to create synthetic data. After fine-tuning the DeepSeekMath 7B model on this synthetic dataset, which comprises 8 million formal statements with proofs, our model achieved whole-proof generation accuracies of 46.3% with 64 samples and 52% cumulatively on the Lean 4 miniF2F test, surpassing the baseline GPT-4 at 23.0% with 64 samples and a tree search reinforcement learning method at 41.0%. Additionally, our model successfully proved 5 out of 148 problems in the Lean 4 Formalized International Mathematical Olympiad (FIMO) benchmark, while GPT-4 failed to prove any. These results demonstrate the potential of leveraging large-scale synthetic data to enhance theorem-proving capabilities in LLMs. Both the synthetic dataset and the model will be made available to facilitate further research in this promising field.
     </details>

210. **Can LLMs Solve longer Math Word Problems Better?** [[pdf]](http://arxiv.org/abs/2405.14804) `2024-05-23` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This study pioneers the exploration of Context Length Generalizability (CoLeG), the ability of LLMs to solve long MWPs, and introduces Extended Grade-School Math (E-GSM), a collection of MWPs with lengthy narratives.
     </details>


     <details>
          <summary>Abstract</summary>
          Math Word Problems (MWPs) are crucial for evaluating the capability of Large Language Models (LLMs), with current research primarily focusing on questions with concise contexts. However, as real-world math problems often involve complex circumstances, LLMs' ability to solve long MWPs is vital for their applications in these scenarios, yet remains under-explored. This study pioneers the exploration of Context Length Generalizability (CoLeG), the ability of LLMs to solve long MWPs. We introduce Extended Grade-School Math (E-GSM), a collection of MWPs with lengthy narratives. Two novel metrics are proposed to assess the efficacy and resilience of LLMs in solving these problems. Our examination of existing zero-shot prompting techniques and both proprietary and open-source LLMs reveals a general deficiency in CoLeG. To alleviate these challenges, we propose distinct approaches for different categories of LLMs. For proprietary LLMs, a new instructional prompt is proposed to mitigate the influence of long context. For open-source LLMs, a new data augmentation task is developed to improve CoLeG. Our comprehensive results demonstrate the effectiveness of our proposed methods, showing not only improved performance on E-GSM but also generalizability across several other MWP benchmarks. Our findings pave the way for future research in employing LLMs for complex, real-world applications, offering practical solutions to current limitations and opening avenues for further exploration of model generalizability and training methodologies.
     </details>

211. **Large Language Models are Contrastive Reasoners** [[pdf]](http://arxiv.org/abs/2403.08211) `2024-05-22` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work explores how contrastive prompting (CP) significantly improves the ability of large language models to perform complex reasoning and demonstrates that LLMs are decent contrastive reasoners by simply adding "Let's give a correct and a wrong answer" before LLMs provide answers.
     </details>


     <details>
          <summary>Abstract</summary>
          Prompting methods play a crucial role in enhancing the capabilities of pre-trained large language models (LLMs). We explore how contrastive prompting (CP) significantly improves the ability of large language models to perform complex reasoning. We demonstrate that LLMs are decent contrastive reasoners by simply adding "Let's give a correct and a wrong answer." before LLMs provide answers. Experiments on various large language models show that zero-shot contrastive prompting improves performance on a range of arithmetic, commonsense, and symbolic reasoning tasks without any hand-crafted few-shot examples, such as increasing the accuracy on GSM8K from 35.9% to 88.8% and AQUA-RAT from 41.3% to 62.2% with the state-of-the-art GPT-4 model. Our method not only surpasses zero-shot CoT and few-shot CoT in most arithmetic and commonsense reasoning tasks but also can seamlessly integrate with existing prompting methods, resulting in improved or comparable results when compared to state-of-the-art methods. Our code is available at https://github.com/yao8839836/cp
     </details>

212. **Investigating Symbolic Capabilities of Large Language Models** [[pdf]](http://arxiv.org/abs/2405.13209) `2024-05-21` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Evaluating Large Language Models on a series of symbolic tasks reveals a significant decline in LLMs' performance on context-free and context-sensitive symbolic tasks as the complexity, represented by the number of symbols, increases.
     </details>


     <details>
          <summary>Abstract</summary>
          Prompting techniques have significantly enhanced the capabilities of Large Language Models (LLMs) across various complex tasks, including reasoning, planning, and solving math word problems. However, most research has predominantly focused on language-based reasoning and word problems, often overlooking the potential of LLMs in handling symbol-based calculations and reasoning. This study aims to bridge this gap by rigorously evaluating LLMs on a series of symbolic tasks, such as addition, multiplication, modulus arithmetic, numerical precision, and symbolic counting. Our analysis encompasses eight LLMs, including four enterprise-grade and four open-source models, of which three have been pre-trained on mathematical tasks. The assessment framework is anchored in Chomsky's Hierarchy, providing a robust measure of the computational abilities of these models. The evaluation employs minimally explained prompts alongside the zero-shot Chain of Thoughts technique, allowing models to navigate the solution process autonomously. The findings reveal a significant decline in LLMs' performance on context-free and context-sensitive symbolic tasks as the complexity, represented by the number of symbols, increases. Notably, even the fine-tuned GPT3.5 exhibits only marginal improvements, mirroring the performance trends observed in other models. Across the board, all models demonstrated a limited generalization ability on these symbol-intensive tasks. This research underscores LLMs' challenges with increasing symbolic complexity and highlights the need for specialized training, memory and architectural adjustments to enhance their proficiency in symbol-based reasoning tasks.
     </details>

213. **DOP: Diagnostic-Oriented Prompting for Large Language Models in Mathematical Correction** [[pdf]](http://arxiv.org/abs/2405.12100) `2024-05-20` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Math world problems correction(MWPC) is a novel task dedicated to rectifying reasoning errors in the process of solving mathematical problems. In this paper, leveraging the advancements in large language models (LLMs), we address two key objectives:(1) Distinguishing between mathematical reasoning and error correction; (2) Exploring strategies to enhance the error correction capabilities of LLMs in mathematics to solve MWPC task. We noticed that, in real-time education,assisting students in recognizing their mistakes is more crucial than simply providing correct answers. However, current research tends to prioritize obtaining accurate solutions to math problems rather than correcting potentially incorrect ones. Therefore, we modify the research paradigm, demonstrating that improving mathematical reasoning abilities does not equate to mastery in error correction. Meanwhile, we propose a novel method called diagnostic-oriented promping(DOP) aimed at facilitating LLMs to excel in error correction. In experiments, DOP has shown outstanding performance, highlighting its significant impact. We argue that in mathematical education, the demand for outstanding correctors surpasses that for proficient reasoners. Codes and data are available on https://github.com/ChenhaoEcnuCS/Reason-Correct.
     </details>

214. **Self-Explore to Avoid the Pit: Improving the Reasoning Capabilities of Language Models with Fine-grained Rewards** [[pdf]](http://arxiv.org/abs/2404.10346) `2024-05-16` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Training on large amounts of rationales (i.e., CoT Fine-tuning) is effective at improving the reasoning capabilities of large language models (LLMs). However, acquiring human-authored rationales or augmenting rationales from proprietary models is costly and not scalable. In this paper, we study the problem of whether LLMs could self-improve their reasoning capabilities. To this end, we propose Self-Explore, where the LLM is tasked to explore the first wrong step (i.e., the first pit) within the rationale and use such signals as fine-grained rewards for further improvement. On the GSM8K and MATH test set, Self-Explore achieves 11.57% and 2.89% improvement on average across three LLMs compared to supervised fine-tuning (SFT). Our code is available at https://github.com/hbin0701/Self-Explore.
     </details>

215. **Improving Automated Distractor Generation for Math Multiple-choice Questions with Overgenerate-and-rank** [[pdf]](http://arxiv.org/abs/2405.05144) `2024-05-13` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel method to enhance the quality of generated distractors through overgenerate-and-rank, training a ranking model to predict how likely distractors are to be selected by real students is proposed.
     </details>


     <details>
          <summary>Abstract</summary>
          Multiple-choice questions (MCQs) are commonly used across all levels of math education since they can be deployed and graded at a large scale. A critical component of MCQs is the distractors, i.e., incorrect answers crafted to reflect student errors or misconceptions. Automatically generating them in math MCQs, e.g., with large language models, has been challenging. In this work, we propose a novel method to enhance the quality of generated distractors through overgenerate-and-rank, training a ranking model to predict how likely distractors are to be selected by real students. Experimental results on a real-world dataset and human evaluation with math teachers show that our ranking model increases alignment with human-authored distractors, although human-authored ones are still preferred over generated ones.
     </details>

216. **MuMath-Code: Combining Tool-Use Large Language Models with Multi-perspective Data Augmentation for Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2405.07551) `NAACL 2024 Findings` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work firstly includes new math questions via multi-perspective data augmenting methods and then synthesize code-nested solutions to them and proposes a two-stage training strategy, which leverages the advantages of both the external tool and data augmentation.
     </details>


     <details>
          <summary>Abstract</summary>
          The tool-use Large Language Models (LLMs) that integrate with external Python interpreters have significantly enhanced mathematical reasoning capabilities for open-source LLMs, while tool-free methods chose another track: augmenting math reasoning data. However, a great method to integrate the above two research paths and combine their advantages remains to be explored. In this work, we firstly include new math questions via multi-perspective data augmenting methods and then synthesize code-nested solutions to them. The open LLMs (i.e., Llama-2) are finetuned on the augmented dataset to get the resulting models, MuMath-Code ($\mu$-Math-Code). During the inference phase, our MuMath-Code generates code and interacts with the external python interpreter to get the execution results. Therefore, MuMath-Code leverages the advantages of both the external tool and data augmentation. To fully leverage the advantages of our augmented data, we propose a two-stage training strategy: In Stage-1, we finetune Llama-2 on pure CoT data to get an intermediate model, which then is trained on the code-nested data in Stage-2 to get the resulting MuMath-Code. Our MuMath-Code-7B achieves 83.8 on GSM8K and 52.4 on MATH, while MuMath-Code-70B model achieves new state-of-the-art performance among open methods -- achieving 90.7% on GSM8K and 55.1% on MATH. Extensive experiments validate the combination of tool use and data augmentation, as well as our two-stage training strategy. We release the proposed dataset along with the associated code for public use.
     </details>

217. **MathDivide: Improved mathematical reasoning by large language models** [[pdf]](http://arxiv.org/abs/2405.13004) `2024-05-12` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A prompting technique called MathDivide is proposed that breaks down the mathematical problem into simpler subproblems and demonstrates that MathDivide was able to significantly outperform the leading prompting technique called Math-prompter.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models have been proven to be capable of handling complex linguistic and cognitive tasks. Therefore their usage has been extended to tasks requiring logical reasoning ability such as Mathematics. In this paper, we propose a prompting technique called MathDivide that breaks down the mathematical problem into simpler subproblems. Each of the subproblems is formulated as an algebraic expression whose value is evaluated by the Python code generated by the LLM for the corresponding algebraic expression. The values fed to the Python code are the numerical values provided in the problem statement. The solutions for the subproblems are composed together to obtain the final answer for the problem statement. Finally, the final answer is compared to the correct answer. If the final answer matches the correct answer, it is produced as output else a refinement prompt is fed to the LLM. We experiment with this prompting technique on both closed-source LLM models and open-source LLM models using GSM8K dataset. The results obtained demonstrate that MathDivide was able to significantly outperform the leading prompting technique called Math-prompter.
     </details>

218. **Natural Language Reasoning, A Survey** [[pdf]](https://dl.acm.org/doi/10.1145/3664194) `2024-05-09` (36 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This survey paper proposes a clearer view of natural language reasoning in the field of Natural Language Processing (NLP), and identifies and views backward reasoning, a powerful paradigm for multi-step reasoning, and introduces defeasible reasoning as one of the most important future directions in natural language reasoning research.
     </details>


     <details>
          <summary>Abstract</summary>
          This survey paper proposes a clearer view of natural language reasoning in the field of Natural Language Processing (NLP), both conceptually and practically. Conceptually, we provide a distinct definition for natural language reasoning in NLP, based on both philosophy and NLP scenarios, discuss what types of tasks require reasoning, and introduce a taxonomy of reasoning. Practically, we conduct a comprehensive literature review on natural language reasoning in NLP, mainly covering classical logical reasoning, natural language inference, multi-hop question answering, and commonsense reasoning. The paper also identifies and views backward reasoning, a powerful paradigm for multi-step reasoning, and introduces defeasible reasoning as one of the most important future directions in natural language reasoning research. We focus on single-modality unstructured natural language text, excluding neuro-symbolic techniques and mathematical reasoning.
     </details>

219. **LLMs Can Find Mathematical Reasoning Mistakes by Pedagogical Chain-of-Thought** [[pdf]](http://arxiv.org/abs/2405.06705) `IJCAI 2024 Knowledge Representation and Reasoning` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Self-correction is emerging as a promising approach to mitigate the issue of hallucination in Large Language Models (LLMs). To facilitate effective self-correction, recent research has proposed mistake detection as its initial step. However, current literature suggests that LLMs often struggle with reliably identifying reasoning mistakes when using simplistic prompting strategies. To address this challenge, we introduce a unique prompting strategy, termed the Pedagogical Chain-of-Thought (PedCoT), which is specifically designed to guide the identification of reasoning mistakes, particularly mathematical reasoning mistakes. PedCoT consists of pedagogical principles for prompts (PPP) design, two-stage interaction process (TIP) and grounded PedCoT prompts, all inspired by the educational theory of the Bloom Cognitive Model (BCM). We evaluate our approach on two public datasets featuring math problems of varying difficulty levels. The experiments demonstrate that our zero-shot prompting strategy significantly outperforms strong baselines. The proposed method can achieve the goal of reliable mathematical mistake identification and provide a foundation for automatic math answer grading. The results underscore the significance of educational theory, serving as domain knowledge, in guiding prompting strategy design for addressing challenging tasks with LLMs effectively.
     </details>

220. **Learning to Solve Geometry Problems via Simulating Human Dual-Reasoning Process** [[pdf]](http://arxiv.org/abs/2405.06232) `IJCAI 2024 Natural Language Processing` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A Dual-Reasoning Geometry Solver (DualGeoSolver) to simulate the dual-reasoning process of humans for GPS and demonstrates the superiority of DualGeoSolver in both solving accuracy and robustness from explicitly modeling human reasoning process and knowledge application.
     </details>


     <details>
          <summary>Abstract</summary>
          Geometry Problem Solving (GPS), which is a classic and challenging math problem, has attracted much attention in recent years. It requires a solver to comprehensively understand both text and diagram, master essential geometry knowledge, and appropriately apply it in reasoning. However, existing works follow a paradigm of neural machine translation and only focus on enhancing the capability of encoders, which neglects the essential characteristics of human geometry reasoning. In this paper, inspired by dual-process theory, we propose a Dual-Reasoning Geometry Solver (DualGeoSolver) to simulate the dual-reasoning process of humans for GPS. Specifically, we construct two systems in DualGeoSolver, namely Knowledge System and Inference System. Knowledge System controls an implicit reasoning process, which is responsible for providing diagram information and geometry knowledge according to a step-wise reasoning goal generated by Inference System. Inference System conducts an explicit reasoning process, which specifies the goal in each reasoning step and applies the knowledge to generate program tokens for resolving it. The two systems carry out the above process iteratively, which behaves more in line with human cognition. We conduct extensive experiments on two benchmark datasets, GeoQA and GeoQA+. The results demonstrate the superiority of DualGeoSolver in both solving accuracy and robustness from explicitly modeling human reasoning process and knowledge application.
     </details>

221. **Key-Point-Driven Data Synthesis with its Enhancement on Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2403.02333) `2024-05-07` (13 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes Key-Point-Driven Data Synthesis (KPDDS), a novel data synthesis framework that synthesizes question-answer pairs by leveraging key points and exemplar practices from authentic data sources to ensure the generation of novel questions with rigorous quality control and substantial scalability.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have shown great potential in complex reasoning tasks, yet their performance is often hampered by the scarcity of high-quality and reasoning-focused training datasets. Addressing this challenge, we propose Key-Point-Driven Data Synthesis (KPDDS), a novel data synthesis framework that synthesizes question-answer pairs by leveraging key points and exemplar practices from authentic data sources. KPDDS ensures the generation of novel questions with rigorous quality control and substantial scalability. As a result, we present KPMath, an extensive synthetic dataset tailored for mathematical reasoning, comprising over 800K question-answer pairs. Utilizing KPMath and augmenting it with additional reasoning-intensive corpora, we create the comprehensive KPMath-Plus dataset. The Qwen1.5-72B model, fine-tuned on KPMath-Plus, achieves 87.0% PASS@1 accuracy on GSM8K and 58.3% on MATH, surpassing competitors in the 7B to 70B range and best commercial models like GPT-4 across multiple math reasoning datasets.
     </details>

222. **ATG: Benchmarking Automated Theorem Generation for Generative Language Models** [[pdf]](http://arxiv.org/abs/2405.06677) `NAACL 2024 Findings` `MetaMath` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          An Automated Theorem Generation (ATG) benchmark that evaluates whether an agent can automatically generate valuable (and possibly brand new) theorems that are applicable for downstream theorem proving as reusable knowledge is proposed.
     </details>


     <details>
          <summary>Abstract</summary>
          Humans can develop new theorems to explore broader and more complex mathematical results.While current generative language models (LMs) have achieved significant improvement in automatically proving theorems, their ability to generate new or reusable theorems is still under-explored. Without the new theorems, current LMs struggle to prove harder theorems that are distant from the given hypotheses with the exponentially growing search space.More advanced theorem proving is if an agent (for instance, a generative LM) can leverage its creativity to generate new but also reasonable theorems that properly substitute part of a proof and also be saved as reusable knowledge for future theorem proving.Therefore, this paper proposes an Automated Theorem Generation (ATG) benchmark that evaluates whether an agent can automatically generate valuable (and possibly brand new) theorems that are applicable for downstream theorem proving as reusable knowledge. Specifically, we construct the ATG benchmark by splitting the Metamath library into three sets: axioms, library, and problem based on their proving depth.We conduct extensive experiments to investigate whether current LMs can generate theorems in the library and benefit the problem theorems proving. The results demonstrate that high-quality ATG data facilitates models’ performances on downstream ATP. However, there is still room for current LMs to develop better ATG and generate more advanced and human-like theorems. We hope the new ATG challenge can shed some light on advanced complex theorem proving.
     </details>

223. **Monte Carlo Tree Search Boosts Reasoning via Iterative Preference Learning** [[pdf]](https://arxiv.org/abs/2405.00451v2) `2024-05-01` (14 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work leverages Monte Carlo Tree Search (MCTS) to iteratively collect preference data, utilizing its look-ahead ability to break down instance-level rewards into more granular step-level signals, to enhance consistency in intermediate steps.
     </details>


     <details>
          <summary>Abstract</summary>
          We introduce an approach aimed at enhancing the reasoning capabilities of Large Language Models (LLMs) through an iterative preference learning process inspired by the successful strategy employed by AlphaZero. Our work leverages Monte Carlo Tree Search (MCTS) to iteratively collect preference data, utilizing its look-ahead ability to break down instance-level rewards into more granular step-level signals. To enhance consistency in intermediate steps, we combine outcome validation and stepwise self-evaluation, continually updating the quality assessment of newly generated data. The proposed algorithm employs Direct Preference Optimization (DPO) to update the LLM policy using this newly generated step-level preference data. Theoretical analysis reveals the importance of using on-policy sampled data for successful self-improving. Extensive evaluations on various arithmetic and commonsense reasoning tasks demonstrate remarkable performance improvements over existing models. For instance, our approach outperforms the Mistral-7B Supervised Fine-Tuning (SFT) baseline on GSM8K, MATH, and ARC-C, with substantial increases in accuracy to $81.8\%$ (+$5.9\%$), $34.7\%$ (+$5.8\%$), and $76.4\%$ (+$15.8\%$), respectively. Additionally, our research delves into the training and inference compute tradeoff, providing insights into how our method effectively maximizes performance gains. Our code is publicly available at https://github.com/YuxiXie/MCTS-DPO.
     </details>

224. **Self-Refine Instruction-Tuning for Aligning Reasoning in Language Models** [[pdf]](http://arxiv.org/abs/2405.00402) `2024-05-01` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper proposes the Self-refine Instruction-tuning method that elicits Smaller Language Models to self-refine their abilities, and shows that this approach significantly outperforms Instruction-tuning in both in-domain and out-domain scenarios, aligning the reasoning abilities of Smaller and Larger Language Models.
     </details>


     <details>
          <summary>Abstract</summary>
          The alignments of reasoning abilities between smaller and larger Language Models are largely conducted via Supervised Fine-Tuning (SFT) using demonstrations generated from robust Large Language Models (LLMs). Although these approaches deliver more performant models, they do not show sufficiently strong generalization ability as the training only relies on the provided demonstrations.   In this paper, we propose the Self-refine Instruction-tuning method that elicits Smaller Language Models to self-refine their abilities. Our approach is based on a two-stage process, where reasoning abilities are first transferred between LLMs and Small Language Models (SLMs) via Instruction-tuning on demonstrations provided by LLMs, and then the instructed models Self-refine their abilities through preference optimization strategies. In particular, the second phase operates refinement heuristics based on the Direct Preference Optimization algorithm, where the SLMs are elicited to deliver a series of reasoning paths by automatically sampling the generated responses and providing rewards using ground truths from the LLMs. Results obtained on commonsense and math reasoning tasks show that this approach significantly outperforms Instruction-tuning in both in-domain and out-domain scenarios, aligning the reasoning abilities of Smaller and Larger Language Models.
     </details>

225. **General Purpose Verification for Chain of Thought Prompting** [[pdf]](http://arxiv.org/abs/2405.00204) `2024-04-30` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper explores ways to improve reasoning capabilities of LLMs through (1) exploration of different chains of thought and (2) validation of the individual steps of the reasoning process through three general principles that a model should adhere to while reasoning.
     </details>


     <details>
          <summary>Abstract</summary>
          Many of the recent capabilities demonstrated by Large Language Models (LLMs) arise primarily from their ability to exploit contextual information. In this paper, we explore ways to improve reasoning capabilities of LLMs through (1) exploration of different chains of thought and (2) validation of the individual steps of the reasoning process. We propose three general principles that a model should adhere to while reasoning: (i) Relevance, (ii) Mathematical Accuracy, and (iii) Logical Consistency. We apply these constraints to the reasoning steps generated by the LLM to improve the accuracy of the final generation. The constraints are applied in the form of verifiers: the model itself is asked to verify if the generated steps satisfy each constraint. To further steer the generations towards high-quality solutions, we use the perplexity of the reasoning steps as an additional verifier. We evaluate our method on 4 distinct types of reasoning tasks, spanning a total of 9 different datasets. Experiments show that our method is always better than vanilla generation, and, in 6 out of the 9 datasets, it is better than best-of N sampling which samples N reasoning chains and picks the lowest perplexity generation.
     </details>

226. **Can Large Language Models put 2 and 2 together? Probing for Entailed Arithmetical Relationships** [[pdf]](https://arxiv.org/abs/2404.19432v1) `2024-04-30` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is empirical demonstrate that although LLMs make steady progress in knowledge acquisition and (pseudo)reasoning with each new GPT release, their capabilities are limited to statistical inference only, and it is argued that bigger is not always better and chasing purely statistical improvements is flawed at the core.
     </details>


     <details>
          <summary>Abstract</summary>
          Two major areas of interest in the era of Large Language Models regard questions of what do LLMs know, and if and how they may be able to reason (or rather, approximately reason). Since to date these lines of work progressed largely in parallel (with notable exceptions), we are interested in investigating the intersection: probing for reasoning about the implicitly-held knowledge. Suspecting the performance to be lacking in this area, we use a very simple set-up of comparisons between cardinalities associated with elements of various subjects (e.g. the number of legs a bird has versus the number of wheels on a tricycle). We empirically demonstrate that although LLMs make steady progress in knowledge acquisition and (pseudo)reasoning with each new GPT release, their capabilities are limited to statistical inference only. It is difficult to argue that pure statistical learning can cope with the combinatorial explosion inherent in many commonsense reasoning tasks, especially once arithmetical notions are involved. Further, we argue that bigger is not always better and chasing purely statistical improvements is flawed at the core, since it only exacerbates the dangerous conflation of the production of correct answers with genuine reasoning ability.
     </details>

227. **Exploring Internal Numeracy in Language Models: A Case Study on ALBERT** [[pdf]](http://arxiv.org/abs/2404.16574) `2024-04-25` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          These findings illustrate that language models, trained purely to model text, can intuit basic mathematical concepts, opening avenues for NLP applications that intersect with quantitative reasoning.
     </details>


     <details>
          <summary>Abstract</summary>
          It has been found that Transformer-based language models have the ability to perform basic quantitative reasoning. In this paper, we propose a method for studying how these models internally represent numerical data, and use our proposal to analyze the ALBERT family of language models. Specifically, we extract the learned embeddings these models use to represent tokens that correspond to numbers and ordinals, and subject these embeddings to Principal Component Analysis (PCA). PCA results reveal that ALBERT models of different sizes, trained and initialized separately, consistently learn to use the axes of greatest variation to represent the approximate ordering of various numerical concepts. Numerals and their textual counterparts are represented in separate clusters, but increase along the same direction in 2D space. Our findings illustrate that language models, trained purely to model text, can intuit basic mathematical concepts, opening avenues for NLP applications that intersect with quantitative reasoning.
     </details>

228. **SAAS: Solving Ability Amplification Strategy for Enhanced Mathematical Reasoning in Large Language Models** [[pdf]](http://arxiv.org/abs/2404.03887) `2024-04-24` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This study proposes a sequential learning approach, named SAAS (Solving Ability Amplification Strategy), which strategically transitions from CoT learning to PoT learning, and demonstrates that the SAAS achieves state-of-the-art (SOTA) performance.
     </details>


     <details>
          <summary>Abstract</summary>
          This study presents a novel learning approach designed to enhance both mathematical reasoning and problem-solving abilities of Large Language Models (LLMs). We focus on integrating the Chain-of-Thought (CoT) and the Program-of-Thought (PoT) learning, hypothesizing that prioritizing the learning of mathematical reasoning ability is helpful for the amplification of problem-solving ability. Thus, the initial learning with CoT is essential for solving challenging mathematical problems. To this end, we propose a sequential learning approach, named SAAS (Solving Ability Amplification Strategy), which strategically transitions from CoT learning to PoT learning. Our empirical study, involving an extensive performance comparison using several benchmarks, demonstrates that our SAAS achieves state-of-the-art (SOTA) performance. The results underscore the effectiveness of our sequential learning approach, marking a significant advancement in the field of mathematical reasoning in LLMs.
     </details>

229. **Describe-then-Reason: Improving Multimodal Mathematical Reasoning through Visual Comprehension Training** [[pdf]](https://arxiv.org/abs/2404.14604v3) `2024-04-22` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A two-step training pipeline VCAR, which improves the visual comprehension ability of MLLMs through the visual description generation task, followed by another training step on generating rationales with the assistance of descriptions, demonstrates that VCAR substantially outperforms baseline methods solely relying on rationale supervision.
     </details>


     <details>
          <summary>Abstract</summary>
          Open-source multimodal large language models (MLLMs) excel in various tasks involving textual and visual inputs but still struggle with complex multimodal mathematical reasoning, lagging behind proprietary models like GPT-4V(ision) and Gemini-Pro. Although fine-tuning with intermediate steps (i.e., rationales) elicits some mathematical reasoning skills, the resulting models still fall short in visual comprehension due to inadequate visual-centric supervision, which leads to inaccurate interpretation of math figures. To address this issue, we propose a two-step training pipeline VCAR, which emphasizes the Visual Comprehension training in Addition to mathematical Reasoning learning. It first improves the visual comprehension ability of MLLMs through the visual description generation task, followed by another training step on generating rationales with the assistance of descriptions. Experimental results on two popular benchmarks demonstrate that VCAR substantially outperforms baseline methods solely relying on rationale supervision, especially on problems with high visual demands.
     </details>

230. **FGeo-HyperGNet: Geometric Problem Solving Integrating Formal Symbolic System and Hypergraph Neural Network** [[pdf]](http://arxiv.org/abs/2402.11461) `2024-04-22` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Geometric problem solving has always been a long-standing challenge in the fields of automated reasoning and artificial intelligence. We built a neural-symbolic system to automatically perform human-like geometric deductive reasoning. The symbolic part is a formal system built on FormalGeo, which can automatically perform geomertic relational reasoning and algebraic calculations and organize the solving process into a solution hypertree with conditions as hypernodes and theorems as hyperedges. The neural part, called HyperGNet, is a hypergraph neural network based on the attention mechanism, including a encoder to effectively encode the structural and semantic information of the hypertree, and a solver to provide problem-solving guidance. The neural part predicts theorems according to the hypertree, and the symbolic part applies theorems and updates the hypertree, thus forming a predict-apply cycle to ultimately achieve readable and traceable automatic solving of geometric problems. Experiments demonstrate the correctness and effectiveness of this neural-symbolic architecture. We achieved a step-wised accuracy of 87.65% and an overall accuracy of 85.53% on the formalgeo7k datasets.
     </details>

231. **MARIO Eval: Evaluate Your Math LLM with your Math LLM--A mathematical dataset evaluation toolkit** [[pdf]](http://arxiv.org/abs/2404.13925) `2024-04-22` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have been explored in a variety of reasoning tasks including solving of mathematical problems. Each math dataset typically includes its own specially designed evaluation script, which, while suitable for its intended use, lacks generalizability across different datasets. Consequently, updates and adaptations to these evaluation tools tend to occur without being systematically reported, leading to inconsistencies and obstacles to fair comparison across studies. To bridge this gap, we introduce a comprehensive mathematical evaluation toolkit that not only utilizes a python computer algebra system (CAS) for its numerical accuracy, but also integrates an optional LLM, known for its considerable natural language processing capabilities. To validate the effectiveness of our toolkit, we manually annotated two distinct datasets. Our experiments demonstrate that the toolkit yields more robust evaluation results compared to prior works, even without an LLM. Furthermore, when an LLM is incorporated, there is a notable enhancement. The code for our method will be made available at \url{https://github.com/MARIO-Math-Reasoning/math_evaluation}.
     </details>

232. **PARAMANU-GANITA: Language Model with Mathematical Capabilities** [[pdf]](http://arxiv.org/abs/2404.14395) `2024-04-22` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Paramanu-Ganita despite being 35 times smaller than 7B LLMs, outperformed generalist LLMs and math specialised LLMs such as Minerva 8B, LLEMMA 7B, and LLEMMA-7B in GSM8k test accuracy metric respectively and concludes that for strong mathematical reasoning abilities of language model, the authors do not need giant LLMs and immense computing power to their end.
     </details>


     <details>
          <summary>Abstract</summary>
          In this paper, we present Paramanu-Ganita, a 208 million parameter novel Auto Regressive (AR) decoder based language model on mathematics. The model is pretrained from scratch at context size of 4096 on our curated mixed mathematical corpus. We evaluate our model on both perplexity metric and GSM8k mathematical benchmark. Paramanu-Ganita despite being 35 times smaller than 7B LLMs, outperformed generalist LLMs such as LLaMa-1 7B by 28.4% points, LLaMa-2 7B by 27.6% points, Falcon 7B by 32.6% points, PaLM 8B by 35.3% points, and math specialised LLMs such as Minerva 8B by 23.2% points, and LLEMMA-7B by 3.0% points in GSM8k test accuracy metric respectively. Paramanu-Ganita also outperformed giant LLMs like PaLM 62B by 6.4% points, Falcon 40B by 19.8% points, LLaMa-1 33B by 3.8% points and Vicuna 13B by 11.8% points respectively. The large significant margin improvement in performance of our math model over the existing LLMs signifies that reasoning capabilities of language model are just not restricted to LLMs with humongous number of parameters. Paramanu-Ganita took 146 hours of A100 training whereas math specialised LLM, LLEMMA 7B, was trained for 23,000 A100 hours of training equivalent. Thus, our approach of pretraining powerful domain specialised language models from scratch for domain adaptation is much more cost-effective than performing continual training of LLMs for domain adaptation. Hence, we conclude that for strong mathematical reasoning abilities of language model, we do not need giant LLMs and immense computing power to our end. In the end, we want to point out that we have only trained Paramanu-Ganita only on a part of our entire mathematical corpus and yet to explore the full potential of our model.
     </details>

233. **Mathify: Evaluating Large Language Models on Mathematical Problem Solving Tasks** [[pdf]](http://arxiv.org/abs/2404.13099) `NeurIPS 2023 Workshop on Generative AI for Education` (7 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          An extensive mathematics dataset called "MathQuest"ourced from the 11th and 12th standard Mathematics NCERT textbooks is introduced, and MAmmoTH-13B establishes itself as a robust and dependable benchmark for addressing NCERT mathematics problems.
     </details>


     <details>
          <summary>Abstract</summary>
          The rapid progress in the field of natural language processing (NLP) systems and the expansion of large language models (LLMs) have opened up numerous opportunities in the field of education and instructional methods. These advancements offer the potential for tailored learning experiences and immediate feedback, all delivered through accessible and cost-effective services. One notable application area for this technological advancement is in the realm of solving mathematical problems. Mathematical problem-solving not only requires the ability to decipher complex problem statements but also the skill to perform precise arithmetic calculations at each step of the problem-solving process. However, the evaluation of the arithmetic capabilities of large language models remains an area that has received relatively little attention. In response, we introduce an extensive mathematics dataset called "MathQuest" sourced from the 11th and 12th standard Mathematics NCERT textbooks. This dataset encompasses mathematical challenges of varying complexity and covers a wide range of mathematical concepts. Utilizing this dataset, we conduct fine-tuning experiments with three prominent LLMs: LLaMA-2, WizardMath, and MAmmoTH. These fine-tuned models serve as benchmarks for evaluating their performance on our dataset. Our experiments reveal that among the three models, MAmmoTH-13B emerges as the most proficient, achieving the highest level of competence in solving the presented mathematical problems. Consequently, MAmmoTH-13B establishes itself as a robust and dependable benchmark for addressing NCERT mathematics problems.
     </details>

234. **Enhancing Length Extrapolation in Sequential Models with Pointer-Augmented Neural Memory** [[pdf]](https://arxiv.org/abs/2404.11870v1) `2024-04-18` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          PANM helps Transformer achieve up to 100% generalization accuracy in compositional learning tasks and significantly better results in mathematical reasoning, question answering and machine translation tasks.
     </details>


     <details>
          <summary>Abstract</summary>
          We propose Pointer-Augmented Neural Memory (PANM) to help neural networks understand and apply symbol processing to new, longer sequences of data. PANM integrates an external neural memory that uses novel physical addresses and pointer manipulation techniques to mimic human and computer symbol processing abilities. PANM facilitates pointer assignment, dereference, and arithmetic by explicitly using physical pointers to access memory content. Remarkably, it can learn to perform these operations through end-to-end training on sequence data, powering various sequential models. Our experiments demonstrate PANM's exceptional length extrapolating capabilities and improved performance in tasks that require symbol processing, such as algorithmic reasoning and Dyck language recognition. PANM helps Transformer achieve up to 100% generalization accuracy in compositional learning tasks and significantly better results in mathematical reasoning, question answering and machine translation tasks.
     </details>

235. **Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models** [[pdf]](http://arxiv.org/abs/2312.06585) `2024-04-17` (72 cite) (7 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Testing on advanced MATH reasoning and APPS coding benchmarks using PaLM-2 models, it is found that ReST$^{EM}$ scales favorably with model size and significantly surpasses fine-tuning only on human data.
     </details>


     <details>
          <summary>Abstract</summary>
          Fine-tuning language models~(LMs) on human-generated data remains a prevalent practice. However, the performance of such models is often limited by the quantity and diversity of high-quality human data. In this paper, we explore whether we can go beyond human data on tasks where we have access to scalar feedback, for example, on math problems where one can verify correctness. To do so, we investigate a simple self-training method based on expectation-maximization, which we call ReST$^{EM}$, where we (1) generate samples from the model and filter them using binary feedback, (2) fine-tune the model on these samples, and (3) repeat this process a few times. Testing on advanced MATH reasoning and APPS coding benchmarks using PaLM-2 models, we find that ReST$^{EM}$ scales favorably with model size and significantly surpasses fine-tuning only on human data. Overall, our findings suggest self-training with feedback can substantially reduce dependence on human-generated data.
     </details>

236. **Reformatted Alignment** [[pdf]](http://arxiv.org/abs/2402.12219) `2024-04-17` (6 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A simple and effective approach named ReAlign is introduced, which reformats the responses of instruction data into a format that better aligns with pre-established criteria and the collated evidence, and significantly boosts the general alignment ability, math reasoning, factuality, and readability of the LLMs.
     </details>


     <details>
          <summary>Abstract</summary>
          The quality of finetuning data is crucial for aligning large language models (LLMs) with human values. Current methods to improve data quality are either labor-intensive or prone to factual errors caused by LLM hallucinations. This paper explores elevating the quality of existing instruction data to better align with human values, introducing a simple and effective approach named ReAlign, which reformats the responses of instruction data into a format that better aligns with pre-established criteria and the collated evidence. This approach minimizes human annotation, hallucination, and the difficulty in scaling, remaining orthogonal to existing alignment techniques. Experimentally, ReAlign significantly boosts the general alignment ability, math reasoning, factuality, and readability of the LLMs.   Encouragingly, without introducing any additional data or advanced training techniques, and merely by reformatting the response, LLaMA-2-13B's mathematical reasoning ability on GSM8K can be improved from 46.77% to 56.63% in accuracy. Additionally, a mere 5% of ReAlign data yields a 67% boost in general alignment ability measured by the Alpaca dataset. This work highlights the need for further research into the science and mechanistic interpretability of LLMs. We have made the associated code and data publicly accessible to support future studies at https://github.com/GAIR-NLP/ReAlign.
     </details>

237. **A Survey on Deep Learning for Theorem Proving** [[pdf]](http://arxiv.org/abs/2404.09939) `COLM 2024` (6 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper presents a comprehensive survey of deep learning for theorem proving by offering a thorough review of existing approaches across various tasks such as autoformalization, premise selection, proofstep generation, and proof search.
     </details>


     <details>
          <summary>Abstract</summary>
          Theorem proving is a fundamental aspect of mathematics, spanning from informal reasoning in natural language to rigorous derivations in formal systems. In recent years, the advancement of deep learning, especially the emergence of large language models, has sparked a notable surge of research exploring these techniques to enhance the process of theorem proving. This paper presents a comprehensive survey of deep learning for theorem proving by offering (i) a thorough review of existing approaches across various tasks such as autoformalization, premise selection, proofstep generation, and proof search; (ii) an extensive summary of curated datasets and strategies for synthetic data generation; (iii) a detailed analysis of evaluation metrics and the performance of state-of-the-art methods; and (iv) a critical discussion on the persistent challenges and the promising avenues for future exploration. Our survey aims to serve as a foundational reference for deep learning approaches in theorem proving, inspiring and catalyzing further research endeavors in this rapidly growing field. A curated list of papers is available at https://github.com/zhaoyu-li/DL4TP.
     </details>

238. **Wu's Method can Boost Symbolic AI to Rival Silver Medalists and AlphaGeometry to Outperform Gold Medalists at IMO Geometry** [[pdf]](http://arxiv.org/abs/2404.06405) `2024-04-11` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          By combining AlphaGeometry with Wu's method, the IMO-AG-30 Challenge is revisited, and Wu's method is surprisingly strong, setting a new state-of-the-art for automated theorem proving on IMO-AG-30, solving 27 out of 30 problems, the first AI method which outperforms an IMO gold medalist.
     </details>


     <details>
          <summary>Abstract</summary>
          Proving geometric theorems constitutes a hallmark of visual reasoning combining both intuitive and logical skills. Therefore, automated theorem proving of Olympiad-level geometry problems is considered a notable milestone in human-level automated reasoning. The introduction of AlphaGeometry, a neuro-symbolic model trained with 100 million synthetic samples, marked a major breakthrough. It solved 25 of 30 International Mathematical Olympiad (IMO) problems whereas the reported baseline based on Wu's method solved only ten. In this note, we revisit the IMO-AG-30 Challenge introduced with AlphaGeometry, and find that Wu's method is surprisingly strong. Wu's method alone can solve 15 problems, and some of them are not solved by any of the other methods. This leads to two key findings: (i) Combining Wu's method with the classic synthetic methods of deductive databases and angle, ratio, and distance chasing solves 21 out of 30 methods by just using a CPU-only laptop with a time limit of 5 minutes per problem. Essentially, this classic method solves just 4 problems less than AlphaGeometry and establishes the first fully symbolic baseline strong enough to rival the performance of an IMO silver medalist. (ii) Wu's method even solves 2 of the 5 problems that AlphaGeometry failed to solve. Thus, by combining AlphaGeometry with Wu's method we set a new state-of-the-art for automated theorem proving on IMO-AG-30, solving 27 out of 30 problems, the first AI method which outperforms an IMO gold medalist.
     </details>

239. **A Survey in Mathematical Language Processing** [[pdf]](http://arxiv.org/abs/2205.15231) `2024-04-08` (11 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work tracks the development of informal mathematical language processing approaches across five strategic sub-areas in recent years, highlighting the prevailing successful methodological elements along with existing limitations.
     </details>


     <details>
          <summary>Abstract</summary>
          Informal mathematical text underpins real-world quantitative reasoning and communication. Developing sophisticated methods of retrieval and abstraction from this dual modality is crucial in the pursuit of the vision of automating discovery in quantitative science and mathematics. We track the development of informal mathematical language processing approaches across five strategic sub-areas in recent years, highlighting the prevailing successful methodological elements along with existing limitations.
     </details>

240. **Evaluating Mathematical Reasoning Beyond Accuracy** [[pdf]](http://arxiv.org/abs/2404.05692) `2024-04-08` (5 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          The leaderboard of Large Language Models (LLMs) in mathematical tasks has been continuously updated. However, the majority of evaluations focus solely on the final results, neglecting the quality of the intermediate steps. This oversight can mask underlying problems, such as logical errors or unnecessary steps in the reasoning process. To measure reasoning beyond final-answer accuracy, we introduce ReasonEval, a new methodology for evaluating the quality of reasoning steps. ReasonEval employs $\textit{validity}$ and $\textit{redundancy}$ to characterize the reasoning quality, as well as accompanying LLMs to assess them automatically. Instantiated by base models that possess strong mathematical knowledge and trained with high-quality labeled data, ReasonEval achieves state-of-the-art performance on human-labeled datasets and can accurately detect different types of errors generated by perturbation. When applied to evaluate LLMs specialized in math, we find that an increase in final-answer accuracy does not necessarily guarantee an improvement in the overall quality of the reasoning steps for challenging mathematical problems. Additionally, we observe that ReasonEval can play a significant role in data selection. We release the best-performing model, meta-evaluation script, and all evaluation results at https://github.com/GAIR-NLP/ReasonEval.
     </details>

241. **FRACTAL: Fine-Grained Scoring from Aggregate Text Labels** [[pdf]](http://arxiv.org/abs/2404.04817) `2024-04-07` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work is the first to develop response-level feedback to sentence-level scoring techniques, leveraging sentence-level prior information, along with comprehensive evaluations on multiple tasks as well as end-to-end finetuning evaluation showing performance comparable to a model trained on fine-grained human annotated labels.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) are being increasingly tuned to power complex generation tasks such as writing, fact-seeking, querying and reasoning. Traditionally, human or model feedback for evaluating and further tuning LLM performance has been provided at the response level, enabling faster and more cost-effective assessments. However, recent works (Amplayo et al. [2022], Wu et al. [2023]) indicate that sentence-level labels may provide more accurate and interpretable feedback for LLM optimization. In this work, we introduce methods to disaggregate response-level labels into sentence-level (pseudo-)labels. Our approach leverages multiple instance learning (MIL) and learning from label proportions (LLP) techniques in conjunction with prior information (e.g., document-sentence cosine similarity) to train a specialized model for sentence-level scoring. We also employ techniques which use model predictions to pseudo-label the train-set at the sentence-level for model training to further improve performance.   We conduct extensive evaluations of our methods across six datasets and four tasks: retrieval, question answering, summarization, and math reasoning. Our results demonstrate improved performance compared to multiple baselines across most of these tasks. Our work is the first to develop response-level feedback to sentence-level scoring techniques, leveraging sentence-level prior information, along with comprehensive evaluations on multiple tasks as well as end-to-end finetuning evaluation showing performance comparable to a model trained on fine-grained human annotated labels.
     </details>

242. **MM-MATH: Advancing Multimodal Math Evaluation with Process Evaluation and Fine-grained Classification** [[pdf]](https://arxiv.org/abs/2404.05091v4) `2024-04-07` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduces a novel benchmark, MM-MATH, consisting of 5,929 open-ended middle school math problems with visual contexts, with fine-grained classification across difficulty, grade level, and knowledge points, which incorporates both outcome and process evaluations.
     </details>


     <details>
          <summary>Abstract</summary>
          To advance the evaluation of multimodal math reasoning in large multimodal models (LMMs), this paper introduces a novel benchmark, MM-MATH. MM-MATH consists of 5,929 open-ended middle school math problems with visual contexts, with fine-grained classification across difficulty, grade level, and knowledge points. Unlike existing benchmarks relying on binary answer comparison, MM-MATH incorporates both outcome and process evaluations. Process evaluation employs LMM-as-a-judge to automatically analyze solution steps, identifying and categorizing errors into specific error types. Extensive evaluation of ten models on MM-MATH reveals significant challenges for existing LMMs, highlighting their limited utilization of visual information and struggles with higher-difficulty problems. The best-performing model achieves only 31% accuracy on MM-MATH, compared to 82% for humans. This highlights the challenging nature of our benchmark for existing models and the significant gap between the multimodal reasoning capabilities of current models and humans. Our process evaluation reveals that diagram misinterpretation is the most common error, accounting for more than half of the total error cases, underscoring the need for improved image comprehension in multimodal reasoning.
     </details>

243. **Large Language Models for Mathematical Reasoning: Progresses and Challenges** [[pdf]](http://arxiv.org/abs/2402.00157) `EACL 2024 student research workshop` (53 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This survey stands as one of the first extensive examinations of the landscape of LLM-oriented techniques in the realm of mathematics, providing a holistic perspective on the current state, accomplishments, and future challenges in this rapidly evolving field.
     </details>


     <details>
          <summary>Abstract</summary>
          Mathematical reasoning serves as a cornerstone for assessing the fundamental cognitive capabilities of human intelligence. In recent times, there has been a notable surge in the development of Large Language Models (LLMs) geared towards the automated resolution of mathematical problems. However, the landscape of mathematical problem types is vast and varied, with LLM-oriented techniques undergoing evaluation across diverse datasets and settings. This diversity makes it challenging to discern the true advancements and obstacles within this burgeoning field. This survey endeavors to address four pivotal dimensions: i) a comprehensive exploration of the various mathematical problems and their corresponding datasets that have been investigated; ii) an examination of the spectrum of LLM-oriented techniques that have been proposed for mathematical problem-solving; iii) an overview of factors and concerns affecting LLMs in solving math; and iv) an elucidation of the persisting challenges within this domain. To the best of our knowledge, this survey stands as one of the first extensive examinations of the landscape of LLMs in the realm of mathematics, providing a holistic perspective on the current state, accomplishments, and future challenges in this rapidly evolving field.
     </details>

244. **Data Augmentation with In-Context Learning and Comparative Evaluation in Math Word Problem Solving** [[pdf]](https://arxiv.org/abs/2404.03938v1) `2024-04-05` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This study aims to provide MWP solvers with a more diverse training set, ultimately improving their ability to solve various math problems by introducing a new in-context learning augmentation method, employing the Llama-7b language model.
     </details>


     <details>
          <summary>Abstract</summary>
          Math Word Problem (MWP) solving presents a challenging task in Natural Language Processing (NLP). This study aims to provide MWP solvers with a more diverse training set, ultimately improving their ability to solve various math problems. We propose several methods for data augmentation by modifying the problem texts and equations, such as synonym replacement, rule-based: question replacement, and rule based: reversing question methodologies over two English MWP datasets. This study extends by introducing a new in-context learning augmentation method, employing the Llama-7b language model. This approach involves instruction-based prompting for rephrasing the math problem texts. Performance evaluations are conducted on 9 baseline models, revealing that augmentation methods outperform baseline models. Moreover, concatenating examples generated by various augmentation methods further improves performance.
     </details>

245. **Proceedings 12th International Workshop on Theorem proving components for Educational software** [[pdf]](https://arxiv.org/abs/2404.03709v1) `2024-04-04` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The volume editors hope that this collection of papers will further promote the development of theorem-proving based software, and that it will allow to improve the mutual understanding between computer scientists, mathematicians and stakeholders in education.
     </details>


     <details>
          <summary>Abstract</summary>
          The ThEdu series pursues the smooth transition from an intuitive way of doing mathematics at secondary school to a more formal approach to the subject in STEM education, while favouring software support for this transition by exploiting the power of theorem-proving technologies. What follows is a brief description of how the present volume contributes to this enterprise.   The 12th International Workshop on Theorem Proving Components for Educational Software(ThEdu'23), was a satellite event of the 29th international Conference on Automated Deduction (CADE 2023), July 1-4, 2023, Rome, Italy. ThEdu'23 was very successful, with one invited talk, by Yves Bertot (Inria, France), "The challenges of using Type Theory to teach Mathematics", and seven regular contributions. An open call for papers was then issued, to which eight contributions were submitted. Seven submissions have been accepted by our reviewers, who jointly produced at least three careful reports on each of the contributions. The resulting revised papers are collected in the present volume.   We, the volume editors, hope that this collection of papers will further promote the development of theorem-proving based software, and that it will allow to improve the mutual understanding between computer scientists, mathematicians and stakeholders in education.   PC Chairs:Julien Narboux (University of Strasbourg, France); Walther Neuper (JKU, Johannes Kepler University, Linz, Austria); Pedro Quaresma (University of Coimbra, Portugal)
     </details>

246. **ChatGLM-Math: Improving Math Problem-Solving in Large Language Models with a Self-Critique Pipeline** [[pdf]](https://arxiv.org/abs/2404.02893v1) `2024-04-03` (18 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work tailor the Self-Critique pipeline, which addresses the challenge in the feedback learning stage of LLM alignment, and significantly enhances the LLM's mathematical problem-solving while still improving its language ability, outperforming LLMs that could be two times larger.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have shown excellent mastering of human language, but still struggle in real-world applications that require mathematical problem-solving. While many strategies and datasets to enhance LLMs' mathematics are developed, it remains a challenge to simultaneously maintain and improve both language and mathematical capabilities in deployed LLM systems.In this work, we tailor the Self-Critique pipeline, which addresses the challenge in the feedback learning stage of LLM alignment. We first train a general Math-Critique model from the LLM itself to provide feedback signals. Then, we sequentially employ rejective fine-tuning and direct preference optimization over the LLM's own generations for data collection. Based on ChatGLM3-32B, we conduct a series of experiments on both academic and our newly created challenging dataset, MathUserEval. Results show that our pipeline significantly enhances the LLM's mathematical problem-solving while still improving its language ability, outperforming LLMs that could be two times larger. Related techniques have been deployed to ChatGLM\footnote{\url{https://chatglm.cn}}, an online serving LLM. Related evaluation dataset and scripts are released at \url{https://github.com/THUDM/ChatGLM-Math}.
     </details>

247. **Advancing LLM Reasoning Generalists with Preference Trees** [[pdf]](http://arxiv.org/abs/2404.02078) `2024-04-02` (41 cite) (3 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces Eurus, a suite of large language models (LLMs) optimized for reasoning, and derives a novel reward modeling objective which, together with UltraInteract, leads to a strong reward model.
     </details>


     <details>
          <summary>Abstract</summary>
          We introduce Eurus, a suite of large language models (LLMs) optimized for reasoning. Finetuned from Mistral-7B and CodeLlama-70B, Eurus models achieve state-of-the-art results among open-source models on a diverse set of benchmarks covering mathematics, code generation, and logical reasoning problems. Notably, Eurus-70B beats GPT-3.5 Turbo in reasoning through a comprehensive benchmarking across 12 tests covering five tasks, and achieves a 33.3% pass@1 accuracy on LeetCode and 32.6% on TheoremQA, two challenging benchmarks, substantially outperforming existing open-source models by margins more than 13.3%. The strong performance of Eurus can be primarily attributed to UltraInteract, our newly-curated large-scale, high-quality alignment dataset specifically designed for complex reasoning tasks. UltraInteract can be used in both supervised fine-tuning and preference learning. For each instruction, it includes a preference tree consisting of (1) reasoning chains with diverse planning strategies in a unified format, (2) multi-turn interaction trajectories with the environment and the critique, and (3) pairwise data to facilitate preference learning. UltraInteract allows us to conduct an in-depth exploration of preference learning for reasoning tasks. Our investigation reveals that some well-established preference learning algorithms may be less suitable for reasoning tasks compared to their effectiveness in general conversations. Inspired by this, we derive a novel reward modeling objective which, together with UltraInteract, leads to a strong reward model.
     </details>

248. **Autonomous Data Selection with Language Models for Mathematical Texts** [[pdf]](http://arxiv.org/abs/2402.07625) `ICLR 2024 Workshop DPFM` (15 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work continuously pretrained a 7B-parameter language model on the authors' curated dataset, achieving substantial improvements in downstream performance on the MATH, GSM8K, and BIG-Bench Hard (BBH) tasks with a token amount reduced by orders of magnitude compared to previous continual pretraining works.
     </details>


     <details>
          <summary>Abstract</summary>
          To improve language models' proficiency in mathematical reasoning via continual pretraining, we introduce a novel strategy that leverages base language models for autonomous data selection. Departing from conventional supervised fine-tuning or trained classifiers with human-annotated data, our approach Autonomous Data Selection (AutoDS) utilizes meta-prompted language models as zero-shot verifiers to evaluate and select high-quality mathematical content autonomously. To demonstrate the efficacy of our method, we continuously pretrained a 7B-parameter language model on our curated dataset, achieving substantial improvements in downstream performance on the MATH, GSM8K, and BIG-Bench Hard (BBH) tasks with a token amount reduced by orders of magnitude compared to previous continual pretraining works. Our method showcases a 2 times increase in pretraining token efficiency compared to state-of-the-art baselines, underscoring the potential of our approach in enhancing models' mathematical reasoning capabilities. The AutoMathText dataset is available at https://huggingface.co/datasets/math-ai/AutoMathText. The code is available at https://github.com/yifanzhang-pro/AutoMathText.
     </details>

249. **Large Language Models for Mathematicians** [[pdf]](http://arxiv.org/abs/2312.04556) `2024-04-02` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A mathematical description of the transformer model used in all modern language models is provided and light is shed on the potential of LLMs to change how mathematicians work.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) such as ChatGPT have received immense interest for their general-purpose language understanding and, in particular, their ability to generate high-quality text or computer code. For many professions, LLMs represent an invaluable tool that can speed up and improve the quality of work. In this note, we discuss to what extent they can aid professional mathematicians. We first provide a mathematical description of the transformer model used in all modern language models. Based on recent studies, we then outline best practices and potential issues and report on the mathematical abilities of language models. Finally, we shed light on the potential of LLMs to change how mathematicians work.
     </details>

250. **$\texttt{LM}^\texttt{2}$: A Simple Society of Language Models Solves Complex Reasoning** [[pdf]](https://arxiv.org/abs/2404.02255v1) `2024-04-02` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Despite demonstrating emergent reasoning abilities, Large Language Models (LLMS) often lose track of complex, multi-step reasoning. Existing studies show that providing guidance via decomposing the original question into multiple subproblems elicits more robustness in LLM reasoning -- a decomposer generates the subproblems, and a solver solves each of these subproblems. However, these techniques fail to accommodate coordination between the decomposer and the solver modules (either in a single model or different specialized ones) -- the decomposer does not keep track of the ability of the solver to follow the decomposed reasoning. In this paper, we propose LM2 to address these challenges. LM2 modularizes the decomposition, solution, and verification into three different language models. The decomposer module identifies the key concepts necessary to solve the problem and generates step-by-step subquestions according to the reasoning requirement. The solver model generates the solution to the subproblems that are then checked by the verifier module; depending upon the feedback from the verifier, the reasoning context is constructed using the subproblems and the solutions. These models are trained to coordinate using policy learning. Exhaustive experimentation suggests the superiority of LM2 over existing methods on in- and out-domain reasoning problems, outperforming the best baselines by $8.1\%$ on MATH, $7.71\%$ on JEEBench, and $9.7\%$ on MedQA problems (code available at https://github.com/LCS2-IIITD/Language_Model_Multiplex).
     </details>

251. **Enhancing Formal Theorem Proving: A Comprehensive Dataset for Training AI Models on Coq Code** [[pdf]](http://arxiv.org/abs/2403.12627) `2024-04-02` `Coq` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper presents a comprehensive dataset specifically designed to enhance LLMs' proficiency in interpreting and generating Coq code, and discusses the dataset's composition, the methodology behind its creation, and the implications for the future of machine learning in formal verification.
     </details>


     <details>
          <summary>Abstract</summary>
          In the realm of formal theorem proving, the Coq proof assistant stands out for its rigorous approach to verifying mathematical assertions and software correctness. Despite the advances in artificial intelligence and machine learning, the specialized nature of Coq syntax and semantics poses unique challenges for Large Language Models (LLMs). Addressing this gap, we present a comprehensive dataset specifically designed to enhance LLMs' proficiency in interpreting and generating Coq code. This dataset, derived from a collection of over 10,000 Coq source files, encompasses a wide array of propositions, proofs, and definitions, enriched with metadata including source references and licensing information. Our primary aim is to facilitate the development of LLMs capable of generating syntactically correct and semantically meaningful Coq constructs, thereby advancing the frontier of automated theorem proving. Initial experiments with this dataset have showcased its significant potential; models trained on this data exhibited enhanced accuracy in Coq code generation. Notably, a particular experiment revealed that a fine-tuned LLM was capable of generating 141 valid proofs for a basic lemma, highlighting the dataset's utility in facilitating the discovery of diverse and valid proof strategies. This paper discusses the dataset's composition, the methodology behind its creation, and the implications of our findings for the future of machine learning in formal verification. The dataset is accessible for further research and exploration: https://huggingface.co/datasets/florath/coq-facts-props-proofs-gen0-v1
     </details>

252. **Can Large Language Models Reason and Plan?** [[pdf]](http://arxiv.org/abs/2403.04121) `2024-04-01` (27 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          While humans sometimes do show the capability of correcting their own erroneous guesses with self-critiquing, there seems to be no basis for that assumption in the case of LLMs.
     </details>

253. **OVM, Outcome-supervised Value Models for Planning in Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2311.09724) `NAACL 2024 Findings` (18 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Outcome-supervised Value Model (OVM) is proposed that employs outcome supervision for training a value model, which prioritizes steps that lead to accurate conclusions, thereby significantly enhancing its scalability and eliminating the need for labor-intensive annotations of step-level correctness.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) often struggle with maintaining accuracy throughout multiple multiple reasoning steps, especially in mathematical reasoning where an error in earlier steps can propagate to subsequent ones and it ultimately leading to an incorrect answer.To reduce error propagation, guided decoding is employed to direct the LM decoding on a step-by-step basis. We argue that in guided decoding, assessing the potential of an incomplete reasoning path can be more advantageous than simply ensuring per-step correctness, as the former approach leads towards a correct final answer. This transforms the task into a value estimation problem in planning.Inspired by the findings that outcome supervision for guided decoding essentially acts as a value model, we propose Outcome-supervised Value Model (OVM) that employs outcome supervision for training a value model, which prioritizes steps that lead to accurate conclusions. Furthermore, the OVM eliminates the need for labor-intensive annotations of step-level correctness, thereby significantly enhancing its scalability. Our experiments on two multi-step mathematical reasoning datasets, GSM8K and Game of 24, demonstrate the superior performance of the OVM model. Notably, in GSM8K, our OVM-7B model achieves state-of-the-art results among LLMs up to 13B parameters; especially it does not utilize GPT-4 or code execution. These findings offer a novel perspective on the role of outcome supervision in training value models for multi-step reasoning tasks and provide theoretical justification for its advantage in value estimation for guided decoding.
     </details>

254. **Do language models plan ahead for future tokens?** [[pdf]](https://arxiv.org/abs/2404.00859v2) `2024-04-01` (10 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          In the autoregressive language modeling setting, the experiments are more suggestive of the breadcrumbs hypothesis, though pre-caching increases with model scale, and in a constructed synthetic data setting, they find clear evidence for pre-caching.
     </details>


     <details>
          <summary>Abstract</summary>
          Do transformers "think ahead" during inference at a given position? It is known transformers prepare information in the hidden states of the forward pass at time step $t$ that is then used in future forward passes $t+\tau$. We posit two explanations for this phenomenon: pre-caching, in which off-diagonal gradient terms present during training result in the model computing features at $t$ irrelevant to the present inference task but useful for the future, and breadcrumbs, in which features most relevant to time step $t$ are already the same as those that would most benefit inference at time $t+\tau$. We test these hypotheses by training language models without propagating gradients to past timesteps, a scheme we formalize as myopic training. In a constructed synthetic data setting, we find clear evidence for pre-caching. In the autoregressive language modeling setting, our experiments are more suggestive of the breadcrumbs hypothesis, though pre-caching increases with model scale.
     </details>

255. **Exploring the Mystery of Influential Data for Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2404.01067) `2024-04-01` (5 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a Quality-aware Diverse Selection (QaDS) strategy adaptable for mathematical reasoning, and showcases the use of QaDS in creating efficient fine-tuning mixtures with various selection ratios.
     </details>


     <details>
          <summary>Abstract</summary>
          Selecting influential data for fine-tuning on downstream tasks is a key factor for both performance and computation efficiency. Recent works have shown that training with only limited data can show a superior performance on general tasks. However, the feasibility on mathematical reasoning tasks has not been validated. To go further, there exist two open questions for mathematical reasoning: how to select influential data and what is an influential data composition. For the former one, we propose a Quality-aware Diverse Selection (QaDS) strategy adaptable for mathematical reasoning. A comparison with other selection strategies validates the superiority of QaDS. For the latter one, we first enlarge our setting and explore the influential data composition. We conduct a series of experiments and highlight: scaling up reasoning data, and training with general data selected by QaDS is helpful. Then, we define our optimal mixture as OpenMathMix, an influential data mixture with open-source data selected by QaDS. With OpenMathMix, we achieve a state-of-the-art 48.8% accuracy on MATH with 7B base model. Additionally, we showcase the use of QaDS in creating efficient fine-tuning mixtures with various selection ratios, and analyze the quality of a wide range of open-source datasets, which can perform as a reference for future works on mathematical reasoning tasks.
     </details>

256. **GFLean: An Autoformalisation Framework for Lean via GF** [[pdf]](http://arxiv.org/abs/2404.01234) `2024-04-01` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          An autoformalisation framework for the Lean theorem prover, called GFLean, which uses a high-level grammar writing tool called Grammatical Framework for parsing and linearisation and is implemented in Haskell.
     </details>


     <details>
          <summary>Abstract</summary>
          We present an autoformalisation framework for the Lean theorem prover, called GFLean. GFLean uses a high-level grammar writing tool called Grammatical Framework (GF) for parsing and linearisation. GFLean is implemented in Haskell. We explain the functionalities of GFLean, its inner working and discuss its limitations. We also discuss how we can use neural network based translation programs and rule based translation programs together complimenting each other to build robust autoformalisation frameworks.
     </details>

257. **NumeroLogic: Number Encoding for Enhanced LLMs' Numerical Reasoning** [[pdf]](http://arxiv.org/abs/2404.00459) `2024-03-30` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a simple adjustment to how numbers are represented by including the count of digits before each number, which offers an added advantage in number generation by serving as a Chain of Thought (CoT) by requiring the model to consider the number of digits first.
     </details>


     <details>
          <summary>Abstract</summary>
          Language models struggle with handling numerical data and performing arithmetic operations. We hypothesize that this limitation can be partially attributed to non-intuitive textual numbers representation. When a digit is read or generated by a causal language model it does not know its place value (e.g. thousands vs. hundreds) until the entire number is processed. To address this issue, we propose a simple adjustment to how numbers are represented by including the count of digits before each number. For instance, instead of "42", we suggest using "{2:42}" as the new format. This approach, which we term NumeroLogic, offers an added advantage in number generation by serving as a Chain of Thought (CoT). By requiring the model to consider the number of digits first, it enhances the reasoning process before generating the actual number. We use arithmetic tasks to demonstrate the effectiveness of the NumeroLogic formatting. We further demonstrate NumeroLogic applicability to general natural language modeling, improving language understanding performance in the MMLU benchmark.
     </details>

258. **Can LLMs Master Math? Investigating Large Language Models on Math Stack Exchange** [[pdf]](https://arxiv.org/abs/2404.00344v1) `2024-03-30` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The case analysis indicates that while the GPT-4 can generate relevant responses in certain instances, it does not consistently answer all questions accurately, and shed light on the gaps in LLM capabilities within mathematics, thereby setting the stage for future research and advancements in AI-driven mathematical reasoning.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have demonstrated exceptional capabilities in various natural language tasks, often achieving performances that surpass those of humans. Despite these advancements, the domain of mathematics presents a distinctive challenge, primarily due to its specialized structure and the precision it demands. In this study, we adopted a two-step approach for investigating the proficiency of LLMs in answering mathematical questions. First, we employ the most effective LLMs, as identified by their performance on math question-answer benchmarks, to generate answers to 78 questions from the Math Stack Exchange (MSE). Second, a case analysis is conducted on the LLM that showed the highest performance, focusing on the quality and accuracy of its answers through manual evaluation. We found that GPT-4 performs best (nDCG of 0.48 and P@10 of 0.37) amongst existing LLMs fine-tuned for answering mathematics questions and outperforms the current best approach on ArqMATH3 Task1, considering P@10. Our Case analysis indicates that while the GPT-4 can generate relevant responses in certain instances, it does not consistently answer all questions accurately. This paper explores the current limitations of LLMs in navigating complex mathematical problem-solving. Through case analysis, we shed light on the gaps in LLM capabilities within mathematics, thereby setting the stage for future research and advancements in AI-driven mathematical reasoning. We make our code and findings publicly available for research: \url{https://github.com/gipplab/LLM-Investig-MathStackExchange}
     </details>

259. **Large Language Models Are Unconscious of Unreasonability in Math Problems** [[pdf]](https://arxiv.org/abs/2403.19346v2) `2024-03-28` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          With a strategic prompt template called Critical Calculation and Conclusion (CCC), LLMs can better self-evaluate and detect unreasonable errors in math questions, making them more reliable and safe in practical application scenarios.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) demonstrate substantial capabilities in solving math problems. However, they tend to produce hallucinations when given questions containing unreasonable errors. In this paper, we study the behavior of LLMs when faced with unreasonable math problems and further explore their potential to address these problems. We construct the Unreasonable Math Problem (UMP) benchmark to examine the error detection ability of LLMs. Experiments show that LLMs are able to detect unreasonable errors, but still fail in generating non-hallucinatory content. In order to improve their ability of error detection and correction, we further design a strategic prompt template called Critical Calculation and Conclusion(CCC). With CCC, LLMs can better self-evaluate and detect unreasonable errors in math questions, making them more reliable and safe in practical application scenarios.
     </details>

260. **Dual Instruction Tuning with Large Language Models for Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2403.18295) `2024-03-27` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a dual instruction tuning strategy to meticulously model mathematical reasoning from both forward and reverse directions, and introduces the Intermediate Reasoning State Prediction task (forward reasoning) and the Instruction Reconstruction task (reverse reasoning) to enhance the LLMs' understanding and execution of instructions.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent advancements highlight the success of instruction tuning with large language models (LLMs) utilizing Chain-of-Thought (CoT) data for mathematical reasoning tasks. Despite the fine-tuned LLMs, challenges persist, such as incorrect, missing, and redundant steps in CoT generation leading to inaccuracies in answer predictions. To alleviate this problem, we propose a dual instruction tuning strategy to meticulously model mathematical reasoning from both forward and reverse directions. This involves introducing the Intermediate Reasoning State Prediction task (forward reasoning) and the Instruction Reconstruction task (reverse reasoning) to enhance the LLMs' understanding and execution of instructions. Training instances for these tasks are constructed based on existing mathematical instruction tuning datasets. Subsequently, LLMs undergo multi-task fine-tuning using both existing mathematical instructions and the newly created data. Comprehensive experiments validate the effectiveness and domain generalization of the dual instruction tuning strategy across various mathematical reasoning tasks.
     </details>

261. **Look Before You Leap: Problem Elaboration Prompting Improves Mathematical Reasoning in Large Language Models** [[pdf]](http://arxiv.org/abs/2402.15764) `2024-03-26` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A new approach named Problem Elaboration Prompting (PEP) is proposed to enhance the mathematical capacities of LLMs by decomposing and elucidates the problem context before reasoning, therefore enhancing the context modeling and parsing efficiency.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) still grapple with complex tasks like mathematical reasoning. Despite significant efforts invested in improving prefix prompts or reasoning process, the crucial role of problem context might have been neglected. Accurate recognition of inputs is fundamental for solving mathematical tasks, as ill-formed problems could potentially mislead LLM's reasoning. In this study, we propose a new approach named Problem Elaboration Prompting (PEP) to enhance the mathematical capacities of LLMs. Specifically, PEP decomposes and elucidates the problem context before reasoning, therefore enhancing the context modeling and parsing efficiency. Experiments across datasets and models demonstrate promising performances: (1) PEP demonstrates an overall enhancement in various mathematical tasks. For instance, with the GPT-3.5 model, PEP exhibits improvements of 9.93% and 8.80% on GSM8k through greedy decoding and self-consistency, respectively. (2) PEP can be easily implemented and integrated with other prompting methods. (3) PEP shows particular strength in handling distraction problems.
     </details>

262. **Graph of Thoughts: Solving Elaborate Problems with Large Language Models** [[pdf]](http://arxiv.org/abs/2308.09687) `AAAI 2024 Natural Language Processing` (335 cite) (14 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Graph of Thoughts is introduced: a framework that advances prompting capabilities in large language models (LLMs) beyond those offered by paradigms such as Chain-of-Thought or Tree of Thoughts, and is ensured that GoT is extensible with new thought transformations and thus can be used to spearhead new prompting schemes.
     </details>


     <details>
          <summary>Abstract</summary>
          We introduce Graph of Thoughts (GoT): a framework that advances prompting capabilities in large language models (LLMs) beyond those offered by paradigms such as Chain-of-Thought or Tree of Thoughts (ToT). The key idea and primary advantage of GoT is the ability to model the information generated by an LLM as an arbitrary graph, where units of information ("LLM thoughts") are vertices, and edges correspond to dependencies between these vertices. This approach enables combining arbitrary LLM thoughts into synergistic outcomes, distilling the essence of whole networks of thoughts, or enhancing thoughts using feedback loops. We illustrate that GoT offers advantages over state of the art on different tasks, for example increasing the quality of sorting by 62% over ToT, while simultaneously reducing costs by >31%. We ensure that GoT is extensible with new thought transformations and thus can be used to spearhead new prompting schemes. This work brings the LLM reasoning closer to human thinking or brain mechanisms such as recurrence, both of which form complex networks
     </details>

263. **Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Language Models** [[pdf]](http://arxiv.org/abs/2305.16582) `2024-03-22` (6 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Graph-of-Thought reasoning is proposed, which models human thought processes not only as a chain but also as a graph, and achieves significant improvement over the strong CoT baseline on the AQUA-RAT test set and boosts accuracy from 85.19% to 87.59% using the T5-base model.
     </details>


     <details>
          <summary>Abstract</summary>
          With the widespread use of language models (LMs) in NLP tasks, researchers have discovered the potential of Chain-of-thought (CoT) to assist LMs in accomplishing complex reasoning tasks by generating intermediate steps. However, human thought processes are often non-linear, rather than simply sequential chains of thoughts. Therefore, we propose Graph-of-Thought (GoT) reasoning, which models human thought processes not only as a chain but also as a graph. By representing thought units as nodes and connections between them as edges, our approach captures the non-sequential nature of human thinking and allows for a more realistic modeling of thought processes. GoT adopts a two-stage framework with an additional GoT encoder for thought graph representation and fuses the graph representation with the original input representation through a gated fusion mechanism. We evaluate GoT's performance on a text-only reasoning task (AQUA-RAT) and a multimodal reasoning task (ScienceQA). Our model achieves significant improvement over the strong CoT baseline on the AQUA-RAT test set and boosts accuracy from 85.19% to 87.59% using the T5-base model over the state-of-the-art Multimodal-CoT on the ScienceQA test set.
     </details>

264. **RankPrompt: Step-by-Step Comparisons Make Language Models Better Reasoners** [[pdf]](http://arxiv.org/abs/2403.12373) `2024-03-22` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          RankPrompt breaks down the ranking problem into a series of comparisons among diverse responses, leveraging the inherent capabilities of LLMs to generate chains of comparison as contextual exemplars and validate RankPrompt as an effective method for eliciting high-quality feedback from language models.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have achieved impressive performance across various reasoning tasks. However, even state-of-the-art LLMs such as ChatGPT are prone to logical errors during their reasoning processes. Existing solutions, such as deploying task-specific verifiers or voting over multiple reasoning paths, either require extensive human annotations or fail in scenarios with inconsistent responses. To address these challenges, we introduce RankPrompt, a new prompting method that enables LLMs to self-rank their responses without additional resources. RankPrompt breaks down the ranking problem into a series of comparisons among diverse responses, leveraging the inherent capabilities of LLMs to generate chains of comparison as contextual exemplars. Our experiments across 11 arithmetic and commonsense reasoning tasks show that RankPrompt significantly enhances the reasoning performance of ChatGPT and GPT-4, with improvements of up to 13%. Moreover, RankPrompt excels in LLM-based automatic evaluations for open-ended tasks, aligning with human judgments 74% of the time in the AlpacaEval dataset. It also exhibits robustness to variations in response order and consistency. Collectively, our results validate RankPrompt as an effective method for eliciting high-quality feedback from language models.
     </details>

265. **MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems?** [[pdf]](https://arxiv.org/abs/2403.14624v2) `2024-03-21` (55 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The MathVerse benchmark is introduced, an all-around visual math benchmark designed for an equitable and in-depth evaluation of MLLMs, and a Chain-of-Thought evaluation strategy is proposed for a fine-grained assessment of the output answers.
     </details>


     <details>
          <summary>Abstract</summary>
          The remarkable progress of Multi-modal Large Language Models (MLLMs) has garnered unparalleled attention, due to their superior performance in visual contexts. However, their capabilities in visual math problem-solving remain insufficiently evaluated and understood. We investigate current benchmarks to incorporate excessive visual content within textual questions, which potentially assist MLLMs in deducing answers without truly interpreting the input diagrams. To this end, we introduce MathVerse, an all-around visual math benchmark designed for an equitable and in-depth evaluation of MLLMs. We meticulously collect 2,612 high-quality, multi-subject math problems with diagrams from publicly available sources. Each problem is then transformed by human annotators into six distinct versions, each offering varying degrees of information content in multi-modality, contributing to 15K test samples in total. This approach allows MathVerse to comprehensively assess whether and how much MLLMs can truly understand the visual diagrams for mathematical reasoning. In addition, we propose a Chain-of-Thought (CoT) evaluation strategy for a fine-grained assessment of the output answers. Rather than naively judging True or False, we employ GPT-4(V) to adaptively extract crucial reasoning steps, and then score each step with detailed error analysis, which can reveal the intermediate CoT reasoning quality by MLLMs. We hope the MathVerse benchmark may provide unique insights to guide the future development of MLLMs. Project page: https://mathverse-cuhk.github.io
     </details>

266. **Reinforcement Learning from Reflective Feedback (RLRF): Aligning and Improving LLMs via Fine-Grained Self-Reflection** [[pdf]](https://arxiv.org/abs/2403.14238v1) `2024-03-21` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel framework: Reinforcement Learning from Reflective Feedback (RLRF), which leverages fine-grained feedback based on detailed criteria to improve the core capabilities of LLMs beyond superficial surface-level adjustment is proposed.
     </details>


     <details>
          <summary>Abstract</summary>
          Despite the promise of RLHF in aligning LLMs with human preferences, it often leads to superficial alignment, prioritizing stylistic changes over improving downstream performance of LLMs. Underspecified preferences could obscure directions to align the models. Lacking exploration restricts identification of desirable outputs to improve the models. To overcome these challenges, we propose a novel framework: Reinforcement Learning from Reflective Feedback (RLRF), which leverages fine-grained feedback based on detailed criteria to improve the core capabilities of LLMs. RLRF employs a self-reflection mechanism to systematically explore and refine LLM responses, then fine-tuning the models via a RL algorithm along with promising responses. Our experiments across Just-Eval, Factuality, and Mathematical Reasoning demonstrate the efficacy and transformative potential of RLRF beyond superficial surface-level adjustment.
     </details>

267. **From Large to Tiny: Distilling and Refining Mathematical Expertise for Math Word Problems with Weakly Supervision** [[pdf]](http://arxiv.org/abs/2403.14390) `2024-03-21` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          An innovative two-stage framework that adeptly transfers mathematical Expertise from large to tiny language models is introduced that demonstrates significantly improved performance on the Math23K and Weak12K datasets compared to existing small model methods, while maintaining a much lower computational cost than ChatGPT.
     </details>


     <details>
          <summary>Abstract</summary>
          Addressing the challenge of high annotation costs in solving Math Word Problems (MWPs) through full supervision with intermediate equations, recent works have proposed weakly supervised task settings that rely solely on the final answer as a supervised signal. Existing leading approaches typically employ various search techniques to infer intermediate equations, but cannot ensure their semantic consistency with natural language descriptions. The rise of Large Language Models (LLMs) like ChatGPT has opened up new possibilities for addressing MWPs directly. However, the computational demands of LLMs make them less than ideal for use in settings where resources are tight. In light of these challenges, we introduce an innovative two-stage framework that adeptly transfers mathematical Expertise from large to tiny language models. In \emph{Distillation Stage}, we propose a series of extraction processes that satisfy the properties of MWPs to distill mathematical knowledge from LLMs to construct problem-equation pairs required for supervised training. In \emph{Refinement Stage}, Due to Knowledge distilling method cannot guarantee the full utilization of all data, we further utilize the unsuccessfully searched data effectively by Knowledge Refine method. Finally, We train a small model using distilled data generated through two-stage methods. As our method fully leverages the semantic understanding capabilities during the searching 'problem-equation' pair, it demonstrates significantly improved performance on the Math23K and Weak12K datasets compared to existing small model methods, while maintaining a much lower computational cost than ChatGPT.
     </details>

268. **Arithmetic Feature Interaction Is Necessary for Deep Tabular Learning** [[pdf]](http://arxiv.org/abs/2402.02334) `AAAI 2024 Machine Learning` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Results show that AMFormer outperforms strong counterparts in fine-grained tabular data modeling, data efficiency in training, and generalization, suggesting it has established a strong inductive bias for deep learning on tabular data.
     </details>


     <details>
          <summary>Abstract</summary>
          Until recently, the question of the effective inductive bias of deep models on tabular data has remained unanswered. This paper investigates the hypothesis that arithmetic feature interaction is necessary for deep tabular learning. To test this point, we create a synthetic tabular dataset with a mild feature interaction assumption and examine a modified transformer architecture enabling arithmetical feature interactions, referred to as AMFormer. Results show that AMFormer outperforms strong counterparts in fine-grained tabular data modeling, data efficiency in training, and generalization. This is attributed to its parallel additive and multiplicative attention operators and prompt-based optimization, which facilitate the separation of tabular samples in an extended space with arithmetically-engineered features. Our extensive experiments on real-world data also validate the consistent effectiveness, efficiency, and rationale of AMFormer, suggesting it has established a strong inductive bias for deep learning on tabular data. Code is available at https://github.com/aigc-apps/AMFormer.
     </details>

269. **Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking** [[pdf]](http://arxiv.org/abs/2403.09629) `COLM 2024` (25 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Quiet-STaR is presented, a generalization of STaR in which LMs learn to generate rationales at each token to explain future text, improving their predictions and improving the LM's ability to directly answer difficult questions.
     </details>


     <details>
          <summary>Abstract</summary>
          When writing and talking, people sometimes pause to think. Although reasoning-focused works have often framed reasoning as a method of answering questions or completing agentic tasks, reasoning is implicit in almost all written text. For example, this applies to the steps not stated between the lines of a proof or to the theory of mind underlying a conversation. In the Self-Taught Reasoner (STaR, Zelikman et al. 2022), useful thinking is learned by inferring rationales from few-shot examples in question-answering and learning from those that lead to a correct answer. This is a highly constrained setting -- ideally, a language model could instead learn to infer unstated rationales in arbitrary text. We present Quiet-STaR, a generalization of STaR in which LMs learn to generate rationales at each token to explain future text, improving their predictions. We address key challenges, including 1) the computational cost of generating continuations, 2) the fact that the LM does not initially know how to generate or use internal thoughts, and 3) the need to predict beyond individual next tokens. To resolve these, we propose a tokenwise parallel sampling algorithm, using learnable tokens indicating a thought's start and end, and an extended teacher-forcing technique. Encouragingly, generated rationales disproportionately help model difficult-to-predict tokens and improve the LM's ability to directly answer difficult questions. In particular, after continued pretraining of an LM on a corpus of internet text with Quiet-STaR, we find zero-shot improvements on GSM8K (5.9%$\rightarrow$10.9%) and CommonsenseQA (36.3%$\rightarrow$47.2%) and observe a perplexity improvement of difficult tokens in natural text. Crucially, these improvements require no fine-tuning on these tasks. Quiet-STaR marks a step towards LMs that can learn to reason in a more general and scalable way.
     </details>

270. **Self-Consistency Boosts Calibration for Math Reasoning** [[pdf]](http://arxiv.org/abs/2403.09849) `2024-03-14` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Three off-the-shelf calibration methods based on self-consistency for math reasoning tasks better bridge model confidence and accuracy than existing methods based on p(True) (Kadavath et al., 2022) or logit (Kadavath et al., 2022).
     </details>


     <details>
          <summary>Abstract</summary>
          Calibration, which establishes the correlation between accuracy and model confidence, is important for LLM development. We design three off-the-shelf calibration methods based on self-consistency (Wang et al., 2022) for math reasoning tasks. Evaluation on two popular benchmarks (GSM8K and MathQA) using strong open-source LLMs (Mistral and LLaMA2), our methods better bridge model confidence and accuracy than existing methods based on p(True) (Kadavath et al., 2022) or logit (Kadavath et al., 2022).
     </details>

271. **Incorporating Graph Attention Mechanism into Geometric Problem Solving Based on Deep Reinforcement Learning** [[pdf]](https://arxiv.org/abs/2403.14690v1) `2024-03-14` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel algorithm, named Automatically Adding Auxiliary Components using Reinforcement Learning framework (A3C-RL), is proposed by forcing an agent to select top strategies, which incorporates the AttnStrategy and BERT as the memory components.
     </details>


     <details>
          <summary>Abstract</summary>
          In the context of online education, designing an automatic solver for geometric problems has been considered a crucial step towards general math Artificial Intelligence (AI), empowered by natural language understanding and traditional logical inference. In most instances, problems are addressed by adding auxiliary components such as lines or points. However, adding auxiliary components automatically is challenging due to the complexity in selecting suitable auxiliary components especially when pivotal decisions have to be made. The state-of-the-art performance has been achieved by exhausting all possible strategies from the category library to identify the one with the maximum likelihood. However, an extensive strategy search have to be applied to trade accuracy for ef-ficiency. To add auxiliary components automatically and efficiently, we present deep reinforcement learning framework based on the language model, such as BERT. We firstly apply the graph attention mechanism to reduce the strategy searching space, called AttnStrategy, which only focus on the conclusion-related components. Meanwhile, a novel algorithm, named Automatically Adding Auxiliary Components using Reinforcement Learning framework (A3C-RL), is proposed by forcing an agent to select top strategies, which incorporates the AttnStrategy and BERT as the memory components. Results from extensive experiments show that the proposed A3C-RL algorithm can substantially enhance the average precision by 32.7% compared to the traditional MCTS. In addition, the A3C-RL algorithm outperforms humans on the geometric questions from the annual University Entrance Mathematical Examination of China.
     </details>

272. **Laying the Foundation First? Investigating the Generalization from Atomic Skills to Complex Reasoning Tasks** [[pdf]](https://arxiv.org/abs/2403.09479v1) `2024-03-14` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          By leveraging hierarchical curriculum learning, this work successfully induce generalization, significantly improve the performance of open-source LMs on complex reasoning tasks, and offers valuable guidance for designing better training strategies for complex reasoning tasks.
     </details>


     <details>
          <summary>Abstract</summary>
          Current language models have demonstrated their capability to develop basic reasoning, but struggle in more complicated reasoning tasks that require a combination of atomic skills, such as math word problem requiring skills like arithmetic and unit conversion. Previous methods either do not improve the inherent atomic skills of models or not attempt to generalize the atomic skills to complex reasoning tasks. In this paper, we first propose a probing framework to investigate whether the atomic skill can spontaneously generalize to complex reasoning tasks. Then, we introduce a hierarchical curriculum learning training strategy to achieve better skill generalization. In our experiments, we find that atomic skills can not spontaneously generalize to compositional tasks. By leveraging hierarchical curriculum learning, we successfully induce generalization, significantly improve the performance of open-source LMs on complex reasoning tasks. Promisingly, the skill generalization exhibit effective in cross-dataset and cross-domain scenarios. Complex reasoning can also help enhance atomic skills. Our findings offer valuable guidance for designing better training strategies for complex reasoning tasks.
     </details>

273. **APOLLO: An Optimized Training Approach for Long-form Numerical Reasoning** [[pdf]](http://arxiv.org/abs/2212.07249) `2024-03-12` (5 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          APOLLO includes a number-aware negative sampling strategy for the retriever to discriminate key numerical facts, and a consistency-based reinforcement learning with target program augmentation for the generator to ultimately increase the execution accuracy.
     </details>


     <details>
          <summary>Abstract</summary>
          Long-form numerical reasoning in financial analysis aims to generate a reasoning program to calculate the correct answer for a given question. Previous work followed a retriever-generator framework, where the retriever selects key facts from a long-form document, and the generator generates a reasoning program based on retrieved facts. However, they treated all facts equally without considering the different contributions of facts with and without numbers. Meanwhile, the program consistency were ignored under supervised training, resulting in lower training accuracy and diversity. To solve these problems, we proposed APOLLO to improve the long-form numerical reasoning framework. For the retriever, we adopt a number-aware negative sampling strategy to enable the retriever to be more discriminative on key numerical facts. For the generator, we design consistency-based reinforcement learning and target program augmentation strategy based on the consistency of program execution results. Experimental results on the FinQA and ConvFinQA leaderboard verify the effectiveness of our proposed method, achieving the new state-of-the-art.
     </details>

274. **FineMath: A Fine-Grained Mathematical Evaluation Benchmark for Chinese Large Language Models** [[pdf]](http://arxiv.org/abs/2403.07747) `2024-03-12` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The proposed FineMath is a fine-grained mathematical evaluation benchmark dataset for assessing Chinese LLMs, created to cover the major key mathematical concepts taught in elementary school math, and finds that there is still considerable room for improvements in terms of mathematical reasoning capability of Chinese LLMs.
     </details>


     <details>
          <summary>Abstract</summary>
          To thoroughly assess the mathematical reasoning abilities of Large Language Models (LLMs), we need to carefully curate evaluation datasets covering diverse mathematical concepts and mathematical problems at different difficulty levels. In pursuit of this objective, we propose FineMath in this paper, a fine-grained mathematical evaluation benchmark dataset for assessing Chinese LLMs. FineMath is created to cover the major key mathematical concepts taught in elementary school math, which are further divided into 17 categories of math word problems, enabling in-depth analysis of mathematical reasoning abilities of LLMs. All the 17 categories of math word problems are manually annotated with their difficulty levels according to the number of reasoning steps required to solve these problems. We conduct extensive experiments on a wide range of LLMs on FineMath and find that there is still considerable room for improvements in terms of mathematical reasoning capability of Chinese LLMs. We also carry out an in-depth analysis on the evaluation process and methods that have been overlooked previously. These two factors significantly influence the model results and our understanding of their mathematical reasoning capabilities. The dataset will be publicly available soon.
     </details>

275. **Reverse That Number! Decoding Order Matters in Arithmetic Learning** [[pdf]](http://arxiv.org/abs/2403.05845) `2024-03-09` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces a novel strategy that not only reevaluates the digit order by prioritizing output from the least significant digit but also incorporates a step-by-step methodology to substantially reduce complexity.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent advancements in pretraining have demonstrated that modern Large Language Models (LLMs) possess the capability to effectively learn arithmetic operations. However, despite acknowledging the significance of digit order in arithmetic computation, current methodologies predominantly rely on sequential, step-by-step approaches for teaching LLMs arithmetic, resulting in a conclusion where obtaining better performance involves fine-grained step-by-step. Diverging from this conventional path, our work introduces a novel strategy that not only reevaluates the digit order by prioritizing output from the least significant digit but also incorporates a step-by-step methodology to substantially reduce complexity. We have developed and applied this method in a comprehensive set of experiments. Compared to the previous state-of-the-art (SOTA) method, our findings reveal an overall improvement of in accuracy while requiring only a third of the tokens typically used during training. For the purpose of facilitating replication and further research, we have made our code and dataset publicly available at \url{https://anonymous.4open.science/r/RAIT-9FB7/}.
     </details>

276. **RAT: Retrieval Augmented Thoughts Elicit Context-Aware Reasoning in Long-Horizon Generation** [[pdf]](http://arxiv.org/abs/2403.05313) `2024-03-08` (20 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work explores how iterative revising a chain of thoughts with the help of information retrieval significantly improves large language models' reasoning and generation ability in long-horizon generation tasks, while hugely mitigating hallucination.
     </details>


     <details>
          <summary>Abstract</summary>
          We explore how iterative revising a chain of thoughts with the help of information retrieval significantly improves large language models' reasoning and generation ability in long-horizon generation tasks, while hugely mitigating hallucination. In particular, the proposed method -- *retrieval-augmented thoughts* (RAT) -- revises each thought step one by one with retrieved information relevant to the task query, the current and the past thought steps, after the initial zero-shot CoT is generated. Applying RAT to GPT-3.5, GPT-4, and CodeLLaMA-7b substantially improves their performances on various long-horizon generation tasks; on average of relatively increasing rating scores by 13.63% on code generation, 16.96% on mathematical reasoning, 19.2% on creative writing, and 42.78% on embodied task planning. The demo page can be found at https://craftjarvis.github.io/RAT
     </details>

277. **Common 7B Language Models Already Possess Strong Math Capabilities** [[pdf]](http://arxiv.org/abs/2403.04706) `2024-03-07` (33 cite) (3 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper shows that the LLaMA-2 7B model with common pre-training already exhibits strong mathematical abilities, as evidenced by its impressive accuracy when selecting the best response from 256 random generations, and finds that simply scaling up the SFT data can significantly enhance the reliability of generating correct answers.
     </details>


     <details>
          <summary>Abstract</summary>
          Mathematical capabilities were previously believed to emerge in common language models only at a very large scale or require extensive math-related pre-training. This paper shows that the LLaMA-2 7B model with common pre-training already exhibits strong mathematical abilities, as evidenced by its impressive accuracy of 97.7% and 72.0% on the GSM8K and MATH benchmarks, respectively, when selecting the best response from 256 random generations. The primary issue with the current base model is the difficulty in consistently eliciting its inherent mathematical capabilities. Notably, the accuracy for the first answer drops to 49.5% and 7.9% on the GSM8K and MATH benchmarks, respectively. We find that simply scaling up the SFT data can significantly enhance the reliability of generating correct answers. However, the potential for extensive scaling is constrained by the scarcity of publicly available math questions. To overcome this limitation, we employ synthetic data, which proves to be nearly as effective as real data and shows no clear saturation when scaled up to approximately one million samples. This straightforward approach achieves an accuracy of 82.6% on GSM8K and 40.6% on MATH using LLaMA-2 7B models, surpassing previous models by 14.2% and 20.8%, respectively. We also provide insights into scaling behaviors across different reasoning complexities and error types.
     </details>

278. **Machine learning and information theory concepts towards an AI Mathematician** [[pdf]](http://arxiv.org/abs/2403.04571) `2024-03-07` (6 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This essay builds on the idea that current deep learning mostly succeeds at system 1 abilities—which correspond to the authors' intuition and habitual behaviors—but still lacks something important regarding system 2 abilities—which include reasoning and robust uncertainty estimation.
     </details>


     <details>
          <summary>Abstract</summary>
          The current state-of-the-art in artificial intelligence is impressive, especially in terms of mastery of language, but not so much in terms of mathematical reasoning. What could be missing? Can we learn something useful about that gap from how the brains of mathematicians go about their craft? This essay builds on the idea that current deep learning mostly succeeds at system 1 abilities -- which correspond to our intuition and habitual behaviors -- but still lacks something important regarding system 2 abilities -- which include reasoning and robust uncertainty estimation. It takes an information-theoretical posture to ask questions about what constitutes an interesting mathematical statement, which could guide future work in crafting an AI mathematician. The focus is not on proving a given theorem but on discovering new and interesting conjectures. The central hypothesis is that a desirable body of theorems better summarizes the set of all provable statements, for example by having a small description length while at the same time being close (in terms of number of derivation steps) to many provable statements.
     </details>

279. **Benchmarking Hallucination in Large Language Models based on Unanswerable Math Word Problem** [[pdf]](http://arxiv.org/abs/2403.03558) `2024-03-06` (5 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is shown that utilizing MWP is a reliable and effective approach to assess hallucination and in-context learning and reinforcement learning with human feedback (RLHF) training significantly enhance the model’s ability to avoid hallucination.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) are highly effective in various natural language processing (NLP) tasks. However, they are susceptible to producing unreliable conjectures in ambiguous contexts called hallucination. This paper presents a new method for evaluating LLM hallucination in Question Answering (QA) based on the unanswerable math word problem (MWP). To support this approach, we innovatively develop a dataset called Unanswerable Math Word Problem (UMWP) which comprises 5200 questions across five categories. We developed an evaluation methodology combining text similarity and mathematical expression detection to determine whether LLM considers the question unanswerable. The results of extensive experiments conducted on 31 LLMs, including GPT-3, InstructGPT, LLaMA, and Claude, demonstrate that in-context learning and reinforcement learning with human feedback (RLHF) training significantly enhance the model's ability to avoid hallucination. We show that utilizing MWP is a reliable and effective approach to assess hallucination. Our code and data are available at https://github.com/Yuki-Asuuna/UMWP.
     </details>

280. **Exploring the Limitations of Large Language Models in Compositional Relation Reasoning** [[pdf]](http://arxiv.org/abs/2403.02615) `2024-03-04` (4 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The Multilingual Composition Relation (MCR) benchmark aims at investigating the robustness and adaptability of LLMs in handling composition relation reasoning across diverse linguistic contexts.
     </details>


     <details>
          <summary>Abstract</summary>
          We present a comprehensive evaluation of large language models(LLMs)' ability to reason about composition relations through a benchmark encompassing 1,500 test cases in English, designed to cover six distinct types of composition relations: Positional, Comparative, Personal, Mathematical, Identity, and Other. Acknowledging the significance of multilingual capabilities, we expanded our assessment to include translations of these cases into Chinese, Japanese, French, and Korean. Our Multilingual Composition Relation (MCR) benchmark aims at investigating the robustness and adaptability of LLMs in handling composition relation reasoning across diverse linguistic contexts.
     </details>

281. **Functional Benchmarks for Robust Evaluation of Reasoning Performance, and the Reasoning Gap** [[pdf]](http://arxiv.org/abs/2402.19450) `2024-02-29` (22 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Here it is shown that models which anecdotally have good reasoning performance over real-world tasks, have quantifiable lower gaps, motivating the open problem of building"gap 0"models.
     </details>


     <details>
          <summary>Abstract</summary>
          We propose a framework for robust evaluation of reasoning capabilities of language models, using functional variants of benchmarks. Models that solve a reasoning test should exhibit no difference in performance over the static version of a problem compared to a snapshot of the functional variant. We have rewritten the relevant fragment of the MATH benchmark into its functional variant MATH(), with functionalization of other benchmarks to follow. When evaluating current state-of-the-art models over snapshots of MATH(), we find a reasoning gap -- the percentage difference between the static and functional accuracies. We find reasoning gaps from 58.35% to 80.31% among the state-of-the-art closed and open weights models that perform well on static benchmarks, with the caveat that the gaps are likely to be smaller with more sophisticated prompting strategies. Here we show that models which anecdotally have good reasoning performance over real-world tasks, have quantifiable lower gaps, motivating the open problem of building "gap 0" models. Code for evaluation and new evaluation datasets, three MATH() snapshots, are publicly available at https://github.com/consequentai/fneval/.
     </details>

282. **Automatic Prompt Augmentation and Selection with Chain-of-Thought from Labeled Data** [[pdf]](http://arxiv.org/abs/2302.12822) `EMNLP 2023 Findings` (99 cite) (3 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A new strategy, Automate-CoT (Automatic Prompt Augmentation and Selection with Chain-of-Thought), that can bypass human engineering of CoT by automatically augmenting rational chains from a small labeled dataset, and then pruning low-quality chains to construct a candidate pool of machine-generated rationale chains based on the labels.
     </details>


     <details>
          <summary>Abstract</summary>
          Chain-of-thought (CoT) advances the reasoning abilities of large language models (LLMs) and achieves superior performance in complex reasoning tasks. However, most CoT studies rely on carefully designed human-annotated rational chains to prompt LLMs, posing challenges for real-world applications where labeled data is available without rational chains. This paper proposes a new strategy, AutomateCoT (Automatic Prompt Augmentation and Selection with Chain-of-Thought), that can bypass human engineering of CoT by automatically augmenting rational chains from a small labeled dataset, and then pruning low-quality chains to construct a candidate pool of machinegenerated rationale chains based on the labels. Finally, it selects the optimal combination of several rationale chains from the pool for CoT prompting by employing a variance-reduced policy gradient strategy to estimate the significance of each example. Automate-CoT enables a quick adaptation of the CoT technique to different tasks. Experimental results demonstrate the effectiveness of our method, where competitive results are achieved on arithmetic reasoning (+2.7%), commonsense reasoning (+3.4%), symbolic reasoning (+3.2%), and non-reasoning tasks (+2.5%).
     </details>

283. **Stepwise Self-Consistent Mathematical Reasoning with Large Language Models** [[pdf]](http://arxiv.org/abs/2402.17786) `ICML 2024 Workshop ICL` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces a novel algorithm, namely Stepwise Self-Consistent Chain-of-Thought (SSC-CoT), which employs a strategy of selecting intermediate steps based on the intersection of various reasoning chains and enables the model to discover critical intermediate steps by querying a knowledge graph comprising relevant domain knowledge.
     </details>


     <details>
          <summary>Abstract</summary>
          Using Large Language Models for complex mathematical reasoning is difficult, primarily due to the complexity of multi-step reasoning. The main challenges of this process include (1) selecting critical intermediate results to advance the procedure, and (2) limited exploration of potential solutions. To address these issues, we introduce a novel algorithm, namely Stepwise Self-Consistent Chain-of-Thought (SSC-CoT). SSC-CoT employs a strategy of selecting intermediate steps based on the intersection of various reasoning chains. Additionally, SSC-CoT enables the model to discover critical intermediate steps by querying a knowledge graph comprising relevant domain knowledge. To validate SSC-CoT, we present a new dataset, TriMaster100, tailored for complex trigonometry problems. This dataset contains 100 questions, with each solution broken down into scored intermediate steps, facilitating a comprehensive evaluation of the mathematical reasoning process. On TriMaster100, SSC-CoT triples the effectiveness of the state-of-the-art methods. Furthermore, we benchmark SSC-CoT on the widely recognized complex mathematical question dataset, MATH level 5, and it surpasses the second-best method by 7.2% in accuracy. Code and the TriMaster100 dataset can be found at: https://github.com/zhao-zilong/ssc-cot.
     </details>

284. **ConceptMath: A Bilingual Concept-wise Benchmark for Measuring Mathematical Reasoning of Large Language Models** [[pdf]](http://arxiv.org/abs/2402.14660) `ACL 2024 Findings` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          ConceptMath is introduced, a bilingual (English and Chinese), fine-grained benchmark that evaluates concept-wise mathematical reasoning of Large Language Models (LLMs) and an efficient fine-tuning strategy to enhance the weaknesses of existing LLMs is introduced.
     </details>


     <details>
          <summary>Abstract</summary>
          This paper introduces ConceptMath, a bilingual (English and Chinese), fine-grained benchmark that evaluates concept-wise mathematical reasoning of Large Language Models (LLMs). Unlike traditional benchmarks that evaluate general mathematical reasoning with an average accuracy, ConceptMath systemically organizes math problems under a hierarchy of math concepts, so that mathematical reasoning can be evaluated at different granularity with concept-wise accuracies. Based on our ConcepthMath, we then evaluate a broad range of LLMs, and we observe existing LLMs, though achieving high average accuracies on traditional benchmarks, exhibit significant performance variations across different math concepts and may even fail catastrophically on the most basic ones. Besides, we also introduce an efficient fine-tuning strategy to enhance the weaknesses of existing LLMs. Finally, we hope ConceptMath could guide the developers to understand the fine-grained mathematical abilities of their models and facilitate the growth of foundation models. Code is available at https://github.com/conceptmath/conceptmath.
     </details>

285. **Mathematical Language Models: A Survey** [[pdf]](http://arxiv.org/abs/2312.07622) `2024-02-23` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A comprehensive survey of mathematical LMs is conducted, systematically categorizing pivotal research endeavors from two distinct perspectives: tasks and methodologies, revealing a large number of proposed mathematical LLMs.
     </details>


     <details>
          <summary>Abstract</summary>
          In recent years, there has been remarkable progress in leveraging Language Models (LMs), encompassing Pre-trained Language Models (PLMs) and Large-scale Language Models (LLMs), within the domain of mathematics. This paper conducts a comprehensive survey of mathematical LMs, systematically categorizing pivotal research endeavors from two distinct perspectives: tasks and methodologies. The landscape reveals a large number of proposed mathematical LLMs, which are further delineated into instruction learning, tool-based methods, fundamental CoT techniques, and advanced CoT methodologies. In addition, our survey entails the compilation of over 60 mathematical datasets, including training datasets, benchmark datasets, and augmented datasets. Addressing the primary challenges and delineating future trajectories within the field of mathematical LMs, this survey is positioned as a valuable resource, poised to facilitate and inspire future innovation among researchers invested in advancing this domain.
     </details>

286. **An Empirical Study of Data Ability Boundary in LLMs' Math Reasoning** [[pdf]](http://arxiv.org/abs/2403.00799) `2024-02-23` (0 cite) (1 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) are displaying emergent abilities for math reasoning tasks,and there is a growing attention on enhancing the ability of open-source LLMs through supervised fine-tuning (SFT).In this paper, we aim to explore a general data strategy for supervised data to help optimize and expand math reasoning ability.Firstly, we determine the ability boundary of reasoning paths augmentation by identifying these paths' minimal optimal set.Secondly, we validate that different abilities of the model can be cumulatively enhanced by Mix of Minimal Optimal Sets of corresponding types of data, while our models MMOS achieve SOTA performance on series base models under much lower construction costs.Besides, we point out GSM-HARD is not really hard and today's LLMs no longer lack numerical robustness.Also, we provide an Auto Problem Generator for robustness testing and educational applications.Our code and data are publicly available at https://github.com/cyzhh/MMOS.
     </details>

287. **Brain-Inspired Two-Stage Approach: Enhancing Mathematical Reasoning by Imitating Human Thought Processes** [[pdf]](http://arxiv.org/abs/2403.00800) `2024-02-23` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel approach, named Brain, to imitate human thought processes to enhance mathematical reasoning abilities, using the Frontal Lobe Model to generate plans, and then employing the Parietal Lobe Model to generate code and execute to obtain answers.
     </details>


     <details>
          <summary>Abstract</summary>
          Although large language models demonstrate emergent abilities in solving math word problems, there is a challenging task in complex multi-step mathematical reasoning tasks. To improve model performance on mathematical reasoning tasks, previous work has conducted supervised fine-tuning on open-source models by improving the quality and quantity of data. In this paper, we propose a novel approach, named Brain, to imitate human thought processes to enhance mathematical reasoning abilities, using the Frontal Lobe Model to generate plans, and then employing the Parietal Lobe Model to generate code and execute to obtain answers. First, we achieve SOTA performance in comparison with Code LLaMA 7B based models through this method. Secondly, we find that plans can be explicitly extracted from natural language, code, or formal language. Our code and data are publicly available at https://github.com/cyzhh/Brain.
     </details>

288. **Diversity of Thought Improves Reasoning Abilities of LLMs** [[pdf]](http://arxiv.org/abs/2310.07088) `2024-02-23` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a method that automatically improves prompt diversity by soliciting feedback from the LLM to ideate approaches that are apt for the problem, and improves the Pareto frontier of the accuracy-cost trade-off.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) are documented to struggle in settings that require complex reasoning. Nevertheless, instructing the model to break down the problem into smaller reasoning steps, or ensembling various generations through modifying decoding steps boosts performance. However, these methods assume that the input prompt is fixed and expect the decoding strategies to introduce the diversity needed for ensembling. In this work, we discuss how one can create and leverage variations of the input prompt as a means of diversity of thought. We propose a method that automatically improves prompt diversity by soliciting feedback from the LLM to ideate approaches that are apt for the problem. We then ensemble the diverse prompts in our method DIVSE (DIVerse reasoning path Self-Ensemble) across multiple inference calls, or use diverse approaches within a single inference call; we call the latter IDIV-SE (In-call DIVerse reasoning path Self-Ensemble). Apart from our approaches outperforming prior work, DIV-SE(in particular) advances state-of-the-art performance on the challenging planning and graph coloring benchmarks. Our results improve the Pareto frontier of the accuracy-cost trade-off.
     </details>

289. **RevOrder: A Novel Method for Enhanced Arithmetic in Language Models** [[pdf]](http://arxiv.org/abs/2402.03822) `2024-02-23` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          RevOrder, a novel technique aimed at improving arithmetic operations in large language models (LLMs) by reversing the output digits in addition, subtraction, and n-digit by 1-digit (nD by 1D) multiplication tasks, significantly reduces the Count of Sequential Intermediate Digits to $\mathcal{O}(1)$, a new metric to assess equation complexity.
     </details>


     <details>
          <summary>Abstract</summary>
          This paper presents RevOrder, a novel technique aimed at improving arithmetic operations in large language models (LLMs) by reversing the output digits in addition, subtraction, and n-digit by 1-digit (nD by 1D) multiplication tasks. Our method significantly reduces the Count of Sequential Intermediate Digits (CSID) to $\mathcal{O}(1)$, a new metric we introduce to assess equation complexity. Through comprehensive testing, RevOrder not only achieves perfect accuracy in basic arithmetic operations but also substantially boosts LLM performance in division tasks, particularly with large numbers where traditional models struggle. Implementation of RevOrder is cost-effective for both training and inference phases. Moreover, applying RevOrder to fine-tune the LLaMA2-7B model on the GSM8K math task results in a considerable improvement, reducing equation calculation errors by 46% and increasing overall scores from 41.6 to 44.4.
     </details>

290. **Tokenization counts: the impact of tokenization on arithmetic in frontier LLMs** [[pdf]](http://arxiv.org/abs/2402.14903) `2024-02-22` (24 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work performs the first study of how number tokenization choices lead to differences in model performance on arithmetic tasks, accompanied by a thorough analysis of error patterns.
     </details>


     <details>
          <summary>Abstract</summary>
          Tokenization, the division of input text into input tokens, is an often overlooked aspect of the large language model (LLM) pipeline and could be the source of useful or harmful inductive biases. Historically, LLMs have relied on byte pair encoding, without care to specific input domains. With the increased use of LLMs for reasoning, various number-specific tokenization schemes have been adopted, with popular models like LLaMa and PaLM opting for single-digit tokenization while GPT-3.5 and GPT-4 have separate tokens for each 1-, 2-, and 3-digit numbers. In this work, we study the effect this choice has on numerical reasoning through the use of arithmetic tasks. We consider left-to-right and right-to-left tokenization for GPT-3.5 and -4, finding that right-to-left tokenization (enforced by comma separating numbers at inference time) leads to largely improved performance. Furthermore, we find that model errors when using standard left-to-right tokenization follow stereotyped error patterns, suggesting that model computations are systematic rather than approximate. We show that the model is able to convert between tokenizations easily, thus allowing chain-of-thought-inspired approaches to recover performance on left-to-right tokenized inputs. We also find the gap between tokenization directions decreases when models are scaled, possibly indicating that larger models are better able to override this tokenization-dependent inductive bias. In summary, our work performs the first study of how number tokenization choices lead to differences in model performance on arithmetic tasks, accompanied by a thorough analysis of error patterns. We hope this work inspires practitioners to more carefully ablate number tokenization-related choices when working towards general models of numerical reasoning.
     </details>

291. **SciAgent: Tool-augmented Language Models for Scientific Reasoning** [[pdf]](http://arxiv.org/abs/2402.11451) `2024-02-20` (8 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work develops SciAgent to retrieve, understand and, if necessary, use tools for scientific problem solving, andCrafts a benchmark, SciToolBench, spanning five scientific domains to evaluate LLMs' abilities with tool assistance.
     </details>


     <details>
          <summary>Abstract</summary>
          Scientific reasoning poses an excessive challenge for even the most advanced Large Language Models (LLMs). To make this task more practical and solvable for LLMs, we introduce a new task setting named tool-augmented scientific reasoning. This setting supplements LLMs with scalable toolsets, and shifts the focus from pursuing an omniscient problem solver to a proficient tool-user. To facilitate the research of such setting, we construct a tool-augmented training corpus named MathFunc which encompasses over 30,000 samples and roughly 6,000 tools. Building on MathFunc, we develop SciAgent to retrieve, understand and, if necessary, use tools for scientific problem solving. Additionally, we craft a benchmark, SciToolBench, spanning five scientific domains to evaluate LLMs' abilities with tool assistance. Extensive experiments on SciToolBench confirm the effectiveness of SciAgent. Notably, SciAgent-Mistral-7B surpasses other LLMs with the same size by more than 13% in absolute accuracy. Furthermore, SciAgent-DeepMath-7B shows much superior performance than ChatGPT.
     </details>

292. **SymBa: Symbolic Backward Chaining for Structured Natural Language Reasoning** [[pdf]](https://arxiv.org/abs/2402.12806v2) `2024-02-20` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a novel backward chaining framework, SymBa (Symbolic Backward Chaining), in which a symbolic solver controls the whole proof process, and an LLM searches for the relevant natural language premises and translates them into a symbolic form for the solver.
     </details>


     <details>
          <summary>Abstract</summary>
          While Large Language Models (LLMs) have demonstrated remarkable reasoning ability, providing a structured, explainable proof to ensure explainability, i.e. structured reasoning, still remains challenging. Among two directions of structured reasoning, we specifically focus on backward chaining, where the query is recursively decomposed to subgoals by applying inference rules. We point out that current popular backward chaining implementations (Least-to-most prompting and LAMBADA) fail to implement the necessary features of backward chaining, such as arbitrary-depth recursion and binding propagation. To this end, we propose a novel backward chaining framework, SymBa (Symbolic Backward Chaining). In SymBA, a symbolic solver controls the whole proof process, and an LLM searches for the relevant natural language premises and translates them into a symbolic form for the solver. By this LLM-solver integration, while producing a completely structured proof that is symbolically verified, SymBa achieves significant improvement in performance, proof accuracy, and efficiency in diverse structured reasoning benchmarks compared to baselines.
     </details>

293. **Can LLMs Compute with Reasons?** [[pdf]](http://arxiv.org/abs/2402.12080) `2024-02-19` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The goal is to provide a framework that empowers SLMs to approach the level of logic-based applications achieved by high-parameter models, potentially benefiting any language model.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) often struggle with complex mathematical tasks, prone to "hallucinating" incorrect answers due to their reliance on statistical patterns. This limitation is further amplified in average Small LangSLMs with limited context and training data. To address this challenge, we propose an "Inductive Learning" approach utilizing a distributed network of SLMs. This network leverages error-based learning and hint incorporation to refine the reasoning capabilities of SLMs. Our goal is to provide a framework that empowers SLMs to approach the level of logic-based applications achieved by high-parameter models, potentially benefiting any language model. Ultimately, this novel concept paves the way for bridging the logical gap between humans and LLMs across various fields.
     </details>

294. **Learning From Failure: Integrating Negative Examples when Fine-tuning Large Language Models as Agents** [[pdf]](https://arxiv.org/abs/2402.11651v2) `2024-02-18` (6 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is argued that unsuccessful trajectories offer valuable insights, and LLMs can learn from these trajectories through appropriate quality control and fine-tuning strategies, and the first to demonstrate the value of negative trajectories and their application in agent-tunning scenarios is demonstrated.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have achieved success in acting as agents, which interact with environments through tools such as search engines. However, LLMs are optimized for language generation instead of tool use during training or alignment, limiting their effectiveness as agents. To resolve this problem, previous work has first collected interaction trajectories between LLMs and environments, using only trajectories that successfully finished the task to fine-tune smaller models, making fine-tuning data scarce and acquiring it both difficult and costly. Discarding failed trajectories also leads to significant wastage of data and resources and limits the possible optimization paths during fine-tuning. In this paper, we argue that unsuccessful trajectories offer valuable insights, and LLMs can learn from these trajectories through appropriate quality control and fine-tuning strategies. By simply adding a prefix or suffix that tells the model whether to generate a successful trajectory during training, we improve model performance by a large margin on mathematical reasoning, multi-hop question answering, and strategic question answering tasks. We further analyze the inference results and find that our method provides a better trade-off between valuable information and errors in unsuccessful trajectories. To our knowledge, we are the first to demonstrate the value of negative trajectories and their application in agent-tunning scenarios. Our findings offer guidance for developing better agent-tuning methods and low-resource data usage techniques.
     </details>

295. **Orca-Math: Unlocking the potential of SLMs in Grade School Math** [[pdf]](http://arxiv.org/abs/2402.14830) `2024-02-16` (31 cite) (3 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Mathematical word problem-solving has long been recognized as a complex task for small language models (SLMs). A recent study hypothesized that the smallest model size, needed to achieve over 80% accuracy on the GSM8K benchmark, is 34 billion parameters. To reach this level of performance with smaller models, researcher often train SLMs to generate Python code or use tools to help avoid calculation errors. Additionally, they employ ensembling, where outputs of up to 100 model runs are combined to arrive at a more accurate result. Result selection is done using consensus, majority vote or a separate a verifier model used in conjunction with the SLM. Ensembling provides a substantial boost in accuracy but at a significant cost increase with multiple calls to the model (e.g., Phi-GSM uses top-48 to boost the performance from 68.2 to 81.5).   In this work, we present Orca-Math, a 7-billion-parameter SLM based on the Mistral-7B, which achieves 86.81% on GSM8k without the need for multiple model calls or the use of verifiers, code execution or any other external tools. Our approach has the following key elements: (1) A high quality synthetic dataset of 200K math problems created using a multi-agent setup where agents collaborate to create the data, (2) An iterative learning techniques that enables the SLM to practice solving problems, receive feedback on its solutions and learn from preference pairs incorporating the SLM solutions and the feedback. When trained with Supervised Fine-Tuning alone, Orca-Math achieves 81.50% on GSM8k pass@1 metric. With iterative preference learning, Orca-Math achieves 86.81% pass@1. Orca-Math surpasses the performance of significantly larger models such as LLAMA-2-70B, WizardMath-70B, Gemini-Pro, ChatGPT-3.5. It also significantly outperforms other smaller models while using much smaller data (hundreds of thousands vs. millions of problems).
     </details>

296. **AutoTutor meets Large Language Models: A Language Model Tutor with Rich Pedagogy and Guardrails** [[pdf]](https://arxiv.org/abs/2402.09216v3) `2024-02-14` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper creates a sample end-to-end tutoring system named MWPTutor, which uses LLMs to fill in the state space of a pre-defined finite state transducer, and shows that this hybrid approach achieves a better overall tutoring score than an instructed, but otherwise free-form, GPT-4.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have found several use cases in education, ranging from automatic question generation to essay evaluation. In this paper, we explore the potential of using Large Language Models (LLMs) to author Intelligent Tutoring Systems. A common pitfall of LLMs is their straying from desired pedagogical strategies such as leaking the answer to the student, and in general, providing no guarantees. We posit that while LLMs with certain guardrails can take the place of subject experts, the overall pedagogical design still needs to be handcrafted for the best learning results. Based on this principle, we create a sample end-to-end tutoring system named MWPTutor, which uses LLMs to fill in the state space of a pre-defined finite state transducer. This approach retains the structure and the pedagogy of traditional tutoring systems that has been developed over the years by learning scientists but brings in additional flexibility of LLM-based approaches. Through a human evaluation study on two datasets based on math word problems, we show that our hybrid approach achieves a better overall tutoring score than an instructed, but otherwise free-form, GPT-4. MWPTutor is completely modular and opens up the scope for the community to improve its performance by improving individual modules or using different teaching strategies that it can follow.
     </details>

297. **FGeo-DRL: Deductive Reasoning for Geometric Problems through Deep Reinforcement Learning** [[pdf]](http://arxiv.org/abs/2402.09051) `2024-02-14` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A neural-symbolic system, named FGeo-DRL, is built to automatically perform human-like geometric deductive reasoning, capable of autonomously learning problem-solving methods from the feedback of a formalized environment, without the need for human supervision.
     </details>


     <details>
          <summary>Abstract</summary>
          The human-like automatic deductive reasoning has always been one of the most challenging open problems in the interdiscipline of mathematics and artificial intelligence. This paper is the third in a series of our works. We built a neural-symbolic system, called FGeoDRL, to automatically perform human-like geometric deductive reasoning. The neural part is an AI agent based on reinforcement learning, capable of autonomously learning problem-solving methods from the feedback of a formalized environment, without the need for human supervision. It leverages a pre-trained natural language model to establish a policy network for theorem selection and employ Monte Carlo Tree Search for heuristic exploration. The symbolic part is a reinforcement learning environment based on geometry formalization theory and FormalGeo, which models GPS as a Markov Decision Process. In this formal symbolic system, the known conditions and objectives of the problem form the state space, while the set of theorems forms the action space. Leveraging FGeoDRL, we have achieved readable and verifiable automated solutions to geometric problems. Experiments conducted on the formalgeo7k dataset have achieved a problem-solving success rate of 86.40%. The project is available at https://github.com/PersonNoName/FGeoDRL.
     </details>

298. **Beyond Lines and Circles: Unveiling the Geometric Reasoning Gap in Large Language Models** [[pdf]](http://arxiv.org/abs/2402.03877) `2024-02-14` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces a framework that formulates an LLMs-based multi-agents system that enhances their existing reasoning potential by conducting an internal dialogue and underscores LLMs' current limitations in geometric reasoning and improves geometric reasoning capabilities through self-correction, collaboration, and diverse role specializations.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) demonstrate ever-increasing abilities in mathematical and algorithmic tasks, yet their geometric reasoning skills are underexplored. We investigate LLMs' abilities in constructive geometric problem-solving one of the most fundamental steps in the development of human mathematical reasoning. Our work reveals notable challenges that the state-of-the-art LLMs face in this domain despite many successes in similar areas. LLMs exhibit biases in target variable selection and struggle with 2D spatial relationships, often misrepresenting and hallucinating objects and their placements. To this end, we introduce a framework that formulates an LLMs-based multi-agents system that enhances their existing reasoning potential by conducting an internal dialogue. This work underscores LLMs' current limitations in geometric reasoning and improves geometric reasoning capabilities through self-correction, collaboration, and diverse role specializations.
     </details>

299. **FGeo-TP: A Language Model-Enhanced Solver for Geometry Problems** [[pdf]](http://arxiv.org/abs/2402.09047) `2024-02-14` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduced FGeo-TP (Theorem Predictor), which utilizes the language model to predict theorem sequences for solving geometry problems, and compared the effectiveness of various Transformer architectures in theorem prediction, implementing pruning in the search process of FGPS, thereby improving its performance in solving geometry problems.
     </details>


     <details>
          <summary>Abstract</summary>
          The application of contemporary artificial intelligence techniques to address geometric problems and automated deductive proof has always been a grand challenge to the interdiscipline field of mathematics and artificial Intelligence. This is the fourth article in a series of our works, in our previous work, we established of a geometric formalized system known as FormalGeo. Moreover we annotated approximately 7000 geometric problems, forming the FormalGeo7k dataset. Despite the FGPS (Formal Geometry Problem Solver) can achieve interpretable algebraic equation solving and human-like deductive reasoning, it often experiences timeouts due to the complexity of the search strategy. In this paper, we introduced FGeo-TP (Theorem Predictor), which utilizes the language model to predict theorem sequences for solving geometry problems. We compared the effectiveness of various Transformer architectures, such as BART or T5, in theorem prediction, implementing pruning in the search process of FGPS, thereby improving its performance in solving geometry problems. Our results demonstrate a significant increase in the problem-solving rate of the language model-enhanced FGeo-TP on the FormalGeo7k dataset, rising from 39.7% to 80.86%. Furthermore, FGeo-TP exhibits notable reductions in solving time and search steps across problems of varying difficulty levels.
     </details>

300. **FormalGeo: An Extensible Formalized Framework for Olympiad Geometric Problem Solving** [[pdf]](http://arxiv.org/abs/2310.18021) `2024-02-14` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper has constructed a consistent formal plane geometry system that will serve as a crucial bridge between IMO-level plane geometry challenges and readable AI automated reasoning, and has been able to seamlessly integrate modern AI models with this formal system.
     </details>


     <details>
          <summary>Abstract</summary>
          This is the first paper in a series of work we have accomplished over the past three years. In this paper, we have constructed a consistent formal plane geometry system. This will serve as a crucial bridge between IMO-level plane geometry challenges and readable AI automated reasoning. Within this formal framework, we have been able to seamlessly integrate modern AI models with our formal system. AI is now capable of providing deductive reasoning solutions to IMO-level plane geometry problems, just like handling other natural languages, and these proofs are readable, traceable, and verifiable. We propose the geometry formalization theory (GFT) to guide the development of the geometry formal system. Based on the GFT, we have established the FormalGeo, which consists of 88 geometric predicates and 196 theorems. It can represent, validate, and solve IMO-level geometry problems. we also have crafted the FGPS (formal geometry problem solver) in Python. It serves as both an interactive assistant for verifying problem-solving processes and an automated problem solver. We've annotated the formalgeo7k and formalgeo-imo datasets. The former contains 6,981 (expand to 133,818 through data augmentation) geometry problems, while the latter includes 18 (expand to 2,627 and continuously increasing) IMO-level challenging geometry problems. All annotated problems include detailed formal language descriptions and solutions. Implementation of the formal system and experiments validate the correctness and utility of the GFT. The backward depth-first search method only yields a 2.42% problem-solving failure rate, and we can incorporate deep learning techniques to achieve lower one. The source code of FGPS and datasets are available at https://github.com/BitSecret/FGPS.
     </details>

301. **InternLM-Math: Open Math Large Language Models Toward Verifiable Reasoning** [[pdf]](http://arxiv.org/abs/2402.06332) `2024-02-09` `Lean` (33 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduces and open-source their math reasoning LLMs InternLM-Math which is continue pre-trained from InternLM2 and unify chain-of-thought reasoning, reward modeling, formal reasoning, data augmentation, and code interpreter in a unified seq2seq format and supervise the authors' model to be a versatile math reasoner, verifier, prover, and augmenter.
     </details>


     <details>
          <summary>Abstract</summary>
          The math abilities of large language models can represent their abstract reasoning ability. In this paper, we introduce and open-source our math reasoning LLMs InternLM-Math which is continue pre-trained from InternLM2. We unify chain-of-thought reasoning, reward modeling, formal reasoning, data augmentation, and code interpreter in a unified seq2seq format and supervise our model to be a versatile math reasoner, verifier, prover, and augmenter. These abilities can be used to develop the next math LLMs or self-iteration. InternLM-Math obtains open-sourced state-of-the-art performance under the setting of in-context learning, supervised fine-tuning, and code-assisted reasoning in various informal and formal benchmarks including GSM8K, MATH, Hungary math exam, MathBench-ZH, and MiniF2F. Our pre-trained model achieves 30.3 on the MiniF2F test set without fine-tuning. We further explore how to use LEAN to solve math problems and study its performance under the setting of multi-task learning which shows the possibility of using LEAN as a unified platform for solving and proving in math. Our models, codes, and data are released at \url{https://github.com/InternLM/InternLM-Math}.
     </details>

302. **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models** [[pdf]](http://arxiv.org/abs/2402.03300) `2024-02-06` `Isabelle` (123 cite) (7 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Group Relative Policy Optimization (GRPO), a variant of Proximal Policy Optimization (PPO), that enhances mathematical reasoning abilities while concurrently optimizing the memory usage of PPO is introduced.
     </details>


     <details>
          <summary>Abstract</summary>
          Mathematical reasoning poses a significant challenge for language models due to its complex and structured nature. In this paper, we introduce DeepSeekMath 7B, which continues pre-training DeepSeek-Coder-Base-v1.5 7B with 120B math-related tokens sourced from Common Crawl, together with natural language and code data. DeepSeekMath 7B has achieved an impressive score of 51.7% on the competition-level MATH benchmark without relying on external toolkits and voting techniques, approaching the performance level of Gemini-Ultra and GPT-4. Self-consistency over 64 samples from DeepSeekMath 7B achieves 60.9% on MATH. The mathematical reasoning capability of DeepSeekMath is attributed to two key factors: First, we harness the significant potential of publicly available web data through a meticulously engineered data selection pipeline. Second, we introduce Group Relative Policy Optimization (GRPO), a variant of Proximal Policy Optimization (PPO), that enhances mathematical reasoning abilities while concurrently optimizing the memory usage of PPO.
     </details>

303. **Large Language Models as an Indirect Reasoner: Contrapositive and Contradiction for Automated Reasoning** [[pdf]](http://arxiv.org/abs/2402.03667) `2024-02-05` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel Indirect Reasoning (IR) method that employs the logic of contrapositives and contradictions to tackle IR tasks such as factual reasoning and mathematic proof to strengthen the reasoning power of LLMs is proposed.
     </details>


     <details>
          <summary>Abstract</summary>
          Recently, increasing attention has been focused drawn on to improve the ability of Large Language Models (LLMs) to perform complex reasoning. However, previous methods, such as Chain-of-Thought and Self-Consistency, mainly follow Direct Reasoning (DR) frameworks, so they will meet difficulty in solving numerous real-world tasks which can hardly be solved via DR. Therefore, to strengthen the reasoning power of LLMs, this paper proposes a novel Indirect Reasoning (IR) method that employs the logic of contrapositives and contradictions to tackle IR tasks such as factual reasoning and mathematic proof. Specifically, our methodology comprises two steps. Firstly, we leverage the logical equivalence of contrapositive to augment the data and rules to enhance the comprehensibility of LLMs. Secondly, we design a set of prompt templates to trigger LLMs to conduct IR based on proof by contradiction that is logically equivalent to the original DR process. Our IR method is simple yet effective and can be straightforwardly integrated with existing DR methods to further boost the reasoning abilities of LLMs. The experimental results on popular LLMs, such as GPT-3.5-turbo and Gemini-pro, show that our IR method enhances the overall accuracy of factual reasoning by 27.33% and mathematical proof by 31.43%, when compared with traditional DR methods. Moreover, the methods combining IR and DR significantly outperform the methods solely using IR or DR, further demonstrating the effectiveness of our strategy.
     </details>

304. **REFINER: Reasoning Feedback on Intermediate Representations** [[pdf]](http://arxiv.org/abs/2304.01904) `2024-02-04` (108 cite) (16 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          REFINER is a framework for finetuning LMs to explicitly generate intermediate reasoning steps while interacting with a critic model that provides automated feedback on the reasoning that provides structured feedback that the reasoning LM uses to iteratively improve its intermediate arguments.
     </details>


     <details>
          <summary>Abstract</summary>
          Language models (LMs) have recently shown remarkable performance on reasoning tasks by explicitly generating intermediate inferences, e.g., chain-of-thought prompting. However, these intermediate inference steps may be inappropriate deductions from the initial context and lead to incorrect final predictions. Here we introduce REFINER, a framework for finetuning LMs to explicitly generate intermediate reasoning steps while interacting with a critic model that provides automated feedback on the reasoning. Specifically, the critic provides structured feedback that the reasoning LM uses to iteratively improve its intermediate arguments. Empirical evaluations of REFINER on three diverse reasoning tasks show significant improvements over baseline LMs of comparable scale. Furthermore, when using GPT-3.5 or ChatGPT as the reasoner, the trained critic significantly improves reasoning without finetuning the reasoner. Finally, our critic model is trained without expensive human-in-the-loop data but can be substituted with humans at inference time.
     </details>

305. **Multi-step Problem Solving Through a Verifier: An Empirical Analysis on Model-induced Process Supervision** [[pdf]](http://arxiv.org/abs/2402.02658) `2024-02-04` (5 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Model-induced Process Supervision (MiPS), a novel method for automating data curation, is introduced and verification focusing on high predicted scores of the verifier shall be preferred over that of low predicted scores, contrary to prior work.
     </details>


     <details>
          <summary>Abstract</summary>
          Process supervision, using a trained verifier to evaluate the intermediate steps generated by reasoner, has demonstrated significant improvements in multi-step problem solving. In this paper, to avoid expensive human annotation effort on the verifier training data, we introduce Model-induced Process Supervision (MiPS), a novel method for automating data curation. MiPS annotates an intermediate step by sampling completions of this solution through the reasoning model, and obtaining an accuracy defined as the proportion of correct completions. Errors in the reasoner would cause MiPS to underestimate the accuracy of intermediate steps, therefore, we suggest and empirically show that verification focusing on high predicted scores of the verifier shall be preferred over that of low predicted scores, contrary to prior work. Our approach significantly improves the performance of PaLM 2 on math and coding tasks (accuracy +0.67% on GSM8K, +4.16% on MATH, +0.92% on MBPP compared with an output supervision trained verifier). Additionally, our study demonstrates that the verifier exhibits strong generalization ability across different reasoning models.
     </details>
