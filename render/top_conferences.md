# Top Conferences 



1. **OlympicArena: Benchmarking Multi-discipline Cognitive Reasoning for Superintelligent AI** [[pdf]](https://arxiv.org/abs/2406.12753) `NeurIPS 2024` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work argues that the challenges in Olympic competition problems are ideal for evaluating AI's cognitive reasoning due to their complexity and interdisciplinary nature, which are essential for tackling complex scientific challenges and facilitating discoveries.
     </details>


     <details>
          <summary>Abstract</summary>
          The evolution of Artificial Intelligence (AI) has been significantly accelerated by advancements in Large Language Models (LLMs) and Large Multimodal Models (LMMs), gradually showcasing potential cognitive reasoning abilities in problem-solving and scientific discovery (i.e., AI4Science) once exclusive to human intellect. To comprehensively evaluate current models' performance in cognitive reasoning abilities, we introduce OlympicArena, which includes 11,163 bilingual problems across both text-only and interleaved text-image modalities. These challenges encompass a wide range of disciplines spanning seven fields and 62 international Olympic competitions, rigorously examined for data leakage. We argue that the challenges in Olympic competition problems are ideal for evaluating AI's cognitive reasoning due to their complexity and interdisciplinary nature, which are essential for tackling complex scientific challenges and facilitating discoveries. Beyond evaluating performance across various disciplines using answer-only criteria, we conduct detailed experiments and analyses from multiple perspectives. We delve into the models' cognitive reasoning abilities, their performance across different modalities, and their outcomes in process-level evaluations, which are vital for tasks requiring complex reasoning with lengthy solutions. Our extensive evaluations reveal that even advanced models like GPT-4o only achieve a 39.97\%  overall accuracy (28.67\%  for mathematics and 29.71\%  for physics), illustrating current AI limitations in complex reasoning and multimodal integration. Through the OlympicArena, we aim to advance AI towards superintelligence, equipping it to address more complex challenges in science and beyond. We also provide a comprehensive set of resources to support AI research, including a benchmark dataset, an open-source annotation platform, a detailed evaluation tool, and a leaderboard with automatic submission features.
     </details>

2. **StrategyLLM: Large Language Models as Strategy Generators, Executors, Optimizers, and Evaluators for Problem Solving** [[pdf]](https://arxiv.org/abs/2311.08803) `NeurIPS 2024` (3 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A comprehensive framework, StrategyLLM, allowing LLMs to perform inductive reasoning, deriving general strategies from specific task instances, and deductive reasoning, applying these general strategies to particular task examples, for constructing generalizable and consistent few-shot prompts.
     </details>


     <details>
          <summary>Abstract</summary>
          Most existing prompting methods suffer from the issues of generalizability and consistency, as they often rely on instance-specific solutions that may not be applicable to other instances and lack task-level consistency across the selected few-shot examples. To address these limitations, we propose a comprehensive framework, StrategyLLM, allowing LLMs to perform inductive reasoning, deriving general strategies from specific task instances, and deductive reasoning, applying these general strategies to particular task examples, for constructing generalizable and consistent few-shot prompts. It employs four LLM-based agents: strategy generator, executor, optimizer, and evaluator, working together to generate, evaluate, and select promising strategies for a given task. Experimental results demonstrate that StrategyLLM outperforms the competitive baseline CoT-SC that requires human-annotated solutions on 13 datasets across 4 challenging tasks without human involvement, including math reasoning (34.2\% $\rightarrow$ 38.8\%), commonsense reasoning (70.3\% $\rightarrow$ 72.5\%), algorithmic reasoning (73.7\% $\rightarrow$ 85.0\%), and symbolic reasoning (30.0\% $\rightarrow$ 79.2\%). Further analysis reveals that StrategyLLM is applicable to various LLMs and demonstrates advantages across numerous scenarios.
     </details>

3. **Calibrating Reasoning in Language Models with Internal Consistency** [[pdf]](https://arxiv.org/abs/2405.18711) `NeurIPS 2024` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The results demonstrate the potential of using internal representations for self-evaluation of LLMs by up-weighting reasoning paths with high internal consistency, resulting in a significant boost in reasoning performance.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have demonstrated impressive capabilities in various reasoning tasks, aided by techniques like chain-of-thought (CoT) prompting that elicits verbalized reasoning. However, LLMs often generate text with obvious mistakes and contradictions, raising doubts about their ability to robustly process and utilize generated rationales. In this work, we investigate CoT reasoning in LLMs through the lens of internal representations, focusing on how these representations are influenced by generated rationales. Our preliminary analysis reveals that while generated rationales improve answer accuracy, inconsistencies emerge between the model's internal representations in middle layers and those in final layers, potentially undermining the reliability of their reasoning processes. To address this, we propose internal consistency as a measure of the model's confidence by examining the agreement of latent predictions decoded from intermediate layers. Extensive empirical studies across different models and datasets demonstrate that internal consistency effectively distinguishes between correct and incorrect reasoning paths. Motivated by this, we propose a new approach to calibrate CoT reasoning by up-weighting reasoning paths with high internal consistency, resulting in a significant boost in reasoning performance. Further analysis uncovers distinct patterns in attention and feed-forward modules across layers, providing insights into the emergence of internal inconsistency. In summary, our results demonstrate the potential of using internal representations for self-evaluation of LLMs.
     </details>

4. **Autoformalize Mathematical Statements by Symbolic Equivalence and Semantic Consistency** [[pdf]](https://neurips.cc/virtual/2024/poster/96359) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Autoformalization, the task of automatically translating natural language descriptions into a formal language, poses a significant challenge across various domains, especially in mathematics. Recent advancements in large language models (LLMs) have unveiled their promising capabilities to formalize even competition-level math problems. However, we observe a considerable discrepancy between pass@1 and pass@k accuracies in LLM-generated formalizations. To address this gap, we introduce a novel framework that scores and selects the best result from k autoformalization candidates based on two complementary self-consistency methods: symbolic equivalence and semantic consistency. Elaborately, symbolic equivalence identifies the logical homogeneity among autoformalization candidates using automated theorem provers, and semantic consistency evaluates the preservation of the original meaning by informalizing the candidates and computing the similarity between the embeddings of the original and informalized texts. Our extensive experiments on the MATH and miniF2F datasets demonstrate that our approach significantly enhances autoformalization accuracy, achieving up to 0.22-1.35x relative improvements across various LLMs and baseline methods.
     </details>

5. **Benchmarking the Reasoning Robustness against Noisy Rationales in Chain-of-thought Prompting** [[pdf]](https://neurips.cc/virtual/2024/poster/95956) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          This paper investigates an under-explored challenge in large language models (LLMs): chain-of-thought prompting with noisy rationales—irrelevant or inaccurate reasoning steps—despite advancements in in-context learning. We construct the NoRa dataset, specifically designed to evaluate LLMs’ robustness to noisy rationales, based on which we reveal a widespread vulnerability among LLMs to such noise, with limited efficacy from existing reasoning methods. To combat this, we propose the contrastive denoising with noisy chain-of-thought (CD-CoT) method to enhance denoising-reasoning capabilities by contrasting noisy rationales with only one clean rationale, thereby advancing the robustness of LLMs in reasoning.
     </details>

6. **Counterfactual PPO Enhanced Shared Reflector for LLM-based Multi-agent Collaboration** [[pdf]](https://neurips.cc/virtual/2024/poster/93147) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Benefiting from the powerful language expression and planning capabilities of Large Language Models (LLMs), LLM-based autonomous agents achieve promising performance in various downstream tasks. Recently, based on the development of single-agent systems, researchers propose to construct LLM-based multi-agent systems to tackle more complicated tasks. In this paper, we propose a novel framework, named COPPER, to enhance the collaboration ability of multi-agent systems through learnable self-reflection mechanism. To improve the quality of reflections, we propose to fine-tune a shared reflector, which automatically tunes the prompts of actor models using our counterfactual PPO mechanism. On the one hand, we propose counterfactual rewards to assess the contribution of a single agent’s reflection within the system, alleviating the credit assignment problem. On the other hand, we propose to train a shared reflector, which enables the reflector to personalize generated reflections according to agent roles, while reducing the computational resource requirements and improving training stability. We conduct experiments on three datasets to evaluate the performance of multi-agent systems in multi-hop question answering, mathematics, and chess scenarios. Experimental results show that COPPER possesses stronger reflection capabilities and exhibits excellent generalization performance across different actor models.
     </details>

7. **Diffusion of Thought: Chain-of-Thought Reasoning in Diffusion Language Models** [[pdf]](https://neurips.cc/virtual/2024/poster/95935) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Recently, diffusion models have garnered significant interest in the field of text processing due to their many potential advantages compared to conventional autoregressive models.In this work, we propose Diffusion-of-Thought (DoT),  a novel approach that integrates diffusion models with Chain-of-Thought, a well-established technique for improving the reasoning ability of autoregressive language models. In contrast to autoregressive language models that make decisions in a left-to-right, token-by-token manner, DoT allows reasoning steps to diffuse over time through a diffusion language model and offers greater flexibility in trading-off computation for reasoning performance. Our experimental results demonstrate the effectiveness of DoT in multi-digit multiplication, boolean logic, and grade school math problems. In addition to that, DoT showcases promising self-correction abilities and benefits from existing reasoning-enhancing techniques like self-consistency decoding. Our findings contribute to the understanding and development of reasoning with diffusion language models.
     </details>

8. **Learning Goal-Conditioned Representations in Reward Models for Aligning Language Models** [[pdf]](https://neurips.cc/virtual/2024/poster/95067) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Representation learning is important for the success of Reinforcement Learning (RL) algorithms, but has been less explored for Language Model (LM) alignment with Reinforcement learning from Human Feedback (RLHF).In this work, we present a simple yet effective approach to improve the representations learned by reward models for aligning LMs.Our method uses a contrastive loss that encourages reward models to learn goal-conditioned representations which encode the expected reward at intermediate steps of the input sequence.By enforcing this loss on representations from intermediate steps, we can capture which trajectories are likely to reach a desired goal (e.g., correct solution or helpful response) at different points in the sequence.This method is flexible enough to support different kinds of alignment data and does not require extra annotations.We demonstrate the effectiveness of this approach in 2 domains: mathematical reasoning and natural language alignment.On math benchmarks, such as GSM8k, we show that our approach improves the reward model's ability to discern between correct/incorrect solutions, increasing AUROC score by up to 0.11 points, and that the learned representations can help prune undesirable generations.Using this reward model to improve a policy model via RLHF yields accuracy gains of 1.7\% across several math benchmarks when compared to a standard preference-ranking trained reward model.Additionally, we show the that learned representations can be used to steer LMs toward generations that are more aligned with human preferences via guided decoding.Overall, our study underscores the potential of incorporating feedback signals in RLHF frameworks via learned representations, which we believe is a promising avenue for improving the alignment of LLMs.
     </details>

9. **MathPile: A Billion-Token-Scale Pretraining Corpus for Math** [[pdf]](https://neurips.cc/virtual/2024/poster/97685) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          High-quality, large-scale corpora are the cornerstone of building foundation models. In this work, we introduce MathPile, a diverse and high-quality math-centric corpus comprising about 9.5 billion tokens. Throughout its creation, we adhered to the principle of “less is more”, firmly believing in the supremacy of data quality over quantity, even in the pre-training phase. Our meticulous data collection and processing efforts included a complex suite of preprocessing, prefiltering, language identification, cleaning, filtering, and deduplication, ensuring the high quality of our corpus. Furthermore, we performed data contamination detection on downstream benchmark test sets to eliminate duplicates and conducted continual pre-training experiments, booting the performance on common mathematical reasoning benchmarks. We aim for our MathPile to boost language models’ mathematical reasoning and plan to open-source its different versions and processing scripts to advance the field.
     </details>

10. **Multi-language Diversity Benefits Autoformalization** [[pdf]](https://neurips.cc/virtual/2024/poster/96799) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Autoformalization is the task of translating natural language materials into machine-verifiable formalisations. Progress in autoformalization research is hindered by the lack of a sizeable dataset consisting of informal-formal pairs expressing the same essence. Existing methods tend to circumvent this challenge by manually curating small corpora or using few-shot learning with large language models. But these methods suffer from data scarcity and formal language acquisition difficulty. In this work, we create mma, a large, flexible, multi-language, and multi-domain dataset of informal-formal pairs, by using a language model to translate in the reverse direction, that is, from formal mathematical statements into corresponding informal ones. Experiments show that language models fine-tuned on mma can produce up to $29-31$\% of statements acceptable with minimal corrections on the miniF2F and ProofNet benchmarks, up from $0$\% with the base model. We demonstrate that fine-tuning on multi-language formal data results in more capable autoformalization models even on single-language tasks.
     </details>

11. **Neuro-Symbolic Data Generation for Math Reasoning** [[pdf]](https://neurips.cc/virtual/2024/poster/96151) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          A critical question about Large Language Models (LLMs) is whether their apparent deficiency in mathematical reasoning is inherent, or merely a result of insufficient exposure to high-quality mathematical data. To explore this, we developed an automated method for generating high-quality, supervised mathematical datasets. The method carefully mutates existing math problems, ensuring both diversity and validity of the newly generated problems. This is achieved by a neuro-symbolic data generation framework combining the intuitive informalization strengths of LLMs, and the precise symbolic reasoning of math solvers along with projected Markov chain Monte Carlo sampling in the highly-irregular symbolic space.Empirical experiments demonstrate the high quality of data generated by the proposed method, and that the LLMs, specifically LLaMA-2 and Mistral, when realigned with the generated data, surpass their state-of-the-art counterparts.
     </details>

12. **Not All Tokens Are What You Need for Pretraining** [[pdf]](https://neurips.cc/virtual/2024/poster/96931) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Previous language model pre-training methods have uniformly applied a next-token prediction loss to all training tokens. Challenging this norm, we posit that ''Not all tokens in a corpus are equally important for language model training''. Our initial analysis examines token-level training dynamics of language model, revealing distinct loss patterns for different tokens. Leveraging these insights, we introduce a new language model called Rho-1. Unlike traditional LMs that learn to predict every next token in a corpus, Rho-1 employs Selective Language Modeling (SLM), which selectively trains on useful tokens that aligned with the desired distribution. This approach involves scoring pretraining tokens using a reference model, and then training the language model with a focused loss on tokens with higher scores. When continual pretraining on 15B OpenWebMath corpus, Rho-1 yields an absolute improvement in few-shot accuracy of up to 30% in 9 math tasks. After fine-tuning, Rho-1-1B and 7B achieved state-of-the-art results of 40.6% and 51.8% on MATH dataset, respectively - matching DeepSeekMath with only 3% of the pretraining tokens. Furthermore, when pretraining on 80B general tokens, Rho-1 achieves 6.8% average enhancement across 15 diverse tasks, increasing both efficiency and performance of the language model pre-training.
     </details>

13. **Pretrained Large Language Models Use Fourier Features to Compute Addition** [[pdf]](https://neurips.cc/virtual/2024/poster/94033) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Pre-trained large language models (LLMs) exhibit impressive mathematical reasoning capabilities, yet how they compute basic arithmetic, such as addition, remains unclear. This paper shows that pre-trained LLMs add numbers using Fourier features---dimensions in the hidden state that represent numbers via a set of features sparse in the frequency domain. Within the model, MLP and attention layers use Fourier features in complementary ways: MLP layers primarily approximate the magnitude of the answer using low-frequency features, while attention layers primarily perform modular addition (e.g., computing whether the answer is even or odd) using high-frequency features.Pre-training is crucial for this mechanism: models trained from scratch to add numbers only exploit low-frequency features, leading to lower accuracy.Introducing pre-trained token embeddings to a randomly initialized model rescues its performance.Overall, our analysis demonstrates that appropriate pre-trained representations (e.g., Fourier features) can unlock the ability of Transformers to learn precise mechanisms for algorithmic tasks.
     </details>

14. **Recursive Introspection: Teaching Foundation Model Agents How to Self-Improve** [[pdf]](https://neurips.cc/virtual/2024/poster/96089) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          A central piece in enabling intelligent agentic behavior in foundation models is to make them capable of introspecting upon their behavior, to reason and correct their mistakes. Even strong proprietary large language models (LLMs) do not exhibit the ability of continually improving their responses sequentially, even in scenarios where they are explicitly told that they are making a mistake. In this paper, we develop $\textbf{RISE}$: $\textbf{R}$ecursive $\textbf{I}$ntro$\textbf{s}$p$\textbf{e}$ction, an approach for fine-tuning LLMs to introduce this ability. Our approach prescribes an iterative fine-tuning procedure, which attempts to teach the model how to alter its response after having seen previously unsuccessful attempts to solve a problem with additional environment feedback. RISE poses fine-tuning for a single-turn problem as solving a multi-turn Markov decision process (MDP), where the initial state is the prompt. Inspired by principles in online imitation learning, we derive effective strategies to dictate multi-turn data collection and training so as to imbue in an LLM the capability to recursively detect and correct its previous mistakes in subsequent iterations. Our experiments show that $\textbf{RISE}$ enables 7B Llama2 and Mistral models to improve themselves with more turns on math reasoning tasks, outperforming several single-turn strategies given an equal amount of inference-time computation. Our analysis shows that RISE makes meaningful improvements to responses to arrive at the correct solution for challenging prompts, without disrupting one-turn abilities.
     </details>

15. **SciInstruct: a Self-Reflective Instruction Annotated Dataset for Training Scientific Language Models** [[pdf]](https://neurips.cc/virtual/2024/poster/97744) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have shown promise in assisting scientific discovery. However, such applications are currently limited by LLMs' deficiencies in understanding intricate scientific concepts, deriving symbolic equations, and solving advanced numerical calculations. To bridge these gaps, we introduce SciInstruct, a suite of scientific instructions for training scientific language models capable of college-level scientific reasoning. Central to our approach is a novel self-reflective instruction annotation framework to address the data scarcity challenge in the science domain. This framework leverages existing LLMs to generate step-by-step reasoning for unlabelled scientific questions, followed by a process of self-reflective critic-and-revise. Applying this framework, we curated a diverse and high-quality dataset encompassing physics, chemistry, math, and formal proofs. We analyze the curated SciInstruct from multiple interesting perspectives (e.g., domain, scale, source, question type, answer length, etc.). To verify the effectiveness of SciInstruct, we fine-tuned different language models with SciInstruct, i.e., ChatGLM3 (6B and 32B), Llama3-8b-Instruct, and Mistral-7B, enhancing their scientific and mathematical reasoning capabilities, without sacrificing the language understanding capabilities of the base model. We release code and SciInstruct at https://github.com/THUDM/SciGLM.
     </details>

16. **Solving Intricate Problems with Human-like Decomposition and Rethinking** [[pdf]](https://neurips.cc/virtual/2024/poster/95441) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          In this paper, we introduce a novel reasoning framework DeAR (Decompose-Analyze-Rethink) for large language models (LLMs) to conduct intricate reasoning. Our key idea is inspired by human cognitive reasoning, which refines complex problem-solving by breaking it down into sub-questions within a Reasoning Tree and then updating prior answers based on the responses to these sub-questions. In our framework, we propose a Decompose-Analyze-Rethink cycle, which gradually forms a reasoning tree guiding the reasoning process. Specifically, given the problem, the Decompose stage introduces a prompt-based method to break it into simpler sub-ones at subsequent tree nodes. Then, the Analyze stage generates and self-checks the rationales at the node level. Last, the Rethink stage updates the rationales of parent nodes based on its children's feedback. Our reasoning paradigm is more flexible than state-of-the-art methods including Tree-of-Thoughts (ToT), and Graph-of-Thoughts (GoT), as each branch is autonomously generated without fixed settings, and moreover, allows for timely and globally rationale correction throughout the entire process. We conduct extensive experiments on three reasoning benchmarks including ScienceQA, StrategyQA, and GSM8K. Experimental results show that our approach can significantly reduce logical errors and enhance the performance with different LLMs. Our codes are available at: https://anonymous.4open.science/r/Coarse-to-Fine-F216/.
     </details>

17. **Unsupervised Discovery of Formulas for Mathematical Constants** [[pdf]](https://neurips.cc/virtual/2024/poster/95491) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          In recent years, we are witnessing a rise of AI and machine learning methods for scientific discovery and hypothesis creation. Despite the strides in other fields of science, a persistent challenge lies in the creation of formulas for mathematical constants.In the landscape of formula creation, there is no straightforward ‘’distance metric'' between two samples that can guide progress. Formulas are either true or false, with no continuous adjustments that can enhance their correctness.The absence of a systematic method left the realm of formula discovery elusive for automated methods. In this work, we propose a systematic methodology for categorization, characterization, and pattern identification of such formulas. We demonstrate this methodology on Polynomial Continued Fraction formulas, which are ubiquitous in their intrinsic connections to mathematical constants, and generalize many mathematical functions and structures.We discover organizing metrics for the space of polynomial continued fractions. We test our methodology on a set of 1,768,900 such formulas, identifying many known formulas for mathematical constants, and discover previously unknown formulas for $\pi$, $\ln(2)$, Gauss, and Lemniscate constants. The uncovered patterns enable a direct generalization of individual formulas to infinite families, unveiling rich mathematical structures. This success paves the way towards a generative model that creates continued fractions fulfilling requested mathematical properties, potentially accelerating by orders of magnitude the rate of discovery of useful formulas.
     </details>

18. **When and How Does Synthetic Data Improve Reasoning Capabilities of Language Models?** [[pdf]](https://neurips.cc/virtual/2024/poster/96295) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Training on model-generated synthetic data is a promising approach for finetuning LLMs, but it remains unclear when it helps or hurts. In this paper, we investigate this for reasoning problems via an empirical study, followed by a theoretical formalization of our observations. First, we find that while the typical approach of finetuning a model on synthetic correct or positive problem-solution pairs generated by capable models offers modest performance gains, sampling more correct solutions from the finetuned learner doubles the sample efficiency of synthetic data. At the same time, training on model-generated positives can amplify various spurious  correlations, resulting in flat or even inverse scaling trends as the amount of data increases. Surprisingly, we find that several of these issues can be addressed if we also utilize negative responses, i.e. model-generated responses that are deemed incorrect via final answer checking. Crucially, these negatives must be constructed such that the training can appropriately recover the utility or credit of each intermediate step in the negative response. With this per-step scheme, we are able to attain consistent gains over only positive data, attaining performance similar to amplifying the amount of synthetic data by 8x. We show that training on per-step negatives can help to unlearn spurious correlations in the positive data, and is equivalent to advantage-weighted reinforcement learning (RL), implying that it inherits benefits of RL over imitating positive data alone.
     </details>

19. **Instance-adaptive Zero-shot Chain-of-Thought Prompting** [[pdf]](http://arxiv.org/abs/2409.20441) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces an instance-adaptive prompting algorithm as an alternative zero-shot CoT reasoning scheme by adaptively differentiating good and bad prompts, and proposes an instance-adaptive prompting strategy (IAP) for zero-shot CoT reasoning.
     </details>


     <details>
          <summary>Abstract</summary>
          Zero-shot Chain-of-Thought (CoT) prompting emerges as a simple and effective strategy for enhancing the performance of large language models (LLMs) in real-world reasoning tasks. Nonetheless, the efficacy of a singular, task-level prompt uniformly applied across the whole of instances is inherently limited since one prompt cannot be a good partner for all, a more appropriate approach should consider the interaction between the prompt and each instance meticulously. This work introduces an instance-adaptive prompting algorithm as an alternative zero-shot CoT reasoning scheme by adaptively differentiating good and bad prompts. Concretely, we first employ analysis on LLMs through the lens of information flow to detect the mechanism under zero-shot CoT reasoning, in which we discover that information flows from question to prompt and question to rationale jointly influence the reasoning results most. We notice that a better zero-shot CoT reasoning needs the prompt to obtain semantic information from the question then the rationale aggregates sufficient information from the question directly and via the prompt indirectly. On the contrary, lacking any of those would probably lead to a bad one. Stem from that, we further propose an instance-adaptive prompting strategy (IAP) for zero-shot CoT reasoning. Experiments conducted with LLaMA-2, LLaMA-3, and Qwen on math, logic, and commonsense reasoning tasks (e.g., GSM8K, MMLU, Causal Judgement) obtain consistent improvement, demonstrating that the instance-adaptive zero-shot CoT prompting performs better than other task-level methods with some curated prompts or sophisticated procedures, showing the significance of our findings in the zero-shot CoT reasoning mechanism.
     </details>

20. **On the Inductive Bias of Stacking Towards Improving Reasoning** [[pdf]](http://arxiv.org/abs/2409.19044) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          An intriguing phenomenon is discovered: MIDAS is not only training-efficient but surprisingly also has an inductive bias towards improving downstream tasks, especially tasks that require reasoning abilities like reading comprehension and math problems, despite having similar or slightly worse perplexity compared to baseline training.
     </details>


     <details>
          <summary>Abstract</summary>
          Given the increasing scale of model sizes, novel training strategies like gradual stacking have garnered interest. Stacking enables efficient training by gradually growing the depth of a model in stages and using layers from a smaller model in an earlier stage to initialize the next stage. Although efficient for training, the model biases induced by such growing approaches is largely unexplored. In this work, we examine this fundamental aspect of gradual stacking, going beyond its efficiency benefits. We propose a variant of gradual stacking called MIDAS and discover an intriguing phenomenon for this approach: MIDAS is not only training efficient, but surprisingly also has an inductive bias towards improving downstream tasks, especially tasks that require reasoning abilities, despite having similar or slightly worse perplexity compared to baseline training. To further analyze this inductive bias, we construct {\em reasoning primitives} – simple synthetic tasks that are building blocks for reasoning – and find that a model pretrained with stacking is significantly better than standard pretraining on these primitives, with and without fine-tuning. This provides stronger and more robust evidence for this inductive bias towards reasoning. Furthermore, we conjecture the underlying reason for this inductive bias by exploring the connection of stacking to looped models and provide strong supporting empirical analysis.
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

22. **ReConcile: Round-Table Conference Improves Reasoning via Consensus among Diverse LLMs** [[pdf]](https://aclanthology.org/2024.acl-long.381) `ACL 2024 Long Papers` (54 cite) (4 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Experiments demonstrate that ReConcile significantly improves LLMs' reasoning -- both individually and as a team -- surpassing prior single-agent and multi-agent baselines by up to 11.4% and even outperforming GPT-4 on three datasets.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) still struggle with natural language reasoning tasks. Motivated by the society of minds (Minsky, 1988), we propose ReConcile, a multi-model multi-agent framework designed as a round table conference among diverse LLM agents. ReConcile enhances collaborative reasoning between LLM agents via multiple rounds of discussion, learning to convince other agents to improve their answers, and employing a confidence-weighted voting mechanism that leads to a better consensus. In each round, ReConcile initiates discussion between agents via a ‘discussion prompt’ that consists of (a) grouped answers and explanations generated by each agent in the previous round, (b) their confidence scores, and (c) demonstrations of answer-rectifying human explanations, used for convincing other agents. Experiments on seven benchmarks demonstrate that ReConcile significantly improves LLMs’ reasoning – both individually and as a team – surpassing prior single-agent and multi-agent baselines by up to 11.4% and even outperforming GPT-4 on three datasets. ReConcile also flexibly incorporates different combinations of agents, including API-based, open-source, and domain-specific models, leading to an 8% improvement on MATH. Finally, we analyze the individual components of ReConcile, demonstrating that the diversity originating from different models is critical to its superior performance.
     </details>

23. **Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations** [[pdf]](https://aclanthology.org/2024.acl-long.510) `ACL 2024 Long Papers` (49 cite) (4 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          An innovative process-oriented math process reward model called Math-Shepherd, which assigns a reward score to each step of math problem solutions, which holds significant potential for the future evolution of LLMs.
     </details>


     <details>
          <summary>Abstract</summary>
          In this paper, we present an innovative process-oriented math process reward model called Math-shepherd, which assigns a reward score to each step of math problem solutions. The training of Math-shepherd is achieved using automatically constructed process-wise supervision data, breaking the bottleneck of heavy reliance on manual annotation in existing work. We explore the effectiveness of Math-shepherd in two scenarios: 1) Verification: Math-shepherd is utilized for reranking multiple outputs generated by Large Language Models (LLMs); 2) Reinforcement Learning (RL): Math-shepherd is employed to reinforce LLMs.With Math-shepherd, a series of open-source LLMs demonstrates exceptional performance. For instance, process RL with Math-shepherd significantly enhances Mistral-7B (77.9%→84.1% on GSM8K and 28.6%→33.0% on MATH).The accuracy can be further improved to 89.1% and 43.5% on two benchmarks with verification of Math-shepherd.We believe that automatic process supervision holds significant potential for the future evolution of LLMs.
     </details>

24. **Can LLMs Learn from Previous Mistakes? Investigating LLMs’ Errors to Boost for Reasoning** [[pdf]](https://aclanthology.org/2024.acl-long.169) `ACL 2024 Long Papers` (23 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A series of experiments are conducted to prove LLMs can obtain benefits from mistakes in both directions and design two methods that offer potentially cost-effective strategies by leveraging errors to enhance reasoning capabilities, which costs significantly less than creating meticulously hand-crafted golden references.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have demonstrated striking reasoning capability. Recent works have shown the benefits to LLMs from fine-tuning golden-standard Chain-of-Thought (CoT) rationales or using them as correct examples in few-shot prompting. While humans can indeed imitate correct examples, learning from our mistakes is another vital aspect of human cognition. Hence, a question naturally arises: can LLMs learn and benefit from their mistakes, especially for their reasoning?This study investigates this problem from both the prompting and model-tuning perspectives. We begin by introducing CoTErrorSet, a new benchmark with 609,432 questions, each designed with both correct and error references, and demonstrating the types and reasons for making such mistakes. To explore the effectiveness of those mistakes, we design two methods: (1) Self-rethinking prompting guides LLMs to rethink whether they have made similar previous mistakes; and (2) Mistake tuning involves finetuning models in both correct and incorrect reasoning domains, rather than only tuning models to learn ground truth in traditional methodology. We conduct a series of experiments to prove LLMs can obtain benefits from mistakes in both directions. Our two methods offer potentially cost-effective strategies by leveraging errors to enhance reasoning capabilities, which costs significantly less than creating meticulously hand-crafted golden references. We ultimately make a thorough analysis of the reasons behind LLMs’ errors, which provides directions that future research needs to overcome. CoTErrorSet will be published soon on https://github.com/YookiTong/Learn-from-Mistakes-CotErrorSet.
     </details>

25. **Self-Contrast: Better Reflection Through Inconsistent Solving Perspectives** [[pdf]](https://aclanthology.org/2024.acl-long.197) `ACL 2024 Long Papers` (20 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Self-Contrast is advocated: It adaptively explores diverse solving perspectives tailored to the request, contrasts the differences, and summarizes these discrepancies into a checklist which could be used to re-examine and eliminate discrepancies.
     </details>


     <details>
          <summary>Abstract</summary>
          The reflection capacity of Large Language Model (LLM) has garnered extensive attention. A post-hoc prompting strategy, e.g., reflexion and self-refine, refines LLM’s response based on self-evaluated or external feedback. However, recent research indicates without external feedback, LLM’s intrinsic reflection is unstable. Our investigation unveils that the key bottleneck is the quality of the self-evaluated feedback. We find LLMs often exhibit overconfidence or high randomness when self-evaluate, offering stubborn or inconsistent feedback, which causes poor reflection. To remedy this, we advocate Self-Contrast: It adaptively explores diverse solving perspectives tailored to the request, contrasts the differences, and summarizes these discrepancies into a checklist which could be used to re-examine and eliminate discrepancies. Our method endows LLM with diverse perspectives to alleviate stubborn biases. Moreover, their discrepancies indicate potential errors or inherent uncertainties that LLM often overlooks. Reflecting upon these can catalyze more accurate and stable reflection. Experiments conducted on a series of reasoning and translation tasks with different LLMs serve to underscore the effectiveness and generality of our strategy.
     </details>

26. **Rethinking the Bounds of LLM Reasoning: Are Multi-Agent Discussions the Key?** [[pdf]](https://aclanthology.org/2024.acl-long.331) `ACL 2024 Long Papers` (16 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a novel group discussion framework to enrich the set of discussion mechanisms of LLMs, and shows that a single-agent LLM with strong prompts can achieve almost the same performance as the best existing discussion approach on a wide range of reasoning tasks and backbone LLMs.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent progress in LLMs discussion suggests that multi-agent discussion improves the reasoning abilities of LLMs. In this work, we reevaluate this claim through systematic experiments, where we propose a novel group discussion framework to enrich the set of discussion mechanisms. Interestingly, our results show that a single-agent LLM with strong prompts can achieve almost the same best performance as the best existing discussion approach on a wide range of reasoning tasks and backbone LLMs. We observed that the multi-agent discussion performs better than a single agent only when there is no demonstration in the prompt. Further study reveals the common interaction mechanisms of LLMs during the discussion. Our code can be found in https://github.com/HKUST-KnowComp/LLM-discussion.
     </details>

27. **OlympiadBench: A Challenging Benchmark for Promoting AGI with Olympiad-Level Bilingual Multimodal Scientific Problems** [[pdf]](https://aclanthology.org/2024.acl-long.211) `ACL 2024 Long Papers` (14 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work presents OlympiadBench, an Olympiad-level bilingual multimodal scientific benchmark, featuring 8,476 problems from Olympiad-level mathematics and physics competitions, including the Chinese college entrance exam, and implements a comprehensive assessment methodology to accurately evaluate model responses.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent advancements have seen Large Language Models (LLMs) and Large Multimodal Models (LMMs) surpassing general human capabilities in various tasks, approaching the proficiency level of human experts across multiple domains. With traditional benchmarks becoming less challenging for these models, new rigorous challenges are essential to gauge their advanced abilities. In this work, we present OlympiadBench, an Olympiad-level bilingual multimodal scientific benchmark, featuring 8,476 problems from Olympiad-level mathematics and physics competitions, including the Chinese college entrance exam. Each problem is detailed with expert-level annotations for step-by-step reasoning. Evaluating top-tier models on OlympiadBench, we implement a comprehensive assessment methodology to accurately evaluate model responses. Notably, the best-performing model, GPT-4V, attains an average score of 17.97% on OlympiadBench, with a mere 10.74% in physics, highlighting the benchmark rigor and the intricacy of physical reasoning. Our analysis orienting GPT-4V points out prevalent issues with hallucinations, knowledge omissions, and logical fallacies. We hope that our challenging benchmark can serve as a valuable resource for helping future AGI research endeavors. The data and evaluation code are available at https://github.com/OpenBMB/OlympiadBench
     </details>

28. **ReFT: Reasoning with Reinforced Fine-Tuning** [[pdf]](https://aclanthology.org/2024.acl-long.410) `ACL 2024 Long Papers` (14 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Reformed Fine-Tuning first warmups the model with SFT, and then employs on-line reinforcement learning, specifically the PPO algorithm in this paper, to further fine-tune the model, where an abundance of reasoning paths are automatically sampled given the question and the rewards are naturally derived from the ground-truth answers.
     </details>


     <details>
          <summary>Abstract</summary>
          One way to enhance the reasoning capability of Large Language Models (LLMs) is to conduct Supervised Fine-Tuning (SFT) using Chain-of-Thought (CoT) annotations. This approach does not show sufficiently strong generalization ability, however, because the training only relies on the given CoT data. In math problem-solving, for example, there is usually only one annotated reasoning path for each question in the training data. Intuitively, it would be better for the algorithm to learn from multiple annotated reasoning paths given a question. To address this issue, we propose a simple yet effective approach called Reinforced Fine-Tuning (ReFT) to enhance the generalizability of learning LLMs for reasoning, with math problem-solving as an example. ReFT first warmups the model with SFT, and then employs on-line reinforcement learning, specifically the PPO algorithm in this paper, to further fine-tune the model, where an abundance of reasoning paths are automatically sampled given the question and the rewards are naturally derived from the ground-truth answers. Extensive experiments on GSM8K, MathQA, and SVAMP datasets show that ReFT significantly outperforms SFT, and the performance can be potentially further boosted by combining inference-time strategies such as majority voting and re-ranking. Note that ReFT obtains the improvement by learning from the same training questions as SFT, without relying on extra or augmented training questions. This indicates a superior generalization ability for ReFT.
     </details>

29. **MuggleMath: Assessing the Impact of Query and Response Augmentation on Math Reasoning** [[pdf]](https://aclanthology.org/2024.acl-long.551) `ACL 2024 Long Papers` (12 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is found that MuggleMath is weak in out-of-domain math reasoning generalization from AugGSM8K to MATH and from AugMATH to GSM8K, which suggests that augmenting queries that cover a broader range of subjects is more beneficial for generalization.
     </details>


     <details>
          <summary>Abstract</summary>
          In math reasoning with large language models (LLMs), fine-tuning data augmentation by query evolution and diverse reasoning paths is empirically verified effective, profoundly narrowing the gap between open-sourced LLMs and cutting-edge proprietary LLMs. In this paper, we conduct an investigation for such data augmentation in math reasoning and are intended to answer: (1) What strategies of data augmentation are more effective; (2) What is the scaling relationship between the amount of augmented data and model performance; and (3) Can data augmentation incentivize generalization to out-of-domain mathematical reasoning tasks?To this end, we create two new dataset AugGSM8K and AugMATH, by complicating and diversifying the queries and sampling multiple reasoning paths from GSM8K and MATH.We obtained a series of LLMs called MuggleMath by fine-tuning LLaMA models on AugGSM8K and AugMATH. MuggleMath substantially achieves new state-of-the-art on GSM8K and MATH.A log-linear relationship and a segmented log-linear are presented between MuggleMath’s performance and the amount of augmented data on GSM8K and MATH, respectively.We also find that it is weak in out-of-domain math reasoning generalization from AugGSM8K to MATH and from AugMATH to GSM8K, which suggests that augmenting queries that cover a broader range of subjects is more beneficial for generalization.
     </details>

30. **When is Tree Search Useful for LLM Planning? It Depends on the Discriminator** [[pdf]](https://aclanthology.org/2024.acl-long.738) `ACL 2024 Long Papers` (10 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          In this paper, we examine how large language models (LLMs) solve multi-step problems under a language agent framework with three components: a generator, a discriminator, and a planning method. We investigate the practical utility of two advanced planning methods, iterative correction and tree search. We present a comprehensive analysis of how discrimination accuracy affects the overall performance of agents when using these two methods or a simpler method, re-ranking. Experiments on two tasks, text-to-SQL parsing and mathematical reasoning, show that: (1) advanced planning methods demand discriminators with at least 90% accuracy to achieve significant improvements over re-ranking; (2) current LLMs’ discrimination abilities have not met the needs of advanced planning methods to achieve such improvements; (3) with LLM-based discriminators, advanced planning methods may not adequately balance accuracy and efficiency. For example, compared to the other two methods, tree search is at least 10–20 times slower but leads to negligible performance gains, which hinders its real-world applications.
     </details>

31. **Eliciting Better Multilingual Structured Reasoning from LLMs through Code** [[pdf]](https://aclanthology.org/2024.acl-long.281) `ACL 2024 Long Papers` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces a multilingual structured reasoning and explanation dataset, termed xSTREET, that covers four tasks across six languages and proposes two methods to remedy this gap, building on the insight that LLMs trained on code are better reasoners.
     </details>


     <details>
          <summary>Abstract</summary>
          The development of large language models (LLM) has shown progress on reasoning, though studies have largely considered either English or simple reasoning tasks. To address this, we introduce a multilingual structured reasoning and explanation dataset, termed xSTREET, that covers four tasks across six languages. xSTREET exposes a gap in base LLM performance between English and non-English reasoning tasks.We then propose two methods to remedy this gap, building on the insight that LLMs trained on code are better reasoners. First, at training time, we augment a code dataset with multilingual comments using machine translation while keeping program code as-is. Second, at inference time, we bridge the gap between training and inference by employing a prompt structure that incorporates step-by-step code primitives to derive new facts and find a solution. Our methods show improved multilingual performance on xSTREET, most notably on the scientific commonsense reasoning subtask. Furthermore, the models show no regression on non-reasoning tasks, thus demonstrating our techniques maintain general-purpose abilities.
     </details>

32. **SEER: Facilitating Structured Reasoning and Explanation via Reinforcement Learning** [[pdf]](https://aclanthology.org/2024.acl-long.321) `ACL 2024 Long Papers` (2 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper proposes SEER, a novel method that maximizes a structure-based return to facilitate structured reasoning and explanation, and introduces a fine-grained reward function to meticulously delineate diverse reasoning steps.
     </details>


     <details>
          <summary>Abstract</summary>
          Elucidating the reasoning process with structured explanations from question to answer is crucial, as it significantly enhances the interpretability, traceability, and trustworthiness of question-answering (QA) systems. However, structured explanations demand models to perform intricately structured reasoning, which poses great challenges. Most existing methods focus on single-step reasoning through supervised learning, ignoring logical dependencies between steps. Moreover, existing reinforcement learning (RL) based methods overlook the structured relationships, underutilizing the potential of RL in structured reasoning. In this paper, we propose SEER, a novel method that maximizes a structure-based return to facilitate structured reasoning and explanation. Our proposed structure-based return precisely describes the hierarchical and branching structure inherent in structured reasoning, effectively capturing the intricate relationships between different reasoning steps. In addition, we introduce a fine-grained reward function to meticulously delineate diverse reasoning steps. Extensive experiments show that SEER significantly outperforms state-of-the-art methods, achieving an absolute improvement of 6.9% over RL-based methods on EntailmentBank, a 4.4% average improvement on STREET benchmark, and exhibiting outstanding efficiency and cross-dataset generalization performance.
     </details>

33. **mCoT: Multilingual Instruction Tuning for Reasoning Consistency in Language Models** [[pdf]](https://aclanthology.org/2024.acl-long.649) `ACL 2024 Long Papers` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work compiles the first large-scale multilingual math reasoning dataset, mCoT-MATH, and introduces multilingual CoT instruction tuning to boost reasoning capability across languages, thereby improving model consistency.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) with Chain-of-thought (CoT) have recently emerged as a powerful technique for eliciting reasoning to improve various downstream tasks. As most research mainly focuses on English, with few explorations in a multilingual context, the question of how reliable this reasoning capability is in different languages is still open. To address it directly, we study multilingual reasoning consistency across multiple languages, using popular open-source LLMs. First, we compile the first large-scale multilingual math reasoning dataset, *mCoT-MATH*, covering eleven diverse languages. Then, we introduce multilingual CoT instruction tuning to boost reasoning capability across languages, thereby improving model consistency. While existing LLMs show substantial variation across the languages we consider, and especially low performance for lesser resourced languages, our 7B parameter model *mCoT* achieves impressive consistency across languages, and superior or comparable performance to close- and open-source models even of much larger sizes.
     </details>

34. **Enhancing Numerical Reasoning with the Guidance of Reliable Reasoning Processes** [[pdf]](https://aclanthology.org/2024.acl-long.582) `ACL 2024 Long Papers` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces Enhancing NumeriCal reasOning with Reliable procEsses (Encore), which derives the reliable reasoning process by decomposing the answer formula, ensuring which fully supports the answer.
     </details>


     <details>
          <summary>Abstract</summary>
          Numerical reasoning is an essential ability for NLP systems to handle numeric information. Recent research indicates that fine-tuning a small-scale model to learn generating reasoning processes alongside answers can significantly enhance performance. However, current methods have the limitation that most methods generate reasoning processes with large language models (LLMs), which are “unreliable” since such processes could contain information unrelated to the answer. To address this limitation, we introduce enhancing numerical reasoning with reliable processes (Encore), which derives the reliable reasoning process by decomposing the answer formula, ensuring which fully supports the answer. Nevertheless, models could lack enough data to learn the reasoning process generation adequately, since our method generates only one single reasoning process for one formula. To overcome this difficulty, we present a series of pre-training tasks to help models learn the reasoning process generation with synthesized data. The experiments show that Encore yields improvement on all five experimental datasets with an average of 1.8%, proving the effectiveness of our method.
     </details>

35. **KnowledgeFMath: A Knowledge-Intensive Math Reasoning Dataset in Finance Domains** [[pdf]](https://aclanthology.org/2024.acl-long.693) `ACL 2024 Long Papers` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          We introduce KnowledgeFMath, a novel benchmark designed to evaluate LLMs’ capabilities in solving knowledge-intensive math reasoning problems. Compared to prior works, this study features three core advancements. First, KnowledgeFMath includes 1,259 problems with a hybrid of textual and tabular content. These problems require college-level knowledge in the finance domain for effective resolution. Second, we provide expert-annotated, detailed solution references in Python program format, ensuring a high-quality benchmark for LLM assessment. We also construct a finance-domain knowledge bank and investigate various knowledge integration strategies. Finally, we evaluate a wide spectrum of 26 LLMs with different prompting strategies like Chain-of-Thought and Program-of-Thought. Our experimental results reveal that the current best-performing system (i.e., GPT-4 with CoT prompting) achieves only 56.6% accuracy, leaving substantial room for improvement. Moreover, while augmenting LLMs with external knowledge can improve their performance (e.g., from 33.5% to 47.1% for GPT-3.5), their accuracy remains significantly lower than the estimated human expert performance of 92%. We believe that KnowledgeFMath can advance future research in the area of domain-specific knowledge retrieval and integration, particularly within the context of solving math reasoning problems.
     </details>

36. **Mirror: Multiple-perspective Self-Reflection Method for Knowledge-rich Reasoning** [[pdf]](https://aclanthology.org/2024.acl-long.382) `ACL 2024 Long Papers` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          While Large language models (LLMs) have the capability to iteratively reflect on their own outputs, recent studies have observed their struggles with knowledge-rich problems without access to external resources. In addition to the inefficiency of LLMs in self-assessment, we also observe that LLMs struggle to revisit their predictions despite receiving explicit negative feedback. Therefore, We propose Mirror, a Multiple-perspective self-reflection method for knowledge-rich reasoning, to avoid getting stuck at a particular reflection iteration. Mirror enables LLMs to reflect from multiple-perspective clues, achieved through a heuristic interaction between a Navigator and a Reasoner. It guides agents toward diverse yet plausibly reliable reasoning trajectory without access to ground truth by encouraging (1) diversity of directions generated by Navigator and (2) agreement among strategically induced perturbations in responses generated by the Reasoner. The experiments on five reasoning datasets demonstrate that Mirror’s superiority over several contemporary self-reflection approaches. Additionally, the ablation study studies clearly indicate that our strategies alleviate the aforementioned challenges.
     </details>

37. **Reasoning in Flux: Enhancing Large Language Models Reasoning through Uncertainty-aware Adaptive Guidance** [[pdf]](https://aclanthology.org/2024.acl-long.131) `ACL 2024 Long Papers` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Uncertainty-aware Adaptive Guidance (UAG) is introduced, a novel approach for guiding LLM reasoning onto an accurate and reliable trajectory that not only enhances the reasoning abilities of LLMs but also consistently outperforms several strong baselines with minimal computational overhead.
     </details>


     <details>
          <summary>Abstract</summary>
          Machine reasoning, which involves solving complex problems through step-by-step deduction and analysis, is a crucial indicator of the capabilities of Large Language Models (LLMs). However, as the complexity of tasks escalates, LLMs often encounter increasing errors in their multi-step reasoning process. This study delves into the underlying factors contributing to these reasoning errors and seeks to leverage uncertainty to refine them. Specifically, we introduce Uncertainty-aware Adaptive Guidance (UAG), a novel approach for guiding LLM reasoning onto an accurate and reliable trajectory. UAG first identifies and evaluates uncertainty signals within each step of the reasoning chain. Upon detecting a significant increase in uncertainty, UAG intervenes by retracting to a previously reliable state and then introduces certified reasoning clues for refinement. By dynamically adjusting the reasoning process, UAG offers a plug-and-play solution for improving LLMs’ performance in complex reasoning. Extensive experiments across various reasoning tasks demonstrate that UAG not only enhances the reasoning abilities of LLMs but also consistently outperforms several strong baselines with minimal computational overhead. Further analysis reveals that UAG is notably effective in identifying and diminishing reasoning errors.
     </details>

38. **SEGO: Sequential Subgoal Optimization for Mathematical Problem-Solving** [[pdf]](https://aclanthology.org/2024.acl-long.407) `ACL 2024 Long Papers` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel framework called SEGO is proposed to enhance LLMs' ability to solve mathematical problems by establishing a connection between the subgoal breakdown process and the probability of solving problems, and SEGO aims to identify better subgoals with theoretical guarantees.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have driven substantial progress in artificial intelligence in recent years, exhibiting impressive capabilities across a wide range of tasks, including mathematical problem-solving. Inspired by the success of subgoal-based methods, we propose a novel framework called SEquential subGoal Optimization (SEGO) to enhance LLMs’ ability to solve mathematical problems. By establishing a connection between the subgoal breakdown process and the probability of solving problems, SEGO aims to identify better subgoals with theoretical guarantees. Addressing the challenge of identifying suitable subgoals in a large solution space, our framework generates problem-specific subgoals and adjusts them according to carefully designed criteria. Incorporating these optimized subgoals into the policy model training leads to significant improvements in problem-solving performance. We validate SEGO’s efficacy through experiments on two benchmarks, GSM8K and MATH, where our approach outperforms existing methods, highlighting the potential of SEGO in AI-driven mathematical problem-solving.
     </details>

39. **Self-Training with Direct Preference Optimization Improves Chain-of-Thought Reasoning** [[pdf]](https://aclanthology.org/2024.acl-long.643) `ACL 2024 Long Papers` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work demonstrates that the reasoning abilities of small-scale LMs can be enhanced through self-training, a process where models learn from their own outputs, and shows that the conventional self-training can be further augmented by a preference learning algorithm called Direct Preference Optimization (DPO).
     </details>


     <details>
          <summary>Abstract</summary>
          Teaching small-scale language models to perform math reasoning is a valuable yet challenging task. Besides obtaining labeled data from human experts, one of the most common ways to collect high-quality data is by sampling from a larger and more powerful language model. Although previous works have demonstrated the effectiveness of this method, such a knowledge distillation paradigm can be costly and unstable, especially considering that many large language models, such as GPT-4, are closed-sourced, proprietary, and their behaviors are unpredictable. In this work, to avoid relying on outputs from large models, we demonstrate that the reasoning abilities of small-scale language models can be enhanced through self-training, which involves training models with their own outputs. We also show that the vanilla self-training can be further augmented by an alignment algorithm, direct preference optimization (DPO). We empirically found that models trained with the DPO objective are capable of making better generations that largely benefit multi-turn self-training. The experiments show our models outperform the state-of-the-art models with comparable sizes on a series of downstream math reasoning tasks with minimal resource requirements.
     </details>

40. **MACM: Utilizing a Multi-Agent System for Condition Mining in Solving Complex Mathematical Problems** [[pdf]](http://arxiv.org/abs/2404.04735) `NeurIPS 2024` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduces the MACM prompting method, which not only resolves intricate mathematical problems but also demonstrates strong generalization capabilities across various mathematical contexts.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent advancements in large language models, such as GPT-4, have demonstrated remarkable capabilities in processing standard queries. Despite these advancements, their performance substantially declines in advanced mathematical problems requiring complex, multi-step logical reasoning. To enhance their inferential capabilities, current research has delved into prompting engineering, exemplified by methodologies such as the Tree of Thought and Graph of Thought.Nonetheless, these existing approaches encounter two significant limitations. Firstly, their effectiveness in tackling complex mathematical problems is somewhat constrained. Secondly, the necessity to design distinct prompts for individual problems hampers their generalizability.In response to these limitations, this paper introduces the Multi-Agent System for conditional Mining (MACM) prompting method. It not only resolves intricate mathematical problems but also demonstrates strong generalization capabilities across various mathematical contexts.With the assistance of MACM, the accuracy of GPT-4 Turbo on the most challenging level five mathematical problems in the MATH dataset increase from $\mathbf{54.68\\%}  \text{ to } \mathbf{76.73\\%}$.
     </details>

41. **PutnamBench: Evaluating Neural Theorem-Provers on the Putnam Mathematical Competition** [[pdf]](http://arxiv.org/abs/2407.11214) `NeurIPS 2024` `Lean, Isabelle, Coq` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          PutnamBench consists of 1697 hand-constructed formalizations of 640 theorems sourced from the William Lowell Putnam Mathematical Competition, the premier undergraduate-level mathematics competition in North America.
     </details>


     <details>
          <summary>Abstract</summary>
          We present PutnamBench, a new multilingual benchmark for evaluating the ability of neural theorem-provers to solve competition mathematics problems. PutnamBench consists of 1337 hand-constructed formalizations of 514 theorems sourced from the William Lowell Putnam Mathematical Competition, the premier undergraduate-level mathematics competition in North America.  All the theorems have formalizations in Lean 4 and Isabelle; a substantial subset also has Coq formalizations. Proving the theorems requires significant problem-solving ability and proficiency in a broad range of topics taught in undergraduate mathematics courses. We use PutnamBench to evaluate several established neural and symbolic theorem-provers.  These approaches can only solve a handful of the PutnamBench problems, establishing the benchmark as a difficult open challenge for research on neural theorem-proving. PutnamBench is available at https://github.com/trishullab/PUTNAM.
     </details>

42. **Masked Thought: Simply Masking Partial Reasoning Steps Can Improve Mathematical Reasoning Learning of Language Models** [[pdf]](http://arxiv.org/abs/2403.02178) `ACL 2024 Long Papers` (5 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work develops a method that avoids external resources, relying instead on introducing perturbations to the input, and randomly masks certain tokens within the chain of thought, a technique found to be particularly effective for reasoning tasks.
     </details>


     <details>
          <summary>Abstract</summary>
          In reasoning tasks, even a minor error can cascade into inaccurate results, leading to suboptimal performance of large language models insuch domains. Earlier fine-tuning approaches sought to mitigate this by leveraging more precise supervisory signals from human labeling, larger models, or self-sampling, although at a high cost. Conversely, we develop a method that avoids external resources, relying instead on introducing perturbations to the input. Our training approach randomly masks certain tokens within the chain of thought, a techniquewe found to be particularly effective for reasoning tasks. When applied to fine-tuning with GSM8K on Llama-2-7B, this method achieveda 5% improvement in GSM8K accuracy and a 10% improvement in GSM-IC accuracy over standard supervised fine-tuning with a few codes modified. Furthermore, it is complementary to existing methods. When integrated with related explicit data augmentation methods, it leads to improvements across five datasets of various augmentation methods, as well as two different base models. We further investigate the mechanisms behind this improvement through case studies and quantitative analysis, suggesting that our approach may provide superior support for the model in capturing long-distance dependencies, especially those related to questions. This enhancement could deepen understanding of the premises in questions and prior steps.
     </details>

43. **Visual Sketchpad: Sketching as a Visual Chain of Thought for Multimodal Language Models** [[pdf]](http://arxiv.org/abs/2406.09403) `NeurIPS 2024` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Sketchpad enables LMs to draw with lines, boxes, marks, etc., which is closer to human sketching and better facilitates reasoning, and sets a new state of the art on all tasks, including V*Bench and BLINK spatial reasoning.
     </details>


     <details>
          <summary>Abstract</summary>
          Humans draw to facilitate reasoning: we draw auxiliary lines when solving geometry problems; we mark and circle when reasoning on maps; we use sketches to amplify our ideas and relieve our limited-capacity working memory. However, such actions are missing in current multimodal language models (LMs). Current chain-of-thought and tool-use paradigms only use text as intermediate reasoning steps. In this work, we introduce Sketchpad, a framework that gives multimodal LMs a visual sketchpad and tools to draw on the sketchpad. The LM conducts planning and reasoning according to the visual artifacts it has drawn. Different from prior work, which uses text-to-image models to enable LMs to draw, Sketchpad enables LMs to draw with lines, boxes, marks, etc., which is closer to human sketching and better facilitates reasoning. \name can also use specialist vision models during the sketching process (e.g., draw bounding boxes with object detection models, draw masks with segmentation models), to further enhance visual perception and reasoning. We experiment on a wide range of math tasks (including geometry, functions, graph, chess) and complex visual reasoning tasks. Sketchpad substantially improves performance on all tasks over strong base models with no sketching, yielding an average gain of 12.7% on math tasks, and 8.6% on vision tasks. GPT-4o with Sketchpad sets a new state of the art on all tasks, including V*Bench (80.3%), BLINK spatial reasoning (83.9%), and visual correspondence (80.8%). We will release all code and data.
     </details>

44. **GLoRe: When, Where, and How to Improve LLM Reasoning via Global and Local Refinements** [[pdf]](https://proceedings.mlr.press/v235/havrilla24a.html) `ICML 2024 Poster` (12 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          SORMs are trained to predict the correctness of the final answer when sampling the current policy many times (rather than only once as in the case of ORMs), and are shown to more accurately detect incorrect reasoning steps compared to ORMs.
     </details>


     <details>
          <summary>Abstract</summary>
          State-of-the-art language models can exhibit reasoning refinement capabilities on math, science or coding tasks. However, recent work demonstrates that even the best models struggle to identify *when and where to refine* without access to external feedback. In this paper, we propose Stepwise ORMs (**SORMs**) which are trained, only on synthetic data, to approximate the expected future reward of the optimal policy or $V^{\star}$ as a form of Process-based reward modeling. Our experiments show that SORMs can more accurately detect incorrect reasoning steps compared to ORMs, thus enabling them to give precise step-level feedback to refinement models. We then train *global* refinement models, which take only the question and a draft solution as input and predict a corrected solution, and *local* refinement models which also take as input a critique indicating the location of the first reasoning error. We generate training data for both models synthetically by reusing data used to train the SORM. We find combining global and local refinements, using the ORM as a reranker, significantly outperforms either one individually, as well as a best of three sample baseline. With this strategy we can improve the accuracy of a LLaMA-2 13B model (already fine-tuned with RL) on GSM8K from 53% to 65% when greedily sampled.
     </details>

45. **Generating Chain-of-Thoughts with a Pairwise-Comparison Approach to Searching for the Most Promising Intermediate Thought** [[pdf]](https://proceedings.mlr.press/v235/zhang24t.html) `ICML 2024 Poster` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper uses pairwise-comparison evaluation instead of point-wise scoring to search for promising intermediate thoughts with the noisy feedback from the LLM, and incorporates techniques from ensemble learning and dueling bandits to incorporate the rationale of the pairwise comparison mechanism.
     </details>


     <details>
          <summary>Abstract</summary>
          To improve the ability of the large language model (LLMs) to tackle complex reasoning problems, chain-of-thoughts (CoT) methods were proposed to guide LLMs to reason step-by-step, enabling problem solving from simple to complex. State-of-the-art methods for generating such a chain involve interactive collaboration, where the learner generates candidate intermediate thoughts, evaluated by the LLM, guiding the generation of subsequent thoughts. However, a widespread yet understudied problem is that the evaluation from the LLM is typically noisy and unreliable, potentially misleading the generation process in selecting promising intermediate thoughts. In this paper, motivated by Vapnik's principle, we use pairwise-comparison evaluation instead of point-wise scoring to search for promising intermediate thoughts with the noisy feedback from the LLM. In each round, we randomly pair intermediate thoughts and directly prompt the LLM to select the more promising one from each pair, allowing us to identify the most promising thoughts through an iterative process. To further alleviate the noise in the comparison, we incorporate techniques from ensemble learning and dueling bandits, proposing two variants of the algorithm. Experiments on three real-world tasks demonstrate the effectiveness of our proposed algorithm and verify the rationale of the pairwise comparison mechanism.
     </details>

46. **GSM-Plus: A Comprehensive Benchmark for Evaluating the Robustness of LLMs as Mathematical Problem Solvers** [[pdf]](http://arxiv.org/abs/2402.19255) `ACL 2024 Long Papers` (12 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          While LLMs exhibit different levels of math reasoning abilities, their performances are far from robust, and even for problems that have been solved in GSM8K, LLMs can make mistakes when new statements are added or the question targets are altered.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have achieved impressive performance across various mathematical reasoning benchmarks. However, there are increasing debates regarding whether these models truly understand and apply mathematical knowledge or merely rely on shortcuts for mathematical reasoning. One essential and frequently occurring evidence is that when the math questions are slightly changed, LLMs can behave incorrectly. This motivates us to evaluate the robustness of LLMs’ math reasoning capability by testing a wide range of question variations. We introduce the adversarial grade school math (GSM-Plus) dataset, an extension of GSM8K augmented with various mathematical perturbations. Our experiments on 25 LLMs and 4 prompting techniques show that while LLMs exhibit different levels of math reasoning abilities, their performances are far from robust. In particular, even for problems that have been solved in GSM8K, LLMs can make mistakes when new statements are added or the question targets are altered. We also explore whether more robust performance can be achieved by composing existing prompting methods, in which we try an iterative method that generates and verifies each intermediate thought based on its reasoning goal and calculation result.
     </details>

47. **Learning Formal Mathematics From Intrinsic Motivation** [[pdf]](http://arxiv.org/abs/2407.00695) `NeurIPS 2024` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          How did humanity coax mathematics from the aether? We explore the Platonic view that mathematics can be discovered from its axioms---a game of conjecture and proof. We describe an agent that jointly learns to pose challenging problems for itself (conjecturing) and solve them (theorem proving). Given a mathematical domain axiomatized in dependent type theory, we first combine methods for constrained decoding and type-directed synthesis to sample valid conjectures from a language model. Our method guarantees well-formed conjectures by construction, even as we start with a randomly initialized model. We use the same model to represent a policy and value function for guiding proof search. Our agent targets generating hard but provable conjectures --- a moving target, since its own theorem proving ability also improves as it trains. We propose novel methods for hindsight relabeling on proof search trees to significantly improve the agent's sample efficiency in both tasks. Experiments on 3 axiomatic domains (propositional logic, arithmetic and group theory) demonstrate that our agent can bootstrap from only the axioms, self-improving in generating true and challenging conjectures and in finding proofs.
     </details>

48. **OccamLLM: Fast and Exact Language Model Arithmetic in a Single Step** [[pdf]](http://arxiv.org/abs/2406.06576) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a framework that enables exact arithmetic in a single autoregressive step, providing faster, more secure, and more interpretable LLM systems with arithmetic capabilities.
     </details>


     <details>
          <summary>Abstract</summary>
          Despite significant advancements in text generation and reasoning, Large Language Models (LLMs) still face challenges in accurately performing complex arithmetic operations. To achieve accurate calculations, language model systems often enable LLMs to generate code for arithmetic operations. However, this approach compromises speed and security and, if finetuning is involved, risks the language model losing prior capabilities. We propose a framework that enables exact arithmetic in a single autoregressive step, providing faster, more secure, and more interpretable LLM systems with arithmetic capabilities. We use the hidden states of an LLM to control a symbolic architecture which performs arithmetic. Our implementation using Llama 3 8B Instruct with OccamNet as a symbolic model (OccamLlama) achieves 100% accuracy on single arithmetic operations (+, -, *, /, sin, cos, log, exp, sqrt), outperforming GPT 4o and on par with GPT 4o using a code interpreter. OccamLlama also outperforms both Llama 3 8B Instruct and GPT 3.5 Turbo on multistep reasoning problems involving challenging arithmetic, thus enabling small LLMs to match the arithmetic performance of even much larger models. Our code is available at https://anonymous.4open.science/r/OccamLlama.
     </details>

49. **Iterative Reasoning Preference Optimization** [[pdf]](http://arxiv.org/abs/2404.19733) `NeurIPS 2024` (21 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work develops an iterative approach that optimizes the preference between competing generated Chain-of-Thought (CoT) candidates by optimizing for winning vs. losing reasoning steps that lead to the correct answer.
     </details>


     <details>
          <summary>Abstract</summary>
          Iterative preference optimization methods have recently been shown to perform well for general instruction tuning tasks, but typically make little improvement on reasoning tasks. In this work we develop an iterative approach that optimizes the preference between competing generated Chain-of-Thought (CoT) candidates by optimizing for winning vs. losing reasoning steps that lead to the correct answer. We train using a modified DPO loss with an additional negative log-likelihood term, which we find to be crucial. We show reasoning improves across repeated iterations of this scheme. While only relying on examples in the training set, our approach results in increasing accuracy on GSM8K, MATH, and ARC-Challenge for Llama-2-70B-Chat, outperforming other Llama-2-based models not relying on additionally sourced datasets. For example, we see a large improvement from 55.6% to 81.6% on GSM8K and an accuracy of 88.7% with majority voting out of 32 samples.
     </details>

50. **Evaluating Large Vision-and-Language Models on Children's Mathematical Olympiads** [[pdf]](http://arxiv.org/abs/2406.15736) `NeurIPS 2024` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Evaluating state-of-the-art LVLMs on their mathematical and algorithmic reasoning abilities using visuo-linguistic problems from children's Olympiads shows that modern LVLMs do demonstrate increasingly powerful reasoning skills in solving problems for higher grades, but lack the foundations to correctly answer problems designed for younger children.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent years have seen a significant progress in the general-purpose problem solving abilities of large vision and language models (LVLMs), such as ChatGPT, Gemini, etc.; some of these breakthroughs even seem to enable AI models to outperform human abilities in varied tasks that demand higher-order cognitive skills. Are the current large AI models indeed capable of generalized problem solving as humans do?  A systematic analysis of AI capabilities for joint vision and text reasoning, however, is missing in the current scientific literature. In this paper, we make an effort towards filling this gap, by evaluating state-of-the-art LVLMs on their mathematical and algorithmic reasoning abilities using visuo-linguistic problems from children's Olympiads. Specifically, we consider problems from the Mathematical Kangaroo (MK) Olympiad, which is a popular international competition targeted at children from grades 1-12, that tests children's deeper mathematical abilities using puzzles that are appropriately gauged to their age and skills. Using the puzzles from MK, we created a dataset, dubbed SMART-840, consisting of 840 problems from years 2020-2024. With our dataset, we analyze LVLMs power on mathematical reasoning; their responses on our puzzles offer a direct way to compare against that of children. Our results show that modern LVLMs do demonstrate increasingly powerful reasoning skills in solving problems for higher grades, but lack the foundations to correctly answer problems designed for younger children. Further analysis shows that there is no significant correlation between the reasoning capabilities of AI models and that of young children, and their capabilities appear to be based on a different type of reasoning than the cumulative knowledge that underlies children's mathematical skills.
     </details>

51. **Proving Olympiad Algebraic Inequalities without Human Demonstrations** [[pdf]](http://arxiv.org/abs/2406.14219) `NeurIPS 2024` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes AIPS, an Algebraic Inequality Proving System capable of autonomously generating complex inequality theorems and effectively solving Olympiad-level inequality problems without requiring human demonstrations.
     </details>


     <details>
          <summary>Abstract</summary>
          Solving Olympiad-level mathematical problems represents a significant advancement in machine intelligence and automated reasoning. Current machine learning methods, however, struggle to solve Olympiad-level problems beyond Euclidean plane geometry due to a lack of large-scale, high-quality datasets. The challenge is even greater in algebraic systems, which involves infinite reasoning spaces within finite conditions. To address these issues, we propose \textit{AIPS}, an \textit{Algebraic Inequality Proving System} capable of autonomously generating complex inequality theorems and effectively solving Olympiad-level inequality problems without requiring human demonstrations. During proof search in a mixed reasoning manner, a value curriculum learning strategy on generated datasets is implemented to improve proving performance, demonstrating strong mathematical intuitions. On a test set of 20 International Mathematical Olympiad-level inequality problems, AIPS successfully solved 10, outperforming state-of-the-art methods. Furthermore, AIPS automatically generated a vast array of non-trivial theorems without human intervention, some of which have been evaluated by professional contestants and deemed to reach the level of the International Mathematical Olympiad. Notably, one theorem was selected as a competition problem in a major city 2024 Mathematical Olympiad.All the materials are available at {\it \href{https://sites.google.com/view/aips}{sites.google.com/view/aips}}.
     </details>

52. **DART-Math: Difficulty-Aware Rejection Tuning for Mathematical Problem-Solving** [[pdf]](http://arxiv.org/abs/2407.13690) `NeurIPS 2024` (5 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Hypothesizing that difficult queries are crucial to learn complex reasoning, this work proposes Difficulty-Aware Rejection Tuning (DART), a method that allocates difficult queries more trials during the synthesis phase, enabling more extensive training on difficult samples.
     </details>


     <details>
          <summary>Abstract</summary>
          Solving mathematical problems requires advanced reasoning abilities and presents notable challenges for large language models. Previous works usually synthesize data from proprietary models to augment existing datasets, followed by instruction tuning to achieve top-tier results. However, our analysis of these datasets reveals severe biases towards easy queries, with frequent failures to generate any correct response for the most challenging queries.Hypothesizing that difficult queries are crucial to learn complex reasoning, we propose Difficulty-Aware Rejection Tuning (DART), a method that allocates difficult queries more trials during the synthesis phase, enabling more extensive training on difficult samples.Utilizing DART, we have created new datasets for mathematical problem-solving that focus more on difficult queries and are substantially smaller than previous ones. Remarkably, our synthesis process solely relies on a 7B-sized open-weight model, without reliance on the commonly used proprietary GPT-4.We fine-tune various base models on our datasets ranging from 7B to 70B in size, resulting in a series of strong models called DART-Math.In comprehensive in-domain and out-of-domain evaluation on 6 mathematical benchmarks, DART-Math outperforms vanilla rejection tuning significantly, being superior or comparable to previous arts, despite using much smaller datasets and no proprietary models. Furthermore, our results position our synthetic datasets as the most effective and cost-efficient publicly available resources for advancing mathematical problem-solving. Our datasets and models will be made publicly available following the review period.
     </details>

53. **THOUGHT PROPAGATION: AN ANALOGICAL APPROACH TO COMPLEX REASONING WITH LARGE LANGUAGE MODELS** [[pdf]](http://arxiv.org/abs/2310.03965) `ICLR 2024` (16 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Thought Propagation (TP) is proposed, which explores the analogous problems and leverages their solutions to enhance the complex reasoning ability of LLMs and is compatible with existing prompting approaches, allowing plug-and-play generalization and enhancement in a wide range of tasks without much labor in task-specific prompt engineering.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have achieved remarkable success in reasoning tasks with the development of prompting methods. However, existing prompting approaches cannot reuse insights of solving similar problems and suffer from accumulated errors in multi-step reasoning, since they prompt LLMs to reason \textit{from scratch}.To address these issues, we propose \textbf{\textit{Thought Propagation} (TP)}, which explores the analogous problems and leverages their solutions to enhance the complex reasoning ability of LLMs.These analogous problems are related to the input one, with reusable solutions and problem-solving strategies.Thus, it is promising to propagate insights of solving previous analogous problems to inspire new problem-solving. To achieve this, TP first prompts LLMs to propose and solve a set of analogous problems that are related to the input one. Then, TP reuses the results of analogous problems to directly yield a new solution or derive a knowledge-intensive plan for execution to amend the initial solution obtained from scratch.TP is compatible with existing prompting approaches, allowing plug-and-play generalization and enhancement in a wide range of tasks without much labor in task-specific prompt engineering. Experiments across three challenging tasks demonstrate TP enjoys a substantial improvement over the baselines by an average of 12\% absolute increase in finding the optimal solutions in Shortest-path Reasoning, 13\% improvement of human preference in Creative Writing, and 15\% enhancement in the task completion rate of LLM-Agent Planning.
     </details>

54. **Metacognitive Capabilities of LLMs: An Exploration in Mathematical Problem Solving** [[pdf]](https://openreview.net/forum?id=0MsI3bSmmD&name=pdf) `NeurIPS 2024` (5 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          \emph{Metacognitive knowledge} refers to humans' intuitive knowledge of their own thinking and reasoning processes. Today's best LLMs clearly possess some reasoning processes. The paper gives evidence that they also  have metacognitive knowledge, including ability to name skills and procedures to apply given a task. We explore this primarily in context of math reasoning, developing a prompt-guided interaction procedure  to get a powerful  LLM to assign sensible skill labels to math questions, followed by having it perform semantic clustering to obtain coarser families of skill labels. These coarse skill labels look interpretable to humans.To validate that these skill labels are meaningful and relevant to the LLM's reasoning processes we perform the following experiments. (a) We ask GPT-4 to assign skill labels to training questions in math datasets GSM8K and MATH.  (b) When using an LLM to solve the test questions, we present it with the full list of skill labels and ask it to identify the skill needed. Then it is presented with randomly selected exemplar solved questions associated with that skill label.  This improves accuracy on GSM8k and MATH for several strong LLMs, including code-assisted models. The methodology presented is domain-agnostic,  even though this article applies it to math problems.
     </details>

55. **DACO: Towards Application-Driven and Comprehensive Data Analysis via Code Generation** [[pdf]](https://openreview.net/forum?id=Y5iTZ52yFs&name=pdf) `NeurIPS 2024` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes to automatically generate high-quality answer annotations leveraging the code-generation capabilities of LLMs with a multi-turn prompting technique, and trains a 6B supervised fine-tuning model on DACO dataset, and finds that the SFT model learns reasonable data analysis capabilities.
     </details>


     <details>
          <summary>Abstract</summary>
          Data analysis is a crucial analytical process essential for deriving insights from real-world databases. As shown in Figure 1, the need for data analysis typically arises from specific application scenarios, and requires diverse reasoning skills including mathematical reasoning, logical reasoning, and strategic reasoning. Existing work often focus on simple factual retrieval or arithmetic resolutions and thus are insufficient for addressing complex real-world queries. This work aims to propose new resources and benchmarks on this crucial yet challenging and under-explored task. Due to the prohibitively high cost of collecting expert annotations, we use large language models (LLMs) enhanced by code generation to automatically generate high-quality data analysis, which will later be refined by human annotators. We construct the DACO dataset, containing (1) 440 databases (of tabular data) collected from real-world scenarios, (2) ~2k automatically generated query-answer pairs that can serve as weak supervision for model training, and (3) a concentrated but high-quality test set with human refined annotations that serves as our main evaluation benchmark. Experiments show that while LLMs like GPT-4 exhibit promising data analysis capabilities, they are still evaluated as less helpful than human-written analysis on 58.1% cases. Leveraging our weak supervision data, we experiment with various fine-tuning methods, including supervised fine-tuning (SFT) and reinforcement learning from human feedback (RLHF). Our trained model outperforms existing baselines for table question answering, and RLHF further boosts the helpfulness of generated analysis on 58.5% cases.Data and code are released at https://github.com/shirley-wu/daco.
     </details>

56. **Faithful Logical Reasoning via Symbolic Chain-of-Thought** [[pdf]](http://arxiv.org/abs/2405.18357) `ACL 2024 Long Papers` (6 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          SymbCoT is proposed, a fully LLM-based framework that integrates symbolic expressions and logic rules with CoT for logical reasoning with LLMs, and is believed to be the first to combine symbolic expressions and rules into CoT for logical reasoning with LLMs.
     </details>


     <details>
          <summary>Abstract</summary>
          While the recent Chain-of-Thought (CoT) technique enhances the reasoning ability of large language models (LLMs) with the theory of mind, it might still struggle in handling logical reasoning that relies much on symbolic expressions and rigid deducing rules. To strengthen the logical reasoning capability of LLMs, we propose a novel Symbolic Chain-of-Thought, namely SymbCoT, a fully LLM-based framework that integrates symbolic expressions and logic rules with CoT prompting. Technically, building upon an LLM, SymbCoT 1) first translates the natural language context into the symbolic format, and then 2) derives a step-by-step plan to solve the problem with symbolic logical rules, 3) followed by a verifier to check the translation and reasoning chain. Via thorough evaluations on 5 standard datasets with both First-Order Logic and Constraint Optimization symbolic expressions, SymbCoT shows striking improvements over the CoT method consistently, meanwhile refreshing the current state-of-the-art performances. We further demonstrate that our system advances in more faithful, flexible, and explainable logical reasoning. To our knowledge, this is the first attempt at combining symbolic expressions and rules into CoT for logical reasoning with LLMs. Code is open at https://github.com/Aiden0526/SymbCoT.
     </details>

57. **How Far Can Transformers Reason? The Locality Barrier and Inductive Scratchpad** [[pdf]](http://arxiv.org/abs/2406.06467) `NeurIPS 2024` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The notion of 'distribution locality' is put forward to capture when weak learning is efficiently achievable by regular Transformers, where the locality measures the least number of tokens required in addition to the tokens histogram to correlate nontrivially with the target.
     </details>


     <details>
          <summary>Abstract</summary>
          Can Transformers predict new syllogisms by composing established ones? More generally, what type of targets can be learned by such models from scratch? Recent works show that Transformers can be Turing-complete in terms of expressivity, but this does not address the learnability objective. This paper puts forward the notion of 'distribution locality' to capture when weak learning is efficiently achievable by regular Transformers, where the locality measures the least number of tokens required in addition to the tokens histogram to correlate nontrivially with the target. As shown experimentally and theoretically under additional assumptions, distributions with high locality cannot be learned efficiently. In particular, syllogisms cannot be composed on long chains. Furthermore, we argue that (i) an agnostic scratchpad cannot help to break the locality, (ii) an educated scratchpad can help if it breaks the locality at each step, (iii) a notion of 'inductive scratchpad' can both break the locality and help with out-of-distribution generalization.
     </details>

58. **Lean Workbook: A large-scale Lean problem set formalized from natural language math problems** [[pdf]](http://arxiv.org/abs/2406.03847) `NeurIPS 2024` `Lean` (4 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a novel pipeline that iteratively generates and filters synthetic data to translate natural language mathematical problems into Lean 4 statements, and vice versa, and indicates that the synthetic data pipeline can provide useful training data and improve the performance of LLMs in translating and understanding complex mathematical problems and proofs.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models have demonstrated impressive capabilities across various natural language processing tasks, especially in solving mathematical problems. However, large language models are not good at math theorem proving using formal languages like Lean. A significant challenge in this area is the scarcity of training data available in these formal languages. To address this issue, we propose a novel pipeline that iteratively generates and filters synthetic data to translate natural language mathematical problems into Lean 4 statements, and vice versa. Our results indicate that the synthetic data pipeline can provide useful training data and improve the performance of LLMs in translating and understanding complex mathematical problems and proofs. Our final dataset contains about 57K formal-informal question pairs along with searched proof from the math contest forum and 21 new IMO questions. We open-source our code at \url{https://github.com/InternLM/InternLM-Math} and our data at \url{https://huggingface.co/datasets/InternLM/Lean-Workbook}.
     </details>

59. **In-Context Principle Learning from Mistakes** [[pdf]](https://openreview.net/forum?id=PAPY0cAB3C) `ICML 2024 Poster` (13 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          In-context learning (ICL, also known as few-shot prompting) has been the standard method of adapting LLMs to downstream tasks, by learning from a few input-output examples. Nonetheless, all ICL-based approaches only learn from correct input-output pairs. In this paper, we revisit this paradigm, by learning more from the few given input-output examples. We introduce Learning Principles (LEAP): First, we intentionally induce the model to make mistakes on these few examples; then we reflect on these mistakes, and learn explicit task-specific “principles” from them, which help solve similar problems and avoid common mistakes; finally, we prompt the model to answer unseen test questions using the original few-shot examples and these learned general principles. We evaluate LEAP on a wide range of benchmarks, including multi-hop question answering (Hotpot QA), textual QA (DROP), Big-Bench Hard reasoning, and math problems (GSM8K and MATH); in all these benchmarks, LEAP improves the strongest available LLMs such as GPT-3.5-turbo, GPT-4, GPT-4-turbo and Claude-2.1. For example, LEAP improves over the standard few-shot prompting using GPT-4 by 7.5% in DROP, and by 3.3% in HotpotQA. Importantly, LEAP does not require any more input or examples than the standard few-shot prompting settings.
     </details>

60. **Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning** [[pdf]](https://openreview.net/forum?id=t82Y3fmRtk) `ICML 2024 Poster` (8 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          R$^3$ progressively slides the start state of reasoning from a demonstration's end to its beginning, facilitating easier model exploration at all stages, allowing outcome supervision to offer step-level signals and precisely pinpoint errors.
     </details>


     <details>
          <summary>Abstract</summary>
          In this paper, we propose **R**$^3$: Learning **R**easoning through **R**everse Curriculum **R**einforcement Learning (RL), a novel method that employs only outcome supervision to achieve the benefits of process supervision for large language models. The core challenge in applying RL to complex reasoning is to identify a sequence of actions that result in positive rewards and provide appropriate supervision for optimization. Outcome supervision provides sparse rewards for final results without identifying error locations, whereas process supervision offers step-wise rewards but requires extensive manual annotation. **R**$^3$ overcomes these limitations by learning from correct demonstrations. Specifically, **R**$^3$ progressively slides the start state of reasoning from a demonstration's end to its beginning, facilitating easier model exploration at all stages. Thus, **R**$^3$ establishes a step-wise curriculum, allowing outcome supervision to offer step-level signals and precisely pinpoint errors. Using Llama2-7B, our method surpasses RL baseline on eight reasoning tasks by $4.1$ points on average. Notably, in program-based reasoning, 7B-scale models perform comparably to larger models or closed-source models with our **R**$^3$.
     </details>

61. **Case-Based or Rule-Based: How Do Transformers Do the Math?** [[pdf]](https://openreview.net/forum?id=4Vqr8SRfyX) `ICML 2024 Poster` (5 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A Rule-Following Fine-Tuning technique is proposed to teach transformers to perform rule-based reasoning and successfully enables LLMs fine-tuned on 1-5 digit addition to generalize to up to 12-digit addition with over 95% accuracy, which is over 40% higher than scratchpad.
     </details>


     <details>
          <summary>Abstract</summary>
          Despite the impressive performance in a variety of complex tasks, modern large language models (LLMs) still have trouble dealing with some math problems that are simple and intuitive for humans, such as addition. While we can easily learn basic *rules* of addition and apply them to new problems of any length, LLMs struggle to do the same. Instead, they may rely on similar *cases* seen in the training corpus for help. We define these two different reasoning mechanisms as "*rule-based reasoning*" and "*case-based reasoning*". Since rule-based reasoning is essential for acquiring systematic generalization ability, we aim to explore exactly whether transformers use rule-based or case-based reasoning for math problems. Through carefully designed intervention experiments on five math tasks, we confirm that transformers are performing case-based reasoning, no matter whether scratchpad is used, which aligns with the previous observations that transformers use subgraph matching/shortcut learning to reason. To mitigate such problems, we propose a Rule-Following Fine-Tuning (RFFT) technique to teach transformers to perform rule-based reasoning. Specifically, we provide explicit rules in the input and then instruct transformers to recite and follow the rules step by step. Through RFFT, we successfully enable LLMs fine-tuned on 1-5 digit addition to generalize to up to 12-digit addition with over 95% accuracy, which is over 40% higher than scratchpad. The significant improvement demonstrates that teaching LLMs to use rules explicitly helps them learn rule-based reasoning and generalize better in length. Code is available at https://github.com/GraphPKU/Case_or_Rule.
     </details>

62. **MAGDi: Structured Distillation of Multi-Agent Interaction Graphs Improves Reasoning in Smaller Language Models** [[pdf]](https://openreview.net/forum?id=ffLblkoCw8) `ICML 2024 Poster` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          MAGDi, a new method for structured distillation of the reasoning interactions between multiple LLMs into smaller LMs, improves the reasoning capabilities of smaller models and obtains larger improvements when applying self-consistency -- an inference technique that relies on model diversity.
     </details>


     <details>
          <summary>Abstract</summary>
          Multi-agent interactions between Large Language Model (LLM) agents have shown major improvements on diverse reasoning tasks. However, these involve long generations from multiple models across several rounds, making them expensive. Moreover, these multi-agent approaches fail to provide a final, single model for efficient inference. To address this, we introduce MAGDi, a new method for structured distillation of the reasoning interactions between multiple LLMs into smaller LMs. MAGDi teaches smaller models by representing multi-agent interactions as graphs, augmenting a base student model with a graph encoder, and distilling knowledge using three objective functions: next-token prediction, a contrastive loss between correct and incorrect reasoning, and a graph-based objective to model the interaction structure. Experiments on seven widely used commonsense and math reasoning benchmarks show that MAGDi improves the reasoning capabilities of smaller models, outperforming several methods that distill from a single teacher and multiple teachers. Moreover, MAGDi also demonstrates an order of magnitude higher efficiency over its teachers. We conduct extensive analyses to show that MAGDi (1) enhances the generalizability to out-of-domain tasks, (2) scales positively with the size and strength of the base student model, and (3) obtains larger improvements (via our multi-teacher training) when applying self-consistency – an inference technique that relies on model diversity.
     </details>

63. **Keypoint-based Progressive Chain-of-Thought Distillation for LLMs** [[pdf]](https://openreview.net/forum?id=tgsSKziIEa) `ICML 2024 Poster` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A token weighting module utilizing mask learning to encourage accurate mimicry of keypoint tokens by the student during distillation is proposed and an in-rationale progressive distillation strategy is developed, starting with training the student to generate the final reasoning steps and gradually extending to cover the entire rationale.
     </details>


     <details>
          <summary>Abstract</summary>
          Chain-of-thought distillation is a powerful technique for transferring reasoning abilities from large language models (LLMs) to smaller student models. Previous methods typically require the student to mimic the step-by-step rationale produced by LLMs, often facing the following challenges: (i) Tokens within a rationale vary in significance, and treating them equally may fail to accurately mimic keypoint tokens, leading to reasoning errors. (ii) They usually distill knowledge by consistently predicting all the steps in a rationale, which falls short in distinguishing the learning order of step generation. This diverges from the human cognitive progression of starting with easy tasks and advancing to harder ones, resulting in sub-optimal outcomes. To this end, we propose a unified framework, called KPOD, to address these issues. Specifically, we propose a token weighting module utilizing mask learning to encourage accurate mimicry of keypoint tokens by the student during distillation. Besides, we develop an in-rationale progressive distillation strategy, starting with training the student to generate the final reasoning steps and gradually extending to cover the entire rationale. To accomplish this, a weighted token generation loss is proposed to assess step reasoning difficulty, and a value function is devised to schedule the progressive distillation by considering both step difficulty and question diversity. Extensive experiments on four reasoning benchmarks illustrate our KPOD outperforms previous methods by a large margin.
     </details>

64. **Toward Adaptive Reasoning in Large Language Models with Thought Rollback** [[pdf]](https://openreview.net/forum?id=aoAPOOtN9E) `ICML 2024 Poster` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A new reasoning framework, called Thought Rollback (TR), is proposed, allowing LLMs to adaptively build thought structure while maintaining effective reasoning toward problem-solving under “hallucinations”.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have been routinely used to solve various tasks using step-by-step reasoning. However, the structure of intermediate reasoning steps, or *thoughts*, is rigid and unidirectional, such as chains, trees, or acyclic-directed graphs. Consequently, the resulting inflexible and forward-only reasoning may not address challenging tasks and fail when the LLM frequently gives false responses, i.e., hallucinations. This paper proposes a new reasoning framework, called *Thought Rollback* (TR), allowing LLMs to adaptively build thought structure while maintaining effective reasoning toward problem-solving under hallucinations. The core mechanism of TR is *rolling back thoughts*, which allows LLMs to perform error analysis on thoughts, and thus roll back to any previously mistaken thought for revision. Subsequently, by including such trial-and-error in the prompt to guide the LLM, each rollback leads to one more reliable reasoning path. Therefore, starting with a simple prompt without human annotations, LLM with TR adaptively and gradually explores thoughts for a correct solution. Comprehensive experiments on mathematical problems and multi-task reasoning demonstrate the state-of-the-art performance of TR in terms of problem-solving rate and interaction cost. For instance, the solving rate of GPT-4 with TR outperforms the current best by $9\%$ on the MATH dataset. The source code is available under the folder *examples/ThoughtRollback* of https://github.com/iQua/llmpebase.
     </details>

65. **Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models** [[pdf]](http://arxiv.org/abs/2310.04406) `ICML 2024 Poster` (77 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Experimental evaluation across diverse domains, including programming, interactive question-answering (QA), web navigation, and math, validates the effectiveness and generality of LATS in decision-making while maintaining competitive or improved reasoning performance.
     </details>


     <details>
          <summary>Abstract</summary>
          While language models (LMs) have shown potential across a range of decision-making tasks, their reliance on simple acting processes limits their broad deployment as autonomous agents. In this paper, we introduce Language Agent Tree Search (LATS) -- the first general framework that synergizes the capabilities of LMs in reasoning, acting, and planning. By leveraging the in-context learning ability of LMs, we integrate Monte Carlo Tree Search into LATS to enable LMs as agents, along with LM-powered value functions and self-reflections for proficient exploration and enhanced decision-making. A key feature of our approach is the incorporation of an environment for external feedback, which offers a more deliberate and adaptive problem-solving mechanism that surpasses the constraints of existing techniques. Our experimental evaluation across diverse domains, including programming, interactive question-answering (QA), web navigation, and math, validates the effectiveness and generality of LATS in decision-making while maintaining competitive or improved reasoning performance. Notably, LATS achieves state-of-the-art pass@1 accuracy (92.7%) for programming on HumanEval with GPT-4 and demonstrates gradient-free performance (average score of 75.9) comparable to gradient-based fine-tuning for web navigation on WebShop with GPT-3.5. Code can be found at https://github.com/lapisrocks/LanguageAgentTreeSearch
     </details>

66. **Reasoning or Reciting? Exploring the Capabilities and Limitations of Language Models Through Counterfactual Tasks** [[pdf]](https://aclanthology.org/2024.naacl-long.102) `NAACL 2024 Long Papers` (120 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          An evaluation framework based on “counterfactual” task variants that deviate from the default assumptions underlying standard tasks suggests that while current LMs may possess abstract task-solving skills to an extent, they often also rely on narrow, non-transferable procedures for task-solving.
     </details>


     <details>
          <summary>Abstract</summary>
          The impressive performance of recent language models across a wide range of tasks suggests that they possess a degree of abstract reasoning skills. Are these skills general and transferable, or specialized to specific tasks seen during pretraining? To disentangle these effects, we propose an evaluation framework based on “counterfactual” task variants that deviate from the default assumptions underlying standard tasks. Across a suite of 11 tasks, we observe nontrivial performance on the counterfactual variants, but nevertheless find that performance substantially and consistently degrades compared to the default conditions. This suggests that while current LMs may possess abstract task-solving skills to an extent, they often also rely on narrow, non-transferable procedures for task-solving. These results motivate a more careful interpretation of language model performance that teases apart these aspects.
     </details>

67. **AutoPRM: Automating Procedural Supervision for Multi-Step Reasoning via Controllable Question Decomposition** [[pdf]](https://aclanthology.org/2024.naacl-long.73) `NAACL 2024 Long Papers` (13 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel self-supervised framework **AutoPRM** is proposed that efficiently enhances the fine-tuning of LLMs for intricate reasoning challenges and proposes context-guided decoding to avoid reward tampering and guide the subquestion solver towards the solution of the holistic problem.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent advancements in large language models (LLMs) have shown promise in multi-step reasoning tasks, yet their reliance on extensive manual labeling to provide procedural feedback remains a significant impediment. To address this challenge, in this paper, we propose a novel self-supervised framework **AutoPRM** that efficiently enhances the fine-tuning of LLMs for intricate reasoning challenges. Specifically, **AutoPRM** first decomposes complex problems into more manageable subquestions with a controllable granularity switch, then sequentially apply reinforcement learning to iteratively improve the subquestion solver. Additionally, we propose context-guided decoding to avoid reward tampering and guide the subquestion solver towards the solution of the holistic problem. Extensive experiments show that **AutoPRM** significantly improves performance on mathematical and commonsense reasoning tasks over SOTA. More encouragingly, **AutoPRM** can be easily integrated with other orthogonal reasoning pipelines.
     </details>

68. **The ART of LLM Refinement: Ask, Refine, and Trust** [[pdf]](https://aclanthology.org/2024.naacl-long.327) `NAACL 2024 Long Papers` (13 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a reasoning with a refinement strategy called *ART: Ask, Refine, and Trust*, which asks necessary questions to decide when an LLM should *refine* its output, and uses it to affirm or deny *trust* in its refinement by ranking the refinement and the initial prediction.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have demonstrated remarkable generative abilities, but can they judge the quality of their own generations and self-improve?A popular concept, referred to as *self-refinement*, postulates that LLMs can detect and correct the errors in their generations when asked to do so. However, recent empirical evidence points in the opposite direction, suggesting that LLMs often struggle to accurately identify errors when reasoning is involved. To address this, we propose a reasoning with a refinement strategy called *ART: Ask, Refine, and Trust*, which *asks* necessary questions to decide when an LLM should *refine* its output, and uses it to affirm or deny *trust* in its refinement by ranking the refinement and the initial prediction. On two multistep reasoning tasks of mathematical word problems (GSM8K) and question answering (StrategyQA), *ART* achieves a performance gain of +5 points over self-refinement baselines, while using a much smaller model as the decision maker. We believe that *ART* with smaller models, making refinement decisions can be a cost-effective alternative to fine-tuning LLMs.
     </details>

69. **Teaching Language Models to Self-Improve through Interactive Demonstrations** [[pdf]](https://aclanthology.org/2024.naacl-long.287) `NAACL 2024 Long Papers` (11 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          TriPosT is introduced, a training algorithm that endows smaller models with such self-improvement ability, and it is shown that this approach can improve LLaMA-7B’s performance on math and reasoning tasks by up to 7.13%.
     </details>


     <details>
          <summary>Abstract</summary>
          The self-improving ability of large language models (LLMs), enabled by prompting them to analyze and revise their own outputs, has garnered significant interest in recent research. However, this ability has been shown to be absent and difficult to learn for smaller models, thus widening the performance gap between state-of-the-art LLMs and more cost-effective and faster ones. To reduce this gap, we introduce TriPosT, a training algorithm that endows smaller models with such self-improvement ability, and show that our approach can improve LLaMA-7B’s performance on math and reasoning tasks by up to 7.13%. In contrast to prior work, we achieve this by using the smaller model to interact with LLMs to collect feedback and improvements on *its own generations*. We then replay this experience to train the small model. Our experiments on four math and reasoning datasets show that the interactive experience of learning from and correcting its *own* mistakes is crucial for small models to improve their performance.
     </details>

70. **PaD: Program-aided Distillation Can Teach Small Models Reasoning Better than Chain-of-thought Fine-tuning** [[pdf]](https://aclanthology.org/2024.naacl-long.142) `NAACL 2024 Long Papers` (10 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes Program-aided Distillation (PaD), which introduces reasoning programs to suppress the errors in distilled data, and thus achieves better distillation quality for reasoning tasks, and demonstrates that smaller models using PaD can not only outperform certain LLMs but also achieve strong improvement over baselines with a significantly smaller scale of parameters and data.
     </details>


     <details>
          <summary>Abstract</summary>
          While large language models (LLMs) excel in various natural language processing tasks, their huge size and the inaccessibility of parameters present challenges for practical deployment. Previous studies try to distill task-specific ability from LLMs to smaller models, using data synthesis and chain-of-thought (CoT) fine-tuning. However, synthetic CoT data often contains faulty reasoning, which deteriorates the quality of distillation, especially in reasoning capabilities. In this work, we propose Program-aided Distillation (PaD), which introduces reasoning programs to suppress the errors in distilled data, and thus achieves better distillation quality for reasoning tasks. In PaD, we utilize the reasoning program to substitute the CoT, allowing automated error checking of synthetic data. Further, through error injecting and further training, the small distilling model could iteratively self-refine the reasoning. Moreover, we conduct a step-wise beam search by step-by-step verifying to acquire more exact reasoning chains. We evaluate PaD on arithmetic reasoning, symbolic reasoning, and general ability.Experimental results demonstrate that smaller models using PaD can not only outperform certain LLMs (e.g., LLaMA-1 13B) but also achieve strong improvement over baselines with a significantly smaller scale of parameters and data. The source code is publicly available athttps://github.com/Xuekai-Zhu/pad.
     </details>

71. **Instructing Large Language Models to Identify and Ignore Irrelevant Conditions** [[pdf]](https://aclanthology.org/2024.naacl-long.379) `NAACL 2024 Long Papers` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel approach named I^3C is proposed that instructs LLMs to identify and ignore irrelevant conditions and develops I^3C-Select that selects the most confusing problems based on the semantic relevance measurement.
     </details>


     <details>
          <summary>Abstract</summary>
          Math word problem (MWP) solving requires generating a reasoning path based on a given problem description that often contains irrelevant conditions.Existing chain-of-thought (CoT) prompting methods elicited multi-step reasoning abilities of large language models (LLMs) to solve MWPs.However, they were seriously confused by the irrelevant conditions, resulting in low accuracy.In this paper, we propose a novel approach named I3C that instructs LLMs to identify and ignore irrelevant conditions.It identifies a set of irrelevant condition candidates that have a weak semantic relevance with the question.Then it prompts LLMs to verify the irrelevant conditions.Lastly it instructs the LLMs with the verification on relevant and irrelevant conditions to avoid confusion and improve reasoning paths.Moreover, we propose to select (problem, reasoning paths) pairs as demonstrations to enhance I3C with few-shot reasoning. We develop I3C-Select that selects the most confusing problems based on the semantic relevance measurement.We conduct extensive experiments on eight MWP datasets.I3C can be combined with any CoT prompting methods to improve the performance of solving MWPs.Notably, with GPT-3.5-Turbo and I3C-Select, we achieve an accuracy of 96.0 and 94.1 on GSM-IC2-1K and GSM-ICM-1K, respectively, significantly outperforming the state-of-the-art few-shot prompting method Complex-CoT by +11.7 and +11.1.Our implementation is made publicly available at https://wzy6642.github.io/I3C.github.io/.
     </details>

72. **Program-Aided Reasoners (Better) Know What They Know** [[pdf]](https://aclanthology.org/2024.naacl-long.125) `NAACL 2024 Long Papers` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper compares the calibration of Program Aided Language Models (PAL) and text-based Chain-of-thought (COT) prompting techniques over 5 datasets and 2 model types - LLaMA models and OpenAI models and demonstrates that, in the majority of cases, program-aided reasoners better know what they know than text-based counterparts.
     </details>


     <details>
          <summary>Abstract</summary>
          Prior work shows that program-aided reasoning, in which large language models (LLMs) are combined with programs written in programming languages such as Python, can significantly improve accuracy on various reasoning tasks. However, while accuracy is essential, it is also important for such reasoners to “know what they know”, which can be quantified through the calibration of the model. In this paper, we compare the calibration of Program Aided Language Models (PAL) and text-based Chain-of-thought (COT) prompting techniques over 5 datasets and 2 model types - LLaMA models and OpenAI models. Our results indicate that PAL leads to improved calibration in 75% of the instances. Our analysis uncovers that prompting styles that produce lesser diversity in generations also have more calibrated results, and thus we also experiment with inducing lower generation diversity using temperature scaling and find that for certain temperatures, PAL is not only more accurate but is also more calibrated than COT. Overall, we demonstrate that, in the majority of cases, program-aided reasoners better know what they know than text-based counterparts.
     </details>

73. **Transformers Can Do Arithmetic with the Right Embeddings** [[pdf]](http://arxiv.org/abs/2405.17399) `NeurIPS 2024` (12 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work fixes the problem of poor performance of transformers on arithmetic tasks by adding an embedding to each digit that encodes its position relative to the start of the number, and shows that this fix enables architectural modifications such as input injection and recurrent layers to improve performance.
     </details>


     <details>
          <summary>Abstract</summary>
          The poor performance of transformers on arithmetic tasks seems to stem in large part from their inability to keep track of the exact position of each digit inside of a large span of digits. We mend this problem by adding an embedding to each digit that encodes its position relative to the start of the number. In addition to the boost these embeddings provide on their own, we show that this fix enables architectural modifications such as input injection and recurrent layers to improve performance even further.With positions resolved, we can study the logical extrapolation ability of transformers. Can they solve arithmetic problems that are larger and more complex than those in their training data? We find that training on only 20 digit numbers with a single GPU for one day, we can reach state-of-the-art performance, achieving up to 99% accuracy on 100 digit addition problems. Finally, we show that these gains in numeracy also unlock improvements on other multi-step reasoning tasks including sorting and multiplication.
     </details>

74. **Autoformalizing Euclidean Geometry** [[pdf]](http://arxiv.org/abs/2405.17216) `ICML 2024` `Lean` (2 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A neuro-symbolic framework for autoformalizing Euclidean geometry, which combines domain knowledge, SMT solvers, and large language models (LLMs), is introduced and automatic semantic evaluation for autoformalized theorem statements is provided.
     </details>


     <details>
          <summary>Abstract</summary>
          Autoformalization involves automatically translating informal math into formal theorems and proofs that are machine-verifiable. Euclidean geometry provides an interesting and controllable domain for studying autoformalization. In this paper, we introduce a neuro-symbolic framework for autoformalizing Euclidean geometry, which combines domain knowledge, SMT solvers, and large language models (LLMs). One challenge in Euclidean geometry is that informal proofs rely on diagrams, leaving gaps in texts that are hard to formalize. To address this issue, we use theorem provers to fill in such diagrammatic information automatically, so that the LLM only needs to autoformalize the explicit textual steps, making it easier for the model. We also provide automatic semantic evaluation for autoformalized theorem statements. We construct LeanEuclid, an autoformalization benchmark consisting of problems from Euclid's Elements and the UniGeo dataset formalized in the Lean proof assistant. Experiments with GPT-4 and GPT-4V show the capability and limitations of state-of-the-art LLMs on autoformalizing geometry problems. The data and code are available at https://github.com/loganrjmurphy/LeanEuclid.
     </details>

75. **Chain-of-Thought Reasoning Without Prompting** [[pdf]](http://arxiv.org/abs/2402.10200) `NeurIPS 2024` (38 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          These findings reveal that, intriguingly, CoT reasoning paths can be elicited from pre-trained LLMs by simply altering the decoding process, and that the presence of a CoT in the decoding path correlates with a higher confidence in the model's decoded answer.
     </details>


     <details>
          <summary>Abstract</summary>
          In enhancing the reasoning capabilities of large language models (LLMs), prior research primarily focuses on specific prompting techniques such as few-shot or zero-shot chain-of-thought (CoT) prompting. These methods, while effective, often involve manually intensive prompt engineering. Our study takes a novel approach by asking: Can LLMs reason effectively without any prompting? Our findings reveal that, intriguingly, CoT reasoning paths can be elicited from pre-trained LLMs by simply altering the \textit{decoding} process. Rather than conventional greedy decoding, we investigate the top-$k$ alternative tokens, uncovering that CoT paths are frequently inherent in these sequences. This approach not only bypasses the confounders of prompting but also allows us to assess the LLMs' \textit{intrinsic} reasoning abilities. Moreover, we observe that the presence of a CoT in the decoding path correlates with a higher confidence in the model's decoded answer. This confidence metric effectively differentiates between CoT and non-CoT paths. Extensive empirical studies on various reasoning benchmarks show that the proposed CoT-decoding effectively elicits reasoning capabilities from language models, which were previously obscured by standard greedy decoding.
     </details>

76. **MAmmoTH2: Scaling Instructions from the Web** [[pdf]](http://arxiv.org/abs/2405.03548) `NeurIPS 2024` (34 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work demonstrates how to harvest large-scale, high-quality instruction data without costly human annotation or GPT-4 distillation, providing a new paradigm for building better instruction tuning data.
     </details>


     <details>
          <summary>Abstract</summary>
          Instruction tuning improves the reasoning abilities of large language models (LLMs), with data quality and scalability being the crucial factors. Most instruction tuning data come from human crowd-sourcing or GPT-4 distillation. We propose a paradigm to efficiently harvest 10 million naturally existing instruction data from the pre-training web corpus to enhance LLM reasoning. Our approach involves (1) recalling relevant documents, (2) extracting instruction-response pairs, and (3) refining the extracted pairs using open-source LLMs. Fine-tuning base LLMs on this dataset, we build MAmmoTH2 models, which significantly boost performance on reasoning benchmarks. Notably, MAmmoTH2-7B’s (Mistral) performance increases from 11% to 36.7% on MATH and from 36% to 68.4% on GSM8K without training on any in-domain data. Further training MAmmoTH2 on public instruction tuning datasets yields MAmmoTH2-Plus, achieving state-of-the-art performance on several reasoning and chatbot benchmarks. Our work demonstrates how to harvest large-scale, high-quality instruction data without costly human annotation or GPT-4 distillation, providing a new paradigm for building better instruction tuning data.
     </details>

77. **AlphaMath Almost Zero: Process Supervision Without Process** [[pdf]](http://arxiv.org/abs/2405.03553) `NeurIPS 2024` (16 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This study proposes an innovative framework, AlphaMath, that bypasses the need for process annotations (from humans or GPTs) by leveraging Monte Carlo Tree Search (MCTS), and focuses on unleashing the potential of a well-pretrained LLM to autonomously enhance its mathematical reasoning.
     </details>


     <details>
          <summary>Abstract</summary>
          Although recent advancements in large language models (LLMs) have significantly improved their performance on various tasks, they still face challenges with complex and symbolic multi-step reasoning, particularly in mathematical reasoning. To bolster the mathematical reasoning capabilities of LLMs, most existing efforts concentrate on seeking assistance from either domain experts or GPT-4 for high-quality process-supervised data, which is not only expensive but also labor-intensive. In our study, we propose an innovative framework, AlphaMath, that bypasses the need for process annotations (from humans or GPTs) by leveraging Monte Carlo Tree Search (MCTS). This framework focuses on unleashing the potential of a well-pretrained LLM to autonomously enhance its mathematical reasoning. Specifically, we integrate a value model with the LLM, automatically generating both process supervision and step-level evaluation signals in MCTS. Furthermore, we propose an efficient inference strategy—step-level beam search, where the value model is crafted to assist the policy model (i.e., LLM) in navigating more effective reasoning paths, rather than solely relying on prior probabilities. The experimental results on both in-domain and out-of-domain datasets demonstrate that even without GPT-4 or human-annotated process supervision, our AlphaMath framework achieves comparable or superior results to previous state-of-the-art methods.
     </details>

78. **JiuZhang3.0: Efficiently Improving Mathematical Reasoning by Training Small Data Synthesis Models** [[pdf]](http://arxiv.org/abs/2405.14365) `NeurIPS 2024` (10 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes an efficient way that trains a small LLM for math problem synthesis, to efficiently generate sufficient high-quality pre-training data, and craft a set of prompts based on human education stages to guide GPT-4, to synthesize problems covering diverse math knowledge and difficulty levels.
     </details>


     <details>
          <summary>Abstract</summary>
          Mathematical reasoning is an important capability of large language models~(LLMs) for real-world applications.To enhance this capability, existing work either collects large-scale math-related texts for pre-training, or relies on stronger LLMs (\eg GPT-4) to synthesize massive math problems. Both types of work generally lead to large costs in training or synthesis.To reduce the cost, based on open-source available texts, we propose an efficient way that trains a small LLM for math problem synthesis, to efficiently generate sufficient high-quality pre-training data.To achieve it, we create a dataset using GPT-4 to distill its data synthesis capability into the small LLM.Concretely, we craft a set of prompts based on human education stages to guide GPT-4, to synthesize problems covering diverse math knowledge and difficulty levels.Besides, we adopt the gradient-based influence estimation method to select the most valuable math-related texts.The both are fed into GPT-4 for creating the knowledge distillation dataset to train the small LLM.We leverage it to synthesize 6 million math problems for pre-training our JiuZhang3.0 model, which only needs to invoke GPT-4 API 9.3k times and pre-train on 4.6B data.Experimental results have shown that JiuZhang3.0 achieves state-of-the-art performance on several mathematical reasoning datasets, under both natural language reasoning and tool manipulation settings.
     </details>

79. **Proving Theorems Recursively** [[pdf]](http://arxiv.org/abs/2405.14414) `NeurIPS 2024` `Isabelle` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          POETRY (PrOvE Theorems RecursivelY), which proves theorems in a recursive, level-by-level manner in the Isabelle theorem prover, is proposed, which allows the theorem to be tackled incrementally by outlining the overall theorem at the first level and then solving the intermediate conjectures at deeper levels.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent advances in automated theorem proving leverages language models to explore expanded search spaces by step-by-step proof generation. However, such approaches are usually based on short-sighted heuristics (e.g., log probability or value function scores) that potentially lead to suboptimal or even distracting subgoals, preventing us from finding longer proofs. To address this challenge, we propose POETRY (PrOvE Theorems RecursivelY), which proves theorems in a recursive, level-by-level manner in the Isabelle theorem prover. Unlike previous step-by-step methods, POETRY searches for a verifiable sketch of the proof at each level and focuses on solving the current level's theorem or conjecture. Detailed proofs of intermediate conjectures within the sketch are temporarily replaced by a placeholder tactic called sorry, deferring their proofs to subsequent levels. This approach allows the theorem to be tackled incrementally by outlining the overall theorem at the first level and then solving the intermediate conjectures at deeper levels. Experiments are conducted on the miniF2F and PISA datasets and significant performance gains are observed in our POETRY approach over state-of-the-art methods. POETRY on miniF2F achieves an average proving success rate improvement of 5.1%. Moreover, we observe a substantial increase in the maximum proof length found by POETRY, from 10 to 26.
     </details>

80. **Trajectory Volatility for Out-of-Distribution Detection in Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2405.14039) `NeurIPS 2024` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A trajectory-based method TV score is proposed, which uses trajectory volatility for OOD detection in mathematical reasoning and outperforms all traditional algorithms on GLMs under mathematical reasoning scenarios and can be extended to more applications with high-density features in output spaces, such as multiple-choice questions.
     </details>


     <details>
          <summary>Abstract</summary>
          Real-world data deviating from the independent and identically distributed (\textit{i.i.d.}) assumption of in-distribution training data poses security threats to deep networks, thus advancing out-of-distribution (OOD) detection algorithms. Detection methods in generative language models (GLMs) mainly focus on uncertainty estimation and embedding distance measurement, with the latter proven to be most effective in traditional linguistic tasks like summarization and translation. However, another complex generative scenario mathematical reasoning poses significant challenges to embedding-based methods due to its high-density feature of output spaces, but this feature causes larger discrepancies in the embedding shift trajectory between different samples in latent spaces. Hence, we propose a trajectory-based method TV score, which uses trajectory volatility for OOD detection in mathematical reasoning. Experiments show that our method outperforms all traditional algorithms on GLMs under mathematical reasoning scenarios and can be extended to more applications with high-density features in output spaces, such as multiple-choice questions.
     </details>

81. **RESPROMPT: Residual Connection Prompting Advances Multi-Step Reasoning in Large Language Models** [[pdf]](http://arxiv.org/abs/2310.04743) `NAACL 2023 Long` (4 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes Residual Connection Prompting (ResPrompt), a new prompting strategy that advances multi-step reasoning in LLMs by integrating necessary connections–links present in reasoning graph but missing in the linear CoT flow–into the prompts.
     </details>


     <details>
          <summary>Abstract</summary>
          Chain-of-thought (CoT) has impressively unlocked the reasoning potential of large language models (LLMs). Yet, it falls short when tackling problems that require multiple reasoning steps. This limitation arises from the complex nature of multi-step reasoning processes: later stages often depend not only on the immediately preceding step, but also on the results from several steps earlier. Such complexities indicate the reasoning process is naturally a graph. The almost linear structure of CoT, however, struggles to capture this complex reasoning graph. To address this challenge, we propose Residual Connection Prompting (ResPrompt), a new prompting strategy that advances multi-step reasoning in LLMs. The core of our idea is to reconstruct the reasoning graph within prompts. We achieve this by integrating necessary connections–links present in reasoning graph but missing in the linear CoT flow–into the prompts. Termed “residual connections”, these links can transform linear CoT into the complex reasoning graphs that multi-step problems entail. On benchmarks across math, sequential, and commonsense domains, ResPrompt demonstrates clear improvements in multi-step reasoning compared with CoT. Through extensive ablation studies and analyses, we pinpoint how to effectively build residual connections and also identify situations where it might be unnecessary.
     </details>

82. **A Careful Examination of Large Language Model Performance on Grade School Arithmetic** [[pdf]](https://arxiv.org/abs/2405.00332v3) `NeurIPS 2024` (46 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          GSM1k is designed to mirror the style and complexity of the established GSM8k benchmark, the gold standard for measuring elementary mathematical reasoning, and ensures that the two benchmarks are comparable across important metrics such as human solve rates, number of steps in solution, answer magnitude, and more.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have achieved impressive success on many benchmarks for mathematical reasoning.However, there is growing concern that some of this performance actually reflects dataset contamination, where data closely resembling benchmark questions leaks into the training data, instead of true reasoning ability.To investigate this claim rigorously, we commission Grade School Math 1000 (GSM1k). GSM1k is designed to mirror the style and complexity of the established GSM8k benchmark,the gold standard for measuring elementary mathematical reasoning. We ensure that the two benchmarks are comparable across important metrics such as human solve rates, number of steps in solution, answer magnitude, and more.When evaluating leading open- and closed-source LLMs on GSM1k, we observe accuracy drops of up to 8%, with several families of models showing evidence of systematic overfitting across almost all model sizes.Further analysis suggests a positive relationship (Spearman's r^2=0.36) between a model's probability of generating an example from GSM8k and its performance gap between GSM8k and GSM1k, suggesting that some models may have partially memorized GSM8k.Nevertheless, many models, especially those on the frontier, show minimal signs of overfitting, and all models broadly demonstrate generalization to novel math problems guaranteed to not be in their training data.
     </details>

83. **Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing** [[pdf]](http://arxiv.org/abs/2404.12253) `NeurIPS 2024` (20 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          AlphaLLM is introduced, which integrates Monte Carlo Tree Search with LLMs to establish a self-improving loop, thereby enhancing the capabilities of LLMs without additional annotations, showing the potential for self-improvement in LLMs.
     </details>


     <details>
          <summary>Abstract</summary>
          Despite the impressive capabilities of Large Language Models (LLMs) on various tasks, they still struggle with scenarios that involves complex reasoning and planning. Self-correction and self-learning emerge as viable solutions, employing strategies that allow LLMs to refine their outputs and learn from self-assessed rewards. Yet, the efficacy of LLMs in self-refining its response, particularly in complex reasoning and planning task, remains dubious. In this paper, we introduce AlphaLLM for the self-improvements of LLMs, which integrates Monte Carlo Tree Search (MCTS) with LLMs to establish a self-improving loop, thereby enhancing the capabilities of LLMs without additional annotations. Drawing inspiration from the success of AlphaGo, AlphaLLM addresses the unique challenges of combining MCTS with LLM for self-improvement, including data scarcity, the vastness search spaces of language tasks, and the subjective nature of feedback in language tasks. AlphaLLM is comprised of prompt synthesis component, an efficient MCTS approach tailored for language tasks, and a trio of critic models for precise feedback. Our experimental results in mathematical reasoning tasks demonstrate that AlphaLLM significantly enhances the performance of LLMs without additional annotations, showing the potential for self-improvement in LLMs.
     </details>

84. **Paraphrase and Solve: Exploring and Exploiting the Impact of Surface Form on Mathematical Reasoning in Large Language Models** [[pdf]](http://arxiv.org/abs/2404.11500) `NAACL 2024 Long Papers` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper proposes Self-Consistency-over-Paraphrases (SCoP), which diversifies reasoning paths from specific surface forms of the problem and shows that SCoP improves mathematical reasoning performance over vanilla self-consistency, particularly for problems initially deemed unsolvable.
     </details>


     <details>
          <summary>Abstract</summary>
          This paper studies the relationship between the surface form of a mathematical problem and its solvability by large language models. We find that subtle alterations in the surface form can significantly impact the answer distribution and the solve rate, exposing the language model’s lack of robustness and sensitivity to the surface form in reasoning through complex problems. To improve mathematical reasoning performance, we propose Self-Consistency-over-Paraphrases (SCoP), which diversifies reasoning paths from specific surface forms of the problem. We evaluate our approach on four mathematics reasoning benchmarks over three large language models and show that SCoP improves mathematical reasoning performance over vanilla self-consistency, particularly for problems initially deemed unsolvable. Finally, we provide additional experiments and discussion regarding problem difficulty and surface forms, including cross-model difficulty agreement and paraphrasing transferability, and Variance of Variations (VOV) for language model evaluation.
     </details>

85. **Learn from Failure: Fine-tuning LLMs with Trial-and-Error Data for Intuitionistic Propositional Logic Proving** [[pdf]](http://arxiv.org/abs/2404.07382) `ACL 2024 Long Papers` `Lean` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Facing the lack of such trial-and-error data in existing open-source theorem-proving datasets, this paper curate a dataset on intuitionistic propositional logic theorems and formalize it in Lean, such that it can reliably check the correctness of proofs.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent advances in Automated Theorem Proving have shown the effectiveness of leveraging a (large) language model that generates tactics (i.e. proof steps) to search through proof states. The current model, while trained solely on successful proof paths, faces a discrepancy at the inference stage, as it must sample and try various tactics at each proof state until finding success, unlike its training which does not incorporate learning from failed attempts. Intuitively, a tactic that leads to a failed search path would indicate that similar tactics should receive less attention during the following trials. In this paper, we demonstrate the benefit of training models that additionally learn from failed search paths. Facing the lack of such trial-and-error data in existing open-source theorem-proving datasets, we curate a dataset on intuitionistic propositional logic theorems and formalize it in Lean, such that we can reliably check the correctness of proofs. We compare our model trained on relatively short trial-and-error information (TrialMaster) with models trained only on the correct paths and discover that the former solves more unseen theorems with lower trial searches.
     </details>

86. **A Symbolic Framework for Evaluating Mathematical Reasoning and Generalisation with Transformers** [[pdf]](http://arxiv.org/abs/2305.12563) `NAACL 2024 Long Papers` (6 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The results suggest that the in-distribution performance of smaller open-source models may potentially rival GPT by incorporating appropriately structured derivation dependencies during training, and highlight a shared weakness between BERT and GPT involving a relative inability to decode indirect references to mathematical entities.
     </details>


     <details>
          <summary>Abstract</summary>
          This paper proposes a methodology for generating and perturbing detailed derivations of equations at scale, aided by a symbolic engine, to evaluate the generalisability of Transformers to out-of-distribution mathematical reasoning problems. Instantiating the framework in the context of sequence classification tasks, we compare the capabilities of GPT-4, GPT-3.5, and a canon of fine-tuned BERT models, exploring the relationship between specific operators and generalisation failure via the perturbation of reasoning aspects such as symmetry and variable surface forms. Surprisingly, our empirical evaluation reveals that the average in-distribution performance of fine-tuned models surpasses GPT-3.5, and rivals GPT-4. However, perturbations to input reasoning can reduce their performance by up to 80 F1 points. Overall, the results suggest that the in-distribution performance of smaller open-source models may potentially rival GPT by incorporating appropriately structured derivation dependencies during training, and highlight a shared weakness between BERT and GPT involving a relative inability to decode indirect references to mathematical entities. We release the full codebase, constructed datasets, and fine-tuned models to encourage future progress in the field.
     </details>

87. **Multi-Operational Mathematical Derivations in Latent Space** [[pdf]](http://arxiv.org/abs/2311.01230) `NAACL 2024 Long` (5 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper investigates how different encoding mechanisms can approximate expression manipulation in latent space, exploring the trade-off between learning different operators and specialising within single operations, as well as the ability to support multi-step derivations and out-of-distribution generalisation.
     </details>


     <details>
          <summary>Abstract</summary>
          This paper investigates the possibility of approximating multiple mathematical operations in latent space for expression derivation. To this end, we introduce different multi-operational representation paradigms, modelling mathematical operations as explicit geometric transformations. By leveraging a symbolic engine, we construct a large-scale dataset comprising 1.7M derivation steps stemming from 61K premises and 6 operators, analysing the properties of each paradigm when instantiated with state-of-the-art neural encoders.Specifically, we investigate how different encoding mechanisms can approximate expression manipulation in latent space, exploring the trade-off between learning different operators and specialising within single operations, as well as the ability to support multi-step derivations and out-of-distribution generalisation. Our empirical analysis reveals that the multi-operational paradigm is crucial for disentangling different operators, while discriminating the conclusions for a single operation is achievable in the original expression encoder. Moreover, we show that architectural choices can heavily affect the training dynamics, structural organisation, and generalisation of the latent space, resulting in significant variations across paradigms and classes of encoders.
     </details>

88. **MATHSENSEI: A Tool-Augmented Large Language Model for Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2402.17231) `NAACL 2024 Long Papers` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is observed that TALMs are not as effective for simpler math word problems (in GSM-8K), and the benefit increases as the complexity and required knowledge increases (progressively over AQuA, MMLU-Math, and higher level complex questions in MATH).
     </details>


     <details>
          <summary>Abstract</summary>
          Tool-augmented Large Language Models (TALMs) are known to enhance the skillset of large language models (LLMs), thereby, leading to their improved reasoning abilities across many tasks. While, TALMs have been successfully employed in different question-answering benchmarks, their efficacy on complex mathematical reasoning benchmarks, and the potential complementary benefits offered by tools for knowledge retrieval and mathematical equation solving are open research questions. In this work, we present MathSensei, a tool-augmented large language model for mathematical reasoning. We study the complementary benefits of the tools - knowledge retriever (Bing Web Search), program generator + executor (Python), and symbolic equation solver (Wolfram-Alpha API) through evaluations on mathematical reasoning datasets. We perform exhaustive ablations on MATH, a popular dataset for evaluating mathematical reasoning on diverse mathematical disciplines. We also conduct experiments involving well-known tool planners to study the impact of tool sequencing on the model performance. MathSensei achieves 13.5% better accuracy over gpt-3.5-turbo with Chain-of-Thought on the MATH dataset. We further observe that TALMs are not as effective for simpler math word problems (in GSM-8K), and the benefit increases as the complexity and required knowledge increases (progressively over AQuA, MMLU-Math, and higher level complex questions in MATH). The code and data are available at https://github.com/Debrup-61/MathSensei.
     </details>

89. **Don't Trust: Verify -- Grounding LLM Quantitative Reasoning with Autoformalization** [[pdf]](http://arxiv.org/abs/2403.18120) `ICLR 2024` `Isabelle` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Large language models (LLM), such as Google's Minerva and OpenAI's GPT families, are becoming increasingly capable of solving mathematical quantitative reasoning problems. However, they still make unjustified logical and computational errors in their reasoning steps and answers. In this paper, we leverage the fact that if the training corpus of LLMs contained sufficiently many examples of formal mathematics (e.g. in Isabelle, a formal theorem proving environment), they can be prompted to translate i.e. autoformalize informal mathematical statements into formal Isabelle code --- which can be verified automatically for internal consistency. This provides a mechanism to automatically reject solutions whose formalized versions are inconsistent within themselves or with the formalized problem statement. We evaluate our method on GSM8K, MATH and MultiArith datasets and demonstrate that our approach provides a consistently better heuristic than vanilla majority voting --- the previously best method to identify correct answers, by more than 12\% on GSM8K. In our experiments it improves results consistently across all datasets and LLM model sizes.
     </details>

90. **MUSTARD: Mastering Uniform Synthesis of Theorem and Proof Data** [[pdf]](http://arxiv.org/abs/2402.08957) `ICLR 2024` `Lean` (17 cite) (3 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces MUSTARD, a data generation framework that masters uniform synthesis of theorem and proof data of high quality and diversity, and performs extensive analysis and demonstrates that MUSTARD generates validated high-quality step-by-step data.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent large language models (LLMs) have witnessed significant advancement in various tasks, including mathematical reasoning and theorem proving. As these two tasks require strict and formal multi-step inference, they are appealing domains for exploring the reasoning ability of LLMs but still face important challenges. Previous studies such as Chain-of-Thought (CoT) have revealed the effectiveness of intermediate steps guidance. However, such step-wise annotation requires heavy labor, leading to insufficient training steps for current benchmarks. To fill this gap, this work introduces MUSTARD, a data generation framework that masters uniform synthesis of theorem and proof data of high quality and diversity. MUSTARD synthesizes data in three stages: (1) It samples a few mathematical concept seeds as the problem category. (2) Then, it prompts a generative language model with the sampled concepts to obtain both the problems and their step-wise formal solutions. (3) Lastly, the framework utilizes a proof assistant (e.g., Lean Prover) to filter the valid proofs. With the proposed MUSTARD, we present a theorem-and-proof benchmark MUSTARDSAUCE with 5,866 valid data points. Each data point contains an informal statement, an informal proof, and a translated formal proof that passes the prover validation. We perform extensive analysis and demonstrate that MUSTARD generates validated high-quality step-by-step data. We further apply the MUSTARDSAUCE for fine-tuning smaller language models. The fine-tuned Llama 2-7B achieves a 15.41% average relative performance gain in automated theorem proving, and 8.18% in math word problems. Codes and data are available at https://github.com/Eleanor-H/MUSTARD.
     </details>

91. **MathScale: Scaling Instruction Tuning for Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2403.02884) `ICML 2024` (14 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes MathScale, a simple and scalable method to create high-quality mathematical reasoning data using frontier LLMs, and applies MathScaleQA to fine-tune open-source LLMs (e.g., LLaMA-2 and Mistral), resulting in significantly improved capabilities in mathematical reasoning.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have demonstrated remarkable capabilities in problem-solving. However, their proficiency in solving mathematical problems remains inadequate. We propose MathScale, a simple and scalable method to create high-quality mathematical reasoning data using frontier LLMs (e.g., GPT-3.5). Inspired by the cognitive mechanism in human mathematical learning, it first extracts topics and knowledge points from seed math questions and then build a concept graph, which is subsequently used to generate new math questions. MathScale exhibits effective scalability along the size axis of the math dataset that we generate. As a result, we create a mathematical reasoning dataset (MathScaleQA) containing two million math question-answer pairs. To evaluate mathematical reasoning abilities of LLMs comprehensively, we construct MWPBench, a benchmark of Math Word Problems, which is a collection of 9 datasets (including GSM8K and MATH) covering K-12, college, and competition level math problems. We apply MathScaleQA to fine-tune open-source LLMs (e.g., LLaMA-2 and Mistral), resulting in significantly improved capabilities in mathematical reasoning. Evaluated on MWPBench, MathScale-7B achieves state-of-the-art performance across all datasets, surpassing its best peers of equivalent size by 42.8% in micro average accuracy and 43.6% in macro average accuracy, respectively.
     </details>

92. **Do Large Language Models Latently Perform Multi-Hop Reasoning?** [[pdf]](http://arxiv.org/abs/2402.16837) `ACL 2024 Long Papers` (28 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Strong evidence of latent multi-hop reasoning for the prompts of certain relation types is found, with the reasoning pathway used in more than 80% of the prompts, however, the utilization is highly contextual, varying across different types of prompts.
     </details>


     <details>
          <summary>Abstract</summary>
          We study whether Large Language Models (LLMs) latently perform multi-hop reasoning with complex prompts such as “The mother of the singer of ‘Superstition’ is”. We look for evidence of a latent reasoning pathway where an LLM (1) latently identifies “the singer of ‘Superstition’” as Stevie Wonder, the bridge entity, and (2) uses its knowledge of Stevie Wonder’s mother to complete the prompt. We analyze these two hops individually and consider their co-occurrence as indicative of latent multi-hop reasoning. For the first hop, we test if changing the prompt to indirectly mention the bridge entity instead of any other entity increases the LLM’s internal recall of the bridge entity. For the second hop, we test if increasing this recall causes the LLM to better utilize what it knows about the bridge entity. We find strong evidence of latent multi-hop reasoning for the prompts of certain relation types, with the reasoning pathway used in more than 80% of the prompts. However, the utilization is highly contextual, varying across different types of prompts. Also, on average, the evidence for the second hop and the full multi-hop traversal is rather moderate and only substantial for the first hop. Moreover, we find a clear scaling trend with increasing model size for the first hop of reasoning but not for the second hop. Our experimental findings suggest potential challenges and opportunities for future development and applications of LLMs.
     </details>

93. **MathGenie: Generating Synthetic Data with Question Back-translation for Enhancing Mathematical Reasoning of LLMs** [[pdf]](http://arxiv.org/abs/2402.16352) `ACL 2024 Long Papers` (14 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduces MathGenie, a novel method for generating diverse and reliable math problems from a small-scale problem-solution dataset, and introduces a family of models known as MathGenieLM, which consistently outperform previous open-source models across five representative mathematical reasoning datasets.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have exhibited great potential in mathematical reasoning. However, there remains a performance gap in this area between existing open-source models and closed-source models such as GPT-4. In this paper, we introduce MathGenie, a novel method for generating diverse and reliable math problems by leveraging the ground-truth solutions of the seed data. We augment these ground-truth solutions and use a specially finetuned model to translate these augmented solutions back into new questions. Subsequently, we generate code-integrated solutions for these questions. To ensure the correctness of the code-integrated solutions, we employ rationale-based verification for filtering. Then, we finetune various pretrained models, ranging from 7B to 70B, on the newly curated data, resulting in a family of models known as MathGenie. These models consistently outperform previous open-source models across five representative mathematical reasoning datasets, achieving state-of-the-art performance. In particular, MathGenie-InternLM2 achieves an accuracy of 87.7% on GSM8K and 55.7% on MATH, securing the best overall score.
     </details>

94. **Measuring Multimodal Mathematical Reasoning with MATH-Vision Dataset** [[pdf]](http://arxiv.org/abs/2402.14804) `NeurIPS 2024` (24 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Recent advancements in Large Multimodal Models (LMMs) have shown promising results in mathematical reasoning within visual contexts, with models exceeding human-level performance on existing benchmarks such as MathVista. However, we observe significant limitations in the diversity of questions and breadth of subjects covered by these benchmarks. To address this issue, we present the MATH-Vision (MATH-V) dataset, a meticulously curated collection of 3,040 high-quality mathematical problems with visual contexts sourced from real math competitions. Spanning 16 distinct mathematical disciplines and graded across 5 levels of difficulty, our dataset provides a comprehensive and diverse set of challenges for evaluating the mathematical reasoning abilities of LMMs. Through extensive experimentation, we unveil a notable performance gap between current LMMs and human performance on \datasetname, underscoring the imperative for further advancements in LMMs. Moreover, our detailed categorization allows for a thorough error analysis of LMMs, offering valuable insights to guide future research and development. The dataset is released at MathLLMs/MathVision
     </details>

95. **CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing** [[pdf]](http://arxiv.org/abs/2305.11738) `ICLR 2024 Poster` (215 cite) (8 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A framework called CRITIC is introduced that allows LLMs, which are essentially"black boxes" to validate and progressively amend their own outputs in a manner similar to human interaction with tools.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent developments in large language models (LLMs) have been impressive. However, these models sometimes show inconsistencies and problematic behavior, such as hallucinating facts, generating flawed code, or creating offensive and toxic content. Unlike these models, humans typically utilize external tools to cross-check and refine their initial content, like using a search engine for fact-checking, or a code interpreter for debugging. Inspired by this observation, we introduce a framework called CRITIC that allows LLMs, which are essentially “black boxes” to validate and progressively amend their own outputs in a manner similar to human interaction with tools. More specifically, starting with an initial output, CRITIC interacts with appropriate tools to evaluate certain aspects of the text, and then revises the output based on the feedback obtained during this validation process. Comprehensive evaluations involving free-form question answering, mathematical program synthesis, and toxicity reduction demonstrate that CRITIC consistently enhances the performance of LLMs. Meanwhile, our research highlights the crucial importance of external feedback in promoting the ongoing self-improvement of LLMs.
     </details>

96. **OpenMathInstruct-1: A 1.8 Million Math Instruction Tuning Dataset** [[pdf]](http://arxiv.org/abs/2402.10176) `NeurIPS 2024` (42 cite) (4 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The dataset is constructed by synthesizing code-interpreter solutions for GSM8K and MATH, two popular math reasoning benchmarks, using the recently released and permissively licensed Mixtral model and achieves a score competitive with the best gpt-distilled models.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent work has shown the immense potential of synthetically generated datasets for training large language models (LLMs), especially for acquiring targeted skills. Current large-scale math instruction tuning datasets such as MetaMathQA (Yu et al., 2024) and MAmmoTH (Yue et al., 2024) are constructed using outputs from closed-source LLMs with commercially restrictive licenses. A key reason limiting the use of open-source LLMs in these data generation pipelines has been the wide gap between the mathematical skills of the best closed-source LLMs, such as GPT-4, and the best open-source LLMs. Building on the recent progress in open-source LLMs, our proposed prompting novelty, and some brute-force scaling, we construct OpenMathInstruct-1, a math instruction tuning dataset with 1.8M problem-solution pairs. The dataset is constructed by synthesizing code-interpreter solutions for GSM8K and MATH, two popular math reasoning benchmarks, using the recently released and permissively licensed Mixtral model. Our best model, OpenMath-CodeLlama-70B, trained on a subset of OpenMathInstruct-1, achieves a score of 84.6% on GSM8K and 50.7% on MATH, which is competitive with the best gpt-distilled models. We will release our code, models, and the OpenMathInstruct-1 dataset under a commercially permissive license.
     </details>

97. **SELF-DISCOVER: Large Language Models Self-Compose Reasoning Structures** [[pdf]](http://arxiv.org/abs/2402.03620) `NeurIPS 2024` (27 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          SELF-DISCOVER substantially improves GPT-4 and PaLM 2's performance on challenging reasoning benchmarks such as BigBench-Hard, grounded agent reasoning, and MATH, and outperforms inference-intensive methods such as CoT-Self-Consistency by more than 20%, while requiring 10-40x fewer inference compute.
     </details>


     <details>
          <summary>Abstract</summary>
          We introduce SELF-DISCOVER, a general framework for LLMs to self-discover the task-intrinsic reasoning structures to tackle complex reasoning problems that are challenging for typical prompting methods. Core to the framework is a self-discovery process where LLMs select multiple atomic reasoning modules such as critical thinking and step-by-step thinking, and compose them into an explicit reasoning structure for LLMs to follow during decoding. SELF-DISCOVER substantially improves GPT-4 and PaLM 2’s performance on challenging reasoning benchmarks such as BigBench-Hard, grounded agent reasoning, and MATH, by as much as 32% compared to Chain of Thought (CoT). Furthermore, SELF-DISCOVER outperforms inference-intensive methods such as CoT-Self-Consistency by more than 20%, while requiring 10-40x fewer inference compute. Finally, we show that the self-discovered reasoning structures are universally applicable across model families: from PaLM 2-L to GPT-4, and from GPT-4 to Llama2, and share commonalities with human reasoning patterns.
     </details>

98. **ToolkenGPT: Augmenting Frozen Language Models with Massive Tools via Tool Embeddings** [[pdf]](http://arxiv.org/abs/2305.11554) `NeurIPS 2023` (123 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The proposed ToolkenGPT offers the flexibility to plug in an arbitrary number of tools by expanding the set of toolkens on the fly and improves tool use by allowing extensive demonstration data for learning the toolken embeddings.
     </details>


     <details>
          <summary>Abstract</summary>
          Integrating large language models (LLMs) with various tools has led to increased attention in the field. Existing approaches either involve fine-tuning the LLM, which is both computationally costly and limited to a fixed set of tools, or prompting LLMs by in-context tool demonstrations. Although the latter method offers adaptability to new tools, it struggles with the inherent context length constraint of LLMs when many new tools are presented, and mastering a new set of tools with few-shot examples remains challenging, resulting in suboptimal performance. To address these limitations, we propose a novel solution, named **ToolkenGPT**, wherein LLMs effectively learn to master tools as predicting tokens through **tool embeddings** for solving complex tasks. In this framework, each tool is transformed into vector embeddings and plugged into the language model head. Once the function is triggered during text generation, the LLM enters a special function mode to execute the tool calls. Our experiments show that function embeddings effectively help LLMs understand tool use and improve on several tasks, including numerical reasoning, knowledge-based question answering and embodied decision-making.
     </details>

99. **Graph2Tac: Online Representation Learning of Formal Math Concepts** [[pdf]](http://arxiv.org/abs/2401.02949) `ICML 2024` `Coq` (3 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work extensively benchmarks two online solvers implemented in the Tactician platform for the Coq proof assistant, and introduces a graph neural network, Graph2Tac, with a novel approach to build hierarchical representations for new definitions.
     </details>


     <details>
          <summary>Abstract</summary>
          In proof assistants, the physical proximity between two formal mathematical concepts is a strong predictor of their mutual relevance. Furthermore, lemmas with close proximity regularly exhibit similar proof structures. We show that this _locality_ property can be exploited through online learning techniques to obtain solving agents that far surpass offline learners when asked to prove theorems in an unseen mathematical setting. We extensively benchmark two such online solvers implemented in the Tactician platform for the Coq proof assistant: First, Tactician's online $k$-nearest neighbor solver, which can learn from recent proofs, shows a $1.72\times$ improvement in theorems proved over an offline equivalent. Second, we introduce a graph neural network, Graph2Tac, with a novel approach to build hierarchical representations for new definitions. Graph2Tac's online definition task realizes a $1.5\times$ improvement in theorems solved over an offline baseline. The $k$-NN and Graph2Tac solvers rely on orthogonal online data, making them highly complementary. Their combination improves $1.27\times$ over their individual performances. Both solvers outperform all other general purpose provers for Coq, including CoqHammer, Proverbot9001, and a transformer baseline by at least $1.48\times$ and are available for practical use by end-users.
     </details>

100. **Solving olympiad geometry without human demonstrations** [[pdf]](https://www.nature.com/articles/s41586-023-06747-5) `Nature` (162 cite) (7 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          AlphaGeometry is a neuro-symbolic system that uses a neural language model that uses a neural language model, trained from scratch on large-scale synthetic data, to guide a symbolic deduction engine through infinite branching points in challenging problems.
     </details>


     <details>
          <summary>Abstract</summary>
          Proving mathematical theorems at the olympiad level represents a notable milestone in human-level automated reasoning1–4, owing to their reputed difficulty among the world’s best talents in pre-university mathematics. Current machine-learning approaches, however, are not applicable to most mathematical domains owing to the high cost of translating human proofs into machine-verifiable format. The problem is even worse for geometry because of its unique translation challenges1,5, resulting in severe scarcity of training data. We propose AlphaGeometry, a theorem prover for Euclidean plane geometry that sidesteps the need for human demonstrations by synthesizing millions of theorems and proofs across different levels of complexity. AlphaGeometry is a neuro-symbolic system that uses a neural language model, trained from scratch on our large-scale synthetic data, to guide a symbolic deduction engine through infinite branching points in challenging problems. On a test set of 30 latest olympiad-level problems, AlphaGeometry solves 25, outperforming the previous best method that only solves ten problems and approaching the performance of an average International Mathematical Olympiad (IMO) gold medallist. Notably, AlphaGeometry produces human-readable proofs, solves all geometry problems in the IMO 2000 and 2015 under human expert evaluation and discovers a generalized version of a translated IMO theorem in 2004.
     </details>

101. **TheoremQA: A Theorem-driven Question Answering Dataset** [[pdf]](http://arxiv.org/abs/2305.12524) `EMNLP 2023 Main` (64 cite) (9 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduces TheoremQA, the first theorem-driven question-answering dataset designed to evaluate AI models' capabilities to apply theorems to solve challenging science problems and finds that GPT-4's capabilities to solve these problems are unparalleled, achieving an accuracy of 51% with Program-of-Thoughts Prompting.
     </details>


     <details>
          <summary>Abstract</summary>
          The recent LLMs like GPT-4 and PaLM-2 have made tremendous progress in solving fundamental math problems like GSM8K by achieving over 90% accuracy. However, their capabilities to solve more challenging math problems which require domain-specific knowledge (i.e. theorem) have yet to be investigated. In this paper, we introduce TheoremQA, the first theorem-driven question-answering dataset designed to evaluate AI models’ capabilities to apply theorems to solve challenging science problems. TheoremQA is curated by domain experts containing 800 high-quality questions covering 350 theorems from Math, Physics, EE&CS, and Finance. We evaluate a wide spectrum of 16 large language and code models with different prompting strategies like Chain-of-Thoughts and Program-of-Thoughts. We found that GPT-4’s capabilities to solve these problems are unparalleled, achieving an accuracy of 51% with Program-of-Thoughts Prompting. All the existing open-sourced models are below 15%, barely surpassing the random-guess baseline. Given the diversity and broad coverage of TheoremQA, we believe it can be used as a better benchmark to evaluate LLMs’ capabilities to solve challenging science problems.
     </details>

102. **ReCEval: Evaluating Reasoning Chains via Correctness and Informativeness** [[pdf]](https://aclanthology.org/2023.emnlp-main.622) `EMNLP 2023 Main` (29 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes ReCEval (Reasoning Chain Evaluation), a framework that evaluates reasoning chains via two key properties: correctness, i.e., each step makes a valid inference based on information contained within the step, preceding steps, and input context, and informativeness, and each step provides new information that is helpful towards deriving the generated answer.
     </details>


     <details>
          <summary>Abstract</summary>
          Multi-step reasoning ability is fundamental to many natural language tasks, yet it is unclear what constitutes a good reasoning chain and how to evaluate them. Most existing methods focus solely on whether the reasoning chain leads to the correct conclusion, but this answer-oriented view may confound reasoning quality with other spurious shortcuts to predict the answer. To bridge this gap, we evaluate reasoning chains by viewing them as informal proofs that derive the final answer. Specifically, we propose ReCEval (Reasoning Chain Evaluation), a framework that evaluates reasoning chains via two key properties: (1) correctness, i.e., each step makes a valid inference based on information contained within the step, preceding steps, and input context, and (2) informativeness, i.e., each step provides new information that is helpful towards deriving the generated answer. We evaluate these properties by developing metrics using natural language inference models and 𝒱-Information. On multiple datasets, we show that ReCEval effectively identifies various error types and yields notable improvements compared to prior methods. We analyze the impact of step boundaries, and previous steps on evaluating correctness and demonstrate that our informativeness metric captures the expected flow of information in high-quality reasoning chains. Finally, we show that scoring reasoning chains based on ReCEval improves downstream task performance.
     </details>

103. **MoT: Memory-of-Thought Enables ChatGPT to Self-Improve** [[pdf]](https://aclanthology.org/2023.emnlp-main.392) `EMNLP 2023 Main` (18 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Experimental results show that MoT can help ChatGPT significantly improve its abilities in arithmetic reasoning, commonsense reasoning, factual reasoning, and natural language inference and can lead to consistent improvements across various CoT methods and LLMs.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have shown impressive abilities on various tasks. However, fundamentally improving them depends on high-quality datasets or computationally expensive fine-tuning. On the contrary, humans can easily improve themselves by self-thinking and memory, without external resources. In this paper, we propose a framework, **MoT**, to let the LLM self-improve through **M**emory **o**f **T**houghts, without annotated datasets and parameter updates. Specifically, MoT is divided into two stages: 1. before the test stage, the LLM pre-thinks on the unlabeled dataset and saves the high-confidence thoughts as external memory; 2. During the test stage, given a test question, the LLM recalls relevant memory to help itself reason and answer it. Experimental results show that MoT can help ChatGPT significantly improve its abilities in arithmetic reasoning, commonsense reasoning, factual reasoning, and natural language inference. Further analyses show that each component contributes critically to the improvements and MoT can lead to consistent improvements across various CoT methods and LLMs.
     </details>

104. **UniMath: A Foundational and Multimodal Mathematical Reasoner** [[pdf]](https://aclanthology.org/2023.emnlp-main.440) `EMNLP 2023 Main` (10 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Through comprehensive evaluations, it is shown that joint training across diverse math tasks improves overall model performance and enhances its ability to generalize across different mathematical reasoning tasks.
     </details>


     <details>
          <summary>Abstract</summary>
          While significant progress has been made in natural language processing (NLP), existing methods exhibit limitations in effectively interpreting and processing diverse mathematical modalities. Therefore, we introduce UniMath, a versatile and unified system designed for multimodal mathematical reasoning tasks. Tackling complex problem-solving in arithmetic, geometry, and table-based math, UniMath utilizes a fine-tuned T5 model augmented with a variational autoencoder (VAE)-based image tokenizer. By jointly training and evaluating the model on three diverse datasets - SVAMP, GeoQA, and TableMWP, UniMath achieves state-of-the-art performance. The model’s generalization ability is further demonstrated via fine-tuning on two additional datasets, MathQA and Geo-Proving. Through comprehensive evaluations, we showcase that joint training across diverse math tasks improves overall model performance and enhances its ability to generalize across different mathematical reasoning tasks. This pioneering approach provides a blueprint and inspires further efforts on unified mathematical reasoning with deep learning systems.
     </details>

105. **Calc-X and Calcformers: Empowering Arithmetical Chain-of-Thought through Interaction with Symbolic Systems** [[pdf]](https://aclanthology.org/2023.emnlp-main.742) `EMNLP 2023 Main` (9 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Calc-X, a collection of datasets that demonstrates the appropriate use of a calculator in reasoning chains, is created and is suitable for teaching language models to offload computations to a symbolic system.
     </details>


     <details>
          <summary>Abstract</summary>
          Despite outstanding performance in many tasks, language models are notoriously inclined to make factual errors in tasks requiring arithmetic computation. We address this deficiency by creating Calc-X, a collection of datasets that demonstrates the appropriate use of a calculator in reasoning chains. Calc-X is suitable for teaching language models to offload computations to a symbolic system. We survey and unify several existing chain-of-thought datasets into a proposed format, resulting in a standard collection of over 300,000 samples requiring arithmetic reasoning. Finally, we use the new Calc-X collection to train open-source calculator-using models and show that these models approximately double the accuracy of generating correct results compared to vanilla language model baselines.
     </details>

106. **DialCoT Meets PPO: Decomposing and Exploring Reasoning Paths in Smaller Language Models** [[pdf]](https://aclanthology.org/2023.emnlp-main.501) `EMNLP 2023 Main` (9 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Dialogue-guided Chain-of-Thought (DialCoT) is introduced which employs a dialogue format to generate intermediate reasoning steps, guiding the model toward the final answer, and optimize the model's reasoning path selection using the Proximal Policy Optimization (PPO) algorithm, further enhancing its reasoning capabilities.
     </details>


     <details>
          <summary>Abstract</summary>
          Chain-of-Thought (CoT) prompting has successfully enhanced the reasoning capabilities of Large Language Models (LLMs) with at least 100 billion parameters. However, it is ineffective, or even detrimental, to the performance on reasoning tasks in Smaller Language Models (SLMs) with less than 10 billion parameters. In this paper, we propose Dialogue-guided Chain-of-Thought (DialCoT) to improve the reasoning capabilities of SLMs, with the aim of generating intermediate reasoning steps in a dialogue format to guide the model to the final answer. Furthermore, we optimize the model to choose the optimal reasoning path through the Proximal Policy Optimization (PPO) algorithm, further enhancing its reasoning capabilities. Compared to previous methods, our advantages lie in: 1) We transform the process of solving complex reasoning problems into decomposing problems and solving a series of simpler sub-questions, significantly reducing task difficulty and making it more suitable for SLMs. 2) We optimize the model to choose the optimal reasoning path through the PPO algorithm. Comprehensive experiments on four arithmetic reasoning datasets show that our method can achieve significant performance gains over state-of-the-art competitors.
     </details>

107. **MAF: Multi-Aspect Feedback for Improving Reasoning in Large Language Models** [[pdf]](https://aclanthology.org/2023.emnlp-main.407) `EMNLP 2023 Main` (7 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes Multi-Aspect Feedback, an iterative refinement framework that integrates multiple feedback modules, including frozen LMs and external tools, each focusing on a specific error category in the LM-generated reasoning chain to improve the overall performance of an LM in several reasoning tasks.
     </details>


     <details>
          <summary>Abstract</summary>
          Language Models (LMs) have shown impressive performance in various natural language tasks. However, when it comes to natural language reasoning, LMs still face challenges such as hallucination, generating incorrect intermediate reasoning steps, and making mathematical errors. Recent research has focused on enhancing LMs through *self-improvement* using feedback. Nevertheless, existing approaches relying on a single generic feedback source fail to address the diverse error types found in LM-generated reasoning chains. In this work, we propose **Multi-Aspect Feedback**, an iterative refinement framework that integrates multiple feedback modules, including frozen LMs and external tools, each focusing on a specific error category. Our experimental results demonstrate the efficacy of our approach to addressing several errors in the LM-generated reasoning chain and thus improving the overall performance of an LM in several reasoning tasks. We see an improvement of up to 20% in Mathematical Reasoning and up to 18% in Logical Entailment.
     </details>

108. **It Ain’t Over: A Multi-aspect Diverse Math Word Problem Dataset** [[pdf]](https://aclanthology.org/2023.emnlp-main.927) `EMNLP 2023 Main` (6 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          The math word problem (MWP) is a complex task that requires natural language understanding and logical reasoning to extract key knowledge from natural language narratives. Previous studies have provided various MWP datasets but lack diversity in problem types, lexical usage patterns, languages, and annotations for intermediate solutions. To address these limitations, we introduce a new MWP dataset, named DMath (Diverse Math Word Problems), offering a wide range of diversity in problem types, lexical usage patterns, languages, and intermediate solutions. The problems are available in English and Korean and include an expression tree and Python code as intermediate solutions. Through extensive experiments, we demonstrate that the DMath dataset provides a new opportunity to evaluate the capability of large language models, i.e., GPT-4 only achieves about 75% accuracy on the DMath dataset.
     </details>

109. **An Expression Tree Decoding Strategy for Mathematical Equation Generation** [[pdf]](https://aclanthology.org/2023.emnlp-main.29) `EMNLP 2023 Main` (5 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work integrates tree structure into the expression-level generation and advocates an expression tree decoding strategy, and shows that this method outperforms other baselines, especially for these equations with complex structures.
     </details>


     <details>
          <summary>Abstract</summary>
          Generating mathematical equations from natural language requires an accurate understanding of the relations among math expressions. Existing approaches can be broadly categorized into token-level and expression-level generation. The former treats equations as a mathematical language, sequentially generating math tokens. Expression-level methods generate each expression one by one. However, each expression represents a solving step, and there naturally exist parallel or dependent relations between these steps, which are ignored by current sequential methods. Therefore, we integrate tree structure into the expression-level generation and advocate an expression tree decoding strategy. To generate a tree with expression as its node, we employ a layer-wise parallel decoding strategy: we decode multiple independent expressions (leaf nodes) in parallel at each layer and repeat parallel decoding layer by layer to sequentially generate these parent node expressions that depend on others. Besides, a bipartite matching algorithm is adopted to align multiple predictions with annotations for each layer. Experiments show our method outperforms other baselines, especially for these equations with complex structures.
     </details>

110. **Non-Autoregressive Math Word Problem Solver with Unified Tree Structure** [[pdf]](https://aclanthology.org/2023.emnlp-main.199) `EMNLP 2023 Main` (5 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel non-autoregressive solver is proposed, named \textit{MWP-NAS}, to parse the problem and deduce the solution expression based on the unified tree, where the elements are permutable and identical for all the expression variants.
     </details>


     <details>
          <summary>Abstract</summary>
          Existing MWP solvers employ sequence or binary tree to present the solution expression and decode it from given problem description. However, such structures fail to handle the variants that can be derived via mathematical manipulation, e.g., (a1+a2)*a3 and a1 * a3+a2 * a3 can both be possible valid solutions for a same problem but formulated as different expression sequences or trees. The multiple solution variants depicting different possible solving procedures for the same input problem would raise two issues: 1) making it hard for the model to learn the mapping function between the input and output spaces effectively, and 2) wrongly indicating wrong when evaluating a valid expression variant. To address these issues, we introduce a unified tree structure to present a solution expression, where the elements are permutable and identical for all the expression variants. We propose a novel non-autoregressive solver, named MWP-NAS, to parse the problem and deduce the solution expression based on the unified tree. For evaluating the possible expression variants, we design a path-based metric to evaluate the partial accuracy of expressions of a unified tree. The results from extensive experiments conducted on Math23K and MAWPS demonstrate the effectiveness of our proposed MWP-NAS. The codes and checkpoints are available at: https://github.com/mengqunhan/MWP-NAS.
     </details>

111. **ATHENA: Mathematical Reasoning with Thought Expansion** [[pdf]](https://aclanthology.org/2023.emnlp-main.1014) `EMNLP 2023 Main` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Attention-based THought Expansion Network Architecture (ATHENA) is introduced to tackle the challenges of real-world practices by mimicking human thought expansion mechanisms in the form of neural network propagation.
     </details>


     <details>
          <summary>Abstract</summary>
          Solving math word problems depends on how to articulate the problems, the lens through which models view human linguistic expressions. Real-world settings count on such a method even more due to the diverse practices of the same mathematical operations. Earlier works constrain available thinking processes by limited prediction strategies without considering their significance in acquiring mathematical knowledge. We introduce Attention-based THought Expansion Network Architecture (ATHENA) to tackle the challenges of real-world practices by mimicking human thought expansion mechanisms in the form of neural network propagation. A thought expansion recurrently generates the candidates carrying the thoughts of possible math expressions driven from the previous step and yields reasonable thoughts by selecting the valid pathways to the goal. Our experiments show that ATHENA achieves a new state-of-the-art stage toward the ideal model that is compelling in variant questions even when the informativeness in training examples is restricted.
     </details>

112. **A Generation-based Deductive Method for Math Word Problems** [[pdf]](https://aclanthology.org/2023.emnlp-main.108) `EMNLP 2023 Main` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Math word problems (MWP) involving advanced operators such as linear equation solver cannot be easily tackled by earlier MWP methods, because the existing generation methods suffer from repeated sub-expression generation and deductive methods are restricted to dealing with binary operations. This paper propose a new multivariate directed acyclic graph (mDAG) as an alternative to the generation methods’ binary expression tree or the deductive methods’ binary directed acyclic graph. Then to produce the topological ordering of mDAG, we propose a generation-based deductive (GeDe) model, which equips a generation model with a re-encoder to keep the deductive property but avoid the expensive enumeration of the deductive methods. GeDe performs well on math problems with many operators on the widely used benchmarks as well as solving multivariate operators on our own CMWPA benchmark. Our code is available at https://github.com/hyx1999/GeDe
     </details>

113. **Large Language Models Are Latent Variable Models: Explaining and Finding Good Demonstrations for In-Context Learning** [[pdf]](https://openreview.net/forum?id=BGvkwZEGt7) `NeurIPS 2023 Poster` (62 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The empirical findings support the hypothesis that LLMs implicitly infer a latent variable containing task information, and propose an algorithm to select optimal demonstrations from a set of annotated data with a small LM, and then directly generalize the selected demonstrations to larger LMs.
     </details>


     <details>
          <summary>Abstract</summary>
          In recent years, pre-trained large language models (LLMs) have demonstrated remarkable efficiency in achieving an inference-time few-shot learning capability known as in-context learning. However, existing literature has highlighted the sensitivity of this capability to the selection of few-shot demonstrations. Current understandings of the underlying mechanisms by which this capability arises from regular language model pretraining objectives remain disconnected from the real-world LLMs. This study aims to examine the in-context learning phenomenon through a Bayesian lens, viewing real-world LLMs as latent variable models. On this premise, we propose an algorithm to select optimal demonstrations from a set of annotated data with a small LM, and then directly generalize the selected demonstrations to larger LMs. We demonstrate significant improvement over baselines, averaged over eight GPT models on eight real-world text classification datasets. We also demonstrate the real-world usefulness of our algorithm on GSM8K, a math word problem dataset. Our empirical findings support our hypothesis that LLMs implicitly infer a latent variable containing task information.
     </details>

114. **Self-Evaluation Guided Beam Search for Reasoning** [[pdf]](https://openreview.net/forum?id=Bw82hwg5Q3) `NeurIPS 2023 Poster` (61 cite) (4 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A stepwise self-evaluation mechanism to guide and calibrate the reasoning process of LLMs through stochastic beam search and a decoding algorithm integrating the self- evaluation guidance via stochastically beam search are proposed.
     </details>


     <details>
          <summary>Abstract</summary>
          Breaking down a problem into intermediate steps has demonstrated impressive performance in Large Language Model (LLM) reasoning. However, the growth of the reasoning chain introduces uncertainty and error accumulation, making it challenging to elicit accurate final results. To tackle this challenge of uncertainty in multi-step reasoning, we introduce a stepwise self-evaluation mechanism to guide and calibrate the reasoning process of LLMs. We propose a decoding algorithm integrating the self-evaluation guidance via stochastic beam search. The self-evaluation guidance serves as a better-calibrated automatic criterion, facilitating an efficient search in the reasoning space and resulting in superior prediction quality. Stochastic beam search balances exploitation and exploration of the search space with temperature-controlled randomness. Our approach surpasses the corresponding Codex-backboned baselines in few-shot accuracy by $6.34$%, $9.56$%, and $5.46$% on the GSM8K, AQuA, and StrategyQA benchmarks, respectively. Experiment results with Llama-2 on arithmetic reasoning demonstrate the efficiency of our method in outperforming the baseline methods with comparable computational budgets. Further analysis in multi-step reasoning finds our self-evaluation guidance pinpoints logic failures and leads to higher consistency and robustness. Our code is publicly available at [https://guideddecoding.github.io/](https://guideddecoding.github.io/).
     </details>

115. **Evaluating and Improving Tool-Augmented Computation-Intensive Math Reasoning** [[pdf]](https://openreview.net/forum?id=OB10WTlwmX) `NeurIPS 2023 Poster` (19 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Experimental results on CARP and six other datasets show that the proposed DELI mostly outperforms competitive baselines, and can further boost the performance of existing CoT methods.
     </details>


     <details>
          <summary>Abstract</summary>
          Chain-of-thought prompting (CoT) and tool augmentation have been validated in recent work as effective practices for improving large language models (LLMs) to perform step-by-step reasoning on complex math-related tasks.However, most existing math reasoning datasets may not be able to fully evaluate and analyze the ability of LLMs in manipulating tools and performing reasoning, as they often only require very few invocations of tools or miss annotations for evaluating intermediate reasoning steps, thus supporting only outcome evaluation.To address the issue, we construct **CARP**, a new Chinese dataset consisting of 4,886 computation-intensive algebra problems with formulated annotations on intermediate steps, facilitating the evaluation of the intermediate reasoning process.In CARP, we test four LLMs with CoT prompting, and find that they are all prone to make mistakes at the early steps of the solution, leading to incorrect answers.Based on this finding, we propose a new approach that can facilitate the deliberation on reasoning steps with tool interfaces, namely **DELI**.In DELI, we first initialize a step-by-step solution based on retrieved exemplars, then iterate two deliberation procedures that check and refine the intermediate steps of the generated solution, from both tool manipulation and natural language reasoning perspectives, until solutions converge or the maximum iteration is achieved.Experimental results on CARP and six other datasets show that the proposed DELI mostly outperforms competitive baselines, and can further boost the performance of existing CoT methods.Our data and code are available at https://github.com/RUCAIBox/CARP.
     </details>

116. **Geometric Algebra Transformer** [[pdf]](https://openreview.net/forum?id=M7r2CO4tJC) `NeurIPS 2023 Poster` (12 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The Geometric Algebra Transformer (GATr), a general-purpose architecture for geometric data that consistently outperforms both non-geometric and equivariant baselines in terms of error, data efficiency, and scalability, is introduced.
     </details>


     <details>
          <summary>Abstract</summary>
          Problems involving geometric data arise in physics, chemistry, robotics, computer vision, and many other fields. Such data can take numerous forms, for instance points, direction vectors, translations, or rotations, but to date there is no single architecture that can be applied to such a wide variety of geometric types while respecting their symmetries. In this paper we introduce the Geometric Algebra Transformer (GATr), a general-purpose architecture for geometric data. GATr represents inputs, outputs, and hidden states in the projective geometric (or Clifford) algebra, which offers an efficient 16-dimensional vector-space representation of common geometric objects as well as operators acting on them. GATr is equivariant with respect to E(3), the symmetry group of 3D Euclidean space. As a Transformer, GATr is versatile, efficient, and scalable. We demonstrate GATr in problems from n-body modeling to wall-shear-stress estimation on large arterial meshes to robotic motion planning. GATr consistently outperforms both non-geometric and equivariant baselines in terms of error, data efficiency, and scalability.
     </details>

117. **MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts** [[pdf]](https://openreview.net/forum?id=hrI14X0Ltk) `ICLR 2024 Oral` (212 cite) (4 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The in-depth analysis reveals that the superiority of GPT-4V is mainly attributed to its enhanced visual perception and mathematical reasoning, but it still falls short of human performance by 10.4%, which underscores the critical role that MathVista will play in the development of general-purpose AI agents capable of tackling mathematically intensive and visually rich real-world tasks.
     </details>


     <details>
          <summary>Abstract</summary>
          Although Large Language Models (LLMs) and Large Multimodal Models (LMMs) exhibit impressive skills in various domains, their ability for mathematical reasoning within visual contexts has not been formally examined. Equipping LLMs and LMMs with this capability is vital for general-purpose AI assistants and showcases promising potential in education, data analysis, and scientific discovery. To bridge this gap, we present MathVista, a benchmark designed to amalgamate challenges from diverse mathematical and visual tasks. We first taxonomize the key task types, reasoning skills, and visual contexts from the literature to guide our selection from 28 existing math-focused and visual question answering datasets. Then, we construct three new datasets, IQTest, FunctionQA, and PaperQA, to accommodate for missing types of visual contexts. The problems featured often require deep visual understanding beyond OCR or image captioning, and compositional reasoning with rich domain-specific tools, thus posing a notable challenge to existing models. We conduct a comprehensive evaluation of 11 prominent open-source and proprietary foundation models (LLMs, LLMs augmented with tools, and LMMs). The best-performing model, Multimodal Bard, achieves only 58\% of human performance (34.8\% vs 60.3\%), indicating ample room for further improvement. Given this significant gap, MathVista fuels future research in the development of general-purpose AI agents capable of tackling mathematically intensive and visually rich real-world tasks.
     </details>

118. **Teaching Arithmetic to Small Transformers** [[pdf]](https://openreview.net/forum?id=YfhuG7xHQ8) `ICLR 2024 Poster` (52 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This study investigates how small transformers, trained from random initialization, can efficiently learn arithmetic operations such as addition, multiplication, and elementary functions like square root, using the next-token prediction objective.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models like GPT-4 exhibit emergent capabilities across general-purpose tasks, such as basic arithmetic, when trained on extensive text data, even though these tasks are not explicitly encoded by the unsupervised, next-token prediction objective. This study investigates how even small transformers, trained from random initialization, can efficiently learn arithmetic operations such as addition, multiplication, and elementary functions like square root, using the next-token prediction objective. We first demonstrate that conventional training data is not the most effective for arithmetic learning, and simple formatting changes can significantly improve accuracy. This leads to sharp phase transitions as a function of training data scale, which, in some cases, can be explained through connections to low-rank matrix completion. Building on prior work, we then train on chain-of-thought style data that includes intermediate step results. Even in the complete absence of pretraining, this approach significantly and simultaneously improves accuracy, sample complexity, and convergence speed. We also study the interplay between arithmetic and text data during training and examine the effects of few-shot prompting, pretraining, and parameter scaling. Additionally, we discuss the challenges associated with length generalization. Our work highlights the importance of high-quality, instructive data that considers the particular characteristics of the next-word prediction loss for rapidly eliciting arithmetic capabilities.
     </details>

119. **SciBench: Evaluating College-Level Scientific Problem-Solving Abilities of Large Language Models** [[pdf]](https://openreview.net/forum?id=A3W864NIW2) `ICML 2024 Poster` (43 cite) (3 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          An in-depth benchmarking study of representative open-source and proprietary LLMs with various prompting strategies indicates that no single prompting strategy significantly outperforms the others and some strategies that demonstrate improvements in certain problem-solving skills could result in declines in other skills.
     </details>


     <details>
          <summary>Abstract</summary>
          Most existing Large Language Model (LLM) benchmarks on scientific problem reasoning focus on problems grounded in high-school subjects and are confined to elementary algebraic operations. To systematically examine the reasoning capabilities required for solving complex scientific problems, we introduce an expansive benchmark suite SciBench for LLMs. SciBench contains a carefully curated dataset featuring a range of collegiate-level scientific problems from mathematics, chemistry, and physics domains. Based on the dataset, we conduct an in-depth benchmarking study of representative open-source and proprietary LLMs with various prompting strategies. The results reveal that current LLMs fall short of delivering satisfactory performance, with the best overall score of merely 43.22%. Furthermore, through a detailed user study, we categorize the errors made by LLMs into ten problem-solving abilities. Our analysis indicates that no single prompting strategy significantly outperforms the others and some strategies that demonstrate improvements in certain problem-solving skills could result in declines in other skills. We envision that SciBench will catalyze further developments in the reasoning abilities of LLMs, thereby ultimately contributing to scientific research and discovery.
     </details>

120. **OpenWebMath: An Open Dataset of High-Quality Mathematical Web Text** [[pdf]](https://openreview.net/forum?id=5hZTBUtkeh) `ICLR 2024` (35 cite) (3 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces OpenWebMath, an open dataset inspired by these works containing 14.7B tokens of mathematical webpages from Common Crawl, and describes in detail the method for extracting text and LaTeX content and removing boilerplate from HTML documents, as well as the methods for quality filtering and deduplication.
     </details>


     <details>
          <summary>Abstract</summary>
          There is growing evidence that pretraining on high quality, carefully thought-out tokens such as code or mathematics plays an important role in improving the reasoning abilities of large language models. For example, Minerva, a PaLM model finetuned on billions of tokens of mathematical documents from arXiv and the web, reported dramatically improved performance on problems that require quantitative reasoning. However, because all known open source web datasets employ preprocessing that does not faithfully preserve mathematical notation, the benefits of large scale training on quantitive web documents are unavailable to the research community. We introduce OpenWebMath, an open dataset inspired by these works containing 14.7B tokens of mathematical webpages from Common Crawl. We describe in detail our method for extracting text and LaTeX content and removing boilerplate from HTML documents, as well as our methods for quality filtering and deduplication. Additionally, we run small-scale experiments by training 1.4B language models on OpenWebMath, showing that models trained on 14.7B tokens of our dataset surpass the performance of models trained on over 20x the amount of general language data. We hope that our dataset, open-sourced and released on the Hugging Face Hub, will help spur advances in the reasoning abilities of large language models.
     </details>

121. **TRIGO: Benchmarking Formal Mathematical Proof Reduction for Generative Language Models** [[pdf]](http://arxiv.org/abs/2310.10180) `EMNLP 2023 Main` `Lean` (11 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes TRIGO, an ATP benchmark that not only requires a model to reduce a trigonometric expression with step-by-step proofs but also evaluates a generative LM's reasoning ability on formulas and its capability to manipulate, group, and factor number terms.
     </details>


     <details>
          <summary>Abstract</summary>
          Automated theorem proving (ATP) has become an appealing domain for exploring the reasoning ability of the recent successful generative language models. However, current ATP benchmarks are mainly focus on symbolic inference, but rarely involve the understanding of complex number combination reasoning. In this work, we propose TRIGO, an ATP benchmark that not only requires a model to reduce a trigonometric expression with step-by-step proof but also evaluates a generative LM’s reasoning ability on formulas and capability to manipulate, group, and factor number terms. We gather trigonometric expressions and their reduced forms from web, annotate the simplification process manually, and translate it into the “Lean” formal language system. We then automatically generate additional examples from the annotated samples to expand the dataset. Furthermore, we also create three automatically generated training and testing datasets of varying difficulty and distributions. Our extensive experiments show our proposed TRIGO poses a new challenge for advanced generative LM’s including GPT-4 which is pre-trained on a considerable amount of open-source formal theorem-proving language data, and provide a new tool to study the generative LM’s ability on both formal and mathematical reasoning.
     </details>

122. **MLFMF: Data Sets for Machine Learning for Mathematical Formalization** [[pdf]](http://arxiv.org/abs/2310.16005) `NeurIPS 2023` `Lean, Agda` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The MLFMF data sets provide solid benchmarking support for further investigation of the numerous machine learning approaches to formalized mathematics, and are currently the largest collection of formalized mathematical knowledge in machine learnable format.
     </details>


     <details>
          <summary>Abstract</summary>
          We introduce MLFMF, a collection of data sets for benchmarking recommendation systems used to support formalization of mathematics with proof assistants. These systems help humans identify which previous entries (theorems, constructions, datatypes, and postulates) are relevant in proving a new theorem or carrying out a new construction. Each data set is derived from a library of formalized mathematics written in proof assistants Agda or Lean. The collection includes the largest Lean 4 library Mathlib, and some of the largest Agda libraries: the standard library, the library of univalent mathematics Agda-unimath, and the TypeTopology library. Each data set represents the corresponding library in two ways: as a heterogeneous network, and as a list of s-expressions representing the syntax trees of all the entries in the library. The network contains the (modular) structure of the library and the references between entries, while the s-expressions give complete and easily parsed information about every entry.We report baseline results using standard graph and word embeddings, tree ensembles, and instance-based learning algorithms. The MLFMF data sets provide solid benchmarking support for further investigation of the numerous machine learning approaches to formalized mathematics. The methodology used to extract the networks and the s-expressions readily applies to other libraries, and is applicable to other proof assistants. With more than $250\,000$ entries in total, this is currently the largest collection of formalized mathematical knowledge in machine learnable format.
     </details>

123. **Reasoning with Language Model is Planning with World Model** [[pdf]](http://arxiv.org/abs/2305.14992) `EMNLP 2023 Main` (289 cite) (8 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A new LLM reasoning framework, RAP, which repurposes the LLM as both a world model and a reasoning agent, and incorporates a principled planning algorithm (based on Monto Carlo Tree Search) for strategic exploration in the vast reasoning space.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have shown remarkable reasoning capabilities, particularly with Chain-of-Thought-style prompts. However, LLMs can still struggle with problems that are easy for humans, such as generating action plans for executing tasks or performing complex math or logical reasoning. This is due to LLMs’ absence of an internal world model for predicting world states (e.g., environment status, variable values) and simulating long-term action outcomes of actions. This prevents LLMs from performing deliberate planning akin to human brains, which involves exploring alternative reasoning paths, anticipating future states and rewards, and iteratively refining existing reasoning steps. To overcome the limitations, we propose a new LLM reasoning framework, Reasoning via Planning (RAP). RAP repurposes the LLM as both a world model and a reasoning agent, and incorporates a principled planning algorithm (based on Monte Carlo Tree Search) for strategic exploration in the vast reasoning space. During reasoning, the LLM (as agent) incrementally builds a reasoning tree under the guidance of the LLM (as world model) and task-specific rewards, properly balancing exploration v.s. exploitation to achieve a high-reward reasoning path efficiently. We apply RAP to a variety of challenging reasoning problems, such as plan generation, math reasoning, and logical inference. Empirical results demonstrate the superiority of RAP over various strong baselines, including CoT and least-to-most prompting with self-consistency, e.g., RAP on LLaMA-33B surpasses CoT on GPT-4 with 33% relative improvement in plan generation.
     </details>

124. **Answering Questions by Meta-Reasoning over Multiple Chains of Thought** [[pdf]](http://arxiv.org/abs/2304.13007) `EMNLP 2023 Main` (69 cite) (5 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces Multi-Chain Reasoning (MCR), an approach which prompts large language models to meta-reason over multiple chains of thought, rather than aggregating their answers.
     </details>


     <details>
          <summary>Abstract</summary>
          Modern systems for multi-hop question answering (QA) typically break questions into a sequence of reasoning steps, termed chain-of-thought (CoT), before arriving at a final answer. Often, multiple chains are sampled and aggregated through a voting mechanism over the final answers, but the intermediate steps themselves are discarded. While such approaches improve performance, they do not consider the relations between intermediate steps across chains and do not provide a unified explanation for the predicted answer. We introduce Multi-Chain Reasoning (MCR), an approach which prompts large language models to meta-reason over multiple chains of thought, rather than aggregate their answers. MCR examines different reasoning chains, mixes information between them and selects the most relevant facts in generating an explanation and predicting the answer. MCR outperforms strong baselines on 7 multi-hop QA datasets. Moreover, our analysis reveals that MCR explanations exhibit high quality, enabling humans to verify its answers.
     </details>

125. **Llemma: An Open Language Model for Mathematics** [[pdf]](http://arxiv.org/abs/2310.10631) `ICLR 2024` `Lean, Isabelle` (164 cite) (12 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Llemma is a large language model for mathematics that outperforms all known open base models, as well as the unreleased Minerva model suite on an equi-parameter basis, and is capable of tool use and formal theorem proving without any further finetuning.
     </details>


     <details>
          <summary>Abstract</summary>
          We present Llemma, a large language model for mathematics. We continue pretraining Code Llama on the Proof-Pile-2, a mixture of scientific papers, web data containing mathematics, and mathematical code, yielding Llemma. On the MATH benchmark Llemma outperforms all known openly released models, as well as the unreleased Minerva model suite on an equi-parameter basis. Moreover, Llemma is capable of tool use and formal theorem proving without any finetuning. We openly release all artifacts, including 7 billion and 34 billion parameter models, the Proof-Pile-2, and code to replicate our experiments.
     </details>

126. **Large Language Models Cannot Self-Correct Reasoning Yet** [[pdf]](https://openreview.net/forum?id=IkmD3fKBPQ) `ICLR 2024 Poster` (236 cite) (13 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is indicated that LLMs struggle to self-correct their responses without external feedback, and at times, their performance even degrades after self-correction.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have emerged as a groundbreaking technology with their unparalleled text generation capabilities across various applications. Nevertheless, concerns persist regarding the accuracy and appropriateness of their generated content. A contemporary methodology, self-correction, has been proposed as a remedy to these issues. Building upon this premise, this paper critically examines the role and efficacy of self-correction within LLMs, shedding light on its true potential and limitations. Central to our investigation is the notion of intrinsic self-correction, whereby an LLM attempts to correct its initial responses based solely on its inherent capabilities, without the crutch of external feedback. In the context of reasoning, our research indicates that LLMs struggle to self-correct their responses without external feedback, and at times, their performance might even degrade post self-correction. Drawing from these insights, we offer suggestions for future research and practical applications in this field.
     </details>

127. **MathCoder: Seamless Code Integration in LLMs for Enhanced Mathematical Reasoning** [[pdf]](https://openreview.net/forum?id=z8TW0ttBPp) `ICLR 2024 Poster` (44 cite) (4 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper proposes a method to fine-tune open-source language models, enabling them to use code for modeling and deriving math equations and, consequently, enhancing their mathematical reasoning abilities, and yields the MathCoder models, a family of models capable of generating code-based solutions for solving challenging math problems.
     </details>


     <details>
          <summary>Abstract</summary>
          The recently released GPT-4 Code Interpreter has demonstrated remarkable proficiency in solving challenging math problems, primarily attributed to its ability to seamlessly reason with natural language, generate code, execute code, and continue reasoning based on the execution output. In this paper, we present a method to fine-tune open-source language models, enabling them to use code for modeling and deriving math equations and, consequently, enhancing their mathematical reasoning abilities. We propose a method of generating novel and high-quality datasets with math problems and their code-based solutions, referred to as MathCodeInstruct. Each solution interleaves $\textit{natural language}$, $\textit{code}$, and $\textit{execution results}$. We also introduce a customized supervised fine-tuning and inference approach. This approach yields the MathCoder models, a family of models capable of generating code-based solutions for solving challenging math problems. Impressively, the MathCoder models achieve state-of-the-art scores among open-source LLMs on the MATH (45.2%) and GSM8K (83.9%) datasets, substantially outperforming other open-source alternatives. Notably, the MathCoder model not only surpasses ChatGPT-3.5 and PaLM-2 on GSM8K and MATH but also outperforms GPT-4 on the competition-level MATH dataset. The proposed dataset and models will be released upon acceptance.
     </details>

128. **Large Language Models as Analogical Reasoners** [[pdf]](https://openreview.net/forum?id=AgDICX1h50) `ICLR 2024 Poster` (38 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Experimental results show that this approach outperforms 0-shot CoT and manual few-shot CoT in a variety of reasoning tasks, including math problem solving in GSM8K and MATH, code generation in Codeforces, and other reasoning tasks in BIG-Bench.
     </details>


     <details>
          <summary>Abstract</summary>
          Chain-of-thought (CoT) prompting for language models demonstrates impressive performance across reasoning tasks, but typically needs labeled exemplars of the reasoning process. In this work, we introduce a new prompting approach, analogical prompting, designed to automatically guide the reasoning process of large language models. Inspired by analogical reasoning, a cognitive process in which humans draw from relevant past experiences to tackle new problems, our approach prompts language models to self-generate relevant exemplars or knowledge in the context, before proceeding to solve the given problem. This method presents several advantages: it obviates the need for labeling or retrieving exemplars, offering generality and convenience; it can also tailor the generated exemplars and knowledge to each problem, offering adaptability. Experimental results show that our approach outperforms 0-shot CoT and manual few-shot CoT in a variety of reasoning tasks, including math problem solving in GSM8K and MATH, code generation in Codeforces, and other reasoning tasks in BIG-Bench.
     </details>

129. **On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes** [[pdf]](https://openreview.net/forum?id=3zKtaqxLhW) `ICLR 2024 Poster` (29 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The efficacy of GKD is demonstrated for distilling auto-regressive language models on summarization, translation, and arithmetic reasoning tasks, and task-agnostic distillation for instruction-tuning.
     </details>


     <details>
          <summary>Abstract</summary>
          Knowledge distillation (KD) is widely used for compressing a teacher model to reduce its inference cost and memory footprint, by training a smaller student model. However, current KD methods for auto-regressive sequence models suffer from distribution mismatch between output sequences seen during training and those generated by the student during inference. To address this issue, we introduce Generalized Knowledge Distillation (GKD). Instead of solely relying on a fixed set of output sequences, GKD trains the student on its self-generated output sequences by leveraging feedback from the teacher on such sequences. Unlike supervised KD approaches, GKD also offers the flexibility to employ alternative loss functions between the student and teacher, which can be useful when the student lacks the expressivity to mimic the teacher's distribution. Furthermore, GKD facilitates the seamless integration of distillation with RL fine-tuning (RLHF). We demonstrate the efficacy of GKD for distilling auto-regressive T5 language models on summarization, translation, and arithmetic reasoning tasks.
     </details>

130. **Query-Dependent Prompt Evaluation and Optimization with Offline Inverse RL** [[pdf]](https://openreview.net/forum?id=N6o0ZtPzTg) `ICLR 2024 Poster` (12 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This study introduces Prompt-OIRL, which harnesses offline inverse reinforcement learning to draw insights from offline prompting demonstration data and achieves the query-dependent prompt optimization objective by first learning an offline reward model.
     </details>


     <details>
          <summary>Abstract</summary>
          In this study, we aim to enhance the arithmetic reasoning ability of Large Language Models (LLMs) through zero-shot prompt optimization. We identify a previously overlooked objective of query dependency in such optimization and elucidate two ensuing challenges that impede the successful and economical design of prompt optimization techniques. One primary issue is the absence of an effective method to evaluate prompts during inference when the golden answer is unavailable. Concurrently, learning via interactions with the LLMs to navigate the expansive natural language prompting space proves to be resource-intensive.To address this, we introduce Prompt-OIRL, which harnesses offline inverse reinforcement learning to draw insights from offline prompting demonstration data. Such data exists as by-products when diverse prompts are benchmarked on open-accessible datasets. With Prompt-OIRL, the query-dependent prompt optimization objective is achieved by first learning an offline reward model. This model can evaluate any query-prompt pairs without accessing LLMs. Subsequently, a best-of-N strategy is deployed to recommend the optimal prompt. Our experimental evaluations across various LLM scales and arithmetic reasoning datasets underscore both the efficacy and economic viability of the proposed approach.
     </details>

131. **Escape Sky-high Cost: Early-stopping Self-Consistency for Multi-step Reasoning** [[pdf]](https://openreview.net/forum?id=ndR8Ytrzhh) `ICLR 2024 Poster` (10 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A simple and scalable sampling process, Ely-Stopping-Consistency (ESC), to greatly reduce the cost of SC without sacrificing performance and is conducted on three popular categories of reasoning tasks: arithmetic, commonsense and symbolic reasoning over language models with varying scales.
     </details>


     <details>
          <summary>Abstract</summary>
          Self-consistency (SC) has been a widely used decoding strategy for chain-of-thought reasoning. Despite bringing significant performance improvements across a variety of multi-step reasoning tasks, it is a high-cost method that requires multiple sampling with the preset size. In this paper, we propose a simple and scalable sampling process, Early-Stopping Self-Consistency (ESC), to greatly reduce the cost of SC without sacrificing performance. On this basis, one control scheme for ESC is further derivated to dynamically choose the performance-cost balance for different tasks and models. To demonstrate ESC's effectiveness, we conducted extensive experiments on three popular categories of reasoning tasks: arithmetic, commonsense and symbolic reasoning over language models with varying scales. The empirical results show that ESC reduces the average number of sampling of chain-of-thought reasoning by a significant margin on six benchmarks, including MATH (-33.8%), GSM8K (-80.1%), StrategyQA (-76.8%), CommonsenseQA (-78.5%), Coin Flip (-84.2%) and Last Letters (-67.4%), while attaining comparable performances.
     </details>

132. **Boosting of Thoughts: Trial-and-Error Problem Solving with Large Language Models** [[pdf]](https://openreview.net/forum?id=qBL04XXex6) `ICLR 2024 Poster` (4 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Boosting of Thoughts is presented, an automated prompting framework for problem solving with LLMs by iteratively exploring and self-evaluating many trees of thoughts in order to acquire an ensemble of trial-and-error reasoning experiences, which will serve as a new form of prompting to solve the complex problem.
     </details>


     <details>
          <summary>Abstract</summary>
          The reasoning performance of Large Language Models (LLMs) on a wide range of problems critically relies on chain-of-thought prompting, which involves providing a few chain of thought demonstrations as exemplars in prompts. Recent work, e.g., Tree of Thoughts, has pointed out the importance of exploration and self-evaluation in reasoning step selection for complex problem solving. In this paper, we present Boosting of Thoughts (BoT), an automated prompting framework for problem solving with LLMs by iteratively exploring and self-evaluating many trees of thoughts in order to acquire an ensemble of trial-and-error reasoning experiences, which will serve as a new form of prompting to solve the complex problem. Starting from a simple prompt without requiring examples, BoT iteratively explores and evaluates a large collection of reasoning steps, and more importantly, uses error analysis obtained from the LLM on them to explicitly revise prompting, which in turn enhances reasoning step generation, until a final answer is attained. Our experiments with GPT-4 and Llama2 across extensive complex mathematical problems demonstrate that BoT consistently achieves higher or comparable problem-solving rates than other advanced prompting approaches.
     </details>

133. **DyVal: Graph-informed Dynamic Evaluation of Large Language Models** [[pdf]](https://openreview.net/forum?id=gjfOL9z5Xr) `ICLR 2024 Spotlight` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have achieved remarkable performance in various evaluation benchmarks. However, concerns about their performance are raised on potential data contamination in their considerable volume of training corpus. Moreover, the static nature and fixed complexity of current benchmarks may inadequately gauge the advancing capabilities of LLMs. In this paper, we introduce DyVal, a novel, general, and flexible evaluation protocol for dynamic evaluation of LLMs. Based on our proposed dynamic evaluation framework, we build graph-informed DyVal by leveraging the structural advantage of directed acyclic graphs to dynamically generate evaluation samples with controllable complexities. DyVal generates challenging evaluation sets on reasoning tasks including mathematics, logical reasoning, and algorithm problems. We evaluate various LLMs ranging from Flan-T5-large to ChatGPT and GPT4. Experiments demonstrate that LLMs perform worse in DyVal-generated evaluation samples with different complexities, emphasizing the significance of dynamic evaluation. We also analyze the failure cases and results of different prompting methods. Moreover, DyVal-generated samples are not only evaluation sets, but also helpful data for fine-tuning to improve the performance of LLMs on existing benchmarks. We hope that DyVal can shed light on the future evaluation research of LLMs.
     </details>

134. **LEGO-Prover: Neural Theorem Proving with Growing Libraries** [[pdf]](http://arxiv.org/abs/2310.00656) `ICLR 2024` `Isabelle` (22 cite) (4 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work presents LEGO-Prover, which employs a growing skill library containing verified lemmas as skills to augment the capability of LLMs used in theorem proving, and advances the state-of-the-art pass rate on miniF2F-valid and miniF 2F-test.
     </details>


     <details>
          <summary>Abstract</summary>
          Despite the success of large language models (LLMs), the task of theorem proving still remains one of the hardest reasoning tasks that is far from being fully solved. Prior methods using language models have demonstrated promising results, but they still struggle to prove even middle school level theorems. One common limitation of these methods is that they assume a fixed theorem library during the whole theorem proving process. However, as we all know, creating new useful theorems or even new theories is not only helpful but crucial and necessary for advancing mathematics and proving harder and deeper results.In this work, we present LEGO-Prover, which employs a growing skill library containing verified lemmas as skills to augment the capability of LLMs used in theorem proving. By constructing the proof modularly, LEGO-Prover enables LLMs to utilize existing skills retrieved from the library and to create new skills during the proving process. These skills are further evolved (by prompting an LLM) to enrich the library on another scale. Modular and reusable skills are constantly added to the library to enable tackling increasingly intricate mathematical problems. Moreover, the learned library further bridges the gap between human proofs and formal proofs by making it easier to impute missing steps. LEGO-Prover advances the state-of-the-art pass rate on miniF2F-valid (48.0\% to 57.0\%) and miniF2F-test (45.5\% to 50.0\%). During the proving process, LEGO-Prover also generates over 20,000 skills (theorems/lemmas) and adds them to the growing library. Our ablation study indicates that these newly added skills are indeed helpful for proving theorems, resulting in a 4.9\% improvement in success rate
     </details>

135. **Reflexion: language agents with verbal reinforcement learning** [[pdf]](http://arxiv.org/abs/2303.11366) `NeurIPS 2023 Poster` (615 cite) (6 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Reflexion is a novel framework to reinforce language agents not by updating weights, but instead through linguistic feedback, which obtains significant improvements over a baseline agent across diverse tasks (sequential decision-making, coding, language reasoning).
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have been increasingly used to interact with external environments (e.g., games, compilers, APIs) as goal-driven agents. However, it remains challenging for these language agents to quickly and efficiently learn from trial-and-error as traditional reinforcement learning methods require extensive training samples and expensive model fine-tuning. We propose \emph{Reflexion}, a novel framework to reinforce language agents not by updating weights, but instead through linguistic feedback. Concretely, Reflexion agents verbally reflect on task feedback signals, then maintain their own reflective text in an episodic memory buffer to induce better decision-making in subsequent trials. Reflexion is flexible enough to incorporate various types (scalar values or free-form language) and sources (external or internally simulated) of feedback signals, and obtains significant improvements over a baseline agent across diverse tasks (sequential decision-making, coding, language reasoning). For example, Reflexion achieves a 91\% pass@1 accuracy on the HumanEval coding benchmark, surpassing the previous state-of-the-art GPT-4 that achieves 80\%. We also conduct ablation and analysis studies using different feedback signals, feedback incorporation methods, and agent types, and provide insights into how they affect performance. We release all code, demos, and datasets at \url{https://github.com/noahshinn024/reflexion}.
     </details>

136. **MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models** [[pdf]](http://arxiv.org/abs/2309.12284) `ICLR 2024 Spotlight` (163 cite) (13 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Experimental results on two popular benchmarks for mathematical reasoning demonstrate that MetaMath outperforms a suite of open-source LLMs by a significant margin.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have pushed the limits of natural language understanding and exhibited excellent problem-solving ability. Despite the great success, most existing open-source LLMs (\eg, LLaMA-2) are still far away from satisfactory for solving mathematical problems due to the complex reasoning procedures. To bridge this gap, we propose \emph{MetaMath}, a finetuned language model that specializes in mathematical reasoning. Specifically, we start by bootstrapping mathematical questions by rewriting the question from multiple perspectives, which results in a new dataset called {MetaMathQA}. Then we finetune the LLaMA-2 models on MetaMathQA. Experimental results on two popular benchmarks (\ie, GSM8K and MATH) for mathematical reasoning demonstrate that MetaMath outperforms a suite of open-source LLMs by a significant margin. Our MetaMath-7B model achieves $66.5\%$ on GSM8K and $19.8\%$ on MATH, exceeding the state-of-the-art models of the same size by $11.5\%$ and $8.7\%$. Particularly, MetaMath-70B achieves an accuracy of $82.3\%$ on GSM8K, slightly better than GPT-3.5-Turbo. We release the MetaMathQA dataset, the MetaMath models with different model sizes and the training code for public use.
     </details>

137. **SelfCheck: Using LLMs to Zero-Shot Check Their Own Step-by-Step Reasoning** [[pdf]](http://arxiv.org/abs/2308.00436) `ICLR 2024 Poster` (66 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes SelfCheck, a general-purpose zero-shot verification schema for recognizing errors in large language models and uses the results of these checks to improve question-answering performance by conducting weighted voting on multiple solutions to the question.
     </details>


     <details>
          <summary>Abstract</summary>
          The recent progress in large language models (LLMs), especially the invention of chain-of-thought prompting, has made it possible to automatically answer questions by stepwise reasoning. However, when faced with more complicated problems that require non-linear thinking, even the strongest LLMs make mistakes. To address this, we explore whether LLMs are able to recognize errors in their own step-by-step reasoning, without resorting to external resources. To this end, we propose SelfCheck, a general-purpose zero-shot verification schema for recognizing such errors. We then use the results of these checks to improve question-answering performance by conducting weighted voting on multiple solutions to the question. We test SelfCheck on three datasets---GSM8K, MathQA, and MATH---and find that it successfully recognizes errors and, in turn, increases final answer accuracies.
     </details>

138. **ToRA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving** [[pdf]](http://arxiv.org/abs/2309.17452) `ICLR 2024 Poster` (74 cite) (9 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper proposes ToRA a series of Tool-integrated Reasoning Agents designed to solve challenging mathematical problems by seamlessly integrating natural language reasoning with the utilization of external tools, thereby amalgamating the analytical prowess of language and the computational efficiency of tools.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models have made significant progress in various language tasks, yet they still struggle with complex mathematics. In this paper, we propose ToRA a series of Tool-integrated Reasoning Agents designed to solve challenging mathematical problems by seamlessly integrating natural language reasoning with the utilization of external tools (e.g., computation libraries and symbolic solvers), thereby amalgamating the analytical prowess of language and the computational efficiency of tools. To train ToRA, we curate interactive tool-use trajectories on mathematical datasets, apply imitation learning on the annotations, and propose output space shaping to further refine models' reasoning behavior. As a result, ToRA models significantly outperform open-source models on 10 mathematical reasoning datasets across all scales with 13%-19% absolute improvements on average. Notably, ToRA-7B reaches 44.6% on the competition-level dataset MATH, surpassing the best open-source model WizardMath-70B by 22% absolute. ToRA-34B is also the first open-source model that achieves an accuracy exceeding 50% on MATH, which significantly outperforms GPT-4's CoT result, and is competitive with GPT-4 solving problems with programs. Additionally, we conduct a comprehensive analysis of the benefits and remaining challenges of tool interaction for mathematical reasoning, providing valuable insights for future research.
     </details>

139. **Deductive Verification of Chain-of-Thought Reasoning** [[pdf]](http://arxiv.org/abs/2306.03872) `NeurIPS 2023` (77 cite) (6 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes Natural Program, a natural language-based deductive reasoning format that enables models to generate precise reasoning steps where subsequent steps are more rigorously grounded on prior steps and significantly enhances the rigor and trustfulness of generated reasoning steps.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) significantly benefit from Chain-of-thought (CoT) prompting in performing various reasoning tasks. While CoT allows models to produce more comprehensive reasoning processes, its emphasis on intermediate reasoning steps can inadvertently introduce hallucinations and accumulated errors, thereby limiting models’ ability to solve complex reasoning tasks. Inspired by how humans engage in careful and meticulous deductive logical reasoning processes to solve tasks, we seek to enable language models to perform explicit and rigorous deductive reasoning, and also ensure the trustworthiness of their reasoning process through self-verification. However, directly verifying the validity of an entire deductive reasoning process is challenging, even with advanced models like ChatGPT. In light of this, we propose to decompose a reasoning verification process into a series of step-by-step subprocesses, each only receiving their necessary context and premises. To facilitate this procedure, we propose Natural Program, a natural language-based deductive reasoning format. Our approach enables models to generate precise reasoning steps where subsequent steps are more rigorously grounded on prior steps. It also empowers language models to carry out reasoning self-verification in a step-by-step manner. By integrating this verification process into each deductive reasoning stage, we significantly enhance the rigor and trustfulness of generated reasoning steps. Along this process, we also improve the answer correctness on complex reasoning tasks.
     </details>

140. **MAmmoTH: Building Math Generalist Models through Hybrid Instruction Tuning** [[pdf]](http://arxiv.org/abs/2309.05653) `ICLR 2024` (232 cite) (22 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The MAmmoTH series substantially outperform existing open-source models on nine mathematical reasoning datasets across all scales with an average accuracy gain between 16% and 32%, and underscores the importance of diverse problem coverage and the use of hybrid rationales in developing superior math generalist models.
     </details>


     <details>
          <summary>Abstract</summary>
          We introduce MAmmoTH, a series of open-source large language models (LLMs) specifically tailored for general math problem-solving. The MAmmoTH models are trained on MathInstruct, our meticulously curated instruction tuning dataset. MathInstruct is compiled from 13 math datasets with intermediate rationales, six of which have rationales newly curated by us. It presents a unique hybrid of chain-of-thought (CoT) and program-of-thought (PoT) rationales, and also ensures extensive coverage of diverse fields in math. The hybrid of CoT and PoT not only unleashes the potential of tool use but also allows different thought processes for different math problems. As a result, the MAmmoTH series substantially outperform existing open-source models on nine mathematical reasoning datasets across all scales with an average accuracy gain between 16% and 32%. Remarkably, our MAmmoTH-7B model reaches 33% on MATH (a competition-level dataset), which exceeds the best open-source 7B model (WizardMath) by 23%, and the MAmmoTH-34B model achieves 44% accuracy on MATH, even surpassing GPT-4's CoT result. Our work underscores the importance of diverse problem coverage and the use of hybrid rationales in developing superior math generalist models.
     </details>

141. **Reasoning with Language Model Prompting: A Survey** [[pdf]](http://arxiv.org/abs/2212.09597) `ACL 2023 Long Papers` (220 cite) (5 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper provides a comprehensive survey of cutting-edge research on reasoning with language model prompting with comparisons and summaries and provides systematic resources to help beginners.
     </details>


     <details>
          <summary>Abstract</summary>
          Reasoning, as an essential ability for complex problem-solving, can provide back-end support for various real-world applications, such as medical diagnosis, negotiation, etc. This paper provides a comprehensive survey of cutting-edge research on reasoning with language model prompting. We introduce research works with comparisons and summaries and provide systematic resources to help beginners. We also discuss the potential reasons for emerging such reasoning abilities and highlight future research directions. Resources are available at https://github.com/zjunlp/Prompt4ReasoningPapers (updated periodically).
     </details>

142. **ROSCOE: A Suite of Metrics for Scoring Step-by-Step Reasoning** [[pdf]](http://arxiv.org/abs/2212.07919) `ICLR 2023` (97 cite) (7 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          ROSCOE is presented, a suite of interpretable, unsupervised automatic scores that improve and extend previous text generation evaluation metrics and can measure semantic consistency, logicality, informativeness, fluency, and factuality - among other traits - by leveraging properties of step-by-step rationales.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models show improved downstream task performance when prompted to generate step-by-step reasoning to justify their final answers. These reasoning steps greatly improve model interpretability and verification, but objectively studying their correctness (independent of the final answer) is difficult without reliable methods for automatic evaluation. We simply do not know how often the stated reasoning steps actually support the final end task predictions. In this work, we present ROSCOE, a suite of interpretable, unsupervised automatic scores that improve and extend previous text generation evaluation metrics. To evaluate ROSCOE against baseline metrics, we design a typology of reasoning errors and collect synthetic and human evaluation scores on commonly used reasoning datasets. In contrast with existing metrics, ROSCOE can measure semantic consistency, logicality, informativeness, fluency, and factuality — among other traits — by leveraging properties of step-by-step rationales. We empirically verify the strength of our metrics on five human annotated and six programmatically perturbed diagnostics datasets - covering a diverse set of tasks that require reasoning skills and show that ROSCOE can consistently outperform baseline metrics.
     </details>

143. **Solving Challenging Math Word Problems Using GPT-4 Code Interpreter with Code-based Self-Verification** [[pdf]](http://arxiv.org/abs/2308.07921) `ICLR 2024 Poster` (106 cite) (17 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The effect of code on enhancing LLMs' reasoning capability by introducing different constraints on the Code Usage Frequency of GPT-4 Code Interpreter is explored, and a novel and effective prompting method, explicit \uline{c}ode-based \ULine{s}elf-\uline {v}erification~(CSV), is proposed to further boost the mathematical reasoning potential of GPN.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent progress in large language models (LLMs) like GPT-4 and PaLM-2 has brought significant advancements in addressing math reasoning problems. In particular, OpenAI's latest version of GPT-4, known as GPT-4 Code Interpreter, shows remarkable performance on challenging math datasets. In this paper, we explore the effect of code on enhancing LLMs' reasoning capability by introducing different constraints on the Code Usage Frequency of GPT-4 Code Interpreter. We found that its success can be largely attributed to its powerful skills in generating and executing code, evaluating the output of code execution, and rectifying its solution when receiving unreasonable outputs. Based on this insight, we propose a novel and effective prompting method, explicit $\underline{\text{c}}$ode-based $\underline{\text{s}}$elf-$\underline{\text{v}}$erification (CSV), to further boost the mathematical reasoning potential of GPT-4 Code Interpreter. This method employs a zero-shot prompt on GPT-4 Code Interpreter to encourage it to use code to self-verify its answers. In instances where the verification state registers as "False", the model shall automatically amend its solution, analogous to our approach of rectifying errors during a mathematics examination. Furthermore, we recognize that the states of the verification result indicate the confidence of a solution, which can improve the effectiveness of majority voting. With GPT-4 Code Interpreter and CSV, we achieve an impressive zero-shot accuracy on MATH dataset $\textbf{(53.9}$% → $\textbf{84.3}$%$\textbf{)}$.
     </details>

144. **Large Language Models Can Be Easily Distracted by Irrelevant Context** [[pdf]](https://proceedings.mlr.press/v202/shi23a.html) `ICML 2023 Poster` (353 cite) (6 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work investigates the distractibility of large language models, i.e., how the model problem-solving accuracy can be influenced by irrelevant context, and introduces Grade-School Math with Irrelevant Context (GSM-IC), an arithmetic reasoning dataset with irrelevant information in the problem description.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models have achieved impressive performance on various natural language processing tasks. However, so far they have been evaluated primarily on benchmarks where all information in the input context is relevant for solving the task. In this work, we investigate the *distractibility* of large language models, i.e., how the model prediction can be distracted by irrelevant context. In particular, we introduce Grade-School Math with Irrelevant Context (GSM-IC), an arithmetic reasoning dataset with irrelevant information in the problem description. We use this benchmark to measure the distractibility of different prompting techniques for large language models, and find that the model is easily distracted by irrelevant information. We also identify several approaches for mitigating this deficiency, such as decoding with self-consistency and adding to the prompt an instruction that tells the language model to ignore the irrelevant information.
     </details>

145. **Specializing Smaller Language Models towards Multi-Step Reasoning** [[pdf]](https://proceedings.mlr.press/v202/fu23d.html) `ICML 2023 Oral` (177 cite) (14 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work shows two important aspects of model abilities: there exists a very complex balance/ tradeoff between language models' multi-dimensional abilities, and by paying the price of decreased generic ability, it can clearly lift up the scaling curve of models smaller than 10B towards a specialized multi-step math reasoning ability.
     </details>


     <details>
          <summary>Abstract</summary>
          The surprising ability of Large Language Models (LLMs) to perform well on complex reasoning with only few-shot chain-of-thought prompts is believed to emerge only in very large-scale models. We show that such abilities can, in fact, be distilled down from GPT-3.5 (≥ 175B) to T5 variants (≤ 11B). We propose model specialization, to specialize the model’s ability towards a target task. The hypothesis is that large models (commonly viewed as larger than 100B) have strong modeling power such that they can perform a large spectrum of tasks. Small models (commonly viewed as smaller than 10B) have limited model capacity, but if we specialize their capacity towards a target task, the model can achieve decent performance improvements. We use multi-step math reasoning as our testbed because it is a very typical emergent ability. We show two important aspects of model abilities: (1) balancing language model’s performance on multiple tasks is a delicate matter, as improvements on one task may compromise other tasks; (2) yet by intentionally paying the price of decreased generic ability, we can clearly improve across different model scales smaller than 10B towards a specialized multi-step math reasoning ability. We further give comprehensive discussions about important design choices for better generalization, including the data format mixture and the start model checkpoint. We hope our practice and discoveries can serve as an important attempt towards specialized smaller models in the new research paradigm set by LLMs.
     </details>

146. **Large Language Models Are Reasoning Teachers** [[pdf]](https://aclanthology.org/2023.acl-long.830) `ACL 2023 Long Papers` (232 cite) (15 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper uses very large models as reasoning teachers to enable complex reasoning in smaller models and reduce model size requirements by several orders of magnitude, and proposes Fine-tune-CoT, a method that generates reasoning samples from very large teacher models to fine-tunes smaller models.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent works have shown that chain-of-thought (CoT) prompting can elicit language models to solve complex reasoning tasks, step-by-step. However, prompt-based CoT methods are dependent on very large models such as GPT-3 175B which are prohibitive to deploy at scale. In this paper, we use these large models as reasoning teachers to enable complex reasoning in smaller models and reduce model size requirements by several orders of magnitude. We propose Fine-tune-CoT, a method that generates reasoning samples from very large teacher models to fine-tune smaller models. We evaluate our method on a wide range of public models and complex tasks. We find that Fine-tune-CoT enables substantial reasoning capability in small models, far outperforming prompt-based baselines and even the teacher model in many tasks. Additionally, we extend our method by leveraging the teacher model’s ability to generate multiple distinct rationales for each original sample. Enriching the fine-tuning data with such diverse reasoning results in a substantial performance boost across datasets, even for very small models. We conduct ablations and sample studies to understand the emergence of reasoning capabilities of student models. Our code implementation and data are available at https://github.com/itsnamgyu/reasoning-teacher.
     </details>

147. **Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models** [[pdf]](https://aclanthology.org/2023.acl-long.147) `ACL 2023 Long Papers` (194 cite) (5 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The experimental results over GPT-3 show that the proposed zero-shot prompting consistently outperforms Zero- shot-CoT across all datasets by a large margin, is comparable to or exceeds Zero-shot-Program-of-Thought Prompting, and has comparable performance with 8-shot CoT prompting on the math reasoning problem.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have recently been shown to deliver impressive performance in various NLP tasks. To tackle multi-step reasoning tasks, Few-shot chain-of-thought (CoT) prompting includes a few manually crafted step-by-step reasoning demonstrations which enable LLMs to explicitly generate reasoning steps and improve their reasoning task accuracy. To eliminate the manual efforts, Zero-shot-CoT concatenates the target problem statement with “Let’s think step by step” as an input prompt to LLMs. Despite the success of Zero-shot-CoT, it still suffers from three pitfalls: calculation errors, missing-step errors, and semantic misunderstanding errors. To address the missing-step errors, we propose Plan-and-Solve (PS) Prompting. It consists of two components: first, devising a plan to divide the entire task into smaller subtasks, and then carrying out the subtasks according to the plan. To address the calculation errors and improve the quality of generated reasoning steps, we extend PS prompting with more detailed instructions and derive PS+ prompting. We evaluate our proposed prompting strategy on ten datasets across three reasoning problems. The experimental results over GPT-3 show that our proposed zero-shot prompting consistently outperforms Zero-shot-CoT across all datasets by a large margin, is comparable to or exceeds Zero-shot-Program-of-Thought Prompting, and has comparable performance with 8-shot CoT prompting on the math reasoning problem. The code can be found at https://github.com/AGI-Edgerunners/Plan-and-Solve-Prompting.
     </details>

148. **Making Language Models Better Reasoners with Step-Aware Verifier** [[pdf]](https://aclanthology.org/2023.acl-long.291) `ACL 2023 Long Papers` (138 cite) (11 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper presents DiVeRSe (Diverse Verifier on Reasoning Step), a novel approach that further enhances the reasoning capability of language models and achieves new state-of-the-art results on six of eight reasoning benchmarks.
     </details>


     <details>
          <summary>Abstract</summary>
          Few-shot learning is a challenging task that requires language models to generalize from limited examples. Large language models like GPT-3 and PaLM have made impressive progress in this area, but they still face difficulties in reasoning tasks such as GSM8K, a benchmark for arithmetic problems. To improve their reasoning skills, previous work has proposed to guide the language model with prompts that elicit a series of reasoning steps before giving the final answer, achieving a significant improvement on GSM8K from 17.9% to 58.1% in problem-solving rate. In this paper, we present DiVeRSe (Diverse Verifier on Reasoning Step), a novel approach that further enhances the reasoning capability of language models. DiVeRSe has three main components: first, it generates diverse prompts to explore different reasoning paths for the same question; second, it uses a verifier to filter out incorrect answers based on a weighted voting scheme; and third, it verifies each reasoning step individually instead of the whole chain. We evaluate DiVeRSe on the latest language model code-davinci-002 and show that it achieves new state-of-the-art results on six of eight reasoning benchmarks (e.g., GSM8K 74.4% to 83.2%).
     </details>

149. **LAMBADA: Backward Chaining for Automated Reasoning in Natural Language** [[pdf]](https://aclanthology.org/2023.acl-long.361) `ACL 2023 Long Papers` (62 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A Backward Chaining algorithm, called LAMBADA, that decomposes reasoning into four sub-modules, that are simply implemented by few-shot prompted LLM inference, and achieves sizable accuracy boosts over state-of-the-art forward reasoning methods on two challenging logical reasoning datasets.
     </details>


     <details>
          <summary>Abstract</summary>
          Remarkable progress has been made on automated reasoning with natural text, by using Large Language Models (LLMs) and methods such as Chain-of-Thought prompting and Selection-Inference. These techniques search for proofs in the forward direction from axioms to the conclusion, which suffers from a combinatorial explosion of the search space, and thus high failure rates for problems requiring longer chains of reasoning. The classical automated reasoning literature has shown that reasoning in the backward direction (i.e. from intended conclusion to supporting axioms) is significantly more efficient at proof-finding. Importing this intuition into the LM setting, we develop a Backward Chaining algorithm, called LAMBADA, that decomposes reasoning into four sub-modules, that are simply implemented by few-shot prompted LLM inference. We show that LAMBADA achieves sizable accuracy boosts over state-of-the-art forward reasoning methods on two challenging logical reasoning datasets, particularly when deep and accurate proof chains are required.
     </details>

150. **DT-Solver: Automated Theorem Proving with Dynamic-Tree Sampling Guided by Proof-level Value Function** [[pdf]](https://aclanthology.org/2023.acl-long.706) `ACL 2023` `Lean, Isabelle` (20 cite) (5 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel Dynamic-Tree Driven Theorem Solver (DT-Solver) is proposed, which introduces a dynamic-tree Monte-Carlo search algorithm, which dynamically allocates computing budgets for different state confidences, guided by a new proof-level value function to discover proof states that require substantial exploration.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent advances in neural theorem-proving resort to large language models and tree searches. When proving a theorem, a language model advises single-step actions based on the current proving state and the tree search finds a sequence of correct steps using actions given by the language model. However, prior works often conduct constant computation efforts for each proving state while ignoring that the hard states often need more exploration than easy states. Moreover, they evaluate and guide the proof search solely depending on the current proof state instead of considering the whole proof trajectory as human reasoning does. Here, to accommodate general theorems, we propose a novel Dynamic-Tree Driven Theorem Solver (DT-Solver) by guiding the search procedure with state confidence and proof-level values. Specifically, DT-Solver introduces a dynamic-tree Monte-Carlo search algorithm, which dynamically allocates computing budgets for different state confidences, guided by a new proof-level value function to discover proof states that require substantial exploration. Experiments on two popular theorem-proving datasets, PISA and Mathlib, show significant performance gains by our DT-Solver over the state-of-the-art approaches, with a 6.65% improvement on average in terms of success rate. And especially under low computing resource settings (11.03% improvement on average).
     </details>

151. **Tree-Based Representation and Generation of Natural and Mathematical Language** [[pdf]](https://aclanthology.org/2023.acl-long.205) `ACL 2023 Long Papers` (11 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A series of modifications to existing language models are proposed to jointly represent and generate text and math: representing mathematical expressions as sequences of node tokens in their operator tree format, using math symbol and tree position embeddings to preserve the semantic and structural properties of mathematical expressions, and using a constrained decoding method to generate mathematically valid expressions.
     </details>


     <details>
          <summary>Abstract</summary>
          Mathematical language in scientific communications and educational scenarios is important yet relatively understudied compared to natural languages. Recent works on mathematical language focus either on representing stand-alone mathematical expressions, especially in their natural tree format, or mathematical reasoning in pre-trained natural language models. Existing works on jointly modeling and generating natural and mathematical languages simply treat mathematical expressions as text, without accounting for the rigid structural properties of mathematical expressions. In this paper, we propose a series of modifications to existing language models to jointly represent and generate text and math: representing mathematical expressions as sequences of node tokens in their operator tree format, using math symbol and tree position embeddings to preserve the semantic and structural properties of mathematical expressions, and using a constrained decoding method to generate mathematically valid expressions. We ground our modifications in GPT-2, resulting in a model MathGPT, and demonstrate that it outperforms baselines on mathematical expression generation tasks.
     </details>

152. **FERMAT: An Alternative to Accuracy for Numerical Reasoning** [[pdf]](https://aclanthology.org/2023.acl-long.838) `ACL 2023 Long Papers` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces a multi-view evaluation set for numerical reasoning in English, called FERMAT, which evaluates models on various key numerical reasoning aspects such as number understanding, mathematical operations, and training dependency.
     </details>


     <details>
          <summary>Abstract</summary>
          While pre-trained language models achieve impressive performance on various NLP benchmarks, they still struggle with tasks that require numerical reasoning. Recent advances in improving numerical reasoning are mostly achieved using very large language models that contain billions of parameters and are not accessible to everyone. In addition, numerical reasoning is measured using a single score on existing datasets. As a result, we do not have a clear understanding of the strengths and shortcomings of existing models on different numerical reasoning aspects and therefore, potential ways to improve them apart from scaling them up. Inspired by CheckList (Ribeiro et al., 2020), we introduce a multi-view evaluation set for numerical reasoning in English, called FERMAT. Instead of reporting a single score on a whole dataset, FERMAT evaluates models on various key numerical reasoning aspects such as number understanding, mathematical operations, and training dependency. Apart from providing a comprehensive evaluation of models on different numerical reasoning aspects, FERMAT enables a systematic and automated generation of an arbitrarily large training or evaluation set for each aspect. The datasets and codes are publicly available to generate further multi-view data for ulterior tasks and languages.
     </details>

153. **LeanDojo: Theorem Proving with Retrieval-Augmented Language Models** [[pdf]](http://arxiv.org/abs/2306.15626) `NeurIPS 2023` `Lean` (111 cite) (12 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduces LeanDojo: an open-source Lean playground consisting of toolkits, data, models, and benchmarks, and develops ReProver (Retrieval-Augmented Prover): an LLM-based prover augmented with retrieval for selecting premises from a vast math library.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have shown promise in proving formal theorems using proof assistants such as Lean. However, existing methods are difficult to reproduce or build on, due to private code, data, and large compute requirements. This has created substantial barriers to research on machine learning methods for theorem proving. This paper removes these barriers by introducing LeanDojo: an open-source Lean playground consisting of toolkits, data, models, and benchmarks. LeanDojo extracts data from Lean and enables interaction with the proof environment programmatically. It contains fine-grained annotations of premises in proofs, providing valuable data for premise selection—a key bottleneck in theorem proving. Using this data, we develop ReProver (Retrieval-Augmented Prover): an LLM-based prover augmented with retrieval for selecting premises from a vast math library. It is inexpensive and needs only one GPU week of training. Our retriever leverages LeanDojo's program analysis capability to identify accessible premises and hard negative examples, which makes retrieval much more effective. Furthermore, we construct a new benchmark consisting of 98,734 theorems and proofs extracted from Lean's math library. It features challenging data split requiring the prover to generalize to theorems relying on novel premises that are never used in training. We use this benchmark for training and evaluation, and experimental results demonstrate the effectiveness of ReProver over non-retrieval baselines and GPT-4. We thus provide the first set of open-source LLM-based theorem provers without any proprietary datasets and release it under a permissive MIT license to facilitate further research.
     </details>

154. **AdaPlanner: Adaptive Planning from Feedback with Language Models** [[pdf]](http://arxiv.org/abs/2305.16653) `NeurIPS 2023 Poster` (80 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A closed-loop approach, AdaPlanner, is proposed, which allows the LLM agent to refine its self-generated plan adaptively in response to environmental feedback, and develops a code-style LLM prompt structure that facilitates plan generation across a variety of tasks, environments, and agent capabilities.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have recently demonstrated the potential in acting as autonomous agents for sequential decision-making tasks. However, most existing methods either take actions greedily without planning or rely on static plans that are not adaptable to environmental feedback. Consequently, the sequential decision-making performance of LLM agents degenerates with problem complexity and plan horizons increase. We propose a closed-loop approach, AdaPlanner, which allows the LLM agent to refine its self-generated plan adaptively in response to environmental feedback. In AdaPlanner, the LLM agent adaptively refines its plan from feedback with both in-plan and out-of-plan refinement strategies. To mitigate hallucination, we develop a code-style LLM prompt structure that facilitates plan generation across a variety of tasks, environments, and agent capabilities. Furthermore, we propose a skill discovery mechanism that leverages successful plans as few-shot exemplars, enabling the agent to plan and refine with fewer task demonstrations. Our experiments in the ALFWorld and MiniWoB++ environments demonstrate that AdaPlanner outperforms state-of-the-art baselines by 3.73% and 4.11% while utilizing 2x and 600x fewer samples, respectively. The implementation of AdaPlanner is available at https://github.com/haotiansun14/AdaPlanner.
     </details>

155. **Self-Refine: Iterative Refinement with Self-Feedback** [[pdf]](http://arxiv.org/abs/2303.17651) `NeurIPS 2023 Poster` (846 cite) (30 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Self-Refine is introduced, an approach for improving initial outputs from LLMs through iterative feedback and refinement that demonstrates that even state-of-the-art LLMs like GPT-4 can be further improved at test time using this simple, standalone approach.
     </details>


     <details>
          <summary>Abstract</summary>
          Like humans, large language models (LLMs) do not always generate the best output on their first try. Motivated by how humans refine their written text, we introduce Self-Refine, an approach for improving initial outputs from LLMs through iterative feedback and refinement. The main idea is to generate an initial output using an LLMs; then, the same LLMs provides *feedback* for its output and uses it to *refine* itself, iteratively. Self-Refine does not require any supervised training data, additional training, or reinforcement learning, and instead uses a single LLM as the generator, refiner and the feedback provider. We evaluate Self-Refine across 7 diverse tasks, ranging from dialog response generation to mathematical reasoning, using state-of-the-art (GPT-3.5, ChatGPT, and GPT-4) LLMs. Across all evaluated tasks, outputs generated with Self-Refine are preferred by humans and automatic metrics over those generated with the same LLM using conventional one-step generation, improving by $\sim$20\% absolute on average in task performance. Our work demonstrates that even state-of-the-art LLMs like GPT-4 can be further improved at test-time using our simple, standalone approach.
     </details>

156. **Neural Machine Translation for Mathematical Formulae** [[pdf]](http://arxiv.org/abs/2305.16433) `ACL 2023 Long Papers` (2 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          While recurrent, recursive, and transformer networks struggle with preserving all contained information, it is found that convolutional sequence-to-sequence networks achieve 95.1% and 90.7% exact matches, respectively.
     </details>


     <details>
          <summary>Abstract</summary>
          We tackle the problem of neural machine translation of mathematical formulae between ambiguous presentation languages and unambiguous content languages. Compared to neural machine translation on natural language, mathematical formulae have a much smaller vocabulary and much longer sequences of symbols, while their translation requires extreme precision to satisfy mathematical information needs. In this work, we perform the tasks of translating from LaTeX to Mathematica as well as from LaTeX to semantic LaTeX. While recurrent, recursive, and transformer networks struggle with preserving all contained information, we find that convolutional sequence-to-sequence networks achieve 95.1% and 90.7% exact matches, respectively.
     </details>

157. **Subgoal-based Demonstration Learning for Formal Theorem Proving** [[pdf]](http://arxiv.org/abs/2305.16366) `ICML 2024` `Isabelle` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduces a subgoal-based demonstration learning framework, specifically designed to enhance the efficiency of proof search in LLMs, and builds upon recent advances in diffusion models to predict the optimal organization.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) present a promising pathway for advancing the domain of formal theorem proving. In this paper, we aim to improve the performance of LLMs in formal theorem proving by thoroughly examining the structure and organization of demonstrative in-context examples. We introduce a subgoal-based demonstration learning framework, specifically designed to enhance the efficiency of proof search in LLMs. First, drawing upon the insights of subgoal learning from reinforcement learning and robotics, we propose the construction of distinct subgoals for each demonstration example and refine these subgoals in accordance with the pertinent theories of subgoal learning. Second, we build upon recent advances in diffusion models to predict the optimal organization, simultaneously addressing two intricate issues that persist within the domain of demonstration organization: subset selection and order determination. Our integration of subgoal-based learning has notably increased proof accuracy from 38.9% to 44.1% on the miniF2F benchmark. Furthermore, the adoption of diffusion models for demonstration organization can lead to an additional enhancement in accuracy to 45.5%, or a $5\times$ improvement in sampling efficiency compared to previously established methods.
     </details>

158. **Tree of Thoughts: Deliberate Problem Solving with Large Language Models** [[pdf]](http://arxiv.org/abs/2305.10601) `NeurIPS 2023 Oral` (1000 cite) (25 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A new framework for language model inference, Tree of Thoughts (ToT), which generalizes over the popular Chain of Thought approach to prompting language models, and enables exploration over coherent units of text (thoughts) that serve as intermediate steps toward problem solving.
     </details>


     <details>
          <summary>Abstract</summary>
          Language models are increasingly being deployed for general problem solving across a wide range of tasks, but are still confined to token-level, left-to-right decision-making processes during inference. This means they can fall short in tasks that require exploration, strategic lookahead, or where initial decisions play a pivotal role. To surmount these challenges, we introduce a new framework for language model inference, Tree of Thoughts (ToT), which generalizes over the popular Chain of Thought approach to prompting language models, and enables exploration over coherent units of text (thoughts) that serve as intermediate steps toward problem solving. ToT allows LMs to perform deliberate decision making by considering multiple different reasoning paths and self-evaluating choices to decide the next course of action, as well as looking ahead or backtracking when necessary to make global choices.Our experiments show that ToT significantly enhances language models’ problem-solving abilities on three novel tasks requiring non-trivial planning or search: Game of 24, Creative Writing, and Mini Crosswords. For instance, in Game of 24, while GPT-4 with chain-of-thought prompting only solved 4\% of tasks, our method achieved a success rate of 74\%. Code repo with all prompts: https://github.com/princeton-nlp/tree-of-thought-llm.
     </details>

159. **Verify-and-Edit: A Knowledge-Enhanced Chain-of-Thought Framework** [[pdf]](http://arxiv.org/abs/2305.03268) `ACL 2023 Long Papers` (106 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The Verify-and-Edit framework for CoT prompting is proposed, which seeks to increase prediction factuality by post-editing reasoning chains according to external knowledge and lead to accuracy improvements in multiple open-domain question-answering tasks.
     </details>


     <details>
          <summary>Abstract</summary>
          As large language models (LLMs) have become the norm in NLP, demonstrating good performance in generation and reasoning tasks, one of its most fatal disadvantages is the lack of factual correctness. Generating unfactual texts not only leads to lower performances but also degrades the trust and validity of their applications. Chain-of-Thought (CoT) prompting improves trust and model performance on complex reasoning tasks by generating interpretable reasoning chains, but still suffers from factuality concerns in knowledge-intensive tasks. In this paper, we propose the Verify-and-Edit framework for CoT prompting, which seeks to increase prediction factuality by post-editing reasoning chains according to external knowledge. Building on top of GPT-3, our framework lead to accuracy improvements in multiple open-domain question-answering tasks.
     </details>

160. **ReAct: Synergizing Reasoning and Acting in Language Models** [[pdf]](http://arxiv.org/abs/2210.03629) `ICLR 2023` (1000 cite) (29 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The use of LLMs are explored to generate both reasoning traces and task-specific actions in an interleaved manner, allowing for greater synergy between the two: reasoning traces help the model induce, track, and update action plans as well as handle exceptions, while actions allow it to interface with external sources to gather additional information.
     </details>


     <details>
          <summary>Abstract</summary>
          While large language models (LLMs) have demonstrated impressive capabilities across tasks in language understanding and interactive decision making, their abilities for reasoning (e.g. chain-of-thought prompting) and acting (e.g. action plan generation) have primarily been studied as separate topics. In this paper, we explore the use of LLMs to generate both reasoning traces and task-specific actions in an interleaved manner, allowing for greater synergy between the two: reasoning traces help the model induce, track, and update action plans as well as handle exceptions, while actions allow it to interface with external sources, such as knowledge bases or environments, to gather additional information. We apply our approach, named ReAct, to a diverse set of language and decision making tasks and demonstrate its effectiveness over state-of-the-art baselines, as well as improved human interpretability and trustworthiness over methods without reasoning or acting components. Concretely, on question answering (HotpotQA) and fact verification (Fever), ReAct overcomes issues of hallucination and error propagation prevalent in chain-of-thought reasoning by interacting with a simple Wikipedia API, and generates human-like task-solving trajectories that are more interpretable than baselines without reasoning traces. On two interactive decision making benchmarks (ALFWorld and WebShop), ReAct outperforms imitation and reinforcement learning methods by an absolute success rate of 34% and 10% respectively, while being prompted with only one or two in-context examples.
     </details>

161. **Magnushammer: A Transformer-Based Approach to Premise Selection** [[pdf]](https://arxiv.org/abs/2303.04488v1) `ICLR 2024` `Isabelle` (26 cite) (6 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work demonstrates that contrastive training with the transformer architecture can achieve higher-quality retrieval of relevant premises, without the engineering overhead, in a novel approach to premise selection.
     </details>


     <details>
          <summary>Abstract</summary>
          This paper presents a novel approach to premise selection, a crucial reasoning task in automated theorem proving. Traditionally, symbolic methods that rely on extensive domain knowledge and engineering effort are applied to this task. In contrast, this work demonstrates that contrastive training with the transformer architecture can achieve higher-quality retrieval of relevant premises, without the knowledge or feature engineering overhead. Our method, Magnushammer, outperforms the most advanced and widely used automation tool in interactive theorem proving called Sledgehammer. On the PISA and miniF2f benchmarks Magnushammer achieves $59.5\%$ (against $38.3\%$) and $34.0\%$ (against $20.9\%$) success rates, respectively. By combining Magnushammer with a language-model-based automated theorem prover, we further improve the state-of-the-art proof success rate from $57.0\%$ to $71.0\%$ on the PISA benchmark using $4$x fewer parameters. Moreover, we develop and open source a novel dataset for premise selection, containing textual representations of (proof state, relevant premise) pairs. To the best of our knowledge, this is the largest available premise selection dataset, and the first dataset of this kind for the Isabelle proof assistant.
     </details>

162. **Self-Consistency Improves Chain of Thought Reasoning in Language Models** [[pdf]](http://arxiv.org/abs/2203.11171) `ICLR 2023` (1000 cite) (87 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper proposes a new decoding strategy, self-consistency, to replace the naive greedy decoding used in chain-of-thought prompting that first samples a diverse set of reasoning paths instead of only taking the greedy one, and then selects the most consistent answer by marginalizing out the sampled reasoning paths.
     </details>


     <details>
          <summary>Abstract</summary>
          Chain-of-thought prompting combined with pretrained large language models has achieved encouraging results on complex reasoning tasks. In this paper, we propose a new decoding strategy, self-consistency, to replace the naive greedy decoding used in chain-of-thought prompting. It first samples a diverse set of reasoning paths instead of only taking the greedy one, and then selects the most consistent answer by marginalizing out all possible reasoning paths. Self-consistency leverages the intuition that a complex reasoning problem typically admits multiple different ways of thinking leading to its unique correct answer. Our extensive empirical evaluation shows that self-consistency boosts the performance of chain-of-thought prompting with a striking margin on a range of popular arithmetic and commonsense reasoning benchmarks, including GSM8K (+17.9%), SVAMP (+11.0%), AQuA (+12.2%), StrategyQA (+6.4%) and ARC-challenge (+3.9%).
     </details>

163. **MathPrompter: Mathematical Reasoning using Large Language Models** [[pdf]](http://arxiv.org/abs/2303.05398) `ACL 2023 Industry Track` (0 cite) (4 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have limited performance when solving arithmetic reasoning tasks and often provide incorrect answers. Unlike natural language understanding, math problems typically have a single correct answer, making the task of generating accurate solutions more challenging for LLMs. To the best of our knowledge, we are not aware of any LLMs that indicate their level of confidence in their responses which fuels a trust deficit in these models impeding their adoption. To address this deficiency, we propose ‘MathPrompter’, a technique that improves performance of LLMs on arithmetic problems along with increased reliance in the predictions. MathPrompter uses the Zero-shot chain-of-thought prompting technique to generate multiple algebraic expressions or python functions to solve the same math problem in different ways and thereby raise the confidence level in the output results. This is in contrast to other prompt based CoT methods, where there is no check on the validity of the intermediate steps followed. Our technique improves over state-of-the-art on the ‘MultiArith’ dataset (78.7% - 92.5%) evaluated using 175B parameter GPT-based LLM.
     </details>

164. **Dynamic Prompt Learning via Policy Gradient for Semi-structured Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2209.14610) `ICLR 2023` (181 cite) (17 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel approach is proposed, PromptPG, which utilizes policy gradient to learn to select in-context examples from a small amount of training data and then constructs the corresponding prompt for the test example, which verifies its effectiveness in selecting in- context examples.
     </details>


     <details>
          <summary>Abstract</summary>
          Mathematical reasoning, a core ability of human intelligence, presents unique challenges for machines in abstract thinking and logical reasoning. Recent large pre-trained language models such as GPT-3 have achieved remarkable progress on mathematical reasoning tasks written in text form, such as math word problems (MWP). However, it is unknown if the models can handle more complex problems that involve math reasoning over heterogeneous information, such as tabular data. To fill the gap, we present Tabular Math Word Problems (TabMWP), a new dataset containing 38,431 open-domain grade-level problems that require mathematical reasoning on both textual and tabular data. Each question in TabMWP is aligned with a tabular context, which is presented as an image, semi-structured text, and a structured table. There are two types of questions: free-text and multi-choice, and each problem is annotated with gold solutions to reveal the multi-step reasoning process. We evaluate different pre-trained models on TabMWP, including the GPT-3 model in a few-shot setting. As earlier studies suggest, since few-shot GPT-3 relies on the selection of in-context examples, its performance is unstable and can degrade to near chance. The unstable issue is more severe when handling complex problems like TabMWP. To mitigate this, we further propose a novel approach, PromptPG, which utilizes policy gradient to learn to select in-context examples from a small amount of training data and then constructs the corresponding prompt for the test example. Experimental results show that our method outperforms the best baseline by 5.31% on the accuracy metric and reduces the prediction variance significantly compared to random selection, which verifies its effectiveness in selecting in-context examples. The data and code are available at https://promptpg.github.io.
     </details>

165. **Toolformer: Language Models Can Teach Themselves to Use Tools** [[pdf]](http://arxiv.org/abs/2302.04761) `NeurIPS 2023 Oral` (1000 cite) (23 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduces Toolformer, a model trained to decide which APIs to call, when to call them, what arguments to pass, and how to best incorporate the results into future token prediction, which achieves substantially improved zero-shot performance across a variety of downstream tasks.
     </details>


     <details>
          <summary>Abstract</summary>
          Language models (LMs) exhibit remarkable abilities to solve new tasks from just a few examples or textual instructions, especially at scale. They also, paradoxically, struggle with basic functionality, such as arithmetic or factual lookup, where much simpler and smaller specialized models excel. In this paper, we show that LMs can teach themselves to *use external tools* via simple APIs and achieve the best of both worlds. We introduce *Toolformer*, a model trained to decide which APIs to call, when to call them, what arguments to pass, and how to best incorporate the results into future token prediction. This is done in a self-supervised way, requiring nothing more than a handful of demonstrations for each API. We incorporate a range of tools, including a calculator, a Q&A system, a search engine, a translation system, and a calendar. Toolformer achieves substantially improved zero-shot performance across a variety of downstream tasks, often competitive with much larger models, without sacrificing its core language modeling abilities.
     </details>

166. **Mathematical Capabilities of ChatGPT** [[pdf]](http://arxiv.org/abs/2301.13867) `NeurIPS 2023` (299 cite) (3 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is found that ChatGPT can be used most successfully as a mathematical assistant for querying facts, acting as a Mathematical search engine and knowledge base interface, and GPT-4 can additionally be used for undergraduate-level mathematics but fails on graduate-level difficulty.
     </details>


     <details>
          <summary>Abstract</summary>
          We investigate the mathematical capabilities of two iterations of ChatGPT (released 9-January-2023 and 30-January-2023) and of GPT-4 by testing them on publicly available datasets, as well as hand-crafted ones, using a novel methodology. In contrast to formal mathematics, where large databases of formal proofs are available (e.g., the Lean Mathematical Library), current datasets of natural-language mathematics, used to benchmark language models, either cover only elementary mathematics or are very small. We address this by publicly releasing two new datasets: GHOSTS and miniGHOSTS. These are the first natural-language datasets curated by working researchers in mathematics that (1) aim to cover graduate-level mathematics, (2) provide a holistic overview of the mathematical capabilities of language models, and (3) distinguish multiple dimensions of mathematical reasoning. These datasets also test whether ChatGPT and GPT-4 can be helpful assistants to professional mathematicians by emulating use cases that arise in the daily professional activities of mathematicians. We benchmark the models on a range of fine-grained performance metrics. For advanced mathematics, this is the most detailed evaluation effort to date. We find that ChatGPT can be used most successfully as a mathematical assistant for querying facts, acting as a mathematical search engine and knowledge base interface. GPT-4 can additionally be used for undergraduate-level mathematics but fails on graduate-level difficulty. Contrary to many positive reports in the media about GPT-4 and ChatGPT's exam-solving abilities (a potential case of selection bias), their overall mathematical performance is well below the level of a graduate student. Hence, if your goal is to use ChatGPT to pass a graduate-level math exam, you would be better off copying from your average peer!
     </details>

167. **Complexity-Based Prompting for Multi-step Reasoning** [[pdf]](http://arxiv.org/abs/2210.00720) `ICLR 2023 Poster` (309 cite) (24 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes complexity-based prompting, a simple and effective example selection scheme for multi-step reasoning that substantially improves multi- step reasoning accuracy and achieves new state-of-the-art (SOTA) performance on three math benchmarks and two BigBenchHard tasks.
     </details>


     <details>
          <summary>Abstract</summary>
          We study the task of prompting large-scale language models to perform multi-step reasoning. Existing work shows that when prompted with a chain of thoughts (CoT), sequences of short sentences describing intermediate reasoning steps towards a final answer, large language models can generate new reasoning chains and predict answers for new inputs. A central question is which reasoning examples make the most effective prompts. In this work, we propose complexity-based prompting, a simple and effective example selection scheme for multi-step reasoning. We show that prompts with higher reasoning complexity, i.e., chains with more reasoning steps, achieve substantially better performance on math word reasoning tasks over strong baselines. We further extend our complexity-based criteria from prompting (selecting inputs) to decoding (selecting outputs), where we sample multiple reasoning chains from the model, then choose the majorityof generated answers from complex reasoning chains (over simple chains). When used to prompt GPT-3, our approach substantially improves multi-step reasoning accuracy, with an 8.6% absolute improvement on GSM8K, and 6.4% on MathQA. Compared with existing example selection schemes like manual tuning or retrieval-based selection, selection based on reasoning complexity is intuitive, easy to implement, and annotation-efficient. Further results demonstrate the robustness of performance gains from complex prompts under format perturbation and distribution shift.
     </details>

168. **PAL: Program-aided Language Models** [[pdf]](http://arxiv.org/abs/2211.10435) `ICML 2023 Poster` (330 cite) (26 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper presents Program-Aided Language models (PAL): a novel approach that uses the LLM to read natural language problems and generate programs as the intermediate reasoning steps, but offloads the solution step to a runtime such as a Python interpreter.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have demonstrated an impressive ability to perform arithmetic and symbolic reasoning tasks, when provided with a few examples at test time ("few-shot prompting"). Much of this success can be attributed to prompting methods such as "chain-of-thought", which employ LLMs for both understanding the problem description by decomposing it into steps, as well as solving each step of the problem. While LLMs seem to be adept at this sort of step-by-step decomposition, LLMs often make logical and arithmetic mistakes in the solution part, even when the problem is decomposed correctly. In this paper, we present Program-Aided Language models (PAL): a novel approach that uses the LLM to read natural language problems and generate programs as the intermediate reasoning steps, but offloads the solution step to a runtime such as a Python interpreter. With PAL, decomposing the natural language problem into runnable steps remains the only learning task for the LLM, while solving is delegated to the interpreter. We demonstrate this synergy between a neural LLM and a symbolic interpreter across 13 mathematical, symbolic, and algorithmic reasoning tasks from BIG-Bench Hard and others. In all these natural language reasoning tasks, generating code using an LLM and reasoning using a Python interpreter leads to more accurate results than much larger models. For example, PAL using Codex achieves state-of-the-art few-shot accuracy on GSM8K, surpassing PaLM which uses chain-of-thought by absolute 15% top-1.
     </details>

169. **Let's Verify Step by Step** [[pdf]](https://arxiv.org/abs/2305.20050) `ICLR 2024 Poster` (344 cite) (27 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work conducts its own investigation, finding that process supervision significantly outperforms outcome supervision for training models to solve problems from the challenging MATH dataset and shows that active learning significantly improves the efficacy of process supervision.
     </details>


     <details>
          <summary>Abstract</summary>
          In recent years, large language models have greatly improved in their ability to perform complex multi-step reasoning. However, even state-of-the-art models still regularly produce logical mistakes. To train more reliable models, we can turn either to outcome supervision, which provides feedback for a final result, or process supervision, which provides feedback for each intermediate reasoning step. Given the importance of training reliable models, and given the high cost of human feedback, it is important to carefully compare the both methods. Recent work has already begun this comparison, but many questions still remain. We conduct our own investigation, finding that process supervision significantly outperforms outcome supervision for training models to solve problems from the challenging MATH dataset. Our process-supervised model solves 78% of problems from a representative subset of the MATH test set. Additionally, we show that active learning significantly improves the efficacy of process supervision. To support related research, we also release PRM800K, the complete dataset of 800,000 step-level human feedback labels used to train our best reward model.
     </details>

170. **Let GPT be a Math Tutor: Teaching Math Word Problem Solvers with Customized Exercise Generation** [[pdf]](https://arxiv.org/abs/2305.14386) `EMNLP 2023` (25 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel approach for distilling math word problem solving capabilities from large language models (LLMs) into smaller, more efficient student models designed to consider the student model's weaknesses and foster a tailored learning experience by generating targeted exercises aligned with educational science principles.
     </details>


     <details>
          <summary>Abstract</summary>
          In this paper, we present a novel approach for distilling math word problem solving capabilities from large language models (LLMs) into smaller, more efficient student models. Our approach is designed to consider the student model’s weaknesses and foster a tailored learning experience by generating targeted exercises aligned with educational science principles, such as knowledge tracing and personalized learning. Concretely, we let GPT-3 be a math tutor and run two steps iteratively: 1) assessing the student model’s current learning status on a GPT-generated exercise book, and 2) improving the student model by training it with tailored exercise samples generated by GPT-3. Experimental results reveal that our approach outperforms LLMs (e.g., GPT-3 and PaLM) in accuracy across three distinct benchmarks while employing significantly fewer parameters. Furthermore, we provide a comprehensive analysis of the various components within our methodology to substantiate their efficacy.
     </details>

171. **Interpretable Math Word Problem Solution Generation via Step-by-step Planning** [[pdf]](https://arxiv.org/abs/2306.00784) `ACL 2023 Long Papers` (12 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A step-by-step planning approach for intermediate solution generation, which strategically plans the generation of the next solution step based on the MWP and the previous solution steps, and improves the accuracy and interpretability of the solution on both automatic metrics and human evaluation.
     </details>


     <details>
          <summary>Abstract</summary>
          Solutions to math word problems (MWPs) with step-by-step explanations are valuable, especially in education, to help students better comprehend problem-solving strategies. Most existing approaches only focus on obtaining the final correct answer. A few recent approaches leverage intermediate solution steps to improve final answer correctness but often cannot generate coherent steps with a clear solution strategy. Contrary to existing work, we focus on improving the correctness and coherence of the intermediate solutions steps. We propose a step-by-step planning approach for intermediate solution generation, which strategically plans the generation of the next solution step based on the MWP and the previous solution steps. Our approach first plans the next step by predicting the necessary math operation needed to proceed, given history steps, then generates the next step, token-by-token, by prompting a language model with the predicted math operation. Experiments on the GSM8K dataset demonstrate that our approach improves the accuracy and interpretability of the solution on both automatic metrics and human evaluation.
     </details>

172. **The Art of SOCRATIC QUESTIONING: Recursive Thinking with Large Language Models** [[pdf]](https://aclanthology.org/2023.emnlp-main.255) `EMNLP 2023 Main` (7 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The qualitative analysis clearly shows that the intermediate reasoning steps elicited by SOCRATIC QUESTIONING are similar to humans' recursively thinking process of complex reasoning problems.
     </details>


     <details>
          <summary>Abstract</summary>
          Chain-of-Thought (CoT) prompting enables large language models to solve complex reasoning problems by generating intermediate steps. However, confined by its inherent single-pass and sequential generation process, CoT heavily relies on the initial decisions, causing errors in early steps to accumulate and impact the final answers. In contrast, humans adopt recursive thinking when tackling complex reasoning problems, i.e. iteratively breaking the original problem into approachable sub-problems and aggregating their answers to resolve the original one. Inspired by the human cognitive process, we propose SOCRATIC QUESTIONING, a divide-and-conquer style algorithm that mimics the recursive thinking process. Specifically, SOCRATIC QUESTIONING leverages large language models to raise and answer sub-questions until collecting enough information to tackle the original question. Unlike CoT, SOCRATIC QUESTIONING explicitly navigates the thinking space, stimulates effective recursive thinking, and is more robust towards errors in the thinking process. Extensive experiments on several complex reasoning tasks, including MMLU, MATH, LogiQA, and visual question-answering demonstrate significant performance improvements over the state-of-the-art prompting methods, such as CoT, and Tree-of-Thought. The qualitative analysis clearly shows that the intermediate reasoning steps elicited by SOCRATIC QUESTIONING are similar to humans’ recursively thinking process of complex reasoning problems.
     </details>

173. **A Survey of Deep Learning for Mathematical Reasoning** [[pdf]](http://arxiv.org/abs/2212.10535) `ACL 2023` (99 cite) (16 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This survey paper reviews the key tasks, datasets, and methods at the intersection of mathematical reasoning and deep learning over the past decade, and evaluates existing benchmarks and methods and discusses future research directions in this domain.
     </details>


     <details>
          <summary>Abstract</summary>
          Mathematical reasoning is a fundamental aspect of human intelligence and is applicable in various fields, including science, engineering, finance, and everyday life. The development of artificial intelligence (AI) systems capable of solving math problems and proving theorems in language has garnered significant interest in the fields of machine learning and natural language processing. For example, mathematics serves as a testbed for aspects of reasoning that are challenging for powerful deep learning models, driving new algorithmic and modeling advances. On the other hand, recent advances in large-scale neural language models have opened up new benchmarks and opportunities to use deep learning for mathematical reasoning. In this survey paper, we review the key tasks, datasets, and methods at the intersection of mathematical reasoning and deep learning over the past decade. We also evaluate existing benchmarks and methods, and discuss future research directions in this domain.
     </details>

174. **UniGeo: Unifying Geometry Logical Reasoning via Reformulating Mathematical Expression** [[pdf]](http://arxiv.org/abs/2212.02746) `EMNLP 2022 Main` (51 cite) (7 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A large-scale Unified Geometry problem benchmark, UniGeo, is constructed and a unified multi-task Geometric Transformer framework, Geoformer, is presented to tackle calculation and proving problems simultaneously in the form of sequence generation, which finally shows the reasoning ability can be improved on both two tasks by unifying formulation.
     </details>


     <details>
          <summary>Abstract</summary>
          Geometry problem solving is a well-recognized testbed for evaluating the high-level multi-modal reasoning capability of deep models. In most existing works, two main geometry problems: calculation and proving, are usually treated as two specific tasks, hindering a deep model to unify its reasoning capability on multiple math tasks. However, in essence, these two tasks have similar problem representations and overlapped math knowledge which can improve the understanding and reasoning ability of a deep model on both two tasks. Therefore, we construct a large-scale Unified Geometry problem benchmark, UniGeo, which contains 4,998 calculation problems and 9,543 proving problems. Each proving problem is annotated with a multi-step proof with reasons and mathematical expressions. The proof can be easily reformulated as a proving sequence that shares the same formats with the annotated program sequence for calculation problems. Naturally, we also present a unified multi-task Geometric Transformer framework, Geoformer, to tackle calculation and proving problems simultaneously in the form of sequence generation, which finally shows the reasoning ability can be improved on both two tasks by unifying formulation. Furthermore, we propose a Mathematical Expression Pretraining (MEP) method that aims to predict the mathematical expressions in the problem solution, thus improving the Geoformer model. Experiments on the UniGeo demonstrate that our proposed Geoformer obtains state-of-the-art performance by outperforming task-specific model NGS with over 5.6% and 3.2% accuracies on calculation and proving problems, respectively.
     </details>

175. **UniRPG: Unified Discrete Reasoning over Table and Text as Program Generation** [[pdf]](https://aclanthology.org/2022.emnlp-main.508) `EMNLP 2022 Main` (18 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper proposes UniRPG, a semantic-parsing-based approach advanced in interpretability and scalability, to perform Unified discrete Reasoning over heterogeneous knowledge resources, i.e., table and text, as Program Generation, and designs a distant supervision approach for programmer learning.
     </details>


     <details>
          <summary>Abstract</summary>
          Question answering requiring discrete reasoning, e.g., arithmetic computing, comparison, and counting, over knowledge is a challenging task. In this paper, we propose UniRPG, a semantic-parsing-based approach advanced in interpretability and scalability, to perform Unified discrete Reasoning over heterogeneous knowledge resources, i.e., table and text, as Program Generation. Concretely, UniRPG consists of a neural programmer and a symbolic program executor,where a program is the composition of a set of pre-defined general atomic and higher-order operations and arguments extracted from table and text. First, the programmer parses a question into a program by generating operations and copying arguments, and then, the executor derives answers from table and text based on the program. To alleviate the costly program annotation issue, we design a distant supervision approach for programmer learning, where pseudo programs are automatically constructed without annotated derivations. Extensive experiments on the TAT-QA dataset show that UniRPG achieves tremendous improvements and enhances interpretability and scalability compared with previous state-of-the-art methods, even without derivation annotation. Moreover, it achieves promising performance on the textual dataset DROP without derivation annotation.
     </details>

176. **Analogical Math Word Problems Solving with Enhanced Problem-Solution Association** [[pdf]](https://aclanthology.org/2022.emnlp-main.643) `EMNLP 2022 Main` (13 cite) (3 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The proposed analogical learning strategy promotes the performance of MWP-BERT on Math23k over the state-of-the-art model Generate2Rank, with 5 times fewer parameters in the encoder.
     </details>


     <details>
          <summary>Abstract</summary>
          Math word problem (MWP) solving is an important task in question answering which requires human-like reasoning ability. Analogical reasoning has long been used in mathematical education, as it enables students to apply common relational structures of mathematical situations to solve new problems. In this paper, we propose to build a novel MWP solver by leveraging analogical MWPs, which advance the solver’s generalization ability across different kinds of MWPs. The key idea, named analogy identification, is to associate the analogical MWP pairs in a latent space, i.e., encoding an MWP close to another analogical MWP, while leaving away from the non-analogical ones. Moreover, a solution discriminator is integrated into the MWP solver to enhance the association between an MWP and its true solution. The evaluation results verify that our proposed analogical learning strategy promotes the performance of MWP-BERT on Math23k over the state-of-the-art model Generate2Rank, with 5 times fewer parameters in the encoder. We also find that our model has a stronger generalization ability in solving difficult MWPs due to the analogical learning from easy MWPs.
     </details>

177. **Structure-Unified M-Tree Coding Solver for Math Word Problem** [[pdf]](https://aclanthology.org/2022.emnlp-main.556) `EMNLP 2022 Main` (7 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Experimental results on the widely used MAWPS and Math23K datasets have demonstrated that SUMC-Solver not only outperforms several state-of-the-art models under similar experimental settings but also performs much better under low-resource conditions.
     </details>


     <details>
          <summary>Abstract</summary>
          As one of the challenging NLP tasks, designing math word problem (MWP) solvers has attracted increasing research attention for the past few years. In previous work, models designed by taking into account the properties of the binary tree structure of mathematical expressions at the output side have achieved better performance. However, the expressions corresponding to a MWP are often diverse (e.g., n1+n2 × n3-n4, n3× n2-n4+n1, etc.), and so are the corresponding binary trees, which creates difficulties in model learning due to the non-deterministic output space. In this paper, we propose the Structure-Unified M-Tree Coding Solver (SUMC-Solver), which applies a tree with any M branches (M-tree) to unify the output structures. To learn the M-tree, we use a mapping to convert the M-tree into the M-tree codes, where codes store the information of the paths from tree root to leaf nodes and the information of leaf nodes themselves, and then devise a Sequence-to-Code (seq2code) model to generate the codes. Experimental results on the widely used MAWPS and Math23K datasets have demonstrated that SUMC-Solver not only outperforms several state-of-the-art models under similar experimental settings but also performs much better under low-resource conditions.
     </details>

178. **Improving compositional generalization for multi-step quantitative reasoning in question answering** [[pdf]](https://aclanthology.org/2022.emnlp-main.125) `EMNLP 2022 Main` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper demonstrates how modeling the compositional nature of quantitative text can enhance the performance and robustness of QA models, allowing them to capture arithmetic logic that is expressed verbally.
     </details>


     <details>
          <summary>Abstract</summary>
          Quantitative reasoning is an important aspect of question answering, especially when numeric and verbal cues interact to indicate sophisticated, multi-step programs. In this paper, we demonstrate how modeling the compositional nature of quantitative text can enhance the performance and robustness of QA models, allowing them to capture arithmetic logic that is expressed verbally. Borrowing from the literature on semantic parsing, we propose a method that encourages the QA models to adjust their attention patterns and capture input/output alignments that are meaningful to the reasoning task. We show how this strategy improves program accuracy and renders the models more robust against overfitting as the number of reasoning steps grows. Our approach is designed as a standalone module which can be prepended to many existing models and trained in an end-to-end fashion without the need for additional supervisory signal. As part of this exercise, we also create a unified dataset building on four previously released numerical QA datasets over tabular data.
     </details>

179. **Automatic Generation of Socratic Subquestions for Teaching Math Word Problems** [[pdf]](http://arxiv.org/abs/2211.12835) `EMNLP 2022 Main` (30 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work explores the ability of large language models (LMs) in generating sequential questions for guiding math word problem-solving and proposes various guided question generation schemes based on input conditioning and reinforcement learning.
     </details>


     <details>
          <summary>Abstract</summary>
          Socratic questioning is an educational method that allows students to discover answers to complex problems by asking them a series of thoughtful questions. Generation of didactically sound questions is challenging, requiring understanding of the reasoning process involved in the problem. We hypothesize that such questioning strategy can not only enhance the human performance, but also assist the math word problem (MWP) solvers. In this work, we explore the ability of large language models (LMs) in generating sequential questions for guiding math word problem-solving. We propose various guided question generation schemes based on input conditioning and reinforcement learning. On both automatic and human quality evaluations, we find that LMs constrained with desirable question properties generate superior questions and improve the overall performance of a math word problem solver. We conduct a preliminary user study to examine the potential value of such question generation models in the education domain. Results suggest that the difficulty level of problems plays an important role in determining whether questioning improves or hinders human performance. We discuss the future of using such questioning strategies in education.
     </details>

180. **Draft, Sketch, and Prove: Guiding Formal Theorem Provers with Informal Proofs** [[pdf]](http://arxiv.org/abs/2210.12283) `ICLR 2023` `Isabelle` (99 cite) (17 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Draft, Sketch, and Prove (DSP), a method that maps informal proofs to formal proof sketches, and uses the sketches to guide an automated prover by directing its search to easier sub-problems, is introduced.
     </details>


     <details>
          <summary>Abstract</summary>
          The formalization of existing mathematical proofs is a notoriously difficult process. Despite decades of research on automation and proof assistants, writing formal proofs remains arduous and only accessible to a few experts. While previous studies to automate formalization focused on powerful search algorithms, no attempts were made to take advantage of available informal proofs. In this work, we introduce Draft, Sketch, and Prove (DSP), a method that maps informal proofs to formal proof sketches, and uses the sketches to guide an automated prover by directing its search to easier sub-problems. We investigate two relevant setups where informal proofs are either written by humans or generated by a language model. Our experiments and ablation studies show that large language models are able to produce well-structured formal sketches that follow the same reasoning steps as the informal proofs. Guiding an automated prover with these sketches enhances its performance from $20.9\%$ to $39.3\%$ on a collection of mathematical competition problems.
     </details>

181. **The Unreliability of Explanations in Few-shot Prompting for Textual Reasoning** [[pdf]](https://openreview.net/forum?id=Bct2f8fRd8S) `NeurIPS 2022 Poster` (129 cite) (6 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work studies two NLP tasks that involve reasoning over text, namely question answering and natural language inference, and shows that explanations judged by humans to be good--logically consistent with the input and the prediction--more likely cooccur with accurate predictions.
     </details>


     <details>
          <summary>Abstract</summary>
          Does prompting a large language model (LLM) like GPT-3 with explanations improve in-context learning? We study this question on two NLP tasks that involve reasoning over text, namely question answering and natural language inference. We test the performance of four LLMs on three textual reasoning datasets using prompts that include explanations in multiple different styles. For these tasks, we find that including explanations in the prompts for OPT, GPT-3 (davinci), and InstructGPT (text-davinci-001) only yields small to moderate accuracy improvements over standard few-show learning. However, text-davinci-002 is able to benefit more substantially.We further show that explanations generated by the LLMs may not entail the models’ predictions nor be factually grounded in the input, even on simple tasks with extractive explanations. However, these flawed explanations can still be useful as a way to verify LLMs’ predictions post-hoc. Through analysis in our three settings, we show that explanations judged by humans to be good—logically consistent with the input and the prediction—more likely cooccur with accurate predictions. Following these observations, we train calibrators using automatically extracted scores that assess the reliability of explanations, allowing us to improve performance post-hoc across all of our datasets.
     </details>

182. **ELASTIC: Numerical Reasoning with Adaptive Symbolic Compiler** [[pdf]](https://openreview.net/forum?id=gd7ZI0X7Q-h) `NeurIPS 2022 Poster` (15 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduces the ELASTIC model, which is constituted of the RoBERTa as the Encoder and a Compiler with four modules: Reasoning Manager, Operator Generator, Operands Generator, and Memory Register, and it is domain agnostic by supporting the expansion of diverse operators without caring about the number of operands.
     </details>


     <details>
          <summary>Abstract</summary>
          Numerical reasoning over text is a challenging task of Artificial Intelligence (AI), requiring reading comprehension and numerical reasoning abilities. Previous approaches use numerical reasoning programs to represent the reasoning process. However, most works do not separate the generation of operators and operands, which are key components of a numerical reasoning program, thus limiting their ability to generate such programs for complicated tasks. In this paper, we introduce the numEricaL reASoning with adapTive symbolIc Compiler (ELASTIC) model, which is constituted of the RoBERTa as the Encoder and a Compiler with four modules: Reasoning Manager, Operator Generator, Operands Generator, and Memory Register. ELASTIC is robust when conducting complicated reasoning. Also, it is domain agnostic by supporting the expansion of diverse operators without caring about the number of operands it contains. Experiments show that ELASTIC achieves 68.96 and 65.21 of execution accuracy and program accuracy on the FinQA dataset and 83.00 program accuracy on the MathQA dataset, outperforming previous state-of-the-art models significantly.
     </details>

183. **Automatic Chain of Thought Prompting in Large Language Models** [[pdf]](http://arxiv.org/abs/2210.03493) `ICLR 2023 Poster` (403 cite) (14 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          An automatic CoT prompting method that samples questions with diversity and generates reasoning chains to construct demonstrations and consistently matches or exceeds the performance of the CoT paradigm that requires manual designs of demonstrations.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) can carry out complex reasoning tasks by generating intermediate reasoning steps. These steps are triggered by what is called chain-of-thought (CoT) prompting, which comes in two flavors: one leverages a simple prompt like "Let’s think step by step" to facilitate step-by-step reasoning before answering a question (Zero-Shot-CoT). The other uses manual demonstrations, each composed of a question and a reasoning chain that leads to an answer (Manual-CoT). Unfortunately, the superior performance of the latter strategy crucially hinges on manually generating task-specific demonstrations. This makes it far less scalable and more dependent on the talent of the CoT engineer. We show that such manual efforts may be eliminated by leveraging LLMs to generate the reasoning chains on its own. Since these generated chains often come with mistakes we propose a number of mitigation strategies. Our proposed Auto-CoT method automaticaly samples diverse questions and we perform post-processing quality control to generate usable reasoning chains from Zero-Shot-CoT. On ten public benchmark reasoning tasks, Auto-CoT performs on par with Manual-CoT without the need for human intervention. Code is available at https://github.com/amazon-research/auto-cot.
     </details>

184. **ConvFinQA: Exploring the Chain of Numerical Reasoning in Conversational Finance Question Answering** [[pdf]](http://arxiv.org/abs/2210.03849) `EMNLP 2022 Main` (69 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a new large-scale dataset, ConvFinQA, aiming to study the chain of numerical reasoning in conversational question answering, and conducts comprehensive experiments and analyses with both the neural symbolic methods and the prompting-based methods to provide insights into the reasoning mechanisms.
     </details>


     <details>
          <summary>Abstract</summary>
          With the recent advance in large pre-trained language models, researchers have achieved record performances in NLP tasks that mostly focus on language pattern matching. The community is experiencing the shift of the challenge from how to model language to the imitation of complex reasoning abilities like human beings. In this work, we investigate the application domain of finance that involves real-world, complex numerical reasoning. We propose a new large-scale dataset, ConvFinQA, aiming to study the chain of numerical reasoning in conversational question answering. Our dataset poses great challenge in modeling long-range, complex numerical reasoning paths in real-world conversations. We conduct comprehensive experiments and analyses with both the neural symbolic methods and the prompting-based methods, to provide insights into the reasoning mechanisms of these two divisions. We believe our new dataset should serve as a valuable resource to push forward the exploration of real-world, complex reasoning tasks as the next research focus. Our dataset and code is publicly available at https://github.com/czyssrs/ConvFinQA.
     </details>

185. **Language models are multilingual chain-of-thought reasoners** [[pdf]](http://arxiv.org/abs/2210.03057) `ICLR 2023 Poster` (215 cite) (6 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is found that the ability to solve MGSM problems via chain-of-thought prompting emerges with increasing model scale, and that models have strikingly strong multilingual reasoning abilities, even in underrepresented languages such as Bengali and Swahili.
     </details>


     <details>
          <summary>Abstract</summary>
          We evaluate the reasoning abilities of large language models in multilingual settings. We introduce the Multilingual Grade School Math (MGSM) benchmark, by manually translating 250 grade-school math problems from the GSM8K dataset (Cobbe et al., 2021) into ten typologically diverse languages. We find that the ability to solve MGSM problems via chain-of-thought prompting emerges with increasing model scale, and that models have strikingly strong multilingual reasoning abilities, even in underrepresented languages such as Bengali and Swahili. Finally, we show that multilingual reasoning abilities of language models extend to other tasks such as commonsense reasoning and word-in-context semantic judgment. The MGSM benchmark is publicly available at AnonymousLink and the supplementary material.
     </details>

186. **Least-to-Most Prompting Enables Complex Reasoning in Large Language Models** [[pdf]](https://openreview.net/forum?id=WZH7099tgfM) `ICLR 2023 Poster` (788 cite) (49 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Experimental results on tasks related to symbolic manipulation, compositional generalization, and math reasoning reveal that least-to-most prompting is capable of generalizing to more difficult problems than those seen in the prompts.
     </details>


     <details>
          <summary>Abstract</summary>
          Chain-of-thought prompting has demonstrated remarkable performance on various natural language reasoning tasks. However, it tends to perform poorly on tasks which requires solving problems harder than the exemplars shown in the prompts. To overcome this challenge of easy-to-hard generalization, we propose a novel prompting strategy, least-to-most prompting. The key idea in this strategy is to break down a complex problem into a series of simpler subproblems and then solve them in sequence. Solving each subproblem is facilitated by the answers to previously solved subproblems. Our experimental results on tasks related to symbolic manipulation, compositional generalization, and math reasoning reveal that least-to-most prompting is capable of generalizing to more difficult problems than those seen in the prompts. A notable finding is that when the GPT-3 code-davinci-002 model is used with least-to-most prompting, it can solve the compositional generalization benchmark SCAN in any split (including length split) with an accuracy of at least 99\% using just 14 exemplars, compared to only 16\% accuracy with chain-of-thought prompting. This is particularly noteworthy because neural-symbolic models in the literature that specialize in solving SCAN are trained on the entire training set containing over 15,000 examples. We have included prompts for all the tasks in the Appendix.
     </details>

187. **Decomposed Prompting: A Modular Approach for Solving Complex Tasks** [[pdf]](https://openreview.net/forum?id=_nGgzQjzaRy) `ICLR 2023 Poster` (284 cite) (18 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is shown that the flexibility and modularity of Decomposed Prompting allows it to outperform prior work on few-shot prompting using GPT3 and to incorporate a symbolic information retrieval within the decomposition framework, leading to improved performance on both tasks.
     </details>


     <details>
          <summary>Abstract</summary>
          Few-shot prompting is a surprisingly powerful way to use Large Language Models (LLMs) to solve various tasks. However, this approach struggles as the task complexity increases or when the individual reasoning steps of the task themselves are hard to learn, especially when embedded in more complex tasks. To address this, we propose Decomposed Prompting, a new approach to solve complex tasks by decomposing them (via prompting) into simpler sub-tasks that can be delegated to a library of prompting-based LLMs dedicated to these sub-tasks. This modular structure allows each prompt to be optimized for its specific sub-task, further decomposed if necessary, and even easily replaced with more effective prompts, trained models, or symbolic functions if desired.We show that the flexibility and modularity of Decomposed Prompting allows it to outperform prior work on few-shot prompting using GPT3. On symbolic reasoning tasks, we can further decompose sub-tasks that are hard for LLMs into even simpler solvable sub-tasks. When the complexity comes from the input length, we can recursively decompose the task into the same task but with smaller inputs. We also evaluate our approach on textual multi-step reasoning tasks: on long-context multi-hop QA task, we can more effectively teach the sub-tasks via our separate sub-tasks prompts; and on open-domain multi-hop QA, we can incorporate a symbolic information retrieval within our decomposition framework, leading to improved performance on both tasks. Datasets, Code and Prompts available at https://github.com/allenai/DecomP.
     </details>

188. **Learning Math Reasoning from Self-Sampled Correct and Partially-Correct Solutions** [[pdf]](https://openreview.net/forum?id=4D4TSJE6-K) `ICLR 2023 Poster` (30 cite) (5 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is shown that the use of self-sampled correct and partially-correct solutions can benefit learning and help guide the sampling process, leading to more efficient exploration of the solution space and the effectiveness of the method is shown.
     </details>


     <details>
          <summary>Abstract</summary>
          Pretrained language models have shown superior performance on many natural language processing tasks, yet they still struggle at multi-step formal reasoning tasks like grade school math problems. One key challenge of finetuning them to solve such math reasoning problems is that many existing datasets only contain one reference solution for each problem, despite the fact that there are often alternative solutions resembling different reasoning paths to the final answer. This way, the finetuned models are biased towards the limited reference solutions, which limits their generalization to unseen examples. To mitigate this issue, we propose to let the model perform sampling during training and learn from both self-sampled fully-correct solutions, which yield the correct answer upon execution, and partially-correct solutions, whose intermediate state matches an intermediate state of a known correct solution. We show that our use of self-sampled correct and partially-correct solutions can benefit learning and help guide the sampling process, leading to more efficient exploration of the solution space. Additionally, we explore various training objectives to support learning from multiple solutions per example and find they greatly affect the performance. Experiments on two math reasoning datasets show the effectiveness of our method compared to learning from a single reference solution with MLE, where we improve PASS@100 from 35.5% to 44.5% for GSM8K, and 27.6% to 36.2% PASS@80 for MathQA. Such improvements are also consistent across different model sizes.
     </details>

189. **Learning to Reason Deductively: Math Word Problem Solving as Complex Relation Extraction** [[pdf]](http://arxiv.org/abs/2203.10316) `ACL 2022 Long Papers` (64 cite) (11 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work views the task as a complex relation extraction problem, and proposes a novel approach that presents explainable deductive reasoning steps to iteratively construct target expressions, where each step involves a primitive operation over two quantities defining their relation.
     </details>


     <details>
          <summary>Abstract</summary>
          Solving math word problems requires deductive reasoning over the quantities in the text. Various recent research efforts mostly relied on sequence-to-sequence or sequence-to-tree models to generate mathematical expressions without explicitly performing relational reasoning between quantities in the given context. While empirically effective, such approaches typically do not provide explanations for the generated expressions. In this work, we view the task as a complex relation extraction problem, proposing a novel approach that presents explainable deductive reasoning steps to iteratively construct target expressions, where each step involves a primitive operation over two quantities defining their relation. Through extensive experiments on four benchmark datasets, we show that the proposed model significantly outperforms existing strong baselines. We further demonstrate that the deductive procedure not only presents more explainable steps but also enables us to make more accurate predictions on questions that require more complex reasoning.
     </details>

190. **Practice Makes a Solver Perfect: Data Augmentation for Math Word Problem Solvers** [[pdf]](https://aclanthology.org/2022.naacl-main.310) `NAACL 2022 Main` (11 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper proposes several data augmentation techniques broadly categorized into Substitution and Paraphrasing based methods and shows that proposed methods increase the generalization and robustness of existing solvers.
     </details>


     <details>
          <summary>Abstract</summary>
          Existing Math Word Problem (MWP) solvers have achieved high accuracy on benchmark datasets. However, prior works have shown that such solvers do not generalize well and rely on superficial cues to achieve high performance. In this paper, we first conduct experiments to showcase that this behaviour is mainly associated with the limited size and diversity present in existing MWP datasets. Next, we propose several data augmentation techniques broadly categorized into Substitution and Paraphrasing based methods. By deploying these methods we increase the size of existing datasets by five folds. Extensive experiments on two benchmark datasets across three state-of-the-art MWP solvers shows that proposed methods increase the generalization and robustness of existing solvers. On average, proposed methods significantly increase the state-of-the-art results by over five percentage points on benchmark datasets. Further, the solvers trained on the augmented dataset performs comparatively better on the challenge test set. We also show the effectiveness of proposed techniques through ablation studies and verify the quality of augmented samples through human evaluation.
     </details>

191. **Solving Quantitative Reasoning Problems with Language Models** [[pdf]](http://arxiv.org/abs/2206.14858) `NeurIPS 2022 Poster` (553 cite) (45 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Language models have achieved remarkable performance on a wide range of tasks that require natural language understanding. Nevertheless, state-of-the-art models have generally struggled with tasks that require quantitative reasoning, such as solving mathematics, science, and engineering questions at the college level. To help close this gap, we introduce Minerva, a large language model pretrained on general natural language data and further trained on technical content. The model achieves strong performance in a variety of evaluations, including state-of-the-art performance on the MATH dataset. We also evaluate our model on over two hundred undergraduate-level problems in physics, biology, chemistry, economics, and other sciences that require quantitative reasoning, and find that the model can correctly answer nearly a quarter of them.
     </details>

192. **MultiHiertt: Numerical Reasoning over Multi Hierarchical Tabular and Textual Data** [[pdf]](http://arxiv.org/abs/2206.01347) `ACL 2022 Long Papers` (71 cite) (4 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A new large-scale benchmark, MultiHiertt, with QA pairs over Multi Hierarchical Tabular and Textual data is constructed and a novel QA model termed MT2Net is introduced, which first applies facts retrieving to extract relevant supporting facts from both tables and text and then uses a reasoning module to perform symbolic reasoning over retrieved facts.
     </details>


     <details>
          <summary>Abstract</summary>
          Numerical reasoning over hybrid data containing both textual and tabular content (e.g., financial reports) has recently attracted much attention in the NLP community. However, existing question answering (QA) benchmarks over hybrid data only include a single flat table in each document and thus lack examples of multi-step numerical reasoning across multiple hierarchical tables. To facilitate data analytical progress, we construct a new large-scale benchmark, MultiHiertt, with QA pairs over Multi Hierarchical Tabular and Textual data. MultiHiertt is built from a wealth of financial reports and has the following unique characteristics: 1) each document contain multiple tables and longer unstructured texts; 2) most of tables contained are hierarchical; 3) the reasoning process required for each question is more complex and challenging than existing benchmarks; and 4) fine-grained annotations of reasoning processes and supporting facts are provided to reveal complex numerical reasoning. We further introduce a novel QA model termed MT2Net, which first applies facts retrieving to extract relevant supporting facts from both tables and text and then uses a reasoning module to perform symbolic reasoning over retrieved facts. We conduct comprehensive experiments on various baselines. The experimental results show that MultiHiertt presents a strong challenge for existing baselines whose results lag far behind the performance of human experts. The dataset and code are publicly available at https://github.com/psunlpgroup/MultiHiertt.
     </details>

193. **Large Language Models are Zero-Shot Reasoners** [[pdf]](https://arxiv.org/abs/2205.11916v4) `NeurIPS 2022 Poster` (1000 cite) (73 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Experimental results demonstrate that the Zero-shot-CoT, using the same single prompt template, significantly outperforms zero-shot LLM performances on diverse benchmark reasoning tasks including arithmetics, symbolic reasoning, and other logical reasoning tasks, without any hand-crafted few-shot examples.
     </details>


     <details>
          <summary>Abstract</summary>
          Pretrained large language models (LLMs) are widely used in many sub-fields of natural language processing (NLP) and generally known as excellent few-shot learners with task-specific exemplars. Notably, chain of thought (CoT) prompting, a recent technique for eliciting complex multi-step reasoning through step-by-step answer examples, achieved the state-of-the-art performances in arithmetics and symbolic reasoning, difficult system-2 tasks that do not follow the standard scaling laws for LLMs. While these successes are often attributed to LLMs' ability for few-shot learning, we show that LLMs are decent zero-shot reasoners by simply adding ``Let's think step by step'' before each answer. Experimental results demonstrate that our Zero-shot-CoT, using the same single prompt template, significantly outperforms zero-shot LLM performances on diverse benchmark reasoning tasks including arithmetics (MultiArith, GSM8K, AQUA-RAT, SVAMP), symbolic reasoning (Last Letter, Coin Flip), and other logical reasoning tasks (Date Understanding, Tracking Shuffled Objects), without any hand-crafted few-shot examples, e.g. increasing the accuracy on MultiArith from 17.7% to 78.7% and GSM8K from 10.4% to 40.7% with large-scale InstructGPT model (text-davinci-002), as well as similar magnitudes of improvements with another off-the-shelf large model, 540B parameter PaLM. The versatility of this single prompt across very diverse reasoning tasks hints at untapped and understudied fundamental zero-shot capabilities of LLMs, suggesting high-level, multi-task broad cognitive capabilities may be extracted by simple prompting. We hope our work not only serves as the minimal strongest zero-shot baseline for the challenging reasoning benchmarks, but also highlights the importance of carefully exploring and analyzing the enormous zero-shot knowledge hidden inside LLMs before crafting finetuning datasets or few-shot exemplars.
     </details>

194. **HyperTree Proof Search for Neural Theorem Proving** [[pdf]](http://arxiv.org/abs/2205.11491) `NeurIPS 2022` `Lean, MetaMath` (84 cite) (18 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work shows that with HTPS alone, a model trained on annotated proofs manages to prove 65.4% of a held-out set of Metamath theorems, significantly outperforming the previous state of the art of 56.5% by GPT-f.
     </details>


     <details>
          <summary>Abstract</summary>
          We propose an online training procedure for a transformer-based automated theorem prover. Our approach leverages a new search algorithm, HyperTree Proof Search (HTPS), that learns from previous proof searches through online training, allowing it to generalize to domains far from the training distribution. We report detailed ablations of our pipeline’s main components by studying performance on three environments of increasing complexity. In particular, we show that with HTPS alone, a model trained on annotated proofs manages to prove 65.4% of a held-out set of Metamath theorems, significantly outperforming the previous state of the art of 56.5% by GPT-f. Online training on these unproved theorems increases accuracy to 82.6%. With a similar computational budget, we improve the state of the art on the Lean-based miniF2F-curriculum dataset from 31% to 42% proving accuracy.
     </details>

195. **Thor: Wielding Hammers to Integrate Language Models and Automated Theorem Provers** [[pdf]](http://arxiv.org/abs/2205.10893) `NeurIPS 2022` `Isabelle` (65 cite) (15 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Thor is introduced, a framework integrating language models and automated theorem provers to overcome the difficulty of selecting useful premises from a large library to unlock the proof of a given conjecture.
     </details>


     <details>
          <summary>Abstract</summary>
          In theorem proving, the task of selecting useful premises from a large library to unlock the proof of a given conjecture is crucially important. This presents a challenge for all theorem provers, especially the ones based on language models, due to their relative inability to reason over huge volumes of premises in text form. This paper introduces Thor, a framework integrating language models and automated theorem provers to overcome this difficulty. In Thor, a class of methods called hammers that leverage the power of automated theorem provers are used for premise selection, while all other tasks are designated to language models. Thor increases a language model's success rate on the PISA dataset from $39\%$ to $57\%$, while solving $8.2\%$ of problems neither language models nor automated theorem provers are able to solve on their own. Furthermore, with a significantly smaller computational budget, Thor can achieve a success rate on the MiniF2F dataset that is on par with the best existing methods. Thor can be instantiated for the majority of popular interactive theorem provers via a straightforward protocol we provide.
     </details>

196. **Chain of Thought Imitation with Procedure Cloning** [[pdf]](http://arxiv.org/abs/2205.10816) `NeurIPS 2022 Poster` (22 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Through empirical analysis on navigation, simulated robotic manipulation, and game-playing environments, it is shown that imitating the intermediate computations of an expert's behavior enables procedure cloning to learn policies exhibiting significant generalization to unseen environment configurations, including those configurations for which running the expert's procedure directly is infeasible.
     </details>


     <details>
          <summary>Abstract</summary>
          Imitation learning aims to extract high-performance policies from logged demonstrations of expert behavior. It is common to frame imitation learning as a supervised learning problem in which one fits a function approximator to the input-output mapping exhibited by the logged demonstrations (input observations to output actions). While the framing of imitation learning as a supervised input-output learning problem allows for applicability in a wide variety of settings, it is also an overly simplistic view of the problem in situations where the expert demonstrations provide much richer insight into expert behavior. For example, applications such as path navigation, robot manipulation, and strategy games acquire expert demonstrations via planning, search, or some other multi-step algorithm, revealing not just the output action to be imitated but also the procedure for how to determine this action. While these intermediate computations may use tools not available to the agent during inference (e.g., environment simulators), they are nevertheless informative as a way to explain an expert’s mapping of state to actions. To properly leverage expert procedure information without relying on the privileged tools the expert may have used to perform the procedure, we propose procedure cloning, which applies supervised sequence prediction to imitate the complete series of expert computations. This way, procedure cloning learns not only what to do (i.e., the output action), but how and why to do it (i.e., the procedure). Through empirical analysis on navigation, simulated robotic manipulation, and game-playing environments, we show that imitating the intermediate computations of an expert’s behavior enables procedure cloning to learn policies exhibiting significant generalization to unseen environment configurations, including those configurations for which running the expert’s procedure directly is infeasible.
     </details>

197. **STaR: Bootstrapping Reasoning With Reasoning** [[pdf]](http://arxiv.org/abs/2203.14465) `NeurIPS 2022 Poster` (260 cite) (19 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A technique to iteratively leverage a small number of rationale examples and a large dataset without rationales to bootstrap the ability to perform successively more complex reasoning, called STaR, which lets a model improve itself by learning from its own generated reasoning.
     </details>


     <details>
          <summary>Abstract</summary>
          Generating step-by-step "chain-of-thought" rationales improves language model performance on complex reasoning tasks like mathematics or commonsense question-answering. However, inducing language model rationale generation currently requires either constructing massive rationale datasets or sacrificing accuracy by using only few-shot inference. We propose a technique to iteratively leverage a small number of rationale examples and a large dataset without rationales, to bootstrap the ability to perform successively more complex reasoning. This technique, the "Self-Taught Reasoner" (STaR), relies on a simple loop: generate rationales to answer many questions, prompted with a few rationale examples; if the generated answers are wrong, try again to generate a rationale given the correct answer; fine-tune on all the rationales that ultimately yielded correct answers; repeat. We show that STaR significantly improves performance on multiple datasets compared to a model fine-tuned to directly predict final answers, and performs comparably to fine-tuning a 30$\times$ larger state-of-the-art language model on CommensenseQA. Thus, STaR lets a model improve itself by learning from its own generated reasoning.
     </details>

198. **Selection-Inference: Exploiting Large Language Models for Interpretable Logical Reasoning** [[pdf]](http://arxiv.org/abs/2205.09712) `ICLR 2023 Notable-top-5%25` (275 cite) (14 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A Selection-Inference (SI) framework is proposed that exploits pre-trained LLMs as general processing modules, and alternates between selection and inference to generate a series of interpretable, casual reasoning steps leading to the final answer.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have been shown to be capable of impressive few-shot generalisation to new tasks. However, they still tend to perform poorly on multi-step logical reasoning problems. Here we carry out a comprehensive evaluation of LLMs on 46 tasks that probe different aspects of logical reasoning. We show that language models tend to perform fairly well at single step inference or entailment tasks, but struggle to chain together multiple reasoning steps to solve more complex problems. In light of this, we propose a Selection-Inference (SI) framework that exploits pre-trained LLMs as general processing modules, and alternates between selection and inference to generate a series of interpretable, casual reasoning steps leading to the final answer. We show that a 7B parameter LLM used within the SI framework in a 5-shot generalisation setting, with no fine-tuning, yields a performance improvement of over 100% compared to an equivalent vanilla baseline on a suite of 10 logical reasoning tasks. The same model in the same setting even outperforms a significantly larger 280B parameter baseline on the same suite of tasks. Moreover, answers produced by the SI framework are accompanied by a causal natural-language-based reasoning trace, which has important implications for the safety and trustworthiness of the system.
     </details>

199. **Autoformalization with Large Language Models** [[pdf]](https://arxiv.org/abs/2205.12615) `NeurIPS 2022` `Lean` (108 cite) (16 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is shown large language models provide new prospects towards the long-term goal of autoformalization, and the surprising observation that LLMs can correctly translate a significant portion of mathematical competition problems perfectly to formal specifications in Isabelle/HOL.
     </details>


     <details>
          <summary>Abstract</summary>
          Autoformalization is the process of automatically translating from natural language mathematics to formal specifications and proofs. A successful autoformalization system could advance the fields of formal verification, program synthesis, and artificial intelligence.While the long-term goal of autoformalization seemed elusive for a long time, we show large language models provide new prospects towards this goal. We make the surprising observation that LLMs can correctly translate a significant portion ($25.3\%$) of mathematical competition problems perfectly to formal specifications in Isabelle/HOL. We demonstrate the usefulness of this process by improving a previously introduced neural theorem prover via training on these autoformalized theorems. Our methodology results in a new state-of-the-art result on the MiniF2F theorem proving benchmark, improving the proof rate from~$29.6\%$ to~$35.2\%$.
     </details>

200. **NaturalProver: Grounded Mathematical Proof Generation with Language Models** [[pdf]](http://arxiv.org/abs/2205.12910) `NeurIPS 2022` (50 cite) (7 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          NaturalProver is capable of proving some theorems that require short (2-6 step) proofs, and providing next-step suggestions that are rated as correct and useful over 40% of the time, which is to the authors' knowledge the first demonstration of these capabilities using neural language models.
     </details>


     <details>
          <summary>Abstract</summary>
          Theorem proving in natural mathematical language – the mixture of symbolic and natural language used by humans – plays a central role in mathematical advances and education, and tests aspects of reasoning that are core to intelligence. Yet it has remained underexplored with modern generative models. We study large-scale language models on two new generation tasks: suggesting the next step in a mathematical proof, and full proof generation. We develop NaturalProver, a language model that generates proofs by conditioning on background references (e.g. theorems and definitions that are either retrieved or human-provided), and optionally enforces their presence with constrained decoding. On theorems from the NaturalProofs benchmark, NaturalProver improves the quality of next-step suggestions and generated proofs over fine-tuned GPT-3, according to human evaluations from university-level mathematics students. NaturalProver is capable of proving some theorems that require short (2-6 step) proofs, and providing next-step suggestions that are rated as correct and useful over 40% of the time, which is to our knowledge the first demonstration of these capabilities using neural language models.
     </details>

201. **Continual Pre-training of Language Models for Math Problem Understanding with Syntax-Aware Memory Network** [[pdf]](https://aclanthology.org/2022.acl-long.408) `ACL 2022 Long Papers` (18 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A new approach to continually pre-train language models for math problem understanding with syntax-aware memory network called COMUS, which can model the interaction between the token from the text and its semantic-related nodes within the formulas, which is helpful to capture fine-grained semantic correlations between texts and formulas.
     </details>


     <details>
          <summary>Abstract</summary>
          In this paper, we study how to continually pre-train language models for improving the understanding of math problems. Specifically, we focus on solving a fundamental challenge in modeling math problems, how to fuse the semantics of textual description and formulas, which are highly different in essence. To address this issue, we propose a new approach called COMUS to continually pre-train language models for math problem understanding with syntax-aware memory network. In this approach, we first construct the math syntax graph to model the structural semantic information, by combining the parsing trees of the text and formulas, and then design the syntax-aware memory networks to deeply fuse the features from the graph and text. With the help of syntax relations, we can model the interaction between the token from the text and its semantic-related nodes within the formulas, which is helpful to capture fine-grained semantic correlations between texts and formulas. Besides, we devise three continual pre-training tasks to further align and fuse the representations of the text and math syntax graph. Experimental results on four tasks in the math domain demonstrate the effectiveness of our approach. Our code and data are publicly available at the link: bluehttps://github.com/RUCAIBox/COMUS.
     </details>

202. **EPT-X: An Expression-Pointer Transformer model that generates eXplanations for numbers** [[pdf]](https://aclanthology.org/2022.acl-long.305) `ACL 2022` (7 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A neural model EPT-X (Expression-Pointer Transformer with Explanations), which utilizes natural language explanations to solve an algebraic word problem, and releases a novel dataset PEN, which expands the existing datasets by attaching explanations to each number/variable.
     </details>


     <details>
          <summary>Abstract</summary>
          In this paper, we propose a neural model EPT-X (Expression-Pointer Transformer with Explanations), which utilizes natural language explanations to solve an algebraic word problem. To enhance the explainability of the encoding process of a neural model, EPT-X adopts the concepts of plausibility and faithfulness which are drawn from math word problem solving strategies by humans. A plausible explanation is one that includes contextual information for the numbers and variables that appear in a given math word problem. A faithful explanation is one that accurately represents the reasoning process behind the model’s solution equation. The EPT-X model yields an average baseline performance of 69.59% on our PEN dataset and produces explanations with quality that is comparable to human output. The contribution of this work is two-fold. (1) EPT-X model: An explainable neural model that sets a baseline for algebraic word problem solving task, in terms of model’s correctness, plausibility, and faithfulness. (2) New dataset: We release a novel dataset PEN (Problems with Explanations for Numbers), which expands the existing datasets by attaching explanations to each number/variable.
     </details>

203. **NumGLUE: A Suite of Fundamental yet Challenging Mathematical Reasoning Tasks** [[pdf]](http://arxiv.org/abs/2204.05660) `ACL 2022 Long Papers` (91 cite) (11 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          NumGLUE is proposed, a multi-task benchmark that evaluates the performance of AI systems on eight different tasks, that at their core require simple arithmetic understanding and it is shown that this benchmark is far from being solved with neural models including state-of-the-art large-scale language models performing significantly worse than humans.
     </details>


     <details>
          <summary>Abstract</summary>
          Given the ubiquitous nature of numbers in text, reasoning with numbers to perform simple calculations is an important skill of AI systems. While many datasets and models have been developed to this end, state-of-the-art AI systems are brittle; failing to perform the underlying mathematical reasoning when they appear in a slightly different scenario. Drawing inspiration from GLUE that was proposed in the context of natural language understanding, we propose NumGLUE, a multi-task benchmark that evaluates the performance of AI systems on eight different tasks, that at their core require simple arithmetic understanding. We show that this benchmark is far from being solved with neural models including state-of-the-art large-scale language models performing significantly worse than humans (lower by 46.4 %). Further, NumGLUE promotes sharing knowledge across tasks, especially those with limited training data as evidenced by the superior performance (average gain of 3.4 % on each task) when a model is jointly trained on all the tasks as opposed to task-specific modeling. Finally, we hope that NumGLUE will encourage systems that perform robust and general arithmetic reasoning within language, a first step towards being able to perform more complex mathematical reasoning.
     </details>

204. **Formal Mathematics Statement Curriculum Learning** [[pdf]](http://arxiv.org/abs/2202.01344) `ICLR 2023` `Lean` (0 cite) (17 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          We explore the use of expert iteration in the context of language modeling applied to formal mathematics. We show that at same compute budget, expert iteration, by which we mean proof search interleaved with learning, dramatically outperforms proof search only. We also observe that when applied to a collection of formal statements of sufficiently varied difficulty, expert iteration is capable of finding and solving a curriculum of increasingly difficult problems, without the need for associated ground-truth proofs. Finally, by applying this expert iteration to a manually curated set of problem statements, we surpass previous state-of-the-art on the miniF2F benchmark, automatically solving multiple challenging problems drawn from high school olympiads.
     </details>

205. **LILA: A Unified Benchmark for Mathematical Reasoning** [[pdf]](https://arxiv.org/abs/2210.17517) `EMNLP 2022 Main` (103 cite) (18 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is found that multi-tasking leads to significant improvements (average relative improvement of 21.83% F1 score vs. single-task models), indicating the room for improvement in general mathematical reasoning and understanding.
     </details>


     <details>
          <summary>Abstract</summary>
          Mathematical reasoning skills are essential for general-purpose intelligentsystems to perform tasks from grocery shopping to climate modeling. Towards evaluating and improving AI systems in this domain, we proposeLILA, a unified mathematical reasoning benchmark consisting of 23 diversetasks along four dimensions:(i) mathematical abilities e.g., arithmetic, calculus (ii) language format e.g., question-answering, fill-in-the-blanks (iii) language diversity e.g., no language, simple language (iv) external knowledge e.g., commonsense, physics. We construct our benchmark by extending 20 datasets benchmark by collecting task instructions and solutions in the form of Python programs,thereby obtaining explainable solutions in addition to the correct answer. We additionally introduce two evaluation datasets to measure out-of-distribution performance and robustness to language perturbation. Finally, we introduce BHASKARA,a general-purpose mathematical reasoning model trained on LILA. Importantly, we find that multi-tasking leads to significant improvements (average relative improvement of 21.83% F1 score vs. single-task models),while the best performing model only obtains 60.40%,indicating the room for improvement in general mathematical reasoning and understanding.
     </details>

206. **A Causal Framework to Quantify the Robustness of Mathematical Reasoning with Language Models** [[pdf]](https://arxiv.org/abs/2210.12023) `ACL 2023 Long Papers` (47 cite) (3 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a novel framework, which pins down the causal effect of various factors in the input, e.g., the surface form of the problem text, the operands, and math operators on the output solution, and applies this framework on a test bed of math word problems.
     </details>


     <details>
          <summary>Abstract</summary>
          We have recently witnessed a number of impressive results on hard mathematical reasoning problems with language models. At the same time, the robustness of these models has also been called into question; recent works have shown that models can rely on shallow patterns in the problem description when generating a solution. Building on the idea of behavioral testing, we propose a novel framework, which pins down the causal effect of various factors in the input, e.g., the surface form of the problem text, the operands, and math operators on the output solution. By grounding the behavioral analysis in a causal graph describing an intuitive reasoning process, we study the behavior of language models in terms of robustness and sensitivity to direct interventions in the input space. We apply our framework on a test bed of math word problems. Our analysis shows that robustness does not appear to continuously improve as a function of size, but the GPT-3 Davinci models (175B) achieve a dramatic improvement in both robustness and sensitivity compared to all other GPT variants.
     </details>

207. **Solving Math Word Problems via Cooperative Reasoning induced Language Models** [[pdf]](https://arxiv.org/abs/2210.16257) `ACL 2023 Long` (47 cite) (7 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work develops a cooperative reasoning-induced PLM for solving MWPs, called Cooperative Reasoning (CoRe), resulting in a human-like reasoning architecture with system 1 as the generator and system 2 as the verifier, which achieves decent improvement over state-of-the-art methods.
     </details>


     <details>
          <summary>Abstract</summary>
          Large-scale pre-trained language models (PLMs) bring new opportunities to challenging problems, especially those that need high-level intelligence, such as the math word problem (MWPs). However, directly applying existing PLMs to MWPs can fail as the generation process lacks sufficient supervision and thus lacks fast adaptivity as humans. We notice that human reasoning has a dual reasoning framework that consists of an immediate reaction system (system 1) and a delicate reasoning system (system 2), where the entire reasoning is determined by their interaction. This inspires us to develop a cooperative reasoning-induced PLM for solving MWPs, called Cooperative Reasoning (CoRe), resulting in a human-like reasoning architecture with system 1 as the generator and system 2 as the verifier. In our approach, the generator is responsible for generating reasoning paths, and the verifiers are used to supervise the evaluation in order to obtain reliable feedback for the generator. We evaluate our CoRe framework on several mathematical reasoning datasets and achieve decent improvement over state-of-the-art methods, up to 9.6% increase over best baselines.
     </details>

208. **Proving Theorems using Incremental Learning and Hindsight Experience Replay** [[pdf]](https://arxiv.org/abs/2112.10664) `ICML 2022` `Mizar` (11 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A general incremental learning algorithm for training domain specific provers for first-order logic without equality, based only on a basic given-clause algorithm, but using a learned clause-scoring function, and it is shown that provers trained this way can match and sometimes surpass state-of-the-art traditional provers on the TPTP dataset.
     </details>


     <details>
          <summary>Abstract</summary>
          Traditional automated theorem proving systems for first-order logic depend on speed-optimized search and many handcrafted heuristics designed to work over a wide range of domains. Machine learning approaches in the literature either depend on these traditional provers to bootstrap themselves, by leveraging these heuristics, or can struggle due to limited existing proof data. The latter issue can be explained by the lack of a smooth difficulty gradient in theorem proving datasets; large gaps in difficulty between different theorems can make training harder or even impossible. In this paper, we adapt the idea of hindsight experience replay from reinforcement learning to the automated theorem proving domain, so as to use the intermediate data generated during unsuccessful proof attempts. We build a first-order logic prover by disabling all the smart clause-scoring heuristics of the state-of-the-art E prover and replacing them with a clause-scoring neural network learned by using hindsight experience replay in an incremental learning setting. Clauses are represented as graphs and presented to transformer networks with spectral features. We show that provers trained in this way can outperform previous machine learning approaches and compete with the state of the art heuristic-based theorem prover E in its best configuration, on the popular benchmarks MPTP2078, M2k and Mizar40. The proofs generated by our algorithm are also almost always significantly shorter than E’s proofs.
     </details>

209. **A Primer for Neural Arithmetic Logic Modules** [[pdf]](http://jmlr.org/papers/v23/21-0211.html) `NeurIPS 2022 Poster` (6 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Focusing on the shortcomings of the NALU, an in-depth analysis is provided to reason about design choices of recent modules, and a benchmark is created which compares all existing arithmetic NALMs.
     </details>


     <details>
          <summary>Abstract</summary>
          Neural Arithmetic Logic Modules have become a growing area of interest, though remain a niche field. These modules are neural networks which aim to achieve systematic generalisation in learning arithmetic and/or logic operations such as $\{+, -, \times, \div, \leq, \textrm{AND}\}$ while also being interpretable. This paper is the first in discussing the current state of progress of this field, explaining key works, starting with the Neural Arithmetic Logic Unit (NALU). Focusing on the shortcomings of the NALU, we provide an in-depth analysis to reason about design choices of recent modules. A cross-comparison between modules is made on experiment setups and findings, where we highlight inconsistencies in a fundamental experiment causing the inability to directly compare across papers. To alleviate the existing inconsistencies, we create a benchmark which compares all existing arithmetic NALMs. We finish by providing a novel discussion of existing applications for NALU and research directions requiring further exploration.
     </details>

210. **Advancing mathematics by guiding human intuition with AI** [[pdf]](https://www.nature.com/articles/s41586-021-04086-x) `Nature` (338 cite) (4 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A framework through which machine learning can guide mathematicians in discovering new conjectures and theorems is presented and shown to yield mathematical insight on important open problems in different areas of pure mathematics.
     </details>


     <details>
          <summary>Abstract</summary>
          The practice of mathematics involves discovering patterns and using these to formulate and prove conjectures, resulting in theorems. Since the 1960s, mathematicians have used computers to assist in the discovery of patterns and formulation of conjectures1, most famously in the Birch and Swinnerton-Dyer conjecture2, a Millennium Prize Problem3. Here we provide examples of new fundamental results in pure mathematics that have been discovered with the assistance of machine learning—demonstrating a method by which machine learning can aid mathematicians in discovering new conjectures and theorems. We propose a process of using machine learning to discover potential patterns and relations between mathematical objects, understanding them with attribution techniques and using these observations to guide intuition and propose conjectures. We outline this machine-learning-guided framework and demonstrate its successful application to current research questions in distinct areas of pure mathematics, in each case showing how it led to meaningful mathematical contributions on important open problems: a new connection between the algebraic and geometric structure of knots, and a candidate algorithm predicted by the combinatorial invariance conjecture for symmetric groups4. Our work may serve as a model for collaboration between the fields of mathematics and artificial intelligence (AI) that can achieve surprising results by leveraging the respective strengths of mathematicians and machine learning.
     </details>

211. **Measuring Mathematical Problem Solving With the MATH Dataset** [[pdf]](http://arxiv.org/abs/2103.03874) `NeurIPS 2021` (935 cite) (82 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces MATH, a new dataset of 12,500 challenging competition mathematics problems which can be used to teach models to generate answer derivations and explanations and shows that accuracy remains relatively low, even with enormous Transformer models.
     </details>


     <details>
          <summary>Abstract</summary>
          Many intellectual endeavors require mathematical problem solving, but this skill remains beyond the capabilities of computers. To measure this ability in machine learning models, we introduce MATH, a new dataset of 12,500 challenging competition mathematics problems. Each problem in MATH has a full step-by-step solution which can be used to teach models to generate answer derivations and explanations. To facilitate future research and increase accuracy on MATH, we also contribute a large auxiliary pretraining dataset which helps teach models the fundamentals of mathematics. Even though we are able to increase accuracy on MATH, our results show that accuracy remains relatively low, even with enormous Transformer models. Moreover, we find that simply increasing budgets and model parameter counts will be impractical for achieving strong mathematical reasoning if scaling trends continue. While scaling Transformers is automatically solving most other text-based tasks, scaling is not currently solving MATH. To have more traction on mathematical problem solving we will likely need new algorithmic advancements from the broader research community.
     </details>

212. **Math Word Problem Generation with Mathematical Consistency and Problem Context Constraints** [[pdf]](https://aclanthology.org/2021.emnlp-main.484) `EMNLP 2021` (34 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel MWP generation approach is developed that leverages i) pre-trained language models and a context keyword selection model to improve the language quality of generated MWPs and ii) an equation consistency constraint for math equations to improved the mathematical validity of the generatedMWPs.
     </details>


     <details>
          <summary>Abstract</summary>
          We study the problem of generating arithmetic math word problems (MWPs) given a math equation that specifies the mathematical computation and a context that specifies the problem scenario. Existing approaches are prone to generating MWPs that are either mathematically invalid or have unsatisfactory language quality. They also either ignore the context or require manual specification of a problem template, which compromises the diversity of the generated MWPs. In this paper, we develop a novel MWP generation approach that leverages i) pre-trained language models and a context keyword selection model to improve the language quality of generated MWPs and ii) an equation consistency constraint for math equations to improve the mathematical validity of the generated MWPs. Extensive quantitative and qualitative experiments on three real-world MWP datasets demonstrate the superior performance of our approach compared to various baselines.
     </details>

213. **Improving Math Word Problems with Pre-trained Knowledge and Hierarchical Reasoning** [[pdf]](https://aclanthology.org/2021.emnlp-main.272) `EMNLP 2021 Main` (29 cite) (4 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A Reasoning with Pre-trained Knowledge and Hierarchical Structure (RPKHS) network, which contains a pre-trained knowledge encoder and a hierarchical reasoning encoder, which significantly outperforms state-of-the-art approaches on two large-scale commonly-used datasets.
     </details>


     <details>
          <summary>Abstract</summary>
          The recent algorithms for math word problems (MWP) neglect to use outside knowledge not present in the problems. Most of them only capture the word-level relationship and ignore to build hierarchical reasoning like the human being for mining the contextual structure between words and sentences. In this paper, we propose a Reasoning with Pre-trained Knowledge and Hierarchical Structure (RPKHS) network, which contains a pre-trained knowledge encoder and a hierarchical reasoning encoder. Firstly, our pre-trained knowledge encoder aims at reasoning the MWP by using outside knowledge from the pre-trained transformer-based models. Secondly, the hierarchical reasoning encoder is presented for seamlessly integrating the word-level and sentence-level reasoning to bridge the entity and context domain on MWP. Extensive experiments show that our RPKHS significantly outperforms state-of-the-art approaches on two large-scale commonly-used datasets, and boosts performance from 77.4% to 83.9% on Math23K, from 75.5 to 82.2% on Math23K with 5-fold cross-validation and from 83.7% to 89.8% on MAWPS. More extensive ablations are shown to demonstrate the effectiveness and interpretability of our proposed method.
     </details>

214. **Mathematical Word Problem Generation from Commonsense Knowledge Graph and Equations** [[pdf]](https://aclanthology.org/2021.emnlp-main.348) `EMNLP 2021 Main` (23 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          An end-to-end neural model is developed to generate diverse MWPs in real-world scenarios from commonsense knowledge graph and equations and outperforms the SOTA models in terms of both automatic evaluation metrics and human evaluation metrics.
     </details>


     <details>
          <summary>Abstract</summary>
          There is an increasing interest in the use of mathematical word problem (MWP) generation in educational assessment. Different from standard natural question generation, MWP generation needs to maintain the underlying mathematical operations between quantities and variables, while at the same time ensuring the relevance between the output and the given topic. To address above problem, we develop an end-to-end neural model to generate diverse MWPs in real-world scenarios from commonsense knowledge graph and equations. The proposed model (1) learns both representations from edge-enhanced Levi graphs of symbolic equations and commonsense knowledge; (2) automatically fuses equation and commonsense knowledge information via a self-planning module when generating the MWPs. Experiments on an educational gold-standard set and a large-scale generated MWP set show that our approach is superior on the MWP generation task, and it outperforms the SOTA models in terms of both automatic evaluation metrics, i.e., BLEU-4, ROUGE-L, Self-BLEU, and human evaluation metrics, i.e., equation relevance, topic relevance, and language coherence. To encourage reproducible results, we make our code and MWP dataset public available at https://github.com/tal-ai/MaKE_EMNLP2021.
     </details>

215. **GraphMR: Graph Neural Network for Mathematical Reasoning** [[pdf]](https://aclanthology.org/2021.emnlp-main.273) `EMNLP 2021 Main` (9 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A graph-to-sequence neural network GraphMR is proposed, which can effectively learn the hierarchical information of graphs inputs to solve mathematics and speculate answers, and results show that GraphMR outperforms others in hidden information learning and mathematics resolving.
     </details>


     <details>
          <summary>Abstract</summary>
          Mathematical reasoning aims to infer satisfiable solutions based on the given mathematics questions. Previous natural language processing researches have proven the effectiveness of sequence-to-sequence (Seq2Seq) or related variants on mathematics solving. However, few works have been able to explore structural or syntactic information hidden in expressions (e.g., precedence and associativity). This dissertation set out to investigate the usefulness of such untapped information for neural architectures. Firstly, mathematical questions are represented in the format of graphs within syntax analysis. The structured nature of graphs allows them to represent relations of variables or operators while preserving the semantics of the expressions. Having transformed to the new representations, we proposed a graph-to-sequence neural network GraphMR, which can effectively learn the hierarchical information of graphs inputs to solve mathematics and speculate answers. A complete experimental scenario with four classes of mathematical tasks and three Seq2Seq baselines is built to conduct a comprehensive analysis, and results show that GraphMR outperforms others in hidden information learning and mathematics resolving.
     </details>

216. **Mapping probability word problems to executable representations** [[pdf]](https://aclanthology.org/2021.emnlp-main.294) `EMNLP 2021 Main` (8 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper employs and analyse various neural models for answering probability word problems and applies end-to-end models to this task, which bring out the importance of the two-step approach in obtaining correct solutions to probability problems.
     </details>


     <details>
          <summary>Abstract</summary>
          While solving math word problems automatically has received considerable attention in the NLP community, few works have addressed probability word problems specifically. In this paper, we employ and analyse various neural models for answering such word problems. In a two-step approach, the problem text is first mapped to a formal representation in a declarative language using a sequence-to-sequence model, and then the resulting representation is executed using a probabilistic programming system to provide the answer. Our best performing model incorporates general-domain contextualised word representations that were finetuned using transfer learning on another in-domain dataset. We also apply end-to-end models to this task, which bring out the importance of the two-step approach in obtaining correct solutions to probability problems.
     </details>

217. **Subgoal Search For Complex Reasoning Tasks** [[pdf]](http://arxiv.org/abs/2108.11204) `NeurIPS 2021 Poster` (31 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is shown that a simple approach of generating $k$-th step ahead subgoals is surprisingly efficient on three challenging domains: two popular puzzle games, Sokoban and the Rubik's Cube, and an inequality proving benchmark INT.
     </details>


     <details>
          <summary>Abstract</summary>
          Humans excel in solving complex reasoning tasks through a mental process of moving from one idea to a related one. Inspired by this, we propose Subgoal Search (kSubS) method. Its key component is a learned subgoal generator that produces a diversity of subgoals that are both achievable and closer to the solution. Using subgoals reduces the search space and induces a high-level search graph suitable for efficient planning. In this paper, we implement kSubS using a transformer-based subgoal module coupled with the classical best-first search framework. We show that a simple approach of generating $k$-th step ahead subgoals is surprisingly efficient on three challenging domains: two popular puzzle games, Sokoban and the Rubik's Cube, and an inequality proving benchmark INT. kSubS achieves strong results including state-of-the-art on INT within a modest computational budget.
     </details>

218. **miniF2F: a cross-system benchmark for formal Olympiad-level mathematics** [[pdf]](https://arxiv.org/abs/2109.00110) `ICLR 2022` `Lean, Isabelle, HOL Light, MetaMath` (84 cite) (23 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The miniF2F benchmark currently targets Metamath, Lean, Isabelle, and HOL Light and consists of 488 problem statements drawn from the AIME, AMC, and the International Mathematical Olympiad, as well as material from high-school and undergraduate mathematics courses.
     </details>


     <details>
          <summary>Abstract</summary>
          We present $\textsf{miniF2F}$, a dataset of formal Olympiad-level mathematics problems statements intended to provide a unified cross-system benchmark for neural theorem proving. The $\textsf{miniF2F}$ benchmark currently targets Metamath, Lean, Isabelle (partially) and HOL Light (partially) and consists of 488 problem statements drawn from the AIME, AMC, and the International Mathematical Olympiad (IMO), as well as material from high-school and undergraduate mathematics courses. We report baseline results using GPT-f, a neural theorem prover based on GPT-3 and provide an analysis of its performance. We intend for $\textsf{miniF2F}$ to be a community-driven effort and hope that our benchmark will help spur advances in neural theorem proving.
     </details>

219. **Math Word Problem Solving with Explicit Numerical Values** [[pdf]](https://aclanthology.org/2021.acl-long.455) `ACL 2021 Long Papers` (49 cite) (8 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel approach called NumS2T is proposed, which enhances math word problem solving performance by explicitly incorporating numerical values into a sequence-to-tree network.
     </details>


     <details>
          <summary>Abstract</summary>
          In recent years, math word problem solving has received considerable attention and achieved promising results, but previous methods rarely take numerical values into consideration. Most methods treat the numerical values in the problems as number symbols, and ignore the prominent role of the numerical values in solving the problem. In this paper, we propose a novel approach called NumS2T, which enhances math word problem solving performance by explicitly incorporating numerical values into a sequence-to-tree network. In addition, a numerical properties prediction mechanism is used to capture the category and comparison information of numerals and measure their importance in global expressions. Experimental results on the Math23K and APE datasets demonstrate that our model achieves better performance than existing state-of-the-art models.
     </details>

220. **Neural-Symbolic Solver for Math Word Problems with Auxiliary Tasks** [[pdf]](https://aclanthology.org/2021.acl-long.456) `ACL 2021 Long Papers` (49 cite) (7 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The Neural-Symbolic Solver (NS-Solver) is proposed to explicitly and seamlessly incorporate different levels of symbolic constraints by auxiliary tasks to explicitly incorporate essential math symbolic constraints, leading to unexplainable and unreasonable predictions.
     </details>


     <details>
          <summary>Abstract</summary>
          Previous math word problem solvers following the encoder-decoder paradigm fail to explicitly incorporate essential math symbolic constraints, leading to unexplainable and unreasonable predictions. Herein, we propose Neural-Symbolic Solver (NS-Solver) to explicitly and seamlessly incorporate different levels of symbolic constraints by auxiliary tasks. Our NS-Solver consists of a problem reader to encode problems, a programmer to generate symbolic equations, and a symbolic executor to obtain answers. Along with target expression supervision, our solver is also optimized via 4 new auxiliary objectives to enforce different symbolic reasoning: a) self-supervised number prediction task predicting both number quantity and number locations; b) commonsense constant prediction task predicting what prior knowledge (e.g. how many legs a chicken has) is required; c) program consistency checker computing the semantic loss between predicted equation and target equation to ensure reasonable equation mapping; d) duality exploiting task exploiting the quasi-duality between symbolic equation generation and problem’s part-of-speech generation to enhance the understanding ability of a solver. Besides, to provide a more realistic and challenging benchmark for developing a universal and scalable solver, we also construct a new largescale MWP benchmark CM17K consisting of 4 kinds of MWPs (arithmetic, one-unknown linear, one-unknown non-linear, equation set) with more than 17K samples. Extensive experiments on Math23K and our CM17k demonstrate the superiority of our NS-Solver compared to state-of-the-art methods.
     </details>

221. **Inter-GPS: Interpretable Geometry Problem Solving with Formal Language and Symbolic Reasoning** [[pdf]](http://arxiv.org/abs/2105.04165) `ACL 2021 Long Papers` (126 cite) (12 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work constructs a new largescale benchmark, Geometry3K, consisting of 3,002 geometry problems with dense annotation in formal language, and proposes a novel geometry solving approach with formal language and symbolic reasoning, called Interpretable Geometry Problem Solver (InterGPS).
     </details>


     <details>
          <summary>Abstract</summary>
          Geometry problem solving has attracted much attention in the NLP community recently. The task is challenging as it requires abstract problem understanding and symbolic reasoning with axiomatic knowledge. However, current datasets are either small in scale or not publicly available. Thus, we construct a new large-scale benchmark, Geometry3K, consisting of 3,002 geometry problems with dense annotation in formal language. We further propose a novel geometry solving approach with formal language and symbolic reasoning, called Interpretable Geometry Problem Solver (Inter-GPS). Inter-GPS first parses the problem text and diagram into formal language automatically via rule-based text parsing and neural object detecting, respectively. Unlike implicit learning in existing methods, Inter-GPS incorporates theorem knowledge as conditional rules and performs symbolic reasoning step by step. Also, a theorem predictor is designed to infer the theorem application sequence fed to the symbolic solver for the more efficient and reasonable searching path. Extensive experiments on the Geometry3K and GEOS datasets demonstrate that Inter-GPS achieves significant improvements over existing methods. The project with code and data is available at https://lupantech.github.io/inter-gps.
     </details>

222. **A Diverse Corpus for Evaluating and Developing English Math Word Problem Solvers** [[pdf]](http://arxiv.org/abs/2106.15772) `ACL 2020` (253 cite) (37 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A metric to measure the lexicon usage diversity of a given MWP corpus is proposed, and it is demonstrated that ASDiv (Academia Sinica Diverse MWP Dataset) is more diverse than existing corpora.
     </details>


     <details>
          <summary>Abstract</summary>
          We present ASDiv (Academia Sinica Diverse MWP Dataset), a diverse (in terms of both language patterns and problem types) English math word problem (MWP) corpus for evaluating the capability of various MWP solvers. Existing MWP corpora for studying AI progress remain limited either in language usage patterns or in problem types. We thus present a new English MWP corpus with 2,305 MWPs that cover more text patterns and most problem types taught in elementary school. Each MWP is annotated with its problem type and grade level (for indicating the level of difficulty). Furthermore, we propose a metric to measure the lexicon usage diversity of a given MWP corpus, and demonstrate that ASDiv is more diverse than existing corpora. Experiments show that our proposed corpus reflects the true capability of MWP solvers more faithfully.
     </details>

223. **TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance** [[pdf]](http://arxiv.org/abs/2105.07624) `ACL 2021 Long Papers` (200 cite) (10 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work extracts samples from real financial reports to build a new large-scale QA dataset containing both Tabular And Textual data, named TAT-QA, where numerical reasoning is usually required to infer the answer, such as addition, subtraction, multiplication, division, counting, comparison/sorting, and the compositions.
     </details>


     <details>
          <summary>Abstract</summary>
          Hybrid data combining both tabular and textual content (e.g., financial reports) are quite pervasive in the real world. However, Question Answering (QA) over such hybrid data is largely neglected in existing research. In this work, we extract samples from real financial reports to build a new large-scale QA dataset containing both Tabular And Textual data, named TAT-QA, where numerical reasoning is usually required to infer the answer, such as addition, subtraction, multiplication, division, counting, comparison/sorting, and the compositions. We further propose a novel QA model termed TAGOP, which is capable of reasoning over both tables and text. It adopts sequence tagging to extract relevant cells from the table along with relevant spans from the text to infer their semantics, and then applies symbolic reasoning over them with a set of aggregation operators to arrive at the final answer. TAGOP achieves 58.0% inF1, which is an 11.1% absolute increase over the previous best baseline model, according to our experiments on TAT-QA. But this result still lags far behind performance of expert human, i.e.90.8% in F1. It is demonstrated that our TAT-QA is very challenging and can serve as a benchmark for training and testing powerful QA models that address hybrid form data.
     </details>

224. **Are NLP Models really able to Solve Simple Math Word Problems?** [[pdf]](http://arxiv.org/abs/2103.07191) `NAACL 2021 Main` (574 cite) (63 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is shown that MWP solvers that do not have access to the question asked in the MWP can still solve a large fraction of MWPs, and models that treat MWPs as bag-of-words can also achieve surprisingly high accuracy.
     </details>


     <details>
          <summary>Abstract</summary>
          The problem of designing NLP solvers for math word problems (MWP) has seen sustained research activity and steady gains in the test accuracy. Since existing solvers achieve high performance on the benchmark datasets for elementary level MWPs containing one-unknown arithmetic word problems, such problems are often considered “solved” with the bulk of research attention moving to more complex MWPs. In this paper, we restrict our attention to English MWPs taught in grades four and lower. We provide strong evidence that the existing MWP solvers rely on shallow heuristics to achieve high performance on the benchmark datasets. To this end, we show that MWP solvers that do not have access to the question asked in the MWP can still solve a large fraction of MWPs. Similarly, models that treat MWPs as bag-of-words can also achieve surprisingly high accuracy. Further, we introduce a challenge dataset, SVAMP, created by applying carefully chosen variations over examples sampled from existing datasets. The best accuracy achieved by state-of-the-art models is substantially lower on SVAMP, thus showing that much remains to be done even for the simplest of the MWPs.
     </details>

225. **Proof Artifact Co-Training for Theorem Proving with Language Models** [[pdf]](https://arxiv.org/abs/2102.06203) `ICLR 2022` `Lean` (98 cite) (27 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          PACT is proposed, a general methodology for extracting abundant self-supervised data from kernel-level proof terms for co-training alongside the usual tactic prediction objective and applied to Lean, an interactive proof assistant which hosts some of the most sophisticated formalized mathematics to date.
     </details>


     <details>
          <summary>Abstract</summary>
          Labeled data for imitation learning of theorem proving in large libraries of formalized mathematics is scarce as such libraries require years of concentrated effort by human specialists to be built. This is particularly challenging when applying large Transformer language models to tactic prediction, because the scaling of performance with respect to model size is quickly disrupted in the data-scarce, easily-overfitted regime. We propose PACT (Proof Artifact Co-Training), a general methodology for extracting abundant self-supervised data from kernel-level proof terms for joint training alongside the usual tactic prediction objective. We apply this methodology to Lean,an interactive proof assistant which hosts some of the most sophisticated formalized mathematics to date. We instrument Lean with a neural theorem prover driven by a Transformer language model and show that PACT improves theorem proving success rate on a held-out suite of test theorems from 32% to 48%.
     </details>

226. **Measuring Massive Multitask Language Understanding** [[pdf]](http://arxiv.org/abs/2009.03300) `ICLR 2021` (1000 cite) (22 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          While most recent models have near random-chance accuracy, the very largest GPT-3 model improves over random chance by almost 20 percentage points on average, however, on every one of the 57 tasks, the best models still need substantial improvements before they can reach expert-level accuracy.
     </details>


     <details>
          <summary>Abstract</summary>
          We propose a new test to measure a text model's multitask accuracy. The test covers 57 tasks including elementary mathematics, US history, computer science, law, and more. To attain high accuracy on this test, models must possess extensive world knowledge and problem solving ability. We find that while most recent models have near random-chance accuracy, the very largest GPT-3 model improves over random chance by almost 20 percentage points on average. However, on every one of the 57 tasks, the best models still need substantial improvements before they can reach expert-level accuracy. Models also have lopsided performance and frequently do not know when they are wrong. Worse, they still have near-random accuracy on some socially important subjects such as morality and law. By comprehensively evaluating the breadth and depth of a model's academic and professional understanding, our test can be used to analyze models across many tasks and to identify important shortcomings.
     </details>

227. **NaturalProofs: Mathematical Theorem Proving in Natural Language** [[pdf]](https://arxiv.org/abs/2104.01112) `NeurIPS 2021` (51 cite) (16 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          NaturalProofs is developed, a multi-domain corpus of mathematical statements and their proofs, written in natural mathematical language that unifies broad coverage, deep coverage, and low-resource mathematical sources, allowing for evaluating both in-distribution and zero-shot generalization.
     </details>


     <details>
          <summary>Abstract</summary>
          Understanding and creating mathematics using natural mathematical language - the mixture of symbolic and natural language used by humans - is a challenging and important problem for driving progress in machine learning. As a step in this direction, we develop NaturalProofs, a multi-domain corpus of mathematical statements and their proofs, written in natural mathematical language. NaturalProofs unifies broad coverage, deep coverage, and low-resource mathematical sources, allowing for evaluating both in-distribution and zero-shot generalization. Using NaturalProofs, we benchmark strong neural methods on mathematical reference retrieval and generation tasks which test a system's ability to determine key results that appear in a proof. Large-scale sequence models show promise compared to classical information retrieval methods, yet their performance and out-of-domain generalization leave substantial room for improvement. NaturalProofs opens many avenues for research on challenging mathematical tasks.
     </details>

228. **LIME: Learning Inductive Bias for Primitives of Mathematical Reasoning** [[pdf]](https://arxiv.org/abs/2101.06223) `ICML 2021` `Isabelle, MetaMath, HOL Light, Lean` (47 cite) (8 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A new pre-training methodology called LIME (Learning Inductive bias for Mathematical rEasoning).
     </details>


     <details>
          <summary>Abstract</summary>
          While designing inductive bias in neural architectures has been widely studied, we hypothesize that transformer networks are flexible enough to learn inductive bias from suitable generic tasks. Here, we replace architecture engineering by encoding inductive bias in the form of datasets. Inspired by Peirce's view that deduction, induction, and abduction are the primitives of reasoning, we design three synthetic tasks that are intended to require the model to have these three abilities. We specifically design these tasks to be synthetic and devoid of mathematical knowledge to ensure that only the fundamental reasoning biases can be learned from these tasks. This defines a new pre-training methodology called "LIME" (Learning Inductive bias for Mathematical rEasoning). Models trained with LIME significantly outperform vanilla transformers on four very different large mathematical reasoning benchmarks. Unlike dominating the computation cost as traditional pre-training approaches, LIME requires only a small fraction of the computation cost of the typical downstream task. The code for generating LIME tasks is available at https://github.com/tonywu95/LIME.
     </details>

229. **TacticZero: Learning to Prove Theorems from Scratch with Deep Reinforcement Learning** [[pdf]](https://arxiv.org/abs/2102.09756) `NeurIPS 2021` `HOL 4` (34 cite) (7 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel approach to interactive theorem-proving (ITP) using deep reinforcement learning in which each state represents a set of potential derivation paths, and introduces a novel backtracking mechanism which enables the agent to efficiently discard dead-end derivations and restart from promising alternatives.
     </details>


     <details>
          <summary>Abstract</summary>
          We propose a novel approach to interactive theorem-proving (ITP) using deep reinforcement learning. The proposed framework is able to learn proof search strategies as well as tactic and arguments prediction in an end-to-end manner. We formulate the process of ITP as a Markov decision process (MDP) in which each state represents a set of potential derivation paths. This structure allows us to introduce a novel backtracking mechanism which enables the agent to efficiently discard (predicted) dead-end derivations and restart the derivation from promising alternatives. We implement the framework in the HOL theorem prover. Experimental results show that the framework using learned search strategies outperforms existing automated theorem provers (i.e., hammers) available in HOL when evaluated on unseen problems. We further elaborate the role of key components of the framework using ablation studies.
     </details>

230. **Contrastive Reinforcement Learning of Symbolic Reasoning Domains** [[pdf]](https://arxiv.org/abs/2106.09146) `NeurIPS 2021 Poster` (14 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel learning algorithm, Contrastive Policy Learning (ConPoLe) that explicitly optimizes the InfoNCE loss, which lower bounds the mutual information between the current state and next states that continue on a path to the solution, successfully solves all four domains.
     </details>


     <details>
          <summary>Abstract</summary>
          Abstract symbolic reasoning, as required in domains such as mathematics and logic, is a key component of human intelligence. Solvers for these domains have important applications, especially to computer-assisted education. But learning to solve symbolic problems is challenging for machine learning algorithms. Existing models either learn from human solutions or use hand-engineered features, making them expensive to apply in new domains. In this paper, we instead consider symbolic domains as simple environments where states and actions are given as unstructured text, and binary rewards indicate whether a problem is solved. This flexible setup makes it easy to specify new domains, but search and planning become challenging. We introduce five environments inspired by the Mathematics Common Core Curriculum, and observe that existing Reinforcement Learning baselines perform poorly. We then present a novel learning algorithm, Contrastive Policy Learning (ConPoLe) that explicitly optimizes the InfoNCE loss, which lower bounds the mutual information between the current state and next states that continue on a path to the solution. ConPoLe successfully solves all four domains. Moreover, problem representations learned by ConPoLe enable accurate prediction of the categories of problems in a real mathematics curriculum. Our results suggest new directions for reinforcement learning in symbolic domains, as well as applications to mathematics education.
     </details>

231. **REFACTOR: Learning to Extract Theorems from Proofs** [[pdf]](https://arxiv.org/abs/2402.17032) `ICLR 2024` `MetaMath` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper shows on a set of unseen proofs, REFACTOR is able to extract 19.6% of the theorems that humans would use to write the proofs, and demonstrates that the prover trained on the new-theorem refactored dataset proves more test theorems and outperforms state-of-the-art baselines by frequently leveraging a diverse set of newly extracted theorems.
     </details>


     <details>
          <summary>Abstract</summary>
          Human mathematicians are often good at recognizing modular and reusable theorems that make complex mathematical results within reach. In this paper, we propose a novel method called theoREm-from-prooF extrACTOR (REFACTOR) for training neural networks to mimic this ability in formal mathematical theorem proving. We show on a set of unseen proofs, REFACTOR is able to extract 19.6\% of the theorems that humans would use to write the proofs. When applying the model to the existing Metamath library, REFACTOR extracted 16 new theorems. With newly extracted theorems, we show that the existing proofs in the MetaMath database can be refactored. The new theorems are used very frequently after refactoring, with an average usage of 733.5 times, and help shorten the proof lengths. Lastly, we demonstrate that the prover trained on the new-theorem refactored dataset proves more test theorems and outperforms state-of-the-art baselines by frequently leveraging a diverse set of newly extracted theorems.
     </details>

232. **A Knowledge-Aware Sequence-to-Tree Network for Math Word Problem Solving** [[pdf]](https://aclanthology.org/2020.emnlp-main.579) `EMNLP 2020 Main` (62 cite) (10 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Experimental results revealed that the KA-S2T model can achieve better performance than previously reported best results and use a tree-structured decoder with a state aggregation mechanism to capture the long-distance dependency and global expression information.
     </details>


     <details>
          <summary>Abstract</summary>
          With the advancements in natural language processing tasks, math word problem solving has received increasing attention. Previous methods have achieved promising results but ignore background common-sense knowledge not directly provided by the problem. In addition, during generation, they focus on local features while neglecting global information. To incorporate external knowledge and global expression information, we propose a novel knowledge-aware sequence-to-tree (KA-S2T) network in which the entities in the problem sequences and their categories are modeled as an entity graph. Based on this entity graph, a graph attention network is used to capture knowledge-aware problem representations. Further, we use a tree-structured decoder with a state aggregation mechanism to capture the long-distance dependency and global expression information. Experimental results on the Math23K dataset revealed that the KA-S2T model can achieve better performance than previously reported best results.
     </details>

233. **Point to the Expression: Solving Algebraic Word Problems using the Expression-Pointer Transformer Model** [[pdf]](https://aclanthology.org/2020.emnlp-main.308) `EMNLP 2020 Main` (34 cite) (7 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A pure neural model, EPT, is proposed, which can address the expression fragmentation and the operand-context separation, and yields comparable performance to existing models using hand-crafted features, and achieves better performance than existing pure neural models by at most 40%.
     </details>


     <details>
          <summary>Abstract</summary>
          Solving algebraic word problems has recently emerged as an important natural language processing task. To solve algebraic word problems, recent studies suggested neural models that generate solution equations by using ‘Op (operator/operand)’ tokens as a unit of input/output. However, such a neural model suffered two issues: expression fragmentation and operand-context separation. To address each of these two issues, we propose a pure neural model, Expression-Pointer Transformer (EPT), which uses (1) ‘Expression’ token and (2) operand-context pointers when generating solution equations. The performance of the EPT model is tested on three datasets: ALG514, DRAW-1K, and MAWPS. Compared to the state-of-the-art (SoTA) models, the EPT model achieved a comparable performance accuracy in each of the three datasets; 81.3% on ALG514, 59.5% on DRAW-1K, and 84.5% on MAWPS. The contribution of this paper is two-fold; (1) We propose a pure neural model, EPT, which can address the expression fragmentation and the operand-context separation. (2) The fully automatic EPT model, which does not use hand-crafted features, yields comparable performance to existing models using hand-crafted features, and achieves better performance than existing pure neural models by at most 40%.
     </details>

234. **Semantically-Aligned Universal Tree-Structured Solver for Math Word Problems** [[pdf]](http://arxiv.org/abs/2010.06823) `EMNLP 2020 Main` (58 cite) (12 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A simple but efficient method to make the first attempt to represent the equations of various MWPs uniformly, and a semantically-aligned universal tree-structured solver (SAU-Solver) based on an encoder-decoder framework is proposed to resolve multiple types of MWPs in a unified model, benefiting from the UET representation.
     </details>


     <details>
          <summary>Abstract</summary>
          A practical automatic textual math word problems (MWPs) solver should be able to solve various textual MWPs while most existing works only focused on one-unknown linear MWPs. Herein, we propose a simple but efficient method called Universal Expression Tree (UET) to make the first attempt to represent the equations of various MWPs uniformly. Then a semantically-aligned universal tree-structured solver (SAU-Solver) based on an encoder-decoder framework is proposed to resolve multiple types of MWPs in a unified model, benefiting from our UET representation. Our SAU-Solver generates a universal expression tree explicitly by deciding which symbol to generate according to the generated symbols’ semantic meanings like human solving MWPs. Besides, our SAU-Solver also includes a novel subtree-level semanticallyaligned regularization to further enforce the semantic constraints and rationality of the generated expression tree by aligning with the contextual information. Finally, to validate the universality of our solver and extend the research boundary of MWPs, we introduce a new challenging Hybrid Math Word Problems dataset (HMWP), consisting of three types of MWPs. Experimental results on several MWPs datasets show that our model can solve universal types of MWPs and outperforms several state-of-the-art models.
     </details>

235. **Graph-to-Tree Learning for Solving Math Word Problems** [[pdf]](https://aclanthology.org/2020.acl-main.362) `ACL 2020 Main` (166 cite) (20 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Graph2Tree is proposed, a novel deep learning architecture that combines the merits of the graph-based encoder and tree-based decoder to generate better solution expressions for math word problem (MWP) solution expressions.
     </details>


     <details>
          <summary>Abstract</summary>
          While the recent tree-based neural models have demonstrated promising results in generating solution expression for the math word problem (MWP), most of these models do not capture the relationships and order information among the quantities well. This results in poor quantity representations and incorrect solution expressions. In this paper, we propose Graph2Tree, a novel deep learning architecture that combines the merits of the graph-based encoder and tree-based decoder to generate better solution expressions. Included in our Graph2Tree framework are two graphs, namely the Quantity Cell Graph and Quantity Comparison Graph, which are designed to address limitations of existing methods by effectively representing the relationships and order information among the quantities in MWPs. We conduct extensive experiments on two available datasets. Our experiment results show that Graph2Tree outperforms the state-of-the-art baselines on two benchmark datasets significantly. We also discuss case studies and empirically examine Graph2Tree’s effectiveness in translating the MWP text into solution expressions.
     </details>

236. **INT: An Inequality Benchmark for Evaluating Generalization in Theorem Proving** [[pdf]](https://arxiv.org/abs/2007.02924) `ICLR 2021` (49 cite) (17 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduces INT, an INequality Theorem proving benchmark, specifically designed to test agents' generalization ability, and evaluates the same agents augmented with Monte Carlo Tree Search at test time, and shows that MCTS can help to prove new theorems.
     </details>


     <details>
          <summary>Abstract</summary>
          In learning-assisted theorem proving, one of the most critical challenges is to generalize to theorems unlike those seen at training time. In this paper, we introduce INT, an INequality Theorem proving benchmark designed to test agents’ generalization ability. INT is based on a theorem generator, which provides theoretically infinite data and allows us to measure 6 different types of generalization, each reflecting a distinct challenge, characteristic of automated theorem proving. In addition, provides a fast theorem proving environment with sequence-based and graph-based interfaces, conducive to performing learning-based research. We introduce base-lines with architectures including transformers and graph neural networks (GNNs)for INT. Using INT, we find that transformer-based agents achieve stronger test performance for most of the generalization tasks, despite having much larger out-of-distribution generalization gaps than GNNs. We further find that the addition of Monte Carlo Tree Search (MCTS) at test time helps to prove new theorems.
     </details>

237. **Premise Selection in Natural Language Mathematical Texts** [[pdf]](https://aclanthology.org/2020.acl-main.657) `ACL 2020` (23 cite) (5 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper proposes an approach to solve the natural language premise selection task as a link prediction problem, using Deep Convolutional Graph Neural Networks and shows that a graph structure can provide higher F1-score, especially when considering multi-hop premise selection.
     </details>


     <details>
          <summary>Abstract</summary>
          The discovery of supporting evidence for addressing complex mathematical problems is a semantically challenging task, which is still unexplored in the field of natural language processing for mathematical text. The natural language premise selection task consists in using conjectures written in both natural language and mathematical formulae to recommend premises that most likely will be useful to prove a particular statement. We propose an approach to solve this task as a link prediction problem, using Deep Convolutional Graph Neural Networks. This paper also analyses how different baselines perform in this task and shows that a graph structure can provide higher F1-score, especially when considering multi-hop premise selection.
     </details>

238. **Injecting Numerical Reasoning Skills into Language Models** [[pdf]](http://arxiv.org/abs/2004.04487) `ACL 2020 Main` (205 cite) (10 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work shows that numerical reasoning is amenable to automatic data generation, and thus one can inject this skill into pre-trained LMs, by generating large amounts of data, and training in a multi-task setup.
     </details>


     <details>
          <summary>Abstract</summary>
          Large pre-trained language models (LMs) are known to encode substantial amounts of linguistic information. However, high-level reasoning skills, such as numerical reasoning, are difficult to learn from a language-modeling objective only. Consequently, existing models for numerical reasoning have used specialized architectures with limited flexibility. In this work, we show that numerical reasoning is amenable to automatic data generation, and thus one can inject this skill into pre-trained LMs, by generating large amounts of data, and training in a multi-task setup. We show that pre-training our model, GenBERT, on this data, dramatically improves performance on DROP (49.3 –> 72.3 F1), reaching performance that matches state-of-the-art models of comparable size, while using a simple and general-purpose encoder-decoder architecture. Moreover, GenBERT generalizes well to math word problem datasets, while maintaining high performance on standard RC tasks. Our approach provides a general recipe for injecting skills into large pre-trained LMs, whenever the skill is amenable to automatic data augmentation.
     </details>

239. **Mathematical Reasoning via Self-supervised Skip-tree Training** [[pdf]](https://arxiv.org/abs/2006.04757) `ICLR 2021` (53 cite) (15 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is found that models trained on the skip-tree task show surprisingly strong mathematical reasoning abilities, and outperform modelstrained on standard skip-sequence tasks.
     </details>


     <details>
          <summary>Abstract</summary>
          We demonstrate that self-supervised language modeling applied to mathematical formulas enables logical reasoning. To measure the logical reasoning abilities of language models, we formulate several evaluation (downstream) tasks, such as inferring types, suggesting missing assumptions and completing equalities. For training language models for formal mathematics, we propose a novel skip-tree task. We find that models trained on the skip-tree task show surprisingly strong mathematical reasoning abilities, and outperform models trained on standard skip-sequence tasks. We also analyze the models' ability to formulate new conjectures by measuring how often the predictions are provable and useful in other proofs.
     </details>

240. **IsarStep: a Benchmark for High-level Mathematical Reasoning** [[pdf]](https://arxiv.org/abs/2006.09265) `ICLR 2021` `Isabelle` (52 cite) (17 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A benchmark for high-level mathematical reasoning is presented and the reasoning capabilities of neural sequence-to-sequence models are studied and a hierarchical transformer is designed that outperforms the transformer baseline.
     </details>


     <details>
          <summary>Abstract</summary>
          A well-defined benchmark is essential for measuring and accelerating research progress of machine learning models. In this paper, we present a benchmark for high-level mathematical reasoning and study the reasoning capabilities of neural sequence-to-sequence models. We build a non-synthetic dataset from the largest repository of proofs written by human experts in a theorem prover. The dataset has a broad coverage of undergraduate and research-level mathematical and computer science theorems. In our defined task, a model is required to fill in a missing intermediate proposition given surrounding proofs. This task provides a starting point for the long-term goal of having machines generate human-readable proofs automatically. Our experiments and analysis reveal that while the task is challenging, neural models can capture non-trivial mathematical reasoning. We further design a hierarchical transformer that outperforms the transformer baseline.
     </details>

241. **Learning to Prove Theorems by Learning to Generate Theorems** [[pdf]](https://arxiv.org/abs/2002.07019) `NeurIPS 2020` `Holophrasm` (41 cite) (13 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes to learn a neural generator that automatically synthesizes theorems and proofs for the purpose of training a theorem prover, and demonstrates that synthetic data from this approach improves the theorem provers and advances the state of the art of automated theorem proving in Metamath.
     </details>


     <details>
          <summary>Abstract</summary>
          We consider the task of automated theorem proving, a key AI task. Deep learning has shown promise for training theorem provers, but there are limited human-written theorems and proofs available for supervised learning. To address this limitation, we propose to learn a neural generator that automatically synthesizes theorems and proofs for the purpose of training a theorem prover. Experiments on real-world tasks demonstrate that synthetic data from our approach improves the theorem prover and advances the state of the art of automated theorem proving in Metamath. Code is available at https://github.com/princeton-vl/MetaGen.
     </details>

242. **Learning advanced mathematical computations from examples** [[pdf]](https://arxiv.org/abs/2006.06462) `ICLR 2021 Poster` (22 cite) (4 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work achieves near perfect prediction of qualitative characteristics, and good approximations of numerical features of the system, demonstrating that neural networks can learn to perform complex computations, grounded in advanced theory, from examples, without built-in mathematical knowledge.
     </details>


     <details>
          <summary>Abstract</summary>
          Using transformers over large generated datasets, we train models to learn mathematical properties of differential systems, such as local stability, behavior at infinity and controllability. We achieve near perfect prediction of qualitative characteristics, and good approximations of numerical features of the system. This demonstrates that neural networks can learn to perform complex computations, grounded in advanced theory, from examples, without built-in mathematical knowledge.
     </details>

243. **Deep Learning for Symbolic Mathematics** [[pdf]](http://arxiv.org/abs/1912.01412) `ICLR 2020 Spotlight` (357 cite) (23 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is shown that neural networks can be surprisingly good at more elaborated tasks in mathematics, such as symbolic integration and solving differential equations, and a syntax for representing these mathematical problems, and methods for generating large datasets that can be used to train sequence-to-sequence models.
     </details>


     <details>
          <summary>Abstract</summary>
          Neural networks have a reputation for being better at solving statistical or approximate problems than at performing calculations or working with symbolic data. In this paper, we show that they can be surprisingly good at more elaborated tasks in mathematics, such as symbolic integration and solving differential equations. We propose a syntax for representing mathematical problems, and methods for generating large datasets that can be used to train sequence-to-sequence models. We achieve results that outperform commercial Computer Algebra Systems such as Matlab or Mathematica.
     </details>

244. **Mathematical Reasoning in Latent Space** [[pdf]](http://arxiv.org/abs/1909.11851) `ICLR 2020` `HOL Light` (33 cite) (3 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The experiments show that graph neural networks can make non-trivial predictions about the rewrite-success of statements, even when they propagate predicted latent representations for several steps, a strong indicator for the feasibility of deduction in latent space in general.
     </details>


     <details>
          <summary>Abstract</summary>
          We design and conduct a simple experiment to study whether neural networks can perform several steps of approximate reasoning in a fixed dimensional latent space. The set of rewrites (i.e. transformations) that can be successfully performed on a statement represents essential semantic features of the statement. We can compress this information by embedding the formula in a vector space, such that the vector associated with a statement can be used to predict whether a statement can be rewritten by other theorems. Predicting the embedding of a formula generated by some rewrite rule is naturally viewed as approximate reasoning in the latent space. In order to measure the effectiveness of this reasoning, we perform approximate deduction sequences in the latent space and use the resulting embedding to inform the semantic features of the corresponding formal statement (which is obtained by performing the corresponding rewrite sequence using real formulas). Our experiments show that graph neural networks can make non-trivial predictions about the rewrite-success of statements, even when they propagate predicted latent representations for several steps. Since our corpus of mathematical formulas includes a wide variety of mathematical disciplines, this experiment is a strong indicator for the feasibility of deduction in latent space in general.
     </details>

245. **MathQA: Towards Interpretable Math Word Problem Solving with Operation-Based Formalisms** [[pdf]](http://arxiv.org/abs/1905.13319) `NAACL 2019 Main` (393 cite) (42 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A large-scale dataset of math word problems and an interpretable neural math problem solver by learning to map problems to their operation programs and a new representation language to model operation programs corresponding to each math problem that aim to improve both the performance and the interpretability of the learned models.
     </details>


     <details>
          <summary>Abstract</summary>
          We introduce a large-scale dataset of math word problems and an interpretable neural math problem solver by learning to map problems to their operation programs. Due to annotation challenges, current datasets in this domain have been either relatively small in scale or did not offer precise operational annotations over diverse problem types. We introduce a new representation language to model operation programs corresponding to each math problem that aim to improve both the performance and the interpretability of the learned models. Using this representation language, we significantly enhance the AQUA-RAT dataset with fully-specified operational programs. We additionally introduce a neural sequence-to-program model with automatic problem categorization. Our experiments show improvements over competitive baselines in our dataset as well as the AQUA-RAT dataset. The results are still lower than human performance indicating that the dataset poses new challenges for future research. Our dataset is available at https://math-qa.github.io/math-QA/
     </details>

246. **Analysing Mathematical Reasoning Abilities of Neural Models** [[pdf]](https://www.semanticscholar.org/paper/afed6dc6900d3b37e528b9086661bba583d60bf6) `ICLR 2019` (360 cite) (29 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper conducts a comprehensive analysis of models from two broad classes of the most powerful sequence-to-sequence architectures and finds notable differences in their ability to resolve mathematical problems and generalize their knowledge.
     </details>


     <details>
          <summary>Abstract</summary>
          N/A
     </details>

247. **Learning to Prove Theorems via Interacting with Proof Assistants** [[pdf]](https://www.semanticscholar.org/paper/a596f03145285cd05a6ca57a4e25418b23b24976) `ICML 2019` `Coq` (117 cite) (22 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          ASTactic, a deep learning-based model that generates tactics as programs in the form of abstract syntax trees (ASTs) can generate effective tactics and can be used to prove new theorems not previously provable by automated methods.
     </details>


     <details>
          <summary>Abstract</summary>
          N/A
     </details>

248. **NumNet: Machine Reading Comprehension with Numerical Reasoning** [[pdf]](https://www.aclweb.org/anthology/D19-1251) `EMNLP 2019 Main` (112 cite) (8 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A numerical MRC model named as NumNet is proposed, which utilizes a numerically-aware graph neural network to consider the comparing information and performs numerical reasoning over numbers in the question and passage, outperforming all existing machine reading comprehension models by considering the numerical relations among numbers.
     </details>


     <details>
          <summary>Abstract</summary>
          Numerical reasoning, such as addition, subtraction, sorting and counting is a critical skill in human’s reading comprehension, which has not been well considered in existing machine reading comprehension (MRC) systems. To address this issue, we propose a numerical MRC model named as NumNet, which utilizes a numerically-aware graph neural network to consider the comparing information and performs numerical reasoning over numbers in the question and passage. Our system achieves an EM-score of 64.56% on the DROP dataset, outperforming all existing machine reading comprehension models by considering the numerical relations among numbers.
     </details>

249. **Semantically-Aligned Equation Generation for Solving and Reasoning Math Word Problems** [[pdf]](http://aclweb.org/anthology/N19-1272) `NAACL 2019 Main` (100 cite) (17 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The proposed neural math solver is based on an encoder-decoder framework, where the encoder is designed to understand the semantics of problems, and the decoder focuses on tracking semantic meanings of the generated symbols and then deciding which symbol to generate next.
     </details>


     <details>
          <summary>Abstract</summary>
          Solving math word problems is a challenging task that requires accurate natural language understanding to bridge natural language texts and math expressions. Motivated by the intuition about how human generates the equations given the problem texts, this paper presents a neural approach to automatically solve math word problems by operating symbols according to their semantic meanings in texts. This paper views the process of generating equation as a bridge between the semantic world and the symbolic world, where the proposed neural math solver is based on an encoder-decoder framework. In the proposed model, the encoder is designed to understand the semantics of problems, and the decoder focuses on tracking semantic meanings of the generated symbols and then deciding which symbol to generate next. The preliminary experiments are conducted in a dataset Math23K, and our model significantly outperforms both the state-of-the-art single model and the best non-retrieval-based model over about 10% accuracy, demonstrating the effectiveness of bridging the symbolic and semantic worlds from math word problems.
     </details>

250. **Giving BERT a Calculator: Finding Operations and Arguments with Reading Comprehension** [[pdf]](https://www.aclweb.org/anthology/D19-1609) `EMNLP 2019 Main` (96 cite) (10 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work enables a BERT-based reading comprehension model to perform lightweight numerical reasoning by augmenting the model with a predefined set of executable ‘programs’ which encompass simple arithmetic as well as extraction.
     </details>


     <details>
          <summary>Abstract</summary>
          Reading comprehension models have been successfully applied to extractive text answers, but it is unclear how best to generalize these models to abstractive numerical answers. We enable a BERT-based reading comprehension model to perform lightweight numerical reasoning. We augment the model with a predefined set of executable ‘programs’ which encompass simple arithmetic as well as extraction. Rather than having to learn to manipulate numbers directly, the model can pick a program and execute it. On the recent Discrete Reasoning Over Passages (DROP) dataset, designed to challenge reading comprehension models, we show a 33% absolute improvement by adding shallow programs. The model can learn to predict new operations when appropriate in a math word problem setting (Roy and Roth, 2015) with very few training examples.
     </details>

251. **Modeling Intra-Relation in Math Word Problems with Different Functional Multi-Head Attentions** [[pdf]](https://www.aclweb.org/anthology/P19-1619) `ACL 2019 Main` (89 cite) (13 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The experimental results show that the proposed approach performs significantly better than previous state-of-the-art methods, and boost performance from 66.8% to 66.9% on Math23K with 5-fold cross-validation and from 69.2% to 76.1% on MAWPS.
     </details>


     <details>
          <summary>Abstract</summary>
          Several deep learning models have been proposed for solving math word problems (MWPs) automatically. Although these models have the ability to capture features without manual efforts, their approaches to capturing features are not specifically designed for MWPs. To utilize the merits of deep learning models with simultaneous consideration of MWPs’ specific features, we propose a group attention mechanism to extract global features, quantity-related features, quantity-pair features and question-related features in MWPs respectively. The experimental results show that the proposed approach performs significantly better than previous state-of-the-art methods, and boost performance from 66.9% to 69.5% on Math23K with training-test split, from 65.8% to 66.9% on Math23K with 5-fold cross-validation and from 69.2% to 76.1% on MAWPS.
     </details>

252. **Tree-structured Decoding for Solving Math Word Problems** [[pdf]](https://www.aclweb.org/anthology/D19-1241) `EMNLP 2019 Main` (81 cite) (11 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A tree-structured decoding method that generates the abstract syntax tree of the equation in a top-down manner and can automatically stop during decoding without a redundant stop token is proposed.
     </details>


     <details>
          <summary>Abstract</summary>
          Automatically solving math word problems is an interesting research topic that needs to bridge natural language descriptions and formal math equations. Previous studies introduced end-to-end neural network methods, but these approaches did not efficiently consider an important characteristic of the equation, i.e., an abstract syntax tree. To address this problem, we propose a tree-structured decoding method that generates the abstract syntax tree of the equation in a top-down manner. In addition, our approach can automatically stop during decoding without a redundant stop token. The experimental results show that our method achieves single model state-of-the-art performance on Math23K, which is the largest dataset on this task.
     </details>

253. **HOList: An Environment for Machine Learning of Higher-Order Theorem Proving (extended version)** [[pdf]](https://www.semanticscholar.org/paper/9ef2e09a9e16e176e19c3fdc3b6ee22c5d3f3c97) `ICML 2019` `HOL Light` (45 cite) (7 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work provides an open-source framework based on the HOL Light theorem prover that can be used as a reinforcement learning environment and presents a deep reinforcement learning driven automated theorem provers, DeepHOL, with strong initial results on this benchmark.
     </details>


     <details>
          <summary>Abstract</summary>
          N/A
     </details>

254. **Text2Math: End-to-end Parsing Text into Math Expressions** [[pdf]](https://www.aclweb.org/anthology/D19-1536) `EMNLP 2019 Main` (32 cite) (6 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes Text2Math, a model for semantically parsing text into math expressions that aims to predict the complete math expression at once as a tree structure, where minimal manual efforts are involved in the process.
     </details>


     <details>
          <summary>Abstract</summary>
          We propose Text2Math, a model for semantically parsing text into math expressions. The model can be used to solve different math related problems including arithmetic word problems and equation parsing problems. Unlike previous approaches, we tackle the problem from an end-to-end structured prediction perspective where our algorithm aims to predict the complete math expression at once as a tree structure, where minimal manual efforts are involved in the process. Empirical results on benchmark datasets demonstrate the efficacy of our approach.
     </details>

255. **Learning dynamic polynomial proofs** [[pdf]](https://www.semanticscholar.org/paper/de58d5dc7b7fb8e69928331178c9c24b717612ff) `NeurIPS 2019` (16 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces a machine learning based method to search for a dynamic proof within semi-algebraic proof systems that manipulate polynomial inequalities via elementary inference rules that infer new inequalities from the premises.
     </details>


     <details>
          <summary>Abstract</summary>
          N/A
     </details>

256. **FigureQA: An Annotated Figure Dataset for Visual Reasoning** [[pdf]](http://arxiv.org/abs/1710.07300) `ICLR 2018` (233 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          FigureQA is envisioned as a first step towards developing models that can intuitively recognize patterns from visual representations of data, and preliminary results indicate that the task poses a significant machine learning challenge.
     </details>


     <details>
          <summary>Abstract</summary>
          We introduce FigureQA, a visual reasoning corpus of over one million question-answer pairs grounded in over 100,000 images. The images are synthetic, scientific-style figures from five classes: line plots, dot-line plots, vertical and horizontal bar graphs, and pie charts. We formulate our reasoning task by generating questions from 15 templates; questions concern various relationships between plot elements and examine characteristics like the maximum, the minimum, area-under-the-curve, smoothness, and intersection. To resolve, such questions often require reference to multiple plot elements and synthesis of information distributed spatially throughout a figure. To facilitate the training of machine learning systems, the corpus also includes side data that can be used to formulate auxiliary objectives. In particular, we provide the numerical data used to generate each figure as well as bounding-box annotations for all plot elements. We study the proposed visual reasoning task by training several models, including the recently proposed Relation Network as a strong baseline. Preliminary results indicate that the task poses a significant machine learning challenge. We envision FigureQA as a first step towards developing models that can intuitively recognize patterns from visual representations of data.
     </details>

257. **Reinforcement Learning of Theorem Proving** [[pdf]](https://www.semanticscholar.org/paper/920febb03475b068286a855c10ea09b968fe7ee3) `NeurIPS 2018` `Mizar` (135 cite) (13 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A theorem proving algorithm that uses practically no domain heuristics for guiding its connection-style proof search and solves within the same number of inferences over 40% more problems than a baseline prover, which is an unusually high improvement in this hard AI domain.
     </details>


     <details>
          <summary>Abstract</summary>
          N/A
     </details>

258. **GamePad: A Learning Environment for Theorem Proving** [[pdf]](https://www.semanticscholar.org/paper/87c425f23bcac2f082968abda64a971f91522f73) `ICLR 2019` `Coq` (97 cite) (16 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A system called GamePad is introduced that can be used to explore the application of machine learning methods to theorem proving in the Coq proof assistant and addresses position evaluation and tactic prediction tasks, which arise naturally in tactic-based theorem proving.
     </details>


     <details>
          <summary>Abstract</summary>
          N/A
     </details>

259. **Premise Selection for Theorem Proving by Deep Graph Embedding** [[pdf]](http://arxiv.org/abs/1709.09994) `NeurIPS 2017` (124 cite) (12 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A deep learning-based approach to the problem of premise selection: selecting mathematical statements relevant for proving a given conjecture by representing a higher-order logic formula as a graph that is invariant to variable renaming but still fully preserves syntactic and semantic information.
     </details>


     <details>
          <summary>Abstract</summary>
          We propose a deep learning-based approach to the problem of premise selection: selecting mathematical statements relevant for proving a given conjecture. We represent a higher-order logic formula as a graph that is invariant to variable renaming but still fully preserves syntactic and semantic information. We then embed the graph into a vector via a novel embedding method that preserves the information of edge ordering. Our approach achieves state-of-the-art results on the HolStep dataset, improving the classification accuracy from 83% to 90.3%.
     </details>

260. **Deep Neural Solver for Math Word Problems** [[pdf]](https://aclanthology.org/D17-1088) `EMNLP 2017 Main` (310 cite) (42 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Experiments conducted on a large dataset show that the RNN model and the hybrid model significantly outperform state-of-the-art statistical learning methods for math word problem solving.
     </details>


     <details>
          <summary>Abstract</summary>
          This paper presents a deep neural solver to automatically solve math word problems. In contrast to previous statistical learning approaches, we directly translate math word problems to equation templates using a recurrent neural network (RNN) model, without sophisticated feature engineering. We further design a hybrid model that combines the RNN model and a similarity-based retrieval model to achieve additional performance improvement. Experiments conducted on a large dataset show that the RNN model and the hybrid model significantly outperform state-of-the-art statistical learning methods for math word problem solving.
     </details>

261. **Program Induction by Rationale Generation: Learning to Solve and Explain Algebraic Word Problems** [[pdf]](https://aclanthology.org/P17-1015) `ACL 2017 Long Papers` (543 cite) (65 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Experimental results show that indirect supervision of program learning via answer rationales is a promising strategy for inducing arithmetic programs.
     </details>


     <details>
          <summary>Abstract</summary>
          Solving algebraic word problems requires executing a series of arithmetic operations—a program—to obtain a final answer. However, since programs can be arbitrarily complicated, inducing them directly from question-answer pairs is a formidable challenge. To make this task more feasible, we solve these problems by generating answer rationales, sequences of natural language and human-readable mathematical expressions that derive the final answer through a series of small steps. Although rationales do not explicitly specify programs, they provide a scaffolding for their structure via intermediate milestones. To evaluate our approach, we have created a new 100,000-sample dataset of questions, answers and rationales. Experimental results show that indirect supervision of program learning via answer rationales is a promising strategy for inducing arithmetic programs.
     </details>

262. **HolStep: A Machine Learning Dataset for Higher-order Logic Theorem Proving** [[pdf]](http://arxiv.org/abs/1703.00426) `ICLR 2017` `HOL Light` (79 cite) (10 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A new dataset based on Higher-Order Logic (HOL) proofs is introduced, for the purpose of developing new machine learning-based theorem-proving strategies and the results of these models show the promise of applying machine learning to HOL theorem proving.
     </details>


     <details>
          <summary>Abstract</summary>
          Large computer-understandable proofs consist of millions of intermediate logical steps. The vast majority of such steps originate from manually selected and manually guided heuristics applied to intermediate goals. So far, machine learning has generally not been used to filter or generate these steps. In this paper, we introduce a new dataset based on Higher-Order Logic (HOL) proofs, for the purpose of developing new machine learning-based theorem-proving strategies. We make this dataset publicly available under the BSD license. We propose various machine learning tasks that can be performed on this dataset, and discuss their significance for theorem proving. We also benchmark a set of simple baseline machine learning models suited for the tasks (including logistic regression, convolutional neural networks and recurrent neural networks). The results of our baseline models show the promise of applying machine learning to HOL theorem proving.
     </details>

263. **DeepMath - Deep Sequence Models for Premise Selection** [[pdf]](http://arxiv.org/abs/1606.04442) `NeurIPS 2016` `Mizar` (212 cite) (26 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A two stage approach is proposed that yields good results for the premise selection task on the Mizar corpus while avoiding the hand-engineered features of existing state-of-the-art models.
     </details>


     <details>
          <summary>Abstract</summary>
          We study the effectiveness of neural sequence models for premise selection in automated theorem proving, one of the main bottlenecks in the formalization of mathematics. We propose a two stage approach for this task that yields good results for the premise selection task on the Mizar corpus while avoiding the hand-engineered features of existing state-of-the-art models. To our knowledge, this is the first time deep learning has been applied to theorem proving on a large scale.
     </details>

264. **From Textbooks to Knowledge: A Case Study in Harvesting Axiomatic Knowledge from Textbooks to Solve Geometry Problems** [[pdf]](https://www.semanticscholar.org/paper/81f466a535cdec4957989999f9ca381bc4fe14e9) `EMNLP 2017` (0 cite) (5 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Textbooks are rich sources of information. Harvesting structured knowledge from textbooks is a key challenge in many educational applications. As a case study, we present an approach for harvesting structured axiomatic knowledge from math textbooks. Our approach uses rich contextual and typographical features extracted from raw textbooks. It leverages the redundancy and shared ordering across multiple textbooks to further refine the harvested axioms. These axioms are then parsed into rules that are used to improve the state-of-the-art in solving geometry problems.
     </details>

265. **How well do Computers Solve Math Word Problems? Large-Scale Dataset Construction and Evaluation** [[pdf]](https://aclanthology.org/P16-1084) `ACL 2016 Long Papers` (130 cite) (21 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A large-scale dataset is built which is more than 9 times the size of previous ones, and contains many more problem types, and semi-automatically obtained from community question-answering web pages.
     </details>


     <details>
          <summary>Abstract</summary>
          No summary was provided.
     </details>

266. **Learning To Use Formulas To Solve Simple Arithmetic Problems** [[pdf]](https://aclanthology.org/P16-1202) `ACL 2016 Long Papers` (106 cite) (13 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel method to learn to use formulas to solve simple arithmetic word problems and beats the state-of-the-art by 86.07% of the problems in a corpus of standard primary school test questions.
     </details>


     <details>
          <summary>Abstract</summary>
          No summary was provided.
     </details>

267. **MAWPS: A Math Word Problem Repository** [[pdf]](https://aclanthology.org/N16-1136) `NAACL 2016 Main` (308 cite) (53 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          MAWPS allows for the automatic construction of datasets with particular characteristics, providing tools for tuning the lexical and template overlap of a dataset as well as for filtering ungrammatical problems from web-sourced corpora.
     </details>


     <details>
          <summary>Abstract</summary>
          No summary was provided.
     </details>

268. **Learning to Automatically Solve Algebra Word Problems** [[pdf]](https://aclanthology.org/P14-1026) `ACL 2014 Long Papers` (347 cite) (39 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          An approach for automatically learning to solve algebra word problems by reasons across sentence boundaries to construct and solve a system of linear equations, while simultaneously recovering an alignment of the variables and numbers to the problem text.
     </details>


     <details>
          <summary>Abstract</summary>
          No summary was provided.
     </details>

