# Papers Using Proof Assistants 

This is an incomplete list primarily focused on new papers in 2024 + older papers in top conferences.

1. **Proof Flow: Preliminary Study on Generative Flow Network Language Model Tuning for Formal Reasoning** [[pdf]](http://arxiv.org/abs/2410.13224) `2024-10-17` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper presents a proof of concept in the domain of formal reasoning, specifically in the Neural Theorem Proving (NTP) setting, where proofs specified in a formal language such as Lean can be deterministically and objectively verified.
     </details>


     <details>
          <summary>Abstract</summary>
          Reasoning is a fundamental substrate for solving novel and complex problems. Deliberate efforts in learning and developing frameworks around System 2 reasoning have made great strides, yet problems of sufficient complexity remain largely out of reach for open models. To address this gap, we examine the potential of Generative Flow Networks as a fine-tuning method for LLMs to unlock advanced reasoning capabilities. In this paper, we present a proof of concept in the domain of formal reasoning, specifically in the Neural Theorem Proving (NTP) setting, where proofs specified in a formal language such as Lean can be deterministically and objectively verified. Unlike classical reward-maximization reinforcement learning, which frequently over-exploits high-reward actions and fails to effectively explore the state space, GFlowNets have emerged as a promising approach for sampling compositional objects, improving generalization, and enabling models to maintain diverse hypotheses. Our early results demonstrate GFlowNet fine-tuning's potential for enhancing model performance in a search setting, which is especially relevant given the paradigm shift towards inference time compute scaling and "thinking slowly."
     </details>

2. **3D-Prover: Diversity Driven Theorem Proving With Determinantal Point Processes** [[pdf]](http://arxiv.org/abs/2410.11133) `ICLR 2025 Submission` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          A key challenge in automated formal reasoning is the intractable search space, which grows exponentially with the depth of the proof. This branching is caused by the large number of candidate proof tactics which can be applied to a given goal. Nonetheless, many of these tactics are semantically similar or lead to an execution error, wasting valuable resources in both cases. We address the problem of effectively pruning this search, using only synthetic data generated from previous proof attempts. We first demonstrate that it is possible to generate semantically aware tactic representations which capture the effect on the proving environment, likelihood of success and execution time. We then propose a novel filtering mechanism which leverages these representations to select semantically diverse and high quality tactics, using Determinantal Point Processes. Our approach, 3D-Prover, is designed to be general, and to augment any underlying tactic generator. We demonstrate the effectiveness of 3D-Prover on the miniF2F-valid and miniF2F-test benchmarks by augmenting the ReProver LLM. We show that our approach leads to an increase in the overall proof rate, as well as a significant improvement in the tactic success rate, execution time and diversity.
     </details>

3. **ABEL: Sample Efficient Online Reinforcement Learning for Neural Theorem Proving** [[pdf]](https://openreview.net/forum?id=kk3mSjVCUO) `NeurIPS 2024 Workshop MATH-AI` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          We propose a scalable and efficient reinforcement learning framework as a strong baseline for theorem proving with limited data. This baseline reaches performances comparable to the current state-of-the-art in theorem proving, while only training on a few hundred examples. This a first step toward an efficient and easily reproducible combination of autoformalization, synthetic data generation and reinforcement learning, which could unlock significant advancements in neural theorem proving.
     </details>

4. **Advancing Theorem Proving in LLMs through Large-Scale Synthetic Data** [[pdf]](https://openreview.net/forum?id=TPtXLihkny) `NeurIPS 2024 Workshop MATH-AI` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Proof assistants like Lean have revolutionized mathematical proof verification by providing high levels of accuracy and reliability. Although large language models (LLMs) have demonstrated potential in mathematical reasoning, their advancement in formal theorem proving is hindered by the scarcity of large, high-quality training datasets. To address this challenge, we present a novel approach to generate extensive Lean 4 proof data from natural language mathematical problems at the high school and undergraduate levels. Specifically, we synthesize 8 million formal statements with corresponding proofs, leveraging this dataset to fine-tune the DeepSeekMath 7B model. The resulting model, DS-Prover, achieves a pass rate of 50\% on the Lean 4 miniF2F benchmark, surpassing the previous state-of-the-art result of 41.0\%. These findings underscore the potential of large-scale synthetic data in significantly enhancing the theorem-proving capabilities of LLMs.
     </details>

5. **Library Learning Doesn’t: The Curious Case of the Single-Use “Library”** [[pdf]](https://openreview.net/forum?id=et2T8SKF1O) `NeurIPS 2024 Workshop MATH-AI` `Isabelle` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Advances in Large Language Models (LLMs) have spurred a wave of LLM library learning systems for mathematical reasoning.  These systems aim to learn a reusable library of *tools*, such as formal Isabelle lemmas or Python programs that are tailored to a family of tasks. Many of these systems are inspired by the human structuring of knowledge into reusable and extendable concepts, but do current methods actually learn reusable libraries of tools?    We study two library learning systems for mathematics which both reported increased accuracy: LEGO-Prover and TroVE. We find that function reuse is extremely infrequent on miniF2F and MATH. Our followup ablation experiments suggest that, rather than reuse, self-correction and self-consistency are the primary drivers of the observed performance gains.
     </details>

6. **NLIR: Natural Language Intermediate Representation for Mechanized Theorem Proving** [[pdf]](https://openreview.net/forum?id=QzOc0tpdef) `NeurIPS 2024 Workshop MATH-AI` `Coq` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Formal theorem proving is challenging for humans as well as for machines. Thanks to recent advances in LLM capabilities, we believe natural language can serve as a universal interface for reasoning about formal proofs. In this paper, 1) we introduce Pétanque, a new lightweight environment to interact with the Coq theorem prover; 2) we present two interactive proof protocols leveraging natural language as an intermediate representation for designing proof steps; 3) we implement beam search over these interaction protocols, using natural language to rerank proof candidates;  and 4) we use Pétanque to benchmark our search algorithms. Using our method with GPT-4o we can successfully synthesize proofs for 50% of the first 100/260 lemmas from the newly published Busy Beaver proofs.
     </details>

7. **Probabilistic Proof State Compression: Optimizing LLM-Guided Formal Verification** [[pdf]](https://openreview.net/forum?id=x2yiUEH0f9) `NeurIPS 2024 Workshop MATH-AI` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          While approaches to Large Language Model guided formal proof search have recently seen immense success, the scalability of these techniques is often hindered by the explosive growth of the proof search space. This paper introduces a novel approach that synergistically combines LLMs with conformal prediction techniques to guide and optimize formal proof search. We present a method for compressing the proof state space using adaptive, probability-based binning, informed by conformal prediction intervals. This compression technique significantly reduces the computational resources required for proof search while maintaining statistical guarantees on proof discovery. In addition, we present preliminary empirical results on a subset of the Lean 4 miniF2F test demonstrating the effectiveness of this method leading to a 23\% average reduction in proof search time compared to baseline open models, while maintaining comparable proof success rates.
     </details>

8. **Reasoning in Reasoning: A Hierarchical Framework for (Better and Faster) Neural Theorem Proving** [[pdf]](https://openreview.net/forum?id=H5hePMXKht) `NeurIPS 2024 Workshop MATH-AI` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Learning to do complex reasoning is the central objective of artificial intelligence. Autoregressive language models have shown promise in generating intermediate steps for problem solving; however, complex reasoning tasks such as theorem proving still present challenges due to the vast search spaces. Classical works have considered reasoning by searching, e.g., expanding the reasoning space with tree search to explore intermediate steps, and reasoning by decomposing, i.e., breaking down the problem into higher-level thoughts that prompt lower-level steps. We develop Reasoning in Reasoning (RiR), a hierarchical framework that unifies strategic problem decomposing with goal-driven reasoning step generation and search, via a planner-actor game. Using neural theorem proving as a representative task, our approach breaks down complex theorem problems into achievable subgoals, giving models: (i) improved generalizability for reasoning step generation, (ii) a more compact and informative search space for reasoning paths, and (iii) an efficient mechanism for learning to plan. We support RiR via an information-theoretic analysis, and show it achieves state-of-the-art performance and efficiency on popular theorem proving benchmarks including LeanDojo and miniF2F.
     </details>

9. **LeanAgent: Lifelong Learning for Formal Theorem Proving** [[pdf]](http://arxiv.org/abs/2410.06209) `ICLR 2025 Submission` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          LeanAgent is presented, a novel lifelong learning framework for theorem proving that continuously generalizes to and improves on ever-expanding mathematical knowledge without forgetting previously learned knowledge and achieves exceptional scores in stability and backward transfer.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) have been successful in mathematical reasoning tasks such as formal theorem proving when integrated with interactive proof assistants like Lean. Existing approaches involve training or fine-tuning an LLM on a specific dataset to perform well on particular domains, such as undergraduate-level mathematics. These methods struggle with generalizability to advanced mathematics. A fundamental limitation is that these approaches operate on static domains, failing to capture how mathematicians often work across multiple domains and projects simultaneously or cyclically. We present LeanAgent, a novel lifelong learning framework for theorem proving that continuously generalizes to and improves on ever-expanding mathematical knowledge without forgetting previously learned knowledge. LeanAgent introduces several key innovations, including a curriculum learning strategy that optimizes the learning trajectory in terms of mathematical difficulty, a dynamic database for efficient management of evolving mathematical knowledge, and progressive training to balance stability and plasticity. LeanAgent successfully proves 162 theorems previously unproved by humans across 23 diverse Lean repositories, many from advanced mathematics. It performs up to 11$\times$ better than the static LLM baseline, proving challenging theorems in domains like abstract algebra and algebraic topology while showcasing a clear progression of learning from basic concepts to advanced topics. In addition, we analyze LeanAgent's superior performance on key lifelong learning metrics. LeanAgent achieves exceptional scores in stability and backward transfer, where learning new tasks improves performance on previously learned tasks. This emphasizes LeanAgent's continuous generalizability and improvement, explaining its superior theorem proving performance.
     </details>

10. **ImProver: Agent-Based Automated Proof Optimization** [[pdf]](https://arxiv.org/abs/2410.04753v1) `ICLR 2025 Submission` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          ImProver is presented, a large-language-model agent that rewrites proofs to optimize arbitrary user-defined metrics in Lean and is capable of rewriting proofs so that they are substantially shorter, more modular, and more readable.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have been used to generate formal proofs of mathematical theorems in proofs assistants such as Lean. However, we often want to optimize a formal proof with respect to various criteria, depending on its downstream use. For example, we may want a proof to adhere to a certain style, or to be readable, concise, or modularly structured. Having suitably optimized proofs is also important for learning tasks, especially since human-written proofs may not optimal for that purpose. To this end, we study a new problem of automated proof optimization: rewriting a proof so that it is correct and optimizes for an arbitrary criterion, such as length or readability. As a first method for automated proof optimization, we present ImProver, a large-language-model agent that rewrites proofs to optimize arbitrary user-defined metrics in Lean. We find that naively applying LLMs to proof optimization falls short, and we incorporate various improvements into ImProver, such as the use of symbolic Lean context in a novel Chain-of-States technique, as well as error-correction and retrieval. We test ImProver on rewriting real-world undergraduate, competition, and research-level mathematics theorems, finding that ImProver is capable of rewriting proofs so that they are substantially shorter, more modular, and more readable.
     </details>

11. **Consistent Autoformalization for Constructing Mathematical Libraries** [[pdf]](http://arxiv.org/abs/2410.04194) `2024-10-05` `Isabelle` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper proposes the coordinated use of three mechanisms, most-similar retrieval augmented generation (MS-RAG), denoising steps, and auto-correction with syntax error feedback (Auto-SEF) to improve autoformalization quality.
     </details>


     <details>
          <summary>Abstract</summary>
          Autoformalization is the task of automatically translating mathematical content written in natural language to a formal language expression. The growing language interpretation capabilities of Large Language Models (LLMs), including in formal languages, are lowering the barriers for autoformalization. However, LLMs alone are not capable of consistently and reliably delivering autoformalization, in particular as the complexity and specialization of the target domain grows. As the field evolves into the direction of systematically applying autoformalization towards large mathematical libraries, the need to improve syntactic, terminological and semantic control increases. This paper proposes the coordinated use of three mechanisms, most-similar retrieval augmented generation (MS-RAG), denoising steps, and auto-correction with syntax error feedback (Auto-SEF) to improve autoformalization quality. The empirical analysis, across different models, demonstrates that these mechanisms can deliver autoformalizaton results which are syntactically, terminologically and semantically more consistent. These mechanisms can be applied across different LLMs and have shown to deliver improve results across different model types.
     </details>

12. **Alchemy: Amplifying Theorem-Proving Capability Through Symbolic Mutation** [[pdf]](https://openreview.net/forum?id=7NL74jUiMg) `ICLR 2025 Submission` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Formal proofs are challenging to write even for experienced experts. Recent progress in Neural Theorem Proving (NTP) shows promise in expediting this process. However, the formal corpora available on the Internet are limited compared to the general text, posing a significant data scarcity challenge for NTP. To address this issue, this work proposes Alchemy, a general framework for data synthesis that constructs formal theorems through symbolic mutation. Specifically, for each candidate theorem in Mathlib, we identify all invocable theorems that can be used to rewrite or apply to it. Subsequently, we mutate the candidate theorem by replacing the corresponding term in the statement with its equivalent form or antecedent. As a result, our method increases the number of theorems in Mathlib by an order of magnitude, from 110k to 6M. Furthermore, we perform continual pretraining and supervised finetuning on this augmented corpus for large language models. Experimental results demonstrate the effectiveness of our approach, achieving a 5% absolute performance improvement on Leandojo benchmark. Additionally, our synthetic data achieve a 2.5% absolute performance gain on the out-of-distribution miniF2F benchmark. To provide further insights, we conduct a comprehensive analysis of synthetic data composition and the training paradigm, offering valuable guidance for developing a strong theorem prover.
     </details>

13. **CARTS: Advancing Neural Theorem Proving with Diversified Tactic Calibration and Bias-Resistant Tree Search** [[pdf]](https://openreview.net/forum?id=VQwI055flA) `ICLR 2025 Submission` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Recent advancements in neural theorem proving integrate large language models with tree search algorithms like Monte Carlo Tree Search (MCTS), where the language model suggests tactics and the tree search finds the complete proof path. However, many tactics proposed by the language model converge to semantically or strategically similar, reducing diversity and increasing search costs by expanding redundant proof paths. This issue exacerbates as computation scales and more tactics are explored per state. Furthermore, the trained value function suffers from false negatives, label imbalance, and domain gaps due to biased data construction.  To address these challenges, we propose CARTS (diversified tactic CAlibration and bias-Resistant Tree Search), which balances tactic diversity and importance while calibrating model confidence. CARTS also introduce preference modeling and an adjustment term related to the ratio of valid tactics to improve the bias-resistance of the value function. Experimental results demonstrate that CARTS consistently outperforms previous methods achieving a pass@l rate of 49.6\% on the miniF2F-test benchmark. Further analysis confirms that CARTS improves tactic diversity and leads to a more balanced tree search.
     </details>

14. **Collaborative Theorem Proving with Large Language Models: Enhancing Formal Proofs with ProofRefiner** [[pdf]](https://openreview.net/forum?id=y9xNQZjUJM) `ICLR 2025 Submission` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Abstract： Theorem proving presents a significant challenge for large language models (LLMs) because formal proofs can be rigorously verified by proof assistants like Lean, leaving no room for errors. Existing LLM-based provers typically operate autonomously, but they often struggle with complex and novel theorems where human insights are crucial. We propose a new framework that positions LLMs as collaborative assistants in theorem proving to address this. This framework enables the seamless integration of LLM inference into the Lean environment, allowing developers to build various proof automation tools. These tools offer features such as suggesting proof steps, completing intermediate goals, and selecting relevant premises, thereby enhancing the theorem-proving process. Users can leverage our pretrained models or integrate their own, supporting local and cloud-based execution. Experimental results demonstrate that our approach is more effective in aiding humans and automating the theorem-proving process than existing rule-based systems. Additionally, we introduce a system called ProofRefiner, which refines questions and answers through dynamic dialogue adjustments to ensure relevance and precision.
     </details>

15. **Formal Theorem Proving by Rewarding LLMs to Decompose Proofs Hierarchically** [[pdf]](https://openreview.net/forum?id=D23JcXiUwf) `NeurIPS 2024 Workshop MATH-AI` `Isabelle` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Mathematical theorem proving is an important testbed for large language models’ deep and abstract reasoning capability. This paper focuses on improving LLMs’ ability to write proofs in formal languages that permit automated proof verification/ evaluation. Most previous results provide human-written lemmas to the theorem prover, which is an arguably oversimplified setting that does not sufficiently test the provers' planning and decomposition capabilities. Instead, we work in a more natural setup where the lemmas that are directly relevant to the theorem are not given to the theorem prover in the test time. We design an RL-based training algorithm that encourages the model to decompose the theorem into lemmas, prove the lemmas, and then prove the theorem by using the lemmas. Our reward mechanism is inspired by how mathematicians train themselves: even if a theorem is too challenging to be proved by the current model, a reward is still given (to the model) for any correct and novel lemmas that are proposed and proved in this process. During training, our model proves 37.7% lemmas that are not in the training dataset. When tested on a set of holdout theorems, our model improves the pass rate from 40.8% to 47.5% compared with the supervised fine-tuned model.
     </details>

16. **FormalAlign: Automated Alignment Evaluation for Autoformalization** [[pdf]](https://openreview.net/forum?id=B5RrIFMqbe) `ICLR 2025 Submission` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces \textsc{FormalAlign], the first automated framework designed for evaluating the alignment between natural and formal languages in autoformalization, and significantly reduces the need for manual verification.
     </details>


     <details>
          <summary>Abstract</summary>
          Autoformalization aims to convert informal mathematical proofs into machine-verifiable formats, bridging the gap between natural and formal languages. However, ensuring semantic alignment between the informal and formalized statements remains challenging. Existing approaches heavily rely on manual verification, hindering scalability. To address this, we introduce \textsc{FormalAlign}, the first automated framework designed for evaluating the alignment between natural and formal languages in autoformalization. \textsc{FormalAlign} trains on both the autoformalization sequence generation task and the representational alignment between input and output, employing a dual loss that combines a pair of mutually enhancing autoformalization and alignment tasks. Evaluated across four benchmarks augmented by our proposed misalignment strategies, \textsc{FormalAlign} demonstrates superior performance. In our experiments, \textsc{FormalAlign} outperforms GPT-4, achieving an Alignment-Selection Score 11.58\% higher on \forml-Basic (99.21\% vs. 88.91\%) and 3.19\% higher on MiniF2F-Valid (66.39\% vs. 64.34\%). This effective alignment evaluation significantly reduces the need for manual verification. Both the dataset and code can be accessed via~\url{https://github.com/rookie-joe/FormalAlign}.
     </details>

17. **Harnessing Proof Assistant Feedback for Reinforcement Learning and Monte-Carlo Tree Search** [[pdf]](https://openreview.net/forum?id=I4YAIwrsXa) `ICLR 2025 Submission` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Lean is an advanced proof assistant designed to facilitate formal theorem proving by providing a variety of interactive feedback. In this paper, we explore methodologies to leverage proof assistant feedback to augment the capabilities of large language models in constructing formal proofs. First, we deploy online reinforcement learning using Lean verification outcomes as the reward signal to improve the proof completion policy. This straightforward approach shows great promise in enhancing the model's alignment with the formal verification system. In addition, we propose RMaxTS, a variant of Monte-Carlo tree search that employs an intrinsic-reward-driven exploration strategy to generate diverse proof paths. The tree structure is organized to represent the transitions of intermediate tactic states, extracted from the compilation messages given by Lean's tactic mode. The intrinsic reward is constructed to incentivize the discovery of novel tactic states, which helps to to mitigate the sparse-reward problem inherent in proof search. These techniques lead to a more efficient planning scheme for formal proof generation, achieving new state-of-the-art results on both miniF2F and ProofNet benchmarks.
     </details>

18. **Lean-ing on Quality: How High-Quality Data Beats Diverse Multilingual Data in AutoFormalization** [[pdf]](https://openreview.net/forum?id=Qdp7hlenr6) `ICLR 2025 Submission` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Autoformalization, the process of transforming informal mathematical language into formal specifications and proofs remains a difficult task for state-of-the-art (large) language models. Existing works point to competing explanations for the performance gap. On one hand, large language models exhibit exceptional performance on translation tasks, suggesting their significant potential for autoformalization.  On the other hand, the quantitative reasoning capabilities of standard language models remain limited, leading to suboptimal performance on autoformalization and the subsequent task of formal theorem proving. To this end, we introduce a novel methodology that leverages backtranslation with hand-curated prompts to enhance the mathematical capabilities of language models, particularly addressing the challenge posed by the scarcity of labeled data.  Specifically, we evaluate three primary variations of this strategy: (1) on-the-fly (online) backtranslation, (2) distilled (offline) backtranslation with few-shot amplification, and (3) line-by-line proof analysis integrated with proof state information. Each variant is designed to optimize data quality over quantity, focusing on the high fidelity of generated proofs rather than sheer data scale. Our findings provide evidence that employing our proposed approaches to generate synthetic data, which prioritizes quality over volume, improves the autoformalization performance of LLMs as measured by standard benchmarks such as ProofNet. Crucially, our approach outperforms pretrained models using a minimal number of tokens. We also show, through strategic prompting and backtranslation, that our approaches surpass the performance of finetuning with extensive multilingual datasets such as MMA on ProofNet with only 1/150th of the tokens. Taken together, our methods show a promising new approach to significantly reduce the resources required to formalize proofs, thereby accelerating AI for math.
     </details>

19. **Proving Olympiad Inequalities by Synergizing LLMs and Symbolic Reasoning** [[pdf]](https://openreview.net/forum?id=FiyS0ecSm0) `ICLR 2025 Submission` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) can prove mathematical theorems formally by generating proof steps (\textit{a.k.a.} tactics) within a proof system. However, the space of possible tactics is vast and complex, while the available training data for formal proofs is limited, posing a significant challenge to LLM-based tactic generation. To address this, we introduce a neuro-symbolic tactic generator that synergizes the mathematical intuition learned by LLMs with domain-specific insights encoded by symbolic methods. The key aspect of this integration is identifying which parts of mathematical reasoning are best suited to LLMs and which to symbolic methods. While the high-level idea of neuro-symbolic integration is broadly applicable to various mathematical problems, in this paper, we focus specifically on Olympiad inequalities (Figure~\ref{fig:example}). We analyze how humans solve these problems and distill the techniques into two types of tactics: (1) scaling, handled by symbolic methods, and (2) rewriting, handled by LLMs. In addition, we combine symbolic tools with LLMs to prune and rank the proof goals for efficient proof search. We evaluate our framework on 161 challenging inequalities from multiple mathematics competitions, achieving state-of-the-art performance and significantly outperforming existing LLM and symbolic approaches without requiring additional training data.
     </details>

20. **Rethinking and improving autoformalization: towards a faithful metric and a Dependency Retrieval-based approach** [[pdf]](https://openreview.net/forum?id=hUb2At2DsQ) `ICLR 2025 Submission` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          As a central component in formal verification, statement autoformalization has been widely studied including the recent efforts from machine learning community, but still remains a widely-recognized difficult and open problem. In this paper, we delve into two critical yet under-explored gaps: 1) absence of faithful and universal automated evaluation for autoformalization results; 2) agnosia of contextural information, inducing severe hallucination of formal definitions and theorems. To address the first issue, we propose **BEq** (_**B**idirectional **E**xtended Definitional E**q**uivalence_), an automated neuro-symbolic method to determine the equivalence between two formal statements, which is formal-grounded and well-aligned with human intuition. For the second, we propose **RAutoformalizer** (_**R**etrieval-augmented **Autoformalizer**_), augmenting statement autoformalization by _Dependency Retrieval_, retrieving potentially dependent objects from formal libraries. We parse the dependencies of libraries and propose to _structurally informalise_ formal objects by the topological order of dependencies. To evaluate OOD generalization and research-level capabilities, we build a novel benchmark, _Con-NF_, consisting of 961 informal-formal statement pairs from frontier mathematical researches. Extensive experiments validate the effectiveness of our proposed approaches. In particular, BEq is evaluated on 200 diverse formal statement pairs with expert-annotated equivalence label, exhibiting significantly improved accuracy ($82.50\\% \mapsto 90.50\\%$) and precision ($70.59\\% \mapsto 100.0\\%$). For dependency retrieval, a baseline with excellent performance is established. The proposed RAutoformalizer substantially outperforms SOTA baselines in both in-distribution ProofNet benchmark ($12.83\\% \mapsto 18.18\\%$, BEq@8) and OOD Con-NF scenario ($4.58\\%\mapsto 16.86\\%$, BEq@8). Code, data, and models will be available.
     </details>

21. **StepProof: Step-by-step verification of natural language mathematical proofs** [[pdf]](https://openreview.net/forum?id=EXaKfdsw04) `ICLR 2025 Submission` `Isabelle` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Interactive theorem provers (ITPs) are powerful tools for the formal verification of mathematical proofs down to the axiom level. However, their lack of a natural language interface remains a significant limitation. Recent advancements in large language models (LLMs) have enhanced the understanding of natural language inputs, paving the way for autoformalization—the process of translating natural language proofs into formal proofs that can be verified. Despite these advancements, existing autoformalization approaches are limited to verifying complete proofs and lack the capability for finer, sentence-level verification. To address this gap, we propose StepProof, a novel autoformalization method designed for granular, step-by-step verification. StepProof breaks down complete proofs into multiple verifiable subproofs, enabling sentence-level verification. Experimental results demonstrate that StepProof significantly improves proof success rates and efficiency compared to traditional methods. Additionally, we found that minor manual adjustments to the natural language proofs, tailoring them for step-level verification, further enhanced StepProof’s performance in autoformalization.
     </details>

22. **Synthetic Theorem Generation in Lean** [[pdf]](https://openreview.net/forum?id=EeDSMy5Ruj) `ICLR 2025 Submission` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          The application of large language models (LLMs) to theorem proving presents a promising avenue for advancing formal mathematics. Interactive theorem provers, such as Lean, offer a rigorous framework within which these models can assist in or automate proof discovery, grounding their reasoning capabilities in a sound, verifiable formal system. However, the potential of LLMs in this domain is constrained by the limited availability of formal proof corpora for training. To address this limitation, we introduce a synthetic theorem generator capable of producing novel Lean theorems and their corresponding proofs. Our approach employs forward reasoning to synthesize new propositions from premises drawn from existing Lean libraries. We explore candidate reasoning steps using a search strategy that optimizes for diversity of output, apply them in a linear fashion that avoids irrelevant proof steps, and assess their effect by meta-programmatically executing corresponding Lean tactics. These methods enable the generation of an arbitrary number of new theorems and proofs across various mathematical domains, using common Lean proof tactics while ensuring the correctness of generated theorems by construction.  We demonstrate the efficacy of the generated theorems and training data by fine-tuning models on synthetic theorems and evaluating them on the miniF2F-test benchmark. Our results show improvements in theorem-proving capabilities, with accuracy increasing from 37.3% to 38.5% for the Falcon2-11B model trained solely on Mathlib, and from 38.1% to 39.3% for the same model trained on a mix of rich datasets. These improvements highlight the value of our diverse synthetic data in augmenting limited existing corpora of formal proofs, providing complementary information that enhances LLMs' performance on theorem-proving tasks even when combined with other datasets.
     </details>

23. **ZIP-FIT: Embedding-Free Data Selection via Compression-Based Alignment** [[pdf]](https://openreview.net/forum?id=4JBEpP6eRS) `ICLR 2025 Submission` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Selecting high-quality, aligned fine-tuning data is crucial for improving the downstream performance of language models (LMs). Automatic data selection in these scenarios is challenging and often inefficient due to previous approaches relying on neural embeddings or limited n-gram representations to identify aligned datasets. In addition, traditional data selection methods often focus on increasing the size of the training data, making them computationally expensive to use and data inefficient. In this work, we introduce ZIP-FIT, an embedding-free, data-efficient selection framework that leverages gzip compression to measure the alignment between training data and target domains. We show that ZIP-FIT significantly outperforms two leading baselines, DSIR and D4, in selecting high-quality data for ProofNet, a formal mathematical dataset, and HumanEval, a benchmark for code generation tasks. Specifically, ZIP-FIT demonstrates a computational speed advantage, performing data selection up to  65.8\% faster than DSIR and achieving its lowest cross-entropy loss up to 85.1\% faster. Our findings suggest that ZIP-FIT offers a scalable and adaptable approach for data selection, enabling more precise fine-tuning for code generation domains. By demonstrating that embedding-free data selection can outperform established methods like DSIR and D4, our research opens new avenues for optimizing model training, thereby enhancing the effectiveness and efficiency of machine learning workflows.
     </details>

24. **Proof Automation with Large Language Models** [[pdf]](https://arxiv.org/abs/2409.14274v1) `2024-09-22` `Coq` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          PALM is a novel generate-then-repair approach that first prompts an LLM to generate an initial proof and then leverages targeted symbolic methods to iteratively repair low-level problems, and significantly outperforms other state-of-the-art approaches.
     </details>


     <details>
          <summary>Abstract</summary>
          Interactive theorem provers such as Coq are powerful tools to formally guarantee the correctness of software. However, using these tools requires significant manual effort and expertise. While Large Language Models (LLMs) have shown promise in automatically generating informal proofs in natural language, they are less effective at generating formal proofs in interactive theorem provers. In this paper, we conduct a formative study to identify common mistakes made by LLMs when asked to generate formal proofs. By analyzing 520 proof generation errors made by GPT-3.5, we found that GPT-3.5 often identified the correct high-level structure of a proof, but struggled to get the lower-level details correct. Based on this insight, we propose PALM, a novel generate-then-repair approach that first prompts an LLM to generate an initial proof and then leverages targeted symbolic methods to iteratively repair low-level problems. We evaluate PALM on a large dataset that includes more than 10K theorems. Our results show that PALM significantly outperforms other state-of-the-art approaches, successfully proving 76.6% to 180.4% more theorems. Moreover, PALM proves 1270 theorems beyond the reach of existing approaches. We also demonstrate the generalizability of PALM across different LLMs.
     </details>

25. **CoqPilot, a plugin for LLM-based generation of proofs** [[pdf]](None) `2024-09-01` `Coq` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          We present CoqPilot, a VS Code extension designed to help automate the writing of Coq proofs. The plugin collects the parts of proofs marked with the admit tactic in a Coq file, i.e., proof holes, and combines LLMs along with non-machine-learning methods to generate proof candidates for the holes. Then, CoqPilot checks if each proof candidate solves the given subgoal and, if successful, replaces the hole with it.
     </details>

26. **Project Description: Experiments with Language Models for Isabelle Autoformalization** [[pdf]](None) `2024-09-01` `Isabelle` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          N/A
     </details>

27. **Project Proposal: Forward Reasoning in Hindsight** [[pdf]](https://www.semanticscholar.org/paper/48701c555c6321571d06000767c4f629010b6fd5) `2024-09-01` `Agda` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is argued that Hindsight Experience Replay interpretation in refutational theorem proving is somewhat indirect, and instead its application in reasoning settings where consequences are derived from axioms alone until a goal is reached is proposed.
     </details>


     <details>
          <summary>Abstract</summary>
          Hindsight Experience Replay is a promising technique in reinforcement learning. However, we argue that its interpretation in refutational theorem proving is somewhat indirect, and instead propose its application in reasoning settings where consequences are derived from axioms alone until a goal is reached. Such settings include many sequent-like calculi, condensed detachment, non-trivial fragments of dependently-typed languages such as Agda, and we conjecture that unit equational reasoning is also suitable.
     </details>

28. **Proof By Abduction in Isabelle/HOL** [[pdf]](None) `2024-09-01` `Isabelle` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          When proving an inductive problem, we often prove auxiliary lemmas that are useful for proving the original problem. If these auxiliary lemmas themselves are challenging, we must introduce more lemmas to prove these lemmas. To automate such multi-step conjecturing, we developed Abduction Prover. Given a proof goal, Abduction Prover conjectures a series of lemmas and attempts to prove the original goal using these lemmas. Our working prototype of Abduction Prover for Isabelle/HOL is publicly available on GitHub.
     </details>

29. **Proof Recommendation System for the HOL4 Theorem Prover** [[pdf]](None) `2024-09-01` `HOL 4` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          N/A
     </details>

30. **ProofDB: A prototype natural language Coq search engine** [[pdf]](None) `2024-09-01` `Coq` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          N/A
     </details>

31. **SubgoalXL: Subgoal-based Expert Learning for Theorem Proving** [[pdf]](https://arxiv.org/abs/2408.11172v1) `ICLR 2025 Submission` `Isabelle` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          SubgoalXL is introduced, a novel approach that synergizes subgoal-based proofs with expert learning to enhance LLMs' capabilities in formal theorem proving within the Isabelle environment, and achieves a new state-of-the-art performance.
     </details>


     <details>
          <summary>Abstract</summary>
          Formal theorem proving, a field at the intersection of mathematics and computer science, has seen renewed interest with advancements in large language models (LLMs). This paper introduces SubgoalXL, a novel approach that synergizes subgoal-based proofs with expert learning to enhance LLMs' capabilities in formal theorem proving within the Isabelle environment. SubgoalXL addresses two critical challenges: the scarcity of specialized mathematics and theorem-proving data, and the need for improved multi-step reasoning abilities in LLMs. By optimizing data efficiency and employing subgoal-level supervision, SubgoalXL extracts richer information from limited human-generated proofs. The framework integrates subgoal-oriented proof strategies with an expert learning system, iteratively refining formal statement, proof, and subgoal generators. Leveraging the Isabelle environment's advantages in subgoal-based proofs, SubgoalXL achieves a new state-of-the-art performance of 56.1\% in Isabelle on the standard miniF2F dataset, marking an absolute improvement of 4.9\%. Notably, SubgoalXL successfully solves 41 AMC12, 9 AIME, and 3 IMO problems from miniF2F. These results underscore the effectiveness of maximizing limited data utility and employing targeted guidance for complex reasoning in formal theorem proving, contributing to the ongoing advancement of AI reasoning capabilities. The implementation is available at \url{https://github.com/zhaoxlpku/SubgoalXL}.
     </details>

32. **DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for Reinforcement Learning and Monte-Carlo Tree Search** [[pdf]](http://arxiv.org/abs/2408.08152) `2024-08-15` `Lean` (8 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          An open-source language model designed for theorem proving in Lean 4, which enhances DeepSeek-Prover-V1 by optimizing both training and inference processes, and proposes RMaxTS, a variant of Monte-Carlo tree search that employs an intrinsic-reward-driven exploration strategy to generate diverse proof paths.
     </details>


     <details>
          <summary>Abstract</summary>
          We introduce DeepSeek-Prover-V1.5, an open-source language model designed for theorem proving in Lean 4, which enhances DeepSeek-Prover-V1 by optimizing both training and inference processes. Pre-trained on DeepSeekMath-Base with specialization in formal mathematical languages, the model undergoes supervised fine-tuning using an enhanced formal theorem proving dataset derived from DeepSeek-Prover-V1. Further refinement is achieved through reinforcement learning from proof assistant feedback (RLPAF). Beyond the single-pass whole-proof generation approach of DeepSeek-Prover-V1, we propose RMaxTS, a variant of Monte-Carlo tree search that employs an intrinsic-reward-driven exploration strategy to generate diverse proof paths. DeepSeek-Prover-V1.5 demonstrates significant improvements over DeepSeek-Prover-V1, achieving new state-of-the-art results on the test set of the high school level miniF2F benchmark ($63.5\%$) and the undergraduate level ProofNet benchmark ($25.3\%$).
     </details>

33. **miniCTX: Neural Theorem Proving with (Long-)Contexts** [[pdf]](https://arxiv.org/abs/2408.03350v1) `NeurIPS 2024 Workshop MATH-AI` `Lean` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces miniCTX, which tests a model's ability to prove formal mathematical theorems that depend on new context that is not seen during training, and offers ntp-toolkit for automatically extracting and annotating theorem proving data.
     </details>


     <details>
          <summary>Abstract</summary>
          We introduce miniCTX, which tests a model's ability to prove formal mathematical theorems that depend on new definitions, lemmas, or other contextual information that was not observed during training. miniCTX contains theorems sourced from real Lean projects and textbooks, each associated with a context that can span tens of thousands of tokens. Models are tasked with proving a theorem given access to code from the theorem's repository, which contains context that is helpful or needed for the proof. As a baseline for miniCTX, we introduce file-tuning, a simple recipe that trains a model to generate a proof step conditioned on the preceding file contents. File-tuning substantially outperforms the traditional neural theorem proving approach that fine-tunes on states alone. Additionally, our file-tuned model improves performance on the standard miniF2F benchmark, achieving a pass rate of 33.61%, which is a new state-of-the-art for 1.3B parameter models. Alongside miniCTX, we offer ntp-toolkit for automatically extracting and annotating theorem proving data, making it easy to add new projects into miniCTX to ensure that contexts are not seen during training. miniCTX offers a challenging and realistic perspective on evaluating neural theorem provers.
     </details>

34. **LEAN-GitHub: Compiling GitHub LEAN repositories for a versatile LEAN prover** [[pdf]](http://arxiv.org/abs/2407.17227) `2024-07-24` `Lean` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes LEAN-GitHub, a dataset consisting of large-scale formal data extracted from almost all Lean 4 repositories on GitHub, and achieves state-of-the-art on two other Lean 4 benchmarks targeting different fields/levels of math.
     </details>


     <details>
          <summary>Abstract</summary>
          Recently, large language models have presented promising results in aiding formal mathematical reasoning. However, their performance is restricted due to the scarcity of formal theorem-proving data, which requires additional effort to be extracted from raw formal language corpora. Meanwhile, a significant amount of human-written formal language corpora remains underutilized. To address this issue, we propose LEAN-GitHub, a dataset consisting of large-scale formal data extracted from almost all Lean 4 repositories on GitHub. After fine-tuning InternLM-math-plus on this dataset, our model achieved accuracies of 48.8% with a single pass and 54.5% with 64 passes on the Lean 4 miniF2F test, surpassing state-of-the-art method at 52%. And it also achieves state-of-the-art on two other Lean 4 benchmarks (ProofNet and Putnam) targeting different fields/levels of math. These results demonstrate that our proposed dataset is beneficial for formal reasoning on a wide range of math topics. We open-source our model at https://GitHub. com/InternLM/InternLM-Math and our data at https://huggingface.co/ datasets/InternLM/Lean-GitHub
     </details>

35. **Lean-STaR: Learning to Interleave Thinking and Proving** [[pdf]](http://arxiv.org/abs/2407.10040) `NeurIPS 2024 Workshop MATH-AI` `Lean` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work presents Lean-STaR, a framework for training language models to produce informal thoughts prior to each step of a proof, thereby boosting the model's theorem-proving capabilities.
     </details>


     <details>
          <summary>Abstract</summary>
          Traditional language model-based theorem proving assumes that by training on a sufficient amount of formal proof data, a model will learn to prove theorems. Our key observation is that a wealth of informal information that is not present in formal proofs can be useful for learning to prove theorems. For instance, humans think through steps of a proof, but this thought process is not visible in the resulting code. We present Lean-STaR, a framework for training language models to produce informal thoughts prior to each step of a proof, thereby boosting the model's theorem-proving capabilities. Lean-STaR uses retrospective ground-truth tactics to generate synthetic thoughts for training the language model. At inference time, the trained model directly generates the thoughts prior to the prediction of the tactics in each proof step. Building on the self-taught reasoner framework, we then apply expert iteration to further fine-tune the model on the correct proofs it samples and verifies using the Lean solver. Lean-STaR achieves state-of-the-art results on the miniF2F-test benchmark within the Lean theorem proving environment, significantly outperforming base models ($\boldsymbol{43.4\% \rightarrow 46.3\%,}$ Pass@64). We also analyze the impact of the augmented thoughts on various aspects of the theorem proving process, providing insights into their effectiveness.
     </details>

36. **Reliable Reasoning Beyond Natural Language** [[pdf]](http://arxiv.org/abs/2407.11373) `2024-07-19` `Prolog` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a neurosymbolic approach that prompts LLMs to extract and encode all relevant information from a problem statement as logical code statements, and then uses a logic programming language (Prolog) to conduct the iterative computations of explicit deductive reasoning.
     </details>


     <details>
          <summary>Abstract</summary>
          Despite their linguistic competence, Large Language models (LLMs) often exhibit limitations in their ability to reason reliably and flexibly. To address this, we propose a neurosymbolic approach that prompts LLMs to extract and encode all relevant information from a problem statement as logical code statements, and then use a logic programming language (Prolog) to conduct the iterative computations of explicit deductive reasoning. Our approach significantly enhances the performance of LLMs on the standard mathematical reasoning benchmark, GSM8k, and the Navigate dataset from the BIG-bench dataset. Additionally, we introduce a novel dataset, the Non-Linear Reasoning (NLR) dataset, consisting of 55 unique word problems that target the shortcomings of the next token prediction paradigm of LLMs and require complex non-linear reasoning but only basic arithmetic skills to solve. Our findings demonstrate that the integration of Prolog enables LLMs to achieve high performance on the NLR dataset, which even the most advanced language models (including GPT4) fail to solve using text only.
     </details>

37. **Thought-Like-Pro: Enhancing Reasoning of Large Language Models through Self-Driven Prolog-based Chain-of-Though** [[pdf]](http://arxiv.org/abs/2407.14562) `2024-07-18` `Prolog` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel learning framework, THOUGHT-LIKE-PRO, that enables LLMs to formulate rules and statements from given instructions and leverage the symbolic Prolog engine to derive results, and demonstrates robust generalization across out-of-distribution reasoning tasks.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have shown exceptional performance as general-purpose assistants, excelling across a variety of reasoning tasks. This achievement represents a significant step toward achieving artificial general intelligence (AGI). Despite these advancements, the effectiveness of LLMs often hinges on the specific prompting strategies employed, and there remains a lack of a robust framework to facilitate learning and generalization across diverse reasoning tasks. To address these challenges, we introduce a novel learning framework, THOUGHT-LIKE-PRO In this framework, we utilize imitation learning to imitate the Chain-of-Thought (CoT) process which is verified and translated from reasoning trajectories generated by a symbolic Prolog logic engine. This framework proceeds in a self-driven manner, that enables LLMs to formulate rules and statements from given instructions and leverage the symbolic Prolog engine to derive results. Subsequently, LLMs convert Prolog-derived successive reasoning trajectories into natural language CoT for imitation learning. Our empirical findings indicate that our proposed approach substantially enhances the reasoning abilities of LLMs and demonstrates robust generalization across out-of-distribution reasoning tasks.
     </details>

38. **PutnamBench: Evaluating Neural Theorem-Provers on the Putnam Mathematical Competition** [[pdf]](http://arxiv.org/abs/2407.11214) `NeurIPS 2024` `Lean, Isabelle, Coq` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          PutnamBench consists of 1697 hand-constructed formalizations of 640 theorems sourced from the William Lowell Putnam Mathematical Competition, the premier undergraduate-level mathematics competition in North America.
     </details>


     <details>
          <summary>Abstract</summary>
          We present PutnamBench, a new multilingual benchmark for evaluating the ability of neural theorem-provers to solve competition mathematics problems. PutnamBench consists of 1337 hand-constructed formalizations of 514 theorems sourced from the William Lowell Putnam Mathematical Competition, the premier undergraduate-level mathematics competition in North America.  All the theorems have formalizations in Lean 4 and Isabelle; a substantial subset also has Coq formalizations. Proving the theorems requires significant problem-solving ability and proficiency in a broad range of topics taught in undergraduate mathematics courses. We use PutnamBench to evaluate several established neural and symbolic theorem-provers.  These approaches can only solve a handful of the PutnamBench problems, establishing the benchmark as a difficult open challenge for research on neural theorem-proving. PutnamBench is available at https://github.com/trishullab/PUTNAM.
     </details>

39. **Towards Automated Functional Equation Proving: A Benchmark Dataset and A Domain-Specific In-Context Agent** [[pdf]](https://arxiv.org/abs/2407.14521v1) `2024-07-05` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          FEAS, an agent that enhances the COPRA in-context learning framework within Lean, is introduced, an agent that enhances the COPRA in-context learning framework within Lean and outperforms baselines on FunEq, particularly with the integration of domain-specific heuristics.
     </details>


     <details>
          <summary>Abstract</summary>
          Automated Theorem Proving (ATP) faces challenges due to its complexity and computational demands. Recent work has explored using Large Language Models (LLMs) for ATP action selection, but these methods can be resource-intensive. This study introduces FEAS, an agent that enhances the COPRA in-context learning framework within Lean. FEAS refines prompt generation, response parsing, and incorporates domain-specific heuristics for functional equations. It introduces FunEq, a curated dataset of functional equation problems with varying difficulty. FEAS outperforms baselines on FunEq, particularly with the integration of domain-specific heuristics. The results demonstrate FEAS's effectiveness in generating and formalizing high-level proof strategies into Lean proofs, showcasing the potential of tailored approaches for specific ATP challenges.
     </details>

40. **TheoremLlama: Transforming General-Purpose LLMs into Lean4 Experts** [[pdf]](http://arxiv.org/abs/2407.03203) `2024-07-03` `Lean` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper proposes TheoremLlama, an end-to-end framework that trains a general-purpose LLM to be a Lean4 expert, and provides Open Bootstrapped Theorems (OBT), an NL-FL aligned and bootstrapped dataset.
     </details>


     <details>
          <summary>Abstract</summary>
          Proving mathematical theorems using computer-verifiable formal languages like Lean significantly impacts mathematical reasoning. One approach to formal theorem proving involves generating complete proofs using Large Language Models (LLMs) based on Natural Language (NL) proofs. Similar methods have shown promising results in code generation. However, most modern LLMs exhibit suboptimal performance due to the scarcity of aligned NL and Formal Language (FL) theorem-proving data. This scarcity results in a paucity of methodologies for training LLMs and techniques to fully utilize their capabilities in composing formal proofs. To address the challenges, this paper proposes **TheoremLlama**, an end-to-end framework to train a general-purpose LLM to become a Lean4 expert. This framework encompasses NL-FL aligned dataset generation methods, training approaches for the LLM formal theorem prover, and techniques for LLM Lean4 proof writing. Using the dataset generation method, we provide *Open Bootstrapped Theorems* (OBT), an NL-FL aligned and bootstrapped dataset. A key innovation in this framework is the NL-FL bootstrapping method, where NL proofs are integrated into Lean4 code for training datasets, leveraging the NL reasoning ability of LLMs for formal reasoning. The **TheoremLlama** framework achieves cumulative accuracies of 36.48% and 33.61% on MiniF2F-Valid and Test datasets respectively, surpassing the GPT-4 baseline of 22.95% and 25.41%. We have also open-sourced our model checkpoints and generated dataset, and will soon make all the code publicly available.
     </details>

41. **Lean4trace: Data augmentation for neural theorem proving in Lean** [[pdf]](https://openreview.net/forum?id=sjLWmLeJ6R) `2024-06-13` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Integrating large language models as proof assistants with theorem provers has shown great promise. However, one of the major challenges in this field is the scarcity of training data. To address this, we release a new open-source tool, *Lean4trace*, for training data extraction from Lean 4 sources. Unlike previous approaches, *Lean4trace* is deeply integrated into the Lean elaborator, allowing us to modify proofs on-the-fly. Leveraging this feature, we propose two methods of data augmentation in Lean: (1) decomposing composite proof steps into multiple simpler steps; (2) testing existing proof automation tactics at each proof state and collecting the successful ones. Models trained on this augmented data are capable of proving 58.0% of theorems from a hold-out subset of Mathlib and 35.6% of the test subset of the MiniF2F benchmark.
     </details>

42. **More Details, Please: Improving Autoformalization with More Detailed Proofs** [[pdf]](https://openreview.net/forum?id=AkJvzpYMvK&name=pdf) `2024-06-13` `Isabelle` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          The formalization of mathematical theorems and their proofs is a time-consuming and tedious process which, despite recent advances in the reasoning capabilities of AI systems, remains a challenging task for computers. Existing attempts to automate the process with language models struggle with the difference in level of detail between formal and informal proofs. Successful autoformalization requires models to understand and be able to explain the nuances of logical arguments, a critical aspect of reasoning that is often overlooked in existing research. In this work, we introduce Sketch, Prove, Add Detail & Repeat (SPADeR), an approach that enhances proof autoformalizers by using language models to infer and explicitly incorporate implicit details from informal proofs. With the same number of autoformalization attempts, our method increases the percentage of successfully formalized problems in the miniF2F test dataset from 34.8% to 38.1%.
     </details>

43. **Improving Autoformalization using Type Checking** [[pdf]](http://arxiv.org/abs/2406.07222) `2024-06-11` `Lean` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a method to fix the performance of large language models for autoformalization through decoding with type-check filtering, where they initially sample a diverse set of candidate formalizations for an informal statement, then use the Lean proof assistant to filter out candidates that do not type-check.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models show promise for autoformalization, the task of automatically translating natural language into formal languages. However, current autoformalization methods remain limited. The last reported state-of-the-art performance on the ProofNet formalization benchmark for the Lean proof assistant, achieved using Codex for Lean 3, only showed successful formalization of 16.1% of informal statements. Similarly, our evaluation of GPT-4o for Lean 4 only produces successful translations 34.9% of the time. Our analysis shows that the performance of these models is largely limited by their inability to generate formal statements that successfully type-check (i.e., are syntactically correct and consistent with types) - with a whopping 86.6% of GPT-4o errors starting from a type-check failure. In this work, we propose a method to fix this issue through decoding with type-check filtering, where we initially sample a diverse set of candidate formalizations for an informal statement, then use the Lean proof assistant to filter out candidates that do not type-check. Using GPT-4o as a base model, and combining our method with self-consistency, we obtain a +18.3% absolute increase in formalization accuracy, and achieve a new state-of-the-art of 53.2% on ProofNet with Lean 4.
     </details>

44. **Lean Workbook: A large-scale Lean problem set formalized from natural language math problems** [[pdf]](http://arxiv.org/abs/2406.03847) `NeurIPS 2024` `Lean` (4 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a novel pipeline that iteratively generates and filters synthetic data to translate natural language mathematical problems into Lean 4 statements, and vice versa, and indicates that the synthetic data pipeline can provide useful training data and improve the performance of LLMs in translating and understanding complex mathematical problems and proofs.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models have demonstrated impressive capabilities across various natural language processing tasks, especially in solving mathematical problems. However, large language models are not good at math theorem proving using formal languages like Lean. A significant challenge in this area is the scarcity of training data available in these formal languages. To address this issue, we propose a novel pipeline that iteratively generates and filters synthetic data to translate natural language mathematical problems into Lean 4 statements, and vice versa. Our results indicate that the synthetic data pipeline can provide useful training data and improve the performance of LLMs in translating and understanding complex mathematical problems and proofs. Our final dataset contains about 57K formal-informal question pairs along with searched proof from the math contest forum and 21 new IMO questions. We open-source our code at \url{https://github.com/InternLM/InternLM-Math} and our data at \url{https://huggingface.co/datasets/InternLM/Lean-Workbook}.
     </details>

45. **Process-Driven Autoformalization in Lean 4** [[pdf]](http://arxiv.org/abs/2406.01940) `ICLR 2025 Submission` `Lean` (4 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A new benchmark for autoformalization for large language models designed to evaluate the autoformalization capabilities of large language models (LLMs), and introduces a model that leverages the precise feedback from Lean 4 compilers to enhance autoformalization.
     </details>


     <details>
          <summary>Abstract</summary>
          Autoformalization, the conversion of natural language mathematics into formal languages, offers significant potential for advancing mathematical reasoning. However, existing efforts are limited to formal languages with substantial online corpora and struggle to keep pace with rapidly evolving languages like Lean 4. To bridge this gap, we propose a new benchmark \textbf{Form}alization for \textbf{L}ean~\textbf{4} (\textbf{\name}) designed to evaluate the autoformalization capabilities of large language models (LLMs). This benchmark encompasses a comprehensive assessment of questions, answers, formal statements, and proofs. Additionally, we introduce a \textbf{P}rocess-\textbf{S}upervised \textbf{V}erifier (\textbf{PSV}) model that leverages the precise feedback from Lean 4 compilers to enhance autoformalization. Our experiments demonstrate that the PSV method improves autoformalization, enabling higher accuracy using less filtered training data. Furthermore, when fine-tuned with data containing detailed process information, PSV can leverage the data more effectively, leading to more significant improvements in autoformalization for Lean 4. Our dataset and code are available at \url{https://github.com/rookie-joe/PDA}.
     </details>

46. **An Evaluation Benchmark for Autoformalization in Lean4** [[pdf]](http://arxiv.org/abs/2406.06555) `2024-06-01` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel evaluation benchmark designed for Lean4 is introduced, applying it to test the abilities of state-of-the-art LLMs, including GPT-3.5, GPT-4, and Gemini Pro, revealing that these LLMs still exhibit limitations in autoformalization, particularly in more complex areas of mathematics.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) hold the potential to revolutionize autoformalization. The introduction of Lean4, a mathematical programming language, presents an unprecedented opportunity to rigorously assess the autoformalization capabilities of LLMs. This paper introduces a novel evaluation benchmark designed for Lean4, applying it to test the abilities of state-of-the-art LLMs, including GPT-3.5, GPT-4, and Gemini Pro. Our comprehensive analysis reveals that, despite recent advancements, these LLMs still exhibit limitations in autoformalization, particularly in more complex areas of mathematics. These findings underscore the need for further development in LLMs to fully harness their potential in scientific research and development. This study not only benchmarks current LLM capabilities but also sets the stage for future enhancements in autoformalization.
     </details>

47. **Autoformalizing Euclidean Geometry** [[pdf]](http://arxiv.org/abs/2405.17216) `ICML 2024` `Lean` (2 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A neuro-symbolic framework for autoformalizing Euclidean geometry, which combines domain knowledge, SMT solvers, and large language models (LLMs), is introduced and automatic semantic evaluation for autoformalized theorem statements is provided.
     </details>


     <details>
          <summary>Abstract</summary>
          Autoformalization involves automatically translating informal math into formal theorems and proofs that are machine-verifiable. Euclidean geometry provides an interesting and controllable domain for studying autoformalization. In this paper, we introduce a neuro-symbolic framework for autoformalizing Euclidean geometry, which combines domain knowledge, SMT solvers, and large language models (LLMs). One challenge in Euclidean geometry is that informal proofs rely on diagrams, leaving gaps in texts that are hard to formalize. To address this issue, we use theorem provers to fill in such diagrammatic information automatically, so that the LLM only needs to autoformalize the explicit textual steps, making it easier for the model. We also provide automatic semantic evaluation for autoformalized theorem statements. We construct LeanEuclid, an autoformalization benchmark consisting of problems from Euclid's Elements and the UniGeo dataset formalized in the Lean proof assistant. Experiments with GPT-4 and GPT-4V show the capability and limitations of state-of-the-art LLMs on autoformalizing geometry problems. The data and code are available at https://github.com/loganrjmurphy/LeanEuclid.
     </details>

48. **DeepSeek-Prover: Advancing Theorem Proving in LLMs through Large-Scale Synthetic Data** [[pdf]](http://arxiv.org/abs/2405.14333) `2024-05-23` `Lean` (15 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces an approach to generate extensive Lean 4 proof data derived from high-school and undergraduate-level mathematical competition problems to enhance theorem-proving capabilities in LLMs and demonstrates the potential of leveraging large-scale synthetic data to enhance theorem-proving capabilities in LLMs.
     </details>


     <details>
          <summary>Abstract</summary>
          Proof assistants like Lean have revolutionized mathematical proof verification, ensuring high accuracy and reliability. Although large language models (LLMs) show promise in mathematical reasoning, their advancement in formal theorem proving is hindered by a lack of training data. To address this issue, we introduce an approach to generate extensive Lean 4 proof data derived from high-school and undergraduate-level mathematical competition problems. This approach involves translating natural language problems into formal statements, filtering out low-quality statements, and generating proofs to create synthetic data. After fine-tuning the DeepSeekMath 7B model on this synthetic dataset, which comprises 8 million formal statements with proofs, our model achieved whole-proof generation accuracies of 46.3% with 64 samples and 52% cumulatively on the Lean 4 miniF2F test, surpassing the baseline GPT-4 at 23.0% with 64 samples and a tree search reinforcement learning method at 41.0%. Additionally, our model successfully proved 5 out of 148 problems in the Lean 4 Formalized International Mathematical Olympiad (FIMO) benchmark, while GPT-4 failed to prove any. These results demonstrate the potential of leveraging large-scale synthetic data to enhance theorem-proving capabilities in LLMs. Both the synthetic dataset and the model will be made available to facilitate further research in this promising field.
     </details>

49. **Proving Theorems Recursively** [[pdf]](http://arxiv.org/abs/2405.14414) `NeurIPS 2024` `Isabelle` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          POETRY (PrOvE Theorems RecursivelY), which proves theorems in a recursive, level-by-level manner in the Isabelle theorem prover, is proposed, which allows the theorem to be tackled incrementally by outlining the overall theorem at the first level and then solving the intermediate conjectures at deeper levels.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent advances in automated theorem proving leverages language models to explore expanded search spaces by step-by-step proof generation. However, such approaches are usually based on short-sighted heuristics (e.g., log probability or value function scores) that potentially lead to suboptimal or even distracting subgoals, preventing us from finding longer proofs. To address this challenge, we propose POETRY (PrOvE Theorems RecursivelY), which proves theorems in a recursive, level-by-level manner in the Isabelle theorem prover. Unlike previous step-by-step methods, POETRY searches for a verifiable sketch of the proof at each level and focuses on solving the current level's theorem or conjecture. Detailed proofs of intermediate conjectures within the sketch are temporarily replaced by a placeholder tactic called sorry, deferring their proofs to subsequent levels. This approach allows the theorem to be tackled incrementally by outlining the overall theorem at the first level and then solving the intermediate conjectures at deeper levels. Experiments are conducted on the miniF2F and PISA datasets and significant performance gains are observed in our POETRY approach over state-of-the-art methods. POETRY on miniF2F achieves an average proving success rate improvement of 5.1%. Moreover, we observe a substantial increase in the maximum proof length found by POETRY, from 10 to 26.
     </details>

50. **ATG: Benchmarking Automated Theorem Generation for Generative Language Models** [[pdf]](http://arxiv.org/abs/2405.06677) `NAACL 2024 Findings` `MetaMath` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          An Automated Theorem Generation (ATG) benchmark that evaluates whether an agent can automatically generate valuable (and possibly brand new) theorems that are applicable for downstream theorem proving as reusable knowledge is proposed.
     </details>


     <details>
          <summary>Abstract</summary>
          Humans can develop new theorems to explore broader and more complex mathematical results.While current generative language models (LMs) have achieved significant improvement in automatically proving theorems, their ability to generate new or reusable theorems is still under-explored. Without the new theorems, current LMs struggle to prove harder theorems that are distant from the given hypotheses with the exponentially growing search space.More advanced theorem proving is if an agent (for instance, a generative LM) can leverage its creativity to generate new but also reasonable theorems that properly substitute part of a proof and also be saved as reusable knowledge for future theorem proving.Therefore, this paper proposes an Automated Theorem Generation (ATG) benchmark that evaluates whether an agent can automatically generate valuable (and possibly brand new) theorems that are applicable for downstream theorem proving as reusable knowledge. Specifically, we construct the ATG benchmark by splitting the Metamath library into three sets: axioms, library, and problem based on their proving depth.We conduct extensive experiments to investigate whether current LMs can generate theorems in the library and benefit the problem theorems proving. The results demonstrate that high-quality ATG data facilitates models’ performances on downstream ATP. However, there is still room for current LMs to develop better ATG and generate more advanced and human-like theorems. We hope the new ATG challenge can shed some light on advanced complex theorem proving.
     </details>

51. **Learn from Failure: Fine-tuning LLMs with Trial-and-Error Data for Intuitionistic Propositional Logic Proving** [[pdf]](http://arxiv.org/abs/2404.07382) `ACL 2024 Long Papers` `Lean` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Facing the lack of such trial-and-error data in existing open-source theorem-proving datasets, this paper curate a dataset on intuitionistic propositional logic theorems and formalize it in Lean, such that it can reliably check the correctness of proofs.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent advances in Automated Theorem Proving have shown the effectiveness of leveraging a (large) language model that generates tactics (i.e. proof steps) to search through proof states. The current model, while trained solely on successful proof paths, faces a discrepancy at the inference stage, as it must sample and try various tactics at each proof state until finding success, unlike its training which does not incorporate learning from failed attempts. Intuitively, a tactic that leads to a failed search path would indicate that similar tactics should receive less attention during the following trials. In this paper, we demonstrate the benefit of training models that additionally learn from failed search paths. Facing the lack of such trial-and-error data in existing open-source theorem-proving datasets, we curate a dataset on intuitionistic propositional logic theorems and formalize it in Lean, such that we can reliably check the correctness of proofs. We compare our model trained on relatively short trial-and-error information (TrialMaster) with models trained only on the correct paths and discover that the former solves more unseen theorems with lower trial searches.
     </details>

52. **Enhancing Formal Theorem Proving: A Comprehensive Dataset for Training AI Models on Coq Code** [[pdf]](http://arxiv.org/abs/2403.12627) `2024-04-02` `Coq` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper presents a comprehensive dataset specifically designed to enhance LLMs' proficiency in interpreting and generating Coq code, and discusses the dataset's composition, the methodology behind its creation, and the implications for the future of machine learning in formal verification.
     </details>


     <details>
          <summary>Abstract</summary>
          In the realm of formal theorem proving, the Coq proof assistant stands out for its rigorous approach to verifying mathematical assertions and software correctness. Despite the advances in artificial intelligence and machine learning, the specialized nature of Coq syntax and semantics poses unique challenges for Large Language Models (LLMs). Addressing this gap, we present a comprehensive dataset specifically designed to enhance LLMs' proficiency in interpreting and generating Coq code. This dataset, derived from a collection of over 10,000 Coq source files, encompasses a wide array of propositions, proofs, and definitions, enriched with metadata including source references and licensing information. Our primary aim is to facilitate the development of LLMs capable of generating syntactically correct and semantically meaningful Coq constructs, thereby advancing the frontier of automated theorem proving. Initial experiments with this dataset have showcased its significant potential; models trained on this data exhibited enhanced accuracy in Coq code generation. Notably, a particular experiment revealed that a fine-tuned LLM was capable of generating 141 valid proofs for a basic lemma, highlighting the dataset's utility in facilitating the discovery of diverse and valid proof strategies. This paper discusses the dataset's composition, the methodology behind its creation, and the implications of our findings for the future of machine learning in formal verification. The dataset is accessible for further research and exploration: https://huggingface.co/datasets/florath/coq-facts-props-proofs-gen0-v1
     </details>

53. **GFLean: An Autoformalisation Framework for Lean via GF** [[pdf]](http://arxiv.org/abs/2404.01234) `2024-04-01` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          An autoformalisation framework for the Lean theorem prover, called GFLean, which uses a high-level grammar writing tool called Grammatical Framework for parsing and linearisation and is implemented in Haskell.
     </details>


     <details>
          <summary>Abstract</summary>
          We present an autoformalisation framework for the Lean theorem prover, called GFLean. GFLean uses a high-level grammar writing tool called Grammatical Framework (GF) for parsing and linearisation. GFLean is implemented in Haskell. We explain the functionalities of GFLean, its inner working and discuss its limitations. We also discuss how we can use neural network based translation programs and rule based translation programs together complimenting each other to build robust autoformalisation frameworks.
     </details>

54. **Don't Trust: Verify -- Grounding LLM Quantitative Reasoning with Autoformalization** [[pdf]](http://arxiv.org/abs/2403.18120) `ICLR 2024` `Isabelle` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Large language models (LLM), such as Google's Minerva and OpenAI's GPT families, are becoming increasingly capable of solving mathematical quantitative reasoning problems. However, they still make unjustified logical and computational errors in their reasoning steps and answers. In this paper, we leverage the fact that if the training corpus of LLMs contained sufficiently many examples of formal mathematics (e.g. in Isabelle, a formal theorem proving environment), they can be prompted to translate i.e. autoformalize informal mathematical statements into formal Isabelle code --- which can be verified automatically for internal consistency. This provides a mechanism to automatically reject solutions whose formalized versions are inconsistent within themselves or with the formalized problem statement. We evaluate our method on GSM8K, MATH and MultiArith datasets and demonstrate that our approach provides a consistently better heuristic than vanilla majority voting --- the previously best method to identify correct answers, by more than 12\% on GSM8K. In our experiments it improves results consistently across all datasets and LLM model sizes.
     </details>

55. **MUSTARD: Mastering Uniform Synthesis of Theorem and Proof Data** [[pdf]](http://arxiv.org/abs/2402.08957) `ICLR 2024` `Lean` (17 cite) (3 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces MUSTARD, a data generation framework that masters uniform synthesis of theorem and proof data of high quality and diversity, and performs extensive analysis and demonstrates that MUSTARD generates validated high-quality step-by-step data.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent large language models (LLMs) have witnessed significant advancement in various tasks, including mathematical reasoning and theorem proving. As these two tasks require strict and formal multi-step inference, they are appealing domains for exploring the reasoning ability of LLMs but still face important challenges. Previous studies such as Chain-of-Thought (CoT) have revealed the effectiveness of intermediate steps guidance. However, such step-wise annotation requires heavy labor, leading to insufficient training steps for current benchmarks. To fill this gap, this work introduces MUSTARD, a data generation framework that masters uniform synthesis of theorem and proof data of high quality and diversity. MUSTARD synthesizes data in three stages: (1) It samples a few mathematical concept seeds as the problem category. (2) Then, it prompts a generative language model with the sampled concepts to obtain both the problems and their step-wise formal solutions. (3) Lastly, the framework utilizes a proof assistant (e.g., Lean Prover) to filter the valid proofs. With the proposed MUSTARD, we present a theorem-and-proof benchmark MUSTARDSAUCE with 5,866 valid data points. Each data point contains an informal statement, an informal proof, and a translated formal proof that passes the prover validation. We perform extensive analysis and demonstrate that MUSTARD generates validated high-quality step-by-step data. We further apply the MUSTARDSAUCE for fine-tuning smaller language models. The fine-tuned Llama 2-7B achieves a 15.41% average relative performance gain in automated theorem proving, and 8.18% in math word problems. Codes and data are available at https://github.com/Eleanor-H/MUSTARD.
     </details>

56. **Verified Multi-Step Synthesis using Large Language Models and Monte Carlo Tree Search** [[pdf]](http://arxiv.org/abs/2402.08147) `2024-02-12` `Lean, Coq` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          We present an approach using Monte Carlo Tree Search (MCTS) to guide Large Language Models (LLMs) to generate verified programs in Dafny, Lean and Coq. Our method, which we call VMCTS, leverages the verifier inside the search algorithm by checking partial programs at each step. In combination with the LLM prior, the verifier feedback raises the synthesis capabilities of open source models. On a set of five verified programming problems, we find that in four problems where the base model cannot solve the question even when re-sampling solutions for one hour, VMCTS can solve the problems within 6 minutes. The base model with VMCTS is even competitive with ChatGPT4 augmented with plugins and multiple re-tries on these problems. Our code and benchmarks are available at https://github.com/namin/llm-verified-with-monte-carlo-tree-search .
     </details>

57. **InternLM-Math: Open Math Large Language Models Toward Verifiable Reasoning** [[pdf]](http://arxiv.org/abs/2402.06332) `2024-02-09` `Lean` (33 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduces and open-source their math reasoning LLMs InternLM-Math which is continue pre-trained from InternLM2 and unify chain-of-thought reasoning, reward modeling, formal reasoning, data augmentation, and code interpreter in a unified seq2seq format and supervise the authors' model to be a versatile math reasoner, verifier, prover, and augmenter.
     </details>


     <details>
          <summary>Abstract</summary>
          The math abilities of large language models can represent their abstract reasoning ability. In this paper, we introduce and open-source our math reasoning LLMs InternLM-Math which is continue pre-trained from InternLM2. We unify chain-of-thought reasoning, reward modeling, formal reasoning, data augmentation, and code interpreter in a unified seq2seq format and supervise our model to be a versatile math reasoner, verifier, prover, and augmenter. These abilities can be used to develop the next math LLMs or self-iteration. InternLM-Math obtains open-sourced state-of-the-art performance under the setting of in-context learning, supervised fine-tuning, and code-assisted reasoning in various informal and formal benchmarks including GSM8K, MATH, Hungary math exam, MathBench-ZH, and MiniF2F. Our pre-trained model achieves 30.3 on the MiniF2F test set without fine-tuning. We further explore how to use LEAN to solve math problems and study its performance under the setting of multi-task learning which shows the possibility of using LEAN as a unified platform for solving and proving in math. Our models, codes, and data are released at \url{https://github.com/InternLM/InternLM-Math}.
     </details>

58. **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models** [[pdf]](http://arxiv.org/abs/2402.03300) `2024-02-06` `Isabelle` (123 cite) (7 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Group Relative Policy Optimization (GRPO), a variant of Proximal Policy Optimization (PPO), that enhances mathematical reasoning abilities while concurrently optimizing the memory usage of PPO is introduced.
     </details>


     <details>
          <summary>Abstract</summary>
          Mathematical reasoning poses a significant challenge for language models due to its complex and structured nature. In this paper, we introduce DeepSeekMath 7B, which continues pre-training DeepSeek-Coder-Base-v1.5 7B with 120B math-related tokens sourced from Common Crawl, together with natural language and code data. DeepSeekMath 7B has achieved an impressive score of 51.7% on the competition-level MATH benchmark without relying on external toolkits and voting techniques, approaching the performance level of Gemini-Ultra and GPT-4. Self-consistency over 64 samples from DeepSeekMath 7B achieves 60.9% on MATH. The mathematical reasoning capability of DeepSeekMath is attributed to two key factors: First, we harness the significant potential of publicly available web data through a meticulously engineered data selection pipeline. Second, we introduce Group Relative Policy Optimization (GRPO), a variant of Proximal Policy Optimization (PPO), that enhances mathematical reasoning abilities while concurrently optimizing the memory usage of PPO.
     </details>

59. **Graph2Tac: Online Representation Learning of Formal Math Concepts** [[pdf]](http://arxiv.org/abs/2401.02949) `ICML 2024` `Coq` (3 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work extensively benchmarks two online solvers implemented in the Tactician platform for the Coq proof assistant, and introduces a graph neural network, Graph2Tac, with a novel approach to build hierarchical representations for new definitions.
     </details>


     <details>
          <summary>Abstract</summary>
          In proof assistants, the physical proximity between two formal mathematical concepts is a strong predictor of their mutual relevance. Furthermore, lemmas with close proximity regularly exhibit similar proof structures. We show that this _locality_ property can be exploited through online learning techniques to obtain solving agents that far surpass offline learners when asked to prove theorems in an unseen mathematical setting. We extensively benchmark two such online solvers implemented in the Tactician platform for the Coq proof assistant: First, Tactician's online $k$-nearest neighbor solver, which can learn from recent proofs, shows a $1.72\times$ improvement in theorems proved over an offline equivalent. Second, we introduce a graph neural network, Graph2Tac, with a novel approach to build hierarchical representations for new definitions. Graph2Tac's online definition task realizes a $1.5\times$ improvement in theorems solved over an offline baseline. The $k$-NN and Graph2Tac solvers rely on orthogonal online data, making them highly complementary. Their combination improves $1.27\times$ over their individual performances. Both solvers outperform all other general purpose provers for Coq, including CoqHammer, Proverbot9001, and a transformer baseline by at least $1.48\times$ and are available for practical use by end-users.
     </details>

60. **The Tactician's Web of Large-Scale Formal Knowledge** [[pdf]](http://arxiv.org/abs/2401.02950) `2024-01-09` `Coq` (3 cite) (1 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          The Tactician's Web is a platform offering a large web of strongly interconnected, machine-checked, formal mathematical knowledge conveniently packaged for machine learning, analytics, and proof engineering. Built on top of the Coq proof assistant, the platform exports a dataset containing a wide variety of formal theories, presented as a web of definitions, theorems, proof terms, tactics, and proof states. Theories are encoded both as a semantic graph (rendered below) and as human-readable text, each with a unique set of advantages and disadvantages. Proving agents may interact with Coq through the same rich data representation and can be automatically benchmarked on a set of theorems. Tight integration with Coq provides the unique possibility to make agents available to proof engineers as practical tools.
     </details>

61. **Enhancing Neural Theorem Proving through Data Augmentation and Dynamic Sampling Method** [[pdf]](http://arxiv.org/abs/2312.14188) `2023-12-20` `Lean` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          DS-Prover is introduced, a novel dynamic sampling method for theorem proving that dynamically determines the number of tactics to apply to expand the current goal, taking into account the remaining time compared to the total allocated time for proving a theorem.
     </details>


     <details>
          <summary>Abstract</summary>
          Theorem proving is a fundamental task in mathematics. With the advent of large language models (LLMs) and interactive theorem provers (ITPs) like Lean, there has been growing interest in integrating LLMs and ITPs to automate theorem proving. In this approach, the LLM generates proof steps (tactics), and the ITP checks the applicability of the tactics at the current goal. The two systems work together to complete the proof. In this paper, we introduce DS-Prover, a novel dynamic sampling method for theorem proving. This method dynamically determines the number of tactics to apply to expand the current goal, taking into account the remaining time compared to the total allocated time for proving a theorem. This makes the proof search process more efficient by adjusting the balance between exploration and exploitation as time passes. We also augment the training dataset by decomposing simplification and rewrite tactics with multiple premises into tactics with single premises. This gives the model more examples to learn from and helps it to predict the tactics with premises more accurately. We perform our experiments using the Mathlib dataset of the Lean theorem prover and report the performance on two standard datasets, MiniF2F and ProofNet. Our methods achieve significant performance gains on both datasets. We achieved a state-of-the-art performance (Pass@1) of 14.2% on the ProofNet dataset and a performance of 29.8% on MiniF2F, slightly surpassing the best-reported Pass@1 of 29.6% using Lean.
     </details>

62. **Multilingual Mathematical Autoformalization** [[pdf]](http://arxiv.org/abs/2311.03755) `2023-11-09` `Lean, Isabelle` (9 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work creates a large, flexible, multilingual, and multi-domain dataset of informal-formal pairs, by using a language model to translate in the reverse direction, that is, from formal mathematical statements into corresponding informal ones, and demonstrates that fine-tuning on multilingual formal data results in more capable autoformalization models even when deployed on monolingual tasks.
     </details>


     <details>
          <summary>Abstract</summary>
          Autoformalization is the task of translating natural language materials into machine-verifiable formalisations. Progress in autoformalization research is hindered by the lack of a sizeable dataset consisting of informal-formal pairs expressing the same essence. Existing methods tend to circumvent this challenge by manually curating small corpora or using few-shot learning with large language models. But these methods suffer from data scarcity and formal language acquisition difficulty. In this work, we create $\texttt{MMA}$, a large, flexible, multilingual, and multi-domain dataset of informal-formal pairs, by using a language model to translate in the reverse direction, that is, from formal mathematical statements into corresponding informal ones. Experiments show that language models fine-tuned on $\texttt{MMA}$ produce $16-18\%$ of statements acceptable with minimal corrections on the $\texttt{miniF2F}$ and $\texttt{ProofNet}$ benchmarks, up from $0\%$ with the base model. We demonstrate that fine-tuning on multilingual formal data results in more capable autoformalization models even when deployed on monolingual tasks.
     </details>

63. **Temperature-scaled large language models for Lean proofstep prediction** [[pdf]](https://openreview.net/forum?id=sSgdyY0YJR) `NeurIPS 2023 MATH-AI Workshop` `Lean` (6 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Temperature scaling is proposed as a regularization method for multi-epoch training on small datasets and demonstrated effectiveness empirically, obtaining state-of-the-art supervised tactic generation models for Lean 3 of sizes 1.5B, 7B and 13B parameters.
     </details>


     <details>
          <summary>Abstract</summary>
          Leveraging the reasoning capabilities of large language models (LLMs) for theorem proving is a promising but challenging task because it requires in-domain finetunings on which LLMs are known to be prone to overfit. This issue is exacerbated by two properties that set theorem proving apart from more mainstream applications of LLMs: training data in formal environments like Lean or Isabelle is very scarce and evaluation benchmarks are prohibitively costly to be used extensively for hyperparameter search and model selection. In this work, we propose temperature scaling as a regularization method for multi-epoch training on small datasets. We explain its theoretical purpose heuristically and demonstrate its effectiveness empirically, obtaining state-of-the-art supervised tactic generation models for Lean 3 of sizes 1.5B, 7B and 13B parameters. Model selection based on temperature-scaled perplexity increases scores on theorem proving benchmarks by up to four percentage points. We provide detailed ablations and analyses of the proof search behaviors of the resulting models, allowing practitioners to pick optimal model sizes for their respective use cases.
     </details>

64. **LLM vs ITP** [[pdf]](https://openreview.net/forum?id=EUoe9ujR0C) `NeurIPS 2023 MATH-AI Workshop` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Wiedijk's list of 100 theorems provides a benchmark for comparing interactive theorem provers (ITPs) and their mathematics libraries. As shown by the GHOSTS dataset, large language models (LLMs) can also serve as searchable libraries of mathematics, given their capacity to ingest vast amounts of mathematical literature during their pre-training or finetuning phases. ITP libraries are the only other repositories of comparable size and range of mathematical intricacy. This paper presents the first comparison between these two unique mathematical resources, centered on Wiedijk's list. Beyond the intrinsic interest of such a comparison, we discuss the importance of analyzing whether knowledge contained in LLMs (represented by GPT-4 and Claude 2) matches that encoded in ITPs. This analysis contributes thus further to advance the intersection between LLM and ITP technology (examples being tasks like autoformalization, LLM-guided proof generation, or proof completion) by ensuring LLMs possess, beyond ITP code generation capabilities, sufficient mathematical knowledge to carry out the desired formalization. The dataset with our findings, called "LLMKnow", is made available to the public.
     </details>

65. **LLMSTEP: LLM proofstep suggestions in Lean** [[pdf]](http://arxiv.org/abs/2310.18457) `NeurIPS 2023 MATH-AI Workshop` `Lean` (12 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A baseline language model is provided, along with code for fine-tuning and evaluation to support further development, and server implementations that run on CPU, a CUDA GPU, or a Google Colab notebook are provided as a step towards fast, effective language model suggestions for any user.
     </details>


     <details>
          <summary>Abstract</summary>
          We present LLMSTEP, a tool for integrating a language model into the Lean proof assistant. LLMSTEP is a Lean 4 tactic that sends a user's proof state to a server hosting a language model. The language model generates suggestions, which are checked in Lean and displayed to a user in their development environment. We provide a baseline language model, along with code for fine-tuning and evaluation to support further development. We provide server implementations that run on CPU, a CUDA GPU, or a Google Colab notebook, as a step towards fast, effective language model suggestions for any user.
     </details>

66. **TRIGO: Benchmarking Formal Mathematical Proof Reduction for Generative Language Models** [[pdf]](http://arxiv.org/abs/2310.10180) `EMNLP 2023 Main` `Lean` (11 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes TRIGO, an ATP benchmark that not only requires a model to reduce a trigonometric expression with step-by-step proofs but also evaluates a generative LM's reasoning ability on formulas and its capability to manipulate, group, and factor number terms.
     </details>


     <details>
          <summary>Abstract</summary>
          Automated theorem proving (ATP) has become an appealing domain for exploring the reasoning ability of the recent successful generative language models. However, current ATP benchmarks are mainly focus on symbolic inference, but rarely involve the understanding of complex number combination reasoning. In this work, we propose TRIGO, an ATP benchmark that not only requires a model to reduce a trigonometric expression with step-by-step proof but also evaluates a generative LM’s reasoning ability on formulas and capability to manipulate, group, and factor number terms. We gather trigonometric expressions and their reduced forms from web, annotate the simplification process manually, and translate it into the “Lean” formal language system. We then automatically generate additional examples from the annotated samples to expand the dataset. Furthermore, we also create three automatically generated training and testing datasets of varying difficulty and distributions. Our extensive experiments show our proposed TRIGO poses a new challenge for advanced generative LM’s including GPT-4 which is pre-trained on a considerable amount of open-source formal theorem-proving language data, and provide a new tool to study the generative LM’s ability on both formal and mathematical reasoning.
     </details>

67. **MLFMF: Data Sets for Machine Learning for Mathematical Formalization** [[pdf]](http://arxiv.org/abs/2310.16005) `NeurIPS 2023` `Lean, Agda` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The MLFMF data sets provide solid benchmarking support for further investigation of the numerous machine learning approaches to formalized mathematics, and are currently the largest collection of formalized mathematical knowledge in machine learnable format.
     </details>


     <details>
          <summary>Abstract</summary>
          We introduce MLFMF, a collection of data sets for benchmarking recommendation systems used to support formalization of mathematics with proof assistants. These systems help humans identify which previous entries (theorems, constructions, datatypes, and postulates) are relevant in proving a new theorem or carrying out a new construction. Each data set is derived from a library of formalized mathematics written in proof assistants Agda or Lean. The collection includes the largest Lean 4 library Mathlib, and some of the largest Agda libraries: the standard library, the library of univalent mathematics Agda-unimath, and the TypeTopology library. Each data set represents the corresponding library in two ways: as a heterogeneous network, and as a list of s-expressions representing the syntax trees of all the entries in the library. The network contains the (modular) structure of the library and the references between entries, while the s-expressions give complete and easily parsed information about every entry.We report baseline results using standard graph and word embeddings, tree ensembles, and instance-based learning algorithms. The MLFMF data sets provide solid benchmarking support for further investigation of the numerous machine learning approaches to formalized mathematics. The methodology used to extract the networks and the s-expressions readily applies to other libraries, and is applicable to other proof assistants. With more than $250\,000$ entries in total, this is currently the largest collection of formalized mathematical knowledge in machine learnable format.
     </details>

68. **A New Approach Towards Autoformalization** [[pdf]](http://arxiv.org/abs/2310.07957) `2023-10-19` `Lean` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The task of automatically translating natural language mathematics into a formal language that can be verified by a program is broken into easier and more approachable subtasks: unlinked formalization (formalization with unlinked definitions and theorems), entity linking, and finally adjusting types so it passes the type checker.
     </details>


     <details>
          <summary>Abstract</summary>
          Verifying mathematical proofs is difficult, but can be automated with the assistance of a computer. Autoformalization is the task of automatically translating natural language mathematics into a formal language that can be verified by a program. This is a challenging task, and especially for higher-level mathematics found in research papers. Research paper mathematics requires large amounts of background and context. In this paper, we propose an avenue towards tackling autoformalization for research-level mathematics, by breaking the task into easier and more approachable subtasks: unlinked formalization (formalization with unlinked definitions and theorems), entity linking (linking to the proper theorems and definitions), and finally adjusting types so it passes the type checker. In addition, we present arXiv2Formal, a benchmark dataset for unlinked formalization consisting of 50 theorems formalized for the Lean theorem prover sampled from papers on arXiv.org. We welcome any contributions from the community to future versions of this dataset.
     </details>

69. **Llemma: An Open Language Model for Mathematics** [[pdf]](http://arxiv.org/abs/2310.10631) `ICLR 2024` `Lean, Isabelle` (164 cite) (12 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Llemma is a large language model for mathematics that outperforms all known open base models, as well as the unreleased Minerva model suite on an equi-parameter basis, and is capable of tool use and formal theorem proving without any further finetuning.
     </details>


     <details>
          <summary>Abstract</summary>
          We present Llemma, a large language model for mathematics. We continue pretraining Code Llama on the Proof-Pile-2, a mixture of scientific papers, web data containing mathematics, and mathematical code, yielding Llemma. On the MATH benchmark Llemma outperforms all known openly released models, as well as the unreleased Minerva model suite on an equi-parameter basis. Moreover, Llemma is capable of tool use and formal theorem proving without any finetuning. We openly release all artifacts, including 7 billion and 34 billion parameter models, the Proof-Pile-2, and code to replicate our experiments.
     </details>

70. **LEGO-Prover: Neural Theorem Proving with Growing Libraries** [[pdf]](http://arxiv.org/abs/2310.00656) `ICLR 2024` `Isabelle` (22 cite) (4 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work presents LEGO-Prover, which employs a growing skill library containing verified lemmas as skills to augment the capability of LLMs used in theorem proving, and advances the state-of-the-art pass rate on miniF2F-valid and miniF 2F-test.
     </details>


     <details>
          <summary>Abstract</summary>
          Despite the success of large language models (LLMs), the task of theorem proving still remains one of the hardest reasoning tasks that is far from being fully solved. Prior methods using language models have demonstrated promising results, but they still struggle to prove even middle school level theorems. One common limitation of these methods is that they assume a fixed theorem library during the whole theorem proving process. However, as we all know, creating new useful theorems or even new theories is not only helpful but crucial and necessary for advancing mathematics and proving harder and deeper results.In this work, we present LEGO-Prover, which employs a growing skill library containing verified lemmas as skills to augment the capability of LLMs used in theorem proving. By constructing the proof modularly, LEGO-Prover enables LLMs to utilize existing skills retrieved from the library and to create new skills during the proving process. These skills are further evolved (by prompting an LLM) to enrich the library on another scale. Modular and reusable skills are constantly added to the library to enable tackling increasingly intricate mathematical problems. Moreover, the learned library further bridges the gap between human proofs and formal proofs by making it easier to impute missing steps. LEGO-Prover advances the state-of-the-art pass rate on miniF2F-valid (48.0\% to 57.0\%) and miniF2F-test (45.5\% to 50.0\%). During the proving process, LEGO-Prover also generates over 20,000 skills (theorems/lemmas) and adds them to the growing library. Our ablation study indicates that these newly added skills are indeed helpful for proving theorems, resulting in a 4.9\% improvement in success rate
     </details>

71. **A Language-Agent Approach to Formal Theorem-Proving** [[pdf]](https://arxiv.org/abs/2310.04353v1) `2023-10-06` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Language agents, which use a large language model (LLM) capable of in-context learning to interact with an external environment, have recently emerged as a promising approach to control tasks. We present the first language-agent approach to formal theorem-proving. Our method, COPRA, uses a high-capacity, black-box LLM (GPT-4) as part of a policy for a stateful backtracking search. During the search, the policy can select proof tactics and retrieve lemmas and definitions from an external database. Each selected tactic is executed in the underlying proof framework, and the execution feedback is used to build the prompt for the next policy invocation. The search also tracks selected information from its history and uses it to reduce hallucinations and unnecessary LLM queries. We evaluate COPRA on the miniF2F benchmark for Lean and a set of Coq tasks from the Compcert project. On these benchmarks, COPRA is significantly better than one-shot invocations of GPT-4, as well as state-of-the-art models fine-tuned on proof data, at finding correct proofs quickly.
     </details>

72. **Lyra: Orchestrating Dual Correction in Automated Theorem Proving** [[pdf]](http://arxiv.org/abs/2309.15806) `2023-10-02` `Isabelle` (9 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The Lyra is introduced, a new framework that employs two distinct correction mechanisms: Tool Correction (TC) and Conjecture Correction (CC), an error feedback mechanism designed to interact with prover to refine formal proof conjectures with prover error messages.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) present an intriguing avenue for exploration in the field of formal theorem proving. Nevertheless, their full potential, particularly concerning the mitigation of hallucinations and refinement through prover error messages, remains an area that has yet to be thoroughly investigated. To enhance the effectiveness of LLMs in the field, we introduce the Lyra, a new framework that employs two distinct correction mechanisms: Tool Correction (TC) and Conjecture Correction (CC). To implement Tool Correction in the post-processing of formal proofs, we leverage prior knowledge to utilize predefined prover tools (e.g., Sledgehammer) for guiding the replacement of incorrect tools. Tool Correction significantly contributes to mitigating hallucinations, thereby improving the overall accuracy of the proof. In addition, we introduce Conjecture Correction, an error feedback mechanism designed to interact with prover to refine formal proof conjectures with prover error messages. Compared to the previous refinement framework, the proposed Conjecture Correction refines generation with instruction but does not collect paired (generation, error & refinement) prompts. Our method has achieved state-of-the-art (SOTA) performance on both miniF2F validation (48.0% -> 55.3%) and test (45.5% -> 51.2%). We also present 3 IMO problems solved by Lyra. We believe Tool Correction (post-process for hallucination mitigation) and Conjecture Correction (subgoal adjustment from interaction with environment) could provide a promising avenue for future research in this field.
     </details>

73. **FIMO: A Challenge Formal Dataset for Automated Theorem Proving** [[pdf]](http://arxiv.org/abs/2309.04295) `2023-09-08` `Lean` (18 cite) (6 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Through initial experiments involving GPT-4, the findings underscore the existing limitations in current methodologies, indicating a substantial journey ahead before achieving satisfactory IMO-level automated theorem proving outcomes.
     </details>


     <details>
          <summary>Abstract</summary>
          We present FIMO, an innovative dataset comprising formal mathematical problem statements sourced from the International Mathematical Olympiad (IMO) Shortlisted Problems. Designed to facilitate advanced automated theorem proving at the IMO level, FIMO is currently tailored for the Lean formal language. It comprises 149 formal problem statements, accompanied by both informal problem descriptions and their corresponding LaTeX-based informal proofs. Through initial experiments involving GPT-4, our findings underscore the existing limitations in current methodologies, indicating a substantial journey ahead before achieving satisfactory IMO-level automated theorem proving outcomes.
     </details>

74. **DT-Solver: Automated Theorem Proving with Dynamic-Tree Sampling Guided by Proof-level Value Function** [[pdf]](https://aclanthology.org/2023.acl-long.706) `ACL 2023` `Lean, Isabelle` (20 cite) (5 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel Dynamic-Tree Driven Theorem Solver (DT-Solver) is proposed, which introduces a dynamic-tree Monte-Carlo search algorithm, which dynamically allocates computing budgets for different state confidences, guided by a new proof-level value function to discover proof states that require substantial exploration.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent advances in neural theorem-proving resort to large language models and tree searches. When proving a theorem, a language model advises single-step actions based on the current proving state and the tree search finds a sequence of correct steps using actions given by the language model. However, prior works often conduct constant computation efforts for each proving state while ignoring that the hard states often need more exploration than easy states. Moreover, they evaluate and guide the proof search solely depending on the current proof state instead of considering the whole proof trajectory as human reasoning does. Here, to accommodate general theorems, we propose a novel Dynamic-Tree Driven Theorem Solver (DT-Solver) by guiding the search procedure with state confidence and proof-level values. Specifically, DT-Solver introduces a dynamic-tree Monte-Carlo search algorithm, which dynamically allocates computing budgets for different state confidences, guided by a new proof-level value function to discover proof states that require substantial exploration. Experiments on two popular theorem-proving datasets, PISA and Mathlib, show significant performance gains by our DT-Solver over the state-of-the-art approaches, with a 6.65% improvement on average in terms of success rate. And especially under low computing resource settings (11.03% improvement on average).
     </details>

75. **LeanDojo: Theorem Proving with Retrieval-Augmented Language Models** [[pdf]](http://arxiv.org/abs/2306.15626) `NeurIPS 2023` `Lean` (111 cite) (12 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduces LeanDojo: an open-source Lean playground consisting of toolkits, data, models, and benchmarks, and develops ReProver (Retrieval-Augmented Prover): an LLM-based prover augmented with retrieval for selecting premises from a vast math library.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have shown promise in proving formal theorems using proof assistants such as Lean. However, existing methods are difficult to reproduce or build on, due to private code, data, and large compute requirements. This has created substantial barriers to research on machine learning methods for theorem proving. This paper removes these barriers by introducing LeanDojo: an open-source Lean playground consisting of toolkits, data, models, and benchmarks. LeanDojo extracts data from Lean and enables interaction with the proof environment programmatically. It contains fine-grained annotations of premises in proofs, providing valuable data for premise selection—a key bottleneck in theorem proving. Using this data, we develop ReProver (Retrieval-Augmented Prover): an LLM-based prover augmented with retrieval for selecting premises from a vast math library. It is inexpensive and needs only one GPU week of training. Our retriever leverages LeanDojo's program analysis capability to identify accessible premises and hard negative examples, which makes retrieval much more effective. Furthermore, we construct a new benchmark consisting of 98,734 theorems and proofs extracted from Lean's math library. It features challenging data split requiring the prover to generalize to theorems relying on novel premises that are never used in training. We use this benchmark for training and evaluation, and experimental results demonstrate the effectiveness of ReProver over non-retrieval baselines and GPT-4. We thus provide the first set of open-source LLM-based theorem provers without any proprietary datasets and release it under a permissive MIT license to facilitate further research.
     </details>

76. **Subgoal-based Demonstration Learning for Formal Theorem Proving** [[pdf]](http://arxiv.org/abs/2305.16366) `ICML 2024` `Isabelle` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduces a subgoal-based demonstration learning framework, specifically designed to enhance the efficiency of proof search in LLMs, and builds upon recent advances in diffusion models to predict the optimal organization.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) present a promising pathway for advancing the domain of formal theorem proving. In this paper, we aim to improve the performance of LLMs in formal theorem proving by thoroughly examining the structure and organization of demonstrative in-context examples. We introduce a subgoal-based demonstration learning framework, specifically designed to enhance the efficiency of proof search in LLMs. First, drawing upon the insights of subgoal learning from reinforcement learning and robotics, we propose the construction of distinct subgoals for each demonstration example and refine these subgoals in accordance with the pertinent theories of subgoal learning. Second, we build upon recent advances in diffusion models to predict the optimal organization, simultaneously addressing two intricate issues that persist within the domain of demonstration organization: subset selection and order determination. Our integration of subgoal-based learning has notably increased proof accuracy from 38.9% to 44.1% on the miniF2F benchmark. Furthermore, the adoption of diffusion models for demonstration organization can lead to an additional enhancement in accuracy to 45.5%, or a $5\times$ improvement in sampling efficiency compared to previously established methods.
     </details>

77. **Baldur: Whole-Proof Generation and Repair with Large Language Models** [[pdf]](http://arxiv.org/abs/2303.04910) `ESEC/FSE 2023` `Isabelle` (48 cite) (5 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduces a new method to automate formal verification that uses large language models, trained on natural language and code and fine-tuned on proofs, to generate whole proofs at once and demonstrates that whole-proof generation using transformers is possible and is as effective but more efficient than search-based techniques.
     </details>


     <details>
          <summary>Abstract</summary>
          Formally verifying software properties is a highly desirable but labor-intensive task. Recent work has developed methods to automate formal verification using proof assistants, such as Coq and Isabelle/HOL, e.g., by training a model to predict one proof step at a time, and using that model to search through the space of possible proofs. This paper introduces a new method to automate formal verification: We use large language models, trained on natural language text and code and fine-tuned on proofs, to generate whole proofs for theorems at once, rather than one step at a time. We combine this proof generation model with a fine-tuned repair model to repair generated proofs, further increasing proving power. As its main contributions, this paper demonstrates for the first time that: (1) Whole-proof generation using transformers is possible and is as effective as search-based techniques without requiring costly search. (2) Giving the learned model additional context, such as a prior failed proof attempt and the ensuing error message, results in proof repair and further improves automated proof generation. (3) We establish a new state of the art for fully automated proof synthesis. We reify our method in a prototype, Baldur, and evaluate it on a benchmark of 6,336 Isabelle/HOL theorems and their proofs. In addition to empirically showing the effectiveness of whole-proof generation, repair, and added context, we show that Baldur improves on the state-of-the-art tool, Thor, by automatically generating proofs for an additional 8.7% of the theorems. Together, Baldur and Thor can prove 65.7% of the theorems fully automatically. This paper paves the way for new research into using large language models for automating formal verification.
     </details>

78. **Magnushammer: A Transformer-Based Approach to Premise Selection** [[pdf]](https://arxiv.org/abs/2303.04488v1) `ICLR 2024` `Isabelle` (26 cite) (6 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work demonstrates that contrastive training with the transformer architecture can achieve higher-quality retrieval of relevant premises, without the engineering overhead, in a novel approach to premise selection.
     </details>


     <details>
          <summary>Abstract</summary>
          This paper presents a novel approach to premise selection, a crucial reasoning task in automated theorem proving. Traditionally, symbolic methods that rely on extensive domain knowledge and engineering effort are applied to this task. In contrast, this work demonstrates that contrastive training with the transformer architecture can achieve higher-quality retrieval of relevant premises, without the knowledge or feature engineering overhead. Our method, Magnushammer, outperforms the most advanced and widely used automation tool in interactive theorem proving called Sledgehammer. On the PISA and miniF2f benchmarks Magnushammer achieves $59.5\%$ (against $38.3\%$) and $34.0\%$ (against $20.9\%$) success rates, respectively. By combining Magnushammer with a language-model-based automated theorem prover, we further improve the state-of-the-art proof success rate from $57.0\%$ to $71.0\%$ on the PISA benchmark using $4$x fewer parameters. Moreover, we develop and open source a novel dataset for premise selection, containing textual representations of (proof state, relevant premise) pairs. To the best of our knowledge, this is the largest available premise selection dataset, and the first dataset of this kind for the Isabelle proof assistant.
     </details>

79. **Aesop: White-Box Best-First Proof Search for Lean** [[pdf]](https://dl.acm.org/doi/10.1145/3573105.3575671) `2023-01-11` `Lean` (9 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Aesop performs a tree-based search over a user-specified set of proof rules and uses a best-first search strategy with customisable prioritisation, conjecture that with a fair search strategy, the algorithm is as complete as the given set of rules allows.
     </details>


     <details>
          <summary>Abstract</summary>
          We present Aesop, a proof search tactic for the Lean 4 interactive theorem prover. Aesop performs a tree-based search over a user-specified set of proof rules. It supports safe and unsafe rules and uses a best-first search strategy with customisable prioritisation. Aesop also allows users to register custom normalisation rules and integrates Lean's simplifier to support equational reasoning. Many details of Aesop's search procedure are designed to make it a white-box proof automation tactic, meaning that users should be able to easily predict how their rules will be applied, and thus how powerful and fast their Aesop invocations will be. Since we use a best-first search strategy, it is not obvious how to handle metavariables which appear in multiple goals. The most common strategy for dealing with metavariables relies on backtracking and is therefore not suitable for best-first search. We give an algorithm which addresses this issue. The algorithm works with any search strategy, is independent of the underlying logic and makes few assumptions about how rules interact with metavariables. We conjecture that with a fair search strategy, the algorithm is as complete as the given set of rules allows.
     </details>

80. **Towards Autoformalization of Mathematics and Code Correctness: Experiments with Elementary Proofs** [[pdf]](http://arxiv.org/abs/2301.02195) `2023-01-05` `Coq` (12 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A semantic parsing approach, based on the Universal Transformer architecture, that translates elementary mathematical proofs into an equivalent formalization in the language of the Coq interactive theorem prover, and generalize well to intermediate lengths not seen during training and variations in natural language.
     </details>


     <details>
          <summary>Abstract</summary>
          The ever-growing complexity of mathematical proofs makes their manual verification by mathematicians very cognitively demanding. Autoformalization seeks to address this by translating proofs written in natural language into a formal representation that is computer-verifiable via interactive theorem provers. In this paper, we introduce a semantic parsing approach, based on the Universal Transformer architecture, that translates elementary mathematical proofs into an equivalent formalization in the language of the Coq interactive theorem prover. The same architecture is also trained to translate simple imperative code decorated with Hoare triples into formally verifiable proofs of correctness in Coq. Experiments on a limited domain of artificial and human-written proofs show that the models generalize well to intermediate lengths not seen during training and variations in natural language.
     </details>

81. **Proof Repair Infrastructure for Supervised Models: Building a Large Proof Repair Dataset** [[pdf]](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ITP.2023.26) `2023-01-01` `Coq` (5 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is hoped that this report will make it easier to build datasets and benchmark suites so that machine-learning tools for proofs will move to target the tasks that matter most and do so equitably across proof assistants.
     </details>


     <details>
          <summary>Abstract</summary>
          We report on our efforts building a new, large proof-repair dataset and benchmark suite for the Coq proof assistant. The dataset is made up of Git commits from open-source projects with old and new versions of definitions and proofs aligned across commits. Building this dataset has been a significant undertaking, highlighting a number of challenges and gaps in existing infrastructure. We discuss these challenges and gaps, and we provide recommendations for how the proof assistant community can address them. Our hope is to make it easier to build datasets and benchmark suites so that machine-learning tools for proofs will move to target the tasks that matter most and do so equitably across proof assistants.
     </details>

82. **Draft, Sketch, and Prove: Guiding Formal Theorem Provers with Informal Proofs** [[pdf]](http://arxiv.org/abs/2210.12283) `ICLR 2023` `Isabelle` (99 cite) (17 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Draft, Sketch, and Prove (DSP), a method that maps informal proofs to formal proof sketches, and uses the sketches to guide an automated prover by directing its search to easier sub-problems, is introduced.
     </details>


     <details>
          <summary>Abstract</summary>
          The formalization of existing mathematical proofs is a notoriously difficult process. Despite decades of research on automation and proof assistants, writing formal proofs remains arduous and only accessible to a few experts. While previous studies to automate formalization focused on powerful search algorithms, no attempts were made to take advantage of available informal proofs. In this work, we introduce Draft, Sketch, and Prove (DSP), a method that maps informal proofs to formal proof sketches, and uses the sketches to guide an automated prover by directing its search to easier sub-problems. We investigate two relevant setups where informal proofs are either written by humans or generated by a language model. Our experiments and ablation studies show that large language models are able to produce well-structured formal sketches that follow the same reasoning steps as the informal proofs. Guiding an automated prover with these sketches enhances its performance from $20.9\%$ to $39.3\%$ on a collection of mathematical competition problems.
     </details>

83. **Diversity-Driven Automated Formal Verification** [[pdf]](https://dl.acm.org/doi/10.1145/3510003.3510138) `2022-07-05` `Coq` (22 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This study introduces a completely new idea for using diversity in machine learning to improve the power of state-of-the-art proof-script synthesis techniques, and empirically demonstrates that the improvement is significant on a dataset of 68K theorems from 122 open-source software projects.
     </details>


     <details>
          <summary>Abstract</summary>
          Formally verified correctness is one of the most desirable properties of software systems. But despite great progress made via interactive theorem provers, such as Coq, writing proof scripts for verification remains one of the most effort-intensive (and often prohibitively difficult) software development activities. Recent work has created tools that automatically synthesize proofs or proof scripts. For example, CoqHammer can prove 26.6% of theorems completely automatically by reasoning using precomputed facts, while TacTok and ASTactic, which use machine learning to model proof scripts and then perform biased search through the proof-script space, can prove 12.9% and 12.3% of the theorems, respectively. Further, these three tools are highly complementary; together, they can prove 30.4% of the theorems fully automatically. Our key insight is that control over the learning process can produce a diverse set of models, and that, due to the unique nature of proof synthesis (the existence of the theorem prover, an oracle that infallibly judges a proof's correctness), this diversity can significantly improve these tools' proving power. Accordingly, we develop Diva, which uses a diverse set of models with TacTok's and ASTactic's search mechanism to prove 21.7% of the theorems. That is, Diva proves 68% more theorems than TacTok and 77% more than ASTactic. Complementary to CoqHammer, Diva proves 781 theorems (27% added value) that CoqHammer does not, and 364 theorems no existing tool has proved automatically. Together with CoqHammer, Diva proves 33.8% of the theorems, the largest fraction to date. We explore nine dimensions for learning diverse models, and identify which dimensions lead to the most useful diversity. Further, we develop an optimization to speed up Diva's execution by 40X. Our study introduces a completely new idea for using diversity in machine learning to improve the power of state-of-the-art proof-script synthesis techniques, and empirically demonstrates that the improvement is significant on a dataset of 68K theorems from 122 open-source software projects.
     </details>

84. **HyperTree Proof Search for Neural Theorem Proving** [[pdf]](http://arxiv.org/abs/2205.11491) `NeurIPS 2022` `Lean, MetaMath` (84 cite) (18 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work shows that with HTPS alone, a model trained on annotated proofs manages to prove 65.4% of a held-out set of Metamath theorems, significantly outperforming the previous state of the art of 56.5% by GPT-f.
     </details>


     <details>
          <summary>Abstract</summary>
          We propose an online training procedure for a transformer-based automated theorem prover. Our approach leverages a new search algorithm, HyperTree Proof Search (HTPS), that learns from previous proof searches through online training, allowing it to generalize to domains far from the training distribution. We report detailed ablations of our pipeline’s main components by studying performance on three environments of increasing complexity. In particular, we show that with HTPS alone, a model trained on annotated proofs manages to prove 65.4% of a held-out set of Metamath theorems, significantly outperforming the previous state of the art of 56.5% by GPT-f. Online training on these unproved theorems increases accuracy to 82.6%. With a similar computational budget, we improve the state of the art on the Lean-based miniF2F-curriculum dataset from 31% to 42% proving accuracy.
     </details>

85. **Thor: Wielding Hammers to Integrate Language Models and Automated Theorem Provers** [[pdf]](http://arxiv.org/abs/2205.10893) `NeurIPS 2022` `Isabelle` (65 cite) (15 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Thor is introduced, a framework integrating language models and automated theorem provers to overcome the difficulty of selecting useful premises from a large library to unlock the proof of a given conjecture.
     </details>


     <details>
          <summary>Abstract</summary>
          In theorem proving, the task of selecting useful premises from a large library to unlock the proof of a given conjecture is crucially important. This presents a challenge for all theorem provers, especially the ones based on language models, due to their relative inability to reason over huge volumes of premises in text form. This paper introduces Thor, a framework integrating language models and automated theorem provers to overcome this difficulty. In Thor, a class of methods called hammers that leverage the power of automated theorem provers are used for premise selection, while all other tasks are designated to language models. Thor increases a language model's success rate on the PISA dataset from $39\%$ to $57\%$, while solving $8.2\%$ of problems neither language models nor automated theorem provers are able to solve on their own. Furthermore, with a significantly smaller computational budget, Thor can achieve a success rate on the MiniF2F dataset that is on par with the best existing methods. Thor can be instantiated for the majority of popular interactive theorem provers via a straightforward protocol we provide.
     </details>

86. **The Isabelle ENIGMA** [[pdf]](http://arxiv.org/abs/2205.01981) `2022-05-04` `Isabelle` (13 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The authors' final best single-strategy ENIGMA and premise selection system improves the best previous version of E by 25.3% in 15 seconds, outperforming also all other previous ATP and SMT systems.
     </details>


     <details>
          <summary>Abstract</summary>
          We significantly improve the performance of the E automated theorem prover on the Isabelle Sledgehammer problems by combining learning and theorem proving in several ways. In particular, we develop targeted versions of the ENIGMA guidance for the Isabelle problems, targeted versions of neural premise selection, and targeted strategies for E. The methods are trained in several iterations over hundreds of thousands untyped and typed first-order problems extracted from Isabelle. Our final best single-strategy ENIGMA and premise selection system improves the best previous version of E by 25.3% in 15 seconds, outperforming also all other previous ATP and SMT systems.
     </details>

87. **Autoformalization with Large Language Models** [[pdf]](https://arxiv.org/abs/2205.12615) `NeurIPS 2022` `Lean` (108 cite) (16 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is shown large language models provide new prospects towards the long-term goal of autoformalization, and the surprising observation that LLMs can correctly translate a significant portion of mathematical competition problems perfectly to formal specifications in Isabelle/HOL.
     </details>


     <details>
          <summary>Abstract</summary>
          Autoformalization is the process of automatically translating from natural language mathematics to formal specifications and proofs. A successful autoformalization system could advance the fields of formal verification, program synthesis, and artificial intelligence.While the long-term goal of autoformalization seemed elusive for a long time, we show large language models provide new prospects towards this goal. We make the surprising observation that LLMs can correctly translate a significant portion ($25.3\%$) of mathematical competition problems perfectly to formal specifications in Isabelle/HOL. We demonstrate the usefulness of this process by improving a previously introduced neural theorem prover via training on these autoformalized theorems. Our methodology results in a new state-of-the-art result on the MiniF2F theorem proving benchmark, improving the proof rate from~$29.6\%$ to~$35.2\%$.
     </details>

88. **Formal Mathematics Statement Curriculum Learning** [[pdf]](http://arxiv.org/abs/2202.01344) `ICLR 2023` `Lean` (0 cite) (17 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          We explore the use of expert iteration in the context of language modeling applied to formal mathematics. We show that at same compute budget, expert iteration, by which we mean proof search interleaved with learning, dramatically outperforms proof search only. We also observe that when applied to a collection of formal statements of sufficiently varied difficulty, expert iteration is capable of finding and solving a curriculum of increasingly difficult problems, without the need for associated ground-truth proofs. Finally, by applying this expert iteration to a manually curated set of problem statements, we surpass previous state-of-the-art on the miniF2F benchmark, automatically solving multiple challenging problems drawn from high school olympiads.
     </details>

89. **ProofNet: Autoformalizing and Formally Proving Undergraduate-Level Mathematics** [[pdf]](https://arxiv.org/abs/2302.12433) `NeurIPS 2022 MATH-AI Workshop` `Lean` (37 cite) (5 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces ProofNet, a benchmark for autoformalization and formal proving of undergraduate-level mathematics, and introduces two novel statement auto Formalization methods: prompt retrieval and distilled backtranslation.
     </details>


     <details>
          <summary>Abstract</summary>
          We introduce ProofNet, a benchmark for autoformalization and formal proving of undergraduate-level mathematics. The ProofNet benchmarks consists of 371 examples, each consisting of a formal theorem statement in Lean 3, a natural language theorem statement, and a natural language proof. The problems are primarily drawn from popular undergraduate pure mathematics textbooks and cover topics such as real and complex analysis, linear algebra, abstract algebra, and topology. We intend for ProofNet to be a challenging benchmark that will drive progress in autoformalization and automatic theorem proving. We report baseline results on statement autoformalization via in-context learning. Moreover, we introduce two novel statement autoformalization methods: prompt retrieval and distilled backtranslation.
     </details>

90. **Proving Theorems using Incremental Learning and Hindsight Experience Replay** [[pdf]](https://arxiv.org/abs/2112.10664) `ICML 2022` `Mizar` (11 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A general incremental learning algorithm for training domain specific provers for first-order logic without equality, based only on a basic given-clause algorithm, but using a learned clause-scoring function, and it is shown that provers trained this way can match and sometimes surpass state-of-the-art traditional provers on the TPTP dataset.
     </details>


     <details>
          <summary>Abstract</summary>
          Traditional automated theorem proving systems for first-order logic depend on speed-optimized search and many handcrafted heuristics designed to work over a wide range of domains. Machine learning approaches in the literature either depend on these traditional provers to bootstrap themselves, by leveraging these heuristics, or can struggle due to limited existing proof data. The latter issue can be explained by the lack of a smooth difficulty gradient in theorem proving datasets; large gaps in difficulty between different theorems can make training harder or even impossible. In this paper, we adapt the idea of hindsight experience replay from reinforcement learning to the automated theorem proving domain, so as to use the intermediate data generated during unsuccessful proof attempts. We build a first-order logic prover by disabling all the smart clause-scoring heuristics of the state-of-the-art E prover and replacing them with a clause-scoring neural network learned by using hindsight experience replay in an incremental learning setting. Clauses are represented as graphs and presented to transformer networks with spectral features. We show that provers trained in this way can outperform previous machine learning approaches and compete with the state of the art heuristic-based theorem prover E in its best configuration, on the popular benchmarks MPTP2078, M2k and Mizar40. The proofs generated by our algorithm are also almost always significantly shorter than E’s proofs.
     </details>

91. **Formal Premise Selection With Language Models** [[pdf]](https://www.semanticscholar.org/paper/2443179d421e1faf7474add557b45add554723c7) `2022-01-01` `Isabelle` (7 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work provides a solution to the problem of selecting a useful premise to prove a new theorem by combining a premise selection model with a language model, and shows that this retrieval-augmented prover achieves significant improvements in proof rates compared to the language model alone.
     </details>


     <details>
          <summary>Abstract</summary>
          N/A
     </details>

92. **Learned Provability Likelihood for Tactical Search** [[pdf]](http://arxiv.org/abs/2109.03234) `2021-09-06` `HOL 4` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A method to estimate the provability of a mathematical formula is presented and the tactical theorem prover TacticToe is adapted to factor in these estimations, leading to an improvement in performance and an improved user experience.
     </details>


     <details>
          <summary>Abstract</summary>
          We present a method to estimate the provability of a mathematical formula. We adapt the tactical theorem prover TacticToe to factor in these estimations. Experiments over the HOL4 library show an increase in the number of theorems re-proven by TacticToe thanks to this additional guidance. This amelioration in performance together with concurrent updates to the TacticToe framework lead to an improved user experience.
     </details>

93. **miniF2F: a cross-system benchmark for formal Olympiad-level mathematics** [[pdf]](https://arxiv.org/abs/2109.00110) `ICLR 2022` `Lean, Isabelle, HOL Light, MetaMath` (84 cite) (23 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The miniF2F benchmark currently targets Metamath, Lean, Isabelle, and HOL Light and consists of 488 problem statements drawn from the AIME, AMC, and the International Mathematical Olympiad, as well as material from high-school and undergraduate mathematics courses.
     </details>


     <details>
          <summary>Abstract</summary>
          We present $\textsf{miniF2F}$, a dataset of formal Olympiad-level mathematics problems statements intended to provide a unified cross-system benchmark for neural theorem proving. The $\textsf{miniF2F}$ benchmark currently targets Metamath, Lean, Isabelle (partially) and HOL Light (partially) and consists of 488 problem statements drawn from the AIME, AMC, and the International Mathematical Olympiad (IMO), as well as material from high-school and undergraduate mathematics courses. We report baseline results using GPT-f, a neural theorem prover based on GPT-3 and provide an analysis of its performance. We intend for $\textsf{miniF2F}$ to be a community-driven effort and hope that our benchmark will help spur advances in neural theorem proving.
     </details>

94. **Graph Contrastive Pre-training for Effective Theorem Reasoning** [[pdf]](http://arxiv.org/abs/2108.10821) `ICML 2021 SSL Workshop` `Coq` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          NeuroTactic is proposed, a novel extension with a special focus on improving the representation learning for theorem proving that leverages graph neural networks to represent the theorems and premises, and applies graph contrastive learning for pre-training.
     </details>


     <details>
          <summary>Abstract</summary>
          Interactive theorem proving is a challenging and tedious process, which requires non-trivial expertise and detailed low-level instructions (or tactics) from human experts. Tactic prediction is a natural way to automate this process. Existing methods show promising results on tactic prediction by learning a deep neural network (DNN) based model from proofs written by human experts. In this paper, we propose NeuroTactic, a novel extension with a special focus on improving the representation learning for theorem proving. NeuroTactic leverages graph neural networks (GNNs) to represent the theorems and premises, and applies graph contrastive learning for pre-training. We demonstrate that the representation learning of theorems is essential to predict tactics. Compared with other methods, NeuroTactic achieves state-of-the-art performance on the CoqGym dataset.
     </details>

95. **Learning Theorem Proving Components** [[pdf]](http://arxiv.org/abs/2107.10034) `2021-07-21` `Mizar` (7 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work describes several algorithms and experiments with ENIGMA, advancing the idea of contextual evaluation based on learning important components of the graph of clauses, and equipping the E/ENIGMA system with a graph neural network that chooses the next given clause based on its evaluation in the context of previously selected clauses.
     </details>


     <details>
          <summary>Abstract</summary>
          Saturation-style automated theorem provers (ATPs) based on the given clause procedure are today the strongest general reasoners for classical first-order logic. The clause selection heuristics in such systems are, however, often evaluating clauses in isolation, ignoring other clauses. This has changed recently by equipping the E/ENIGMA system with a graph neural network (GNN) that chooses the next given clause based on its evaluation in the context of previously selected clauses. In this work, we describe several algorithms and experiments with ENIGMA, advancing the idea of contextual evaluation based on learning important components of the graph of clauses.
     </details>

96. **Proof Artifact Co-Training for Theorem Proving with Language Models** [[pdf]](https://arxiv.org/abs/2102.06203) `ICLR 2022` `Lean` (98 cite) (27 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          PACT is proposed, a general methodology for extracting abundant self-supervised data from kernel-level proof terms for co-training alongside the usual tactic prediction objective and applied to Lean, an interactive proof assistant which hosts some of the most sophisticated formalized mathematics to date.
     </details>


     <details>
          <summary>Abstract</summary>
          Labeled data for imitation learning of theorem proving in large libraries of formalized mathematics is scarce as such libraries require years of concentrated effort by human specialists to be built. This is particularly challenging when applying large Transformer language models to tactic prediction, because the scaling of performance with respect to model size is quickly disrupted in the data-scarce, easily-overfitted regime. We propose PACT (Proof Artifact Co-Training), a general methodology for extracting abundant self-supervised data from kernel-level proof terms for joint training alongside the usual tactic prediction objective. We apply this methodology to Lean,an interactive proof assistant which hosts some of the most sophisticated formalized mathematics to date. We instrument Lean with a neural theorem prover driven by a Transformer language model and show that PACT improves theorem proving success rate on a held-out suite of test theorems from 32% to 48%.
     </details>

97. **LIME: Learning Inductive Bias for Primitives of Mathematical Reasoning** [[pdf]](https://arxiv.org/abs/2101.06223) `ICML 2021` `Isabelle, MetaMath, HOL Light, Lean` (47 cite) (8 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A new pre-training methodology called LIME (Learning Inductive bias for Mathematical rEasoning).
     </details>


     <details>
          <summary>Abstract</summary>
          While designing inductive bias in neural architectures has been widely studied, we hypothesize that transformer networks are flexible enough to learn inductive bias from suitable generic tasks. Here, we replace architecture engineering by encoding inductive bias in the form of datasets. Inspired by Peirce's view that deduction, induction, and abduction are the primitives of reasoning, we design three synthetic tasks that are intended to require the model to have these three abilities. We specifically design these tasks to be synthetic and devoid of mathematical knowledge to ensure that only the fundamental reasoning biases can be learned from these tasks. This defines a new pre-training methodology called "LIME" (Learning Inductive bias for Mathematical rEasoning). Models trained with LIME significantly outperform vanilla transformers on four very different large mathematical reasoning benchmarks. Unlike dominating the computation cost as traditional pre-training approaches, LIME requires only a small fraction of the computation cost of the typical downstream task. The code for generating LIME tasks is available at https://github.com/tonywu95/LIME.
     </details>

98. **TacticZero: Learning to Prove Theorems from Scratch with Deep Reinforcement Learning** [[pdf]](https://arxiv.org/abs/2102.09756) `NeurIPS 2021` `HOL 4` (34 cite) (7 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel approach to interactive theorem-proving (ITP) using deep reinforcement learning in which each state represents a set of potential derivation paths, and introduces a novel backtracking mechanism which enables the agent to efficiently discard dead-end derivations and restart from promising alternatives.
     </details>


     <details>
          <summary>Abstract</summary>
          We propose a novel approach to interactive theorem-proving (ITP) using deep reinforcement learning. The proposed framework is able to learn proof search strategies as well as tactic and arguments prediction in an end-to-end manner. We formulate the process of ITP as a Markov decision process (MDP) in which each state represents a set of potential derivation paths. This structure allows us to introduce a novel backtracking mechanism which enables the agent to efficiently discard (predicted) dead-end derivations and restart the derivation from promising alternatives. We implement the framework in the HOL theorem prover. Experimental results show that the framework using learned search strategies outperforms existing automated theorem provers (i.e., hammers) available in HOL when evaluated on unseen problems. We further elaborate the role of key components of the framework using ablation studies.
     </details>

99. **Proof searching and prediction in HOL4 with evolutionary/heuristic and deep learning techniques** [[pdf]](https://www.semanticscholar.org/paper/ce6b127c3db91581419bbf985b6828916586c64a) `2021-01-01` `HOL 4` (4 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Experimental results suggest that combining evolutionary/heuristic and deep learning techniques with proof assistants can greatly facilitate proof finding/optimization and proof prediction.
     </details>


     <details>
          <summary>Abstract</summary>
          N/A
     </details>

100. **REFACTOR: Learning to Extract Theorems from Proofs** [[pdf]](https://arxiv.org/abs/2402.17032) `ICLR 2024` `MetaMath` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper shows on a set of unseen proofs, REFACTOR is able to extract 19.6% of the theorems that humans would use to write the proofs, and demonstrates that the prover trained on the new-theorem refactored dataset proves more test theorems and outperforms state-of-the-art baselines by frequently leveraging a diverse set of newly extracted theorems.
     </details>


     <details>
          <summary>Abstract</summary>
          Human mathematicians are often good at recognizing modular and reusable theorems that make complex mathematical results within reach. In this paper, we propose a novel method called theoREm-from-prooF extrACTOR (REFACTOR) for training neural networks to mimic this ability in formal mathematical theorem proving. We show on a set of unseen proofs, REFACTOR is able to extract 19.6\% of the theorems that humans would use to write the proofs. When applying the model to the existing Metamath library, REFACTOR extracted 16 new theorems. With newly extracted theorems, we show that the existing proofs in the MetaMath database can be refactored. The new theorems are used very frequently after refactoring, with an average usage of 733.5 times, and help shorten the proof lengths. Lastly, we demonstrate that the prover trained on the new-theorem refactored dataset proves more test theorems and outperforms state-of-the-art baselines by frequently leveraging a diverse set of newly extracted theorems.
     </details>

101. **LISA: Language models of ISAbelle proofs** [[pdf]](None) `2021-01-01` `Isabelle` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          We introduce an environment that allows interaction with an Isabelle server in an incremental manner. With this environment, we mined the Isabelle standard library and the Archive of Formal Proofs (AFP) and extracted 183K lemmas and theorems. We built language models on this large corpus and showed their effectiveness in proving AFP theorems.
     </details>

102. **Retrieval-Augmented Proof Step Synthesis** [[pdf]](None) `2021-01-01` `HOL Light` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Automated theorem proving is relying increasingly on sophisticated language modelling approaches for synthesizing proof steps (tactic applications, rewrite rules). However, one of the most signiﬁcant diﬃculties of proof search is ﬁnding the correct premises to be used. This raises the problem of combining premise selection with language modeling. There are two obvious avenues towards this goal: synthesizing the full theorem text to be utilized as a premise, or using a separate premise selection model that is used as an extra component to be used when referencing theorems. In this paper, we suggest a new solution based on language modelling that allows premise selection to become an organic component of the deep learning model and is not trained in separation. We compare this approach to theorem proving using a combination of pretrained premise selection and tactic synthesis on the HOList dataset.
     </details>

103. **Simple Dataset for Proof Method Recommendation in Isabelle/HOL (Dataset Description)** [[pdf]](http://arxiv.org/abs/2004.10667) `2020-05-26` `Isabelle` (4 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A simple dataset that contains data on over 400k proof method applications along with over 100 extracted features for each in a format that can be processed easily without any knowledge about formal logic is presented.
     </details>


     <details>
          <summary>Abstract</summary>
          Recently, a growing number of researchers have applied machine learning to assist users of interactive theorem provers. However, the expressive nature of underlying logics and esoteric structures of proof documents impede machine learning practitioners, who often do not have much expertise in formal logic, let alone Isabelle/HOL, from achieving a large scale success in this field. In this data description, we present a simple dataset that contains data on over 400k proof method applications along with over 100 extracted features for each in a format that can be processed easily without any knowledge about formal logic. Our simple data format allows machine learning practitioners to try machine learning tools to predict proof methods in Isabelle/HOL without requiring domain expertise in logic.
     </details>

104. **IsarStep: a Benchmark for High-level Mathematical Reasoning** [[pdf]](https://arxiv.org/abs/2006.09265) `ICLR 2021` `Isabelle` (52 cite) (17 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A benchmark for high-level mathematical reasoning is presented and the reasoning capabilities of neural sequence-to-sequence models are studied and a hierarchical transformer is designed that outperforms the transformer baseline.
     </details>


     <details>
          <summary>Abstract</summary>
          A well-defined benchmark is essential for measuring and accelerating research progress of machine learning models. In this paper, we present a benchmark for high-level mathematical reasoning and study the reasoning capabilities of neural sequence-to-sequence models. We build a non-synthetic dataset from the largest repository of proofs written by human experts in a theorem prover. The dataset has a broad coverage of undergraduate and research-level mathematical and computer science theorems. In our defined task, a model is required to fill in a missing intermediate proposition given surrounding proofs. This task provides a starting point for the long-term goal of having machines generate human-readable proofs automatically. Our experiments and analysis reveal that while the task is challenging, neural models can capture non-trivial mathematical reasoning. We further design a hierarchical transformer that outperforms the transformer baseline.
     </details>

105. **Learning to Prove Theorems by Learning to Generate Theorems** [[pdf]](https://arxiv.org/abs/2002.07019) `NeurIPS 2020` `Holophrasm` (41 cite) (13 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes to learn a neural generator that automatically synthesizes theorems and proofs for the purpose of training a theorem prover, and demonstrates that synthetic data from this approach improves the theorem provers and advances the state of the art of automated theorem proving in Metamath.
     </details>


     <details>
          <summary>Abstract</summary>
          We consider the task of automated theorem proving, a key AI task. Deep learning has shown promise for training theorem provers, but there are limited human-written theorems and proofs available for supervised learning. To address this limitation, we propose to learn a neural generator that automatically synthesizes theorems and proofs for the purpose of training a theorem prover. Experiments on real-world tasks demonstrate that synthetic data from our approach improves the theorem prover and advances the state of the art of automated theorem proving in Metamath. Code is available at https://github.com/princeton-vl/MetaGen.
     </details>

106. **Reinforcement Learning for Interactive Theorem Proving in HOL4** [[pdf]](None) `2020-01-01` `HOL 4` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          N/A
     </details>

107. **Mathematical Reasoning in Latent Space** [[pdf]](http://arxiv.org/abs/1909.11851) `ICLR 2020` `HOL Light` (33 cite) (3 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The experiments show that graph neural networks can make non-trivial predictions about the rewrite-success of statements, even when they propagate predicted latent representations for several steps, a strong indicator for the feasibility of deduction in latent space in general.
     </details>


     <details>
          <summary>Abstract</summary>
          We design and conduct a simple experiment to study whether neural networks can perform several steps of approximate reasoning in a fixed dimensional latent space. The set of rewrites (i.e. transformations) that can be successfully performed on a statement represents essential semantic features of the statement. We can compress this information by embedding the formula in a vector space, such that the vector associated with a statement can be used to predict whether a statement can be rewritten by other theorems. Predicting the embedding of a formula generated by some rewrite rule is naturally viewed as approximate reasoning in the latent space. In order to measure the effectiveness of this reasoning, we perform approximate deduction sequences in the latent space and use the resulting embedding to inform the semantic features of the corresponding formal statement (which is obtained by performing the corresponding rewrite sequence using real formulas). Our experiments show that graph neural networks can make non-trivial predictions about the rewrite-success of statements, even when they propagate predicted latent representations for several steps. Since our corpus of mathematical formulas includes a wide variety of mathematical disciplines, this experiment is a strong indicator for the feasibility of deduction in latent space in general.
     </details>

108. **Learning to Prove Theorems via Interacting with Proof Assistants** [[pdf]](https://www.semanticscholar.org/paper/a596f03145285cd05a6ca57a4e25418b23b24976) `ICML 2019` `Coq` (117 cite) (22 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          ASTactic, a deep learning-based model that generates tactics as programs in the form of abstract syntax trees (ASTs) can generate effective tactics and can be used to prove new theorems not previously provable by automated methods.
     </details>


     <details>
          <summary>Abstract</summary>
          N/A
     </details>

109. **HOList: An Environment for Machine Learning of Higher-Order Theorem Proving (extended version)** [[pdf]](https://www.semanticscholar.org/paper/9ef2e09a9e16e176e19c3fdc3b6ee22c5d3f3c97) `ICML 2019` `HOL Light` (45 cite) (7 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work provides an open-source framework based on the HOL Light theorem prover that can be used as a reinforcement learning environment and presents a deep reinforcement learning driven automated theorem provers, DeepHOL, with strong initial results on this benchmark.
     </details>


     <details>
          <summary>Abstract</summary>
          N/A
     </details>

110. **Large Scale Deep Learning for Theorem Proving in HOList: First Results and Future Directions** [[pdf]](None) `2019-01-01` `HOL Light` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          N/A
     </details>

111. **Reinforcement Learning of Theorem Proving** [[pdf]](https://www.semanticscholar.org/paper/920febb03475b068286a855c10ea09b968fe7ee3) `NeurIPS 2018` `Mizar` (135 cite) (13 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A theorem proving algorithm that uses practically no domain heuristics for guiding its connection-style proof search and solves within the same number of inferences over 40% more problems than a baseline prover, which is an unusually high improvement in this hard AI domain.
     </details>


     <details>
          <summary>Abstract</summary>
          N/A
     </details>

112. **GamePad: A Learning Environment for Theorem Proving** [[pdf]](https://www.semanticscholar.org/paper/87c425f23bcac2f082968abda64a971f91522f73) `ICLR 2019` `Coq` (97 cite) (16 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A system called GamePad is introduced that can be used to explore the application of machine learning methods to theorem proving in the Coq proof assistant and addresses position evaluation and tactic prediction tasks, which arise naturally in tactic-based theorem proving.
     </details>


     <details>
          <summary>Abstract</summary>
          N/A
     </details>

113. **HolStep: A Machine Learning Dataset for Higher-order Logic Theorem Proving** [[pdf]](http://arxiv.org/abs/1703.00426) `ICLR 2017` `HOL Light` (79 cite) (10 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A new dataset based on Higher-Order Logic (HOL) proofs is introduced, for the purpose of developing new machine learning-based theorem-proving strategies and the results of these models show the promise of applying machine learning to HOL theorem proving.
     </details>


     <details>
          <summary>Abstract</summary>
          Large computer-understandable proofs consist of millions of intermediate logical steps. The vast majority of such steps originate from manually selected and manually guided heuristics applied to intermediate goals. So far, machine learning has generally not been used to filter or generate these steps. In this paper, we introduce a new dataset based on Higher-Order Logic (HOL) proofs, for the purpose of developing new machine learning-based theorem-proving strategies. We make this dataset publicly available under the BSD license. We propose various machine learning tasks that can be performed on this dataset, and discuss their significance for theorem proving. We also benchmark a set of simple baseline machine learning models suited for the tasks (including logistic regression, convolutional neural networks and recurrent neural networks). The results of our baseline models show the promise of applying machine learning to HOL theorem proving.
     </details>

114. **DeepMath - Deep Sequence Models for Premise Selection** [[pdf]](http://arxiv.org/abs/1606.04442) `NeurIPS 2016` `Mizar` (212 cite) (26 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A two stage approach is proposed that yields good results for the premise selection task on the Mizar corpus while avoiding the hand-engineered features of existing state-of-the-art models.
     </details>


     <details>
          <summary>Abstract</summary>
          We study the effectiveness of neural sequence models for premise selection in automated theorem proving, one of the main bottlenecks in the formalization of mathematics. We propose a two stage approach for this task that yields good results for the premise selection task on the Mizar corpus while avoiding the hand-engineered features of existing state-of-the-art models. To our knowledge, this is the first time deep learning has been applied to theorem proving on a large scale.
     </details>

115. **Tree-Structure CNN for Automated Theorem Proving** [[pdf]](https://www.semanticscholar.org/paper/e439d9c607fc7f97b410f9912b99548ca5cbc41c) `2017-01-01` `HOL Light` (4 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper designs a tree-structure CNN, involving bidirectional LSTM, and improves accuracy significantly, reaching 90% accuracy on the test set.
     </details>


     <details>
          <summary>Abstract</summary>
          The most difficult and heavy work of Automated Theorem Proving (ATP) is that people should search in millions of intermediate steps to finish proof. In this paper, we present a novel neural network, which can effectively help people to finish this work. Specifically, we design a tree-structure CNN, involving bidirectional LSTM. We compare our model with other neural network models and make experiments on HOLStep dataset, which is a machine learning dataset for Higher-order logic theorem proving. Being compared to previous approaches, our model improves accuracy significantly, reaching 90% accuracy on the test set.
     </details>

