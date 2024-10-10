# Papers Using Proof Assistants 

This is an incomplete list primarily focused on new papers in 2024 + older papers in top conferences.

1. **SubgoalXL: Subgoal-based Expert Learning for Theorem Proving** [[pdf]](https://arxiv.org/abs/2408.11172v1) `2024-08-20` `Isabelle` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          SubgoalXL is introduced, a novel approach that synergizes subgoal-based proofs with expert learning to enhance LLMs' capabilities in formal theorem proving within the Isabelle environment, and achieves a new state-of-the-art performance.
     </details>


     <details>
          <summary>Abstract</summary>
          Formal theorem proving, a field at the intersection of mathematics and computer science, has seen renewed interest with advancements in large language models (LLMs). This paper introduces SubgoalXL, a novel approach that synergizes subgoal-based proofs with expert learning to enhance LLMs' capabilities in formal theorem proving within the Isabelle environment. SubgoalXL addresses two critical challenges: the scarcity of specialized mathematics and theorem-proving data, and the need for improved multi-step reasoning abilities in LLMs. By optimizing data efficiency and employing subgoal-level supervision, SubgoalXL extracts richer information from limited human-generated proofs. The framework integrates subgoal-oriented proof strategies with an expert learning system, iteratively refining formal statement, proof, and subgoal generators. Leveraging the Isabelle environment's advantages in subgoal-based proofs, SubgoalXL achieves a new state-of-the-art performance of 56.1\% in Isabelle on the standard miniF2F dataset, marking an absolute improvement of 4.9\%. Notably, SubgoalXL successfully solves 41 AMC12, 9 AIME, and 3 IMO problems from miniF2F. These results underscore the effectiveness of maximizing limited data utility and employing targeted guidance for complex reasoning in formal theorem proving, contributing to the ongoing advancement of AI reasoning capabilities. The implementation is available at \url{https://github.com/zhaoxlpku/SubgoalXL}.
     </details>

2. **DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for Reinforcement Learning and Monte-Carlo Tree Search** [[pdf]](http://arxiv.org/abs/2408.08152) `2024-08-15` `Lean` (7 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          An open-source language model designed for theorem proving in Lean 4, which enhances DeepSeek-Prover-V1 by optimizing both training and inference processes, and proposes RMaxTS, a variant of Monte-Carlo tree search that employs an intrinsic-reward-driven exploration strategy to generate diverse proof paths.
     </details>


     <details>
          <summary>Abstract</summary>
          We introduce DeepSeek-Prover-V1.5, an open-source language model designed for theorem proving in Lean 4, which enhances DeepSeek-Prover-V1 by optimizing both training and inference processes. Pre-trained on DeepSeekMath-Base with specialization in formal mathematical languages, the model undergoes supervised fine-tuning using an enhanced formal theorem proving dataset derived from DeepSeek-Prover-V1. Further refinement is achieved through reinforcement learning from proof assistant feedback (RLPAF). Beyond the single-pass whole-proof generation approach of DeepSeek-Prover-V1, we propose RMaxTS, a variant of Monte-Carlo tree search that employs an intrinsic-reward-driven exploration strategy to generate diverse proof paths. DeepSeek-Prover-V1.5 demonstrates significant improvements over DeepSeek-Prover-V1, achieving new state-of-the-art results on the test set of the high school level miniF2F benchmark ($63.5\%$) and the undergraduate level ProofNet benchmark ($25.3\%$).
     </details>

3. **miniCTX: Neural Theorem Proving with (Long-)Contexts** [[pdf]](https://arxiv.org/abs/2408.03350v1) `2024-08-05` `Lean` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces miniCTX, which tests a model's ability to prove formal mathematical theorems that depend on new context that is not seen during training, and offers ntp-toolkit for automatically extracting and annotating theorem proving data.
     </details>


     <details>
          <summary>Abstract</summary>
          We introduce miniCTX, which tests a model's ability to prove formal mathematical theorems that depend on new definitions, lemmas, or other contextual information that was not observed during training. miniCTX contains theorems sourced from real Lean projects and textbooks, each associated with a context that can span tens of thousands of tokens. Models are tasked with proving a theorem given access to code from the theorem's repository, which contains context that is helpful or needed for the proof. As a baseline for miniCTX, we introduce file-tuning, a simple recipe that trains a model to generate a proof step conditioned on the preceding file contents. File-tuning substantially outperforms the traditional neural theorem proving approach that fine-tunes on states alone. Additionally, our file-tuned model improves performance on the standard miniF2F benchmark, achieving a pass rate of 33.61%, which is a new state-of-the-art for 1.3B parameter models. Alongside miniCTX, we offer ntp-toolkit for automatically extracting and annotating theorem proving data, making it easy to add new projects into miniCTX to ensure that contexts are not seen during training. miniCTX offers a challenging and realistic perspective on evaluating neural theorem provers.
     </details>

4. **LEAN-GitHub: Compiling GitHub LEAN repositories for a versatile LEAN prover** [[pdf]](http://arxiv.org/abs/2407.17227) `2024-07-24` `Lean` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes LEAN-GitHub, a dataset consisting of large-scale formal data extracted from almost all Lean 4 repositories on GitHub, and achieves state-of-the-art on two other Lean 4 benchmarks targeting different fields/levels of math.
     </details>


     <details>
          <summary>Abstract</summary>
          Recently, large language models have presented promising results in aiding formal mathematical reasoning. However, their performance is restricted due to the scarcity of formal theorem-proving data, which requires additional effort to be extracted from raw formal language corpora. Meanwhile, a significant amount of human-written formal language corpora remains underutilized. To address this issue, we propose LEAN-GitHub, a dataset consisting of large-scale formal data extracted from almost all Lean 4 repositories on GitHub. After fine-tuning InternLM-math-plus on this dataset, our model achieved accuracies of 48.8% with a single pass and 54.5% with 64 passes on the Lean 4 miniF2F test, surpassing state-of-the-art method at 52%. And it also achieves state-of-the-art on two other Lean 4 benchmarks (ProofNet and Putnam) targeting different fields/levels of math. These results demonstrate that our proposed dataset is beneficial for formal reasoning on a wide range of math topics. We open-source our model at https://GitHub. com/InternLM/InternLM-Math and our data at https://huggingface.co/ datasets/InternLM/Lean-GitHub
     </details>

5. **Reliable Reasoning Beyond Natural Language** [[pdf]](http://arxiv.org/abs/2407.11373) `2024-07-19` `Prolog` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a neurosymbolic approach that prompts LLMs to extract and encode all relevant information from a problem statement as logical code statements, and then uses a logic programming language (Prolog) to conduct the iterative computations of explicit deductive reasoning.
     </details>


     <details>
          <summary>Abstract</summary>
          Despite their linguistic competence, Large Language models (LLMs) often exhibit limitations in their ability to reason reliably and flexibly. To address this, we propose a neurosymbolic approach that prompts LLMs to extract and encode all relevant information from a problem statement as logical code statements, and then use a logic programming language (Prolog) to conduct the iterative computations of explicit deductive reasoning. Our approach significantly enhances the performance of LLMs on the standard mathematical reasoning benchmark, GSM8k, and the Navigate dataset from the BIG-bench dataset. Additionally, we introduce a novel dataset, the Non-Linear Reasoning (NLR) dataset, consisting of 55 unique word problems that target the shortcomings of the next token prediction paradigm of LLMs and require complex non-linear reasoning but only basic arithmetic skills to solve. Our findings demonstrate that the integration of Prolog enables LLMs to achieve high performance on the NLR dataset, which even the most advanced language models (including GPT4) fail to solve using text only.
     </details>

6. **PutnamBench: Evaluating Neural Theorem-Provers on the Putnam Mathematical Competition** [[pdf]](http://arxiv.org/abs/2407.11214) `NeurIPS 2024` `Lean, Isabelle, Coq` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          PutnamBench consists of 1697 hand-constructed formalizations of 640 theorems sourced from the William Lowell Putnam Mathematical Competition, the premier undergraduate-level mathematics competition in North America.
     </details>


     <details>
          <summary>Abstract</summary>
          We present PutnamBench, a new multilingual benchmark for evaluating the ability of neural theorem-provers to solve competition mathematics problems. PutnamBench consists of 1337 hand-constructed formalizations of 514 theorems sourced from the William Lowell Putnam Mathematical Competition, the premier undergraduate-level mathematics competition in North America.  All the theorems have formalizations in Lean 4 and Isabelle; a substantial subset also has Coq formalizations. Proving the theorems requires significant problem-solving ability and proficiency in a broad range of topics taught in undergraduate mathematics courses. We use PutnamBench to evaluate several established neural and symbolic theorem-provers.  These approaches can only solve a handful of the PutnamBench problems, establishing the benchmark as a difficult open challenge for research on neural theorem-proving. PutnamBench is available at https://github.com/trishullab/PUTNAM.
     </details>

7. **TheoremLlama: Transforming General-Purpose LLMs into Lean4 Experts** [[pdf]](http://arxiv.org/abs/2407.03203) `2024-07-03` `Lean` (2 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper proposes TheoremLlama, an end-to-end framework that trains a general-purpose LLM to be a Lean4 expert, and provides Open Bootstrapped Theorems (OBT), an NL-FL aligned and bootstrapped dataset.
     </details>


     <details>
          <summary>Abstract</summary>
          Proving mathematical theorems using computer-verifiable formal languages like Lean significantly impacts mathematical reasoning. One approach to formal theorem proving involves generating complete proofs using Large Language Models (LLMs) based on Natural Language (NL) proofs. Similar methods have shown promising results in code generation. However, most modern LLMs exhibit suboptimal performance due to the scarcity of aligned NL and Formal Language (FL) theorem-proving data. This scarcity results in a paucity of methodologies for training LLMs and techniques to fully utilize their capabilities in composing formal proofs. To address the challenges, this paper proposes **TheoremLlama**, an end-to-end framework to train a general-purpose LLM to become a Lean4 expert. This framework encompasses NL-FL aligned dataset generation methods, training approaches for the LLM formal theorem prover, and techniques for LLM Lean4 proof writing. Using the dataset generation method, we provide *Open Bootstrapped Theorems* (OBT), an NL-FL aligned and bootstrapped dataset. A key innovation in this framework is the NL-FL bootstrapping method, where NL proofs are integrated into Lean4 code for training datasets, leveraging the NL reasoning ability of LLMs for formal reasoning. The **TheoremLlama** framework achieves cumulative accuracies of 36.48% and 33.61% on MiniF2F-Valid and Test datasets respectively, surpassing the GPT-4 baseline of 22.95% and 25.41%. We have also open-sourced our model checkpoints and generated dataset, and will soon make all the code publicly available.
     </details>

8. **Improving Autoformalization using Type Checking** [[pdf]](http://arxiv.org/abs/2406.07222) `2024-06-11` `Lean` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a method to fix the performance of large language models for autoformalization through decoding with type-check filtering, where they initially sample a diverse set of candidate formalizations for an informal statement, then use the Lean proof assistant to filter out candidates that do not type-check.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models show promise for autoformalization, the task of automatically translating natural language into formal languages. However, current autoformalization methods remain limited. The last reported state-of-the-art performance on the ProofNet formalization benchmark for the Lean proof assistant, achieved using Codex for Lean 3, only showed successful formalization of 16.1% of informal statements. Similarly, our evaluation of GPT-4o for Lean 4 only produces successful translations 34.9% of the time. Our analysis shows that the performance of these models is largely limited by their inability to generate formal statements that successfully type-check (i.e., are syntactically correct and consistent with types) - with a whopping 86.6% of GPT-4o errors starting from a type-check failure. In this work, we propose a method to fix this issue through decoding with type-check filtering, where we initially sample a diverse set of candidate formalizations for an informal statement, then use the Lean proof assistant to filter out candidates that do not type-check. Using GPT-4o as a base model, and combining our method with self-consistency, we obtain a +18.3% absolute increase in formalization accuracy, and achieve a new state-of-the-art of 53.2% on ProofNet with Lean 4.
     </details>

9. **Lean Workbook: A large-scale Lean problem set formalized from natural language math problems** [[pdf]](http://arxiv.org/abs/2406.03847) `NeurIPS 2024` `Lean` (4 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes a novel pipeline that iteratively generates and filters synthetic data to translate natural language mathematical problems into Lean 4 statements, and vice versa, and indicates that the synthetic data pipeline can provide useful training data and improve the performance of LLMs in translating and understanding complex mathematical problems and proofs.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models have demonstrated impressive capabilities across various natural language processing tasks, especially in solving mathematical problems. However, large language models are not good at math theorem proving using formal languages like Lean. A significant challenge in this area is the scarcity of training data available in these formal languages. To address this issue, we propose a novel pipeline that iteratively generates and filters synthetic data to translate natural language mathematical problems into Lean 4 statements, and vice versa. Our results indicate that the synthetic data pipeline can provide useful training data and improve the performance of LLMs in translating and understanding complex mathematical problems and proofs. Our final dataset contains about 57K formal-informal question pairs along with searched proof from the math contest forum and 21 new IMO questions. We open-source our code at \url{https://github.com/InternLM/InternLM-Math} and our data at \url{https://huggingface.co/datasets/InternLM/Lean-Workbook}.
     </details>

10. **An Evaluation Benchmark for Autoformalization in Lean4** [[pdf]](http://arxiv.org/abs/2406.06555) `2024-06-01` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel evaluation benchmark designed for Lean4 is introduced, applying it to test the abilities of state-of-the-art LLMs, including GPT-3.5, GPT-4, and Gemini Pro, revealing that these LLMs still exhibit limitations in autoformalization, particularly in more complex areas of mathematics.
     </details>


     <details>
          <summary>Abstract</summary>
          Large Language Models (LLMs) hold the potential to revolutionize autoformalization. The introduction of Lean4, a mathematical programming language, presents an unprecedented opportunity to rigorously assess the autoformalization capabilities of LLMs. This paper introduces a novel evaluation benchmark designed for Lean4, applying it to test the abilities of state-of-the-art LLMs, including GPT-3.5, GPT-4, and Gemini Pro. Our comprehensive analysis reveals that, despite recent advancements, these LLMs still exhibit limitations in autoformalization, particularly in more complex areas of mathematics. These findings underscore the need for further development in LLMs to fully harness their potential in scientific research and development. This study not only benchmarks current LLM capabilities but also sets the stage for future enhancements in autoformalization.
     </details>

11. **Autoformalizing Euclidean Geometry** [[pdf]](http://arxiv.org/abs/2405.17216) `ICML 2024` `Lean` (2 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A neuro-symbolic framework for autoformalizing Euclidean geometry, which combines domain knowledge, SMT solvers, and large language models (LLMs), is introduced and automatic semantic evaluation for autoformalized theorem statements is provided.
     </details>


     <details>
          <summary>Abstract</summary>
          Autoformalization involves automatically translating informal math into formal theorems and proofs that are machine-verifiable. Euclidean geometry provides an interesting and controllable domain for studying autoformalization. In this paper, we introduce a neuro-symbolic framework for autoformalizing Euclidean geometry, which combines domain knowledge, SMT solvers, and large language models (LLMs). One challenge in Euclidean geometry is that informal proofs rely on diagrams, leaving gaps in texts that are hard to formalize. To address this issue, we use theorem provers to fill in such diagrammatic information automatically, so that the LLM only needs to autoformalize the explicit textual steps, making it easier for the model. We also provide automatic semantic evaluation for autoformalized theorem statements. We construct LeanEuclid, an autoformalization benchmark consisting of problems from Euclid's Elements and the UniGeo dataset formalized in the Lean proof assistant. Experiments with GPT-4 and GPT-4V show the capability and limitations of state-of-the-art LLMs on autoformalizing geometry problems. The data and code are available at https://github.com/loganrjmurphy/LeanEuclid.
     </details>

12. **DeepSeek-Prover: Advancing Theorem Proving in LLMs through Large-Scale Synthetic Data** [[pdf]](http://arxiv.org/abs/2405.14333) `2024-05-23` `Lean` (10 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces an approach to generate extensive Lean 4 proof data derived from high-school and undergraduate-level mathematical competition problems to enhance theorem-proving capabilities in LLMs and demonstrates the potential of leveraging large-scale synthetic data to enhance theorem-proving capabilities in LLMs.
     </details>


     <details>
          <summary>Abstract</summary>
          Proof assistants like Lean have revolutionized mathematical proof verification, ensuring high accuracy and reliability. Although large language models (LLMs) show promise in mathematical reasoning, their advancement in formal theorem proving is hindered by a lack of training data. To address this issue, we introduce an approach to generate extensive Lean 4 proof data derived from high-school and undergraduate-level mathematical competition problems. This approach involves translating natural language problems into formal statements, filtering out low-quality statements, and generating proofs to create synthetic data. After fine-tuning the DeepSeekMath 7B model on this synthetic dataset, which comprises 8 million formal statements with proofs, our model achieved whole-proof generation accuracies of 46.3% with 64 samples and 52% cumulatively on the Lean 4 miniF2F test, surpassing the baseline GPT-4 at 23.0% with 64 samples and a tree search reinforcement learning method at 41.0%. Additionally, our model successfully proved 5 out of 148 problems in the Lean 4 Formalized International Mathematical Olympiad (FIMO) benchmark, while GPT-4 failed to prove any. These results demonstrate the potential of leveraging large-scale synthetic data to enhance theorem-proving capabilities in LLMs. Both the synthetic dataset and the model will be made available to facilitate further research in this promising field.
     </details>

13. **Proving Theorems Recursively** [[pdf]](http://arxiv.org/abs/2405.14414) `NeurIPS 2024` `Isabelle` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          POETRY (PrOvE Theorems RecursivelY), which proves theorems in a recursive, level-by-level manner in the Isabelle theorem prover, is proposed, which allows the theorem to be tackled incrementally by outlining the overall theorem at the first level and then solving the intermediate conjectures at deeper levels.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent advances in automated theorem proving leverages language models to explore expanded search spaces by step-by-step proof generation. However, such approaches are usually based on short-sighted heuristics (e.g., log probability or value function scores) that potentially lead to suboptimal or even distracting subgoals, preventing us from finding longer proofs. To address this challenge, we propose POETRY (PrOvE Theorems RecursivelY), which proves theorems in a recursive, level-by-level manner in the Isabelle theorem prover. Unlike previous step-by-step methods, POETRY searches for a verifiable sketch of the proof at each level and focuses on solving the current level's theorem or conjecture. Detailed proofs of intermediate conjectures within the sketch are temporarily replaced by a placeholder tactic called sorry, deferring their proofs to subsequent levels. This approach allows the theorem to be tackled incrementally by outlining the overall theorem at the first level and then solving the intermediate conjectures at deeper levels. Experiments are conducted on the miniF2F and PISA datasets and significant performance gains are observed in our POETRY approach over state-of-the-art methods. POETRY on miniF2F achieves an average proving success rate improvement of 5.1%. Moreover, we observe a substantial increase in the maximum proof length found by POETRY, from 10 to 26.
     </details>

14. **ATG: Benchmarking Automated Theorem Generation for Generative Language Models** [[pdf]](http://arxiv.org/abs/2405.06677) `NAACL 2024 Findings` `MetaMath` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          An Automated Theorem Generation (ATG) benchmark that evaluates whether an agent can automatically generate valuable (and possibly brand new) theorems that are applicable for downstream theorem proving as reusable knowledge is proposed.
     </details>


     <details>
          <summary>Abstract</summary>
          Humans can develop new theorems to explore broader and more complex mathematical results.While current generative language models (LMs) have achieved significant improvement in automatically proving theorems, their ability to generate new or reusable theorems is still under-explored. Without the new theorems, current LMs struggle to prove harder theorems that are distant from the given hypotheses with the exponentially growing search space.More advanced theorem proving is if an agent (for instance, a generative LM) can leverage its creativity to generate new but also reasonable theorems that properly substitute part of a proof and also be saved as reusable knowledge for future theorem proving.Therefore, this paper proposes an Automated Theorem Generation (ATG) benchmark that evaluates whether an agent can automatically generate valuable (and possibly brand new) theorems that are applicable for downstream theorem proving as reusable knowledge. Specifically, we construct the ATG benchmark by splitting the Metamath library into three sets: axioms, library, and problem based on their proving depth.We conduct extensive experiments to investigate whether current LMs can generate theorems in the library and benefit the problem theorems proving. The results demonstrate that high-quality ATG data facilitates models’ performances on downstream ATP. However, there is still room for current LMs to develop better ATG and generate more advanced and human-like theorems. We hope the new ATG challenge can shed some light on advanced complex theorem proving.
     </details>

15. **Learn from Failure: Fine-tuning LLMs with Trial-and-Error Data for Intuitionistic Propositional Logic Proving** [[pdf]](http://arxiv.org/abs/2404.07382) `ACL 2024 Long Papers` `Lean` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Facing the lack of such trial-and-error data in existing open-source theorem-proving datasets, this paper curate a dataset on intuitionistic propositional logic theorems and formalize it in Lean, such that it can reliably check the correctness of proofs.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent advances in Automated Theorem Proving have shown the effectiveness of leveraging a (large) language model that generates tactics (i.e. proof steps) to search through proof states. The current model, while trained solely on successful proof paths, faces a discrepancy at the inference stage, as it must sample and try various tactics at each proof state until finding success, unlike its training which does not incorporate learning from failed attempts. Intuitively, a tactic that leads to a failed search path would indicate that similar tactics should receive less attention during the following trials. In this paper, we demonstrate the benefit of training models that additionally learn from failed search paths. Facing the lack of such trial-and-error data in existing open-source theorem-proving datasets, we curate a dataset on intuitionistic propositional logic theorems and formalize it in Lean, such that we can reliably check the correctness of proofs. We compare our model trained on relatively short trial-and-error information (TrialMaster) with models trained only on the correct paths and discover that the former solves more unseen theorems with lower trial searches.
     </details>

16. **Enhancing Formal Theorem Proving: A Comprehensive Dataset for Training AI Models on Coq Code** [[pdf]](http://arxiv.org/abs/2403.12627) `2024-04-02` `Coq` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper presents a comprehensive dataset specifically designed to enhance LLMs' proficiency in interpreting and generating Coq code, and discusses the dataset's composition, the methodology behind its creation, and the implications for the future of machine learning in formal verification.
     </details>


     <details>
          <summary>Abstract</summary>
          In the realm of formal theorem proving, the Coq proof assistant stands out for its rigorous approach to verifying mathematical assertions and software correctness. Despite the advances in artificial intelligence and machine learning, the specialized nature of Coq syntax and semantics poses unique challenges for Large Language Models (LLMs). Addressing this gap, we present a comprehensive dataset specifically designed to enhance LLMs' proficiency in interpreting and generating Coq code. This dataset, derived from a collection of over 10,000 Coq source files, encompasses a wide array of propositions, proofs, and definitions, enriched with metadata including source references and licensing information. Our primary aim is to facilitate the development of LLMs capable of generating syntactically correct and semantically meaningful Coq constructs, thereby advancing the frontier of automated theorem proving. Initial experiments with this dataset have showcased its significant potential; models trained on this data exhibited enhanced accuracy in Coq code generation. Notably, a particular experiment revealed that a fine-tuned LLM was capable of generating 141 valid proofs for a basic lemma, highlighting the dataset's utility in facilitating the discovery of diverse and valid proof strategies. This paper discusses the dataset's composition, the methodology behind its creation, and the implications of our findings for the future of machine learning in formal verification. The dataset is accessible for further research and exploration: https://huggingface.co/datasets/florath/coq-facts-props-proofs-gen0-v1
     </details>

17. **GFLean: An Autoformalisation Framework for Lean via GF** [[pdf]](http://arxiv.org/abs/2404.01234) `2024-04-01` `Lean` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          An autoformalisation framework for the Lean theorem prover, called GFLean, which uses a high-level grammar writing tool called Grammatical Framework for parsing and linearisation and is implemented in Haskell.
     </details>


     <details>
          <summary>Abstract</summary>
          We present an autoformalisation framework for the Lean theorem prover, called GFLean. GFLean uses a high-level grammar writing tool called Grammatical Framework (GF) for parsing and linearisation. GFLean is implemented in Haskell. We explain the functionalities of GFLean, its inner working and discuss its limitations. We also discuss how we can use neural network based translation programs and rule based translation programs together complimenting each other to build robust autoformalisation frameworks.
     </details>

18. **Don't Trust: Verify -- Grounding LLM Quantitative Reasoning with Autoformalization** [[pdf]](http://arxiv.org/abs/2403.18120) `ICLR 2024` `Isabelle` (0 cite) (0 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          Large language models (LLM), such as Google's Minerva and OpenAI's GPT families, are becoming increasingly capable of solving mathematical quantitative reasoning problems. However, they still make unjustified logical and computational errors in their reasoning steps and answers. In this paper, we leverage the fact that if the training corpus of LLMs contained sufficiently many examples of formal mathematics (e.g. in Isabelle, a formal theorem proving environment), they can be prompted to translate i.e. autoformalize informal mathematical statements into formal Isabelle code --- which can be verified automatically for internal consistency. This provides a mechanism to automatically reject solutions whose formalized versions are inconsistent within themselves or with the formalized problem statement. We evaluate our method on GSM8K, MATH and MultiArith datasets and demonstrate that our approach provides a consistently better heuristic than vanilla majority voting --- the previously best method to identify correct answers, by more than 12\% on GSM8K. In our experiments it improves results consistently across all datasets and LLM model sizes.
     </details>

19. **MUSTARD: Mastering Uniform Synthesis of Theorem and Proof Data** [[pdf]](http://arxiv.org/abs/2402.08957) `ICLR 2024` `Lean` (17 cite) (3 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces MUSTARD, a data generation framework that masters uniform synthesis of theorem and proof data of high quality and diversity, and performs extensive analysis and demonstrates that MUSTARD generates validated high-quality step-by-step data.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent large language models (LLMs) have witnessed significant advancement in various tasks, including mathematical reasoning and theorem proving. As these two tasks require strict and formal multi-step inference, they are appealing domains for exploring the reasoning ability of LLMs but still face important challenges. Previous studies such as Chain-of-Thought (CoT) have revealed the effectiveness of intermediate steps guidance. However, such step-wise annotation requires heavy labor, leading to insufficient training steps for current benchmarks. To fill this gap, this work introduces MUSTARD, a data generation framework that masters uniform synthesis of theorem and proof data of high quality and diversity. MUSTARD synthesizes data in three stages: (1) It samples a few mathematical concept seeds as the problem category. (2) Then, it prompts a generative language model with the sampled concepts to obtain both the problems and their step-wise formal solutions. (3) Lastly, the framework utilizes a proof assistant (e.g., Lean Prover) to filter the valid proofs. With the proposed MUSTARD, we present a theorem-and-proof benchmark MUSTARDSAUCE with 5,866 valid data points. Each data point contains an informal statement, an informal proof, and a translated formal proof that passes the prover validation. We perform extensive analysis and demonstrate that MUSTARD generates validated high-quality step-by-step data. We further apply the MUSTARDSAUCE for fine-tuning smaller language models. The fine-tuned Llama 2-7B achieves a 15.41% average relative performance gain in automated theorem proving, and 8.18% in math word problems. Codes and data are available at https://github.com/Eleanor-H/MUSTARD.
     </details>

20. **InternLM-Math: Open Math Large Language Models Toward Verifiable Reasoning** [[pdf]](http://arxiv.org/abs/2402.06332) `2024-02-09` `Lean` (33 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduces and open-source their math reasoning LLMs InternLM-Math which is continue pre-trained from InternLM2 and unify chain-of-thought reasoning, reward modeling, formal reasoning, data augmentation, and code interpreter in a unified seq2seq format and supervise the authors' model to be a versatile math reasoner, verifier, prover, and augmenter.
     </details>


     <details>
          <summary>Abstract</summary>
          The math abilities of large language models can represent their abstract reasoning ability. In this paper, we introduce and open-source our math reasoning LLMs InternLM-Math which is continue pre-trained from InternLM2. We unify chain-of-thought reasoning, reward modeling, formal reasoning, data augmentation, and code interpreter in a unified seq2seq format and supervise our model to be a versatile math reasoner, verifier, prover, and augmenter. These abilities can be used to develop the next math LLMs or self-iteration. InternLM-Math obtains open-sourced state-of-the-art performance under the setting of in-context learning, supervised fine-tuning, and code-assisted reasoning in various informal and formal benchmarks including GSM8K, MATH, Hungary math exam, MathBench-ZH, and MiniF2F. Our pre-trained model achieves 30.3 on the MiniF2F test set without fine-tuning. We further explore how to use LEAN to solve math problems and study its performance under the setting of multi-task learning which shows the possibility of using LEAN as a unified platform for solving and proving in math. Our models, codes, and data are released at \url{https://github.com/InternLM/InternLM-Math}.
     </details>

21. **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models** [[pdf]](http://arxiv.org/abs/2402.03300) `2024-02-06` `Isabelle` (123 cite) (7 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Group Relative Policy Optimization (GRPO), a variant of Proximal Policy Optimization (PPO), that enhances mathematical reasoning abilities while concurrently optimizing the memory usage of PPO is introduced.
     </details>


     <details>
          <summary>Abstract</summary>
          Mathematical reasoning poses a significant challenge for language models due to its complex and structured nature. In this paper, we introduce DeepSeekMath 7B, which continues pre-training DeepSeek-Coder-Base-v1.5 7B with 120B math-related tokens sourced from Common Crawl, together with natural language and code data. DeepSeekMath 7B has achieved an impressive score of 51.7% on the competition-level MATH benchmark without relying on external toolkits and voting techniques, approaching the performance level of Gemini-Ultra and GPT-4. Self-consistency over 64 samples from DeepSeekMath 7B achieves 60.9% on MATH. The mathematical reasoning capability of DeepSeekMath is attributed to two key factors: First, we harness the significant potential of publicly available web data through a meticulously engineered data selection pipeline. Second, we introduce Group Relative Policy Optimization (GRPO), a variant of Proximal Policy Optimization (PPO), that enhances mathematical reasoning abilities while concurrently optimizing the memory usage of PPO.
     </details>

22. **Graph2Tac: Online Representation Learning of Formal Math Concepts** [[pdf]](http://arxiv.org/abs/2401.02949) `ICML 2024` `Coq` (3 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work extensively benchmarks two online solvers implemented in the Tactician platform for the Coq proof assistant, and introduces a graph neural network, Graph2Tac, with a novel approach to build hierarchical representations for new definitions.
     </details>


     <details>
          <summary>Abstract</summary>
          In proof assistants, the physical proximity between two formal mathematical concepts is a strong predictor of their mutual relevance. Furthermore, lemmas with close proximity regularly exhibit similar proof structures. We show that this _locality_ property can be exploited through online learning techniques to obtain solving agents that far surpass offline learners when asked to prove theorems in an unseen mathematical setting. We extensively benchmark two such online solvers implemented in the Tactician platform for the Coq proof assistant: First, Tactician's online $k$-nearest neighbor solver, which can learn from recent proofs, shows a $1.72\times$ improvement in theorems proved over an offline equivalent. Second, we introduce a graph neural network, Graph2Tac, with a novel approach to build hierarchical representations for new definitions. Graph2Tac's online definition task realizes a $1.5\times$ improvement in theorems solved over an offline baseline. The $k$-NN and Graph2Tac solvers rely on orthogonal online data, making them highly complementary. Their combination improves $1.27\times$ over their individual performances. Both solvers outperform all other general purpose provers for Coq, including CoqHammer, Proverbot9001, and a transformer baseline by at least $1.48\times$ and are available for practical use by end-users.
     </details>

23. **The Tactician's Web of Large-Scale Formal Knowledge** [[pdf]](http://arxiv.org/abs/2401.02950) `2024-01-09` `Coq` (3 cite) (1 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          The Tactician's Web is a platform offering a large web of strongly interconnected, machine-checked, formal mathematical knowledge conveniently packaged for machine learning, analytics, and proof engineering. Built on top of the Coq proof assistant, the platform exports a dataset containing a wide variety of formal theories, presented as a web of definitions, theorems, proof terms, tactics, and proof states. Theories are encoded both as a semantic graph (rendered below) and as human-readable text, each with a unique set of advantages and disadvantages. Proving agents may interact with Coq through the same rich data representation and can be automatically benchmarked on a set of theorems. Tight integration with Coq provides the unique possibility to make agents available to proof engineers as practical tools.
     </details>

24. **TRIGO: Benchmarking Formal Mathematical Proof Reduction for Generative Language Models** [[pdf]](http://arxiv.org/abs/2310.10180) `EMNLP 2023 Main` `Lean` (11 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes TRIGO, an ATP benchmark that not only requires a model to reduce a trigonometric expression with step-by-step proofs but also evaluates a generative LM's reasoning ability on formulas and its capability to manipulate, group, and factor number terms.
     </details>


     <details>
          <summary>Abstract</summary>
          Automated theorem proving (ATP) has become an appealing domain for exploring the reasoning ability of the recent successful generative language models. However, current ATP benchmarks are mainly focus on symbolic inference, but rarely involve the understanding of complex number combination reasoning. In this work, we propose TRIGO, an ATP benchmark that not only requires a model to reduce a trigonometric expression with step-by-step proof but also evaluates a generative LM’s reasoning ability on formulas and capability to manipulate, group, and factor number terms. We gather trigonometric expressions and their reduced forms from web, annotate the simplification process manually, and translate it into the “Lean” formal language system. We then automatically generate additional examples from the annotated samples to expand the dataset. Furthermore, we also create three automatically generated training and testing datasets of varying difficulty and distributions. Our extensive experiments show our proposed TRIGO poses a new challenge for advanced generative LM’s including GPT-4 which is pre-trained on a considerable amount of open-source formal theorem-proving language data, and provide a new tool to study the generative LM’s ability on both formal and mathematical reasoning.
     </details>

25. **MLFMF: Data Sets for Machine Learning for Mathematical Formalization** [[pdf]](http://arxiv.org/abs/2310.16005) `NeurIPS 2023` `Lean, Agda` (3 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The MLFMF data sets provide solid benchmarking support for further investigation of the numerous machine learning approaches to formalized mathematics, and are currently the largest collection of formalized mathematical knowledge in machine learnable format.
     </details>


     <details>
          <summary>Abstract</summary>
          We introduce MLFMF, a collection of data sets for benchmarking recommendation systems used to support formalization of mathematics with proof assistants. These systems help humans identify which previous entries (theorems, constructions, datatypes, and postulates) are relevant in proving a new theorem or carrying out a new construction. Each data set is derived from a library of formalized mathematics written in proof assistants Agda or Lean. The collection includes the largest Lean 4 library Mathlib, and some of the largest Agda libraries: the standard library, the library of univalent mathematics Agda-unimath, and the TypeTopology library. Each data set represents the corresponding library in two ways: as a heterogeneous network, and as a list of s-expressions representing the syntax trees of all the entries in the library. The network contains the (modular) structure of the library and the references between entries, while the s-expressions give complete and easily parsed information about every entry.We report baseline results using standard graph and word embeddings, tree ensembles, and instance-based learning algorithms. The MLFMF data sets provide solid benchmarking support for further investigation of the numerous machine learning approaches to formalized mathematics. The methodology used to extract the networks and the s-expressions readily applies to other libraries, and is applicable to other proof assistants. With more than $250\,000$ entries in total, this is currently the largest collection of formalized mathematical knowledge in machine learnable format.
     </details>

26. **Llemma: An Open Language Model for Mathematics** [[pdf]](http://arxiv.org/abs/2310.10631) `ICLR 2024` `Lean, Isabelle` (164 cite) (12 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Llemma is a large language model for mathematics that outperforms all known open base models, as well as the unreleased Minerva model suite on an equi-parameter basis, and is capable of tool use and formal theorem proving without any further finetuning.
     </details>


     <details>
          <summary>Abstract</summary>
          We present Llemma, a large language model for mathematics. We continue pretraining Code Llama on the Proof-Pile-2, a mixture of scientific papers, web data containing mathematics, and mathematical code, yielding Llemma. On the MATH benchmark Llemma outperforms all known openly released models, as well as the unreleased Minerva model suite on an equi-parameter basis. Moreover, Llemma is capable of tool use and formal theorem proving without any finetuning. We openly release all artifacts, including 7 billion and 34 billion parameter models, the Proof-Pile-2, and code to replicate our experiments.
     </details>

27. **LEGO-Prover: Neural Theorem Proving with Growing Libraries** [[pdf]](http://arxiv.org/abs/2310.00656) `ICLR 2024` `Isabelle` (22 cite) (4 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work presents LEGO-Prover, which employs a growing skill library containing verified lemmas as skills to augment the capability of LLMs used in theorem proving, and advances the state-of-the-art pass rate on miniF2F-valid and miniF 2F-test.
     </details>


     <details>
          <summary>Abstract</summary>
          Despite the success of large language models (LLMs), the task of theorem proving still remains one of the hardest reasoning tasks that is far from being fully solved. Prior methods using language models have demonstrated promising results, but they still struggle to prove even middle school level theorems. One common limitation of these methods is that they assume a fixed theorem library during the whole theorem proving process. However, as we all know, creating new useful theorems or even new theories is not only helpful but crucial and necessary for advancing mathematics and proving harder and deeper results.In this work, we present LEGO-Prover, which employs a growing skill library containing verified lemmas as skills to augment the capability of LLMs used in theorem proving. By constructing the proof modularly, LEGO-Prover enables LLMs to utilize existing skills retrieved from the library and to create new skills during the proving process. These skills are further evolved (by prompting an LLM) to enrich the library on another scale. Modular and reusable skills are constantly added to the library to enable tackling increasingly intricate mathematical problems. Moreover, the learned library further bridges the gap between human proofs and formal proofs by making it easier to impute missing steps. LEGO-Prover advances the state-of-the-art pass rate on miniF2F-valid (48.0\% to 57.0\%) and miniF2F-test (45.5\% to 50.0\%). During the proving process, LEGO-Prover also generates over 20,000 skills (theorems/lemmas) and adds them to the growing library. Our ablation study indicates that these newly added skills are indeed helpful for proving theorems, resulting in a 4.9\% improvement in success rate
     </details>

28. **DT-Solver: Automated Theorem Proving with Dynamic-Tree Sampling Guided by Proof-level Value Function** [[pdf]](https://aclanthology.org/2023.acl-long.706) `ACL 2023` `Lean, Isabelle` (20 cite) (5 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel Dynamic-Tree Driven Theorem Solver (DT-Solver) is proposed, which introduces a dynamic-tree Monte-Carlo search algorithm, which dynamically allocates computing budgets for different state confidences, guided by a new proof-level value function to discover proof states that require substantial exploration.
     </details>


     <details>
          <summary>Abstract</summary>
          Recent advances in neural theorem-proving resort to large language models and tree searches. When proving a theorem, a language model advises single-step actions based on the current proving state and the tree search finds a sequence of correct steps using actions given by the language model. However, prior works often conduct constant computation efforts for each proving state while ignoring that the hard states often need more exploration than easy states. Moreover, they evaluate and guide the proof search solely depending on the current proof state instead of considering the whole proof trajectory as human reasoning does. Here, to accommodate general theorems, we propose a novel Dynamic-Tree Driven Theorem Solver (DT-Solver) by guiding the search procedure with state confidence and proof-level values. Specifically, DT-Solver introduces a dynamic-tree Monte-Carlo search algorithm, which dynamically allocates computing budgets for different state confidences, guided by a new proof-level value function to discover proof states that require substantial exploration. Experiments on two popular theorem-proving datasets, PISA and Mathlib, show significant performance gains by our DT-Solver over the state-of-the-art approaches, with a 6.65% improvement on average in terms of success rate. And especially under low computing resource settings (11.03% improvement on average).
     </details>

29. **LeanDojo: Theorem Proving with Retrieval-Augmented Language Models** [[pdf]](http://arxiv.org/abs/2306.15626) `NeurIPS 2023` `Lean` (111 cite) (12 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduces LeanDojo: an open-source Lean playground consisting of toolkits, data, models, and benchmarks, and develops ReProver (Retrieval-Augmented Prover): an LLM-based prover augmented with retrieval for selecting premises from a vast math library.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) have shown promise in proving formal theorems using proof assistants such as Lean. However, existing methods are difficult to reproduce or build on, due to private code, data, and large compute requirements. This has created substantial barriers to research on machine learning methods for theorem proving. This paper removes these barriers by introducing LeanDojo: an open-source Lean playground consisting of toolkits, data, models, and benchmarks. LeanDojo extracts data from Lean and enables interaction with the proof environment programmatically. It contains fine-grained annotations of premises in proofs, providing valuable data for premise selection—a key bottleneck in theorem proving. Using this data, we develop ReProver (Retrieval-Augmented Prover): an LLM-based prover augmented with retrieval for selecting premises from a vast math library. It is inexpensive and needs only one GPU week of training. Our retriever leverages LeanDojo's program analysis capability to identify accessible premises and hard negative examples, which makes retrieval much more effective. Furthermore, we construct a new benchmark consisting of 98,734 theorems and proofs extracted from Lean's math library. It features challenging data split requiring the prover to generalize to theorems relying on novel premises that are never used in training. We use this benchmark for training and evaluation, and experimental results demonstrate the effectiveness of ReProver over non-retrieval baselines and GPT-4. We thus provide the first set of open-source LLM-based theorem provers without any proprietary datasets and release it under a permissive MIT license to facilitate further research.
     </details>

30. **Subgoal-based Demonstration Learning for Formal Theorem Proving** [[pdf]](http://arxiv.org/abs/2305.16366) `ICML 2024` `Isabelle` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper introduces a subgoal-based demonstration learning framework, specifically designed to enhance the efficiency of proof search in LLMs, and builds upon recent advances in diffusion models to predict the optimal organization.
     </details>


     <details>
          <summary>Abstract</summary>
          Large language models (LLMs) present a promising pathway for advancing the domain of formal theorem proving. In this paper, we aim to improve the performance of LLMs in formal theorem proving by thoroughly examining the structure and organization of demonstrative in-context examples. We introduce a subgoal-based demonstration learning framework, specifically designed to enhance the efficiency of proof search in LLMs. First, drawing upon the insights of subgoal learning from reinforcement learning and robotics, we propose the construction of distinct subgoals for each demonstration example and refine these subgoals in accordance with the pertinent theories of subgoal learning. Second, we build upon recent advances in diffusion models to predict the optimal organization, simultaneously addressing two intricate issues that persist within the domain of demonstration organization: subset selection and order determination. Our integration of subgoal-based learning has notably increased proof accuracy from 38.9% to 44.1% on the miniF2F benchmark. Furthermore, the adoption of diffusion models for demonstration organization can lead to an additional enhancement in accuracy to 45.5%, or a $5\times$ improvement in sampling efficiency compared to previously established methods.
     </details>

31. **Magnushammer: A Transformer-Based Approach to Premise Selection** [[pdf]](https://arxiv.org/abs/2303.04488v1) `ICLR 2024` `Isabelle` (26 cite) (6 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work demonstrates that contrastive training with the transformer architecture can achieve higher-quality retrieval of relevant premises, without the engineering overhead, in a novel approach to premise selection.
     </details>


     <details>
          <summary>Abstract</summary>
          This paper presents a novel approach to premise selection, a crucial reasoning task in automated theorem proving. Traditionally, symbolic methods that rely on extensive domain knowledge and engineering effort are applied to this task. In contrast, this work demonstrates that contrastive training with the transformer architecture can achieve higher-quality retrieval of relevant premises, without the knowledge or feature engineering overhead. Our method, Magnushammer, outperforms the most advanced and widely used automation tool in interactive theorem proving called Sledgehammer. On the PISA and miniF2f benchmarks Magnushammer achieves $59.5\%$ (against $38.3\%$) and $34.0\%$ (against $20.9\%$) success rates, respectively. By combining Magnushammer with a language-model-based automated theorem prover, we further improve the state-of-the-art proof success rate from $57.0\%$ to $71.0\%$ on the PISA benchmark using $4$x fewer parameters. Moreover, we develop and open source a novel dataset for premise selection, containing textual representations of (proof state, relevant premise) pairs. To the best of our knowledge, this is the largest available premise selection dataset, and the first dataset of this kind for the Isabelle proof assistant.
     </details>

32. **Draft, Sketch, and Prove: Guiding Formal Theorem Provers with Informal Proofs** [[pdf]](http://arxiv.org/abs/2210.12283) `ICLR 2023` `Isabelle` (99 cite) (17 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Draft, Sketch, and Prove (DSP), a method that maps informal proofs to formal proof sketches, and uses the sketches to guide an automated prover by directing its search to easier sub-problems, is introduced.
     </details>


     <details>
          <summary>Abstract</summary>
          The formalization of existing mathematical proofs is a notoriously difficult process. Despite decades of research on automation and proof assistants, writing formal proofs remains arduous and only accessible to a few experts. While previous studies to automate formalization focused on powerful search algorithms, no attempts were made to take advantage of available informal proofs. In this work, we introduce Draft, Sketch, and Prove (DSP), a method that maps informal proofs to formal proof sketches, and uses the sketches to guide an automated prover by directing its search to easier sub-problems. We investigate two relevant setups where informal proofs are either written by humans or generated by a language model. Our experiments and ablation studies show that large language models are able to produce well-structured formal sketches that follow the same reasoning steps as the informal proofs. Guiding an automated prover with these sketches enhances its performance from $20.9\%$ to $39.3\%$ on a collection of mathematical competition problems.
     </details>

33. **HyperTree Proof Search for Neural Theorem Proving** [[pdf]](http://arxiv.org/abs/2205.11491) `NeurIPS 2022` `Lean, MetaMath` (84 cite) (18 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work shows that with HTPS alone, a model trained on annotated proofs manages to prove 65.4% of a held-out set of Metamath theorems, significantly outperforming the previous state of the art of 56.5% by GPT-f.
     </details>


     <details>
          <summary>Abstract</summary>
          We propose an online training procedure for a transformer-based automated theorem prover. Our approach leverages a new search algorithm, HyperTree Proof Search (HTPS), that learns from previous proof searches through online training, allowing it to generalize to domains far from the training distribution. We report detailed ablations of our pipeline’s main components by studying performance on three environments of increasing complexity. In particular, we show that with HTPS alone, a model trained on annotated proofs manages to prove 65.4% of a held-out set of Metamath theorems, significantly outperforming the previous state of the art of 56.5% by GPT-f. Online training on these unproved theorems increases accuracy to 82.6%. With a similar computational budget, we improve the state of the art on the Lean-based miniF2F-curriculum dataset from 31% to 42% proving accuracy.
     </details>

34. **Thor: Wielding Hammers to Integrate Language Models and Automated Theorem Provers** [[pdf]](http://arxiv.org/abs/2205.10893) `NeurIPS 2022` `Isabelle` (65 cite) (15 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          Thor is introduced, a framework integrating language models and automated theorem provers to overcome the difficulty of selecting useful premises from a large library to unlock the proof of a given conjecture.
     </details>


     <details>
          <summary>Abstract</summary>
          In theorem proving, the task of selecting useful premises from a large library to unlock the proof of a given conjecture is crucially important. This presents a challenge for all theorem provers, especially the ones based on language models, due to their relative inability to reason over huge volumes of premises in text form. This paper introduces Thor, a framework integrating language models and automated theorem provers to overcome this difficulty. In Thor, a class of methods called hammers that leverage the power of automated theorem provers are used for premise selection, while all other tasks are designated to language models. Thor increases a language model's success rate on the PISA dataset from $39\%$ to $57\%$, while solving $8.2\%$ of problems neither language models nor automated theorem provers are able to solve on their own. Furthermore, with a significantly smaller computational budget, Thor can achieve a success rate on the MiniF2F dataset that is on par with the best existing methods. Thor can be instantiated for the majority of popular interactive theorem provers via a straightforward protocol we provide.
     </details>

35. **The Isabelle ENIGMA** [[pdf]](http://arxiv.org/abs/2205.01981) `2022-05-04` `Isabelle` (13 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The authors' final best single-strategy ENIGMA and premise selection system improves the best previous version of E by 25.3% in 15 seconds, outperforming also all other previous ATP and SMT systems.
     </details>


     <details>
          <summary>Abstract</summary>
          We significantly improve the performance of the E automated theorem prover on the Isabelle Sledgehammer problems by combining learning and theorem proving in several ways. In particular, we develop targeted versions of the ENIGMA guidance for the Isabelle problems, targeted versions of neural premise selection, and targeted strategies for E. The methods are trained in several iterations over hundreds of thousands untyped and typed first-order problems extracted from Isabelle. Our final best single-strategy ENIGMA and premise selection system improves the best previous version of E by 25.3% in 15 seconds, outperforming also all other previous ATP and SMT systems.
     </details>

36. **Autoformalization with Large Language Models** [[pdf]](https://arxiv.org/abs/2205.12615) `NeurIPS 2022` `Lean` (108 cite) (16 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          It is shown large language models provide new prospects towards the long-term goal of autoformalization, and the surprising observation that LLMs can correctly translate a significant portion of mathematical competition problems perfectly to formal specifications in Isabelle/HOL.
     </details>


     <details>
          <summary>Abstract</summary>
          Autoformalization is the process of automatically translating from natural language mathematics to formal specifications and proofs. A successful autoformalization system could advance the fields of formal verification, program synthesis, and artificial intelligence.While the long-term goal of autoformalization seemed elusive for a long time, we show large language models provide new prospects towards this goal. We make the surprising observation that LLMs can correctly translate a significant portion ($25.3\%$) of mathematical competition problems perfectly to formal specifications in Isabelle/HOL. We demonstrate the usefulness of this process by improving a previously introduced neural theorem prover via training on these autoformalized theorems. Our methodology results in a new state-of-the-art result on the MiniF2F theorem proving benchmark, improving the proof rate from~$29.6\%$ to~$35.2\%$.
     </details>

37. **Formal Mathematics Statement Curriculum Learning** [[pdf]](http://arxiv.org/abs/2202.01344) `ICLR 2023` `Lean` (0 cite) (17 AI4Math cite) 


     <details>
          <summary>Abstract</summary>
          We explore the use of expert iteration in the context of language modeling applied to formal mathematics. We show that at same compute budget, expert iteration, by which we mean proof search interleaved with learning, dramatically outperforms proof search only. We also observe that when applied to a collection of formal statements of sufficiently varied difficulty, expert iteration is capable of finding and solving a curriculum of increasingly difficult problems, without the need for associated ground-truth proofs. Finally, by applying this expert iteration to a manually curated set of problem statements, we surpass previous state-of-the-art on the miniF2F benchmark, automatically solving multiple challenging problems drawn from high school olympiads.
     </details>

38. **ProofNet: Autoformalizing and Formally Proving Undergraduate-Level Mathematics** [[pdf]](https://arxiv.org/abs/2302.12433) `NeurIPS 2022 MATH-AI Workshop` `Lean` (37 cite) (5 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work introduces ProofNet, a benchmark for autoformalization and formal proving of undergraduate-level mathematics, and introduces two novel statement auto Formalization methods: prompt retrieval and distilled backtranslation.
     </details>


     <details>
          <summary>Abstract</summary>
          We introduce ProofNet, a benchmark for autoformalization and formal proving of undergraduate-level mathematics. The ProofNet benchmarks consists of 371 examples, each consisting of a formal theorem statement in Lean 3, a natural language theorem statement, and a natural language proof. The problems are primarily drawn from popular undergraduate pure mathematics textbooks and cover topics such as real and complex analysis, linear algebra, abstract algebra, and topology. We intend for ProofNet to be a challenging benchmark that will drive progress in autoformalization and automatic theorem proving. We report baseline results on statement autoformalization via in-context learning. Moreover, we introduce two novel statement autoformalization methods: prompt retrieval and distilled backtranslation.
     </details>

39. **Proving Theorems using Incremental Learning and Hindsight Experience Replay** [[pdf]](https://arxiv.org/abs/2112.10664) `ICML 2022` `Mizar` (11 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A general incremental learning algorithm for training domain specific provers for first-order logic without equality, based only on a basic given-clause algorithm, but using a learned clause-scoring function, and it is shown that provers trained this way can match and sometimes surpass state-of-the-art traditional provers on the TPTP dataset.
     </details>


     <details>
          <summary>Abstract</summary>
          Traditional automated theorem proving systems for first-order logic depend on speed-optimized search and many handcrafted heuristics designed to work over a wide range of domains. Machine learning approaches in the literature either depend on these traditional provers to bootstrap themselves, by leveraging these heuristics, or can struggle due to limited existing proof data. The latter issue can be explained by the lack of a smooth difficulty gradient in theorem proving datasets; large gaps in difficulty between different theorems can make training harder or even impossible. In this paper, we adapt the idea of hindsight experience replay from reinforcement learning to the automated theorem proving domain, so as to use the intermediate data generated during unsuccessful proof attempts. We build a first-order logic prover by disabling all the smart clause-scoring heuristics of the state-of-the-art E prover and replacing them with a clause-scoring neural network learned by using hindsight experience replay in an incremental learning setting. Clauses are represented as graphs and presented to transformer networks with spectral features. We show that provers trained in this way can outperform previous machine learning approaches and compete with the state of the art heuristic-based theorem prover E in its best configuration, on the popular benchmarks MPTP2078, M2k and Mizar40. The proofs generated by our algorithm are also almost always significantly shorter than E’s proofs.
     </details>

40. **Formal Premise Selection With Language Models** [[pdf]](https://www.semanticscholar.org/paper/2443179d421e1faf7474add557b45add554723c7) `2022-01-01` `Isabelle` (7 cite) (2 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work provides a solution to the problem of selecting a useful premise to prove a new theorem by combining a premise selection model with a language model, and shows that this retrieval-augmented prover achieves significant improvements in proof rates compared to the language model alone.
     </details>


     <details>
          <summary>Abstract</summary>
          N/A
     </details>

41. **miniF2F: a cross-system benchmark for formal Olympiad-level mathematics** [[pdf]](https://arxiv.org/abs/2109.00110) `ICLR 2022` `Lean, Isabelle, HOL Light, MetaMath` (84 cite) (23 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The miniF2F benchmark currently targets Metamath, Lean, Isabelle, and HOL Light and consists of 488 problem statements drawn from the AIME, AMC, and the International Mathematical Olympiad, as well as material from high-school and undergraduate mathematics courses.
     </details>


     <details>
          <summary>Abstract</summary>
          We present $\textsf{miniF2F}$, a dataset of formal Olympiad-level mathematics problems statements intended to provide a unified cross-system benchmark for neural theorem proving. The $\textsf{miniF2F}$ benchmark currently targets Metamath, Lean, Isabelle (partially) and HOL Light (partially) and consists of 488 problem statements drawn from the AIME, AMC, and the International Mathematical Olympiad (IMO), as well as material from high-school and undergraduate mathematics courses. We report baseline results using GPT-f, a neural theorem prover based on GPT-3 and provide an analysis of its performance. We intend for $\textsf{miniF2F}$ to be a community-driven effort and hope that our benchmark will help spur advances in neural theorem proving.
     </details>

42. **Learning Theorem Proving Components** [[pdf]](http://arxiv.org/abs/2107.10034) `2021-07-21` `Mizar` (7 cite) (1 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work describes several algorithms and experiments with ENIGMA, advancing the idea of contextual evaluation based on learning important components of the graph of clauses, and equipping the E/ENIGMA system with a graph neural network that chooses the next given clause based on its evaluation in the context of previously selected clauses.
     </details>


     <details>
          <summary>Abstract</summary>
          Saturation-style automated theorem provers (ATPs) based on the given clause procedure are today the strongest general reasoners for classical first-order logic. The clause selection heuristics in such systems are, however, often evaluating clauses in isolation, ignoring other clauses. This has changed recently by equipping the E/ENIGMA system with a graph neural network (GNN) that chooses the next given clause based on its evaluation in the context of previously selected clauses. In this work, we describe several algorithms and experiments with ENIGMA, advancing the idea of contextual evaluation based on learning important components of the graph of clauses.
     </details>

43. **Proof Artifact Co-Training for Theorem Proving with Language Models** [[pdf]](https://arxiv.org/abs/2102.06203) `ICLR 2022` `Lean` (98 cite) (27 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          PACT is proposed, a general methodology for extracting abundant self-supervised data from kernel-level proof terms for co-training alongside the usual tactic prediction objective and applied to Lean, an interactive proof assistant which hosts some of the most sophisticated formalized mathematics to date.
     </details>


     <details>
          <summary>Abstract</summary>
          Labeled data for imitation learning of theorem proving in large libraries of formalized mathematics is scarce as such libraries require years of concentrated effort by human specialists to be built. This is particularly challenging when applying large Transformer language models to tactic prediction, because the scaling of performance with respect to model size is quickly disrupted in the data-scarce, easily-overfitted regime. We propose PACT (Proof Artifact Co-Training), a general methodology for extracting abundant self-supervised data from kernel-level proof terms for joint training alongside the usual tactic prediction objective. We apply this methodology to Lean,an interactive proof assistant which hosts some of the most sophisticated formalized mathematics to date. We instrument Lean with a neural theorem prover driven by a Transformer language model and show that PACT improves theorem proving success rate on a held-out suite of test theorems from 32% to 48%.
     </details>

44. **LIME: Learning Inductive Bias for Primitives of Mathematical Reasoning** [[pdf]](https://arxiv.org/abs/2101.06223) `ICML 2021` `Isabelle, MetaMath, HOL Light, Lean` (47 cite) (8 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A new pre-training methodology called LIME (Learning Inductive bias for Mathematical rEasoning).
     </details>


     <details>
          <summary>Abstract</summary>
          While designing inductive bias in neural architectures has been widely studied, we hypothesize that transformer networks are flexible enough to learn inductive bias from suitable generic tasks. Here, we replace architecture engineering by encoding inductive bias in the form of datasets. Inspired by Peirce's view that deduction, induction, and abduction are the primitives of reasoning, we design three synthetic tasks that are intended to require the model to have these three abilities. We specifically design these tasks to be synthetic and devoid of mathematical knowledge to ensure that only the fundamental reasoning biases can be learned from these tasks. This defines a new pre-training methodology called "LIME" (Learning Inductive bias for Mathematical rEasoning). Models trained with LIME significantly outperform vanilla transformers on four very different large mathematical reasoning benchmarks. Unlike dominating the computation cost as traditional pre-training approaches, LIME requires only a small fraction of the computation cost of the typical downstream task. The code for generating LIME tasks is available at https://github.com/tonywu95/LIME.
     </details>

45. **TacticZero: Learning to Prove Theorems from Scratch with Deep Reinforcement Learning** [[pdf]](https://arxiv.org/abs/2102.09756) `NeurIPS 2021` `HOL 4` (34 cite) (7 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A novel approach to interactive theorem-proving (ITP) using deep reinforcement learning in which each state represents a set of potential derivation paths, and introduces a novel backtracking mechanism which enables the agent to efficiently discard dead-end derivations and restart from promising alternatives.
     </details>


     <details>
          <summary>Abstract</summary>
          We propose a novel approach to interactive theorem-proving (ITP) using deep reinforcement learning. The proposed framework is able to learn proof search strategies as well as tactic and arguments prediction in an end-to-end manner. We formulate the process of ITP as a Markov decision process (MDP) in which each state represents a set of potential derivation paths. This structure allows us to introduce a novel backtracking mechanism which enables the agent to efficiently discard (predicted) dead-end derivations and restart the derivation from promising alternatives. We implement the framework in the HOL theorem prover. Experimental results show that the framework using learned search strategies outperforms existing automated theorem provers (i.e., hammers) available in HOL when evaluated on unseen problems. We further elaborate the role of key components of the framework using ablation studies.
     </details>

46. **REFACTOR: Learning to Extract Theorems from Proofs** [[pdf]](https://arxiv.org/abs/2402.17032) `ICLR 2024` `MetaMath` (1 cite) (0 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This paper shows on a set of unseen proofs, REFACTOR is able to extract 19.6% of the theorems that humans would use to write the proofs, and demonstrates that the prover trained on the new-theorem refactored dataset proves more test theorems and outperforms state-of-the-art baselines by frequently leveraging a diverse set of newly extracted theorems.
     </details>


     <details>
          <summary>Abstract</summary>
          Human mathematicians are often good at recognizing modular and reusable theorems that make complex mathematical results within reach. In this paper, we propose a novel method called theoREm-from-prooF extrACTOR (REFACTOR) for training neural networks to mimic this ability in formal mathematical theorem proving. We show on a set of unseen proofs, REFACTOR is able to extract 19.6\% of the theorems that humans would use to write the proofs. When applying the model to the existing Metamath library, REFACTOR extracted 16 new theorems. With newly extracted theorems, we show that the existing proofs in the MetaMath database can be refactored. The new theorems are used very frequently after refactoring, with an average usage of 733.5 times, and help shorten the proof lengths. Lastly, we demonstrate that the prover trained on the new-theorem refactored dataset proves more test theorems and outperforms state-of-the-art baselines by frequently leveraging a diverse set of newly extracted theorems.
     </details>

47. **IsarStep: a Benchmark for High-level Mathematical Reasoning** [[pdf]](https://arxiv.org/abs/2006.09265) `ICLR 2021` `Isabelle` (52 cite) (17 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A benchmark for high-level mathematical reasoning is presented and the reasoning capabilities of neural sequence-to-sequence models are studied and a hierarchical transformer is designed that outperforms the transformer baseline.
     </details>


     <details>
          <summary>Abstract</summary>
          A well-defined benchmark is essential for measuring and accelerating research progress of machine learning models. In this paper, we present a benchmark for high-level mathematical reasoning and study the reasoning capabilities of neural sequence-to-sequence models. We build a non-synthetic dataset from the largest repository of proofs written by human experts in a theorem prover. The dataset has a broad coverage of undergraduate and research-level mathematical and computer science theorems. In our defined task, a model is required to fill in a missing intermediate proposition given surrounding proofs. This task provides a starting point for the long-term goal of having machines generate human-readable proofs automatically. Our experiments and analysis reveal that while the task is challenging, neural models can capture non-trivial mathematical reasoning. We further design a hierarchical transformer that outperforms the transformer baseline.
     </details>

48. **Learning to Prove Theorems by Learning to Generate Theorems** [[pdf]](https://arxiv.org/abs/2002.07019) `NeurIPS 2020` `Holophrasm` (41 cite) (13 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work proposes to learn a neural generator that automatically synthesizes theorems and proofs for the purpose of training a theorem prover, and demonstrates that synthetic data from this approach improves the theorem provers and advances the state of the art of automated theorem proving in Metamath.
     </details>


     <details>
          <summary>Abstract</summary>
          We consider the task of automated theorem proving, a key AI task. Deep learning has shown promise for training theorem provers, but there are limited human-written theorems and proofs available for supervised learning. To address this limitation, we propose to learn a neural generator that automatically synthesizes theorems and proofs for the purpose of training a theorem prover. Experiments on real-world tasks demonstrate that synthetic data from our approach improves the theorem prover and advances the state of the art of automated theorem proving in Metamath. Code is available at https://github.com/princeton-vl/MetaGen.
     </details>

49. **Mathematical Reasoning in Latent Space** [[pdf]](http://arxiv.org/abs/1909.11851) `ICLR 2020` `HOL Light` (33 cite) (3 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          The experiments show that graph neural networks can make non-trivial predictions about the rewrite-success of statements, even when they propagate predicted latent representations for several steps, a strong indicator for the feasibility of deduction in latent space in general.
     </details>


     <details>
          <summary>Abstract</summary>
          We design and conduct a simple experiment to study whether neural networks can perform several steps of approximate reasoning in a fixed dimensional latent space. The set of rewrites (i.e. transformations) that can be successfully performed on a statement represents essential semantic features of the statement. We can compress this information by embedding the formula in a vector space, such that the vector associated with a statement can be used to predict whether a statement can be rewritten by other theorems. Predicting the embedding of a formula generated by some rewrite rule is naturally viewed as approximate reasoning in the latent space. In order to measure the effectiveness of this reasoning, we perform approximate deduction sequences in the latent space and use the resulting embedding to inform the semantic features of the corresponding formal statement (which is obtained by performing the corresponding rewrite sequence using real formulas). Our experiments show that graph neural networks can make non-trivial predictions about the rewrite-success of statements, even when they propagate predicted latent representations for several steps. Since our corpus of mathematical formulas includes a wide variety of mathematical disciplines, this experiment is a strong indicator for the feasibility of deduction in latent space in general.
     </details>

50. **Learning to Prove Theorems via Interacting with Proof Assistants** [[pdf]](https://www.semanticscholar.org/paper/a596f03145285cd05a6ca57a4e25418b23b24976) `ICML 2019` `Coq` (117 cite) (22 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          ASTactic, a deep learning-based model that generates tactics as programs in the form of abstract syntax trees (ASTs) can generate effective tactics and can be used to prove new theorems not previously provable by automated methods.
     </details>


     <details>
          <summary>Abstract</summary>
          N/A
     </details>

51. **HOList: An Environment for Machine Learning of Higher-Order Theorem Proving (extended version)** [[pdf]](https://www.semanticscholar.org/paper/9ef2e09a9e16e176e19c3fdc3b6ee22c5d3f3c97) `ICML 2019` `HOL Light` (45 cite) (7 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          This work provides an open-source framework based on the HOL Light theorem prover that can be used as a reinforcement learning environment and presents a deep reinforcement learning driven automated theorem provers, DeepHOL, with strong initial results on this benchmark.
     </details>


     <details>
          <summary>Abstract</summary>
          N/A
     </details>

52. **Reinforcement Learning of Theorem Proving** [[pdf]](https://www.semanticscholar.org/paper/920febb03475b068286a855c10ea09b968fe7ee3) `NeurIPS 2018` `Mizar` (135 cite) (13 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A theorem proving algorithm that uses practically no domain heuristics for guiding its connection-style proof search and solves within the same number of inferences over 40% more problems than a baseline prover, which is an unusually high improvement in this hard AI domain.
     </details>


     <details>
          <summary>Abstract</summary>
          N/A
     </details>

53. **GamePad: A Learning Environment for Theorem Proving** [[pdf]](https://www.semanticscholar.org/paper/87c425f23bcac2f082968abda64a971f91522f73) `ICLR 2019` `Coq` (97 cite) (16 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A system called GamePad is introduced that can be used to explore the application of machine learning methods to theorem proving in the Coq proof assistant and addresses position evaluation and tactic prediction tasks, which arise naturally in tactic-based theorem proving.
     </details>


     <details>
          <summary>Abstract</summary>
          N/A
     </details>

54. **HolStep: A Machine Learning Dataset for Higher-order Logic Theorem Proving** [[pdf]](http://arxiv.org/abs/1703.00426) `ICLR 2017` `HOL Light` (79 cite) (10 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A new dataset based on Higher-Order Logic (HOL) proofs is introduced, for the purpose of developing new machine learning-based theorem-proving strategies and the results of these models show the promise of applying machine learning to HOL theorem proving.
     </details>


     <details>
          <summary>Abstract</summary>
          Large computer-understandable proofs consist of millions of intermediate logical steps. The vast majority of such steps originate from manually selected and manually guided heuristics applied to intermediate goals. So far, machine learning has generally not been used to filter or generate these steps. In this paper, we introduce a new dataset based on Higher-Order Logic (HOL) proofs, for the purpose of developing new machine learning-based theorem-proving strategies. We make this dataset publicly available under the BSD license. We propose various machine learning tasks that can be performed on this dataset, and discuss their significance for theorem proving. We also benchmark a set of simple baseline machine learning models suited for the tasks (including logistic regression, convolutional neural networks and recurrent neural networks). The results of our baseline models show the promise of applying machine learning to HOL theorem proving.
     </details>

55. **DeepMath - Deep Sequence Models for Premise Selection** [[pdf]](http://arxiv.org/abs/1606.04442) `NeurIPS 2016` `Mizar` (212 cite) (26 AI4Math cite) 


     <details>
          <summary>SemanticsScholar tldr</summary>
          A two stage approach is proposed that yields good results for the premise selection task on the Mizar corpus while avoiding the hand-engineered features of existing state-of-the-art models.
     </details>


     <details>
          <summary>Abstract</summary>
          We study the effectiveness of neural sequence models for premise selection in automated theorem proving, one of the main bottlenecks in the formalization of mathematics. We propose a two stage approach for this task that yields good results for the premise selection task on the Mizar corpus while avoiding the hand-engineered features of existing state-of-the-art models. To our knowledge, this is the first time deep learning has been applied to theorem proving on a large scale.
     </details>

