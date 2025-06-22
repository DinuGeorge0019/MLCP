The models in this collection were introduced in the paper **Domain Adaptation for Automated Tag Prediction in Competitive Programming**. They fall into two broad groups:

1. **Domain-Adapted MPNet Encoders**
   All begin from the `all-mpnet-base-v2` checkpoint, which has been domain-adapted on competitive-programming text and serves as the backbone for three downstream architectures—Unified Model, One-Vs-All classifier, and ChainClassifier. For each training dataset we release three variants optimized for Top 5, Top 10, and Top 20 tag recall:

   * **Trained on our proposed dataset** (suffix `OurDataset`):

     * `Sauron0019/DA-all-mpnet-base-v2-Top20-OurDataset`
     * `Sauron0019/DA-all-mpnet-base-v2-Top10-OurDataset`
     * `Sauron0019/DA-all-mpnet-base-v2-Top5-OurDataset` 
   * **Trained on the public dataset of Kim et al. (2023)** (suffix `OutsideDataset`):

     * `Sauron0019/DA-all-mpnet-base-v2-Top20-OutsideDataset`
     * `Sauron0019/DA-all-mpnet-base-v2-Top10-OutsideDataset`
     * `Sauron0019/DA-all-mpnet-base-v2-Top5-OutsideDataset` 

2. **Fine-Tuned Large Language Models**
   These are full LLMs (DeepSeek-LLM-7B, Llama-3-8B, Gemma-3-12B) further fine-tuned on editorial metadata to predict CP problem tags directly. Each comes in Top 5, Top 10, and Top 20 variants:

   * **DeepSeek-LLM-7B**

     * `Sauron0019/DeepSeek-LLM-7B-Base-TagPrediction-Top20-Editorial`
     * `Sauron0019/DeepSeek-LLM-7B-Base-TagPrediction-Top10-Editorial`
     * `Sauron0019/DeepSeek-LLM-7B-Base-TagPrediction-Top5-Editorial` 
   * **Llama-3-8B**

     * `Sauron0019/Llama-3-8B-TagPrediction-Top20-Editorial` 
     * `Sauron0019/Llama-3-8B-TagPrediction-Top10-Editorial` 
     * `Sauron0019/Llama-3-8B-TagPrediction-Top5-Editorial`  
   * **Gemma-3-12B**

     * `Sauron0019/Gemma-3-12B-TagPrediction-Top20-Editorial` 
     * `Sauron0019/Gemma-3-12B-TagPrediction-Top10-Editorial` 
     * `Sauron0019/Gemma-3-12B-TagPrediction-Top5-Editorial`  

