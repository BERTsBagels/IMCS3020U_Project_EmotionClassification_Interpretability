# **IMCS3020U Project: Investigating Interpretability of BERT for Emotion Classification by Comparing Shapley Additive Explanations (SHAP), Local Interpretable Model-Agnostic Explanations (LIME), and Integrated Gradients**

## **Table of Contents**
- [***Project Information***](#project-information)
- [***Mathematical & Computational Backgrounds***](#mathematical--computational-backgrounds)
- [***Getting Started***](#getting-started)
- [***Methodology Overview***](#methodology-overview)
- [***Discussion***](#discussion)
- [***Project Testing Results***](#project-testing-results)
- [***Author(s)***](#authors)

## **Project Information**

### **Overview**
This project was developed as a chosen topical project for the **IMCS3020U: Integrated Project Course II** during the 2026 Winter Semester at Ontario Tech University. Its main purpose is aiming to compare methods of explaining the predictions made by BERT for applications such as mental health support and diagnostics or hate speech detection so that the process by which they are made is transparent, quantifiable, and trustworthy.

### **Description**
In a previous project, BERT was compared with MLkNN at the task of emotion classification on the GoEmotions dataset. While BERT performed considerably better than MLkNN based on the F1-score, the process that allows it to categorize text by emotion content is not as easy to understand as with the MLkNN algorithm. In this project we seek to explain and interpret the results of using BERT to categorize emotion in text samples, by applying three frameworks to the existing BERT pipeline from the previous project: LIME, SHAP, and Integrated Gradients.

### **Project Contributor(s)**
- **Tobenna Nnaobi**
- **Marian Waffle**
- **Jin Sutharman**

## **Mathematical & Computational Backgrounds**
- **BERT (Bidirectional Encoder Representations from Transformers)**: A model capable of powerful text classification by capturing contextual meaning in its predictions. This model has revolutionized natural language processing, yet its complexity makes explaining its decisions challenging. Understanding BERT’s predictions and their basis is crucial to the safety and trustworthiness of the many applications of this technology.

- **SHapley Additive exPlanations (SHAP)**: This concept explains the contributions of various tokens to the end prediction by BERT. It is found in game theory and seeks to fairly divide proceeds of a winning game to players on the same team that have not contributed equally to the outcome. SHAP is represented by the following equation:
    - $`\begin{align} \phi_{i} = \sum_{S\subseteq\{1,...,p\}\{i\}} \frac{\left|S\right|!(p-\left|S\right|-1)!}{p!}[val(S \cup \{i\}) - val(S)] \end{align}`$

    - ***Where***:
        - $`N`$ is the set of all features or words in the sentence, demarcated by [SEP],
        - $`S`$ is a subset of tokens minus $`i`$,
        - $`val(S)`$ is the prediction the model makes based on the words in subset S,
        - $`[val(S \cup \{i\}) - val(S)]`$ is called the **marginal contribution**, the amount that the prediction changes when $i joins subset S, and 
        - $`\frac{\left|S\right|!(p-\left|S\right|-1)!}{p!}`$ is the probability of forming this specific subset. 

-  **Local Interpretable Model-Agnostic Explanations (LIME)**: A model that can be used to produce an explanation for each features contribution to a model's final output. LIME does this by exploring specific instances of a models predictions and perturbing them, essentially removing combinations of features in that instance to examine how it changes the models output. This information is used to fit the instance of data into a more interpretable model. This provides an explainable output for feature contribution. LIME is represented by the following equation:
    - $`\begin{align}\xi(x) = \arg\min\limits_{g \in G} \mathcal{L}(f,g,\pi_{x}) + \Omega(g) \end{align}`$

    - ***Where***:
        - $`f`$ is the black model
        - $`g`$ is the surrogate, interpretable model.
        - $`G`$ is the set of possible interpretable models.
        - $`\pi_{x}`$ is the weight of how similar a perturbed instance of text is to the original instance, defining a local neighbourhood.
        - $`\mathcal{L}(f,g,\pi_{x})`$ is the weighted losss that measures how close $`g`$ approximates $`f`$ on the perturbed instance.
        - $`\Omega(g)`$ is a penalty value to $`g`$ to discourage excess model complexity.

- **Integrated Gradients**: An attribution technique that explains a model’s prediction by quantifying the contribution of each input feature. It works by accumulating gradients along a straight path from a user-defined baseline input to the actual input. This path integral ensures that the attributions satisfy the fundamental axioms like completeness and sensitivity (non-zero attributions for features that change the prediction). Integrated Gradients is represented by the following equation:
    - $`\begin{align}\text{IntegratedGrads}_{i}(x) := (x_{i} - x^{\prime}_{i}) \times \int_{\alpha = 0}^{1} \frac{\partial F(x^{\prime} + \alpha \times (x - x^{\prime}))}{\partial x_{i}} d\alpha\end{align}`$

    - ***Where***:
        - $`\text{IntegratedGrads}_{i}(x)`$ is the integrated gradient for the $`i`$-th input feature.
        - $`x`$ is the actual input
        - $`x^{\prime}`$ is the baseline input
        - $`F`$ is the function of the neural network
        - $`\alpha`$ is a parameter that varies from 0 to 1 on a straight path.

## **Getting Started**

### **Requirements**
- Any Version $\leq$ **Python 3.12.10** Is Required
- Any IDE With Jupyter Notebook Or Any Notebook-like Adajecent Functionalities Are Recommended (e.g. **Visual Studio Code**, **Google Colab**, etc.) 

### **Dependencies & Modules**
- **Pandas**: A fast, powerful, and flexible Python dependency that used for primarily for data analysis and manipulation purposes.

- **PyTorch**: An open-source deep learning library. The successor to **Torch**, PyTorch provides a high-level API that builds upon optimized, low-level implementations of deep learning algorithms and architectures (e.g. the **Transformer**, the **Stochastic Gradient Descent (SGD)**)

- **Sci-kit Learn**: A Python module that is utilized for machine learning purposes and contains
tools for predictive data analysis. It is built on NumPy, SciPy, and matplotlib.

- **Captum**: An open-source, extensible library for model interpretability built on **PyTorch**.

- **BERTTransformer**: A model capable of powerful text classification by capturing contextual meaning in its predictions.

- **SHAP**: A game theoretic approach explain the output of any machine learning model. It connects optimal credit allocation with local explanations using classic Shapley values from game theory and their related extensions.

- **LIME**: An approach that can be used to produce an explanation for each features contribution to a model's final output.

- **NumPy**: A powerful open-source Python library for scientific computing in Python that provides a multidimensional array object, various dervied (e.g. masked arrays and matrices), and an assortment of routines for fast operations on arrays (e.g. Mathematical, Logical, Shape Manipulation, Sorting).

- **Iterative Stratification**: A project that provides **Sci-kit Learn** compatible cross validators with stratification for multilabel data.

## **Methodology Overview**
### **Environment and Model Setup**
The environment was configured to use a local GPU via CUDA for computationally intensive tasks.
Finding the computational resources necessary to run BERT efficiently was initially a challenge.
To address this, the fine-tuned model weights were downloaded and migrated to a local Jupyter
Notebook from their initial Google Colab environment.

### **Architecture**
The BERT model used is the same as the model from the previous project, the bert-base-uncased
sourced from Hugging Face. The uncased version was selected for its insensitivity to casing. The GoEmotions data set contains many samples with inconsistent casing, as social media text often contains unpredictable capitalization. While casing can contain important information, such as
emphasis, the erratic nature of casing in social media makes the uncased model better suited. BERT is available in both large and base configurations, containing approximately 340 million and 110, respectively. While the large model offers an improved ability to capture complex patterns, the computational efficiency is directly proportional to this complexity. To this end, the base model was selected as a compromise between precision and computational efficiency.

### **Hyperparameters**
The following user-controller hyperparameters were selected for the fine-tuning of BERT:
- **Optimzer:** The **Adam with Decoupled Weight Decay (AdamW)** optimizer allows for more flexible tuning of hyperparameters by preventing overfitting of the model.

- **Learning Rate:** A base learning rate of $`1 \times 10^{−5}`$ was used during the fine tuning process. Using a scheduler, this was increased linearly at a rate of 0.1. For BERT, small learning rates are typical to reduce the risk of exploding gradients and improve convergence speed.

- **Batch Size:** During the fine-tuning process, a batch size of 16 was used. This value provides balance between computational efficiency, convergence speed and model stability.

- **Epochs:** The model was fine-tuned for 5 epochs.

- **Loss Function:** During training, a weighted binary cross-entropy loss was used via the `BCEWithLogitsLoss` function. This function improves training stability by combining a sigmoid layer with Binary Cross-Entropy. To improve classification of infrequent emotion labels, the `pos_weight` hyperparameter, defined as the ratio of negative to positive samples for each label was passed to this function.

### **Data Pipeline**
Since LIME, SHAP and IG work to explain the output of BERT, they use a shared data processing procedure to transform the GoEmotions dataset into the necessary format. This includes
tokenization using WordPiece and a conversion of the dataset from CSV to a tensor representation.

#### **Tokenization with WordPiece**
Using the WordPiece algorithm, the raw text of each sample in the dataset is broken down into words and subwords, resulting in tokens. These tokens are assigned a vector ID based on its index in a fixed vocabulary used by BERT. Special tokens are used to signal the structure of the text to the model. The token [CLS] is used to represent the beginning of a text sample, [SEP] to mark the separation of sentences, and [PAD], a null token that fills samples to ensure all samples are all the same length.

#### **PyTorch**
WordPiece tokens need to be mapped to three kinds of tensors:
- **Input IDs**, which represent the token indices from the fixed vocabulary.

- **Attention Masks**, which will allow the model to ignore [PAD] tokens using a binary value, and

- **Token type IDs**, which will mark which sentence in a sample the token belongs to.

These tensors are encapsulated within the **DataLoader** class, which handles iteration and batching. An initial batch size of 16 was used for testing and later increased to 64 to better utilize the GPU. Since the model is initialized in evaluation mode, increasing the batch size does not affect
performance metrics such as the F1-score, but significantly improves the computational efficiency of the XAI explainability methods.

#### **Ekman Label Mapping**
The process of using explanation methods on a multilabel dataset with 28 labels posed a challenge for multiple reasons: there are too many labels to look at individual comparisons of each one, and the labels are not all represented equally so sampled data might miss out entirely on rare emotions such as “grief”. To overcome this problem we used the 6 main Ekman emotion groups to map the 27 emotions excluding samples that only had a “neutral” label. So labels like “grief” and “sadness” are grouped in one label, improving on the issue of having a scarcity of samples in a label group. After collapsing the 27 labels into 6 they can be sampled, explained, and compared more effectively.

#### **Iterative Stratification for Sampling**
In a dataset with 58k samples, it would not be computationally feasible to run the explanations on
every BERT prediction for those samples. The **iterative stratification** method was chosen for its ability to take representative samples of multilabel data. It does this while preserving proportions of label combinations, not just labels themselves. This ensures that the sample we take to explain predictions made by BERT will exhibit the same patterns and correlations as the overall dataset.

### **Methodology of SHAP**
#### **Implementation of SHAP**
Once the data has been sampled, it is translated from input IDs back to raw text, which SHAP needs to explain BERT’s predictions.

**Wrapper Function** The wrapper function converts perturbed samples from raw text to input IDs and attention masks that BERT can process in a forward pass. Logits representing the probabilities are returned, ready for evaluation.

**PartitionExplainer** The `PartitionExplainer` algorithm is called to give the importance scores of each token in each sample taken with iterative stratification. Treating groups of tokens as if they were individuals and iterating through the ones that are estimated to be most important allows the `PartitionExplainer` algorithm to improve time complexity. The output importance scores are saved to a CSV for direct comparison with other explainability methods.

#### **Visualization**
To directly compare the three methods, the importance values are displayed in a 3D scatterplot, which shows the importance score generated by each method on each axis. Barplots of the top tokens by importance score are also generated, as well as the Jaccard similarity scores of the
top-$`n`$ words for each method, displayed in a connected scatterplot. As well as the comparison visualizations generated, Shapley values for some text samples are visualized using the shap library in Python. In the text plots generated in testing, the Shapley values of each token determine the colour and intensity of highlighting over tokens in their raw text form.

### **Methodology of LIME**
#### **Implementation of LIME**
To interpret across all possible emotion labels from the dataset, a LimeTextExplainer is initialized with the list of the 6 Ekman emotion groupings.

**Wrapper Function** As another model-agnostic method, LIME utilizes the same wrapper function used by SHAP to interact with BERT’s output behavior. After accepting raw text as an input, the function tokenizes then converts it to a tensor of token IDs and attention masks. These outputs are applied to a sigmoid function to convert logits into probability scores. LIME can explain a single instance using this wrapper function, resulting in a 6-dimensional vector where each dimension corresponds with a probability score for a given Ekman emotion group.

**Explaining an Instance** The wrapper function is passed to LIME to generate an explanation for a sample of text. The resulting vector of probability scores represents the likelihood of BERT predicting an emotion label. The index corresponding to an Ekman emotion group is selected from the probability vector and passed to the explainer. The explainer computes feature contribution for that label by comparing it to perturbed samples in the local neighbourhood. Since LIME explains individual instances, this process is repeated for all 1000 samples taken using iterative stratification.

#### **Instance Visualization**
LIME explanations can be visualized using a built-in rendering. The feature contributions are represented in two ways:
- A highlight over each word in the instance, where stronger highlights indicate a larger magnitude of contribution towards the selected emotion label.
- A bar plot, where contribution scores are displayed and ranked by magnitude.

These visualizations allow for qualitative analysis of whether BERT is making predictions for appropriate reasons.

#### **Comparison Visualization**
Visualizations of importance values for the top-n tokens are generated and displayed using barplots. Jaccard similarity scores are generated to analyze similarities of these results between each explainability method. Jaccard comparison plots are generated using these scores, along with a 3D scatterplot to compare individual token importance for each method. These visualization methods allow for comparison and analysis of how each method differs in it’s approach to explainability.

### **Methodology of Integrated Gradients**
#### **Implementation of Integrated Gradients**
**Forward Function** The forward function processes input data through the neural network. Logits that represent prediction values are output from the forward function. Input IDs and attention masks are taken as input.

**Baseline Input** Next is the generation of the baseline input. The baseline input, is a sequence of [CLS], [SEP], and [PAD] tokens that together represent an empty input of the length of the text sample. The sequence begins with [CLS], is padded with [PAD], and [SEP] tokens that are placed corresponding to sentence breaks in the input IDs. A custom function is used to create the baseline based on the input IDs in the sample of GoEmotions data. The baseline is then evaluated for feature attribution.

**The Evaluation Loop** The evaluation loop will analyze the 1000 textual examples in the GoEmotions dataset, provided by the iterative stratification step. This step is repeated once per emotion label, demonstrating the necessity of the Ekman mapping step. The evaluation loop calculates the importance score for each token in each sample. This step prepares these scores for each sample as data to be saved and visualized, in comparison with the other two methods.

#### **Visualization**
As well as the comparison visualizations that are generated to test the importance scores and similarity with the other methods, visualizations are used to test the validity of the data and pipeline. Within the evaluation loop, visualizations are generated to explain each piece of text. The visualization first produces a textual output that outlines the current example and the top predicted emotion found within the input, along with the confidence score from BERT’s prediction. The true label of each emotion, the importance scores, and the display of the raw text with colour highlighting corresponding to the intensity and sign of the importance score are displayed for each sample.

### **Evaluation Metrics**
In our comparison analysis, we use Jaccard similarity and a comparison of importance scores to assess each explanation method at quantifying token importance.
#### **Importance Score Comparison**
In both the Top Tokens by Importance Score and Importance Score Distribution Sections, importance scores of individual tokens are used to generate comparative visualizations. Each explanation method outputs an **importance score** for each token between -1 and 1. The tokens with the highest magnitude of impact on the output prediction by BERT have the largest absolute value importance score. A positively-signed importance score pushes the BERT prediction toward a target emotion label, making it likely to have a higher degree of confidence, while a negatively-signed importance score pushes the prediction away from the label.

In the section [**Project Testing Results**](#project-testing-results), Top Tokens by Importance Score, the top 20 tokens are displayed alongside their
importance scores in barchart format for each of the three explanation methods. The Importance Score Distribution, the importance score obtained by each of the three methods is plotted at once for each sample by label, using 3 axes to represent the three explanation methods in a 3D scatter plot.

Different areas of the output graph will correspond to varying degrees of agreement or disagreement by the methods on the importance magnitude of various tokens. The line where $`x = y = z`$ describes perfect agreement between the three methods. Outliers from this line show where the methods disagree on those tokens. The closer a point is to the origin, the smaller its importance score, while the tokens farther from the origin have a greater importance score.

#### **Jaccard Similarity**
When measuring how similar the top-n tokens are between different explanation methods, Jaccard Similarity index is used as the measurement of agreement between the resulting sets. When looking at two sets of top-n tokens, Jaccard Similarity is the ratio between the intersection and the union of the two sets, or $`\begin{align}J(A,B) = \frac{|A \cap B|}{|A \cup B|}\end{align}`$. This results in a number between 0 and 1, with 0 being no agreement at all and 1 being perfect agreement between the two sets. This project evaluated the Jaccard similarity index for the top-n tokens of each explanation method with n between 5 and 50, capturing the most impactful tokens by importance score. For $`n > 50`$, the importance scores, at least for LIME and SHAP, drop off sharply to < 0.2, signifying less impact by those tokens on the predictions by BERT.

## **Discussion**
#### **Inter-Method Consistency**
SHAP and LIME have a greater Jaccard similarity score with each other than either do with Integrated Gradients, consistently for all the Ekman emotion groups. In part, this is because they differ in their ability to capture the tokens with a negative impact on the BERT prediction. Integrated Gradients evaluates clear groups of words that detract from a given label, while SHAP and LIME have very few negative attributions that are very small in magnitude. SHAP and LIME rely on perturbation to function, a model-agnostic method that does not access the internal
gradients of the model. Perturbation methods see the surface-level functioning of the model, but may not be as sensitive as Integrated Gradients is to the impact of each token. Perturbation methods may also be unable to capture the ability of a token to detract from a label consistently,
as Integrated Gradients does.

There are many words that Integrated Gradients identifies as important, which are neutral according to SHAP and LIME. This also detracts from the Jaccard similarity score when comparing LIME and SHAP with IG. Integrated Gradients spreads attribution over many tokens and is not parsimonious, while SHAP and LIME focus importance scores on the top-20 tokens. The gradient based method seems more sensitive to words that do not have a large impact, but still have a measurable impact on the prediction made about a sample.

## **Project Testing Results**
<img width="1984" height="784" alt="topwords_SURPRISE" src="https://github.com/user-attachments/assets/5e70963a-5812-4928-88dd-71ee9680eea4"/>
<img width="1000" height="800" alt="3d_plot_SURPRISE" src="https://github.com/user-attachments/assets/3cafd3b0-5e31-4574-a660-0c249ddc8115"/>
<img width="906" height="853" alt="SHAPglobalSURPRISE" src="https://github.com/user-attachments/assets/868942ad-1079-44b1-882b-be2440e56e88" />
<img width="850" height="554" alt="jaccard_surprise" src="https://github.com/user-attachments/assets/2fb743ab-1906-42ce-94b3-0295b2194d20" />


## **Author(s)**
- **Tobenna Nnaobi**
- **Marian Waffle**
- **Jin Sutharman**

**Copyright &copy; 2026. All Rights Reserved.**
