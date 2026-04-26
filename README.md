# **IMCS3020U Project: Investigating Interpretability of BERT for Emotion Classification by Comparing Shapley Additive Explanations (SHAP), Local Interpretable Model-Agnostic Explanations (LIME), and Integrated Gradients**

## **Table of Contents**
- [***Project Information***](#project-information)
- [***Mathematical & Computational Backgrounds***](#mathematical--computational-backgrounds)
- [***Getting Started***](#getting-started)
- [***Project Testing Results***](#project-testing-results)

## **Project Information**

### **Overview**
This project was developed as a chosen topical project for the **IMCS3020U: Integrated Project Course II** during the 2026 Winter Semester at Ontario Tech University. Its main purpose is aiming to compare methods of explaining the predictions made by BERT for applications such as mental health support and diagnostics or hate speech detection so that the process by which they are made is transparent, quantifiable, and trustworthy.

### **Description**
In a previous project, BERT was compared with MLkNN at the task of emotion classification on the GoEmotions dataset. While BERT performed considerably better than MLkNN based on the F1-score, the process that allows it to categorize text by emotion content is not as easy to understand as with the MLkNN algorithm. In this project we seek to explain and interpret the results of using BERT to categorize emotion in text samples, by applying three frameworks to the existing BERT pipeline from the previous project: LIME, SHAP, and Integrated Gradients.

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

### **Project Contributor(s)**
- **Tobenna Nnaobi**
- **Marian Waffle**
- **Jin Sutharman**

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


## **Project Testing Results**
<img width="1984" height="784" alt="topwords_SURPRISE" src="https://github.com/user-attachments/assets/5e70963a-5812-4928-88dd-71ee9680eea4"/>
<img width="1000" height="800" alt="3d_plot_SURPRISE" src="https://github.com/user-attachments/assets/3cafd3b0-5e31-4574-a660-0c249ddc8115"/>
<img width="906" height="853" alt="SHAPglobalSURPRISE" src="https://github.com/user-attachments/assets/868942ad-1079-44b1-882b-be2440e56e88" />
<img width="850" height="554" alt="jaccard_surprise" src="https://github.com/user-attachments/assets/2fb743ab-1906-42ce-94b3-0295b2194d20" />


## **Author(s)**
- **Tobenna Nnaobi**
- **Marian Waffle**
- **Jin Sutharman**
