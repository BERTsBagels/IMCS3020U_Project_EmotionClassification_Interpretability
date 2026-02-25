# **IMCS3020U Project: Investigating Interpretability of BERT for Emotion Classification by Comparing Shapley Additive Explanations (SHAP), Local Interpretable Model-Agnostic Explanations (LIME), and Integrated Gradients**

## **Project Information**

### **Overview**
This project was developed as a chosen topical project for the **IMCS3020U: Integrated Project Course II** during the 2026 Winter Semester at Ontario Tech University. Its main purpose is aiming to compare methods of explaining the predictions made by BERT for applications such as mental health support and diagnostics or hate speech detection so that the process by which they are made is transparent, quantifiable, and trustworthy.

### **Description**
In a previous project, BERT was compared with MLkNN at the task of emotion classification on the GoEmotions dataset. While BERT performed considerably better than MLkNN based on the F1-score, the process that allows it to categorize text by emotion content is not as easy to understand as with the MLkNN algorithm. In this project we seek to explain and interpret the results of using BERT to categorize emotion in text samples, by applying three frameworks to the existing BERT pipeline from the previous project: LIME, SHAP, and Integrated Gradients.

### **Mathematical & Computational Backgrounds**
- **BERT (Bidirectional Encoder Representations from Transformers)**: A model capable of powerful text classification by capturing contextual meaning in its predictions. This model has revolutionized natural language processing, yet its complexity makes explaining its decisions challenging. Understanding BERT’s predictions and their basis is crucial to the safety and trustworthiness of the many applications of this technology.

- **Shapley Additive Explanations (SHAP)**: This concept explains the contributions of various tokens to the end prediction by BERT. It is found in game theory and seeks to fairly divide proceeds of a winning game to players on the same team that have not contributed equally to the outcome. 
    - $`\begin{align} \phi_{i} = \sum_{S\subseteq\{1,...,p\}\{i\}} \frac{\left|S\right|!(p-\left|S\right|-1)!}{p!}[val(S \cup \{i\}) - val(S)] \end{align}`$

-  **Local Interpretable Model-Agnostic Explanations (LIME)**: LIME can be used to produce an explanation for each features contribution to a model's final output. LIME does this by exploring specific instances of a models predictions and perturbing them, essentially removing combinations of features in that instance to examine how it changes the models output. This information is used to fit the instance of data into a more interpretable model. This provides an explainable output for feature contribution.
    - $`\begin{align}\xi(x) = \arg\min\limits_{g \in G} \mathcal{L}(f,g,\pi_{x}) + \Omega(g) \end{align}`$

- **Integrated Gradients**: An attribution technique that explains a model’s prediction by quantifying the contribution of each input feature. It works by accumulating gradients along a straight path from a user-defined baseline input to the actual input. This path integral ensures that the attributions satisfy the fundamental axioms like completeness and sensitivity (non-zero attributions for features that change the prediction).
    - $`\begin{align}\text{IntegratedGrads}_{i}(x) := (x_{i} - x^{\prime}_{i}) \times \int_{\alpha = 0}^{1} \frac{\partial F(x^{\prime} + \alpha \times (x - x^{\prime}))}{\partial x_{i}} d\alpha\end{align}`$

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
-
-
-

## **Project Testing Results**


## **Author(s)**
- **Tobenna Nnaobi**
- **Marian Waffle**
- **Jin Sutharman**
