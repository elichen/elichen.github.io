const allPuzzles = [
    {
        id: 1,
        domain: "Neural Networks",
        term: "NEURON",
        hint: "The basic computational unit of the brain and artificial neural networks.",
        explanation: "A neuron is the fundamental unit in neural networks, inspired by biological neurons. It receives inputs, processes them, and produces an output."
    },
    {
        id: 2,
        domain: "Machine Learning",
        term: "SIGMOID",
        hint: "An S-shaped activation function commonly used in neural networks.",
        explanation: "The sigmoid function maps any input to a value between 0 and 1, making it useful for binary classification problems and as an activation function in neural networks."
    },
    {
        id: 3,
        domain: "Natural Language Processing",
        term: "TOKENIZE",
        hint: "The process of breaking down text into smaller units for analysis.",
        explanation: "Tokenization is a fundamental step in NLP where text is divided into individual words, subwords, or characters to be processed by algorithms."
    },
    {
        id: 4,
        domain: "Computer Vision",
        term: "CONVOLUTION",
        hint: "A key operation in CNNs for feature extraction from images.",
        explanation: "Convolution involves sliding a small matrix (kernel) over an image to detect features, forming the basis of convolutional neural networks used in image processing."
    },
    {
        id: 5,
        domain: "Reinforcement Learning",
        term: "QLEARNING",
        hint: "A model-free algorithm for learning optimal action-selection policy.",
        explanation: "Q-learning is a reinforcement learning technique that learns the value of actions in states, allowing an agent to make optimal decisions without a model of the environment."
    },
    {
        id: 6,
        domain: "Optimization",
        term: "GRADIENT",
        hint: "The direction of steepest increase in a function, crucial for many optimization algorithms.",
        explanation: "Gradients are used in various optimization techniques, particularly in training neural networks through backpropagation to minimize the loss function."
    },
    {
        id: 7,
        domain: "Clustering",
        term: "KMEANS",
        hint: "An unsupervised learning algorithm that groups similar data points.",
        explanation: "K-means clustering partitions data into K clusters, each represented by the mean of its points, widely used for data segmentation and feature learning."
    },
    {
        id: 8,
        domain: "Dimensionality Reduction",
        term: "PCA",
        hint: "A technique for reducing the dimensionality of data while preserving its variance.",
        explanation: "Principal Component Analysis (PCA) is used to simplify complex datasets by transforming them into fewer dimensions which still retain most of the original information."
    },
    {
        id: 9,
        domain: "Deep Learning",
        term: "BACKPROP",
        hint: "The primary algorithm for training neural networks by adjusting weights.",
        explanation: "Backpropagation calculates the gradient of the loss function with respect to the network's weights, enabling effective training of deep neural networks."
    },
    {
        id: 10,
        domain: "Ethics in AI",
        term: "BIAS",
        hint: "A systematic error in AI systems that can lead to unfair outcomes.",
        explanation: "Bias in AI can result from skewed training data or flawed algorithms, leading to discriminatory or unfair decisions, a critical concern in AI ethics."
    },
    {
        id: 11,
        domain: "Natural Language Processing",
        term: "BERT",
        hint: "A transformer-based machine learning technique for NLP pre-training.",
        explanation: "BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based machine learning technique for NLP. It has advanced the state of the art for many NLP tasks."
    },
    {
        id: 12,
        domain: "Computer Vision",
        term: "YOLO",
        hint: "A real-time object detection system, abbreviated name.",
        explanation: "YOLO (You Only Look Once) is a state-of-the-art, real-time object detection system that can detect multiple objects in an image in a single forward pass of the network."
    },
    {
        id: 13,
        domain: "Reinforcement Learning",
        term: "POLICY",
        hint: "A strategy or rule for an agent to choose actions in reinforcement learning.",
        explanation: "In reinforcement learning, a policy is a strategy that the agent employs to determine the next action based on the current state. It maps states to actions."
    },
    {
        id: 14,
        domain: "Neural Networks",
        term: "LSTM",
        hint: "A type of RNN architecture designed to handle long-term dependencies.",
        explanation: "Long Short-Term Memory (LSTM) networks are a type of recurrent neural network capable of learning long-term dependencies, useful for sequential data like text or time series."
    },
    {
        id: 15,
        domain: "Machine Learning",
        term: "ENSEMBLE",
        hint: "A method that combines multiple learning algorithms to improve performance.",
        explanation: "Ensemble methods use multiple learning algorithms to obtain better predictive performance than could be obtained from any of the constituent learning algorithms alone."
    },
    {
        id: 16,
        domain: "Natural Language Processing",
        term: "WORDVEC",
        hint: "A technique to map words or phrases to vectors of real numbers.",
        explanation: "Word Vector (Word2Vec) is a technique for natural language processing that represents words as vectors in a multidimensional space, capturing semantic relationships."
    },
    {
        id: 17,
        domain: "Computer Vision",
        term: "SEGMENTATION",
        hint: "The process of partitioning a digital image into multiple segments.",
        explanation: "Image segmentation is the process of dividing an image into multiple segments or objects, often used to locate objects and boundaries in images."
    },
    {
        id: 18,
        domain: "Reinforcement Learning",
        term: "MARKOV",
        hint: "A property where the future state depends only on the current state.",
        explanation: "The Markov property in reinforcement learning states that the future is independent of the past given the present, simplifying the modeling of decision processes."
    },
    {
        id: 19,
        domain: "Optimization",
        term: "ADAM",
        hint: "An algorithm for first-order gradient-based optimization of stochastic objective functions.",
        explanation: "Adam (Adaptive Moment Estimation) is an optimization algorithm that can handle sparse gradients on noisy problems, often used in training deep learning models."
    },
    {
        id: 20,
        domain: "Neural Networks",
        term: "DROPOUT",
        hint: "A regularization technique to prevent overfitting in neural networks.",
        explanation: "Dropout is a technique where randomly selected neurons are ignored during training, helping to prevent overfitting by making the network less sensitive to specific weights."
    },
    {
        id: 21,
        domain: "Machine Learning",
        term: "BAGGING",
        hint: "An ensemble meta-algorithm to improve stability and accuracy.",
        explanation: "Bagging (Bootstrap Aggregating) involves training multiple models on random subsets of the training data and then aggregating their predictions to reduce variance and overfitting."
    },
    {
        id: 22,
        domain: "Natural Language Processing",
        term: "STEMMING",
        hint: "The process of reducing words to their word stem or root form.",
        explanation: "Stemming is a text normalization technique used to reduce words to their base or root form, helping to treat different word forms as a single item and improve text analysis."
    },
    {
        id: 23,
        domain: "Computer Vision",
        term: "IMAGENET",
        hint: "A large visual database designed for use in visual object recognition software research.",
        explanation: "ImageNet is a dataset of over 14 million images designed to train and benchmark computer vision models, playing a crucial role in the deep learning revolution in computer vision."
    },
    {
        id: 24,
        domain: "Reinforcement Learning",
        term: "REWARD",
        hint: "A fundamental concept in reinforcement learning, signaling the desirability of an event.",
        explanation: "In reinforcement learning, a reward is a scalar feedback signal that indicates how well an agent is doing at a given time step, guiding the agent to learn optimal behavior."
    },
    {
        id: 25,
        domain: "Ethics in AI",
        term: "FAIRNESS",
        hint: "The impartial and just treatment of people in AI systems.",
        explanation: "Fairness in AI refers to the absence of discrimination or bias in machine learning models and AI systems, ensuring equitable treatment across different groups."
    },
    {
        id: 26,
        domain: "Deep Learning",
        term: "GAN",
        hint: "A class of ML systems where two neural networks contest with each other.",
        explanation: "Generative Adversarial Networks (GANs) consist of two networks, a generator and a discriminator, that compete against each other, often used to generate realistic synthetic data."
    },
    {
        id: 27,
        domain: "Machine Learning",
        term: "SVM",
        hint: "A supervised learning model used for classification and regression analysis.",
        explanation: "Support Vector Machines (SVM) are powerful and flexible supervised algorithms used for both classification and regression tasks, known for their ability to handle non-linear decision boundaries."
    },
    {
        id: 28,
        domain: "Natural Language Processing",
        term: "CORPUS",
        hint: "A large and structured set of texts used in language studies.",
        explanation: "A corpus is a large collection of texts used for linguistic analysis and natural language processing tasks, serving as a fundamental resource for training and testing NLP models."
    },
    {
        id: 29,
        domain: "Computer Vision",
        term: "OPENCV",
        hint: "An open-source computer vision and machine learning software library.",
        explanation: "OpenCV (Open Source Computer Vision Library) is a library of programming functions mainly aimed at real-time computer vision, widely used in both academic and commercial applications."
    },
    {
        id: 30,
        domain: "Reinforcement Learning",
        term: "EXPLOIT",
        hint: "In RL, the strategy of choosing actions known to have high rewards.",
        explanation: "Exploitation in reinforcement learning refers to the agent's strategy of choosing actions that are known to yield high rewards based on current knowledge, as opposed to exploration."
    },
    {
        id: 31,
        domain: "Optimization",
        term: "SGD",
        hint: "A stochastic approximation of gradient descent optimization.",
        explanation: "Stochastic Gradient Descent (SGD) is an iterative method for optimizing an objective function, used widely in machine learning due to its efficiency with large datasets."
    },
    {
        id: 32,
        domain: "Neural Networks",
        term: "RNN",
        hint: "A class of neural networks where connections between nodes form a directed graph along a temporal sequence.",
        explanation: "Recurrent Neural Networks (RNNs) are a class of neural networks designed to recognize patterns in sequences of data, such as text, genomes, handwriting, or numerical time series data."
    },
    {
        id: 33,
        domain: "Machine Learning",
        term: "BOOSTING",
        hint: "An ensemble technique that combines weak learners to create a strong learner.",
        explanation: "Boosting is a machine learning ensemble technique that combines several weak learners to produce a strong learner, often used to improve model accuracy and reduce bias."
    },
    {
        id: 34,
        domain: "Natural Language Processing",
        term: "GLOVE",
        hint: "An unsupervised learning algorithm for obtaining vector representations for words.",
        explanation: "GloVe (Global Vectors for Word Representation) is an unsupervised learning algorithm for generating word embeddings by capturing global word-word co-occurrence statistics from a corpus."
    },
    {
        id: 35,
        domain: "Computer Vision",
        term: "RCNN",
        hint: "A two-stage object detection algorithm using region proposals.",
        explanation: "Region-based Convolutional Neural Networks (R-CNN) is a family of techniques for object detection in images that combine region proposals with CNNs to efficiently detect and localize objects."
    },
    {
        id: 36,
        domain: "Reinforcement Learning",
        term: "SARSA",
        hint: "An on-policy reinforcement learning algorithm for temporal difference learning.",
        explanation: "SARSA (State-Action-Reward-State-Action) is an algorithm for learning a Markov decision process policy, used in reinforcement learning to learn from interactions with the environment."
    },
    {
        id: 37,
        domain: "Ethics in AI",
        term: "PRIVACY",
        hint: "The protection of personal information in AI systems.",
        explanation: "Privacy in AI refers to the responsible handling and protection of personal data used in machine learning models and AI systems, a critical aspect of ethical AI development."
    },
    {
        id: 38,
        domain: "Deep Learning",
        term: "AUTOENCODER",
        hint: "A type of neural network used to learn efficient codings of unlabeled data.",
        explanation: "Autoencoders are a type of artificial neural network used to learn efficient data codings in an unsupervised manner, often used for dimensionality reduction and generative modeling."
    },
    {
        id: 39,
        domain: "Machine Learning",
        term: "KNN",
        hint: "A non-parametric method used for classification and regression.",
        explanation: "K-Nearest Neighbors (KNN) is a simple, versatile algorithm that stores all available cases and classifies new cases based on a similarity measure, used in both classification and regression tasks."
    },
    {
        id: 40,
        domain: "Natural Language Processing",
        term: "NLTK",
        hint: "A leading platform for building Python programs to work with human language data.",
        explanation: "The Natural Language Toolkit (NLTK) is a suite of libraries and programs for symbolic and statistical natural language processing for Python, widely used in NLP research and development."
    },
    {
        id: 41,
        domain: "Computer Vision",
        term: "HAAR",
        hint: "A machine learning object detection method used to detect faces.",
        explanation: "Haar Cascade is a machine learning based approach where a cascade function is trained from a lot of positive and negative images, commonly used for face detection in computer vision."
    },
    {
        id: 42,
        domain: "Reinforcement Learning",
        term: "AGENT",
        hint: "The learner or decision maker in a reinforcement learning scenario.",
        explanation: "In reinforcement learning, an agent is the entity that learns to make decisions by interacting with an environment, aiming to maximize some notion of cumulative reward."
    },
    {
        id: 43,
        domain: "Optimization",
        term: "MOMENTUM",
        hint: "A method to accelerate gradient descent that accumulates a velocity vector.",
        explanation: "Momentum is a method that helps accelerate gradient descent in the relevant direction and dampens oscillations, often leading to faster convergence in optimization problems."
    },
    {
        id: 44,
        domain: "Neural Networks",
        term: "CNN",
        hint: "A class of deep neural networks, most commonly applied to analyzing visual imagery.",
        explanation: "Convolutional Neural Networks (CNNs) are a class of deep learning models most commonly applied to analyzing visual imagery and are inspired by biological processes in the visual cortex."
    },
    {
        id: 45,
        domain: "Machine Learning",
        term: "CROSSVAL",
        hint: "A model validation technique for assessing how results will generalize to an independent data set.",
        explanation: "Cross-validation is a statistical method used to estimate the skill of machine learning models, helping to assess how the model will generalize to an independent dataset."
    },
    {
        id: 46,
        domain: "Natural Language Processing",
        term: "TFIDF",
        hint: "A numerical statistic intended to reflect how important a word is to a document in a collection.",
        explanation: "TF-IDF (Term Frequency-Inverse Document Frequency) is a statistical measure used to evaluate the importance of a word to a document in a collection or corpus, often used in information retrieval and text mining."
    },
    {
        id: 47,
        domain: "Computer Vision",
        term: "SIFT",
        hint: "A feature detection algorithm in computer vision to detect and describe local features in images.",
        explanation: "Scale-Invariant Feature Transform (SIFT) is an algorithm to detect and describe local features in images, useful for tasks like object recognition, robotic mapping, and navigation."
    },
    {
        id: 48,
        domain: "Reinforcement Learning",
        term: "EPSILON",
        hint: "A parameter used in the epsilon-greedy strategy for balancing exploration and exploitation.",
        explanation: "Epsilon in the epsilon-greedy strategy determines the probability of choosing a random action (exploration) versus choosing the best known action (exploitation) in reinforcement learning."
    },
    {
        id: 49,
        domain: "Ethics in AI",
        term: "TRANSPARENCY",
        hint: "The quality of being clear, obvious, and understandable in AI decision-making processes.",
        explanation: "Transparency in AI refers to the openness and clarity in how AI systems make decisions, crucial for building trust and ensuring accountability in AI applications."
    },
    {
        id: 50,
        domain: "Deep Learning",
        term: "TRANSFER",
        hint: "A machine learning method where a model developed for a task is reused as the starting point for a model on a second task.",
        explanation: "Transfer learning is a technique where a model trained on one task is re-purposed on a second related task, often resulting in faster training and better performance, especially with limited data."
    },
    {
        id: 51,
        domain: "Machine Learning",
        term: "OVERFITTING",
        hint: "When a model learns the training data too well, including noise and fluctuations.",
        explanation: "Overfitting occurs when a model learns the training data too well, capturing noise and random fluctuations, leading to poor generalization on new, unseen data."
    },
    {
        id: 52,
        domain: "Natural Language Processing",
        term: "LEMMATIZATION",
        hint: "The process of reducing words to their base or dictionary form.",
        explanation: "Lemmatization is the process of reducing words to their lemma or dictionary form, considering the context and part of speech to determine the correct base form."
    },
    {
        id: 53,
        domain: "Computer Vision",
        term: "RESNET",
        hint: "A specific architecture of CNN that uses skip connections.",
        explanation: "ResNet (Residual Networks) is a specific architecture of convolutional neural networks that utilizes skip connections to allow training of very deep networks, often used in image recognition tasks."
    },
    {
        id: 54,
        domain: "Reinforcement Learning",
        term: "BANDIT",
        hint: "A simplified reinforcement learning scenario with a single state and multiple actions.",
        explanation: "The Multi-armed Bandit problem is a classic reinforcement learning scenario where an agent must choose between multiple actions (arms) to maximize cumulative reward, balancing exploration and exploitation."
    },
    {
        id: 55,
        domain: "Optimization",
        term: "BATCHNORM",
        hint: "A technique to improve the speed, performance, and stability of artificial neural networks.",
        explanation: "Batch Normalization is a technique used to improve the training of artificial neural networks by normalizing the inputs to each layer, which can lead to faster training and better performance."
    },
    {
        id: 56,
        domain: "Neural Networks",
        term: "ATTENTION",
        hint: "A mechanism in neural networks that allows the model to focus on specific parts of the input.",
        explanation: "Attention mechanisms in neural networks allow the model to focus on specific parts of the input when producing output, greatly improving performance in tasks like machine translation and image captioning."
    },
    {
        id: 57,
        domain: "Machine Learning",
        term: "REGULARIZATION",
        hint: "A technique used to prevent overfitting by adding a penalty term to the loss function.",
        explanation: "Regularization is a set of techniques used to prevent overfitting in machine learning models by adding a penalty term to the loss function, discouraging complex models."
    },
    {
        id: 58,
        domain: "Natural Language Processing",
        term: "ELMO",
        hint: "A deep contextualized word representation that models both complex characteristics of word use and how these uses vary across linguistic contexts.",
        explanation: "ELMo (Embeddings from Language Models) provides deep contextualized word representations that capture complex characteristics of word use and how they vary across linguistic contexts, improving many NLP tasks."
    },
    {
        id: 59,
        domain: "Computer Vision",
        term: "MASKRCNN",
        hint: "An extension of Faster R-CNN that adds a branch for predicting segmentation masks on each Region of Interest.",
        explanation: "Mask R-CNN extends Faster R-CNN by adding a branch for predicting segmentation masks on each Region of Interest, allowing for instance segmentation in addition to object detection."
    },
    {
        id: 60,
        domain: "Reinforcement Learning",
        term: "MONTE CARLO",
        hint: "A class of algorithms that rely on repeated random sampling to obtain numerical results.",
        explanation: "Monte Carlo methods in reinforcement learning use repeated random sampling to solve problems, often used for estimating state values and action-value functions based on experience."
    },
    {
        id: 61,
        domain: "Ethics in AI",
        term: "ACCOUNTABILITY",
        hint: "The quality of being responsible and answerable for AI system decisions and actions.",
        explanation: "Accountability in AI refers to the principle that organizations and individuals should be answerable for the decisions and actions of AI systems they develop or deploy."
    },
    {
        id: 62,
        domain: "Deep Learning",
        term: "FINETUNING",
        hint: "The process of taking a pre-trained model and adapting it to a similar but distinct task.",
        explanation: "Fine-tuning involves taking a pre-trained model and further training it on a similar but distinct task, often with a smaller dataset, to adapt the model's knowledge to the new task."
    },
    {
        id: 63,
        domain: "Machine Learning",
        term: "FEATUREENG",
        hint: "The process of using domain knowledge to extract features from raw data.",
        explanation: "Feature engineering is the process of using domain knowledge to select and transform the most relevant variables from raw data when creating a predictive model."
    },
    {
        id: 64,
        domain: "Natural Language Processing",
        term: "BLEU",
        hint: "An algorithm for evaluating the quality of machine-translated text.",
        explanation: "BLEU (Bilingual Evaluation Understudy) is an algorithm for evaluating the quality of text which has been machine-translated from one natural language to another, widely used in machine translation evaluation."
    },
    {
        id: 65,
        domain: "Computer Vision",
        term: "HOUGH",
        hint: "A feature extraction technique used in image analysis, computer vision, and digital image processing.",
        explanation: "The Hough transform is a feature extraction technique used to detect simple shapes like lines, circles, or ellipses in an image, commonly used in computer vision applications."
    },
    {
        id: 66,
        domain: "Reinforcement Learning",
        term: "TRAJECTORY",
        hint: "A sequence of states and actions in reinforcement learning.",
        explanation: "In reinforcement learning, a trajectory (or episode) is a sequence of states and actions that an agent experiences while interacting with an environment."
    },
    {
        id: 67,
        domain: "Optimization",
        term: "RMSPROP",
        hint: "An optimization algorithm that adapts the learning rate for each parameter.",
        explanation: "RMSprop (Root Mean Square Propagation) is an optimization algorithm that adapts the learning rate for each parameter, helping to address the diminishing learning rate problem in standard gradient descent."
    },
    {
        id: 68,
        domain: "Neural Networks",
        term: "PRUNING",
        hint: "A technique for reducing the size of neural networks by removing unnecessary connections.",
        explanation: "Pruning is a technique used to reduce the size and computational requirements of neural networks by removing unnecessary connections or neurons, often with minimal impact on performance."
    },
    {
        id: 69,
        domain: "Machine Learning",
        term: "ANOMALY",
        hint: "The identification of rare items, events or observations that differ significantly from the majority of the data.",
        explanation: "Anomaly detection is the identification of rare items, events or observations which raise suspicions by differing significantly from the majority of the data, often used in fraud detection and system health monitoring."
    },
    {
        id: 70,
        domain: "Natural Language Processing",
        term: "GLUE",
        hint: "A collection of tasks used as a benchmark for natural language understanding systems.",
        explanation: "The General Language Understanding Evaluation (GLUE) benchmark is a collection of diverse natural language understanding tasks, used to evaluate and compare the performance of different NLP models."
    },
    {
        id: 71,
        domain: "Computer Vision",
        term: "GANS",
        hint: "A class of machine learning frameworks for generating new data.",
        explanation: "Generative Adversarial Networks (GANs) are a class of machine learning frameworks where two neural networks contest with each other to generate new, synthetic instances of data that can pass for real data."
    },
    {
        id: 72,
        domain: "Reinforcement Learning",
        term: "BELLMAN",
        hint: "A fundamental equation in reinforcement learning that expresses the value of a state in terms of the values of its successor states.",
        explanation: "The Bellman equation is a fundamental equation in reinforcement learning that expresses the value of a state in terms of the immediate reward and the discounted value of the next state, forming the basis for many RL algorithms."
    },
    {
        id: 73,
        domain: "Ethics in AI",
        term: "EXPLAINABLE",
        hint: "The quality of AI systems that allows humans to understand the reasoning behind their decisions.",
        explanation: "Explainable AI refers to methods and techniques in the application of artificial intelligence such that the results of the solution can be understood by humans, crucial for trust and accountability."
    },
    {
        id: 74,
        domain: "Deep Learning",
        term: "EMBEDDING",
        hint: "A learned representation for text where words with similar meaning have a similar representation.",
        explanation: "Word embeddings are learned representations of text where words that have the same meaning have a similar representation, often used as input to deep learning models in NLP tasks."
    },
    {
        id: 75,
        domain: "Machine Learning",
        term: "TSNE",
        hint: "A technique for dimensionality reduction that is particularly well suited for the visualization of high-dimensional datasets.",
        explanation: "t-SNE (t-distributed stochastic neighbor embedding) is a machine learning algorithm for visualization that reduces dimensionality while trying to keep similar instances close and dissimilar instances apart."
    },
    {
        id: 76,
        domain: "Natural Language Processing",
        term: "SENTIMENT",
        hint: "The process of determining the emotional tone behind a series of words.",
        explanation: "Sentiment analysis is the use of natural language processing to determine the emotional tone behind words, used to gain an understanding of attitudes, opinions and emotions in written text."
    },
    {
        id: 77,
        domain: "Computer Vision",
        term: "UNET",
        hint: "A convolutional network architecture for fast and precise segmentation of images.",
        explanation: "U-Net is a convolutional neural network architecture designed for fast and precise image segmentation, originally developed for biomedical image segmentation but now used in various domains."
    },
    {
        id: 78,
        domain: "Reinforcement Learning",
        term: "ACTOR CRITIC",
        hint: "A framework that combines value-based and policy-based methods.",
        explanation: "Actor-Critic methods in reinforcement learning combine the advantages of value-based and policy-based methods, using an actor to determine the best action and a critic to evaluate the action."
    },
    {
        id: 79,
        domain: "Optimization",
        term: "ADAGRAD",
        hint: "An algorithm for gradient-based optimization that adapts the learning rate to the parameters.",
        explanation: "Adagrad is an optimization algorithm that adapts the learning rate to the parameters, performing smaller updates for frequently occurring features and larger updates for infrequent features."
    },
    {
        id: 80,
        domain: "Neural Networks",
        term: "CAPSULE",
        hint: "A type of artificial neural network that adds structures called capsules to a convolutional neural network.",
        explanation: "Capsule Networks are a type of artificial neural network that adds structures called capsules to CNNs, aiming to address certain limitations of CNNs like viewpoint invariance."
    },
    {
        id: 81,
        domain: "Machine Learning",
        term: "GRIDCV",
        hint: "A technique to perform hyperparameter tuning for estimators.",
        explanation: "Grid Search Cross-Validation is an approach to parameter tuning that will methodically build and evaluate a model for each combination of algorithm parameters specified in a grid."
    },
    {
        id: 82,
        domain: "Natural Language Processing",
        term: "WORDNET",
        hint: "A lexical database of semantic relations between words in more than 200 languages.",
        explanation: "WordNet is a large lexical database of English. Nouns, verbs, adjectives and adverbs are grouped into sets of cognitive synonyms, each expressing a distinct concept."
    },
    {
        id: 83,
        domain: "Computer Vision",
        term: "DEPTHMAP",
        hint: "An image that contains information relating to the distance of the surfaces of scene objects from a viewpoint.",
        explanation: "A depth map is an image or image channel that contains information relating to the distance of the surfaces of scene objects from a viewpoint, often used in 3D computer vision applications."
    },
    {
        id: 84,
        domain: "Reinforcement Learning",
        term: "MDPOLICY",
        hint: "A solution to a Markov Decision Process.",
        explanation: "An MDP Policy is a solution to a Markov Decision Process, specifying the action an agent should take in each state to maximize its expected cumulative reward."
    },
    {
        id: 85,
        domain: "Optimization",
        term: "ADADELTA",
        hint: "An extension of Adagrad that seeks to reduce its aggressive, monotonically decreasing learning rate.",
        explanation: "Adadelta is an extension of Adagrad that seeks to reduce its aggressive, monotonically decreasing learning rate. It adapts learning rates based on a moving window of gradient updates, instead of accumulating all past gradients."
    },
    {
        id: 86,
        domain: "Neural Networks",
        term: "BPTT",
        hint: "The main algorithm used to train recurrent neural networks through time.",
        explanation: "Backpropagation Through Time (BPTT) is the main algorithm used to train recurrent neural networks, allowing the network to learn from sequential data by unrolling the network through time."
    },
    {
        id: 87,
        domain: "Machine Learning",
        term: "SHAP",
        hint: "A game theoretic approach to explain the output of any machine learning model.",
        explanation: "SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model, providing a unified measure of feature importance across different models."
    },
    {
        id: 88,
        domain: "Natural Language Processing",
        term: "ROUGE",
        hint: "A set of metrics used for evaluating automatic summarization of texts.",
        explanation: "ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a set of metrics used for evaluating automatic summarization and machine translation in natural language processing."
    },
    {
        id: 89,
        domain: "Computer Vision",
        term: "FASTRCNN",
        hint: "An object detection algorithm that improves training and testing speed while also increasing detection accuracy.",
        explanation: "Fast R-CNN is an object detection algorithm that improves both speed and accuracy compared to R-CNN by processing the entire image with a CNN and then using ROI pooling to extract features for each region proposal."
    },
    {
        id: 90,
        domain: "Reinforcement Learning",
        term: "ELIGIBILITY",
        hint: "A mechanism used in reinforcement learning to assign credit to actions that led to rewards.",
        explanation: "Eligibility traces in reinforcement learning provide a mechanism for assigning credit to actions that led to rewards, even if those actions occurred several steps earlier in the episode."
    },
    {
        id: 91,
        domain: "Ethics in AI",
        term: "ALIGNMENT",
        hint: "The challenge of creating AI systems that behave in accordance with human values and intentions.",
        explanation: "AI alignment refers to the challenge of creating artificial intelligence systems that behave in accordance with human values and intentions, ensuring that AI systems remain beneficial to humanity as they become more powerful."
    },
    {
        id: 92,
        domain: "Deep Learning",
        term: "DISTILLATION",
        hint: "A technique for transferring knowledge from a large model to a smaller one.",
        explanation: "Knowledge distillation is a technique for transferring knowledge from a large, complex model (the teacher) to a smaller, simpler model (the student), often used to create more efficient models for deployment."
    },
    {
        id: 93,
        domain: "Machine Learning",
        term: "DBSCAN",
        hint: "A density-based clustering algorithm that groups together points that are closely packed together.",
        explanation: "DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a clustering algorithm that groups together points that are closely packed together, marking as outliers points that lie alone in low-density regions."
    },
    {
        id: 94,
        domain: "Natural Language Processing",
        term: "COREFERENCE",
        hint: "The task of finding all expressions that refer to the same entity in a text.",
        explanation: "Coreference resolution is the task of finding all expressions that refer to the same entity in a text. It's a crucial step in many NLP applications, including machine translation and text summarization."
    },
    {
        id: 95,
        domain: "Computer Vision",
        term: "RANSAC",
        hint: "An iterative method to estimate parameters of a mathematical model from a set of observed data that contains outliers.",
        explanation: "RANSAC (Random Sample Consensus) is an iterative method used in computer vision to estimate parameters of a model from a set of observed data containing outliers, often used in image stitching and 3D reconstruction."
    },
    {
        id: 96,
        domain: "Reinforcement Learning",
        term: "DREAMER",
        hint: "A reinforcement learning agent that learns a world model and uses it for planning.",
        explanation: "Dreamer is a reinforcement learning agent that learns a world model of its environment and uses it for planning. It can solve long-horizon tasks from images with significantly better data-efficiency and computation time than previous approaches."
    },
    {
        id: 97,
        domain: "Optimization",
        term: "NADAM",
        hint: "An optimization algorithm that combines Adam and Nesterov accelerated gradient.",
        explanation: "Nadam (Nesterov-accelerated Adaptive Moment Estimation) is an optimization algorithm that combines Adam optimizer with Nesterov accelerated gradient, often leading to faster convergence in training neural networks."
    },
    {
        id: 98,
        domain: "Neural Networks",
        term: "SQUEEZE",
        hint: "A type of neural network layer that reduces the number of channels in a feature map.",
        explanation: "Squeeze layers in neural networks reduce the number of channels in a feature map, often used in architectures like SqueezeNet to create compact models with fewer parameters while maintaining performance."
    },
    {
        id: 99,
        domain: "Machine Learning",
        term: "SMOTE",
        hint: "A technique to address class imbalance by synthesizing new examples.",
        explanation: "SMOTE (Synthetic Minority Over-sampling Technique) is an oversampling method that creates synthetic examples in a feature space to address class imbalance problems in a dataset."
    },
    {
        id: 100,
        domain: "Ethics in AI",
        term: "ROBUSTNESS",
        hint: "The ability of an AI system to perform well under a wide range of conditions, including unexpected or adversarial inputs.",
        explanation: "Robustness in AI refers to the ability of a system to maintain its performance and reliability under various conditions, including unexpected inputs, adversarial attacks, or changes in the environment, which is crucial for deploying AI systems in real-world scenarios."
    }
];