import pandas as pd
import re
import emoji
import nltk
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
from pyarabic.araby import strip_tashkeel
from gensim.models import Word2Vec, FastText
from scipy.sparse import hstack, csr_matrix
from collections import Counter


# Machine Learning imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


#Data Loading + Cleaning (Label Normalization) + Initial Analysis
texts = []
labels = []

with open("dataset.txt", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        if "\t" in line:
            text, label = line.rsplit("\t", 1)
            texts.append(text)
            labels.append(label)

df = pd.DataFrame({
    "text": texts,
    "label": labels
})

df["label"] = df["label"].replace({
    "NEUTRAL": "OBJ",
    "Neutral": "OBJ",
    "Natural": "OBJ"
})

print("\nLabel distribution: ")
print(df["label"].unique())
print(df["label"].value_counts())


df.to_csv("dataset.csv", index=False, encoding="utf-8-sig")

#Data Preprocessing

arabic_stopwords = set(stopwords.words('arabic'))
# Stemming
stemmer = ISRIStemmer()

# -------------------------
# 1. Normalize Arabic text
# -------------------------
def normalize_arabic(text):
    text = re.sub("#", "", text)
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    text = re.sub("ك\b", "ك", text)
    return text

# -------------------------
# 2. Remove elongation 
# -------------------------
def remove_elongation(text):
    return re.sub(r'(.)\1{2,}', r'\1', text)

# -------------------------
# Main preprocessing function
# -------------------------

months_ar = [
    "يناير", "فبراير", "مارس", "ابريل", "مايو", "يونيو",
    "يوليو", "اغسطس", "سبتمبر", "اكتوبر", "نوفمبر", "ديسمبر"
    ]

def remove_dates(text):
    
    months_pattern = "|".join(months_ar)
    text = re.sub(months_pattern, ' ', text, flags=re.IGNORECASE)
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_arabic_text(text):
    if pd.isna(text):
        return " "

    # Convert to lowercase
    text = text.lower()

    # Remove URLs & HTML tags
    text = re.sub(r"http\S+|www\S+|<.*?>", " ", text)

    # Remove numbers
    text = re.sub(r"\d+", " ", text)

    # Remove emojis (or replace with token if needed)
    text = emoji.replace_emoji(text, replace=" ")

    text = re.sub(r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~؟،؛٪]", " ", text)

    # Remove special characters
    text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)

    # Remove tashkeel
    text = strip_tashkeel(text)

    # Normalize Arabic characters
    text = normalize_arabic(text)

    text = remove_dates(text)

    # Remove elongation
    text = remove_elongation(text)

    tokens = text.split()
    tokens = [w for w in tokens if w not in arabic_stopwords]

    clean_text = " ".join(tokens)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    return clean_text


df = pd.read_csv("dataset.csv", encoding="utf-8-sig")

df['clean_text'] = df['text'].apply(preprocess_arabic_text)

df.to_csv("dataset_preprocessed.csv", index=False, encoding="utf-8-sig")

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================
#1- Handcrafted Features

def extract_handcrafted_features(original_text, clean_text):
    features = {}

    if not isinstance(original_text, str):
        original_text = ""
    if not isinstance(clean_text, str):
        clean_text = ""

    # 1. Tweet Length Features
    features['char_count'] = len(original_text)
    features['word_count'] = len(re.findall(r'\w+', original_text))
    features['avg_word_length'] = np.mean([len(w) for w in original_text.split()]) if original_text.split() else 0
    features['clean_word_count'] = len(clean_text.split())

    # 2. Punctuation Features
    features['exclamation_count'] = original_text.count('!')
    features['question_count'] = original_text.count('?') + original_text.count('؟')
    features['comma_count'] = original_text.count(',') + original_text.count('،')
    features['dots_count'] = len(re.findall(r'\.{2,}', original_text))
    features['total_punctuation'] = sum([
        original_text.count(p) for p in '!?؟،,.:;'
    ])


     # 3. Emojis
    features['has_emoji'] = 1 if emoji.emoji_count(original_text) > 0 else 0
    features['emoji_count'] = emoji.emoji_count(original_text)

    # 4. Hashtags and Mentions
    features['hashtag_count'] = len(re.findall(r'#\w+', original_text))
    features['mention_count'] = len(re.findall(r'@\w+', original_text))
    features['has_hashtag'] = 1 if '#' in original_text else 0
    features['has_mention'] = 1 if '@' in original_text else 0

    # 5. Character Repetition (Elongation)
    elongation_pattern = r'(.)\1{2,}'
    features['elongation_count'] = len(re.findall(elongation_pattern, original_text))
    features['has_elongation'] = 1 if re.search(elongation_pattern, original_text) else 0

    # 6. Word Repetition
    words = clean_text.split()

    if words:
        word_counts = Counter(words)
        repeated_words = [w for w, c in word_counts.items() if c > 1]

        features['repeated_word_count'] = len(repeated_words)
        features['repetition_ratio'] = sum(c for c in word_counts.values() if c > 1) / len(words)
        features['has_repetition'] = 1 if features['repeated_word_count'] > 0 else 0

    else:
        features['repeated_word_count'] = 0
        features['repetition_ratio'] = 0
        features['has_repetition'] = 0


    #7. Negation Indicators 

    negation_words = ['لا', 'ما', 'ليس', 'لم', 'لن', 'غير', 'بدون', 'مش']
    text_lower = original_text.lower()

    features['negation_count'] = sum(text_lower.count(word) for word in negation_words)
    features['has_negation'] = 1 if features['negation_count'] > 0 else 0


    #8. Dialectal Indicators
    egyptian_words = ['فشيخ','وحش','ده','عايز', 'عاوز', 'ازيك', 'ايه', 'كده', 'بتاع', 'اوي', 'خالص']
    features['has_egyptian'] = 1 if any(word in text_lower for word in egyptian_words) else 0

    religious_terms = [
    'الله', 'الرسول', 'القرآن', 'سورة', 'آية', 'الصلاة', 
    'الزكاة', 'الجنة', 'رب', 'اللهم', 'أستغفر الله',
    'الحمد لله', 'سبحان الله', 'إن شاء الله' , 'لا إله إلا الله'
    ]   

    features['religious_term_count'] = sum(1 for term in religious_terms if term in text_lower)
    features['has_religious_term'] = 1 if features['religious_term_count'] > 0 else 0

    positive_words = [
    'جميل', 'الرائع', 'سعيد', 'ممتاز', 'جيد', 'حلو', 'طيب', 'عظيم', 
    'ناجح', 'مفيد', 'راضي', 'مبسوط', 'فرحان', 'سعادة', 'محبوب', 'شكرا', 'أحسن', 'أفضل',
    'احلى','ناجح', 'ممتاز'
    ]

    negative_words = [
         'حزين', 'فاشل', 'مقرف', 'قاتل', 'مؤسف', 'خايب', 'خيبة', 'غاضب', 'خوف', 
        'العنصرية', 'مرعب', 'العنف', 'مؤلم', 'عميل', 'بائس',
        'باطل', 'تفجير', 'خاين', 'كذاب', 'منافق', 'الفاسد', 'حرامي', 'إخوان', 'شيطان', 'إرهاب'
    ]

    features['positive_words_count'] = sum(text_lower.count(word) for word in positive_words)
    features['negative_words_count'] = sum(text_lower.count(word) for word in negative_words)
    features['has_positive_word'] = 1 if features['positive_words_count'] > 0 else 0
    features['has_negative_word'] = 1 if features['negative_words_count'] > 0 else 0

    return features

#1- Extract handcrafted features
handcrafted_features_list = []
for idx, (original, clean) in enumerate(zip(df['text'], df['clean_text'])):
    features = extract_handcrafted_features(str(original), str(clean))
    handcrafted_features_list.append(features)
handcrafted_df = pd.DataFrame(handcrafted_features_list)

#2- TF-IDF Features
tfidf_word = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2, max_df=0.8)
tfidf_word_features = tfidf_word.fit_transform(df['clean_text'])

tfidf_char = TfidfVectorizer(analyzer='char', ngram_range=(2,4), max_features=1000)
tfidf_char_features = tfidf_char.fit_transform(df['clean_text'])

#3- Word Embeddings
sentences = [text.split() for text in df['clean_text'] if text.strip()]
w2v_model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=2, workers=4, epochs=10)
ft_model = FastText(sentences=sentences, vector_size=100, window=5, min_count=2, workers=4, epochs=10)

def get_average_embedding(text, model):
    words = text.split()
    valid_words = [w for w in words if w in model.wv]
    return np.mean([model.wv[w] for w in valid_words], axis=0) if valid_words else np.zeros(model.vector_size)

w2v_features = np.array([get_average_embedding(text, w2v_model) for text in df['clean_text']])
ft_features = np.array([get_average_embedding(text, ft_model) for text in df['clean_text']])


#Combine All Features

# Convert handcrafted features to sparse matrix
handcrafted_array = handcrafted_df.values.astype(np.float32)
handcrafted_sparse = csr_matrix(handcrafted_array)

# Convert word embeddings to sparse matrices
w2v_sparse = csr_matrix(w2v_features.astype(np.float32))
ft_sparse = csr_matrix(ft_features.astype(np.float32))

# Combine ALL features together
all_features_sparse = hstack([
    tfidf_word_features,      # 5000 features
    tfidf_char_features,      # 1000 features  
    w2v_sparse,               # 100 features
    ft_sparse,                # 100 features
    handcrafted_sparse        # ~30 handcrafted features
])

print(f"\nFEATURE SUMMARY:")
print("-" * 80)
print(f"1. Word TF-IDF: {tfidf_word_features.shape[1]:,} features")
print(f"2. Char TF-IDF: {tfidf_char_features.shape[1]:,} features")
print(f"3. Word2Vec: {w2v_features.shape[1]:,} features")
print(f"4. FastText: {ft_features.shape[1]:,} features")
print(f"5. Handcrafted: {handcrafted_df.shape[1]:,} features")
print("-" * 80)
print(f"TOTAL: {all_features_sparse.shape[1]:,} features")
print("="*80)

# Save the combined features
print("\nSaving combined features...")
with open('combined_features.pkl', 'wb') as f:
    pickle.dump({
        'X': all_features_sparse,
        'y': df['label'].values,
        'handcrafted_feature_names': list(handcrafted_df.columns),
        'tfidf_word_vocab': list(tfidf_word.get_feature_names_out()),
        'tfidf_char_vocab': list(tfidf_char.get_feature_names_out())
    }, f)
print("✓ Saved to combined_features.pkl")


# Also save the dataframe with handcrafted features
df_with_handcrafted = pd.concat([df.reset_index(drop=True), 
                                 handcrafted_df.reset_index(drop=True)], axis=1)
df_with_handcrafted.to_csv('dataset_with_all_features.csv', 
                          index=False, encoding='utf-8-sig')
print("✓ Saved to dataset_with_all_features.csv")



# ============================================================================
# DATA SPLITTING (60% train, 20% validation, 20% test)
# ============================================================================

# Load the features from the pkl file
with open('combined_features.pkl', 'rb') as f:
    data = pickle.load(f)

X = data['X']
y = data['y']

print(f"Dataset shape: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Labels shape: {y.shape}")
#------------------------------------------------------------------------

print("\n" + "="*60)
print("SPLITTING DATA: 60% train, 20% validation, 20% test")
print("="*60)

# First split: 80% temp, 20% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # Maintain class distribution
)

# Second split: From temp, split 75% train, 25% validation (which is 60% and 20% of original)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, 
    test_size=0.25,  # 0.25 * 0.8 = 0.2 of original
    random_state=42, 
    stratify=y_temp
)

print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(y)*100:.1f}%)")
print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(y)*100:.1f}%)")
print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(y)*100:.1f}%)")


#----------------------------------------------------------------------------------------------


def evaluate_model(model, X, y, set_name="Test"):
    """Evaluate model and print metrics"""
    y_pred = model.predict(X)
    
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')
    
    print(f"\n{set_name} Set Results:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_pred': y_pred
    }


def plot_confusion_matrix(y_true, y_pred, title, classes):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    return cm


# ============================================================================
# MODEL 1: DECISION TREE 
# ============================================================================

print("\n" + "="*80)
print("MODEL 1: DECISION TREE")
print("="*80)

# Decision Tree for choosing the best parameter
"""
max_depth = [10,20,30]
min_samples_split = [2,4,10]
criterion = ['entropy', 'gini']
min_samples_leaf = [2,8,10]


best_val_f1 = 0
best_params = None
best_model = None

for depth in max_depth:
    for min_split in min_samples_split:
        for min_leaf in min_samples_leaf:
            for crit in criterion:

                model = DecisionTreeClassifier(max_depth=depth,min_samples_split=min_split,min_samples_leaf=min_leaf,criterion=crit,random_state=42)
                
                #tarining
                model.fit(X_train, y_train)

                #VALIDATION
                y_val_pred = model.predict(X_val)
                val_f1 = f1_score(y_val, y_val_pred, average='weighted')

                print(f"depth={depth}, min_split={min_split}, "
                    f"min_leaf={min_leaf}, criterion={crit} "
                    f"--> Val F1={val_f1:.4f}")

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_params = {'max_depth': depth,'min_samples_split': min_split,'min_samples_leaf': min_leaf,'criterion': crit}
                    best_model = model

print("\nBEST MODEL FOUND:")
print(best_params)
print(f"Best Validation F1: {best_val_f1:.4f}")

#test
dt_test_results = evaluate_model(best_model,X_test,y_test,set_name="Decision Tree Test")

#plot dt
plot_confusion_matrix(y_test, dt_test_results['y_pred'],title="Decision Tree - Test Confusion Matrix",classes=np.unique(y))
"""



# BEST PARAMETERS (from previous tuning)
best_params = {
    'max_depth': 30,
    'min_samples_split': 2,
    'min_samples_leaf': 8,
    'criterion': 'gini'
}

# Build model
best_model = DecisionTreeClassifier(
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    criterion=best_params['criterion'],
    random_state=42
)

# Training
best_model.fit(X_train, y_train)

# Validation
y_val_pred = best_model.predict(X_val)
val_f1 = f1_score(y_val, y_val_pred, average='weighted')

print("BEST MODEL PARAMETERS:")
print(best_params)
print(f"Validation F1: {val_f1:.4f}")

# Test
dt_test_results = evaluate_model(
    best_model,
    X_test,
    y_test,
    set_name="Decision Tree Test"
)

# Confusion Matrix
plot_confusion_matrix(
    y_test,
    dt_test_results['y_pred'],
    title="Decision Tree - Test Confusion Matrix",
    classes=np.unique(y)
)

# ============================================================================
# MODEL 2: Random Forest
# ============================================================================s

print("\n" + "="*80)
print("MODEL 2: RANDOM FOREST")
print("="*80)

"""
# Random Forest Hyperparameters tuning code 
n_estimators_rf = [300, 400]      
max_depth_rf = [25,35]                
min_samples_split_rf = [2,8]         
min_samples_leaf_rf = [5,10]          
criterion_rf = ['gini', 'entropy']   
max_features = ['sqrt', 'log2']
bootstrap =  [False, True] 
class_weight = ["balanced"] 

best_val_f1_rf = 0
best_params_rf = None
best_rf_model = None

for n_estimators in n_estimators_rf:
    for depth in max_depth_rf:
        for min_split in min_samples_split_rf:
            for min_leaf in min_samples_leaf_rf:
                for crit in criterion_rf:
                    for max_feat in max_features:
                        for boot in bootstrap:
                            for w in class_weight:

                                rf_model = RandomForestClassifier(
                                    n_estimators=n_estimators,
                                    max_depth=depth,
                                    min_samples_split=min_split,
                                    min_samples_leaf=min_leaf,
                                    criterion=crit,
                                    max_features=max_feat,
                                    bootstrap=boot,
                                    class_weight = w,
                                    random_state=42
                                )

                                # Training
                                rf_model.fit(X_train, y_train)

                                # Validation
                                y_val_pred = rf_model.predict(X_val)
                                val_f1 = f1_score(y_val, y_val_pred, average='weighted')

                                print(f"n_estimators={n_estimators}, depth={depth}, "
                                    f"min_split={min_split}, min_leaf={min_leaf}, criterion={crit},max_features={max_feat}, bootstrap={boot} "
                                    f"--> Val F1={val_f1:.4f}")

                                if val_f1 > best_val_f1_rf:
                                    best_val_f1_rf = val_f1
                                    best_params_rf = {
                                        'n_estimators': n_estimators,
                                        'max_depth': depth,
                                        'min_samples_split': min_split,
                                        'min_samples_leaf': min_leaf,
                                        'criterion': crit,
                                        'max_features': max_feat,
                                        'bootstrap': boot,
                                        'class_weight' : w
                                    }
                                    best_rf_model = rf_model

print("\nBEST RANDOM FOREST MODEL FOUND:")
print(best_params_rf)
print(f"Best Validation F1: {best_val_f1_rf:.4f}")

# Test Evaluation
rf_test_results = evaluate_model(best_rf_model, X_test, y_test, set_name="Random Forest Test")

# Plot Confusion Matrix
plot_confusion_matrix(y_test, rf_test_results['y_pred'], title="Random Forest - Test Confusion Matrix", classes=np.unique(y))

"""


# BEST PARAMETERS (from previous tuning)
best_params_rf = {
    'n_estimators': 400,       
    'max_depth': 25,
    'min_samples_split': 2,
    'min_samples_leaf': 5,
    'criterion': 'gini',
    'max_features': 'sqrt',
    'bootstrap': False,
    'class_weight': 'balanced'
}

# Build model
best_rf_model = RandomForestClassifier(
    n_estimators=best_params_rf['n_estimators'],
    max_depth=best_params_rf['max_depth'],
    min_samples_split=best_params_rf['min_samples_split'],
    min_samples_leaf=best_params_rf['min_samples_leaf'],
    criterion=best_params_rf['criterion'],
    max_features=best_params_rf['max_features'],
    bootstrap=best_params_rf['bootstrap'],
    class_weight=best_params_rf['class_weight'],
    random_state=42
)

# Training
best_rf_model.fit(X_train, y_train)

# Validation
y_val_pred = best_rf_model.predict(X_val)
val_f1 = f1_score(y_val, y_val_pred, average='weighted')

print("BEST RANDOM FOREST PARAMETERS:")
print(best_params_rf)
print(f"Validation F1: {val_f1:.4f}")

# Test Evaluation
rf_test_results = evaluate_model(
    best_rf_model,
    X_test,
    y_test,
    set_name="Random Forest Test"
)

# Confusion Matrix
plot_confusion_matrix(
    y_test,
    rf_test_results['y_pred'],
    title="Random Forest - Test Confusion Matrix",
    classes=np.unique(y)
)


"""

# ============================================================================
# MODEL: ARTIFICIAL NEURAL NETWORK (ANN) - FULL HYPERPARAMETER TUNING
# ============================================================================



print("\n" + "="*80)
print("MODEL: ARTIFICIAL NEURAL NETWORK (ANN) - FULL TUNING")
print("="*80)

scaler = StandardScaler(with_mean=False)
X_train_nn = scaler.fit_transform(X_train)
X_val_nn   = scaler.transform(X_val)
X_test_nn  = scaler.transform(X_test)


# ---------------------------------------------------------------------------
# Hyperparameter search space
# ---------------------------------------------------------------------------

hidden_layer_options = [(128,), (256,), (256, 128)]   
activation_functions = ['relu']                    
optimizers = ['adam']                              
learning_rates = [0.001, 0.0005]                    
batch_sizes = [64]                                   
max_epochs = 40                                   

# ---------------------------------------------------------------------------
# Tuning loop
# ---------------------------------------------------------------------------

best_ann_f1 = 0
best_ann_params = None
best_ann_model = None

print("\nStarting ANN hyperparameter tuning...")
print("-"*80)

for layers in hidden_layer_options:
    for activation in activation_functions:
        for optimizer in optimizers:
            for lr in learning_rates:
                for batch in batch_sizes:

                    ann_model = MLPClassifier(
                        hidden_layer_sizes=layers,
                        activation=activation,
                        solver=optimizer,
                        learning_rate_init=lr,
                        batch_size=batch,
                        max_iter=max_epochs,
                        alpha=0.0001,           
                        early_stopping= False,    
                        random_state=42
                    )

                    # Training
                    ann_model.fit(X_train_nn, y_train)

                    # Validation
                    y_val_pred = ann_model.predict(X_val_nn)
                    val_f1 = f1_score(y_val, y_val_pred, average='weighted')

                    print(
                        f"Layers={layers}, Act={activation}, "
                        f"Opt={optimizer}, LR={lr}, Batch={batch} "
                        f"→ Val F1={val_f1:.4f}"
                    )

                    if val_f1 > best_ann_f1:
                        best_ann_f1 = val_f1
                        best_ann_params = {
                            'hidden_layer_sizes': layers,
                            'activation': activation,
                            'solver': optimizer,
                            'learning_rate_init': lr,
                            'batch_size': batch
                        }
                        best_ann_model = ann_model

# ---------------------------------------------------------------------------
# Best ANN configuration
# ---------------------------------------------------------------------------

print("\n" + "="*80)
print("BEST ANN CONFIGURATION (SELECTED USING VALIDATION SET)")
print("="*80)
print(best_ann_params)
print(f"Best Validation F1: {best_ann_f1:.4f}")

# ---------------------------------------------------------------------------
# Final evaluation on TEST set
# ---------------------------------------------------------------------------

ann_test_results = evaluate_model(
    best_ann_model,
    X_test_nn,
    y_test,
    set_name="ANN Test"
)

plot_confusion_matrix(
    y_test,
    ann_test_results['y_pred'],
    title="ANN - Test Confusion Matrix",
    classes=np.unique(y)
)
"""


print("\n" + "="*80)
print("ARTIFICIAL NEURAL NETWORK")
print("="*80)

# Scaling
scaler = StandardScaler(with_mean=False)
X_train_nn = scaler.fit_transform(X_train)
X_val_nn   = scaler.transform(X_val)
X_test_nn  = scaler.transform(X_test)

best_ann_params = {
    "hidden_layer_sizes": (256, 128),
    "activation": "relu",
    "solver": "adam",
    "learning_rate_init": 0.001,
    "batch_size": 64,
    "max_iter": 60,
    "alpha": 0.0005,
    "early_stopping": False,
    "random_state": 42
}

best_ann_model = MLPClassifier(
    hidden_layer_sizes=best_ann_params["hidden_layer_sizes"],
    activation=best_ann_params["activation"],
    solver=best_ann_params["solver"],
    learning_rate_init=best_ann_params["learning_rate_init"],
    batch_size=best_ann_params["batch_size"],
    max_iter=best_ann_params["max_iter"],
    alpha=best_ann_params["alpha"],
    early_stopping=best_ann_params["early_stopping"],
    random_state=best_ann_params["random_state"]
)

# Training
best_ann_model.fit(X_train_nn, y_train)

# Validation
y_val_pred = best_ann_model.predict(X_val_nn)
val_f1 = f1_score(y_val, y_val_pred, average="weighted")

print("BEST ANN PARAMETERS:")
print(best_ann_params)
print(f"Validation F1: {val_f1:.4f}")

# Test
ann_test_results = evaluate_model(
    best_ann_model,
    X_test_nn,
    y_test,
    set_name="ANN Test"
)

# Confusion Matrix
plot_confusion_matrix(
    y_test,
    ann_test_results['y_pred'],
    title="ANN - Test Confusion Matrix",
    classes=np.unique(y)
)


#------------------------------------------------------------------------------------------------------
#Model 4: Naive bais
# ============================================================================
# PART 1: MULTINOMIAL NAÏVE BAYES
# ============================================================================

print("\n" + "="*80)
print("MODEL 3: NAÏVE BAYES")
print("="*80)


# Use only TF-IDF features (combine word and char level)
idx = np.arange(len(df))

train_idx, temp_idx, y_train, y_temp = train_test_split(
    idx, y, test_size=0.4, random_state=42, stratify=y
)

val_idx, test_idx, y_val, y_test = train_test_split(
    temp_idx, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

X_train_nb = hstack([tfidf_word_features[train_idx],
                     tfidf_char_features[train_idx]], format="csr")

X_val_nb   = hstack([tfidf_word_features[val_idx],
                     tfidf_char_features[val_idx]], format="csr")

X_test_nb  = hstack([tfidf_word_features[test_idx],
                     tfidf_char_features[test_idx]], format="csr")


                     


"""# ============================================================================
# PART 1: MULTINOMIAL NAÏVE BAYES
# ============================================================================

print("\n" + "-"*80)
print("PART 1: MULTINOMIAL NAÏVE BAYES")
print("-"*80)

alpha_values = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
fit_prior_values = [True, False]

best_mnb_f1 = 0
best_mnb_params = None
best_mnb_model = None

print("\nStarting Multinomial NB tuning...")
print("-" * 80)

for alpha in alpha_values:
    for fit_prior in fit_prior_values:
        mnb_model = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
        mnb_model.fit(X_train_nb, y_train)
        
        y_val_pred = mnb_model.predict(X_val_nb)
        val_f1 = f1_score(y_val, y_val_pred, average='weighted')
        
        print(f"alpha={alpha:6.2f}, fit_prior={fit_prior:5} "
              f"→ Val F1={val_f1:.4f}")
        
        if val_f1 > best_mnb_f1:
            best_mnb_f1 = val_f1
            best_mnb_params = {'alpha': alpha, 'fit_prior': fit_prior}
            best_mnb_model = mnb_model

print("\n" + "="*80)
print("BEST MULTINOMIAL NB PARAMETERS")
print("="*80)
print(f"Best parameters: {best_mnb_params}")
print(f"Best Validation F1: {best_mnb_f1:.4f}")

# BEST PARAMETERS (from previous tuning)
best_mnb_params = {
    'alpha': best_mnb_params['alpha'],
    'fit_prior': best_mnb_params['fit_prior']
}

# Build model
best_mnb_model = MultinomialNB(
    alpha=best_mnb_params['alpha'],
    fit_prior=best_mnb_params['fit_prior']
)

# Training
best_mnb_model.fit(X_train_nb, y_train)

# Validation
y_val_pred = best_mnb_model.predict(X_val_nb)
val_f1 = f1_score(y_val, y_val_pred, average='weighted')

print("BEST NAIVE BAYES PARAMETERS:")
print(best_mnb_params)
print(f"Validation F1: {val_f1:.4f}")

# Test
mnb_test_results = evaluate_model(
    best_mnb_model,
    X_test_nb,
    y_test,
    set_name="Naive Bayes Test"
)

# Confusion Matrix
plot_confusion_matrix(
    y_test,
    mnb_test_results['y_pred'],
    title="Naive Bayes - Test Confusion Matrix",
    classes=np.unique(y)
)"""


print("PART 1: MULTINOMIAL NAÏVE BAYES")
print("-"*80)

best_mnb_params = {
    'alpha': 0.1,      
    'fit_prior': True  
}

best_mnb_model = MultinomialNB(
    alpha=best_mnb_params['alpha'],
    fit_prior=best_mnb_params['fit_prior']
)

# Training
best_mnb_model.fit(X_train_nb, y_train)

# Validation
y_val_pred = best_mnb_model.predict(X_val_nb)
val_f1 = f1_score(y_val, y_val_pred, average='weighted')

print("BEST NAIVE BAYES PARAMETERS:")
print(best_mnb_params)
print(f"Validation F1: {val_f1:.4f}")

# Test
mnb_test_results = evaluate_model(
    best_mnb_model,
    X_test_nb,
    y_test,
    set_name="Naive Bayes Test"
)

# Confusion Matrix
plot_confusion_matrix(
    y_test,
    mnb_test_results['y_pred'],
    title=" MULTINOMIAL NB - Test Confusion Matrix",
    classes=np.unique(y)
)

# ============================================================================
# PART 2: COMPLEMENT NAÏVE BAYES
# ============================================================================

print("\n" + "-"*80)
print("PART 2: COMPLEMENT NAÏVE BAYES")
print("-"*80)


"""
alpha_cm = [ 0.5, 1.0, 2.0, 5.0]
fit_prior_cm = [True, False]

best_cnb_f1 = 0
best_cnb_params = None
best_cnb_model = None

print("\nStarting Complement NB tuning...")
print("-" * 80)

for alpha in alpha_cm:
    for fit_prior in fit_prior_cm:
        cnb_model = ComplementNB(alpha=alpha, fit_prior=fit_prior)
        cnb_model.fit(X_train_nb, y_train)

        y_val_pred = cnb_model.predict(X_val_nb)
        val_f1 = f1_score(y_val, y_val_pred, average='weighted')

        print(f"alpha={alpha:6.2f}, fit_prior={fit_prior:1} "f"→ Val F1={val_f1:.4f}")

        if val_f1 > best_cnb_f1:
            best_cnb_f1 = val_f1
            best_cnb_params = {'alpha': alpha, 'fit_prior': fit_prior}
            best_cnb_model = cnb_model

print("\n" + "="*80)
print("BEST COMPLEMENT NB PARAMETERS")
print("="*80)
print(f"Best parameters: {best_cnb_params}")
print(f"Best Validation F1: {best_cnb_f1:.4f}")

# Build best ComplementNB model
best_cnb_model = ComplementNB(
    alpha=best_cnb_params['alpha'],
    fit_prior=best_cnb_params['fit_prior']
)

# Training
best_cnb_model.fit(X_train_nb, y_train)

# Validation
y_val_pred = best_cnb_model.predict(X_val_nb)
val_f1 = f1_score(y_val, y_val_pred, average='weighted')

print("BEST COMPLEMENT NB PARAMETERS (RETRAINED):")
print(best_cnb_params)
print(f"Validation F1: {val_f1:.4f}")

# Test
cnb_test_results = evaluate_model(
    best_cnb_model,
    X_test_nb,
    y_test,
    set_name="Complement NB Test"
)

# Confusion Matrix
plot_confusion_matrix(
    y_test,
    cnb_test_results['y_pred'],
    title="Complement NB - Test Confusion Matrix",
    classes=np.unique(y)
)"""


best_cnb_params = {
    'alpha': 1.0,      
    'fit_prior': True  
}

best_cnb_model = ComplementNB(
    alpha=best_cnb_params['alpha'],
    fit_prior=best_cnb_params['fit_prior']
)
# Training
best_cnb_model.fit(X_train_nb, y_train)

# Validation
y_val_pred = best_cnb_model.predict(X_val_nb)
val_f1 = f1_score(y_val, y_val_pred, average='weighted')

print("BEST NAIVE BAYES PARAMETERS:")
print(best_mnb_params)
print(f"Validation F1: {val_f1:.4f}")


# Test
cnb_test_results = evaluate_model(
    best_cnb_model,
    X_test_nb,
    y_test,
    set_name="Complement NB Test"
)

# Confusion Matrix
plot_confusion_matrix(
    y_test,
    cnb_test_results['y_pred'],
    title="Complement NB - Test Confusion Matrix",
    classes=np.unique(y)
)