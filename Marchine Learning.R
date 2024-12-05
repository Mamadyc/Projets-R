## Projet Tutoré 2
##Prédire la mortalité hospitalière à l'aide d'un modèle de machine learning (ou deep learning)
# Charger les bibliothèques nécessaires

library(tidyverse)
library(caret)
library(xgboost)
library(lattice)
library(dplyr)
library(randomForest)

# Charger les donnée

dataPT2.csv <- read.csv("C:/Users/conde/Downloads/dataPT2.csv.gz")
View(dataPT2.csv)
str(dataPT2.csv)
head(dataPT2.csv)

# Charger le cahier des variables

cahierVariables <- read.csv("C:/Users/conde/Downloads/cahierVariables.csv")
View(cahierVariables)

# on a 82542 observation et 84 Variables

# Colonnes à supprimer

colonne_sup <- c('ethnicity', 'encounter_id', 'hospital_id', 'icu_id', 'patient_id', 'apache_4a_icu_death_prob', 'apache_4a_hospital_death_prob')

# Supprimer les colonnes inutiles

data <- dataPT2.csv[, !(names(dataPT2.csv) %in% colonne_sup)]
view(data)
## data ( 82542 observation et 77 variables)

# les valeurs manquantes et encodage des variables catégorielles

data <- data.frame(lapply(data, function(x) ifelse(is.na(x), 0, x)))

# Remplacer les valeurs dans la colonne "gender" M= 1 et F=0

data$gender <- ifelse(data$gender == 'M', 1, ifelse(data$gender == 'F', 0, data$gender))
str(data)
hist(data$hospital_death) ## trop de différence entre les 0 et les 1 

### convertir les données 
# Liste des noms de colonnes binaires

columns_binaire <- c('elective_surgery', 'apache_post_operative', 'hospital_death', 'arf_apache',
                     'intubated_apache', 'ventilated_apache', 'cirrhosis', 'diabetes_mellitus',
                     'hepatic_failure', 'immunosuppression', 'leukemia', 'lymphoma',
                     'solid_tumor_with_metastasis', 'aids', 'gcs_eyes_apache',
                     'gcs_motor_apache', 'gcs_unable_apache', 'gcs_verbal_apache')

# Convertir chaque colonne en type de données catégorie
df[, columns_binaire] <- lapply(df[, columns_binaire], as.factor)


# Afficher le résultat
print(df)


# Assuming 'data' is your data frame
# Convert numeric columns to numeric class
numeric_cols <- c("age", "bmi", "pre_icu_los_days", "weight", "apache_2_diagnosis", "apache_3j_diagnosis", 
                  "gcs_eyes_apache", "gcs_motor_apache", "gcs_unable_apache", "gcs_verbal_apache",
                  "heart_rate_apache", "map_apache", "resprate_apache", "temp_apache",
                  "d1_diasbp_max", "d1_diasbp_min", "d1_diasbp_noninvasive_max", "d1_diasbp_noninvasive_min",
                  "d1_heartrate_max", "d1_heartrate_min", "d1_mbp_max", "d1_mbp_min",
                  "d1_mbp_noninvasive_max", "d1_mbp_noninvasive_min", "d1_resprate_max", "d1_resprate_min",
                  "d1_spo2_max", "d1_spo2_min", "d1_sysbp_max", "d1_sysbp_min",
                  "d1_sysbp_noninvasive_max", "d1_sysbp_noninvasive_min", "d1_temp_max", "d1_temp_min",
                  "h1_diasbp_max", "h1_diasbp_min", "h1_diasbp_noninvasive_max", "h1_diasbp_noninvasive_min",
                  "h1_heartrate_max", "h1_heartrate_min", "h1_mbp_max", "h1_mbp_min",
                  "h1_mbp_noninvasive_max", "h1_mbp_noninvasive_min", "h1_resprate_max", "h1_resprate_min",
                  "h1_spo2_max", "h1_spo2_min", "h1_sysbp_max", "h1_sysbp_min",
                  "h1_sysbp_noninvasive_max", "h1_sysbp_noninvasive_min", "d1_glucose_max", "d1_glucose_min",
                  "d1_potassium_max", "d1_potassium_min")

data[numeric_cols] <- lapply(data[numeric_cols], as.numeric)

# Convert integer columns to integer class
integer_cols <- c("elective_surgery", "apache_post_operative", "arf_apache", "intubated_apache", "ventilated_apache",
                  "aids", "cirrhosis", "diabetes_mellitus", "hepatic_failure", "immunosuppression",
                  "leukemia", "lymphoma", "solid_tumor_with_metastasis", "hospital_death")

data[integer_cols] <- lapply(data[integer_cols], as.integer)

# Convert character columns to factor class
character_cols <- c("gender", "icu_admit_source", "icu_stay_type", "icu_type", "apache_3j_bodysystem", "apache_2_bodysystem")

data[character_cols] <- lapply(data[character_cols], as.factor)

str(data)


# Installer les packages si nécessaire
install.packages("caret")
install.packages("ROSE")

# Charger les bibliothèques
library(caret)
library(ROSE)

# Spécifier la graine pour la reproductibilité
set.seed(123)
## melanger les lignes de fançons aléatoire  avec la fonction 
head(data)
str(data)
rows<-sample(nrow(data))
data_mel <- data[rows,]
View(data_mel)
head(data_mel)

# Créer une partition stratifiée des données

index <- createDataPartition(data$hospital_death, p = 0.8, list = FALSE, times = 1)

# Séparer les données en ensembles d'entraînement et de test

train_data <- data[index, ]
test_data <- data[-index, ]

# Vérifier l'équilibre des classes dans l'ensemble d'entraînement

table(train_data$hospital_death)

# Effectuer un suréchantillonnage de la classe minoritaire avec ROS
library(ROSE)

# Obtenez la taille de la classe minoritaire dans l'échantillon d'entraînement
minority_size <- sum(train_data$hospital_death == 1)

# Obtenez la taille de l'échantillon initial
n <- nrow(train_data)

# Obtenez la taille de la classe minoritaire dans l'échantillon d'entraînement
minority_size <- sum(train_data$hospital_death == 1)

# Ajustez la valeur de N pour qu'elle soit supérieure ou égale à la taille de l'échantillon initial
N <- max(2 * minority_size, n)


# Utilisez ovun.sample en ajustant la valeur de N
rose_train_data <- ovun.sample(hospital_death ~ ., 
                               data = train_data, 
                               method = "over", 
                               N = N, 
                               seed = 123)$data

# Vérifier l'équilibre des classes après le suréchantillonnage
table(rose_train_data$hospital_death)
hist(rose_train_data$hospital_death)
train<- rose_train_data


### Model GLM ########"

model <- glm(hospital_death ~ ., data = train_data, family = "binomial")
library(caret)

predictions <- predict(model, newdata = test_data, type = "response")
predictions_class <- ifelse(predictions > 0.5, 1, 0)

confusion_matrix <- confusionMatrix(table(predictions_class, test_data$hospital_death))
print(confusion_matrix)

# Installer le package pROC si nécessaire
# install.packages("pROC")

# Charger la bibliothèque
library(pROC)

# Obtenir les probabilités prédites du modèle pour la classe positive
predicted_probs <- predict(model, newdata = test_data, type = "response")

# Créer un objet de courbe ROC
roc_curve <- roc(test_data$hospital_death, predicted_probs)

# Afficher la courbe ROC
plot(roc_curve, main = "Courbe ROC", col = "blue", lwd = 2)

# Ajouter l'aire sous la courbe (AUC) au graphique
text(0.8, 0.2, paste("AUC =", round(auc(roc_curve), 3)), col = "blue")

# Afficher la légende
legend("bottomright", legend = paste("AUC =", round(auc(roc_curve), 3)), col = "blue", lty = 1, cex = 0.8)

# Calcul des métriques de la matrice de confusion
conf_matrix <- confusionMatrix(data = as.factor(predictions_class), reference = as.factor(test_data$hospital_death))

# Récupération des valeurs nécessaires pour le F1-score
TP <- conf_matrix$byClass[["Sensitivity"]] * conf_matrix$byClass[["Pos Pred Value"]]
FP <- (1 - conf_matrix$byClass[["Specificity"]]) * conf_matrix$byClass[["Neg Pred Value"]]
FN <- (1 - conf_matrix$byClass[["Sensitivity"]]) * conf_matrix$byClass[["Pos Pred Value"]]

# Calcul du F1-score
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)
f1_score <- 2 * (precision * recall) / (precision + recall)

# Affichage du score F1
print(paste("F1 Score:", round(f1_score, 3)))

























































































